"""Chunked rollout collection with a shrinking cache-preserving active batch.

Place beside actor.py. In train.py, change `from actor import Actor` to
`from continuous_actor import Actor`. Keep actor.py: this module reuses its
Turn/Rollout types, tokenizer setup, and conversation formatting.

Targets the Transformers 5.19 Cache API used by this project. This is a
synchronous scheduler, not Hugging Face's continuous-batching backend.
"""

from dataclasses import dataclass, field
import time

import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import DynamicCache

from actor import Actor as OriginalActor, Rollout, Turn
from env import calculator, execute_tools, verify_solution
from utils.prompts import prompt

CHUNK_SIZE = 512
REBUILD_THRESHOLD = 0.25


@dataclass
class _State:
    group_index: int
    rollout_index: int
    puzzle: list[int]
    history: torch.Tensor                 # 1-D CPU tensor; NEVER contains batch padding
    turn_ids: torch.Tensor                # Current assistant turn, possibly unfinished
    completion_start: int = 0             # Injected prefix length within turn_ids
    turns: list[Turn] = field(default_factory=list)
    reward: float = 0.0


class Actor(OriginalActor):
    def __init__(
        self,
        model,
        max_output_tokens: int = 4096,
        *,
        chunk_size: int = CHUNK_SIZE,
        rebuild_threshold: float = REBUILD_THRESHOLD,
        log_chunks: bool = True,
    ):
        super().__init__(model, max_output_tokens)
        if max_output_tokens <= 0 or chunk_size <= 0:
            raise ValueError("max_output_tokens and chunk_size must be positive")
        if not 0 < rebuild_threshold <= 1:
            raise ValueError("rebuild_threshold must be in (0, 1]")
        if self.tokenizer.eos_token_id is None:
            raise ValueError("An EOS token is required to detect assistant-turn boundaries")

        self.chunk_size = chunk_size
        self.rebuild_threshold = rebuild_threshold
        self.log_chunks = log_chunks
        self.pad_id = self.tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = self.tokenizer.eos_token_id
        self.last_stats: dict[str, int | float] = {}

    def _initial_states(self, puzzles: list[list[int]], group_size: int) -> list[_State]:
        states = []
        for group_index, puzzle in enumerate(puzzles):
            ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt(self.max_output_tokens, puzzle)}],
                tools=[calculator],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=True,
                return_dict=True,
                return_tensors="pt",
            )["input_ids"][0].detach().cpu()
            if ids.numel() >= self.max_output_tokens:
                raise ValueError(
                    f"Puzzle {puzzle}: prompt uses {ids.numel()} tokens, leaving no "
                    f"generation room in the {self.max_output_tokens}-token budget"
                )
            for rollout_index in range(group_size):
                states.append(_State(
                    group_index=group_index,
                    rollout_index=rollout_index,
                    puzzle=list(puzzle),
                    history=ids.clone(),
                    turn_ids=torch.empty(0, dtype=torch.long),
                ))
        return states

    def _fresh_batch(self, states: list[_State]):
        histories = [s.history for s in states]
        ids = pad_sequence(
            histories, batch_first=True, padding_value=self.pad_id, padding_side="left"
        ).to(self.model.device)
        mask = pad_sequence(
            [torch.ones_like(h) for h in histories],
            batch_first=True, padding_value=0, padding_side="left",
        ).to(self.model.device)
        return ids, mask

    def _consume_response(self, state: _State, response: torch.Tensor) -> tuple[str, int]:
        """Append a chunk; return active/waiting/done and its real token count."""
        stops = (response == self.tokenizer.eos_token_id).nonzero().flatten()
        reached_eos = stops.numel() > 0
        if reached_eos:
            response = response[:int(stops[0]) + 1]  # Include EOS, exclude padding after it.

        count = response.numel()
        state.history = torch.cat([state.history, response])
        state.turn_ids = torch.cat([state.turn_ids, response])
        exhausted = state.history.numel() >= self.max_output_tokens
        if not reached_eos and not exhausted:
            return "active", count

        # A scheduling chunk is NOT a new Turn. Assemble all chunks before parsing.
        state.turns.append(Turn(
            token_ids=state.turn_ids.unsqueeze(0),
            completion_start=state.completion_start,
        ))
        text = self.tokenizer.decode(
            state.turn_ids[state.completion_start:], skip_special_tokens=True
        )
        tool_results = execute_tools(text, state.puzzle)
        solved = any(verify_solution(result, state.puzzle) for result in tool_results)
        state.reward = float(solved)

        # Match the original actor: a complete tool call at the final token limit
        # can still earn a reward, even if that final truncated turn has no EOS.
        if solved or exhausted:
            return "done", count

        # Only a genuine EOS can get here. Do not insert a header at chunk limits.
        suffix = self.continuation_suffix(state.history.unsqueeze(0), tool_results)
        suffix_ids = torch.tensor(
            self.tokenizer.encode(suffix, add_special_tokens=False), dtype=torch.long
        )
        if state.history.numel() + suffix_ids.numel() >= self.max_output_tokens:
            return "done", count  # No room for another model-generated token.

        state.history = torch.cat([state.history, suffix_ids])
        state.turn_ids = suffix_ids
        state.completion_start = suffix_ids.numel()
        return "waiting", count

    def _synchronize(self) -> None:
        device = self.model.device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

    def collect_rollouts(
        self, puzzles: list[list[int]], group_size: int = 8
    ) -> list[list[Rollout]]:
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        self.last_stats = dict(
            chunks=0, prefills=0, cached_chunks=0, generated_tokens=0,
            generate_seconds=0.0, wall_seconds=0.0,
        )
        if not puzzles:
            return []

        # Flatten across puzzles for generation, but preserve each GRPO group.
        waiting = self._initial_states(puzzles, group_size)
        active: list[_State] = []
        results: list[list[Rollout | None]] = [
            [None] * group_size for _ in puzzles
        ]
        cache = None
        input_ids = attention_mask = None
        cohort_start_size = 0
        completed = 0
        total = len(waiting)

        # Restore even mixed train/eval module settings after collection.
        previous_modes = [(module, module.training) for module in self.model.modules()]
        self.model.eval()
        started = time.perf_counter()
        try:
            while active or waiting:
                fresh = cache is None
                if fresh:
                    # Unfinished active turns are resumed verbatim; no new headers.
                    active.extend(waiting)
                    waiting.clear()
                    cohort_start_size = len(active)
                    input_ids, attention_mask = self._fresh_batch(active)
                    cache = DynamicCache(config=self.model.config)
                    self.last_stats["prefills"] += 1
                else:
                    self.last_stats["cached_chunks"] += 1

                remaining_budgets = [
                    self.max_output_tokens - s.history.numel() for s in active
                ]
                max_new_tokens = min(self.chunk_size, min(remaining_budgets))
                if max_new_tokens <= 0:
                    raise RuntimeError("An exhausted rollout was left in the active batch")

                batch_size = len(active)
                prefix_width = input_ids.shape[1]
                self._synchronize()
                tick = time.perf_counter()
                with torch.inference_mode():
                    output = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        past_key_values=cache,
                        # Explicit overrides prevent inherited static-cache settings
                        # or forced EOS from interfering with chunk continuation.
                        cache_implementation=None,
                        compile_config=None,
                        disable_compile=True,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.pad_id,
                        forced_eos_token_id=None,
                        min_length=0,
                        min_new_tokens=0,
                        max_new_tokens=max_new_tokens,
                        num_return_sequences=1,  # Histories are already expanded.
                        num_beams=1,
                        do_sample=True,
                        temperature=1.0,
                        top_k=0,
                        top_p=1.0,
                        repetition_penalty=1.0,
                        use_cache=True,
                        return_dict_in_generate=True,
                        output_scores=False,
                        output_logits=False,
                        output_attentions=False,
                        output_hidden_states=False,
                    )
                self._synchronize()
                elapsed = time.perf_counter() - tick
                self.last_stats["chunks"] += 1
                self.last_stats["generate_seconds"] += elapsed

                generated = output.sequences[:, prefix_width:].detach().cpu()
                generated_width = generated.shape[1]
                if not 0 < generated_width <= max_new_tokens:
                    raise RuntimeError("generate() returned an unexpected number of tokens")
                if output.past_key_values is None:
                    raise RuntimeError("generate() did not return the requested cache")

                survivors: list[_State] = []
                keep_rows: list[int] = []
                real_tokens = 0
                for row_index, state in enumerate(active):
                    status, count = self._consume_response(state, generated[row_index])
                    real_tokens += count
                    if status == "active":
                        survivors.append(state)
                        keep_rows.append(row_index)
                    elif status == "waiting":
                        waiting.append(state)
                    else:
                        results[state.group_index][state.rollout_index] = Rollout(
                            turns=state.turns, reward=state.reward
                        )
                        completed += 1
                self.last_stats["generated_tokens"] += real_tokens

                # Only rebuild a small active batch if there is waiting work to add.
                rebuild = bool(waiting) and (
                    len(survivors) <= self.rebuild_threshold * cohort_start_size
                )
                if survivors and not rebuild:
                    cache = output.past_key_values
                    if len(survivors) == batch_size:
                        input_ids = output.sequences
                        kept_mask = attention_mask
                    else:
                        indices = torch.tensor(
                            keep_rows, dtype=torch.long, device=output.sequences.device
                        )
                        # Select BOTH attention KV tensors and recurrent/conv states.
                        with torch.inference_mode():
                            cache.reorder_cache(indices)
                        input_ids = output.sequences.index_select(0, indices)
                        kept_mask = attention_mask.index_select(0, indices)

                    # Survivors emitted NO EOS, so every new position is a real token.
                    # Keep old left padding unchanged while reusing this cache.
                    attention_mask = torch.cat([
                        kept_mask,
                        kept_mask.new_ones((len(survivors), generated_width)),
                    ], dim=1)
                else:
                    # Drop the active batch/cache before the next fresh prefill.
                    cache = None
                    input_ids = attention_mask = None

                active = survivors
                del output
                if self.log_chunks:
                    mode = "prefill" if fresh else "cached"
                    tqdm.write(
                        f"[{mode}] batch={batch_size}, limit={max_new_tokens}, "
                        f"tokens={real_tokens}, {elapsed:.2f}s | "
                        f"active={len(active)}, waiting={len(waiting)}, "
                        f"done={completed}/{total}"
                        + (" | rebuild next" if rebuild else "")
                    )
        finally:
            cache = None
            input_ids = attention_mask = None
            for module, was_training in previous_modes:
                module.training = was_training
            self.last_stats["wall_seconds"] = time.perf_counter() - started

        if completed != total or any(r is None for group in results for r in group):
            raise RuntimeError("Scheduler finished without returning every rollout")
        return [[r for r in group if r is not None] for group in results]


if __name__ == "__main__":
    from transformers import Qwen3_5ForCausalLM

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    model = Qwen3_5ForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B", dtype=dtype
    ).to(device).eval()
    actor = Actor(model, max_output_tokens=2048)
    puzzles = [[6, 6, 6, 6]]
    groups = actor.collect_rollouts(puzzles, group_size=8)
    for puzzle, group in zip(puzzles, groups):
        print(f"Puzzle {puzzle}: {sum(r.reward for r in group):g}/{len(group)} successes")
    print(actor.last_stats)