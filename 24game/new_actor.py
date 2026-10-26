import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import DynamicCache
from transformers import AutoTokenizer, Qwen3_5ForCausalLM, LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList
from dataclasses import dataclass
import time
from env import calculator, verify_solution, execute_tools

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.prompts import prompt

@dataclass
class Turn:
    """
    A single turn of the actor, containing this turn's token ids. Former is prompt, latter is completion
    The very first turn omits the prompt (as it's just the initial prompt), future turns are just continuations
    """
    token_ids: torch.Tensor # [N,]
    completion_start: int

@dataclass
class Rollout:
    turns: list[Turn]
    reward: float
    history_token_count: int

    @property
    def len_until_last_turn_prompt(self) -> int:
        return self.history_token_count - (len(self.turns[-1].token_ids) - self.turns[-1].completion_start) 

@dataclass
class RolloutManager:
    """
    Storing every Tensor in RolloutManager, Rollout, and Turn as just [N,] 1D not 1xN
    """
    puzzle: list[int]
    starting_rollout_id: int
    prompt_ids: torch.Tensor
    rollouts: list[Rollout]

    def rollout(self, rollout_id: int) -> Rollout:
        return self.rollouts[rollout_id - self.starting_rollout_id]

    def history(self, rollout_id: int) -> torch.Tensor:
        return torch.cat([self.prompt_ids, *[turn.token_ids for turn in self.rollout(rollout_id).turns]], dim=0)

@dataclass
class Batch:
    input_ids: torch.Tensor # (B, N)
    attention_mask: torch.Tensor
    rollout_ids: list[int]
    left_padding: list[int]
    cache: DynamicCache
    cohort_start_size: int

def continuation_suffix(tokenizer: AutoTokenizer, prefix_ids: torch.Tensor, tool_results: list[str]) -> str:
    ids = prefix_ids.tolist()
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    newline_id = tokenizer.encode("\n", add_special_tokens=False)[0]
    already_closed = len(ids) >= 2 and ids[-2] == end_id and ids[-1] == newline_id
    close = "" if already_closed else "\n" if ids[-1] == end_id else "<|im_end|>\n"
    tools = "".join(f"\n<tool_response>\n{result}\n</tool_response>" for result in tool_results)
    if tools:
        tools = f"<|im_start|>user{tools}<|im_end|>\n"
    return close + tools + "<|im_start|>assistant\n<think>\n"

class Actor:
    def get_next_injected_token_index(self, rollout_id: int, tokens_produced: int) -> int:
        """
        Calculates the index of the next token to inject from the back. 1 means access[-1] so last token

        Args:
            tokens_produced: The number of tokens produced so far, including prompt + generated tokens.
        """
        rollout = self.rollout_manager(rollout_id).rollout(rollout_id)
        should_generate = rollout.history_token_count
        
        next_id_from_back = should_generate - tokens_produced
        return next_id_from_back

    class Injector(LogitsProcessor):
        def __init__(self, actor: Actor):
            self.actor = actor

        def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
            for row, rollout_id in enumerate(self.actor.batch.rollout_ids):
                next_id_from_back = self.actor.get_next_injected_token_index(rollout_id, input_ids.shape[1] - self.actor.batch.left_padding[row])
                if next_id_from_back <= 0:
                    continue # normal model sampling
                token_id = int(self.actor.rollout_manager(rollout_id).rollout(rollout_id).turns[-1].token_ids[-next_id_from_back])
                scores[row].fill_(-torch.inf)
                scores[row, token_id] = 0.0

            return scores

    class Stopper(StoppingCriteria):
        def __init__(self, actor: Actor):
            self.actor = actor
        
        def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs) -> torch.Tensor:
            sampled_eos = input_ids[:, -1].eq(self.actor.eos_id)
            generated_beyond_queued_prefix = torch.zeros(input_ids.shape[0], dtype=torch.bool, device="cpu")
            for index in range(input_ids.shape[0]):
                if input_ids.shape[1] > self.actor.batch.left_padding[index] + self.actor.rollout_manager(self.actor.batch.rollout_ids[index]).rollout(self.actor.batch.rollout_ids[index]).history_token_count:
                    generated_beyond_queued_prefix[index] = True
            generated_beyond_queued_prefix = generated_beyond_queued_prefix.to(self.actor.model.device)
            eligible_eos = sampled_eos & generated_beyond_queued_prefix
            return eligible_eos.any().expand(input_ids.shape[0])

    def __init__(self, model: Qwen3_5ForCausalLM, max_output_tokens: int = 4096, tail_threshold: int = 24, rebuild_threshold: float = 0.25):
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
        self.eos_id = int(self.tokenizer.eos_token_id)
        self.pad_id = self.tokenizer.pad_token_id
        self.tail_threshold = tail_threshold
        self.injector = LogitsProcessorList([self.Injector(self)])
        self.stopper = StoppingCriteriaList([self.Stopper(self)])
        self.rebuild_threshold = rebuild_threshold

    def _initialize(self, puzzles: list[list[int]]) -> None:
        self.rollout_managers: list[RolloutManager] = []
        starting_rollout_id = 0

        for puzzle in puzzles:
            prompt_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt(self.max_output_tokens, puzzle)}],
                tools=[calculator],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=True,
                return_dict=True,
                return_tensors="pt",
            )["input_ids"].squeeze(0)

            self.rollout_managers.append(RolloutManager(
                puzzle, starting_rollout_id, prompt_ids,
                [Rollout([Turn(torch.empty(0, dtype=torch.long), 0)], None, prompt_ids.numel()) for _ in range(self.group_size)]
            ))
            starting_rollout_id += self.group_size
        self.batch = None

    def rollout_manager(self, rollout_id: int) -> RolloutManager:
        return self.rollout_managers[rollout_id // self.group_size]

    def _unfinished_rollouts(self) -> list[int]:
        return [
            rollout_manager.starting_rollout_id + index
            for rollout_manager in self.rollout_managers
            for index, rollout in enumerate(rollout_manager.rollouts)
            if rollout.reward is None
        ]

    def _rebuild(self, rollout_ids: list[int]) -> None:
        histories = [self.rollout_manager(rollout_id).history(rollout_id) for rollout_id in rollout_ids]
        width = max(h.numel() for h in histories)
        ids = pad_sequence(histories, batch_first=True, padding_side="left", padding_value=self.pad_id).to(self.model.device)
        mask = pad_sequence([torch.ones_like(h) for h in histories], batch_first=True, padding_side="left", padding_value=0).to(self.model.device)
        cache = DynamicCache(config=self.model.config)
        self.batch = Batch(ids, mask, rollout_ids, [width - h.numel() for h in histories], cache, len(rollout_ids))
        for rollout_id, history in zip(rollout_ids, histories):
            self.rollout_manager(rollout_id).rollout(rollout_id).history_token_count = history.numel()

    def _evict_batch(self, rows_to_keep: list[int]) -> None:
        if not rows_to_keep:
            self.batch = None
            return

        self.batch.rollout_ids = [self.batch.rollout_ids[i] for i in rows_to_keep]
        self.batch.left_padding = [self.batch.left_padding[i] for i in rows_to_keep]
        rows_to_keep = torch.tensor(rows_to_keep, dtype=torch.long, device=self.model.device)
        self.batch.input_ids = self.batch.input_ids.index_select(0, rows_to_keep)
        self.batch.attention_mask = self.batch.attention_mask.index_select(0, rows_to_keep)
        self.batch.cache.reorder_cache(rows_to_keep)

    def _advance(self, remaining: list[int]) -> None:
        min_left_padding = min(self.batch.left_padding)
        limit = self.max_output_tokens - self.batch.input_ids.shape[1] + min_left_padding
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=self.batch.input_ids,
                attention_mask=self.batch.attention_mask,
                past_key_values=self.batch.cache,
                logits_processor=self.injector,
                stopping_criteria=self.stopper,
                max_new_tokens=limit,
                eos_token_id=None,
                return_dict_in_generate=True,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                repetition_penalty=1.0,
                top_k=0,
            )
            sequences = output.sequences
            last_tokens = sequences[:, -1].cpu().tolist()
            generated_len = sequences.shape[1] - self.batch.input_ids.shape[1]
            self.batch.input_ids = sequences
            self.batch.attention_mask = torch.cat([
                self.batch.attention_mask,
                self.batch.attention_mask.new_ones(self.batch.attention_mask.shape[0], generated_len),
            ], dim=1)
            self.batch.cache = output.past_key_values

        # check for things to evict (last token is eos)
        rows_to_keep = []
        refreshed_rows = [] # rows that have been refreshed with a new turn
        for index, row in enumerate(sequences):
            rollout_id = self.batch.rollout_ids[index]
            rollout = self.rollout_manager(self.batch.rollout_ids[index]).rollout(self.batch.rollout_ids[index])
            real_length = sequences.shape[1] - self.batch.left_padding[index]
            ended_on_eos = last_tokens[index] == self.eos_id and real_length > rollout.history_token_count
            if not (ended_on_eos or real_length >= self.max_output_tokens):
                # didn't EOS or reached max output tokens
                rows_to_keep.append(index)
                continue
            rollout.turns[-1].token_ids = sequences[index, self.batch.left_padding[index] + (rollout.history_token_count - rollout.turns[-1].token_ids.numel()):].cpu()
            rollout.history_token_count = real_length
            response = rollout.turns[-1].token_ids[rollout.turns[-1].completion_start:]

            text = self.tokenizer.decode(response, skip_special_tokens=True)
            tool_results = execute_tools(text, self.rollout_manager(rollout_id).puzzle)

            if any(verify_solution(result, self.rollout_manager(rollout_id).puzzle) for result in tool_results):
                rollout.reward = 1
                continue

            next_prompt = continuation_suffix(self.tokenizer, rollout.turns[-1].token_ids, tool_results)
            next_prompt_ids = self.tokenizer(next_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze(0)

            if real_length + next_prompt_ids.numel() >= self.max_output_tokens:
                rollout.reward = 0
            else:
                rollout.turns.append(Turn(next_prompt_ids, next_prompt_ids.numel()))
                rollout.history_token_count = self.rollout_manager(rollout_id).history(rollout_id).numel()
                refreshed_rows.append(index)

        # phase 1 just keep stuff we are still actively decoding, phase 2 (continuous) then keep everything except finished trajectories and inject user tokens
        if len(remaining) > self.tail_threshold and len(rows_to_keep) <= self.batch.cohort_start_size * self.rebuild_threshold:
            print(f"[Prefill Next] {len(rows_to_keep)} remain | {len(remaining) - len(rows_to_keep)} waiting")
            for row in rows_to_keep:
                rollout_id = self.batch.rollout_ids[row]
                rollout = self.rollout_manager(rollout_id).rollout(rollout_id)
                rollout.turns[-1].token_ids = sequences[row, self.batch.left_padding[row] + rollout.len_until_last_turn_prompt - rollout.turns[-1].completion_start:].cpu()
            self.batch = None
            return

        rows_to_keep = rows_to_keep if len(remaining) > self.tail_threshold else sorted(rows_to_keep + refreshed_rows)
        print(f"[Decode] {len(rows_to_keep)} rollouts | {len(remaining) - len(rows_to_keep)} waiting")
        if len(rows_to_keep) < len(self.batch.rollout_ids):
            self._evict_batch(rows_to_keep)

    def collect_rollouts(self, puzzles: list[list[int]], group_size: int = 8) -> list[list[Rollout]]:
        self.group_size = group_size
        start_time = time.time()

        try:
            self._initialize(puzzles)
            self.model.eval()
            while self._unfinished_rollouts():
                remaining = self._unfinished_rollouts()
                if self.batch is None:
                    self._rebuild(remaining)
                self._advance(remaining)
        finally:
            print(f"Collected rollouts in {time.time() - start_time:.2f} seconds")
        
        return [rollout_manager.rollouts for rollout_manager in self.rollout_managers]

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = Qwen3_5ForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16).to(device).eval()
    MAX_OUTPUT_TOKENS = 2048
    actor = Actor(model, max_output_tokens=MAX_OUTPUT_TOKENS, tail_threshold=4)
    puzzles = [[8, 3, 8, 3], [6, 6, 6, 6]]
    group_size = 8
    rollouts = actor.collect_rollouts(puzzles, group_size=group_size)
    for puzzle, group in zip(puzzles, rollouts):
        successes = sum(rollout.reward for rollout in group)
        prompt_len = actor.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt(MAX_OUTPUT_TOKENS, puzzle)}],
            tools=[calculator],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
            return_dict=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0).numel()
        print(f"Puzzle {puzzle}: {successes}/{group_size} successes")
        for i, rollout in enumerate(group, start=1):
            print(f"Rollout {i}: {len(rollout.turns)} turns, reward {rollout.reward}")
            summed_len = prompt_len
            for turn in rollout.turns:
                summed_len += turn.token_ids.numel()
            print(f"Summed length: {summed_len} / {MAX_OUTPUT_TOKENS}")