import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import DynamicCache
from typing import NamedTuple
from transformers import AutoTokenizer, Qwen3_5ForCausalLM, LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList
from dataclasses import dataclass
import time
from env import calculator, verify_solution, execute_tools

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.prompts import prompt

class Turn(NamedTuple):
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
            for index, candidate in enumerate(sampled_eos):
                if candidate:
                    rollout_id = self.actor.batch.rollout_ids[index]
                    next_id_from_back = self.actor.get_next_injected_token_index(rollout_id, input_ids.shape[1])
                    if next_id_from_back <= 0:
                        return torch.tensor(True, device=self.actor.model.device).expand(input_ids.shape[0])

            return torch.tensor(False, device=self.actor.model.device).expand(input_ids.shape[0])

    def __init__(self, model: Qwen3_5ForCausalLM, max_output_tokens: int = 4096, tail_threshold: int = -1, rebuild_threshold: float = 0.25):
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
        self.eos_id = int(self.tokenizer.eos_token_id)
        self.pad_id = self.tokenizer.pad_token_id
        self.tail_threshold = tail_threshold
        self.injector = LogitsProcessorList([self.Injector(self)])
        self.stopper = StoppingCriteriaList([self.Stopper(self)])

    def _initialize(self, puzzles: list[list[int]]) -> None:
        self.rollout_managers: list[RolloutManager] = []
        starting_rollout_id = 0
        if self.tail_threshold < 0: self.tail_threshold = len(puzzles) * self.group_size // 8

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
            sequences = output.sequences.cpu()

        # check for things to evict (last token is eos)
        rows_to_keep = []
        refreshed_rows = [] # rows that have been refreshed with a new turn
        for index, row in enumerate(sequences):
            rollout = self.rollout_manager(self.batch.rollout_ids[index]).rollout(self.batch.rollout_ids[index])
            response = row[(self.batch.left_padding[index] + rollout.history_token_count):]
            real_length = sequences.shape[1] - self.batch.left_padding[index]
            if not (response[-1] == self.eos_id or real_length >= self.max_output_tokens):
                # didn't EOS or reached max output tokens
                rows_to_keep.append(index)
                continue

            text = self.tokenizer.decode(response, skip_special_tokens=True)
            tool_results = execute_tools(text, self.rollout_manager(self.batch.rollout_ids[index]).puzzle)
            rollout.turns[-1].token_ids = torch.cat([rollout.turns[-1].token_ids, response])

            if any(verify_solution(result, self.rollout_manager(self.batch.rollout_ids[index]).puzzle) for result in tool_results):
                rollout.reward = 1
            elif real_length >= self.max_output_tokens:
                rollout.reward = 0
            else:
                # make sure to handle edge case where new next_prompt will push over generation limit
                next_prompt = continuation_suffix(self.tokenizer, rollout.turns[-1].token_ids, tool_results)
                next_prompt_ids = self.tokenizer(next_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze(0)
                rollout.turns[-1].append(Turn(next_prompt_ids, next_prompt_ids.numel()))
                rollout.history_token_count += next_prompt_ids.numel() + response.numel()
                refreshed_rows.append(index)
        
        # phase 1 just keep stuff we are still actively decoding, phase 2 (continuous) then keep everything except finished trajectories and inject user tokens
        if len(remaining) > self.tail_threshold and len(remaining) <= self.batch.cohort_start_size * self.rebuild_threshold:
            self.batch = None
            return

        rows_to_keep = rows_to_keep if len(remaining) > self.tail_threshold else sorted(rows_to_keep + refreshed_rows)
        if len(rows_to_keep) < len(remaining):
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
                    # add logging if we ended up prefilling, and if not decode how many active right now
                    self._rebuild(remaining)
                self._advance(remaining)
        finally:
            print(f"Collected rollouts in {time.time() - start_time:.2f} seconds")
        
        return [rollout_manager.rollouts for rollout_manager in self.rollout_managers]