import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM
from transformers.generation.utils import GenerateOutput
from env import calculator, verify_solution, execute_tools
from typing import NamedTuple
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.prompts import prompt

"""
Notes:
Usually I'd collect old_logprobs of the selected action in the Actor during rollout, but since I'm doing one gradient update per batch, this is unnecessary
"""

class Turn(NamedTuple):
    """
    A single turn of the actor, containing this turn's token ids. Former is prompt, latter is completion
    The very first turn omits the prompt (as it's just the initial prompt), future turns are just continuations
    """
    token_ids: torch.Tensor
    completion_start: int

class Rollout(NamedTuple):
    turns: list[Turn]
    reward: float

class ExecutionResult(NamedTuple):
    turn: Turn
    cache: dict[str, torch.Tensor]
    next_prompt: str
    tool_results: list[str]

class Actor:
    def __init__(self, model: Qwen3_5ForCausalLM, max_output_tokens: int = 4096):
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
        self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    def continuation_suffix(self, prefix_ids: torch.Tensor, tool_results: list[str]) -> str:
        ids = prefix_ids[0].tolist()
        end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        newline_id = self.tokenizer.encode("\n", add_special_tokens=False)[0]
        already_closed = len(ids) >= 2 and ids[-2] == end_id and ids[-1] == newline_id
        close = "" if already_closed else "\n" if ids[-1] == end_id else "<|im_end|>\n"
        tools = "".join(f"\n<tool_response>\n{result}\n</tool_response>" for result in tool_results)
        if tools:
            tools = f"<|im_start|>user{tools}<|im_end|>\n"
        return close + tools + "<|im_start|>assistant\n<think>\n"


    def execute_turn(self, cache: dict[str, torch.Tensor] | None, input_ids: torch.Tensor, prompt: str, puzzle: list[int]) -> ExecutionResult:
        """
        Execute a turn of the actor, modifying the cache reference in-place
        """
        original_input_length = input_ids.shape[1]
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
        input_ids = torch.cat([input_ids, prompt_ids.to(self.model.device)], dim=1)

        with torch.inference_mode():
            outputs = self.model.generate(
				input_ids=input_ids,
				attention_mask=torch.ones_like(input_ids),
				past_key_values=cache,
				return_dict_in_generate=True,
				max_new_tokens=self.max_output_tokens - input_ids.shape[1],
				do_sample=True, temperature=1.0, top_k=0,
				top_p=1.0, repetition_penalty=1.0,
			)

        sequences = outputs.sequences
        raw_response = sequences[0, input_ids.shape[1]:]
        text = self.tokenizer.decode(raw_response, skip_special_tokens=True)
        print(f"Generated text: {text}")
        tool_results = execute_tools(text, puzzle)
        print(f"Tool results: {tool_results}")

        next_prompt = self.continuation_suffix(sequences, tool_results)

        generated = sequences[:, original_input_length + prompt_ids.shape[1]:]
        turn_ids = torch.cat([prompt_ids, generated.cpu()], dim=1)

        return ExecutionResult(
            turn=Turn(token_ids=turn_ids, completion_start=prompt_ids.shape[1]),
            cache=outputs.past_key_values,
            next_prompt=next_prompt,
            tool_results=tool_results,
        )

    def execute_first_turn(self, prompt_ids: torch.Tensor, puzzle: list[int], group_size: int) -> list[ExecutionResult]:
        prompt_ids = prompt_ids.to(self.model.device)
        with torch.inference_mode():
            sequences = self.model.generate(
                input_ids=prompt_ids,
                attention_mask=torch.ones_like(prompt_ids),
                num_return_sequences=group_size,
                num_beams=1,
                do_sample=True,
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                repetition_penalty=1.0,
                max_new_tokens=self.max_output_tokens - prompt_ids.shape[1],
                use_cache=True,
                return_dict_in_generate=False,
                output_scores=False,
                output_logits=False,
            )

        eos = self.model.generation_config.eos_token_id
        ret = []

        for row in sequences.cpu():
            response = row[prompt_ids.shape[1]:]
            stops = (response == eos).nonzero().flatten()
            response = response[:stops[0] + 1] if stops.numel() else response

            text = self.tokenizer.decode(response, skip_special_tokens=True)
            tool_results = execute_tools(text, puzzle)

            ret.append(ExecutionResult(
                turn=Turn(response.unsqueeze(0), completion_start=0),
                cache=None,
                next_prompt=self.continuation_suffix(row[:prompt_ids.shape[1] + len(response)].unsqueeze(0), tool_results),
                tool_results=tool_results,
            ))
        return ret

    def collect_rollouts(self, puzzles: list[list[int]], group_size: int = 8) -> list[list[Rollout]]:
        ret: list[list[Rollout]] = []
        for puzzle in puzzles:
            prompt_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt(self.max_output_tokens, puzzle)}],
                tools=[calculator],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)["input_ids"]

            first_turns = self.execute_first_turn(prompt_ids, puzzle, group_size)

            group = []
            for curr_exec in tqdm(first_turns, desc=f"Finishing rollouts for {puzzle}"):
                turns = [curr_exec.turn]
                input_ids = torch.cat([prompt_ids, curr_exec.turn.token_ids.to(self.model.device)], dim=1)

                found_sol = any(verify_solution(result, puzzle) for result in curr_exec.tool_results)
                while not found_sol:
                    suffix_len = len(self.tokenizer.encode(curr_exec.next_prompt, add_special_tokens=False))
                    if input_ids.shape[1] + suffix_len >= self.max_output_tokens:
                        break

                    curr_exec = self.execute_turn(curr_exec.cache, input_ids, curr_exec.next_prompt, puzzle)
                    turns.append(curr_exec.turn)
                    input_ids = torch.cat([input_ids, curr_exec.turn.token_ids.to(self.model.device)], dim=1)
                    found_sol = any(verify_solution(result, puzzle) for result in curr_exec.tool_results)
                group.append(Rollout(turns=turns, reward=found_sol))
                print(f"Found solution: {found_sol} for puzzle {puzzle}\n")
            ret.append(group)
        return ret
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = Qwen3_5ForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16).to(device).eval()
    actor = Actor(model)
    puzzles = [[8, 3, 8, 3]]
    group_size = 8
    rollouts = actor.collect_rollouts(puzzles, group_size=group_size)
    for puzzle, group in zip(puzzles, rollouts):
        successes = sum(rollout.reward for rollout in group)
        print(f"Puzzle {puzzle}: {successes}/{group_size} successes")
        for i, rollout in enumerate(group, start=1):
            print(f"  Rollout {i}: {len(rollout.turns)} turns, reward {rollout.reward}")