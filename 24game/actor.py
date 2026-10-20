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
        input_ids = torch.cat([input_ids, prompt_ids.to(input_ids.device)], dim=1)

        with torch.inference_mode():
            outputs = self.model.generate(
				input_ids=input_ids,
				attention_mask=torch.ones_like(input_ids),
				past_key_values=cache,
				return_dict_in_generate=True,
				max_new_tokens=self.max_output_tokens - input_ids.shape[1],
				do_sample=True, temperature=1.0, top_k=20,
				top_p=0.95, repetition_penalty=1.1,
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

    def collect_rollouts(self, puzzles: list[list[int]], group_size: int = 8) -> list[list[Rollout]]:
        ret: list[list[Rollout]] = []
        for puzzle in puzzles:
            ret.append([])
            prompt_text = prompt(self.max_output_tokens, puzzle)
            for _ in tqdm(range(group_size), desc=f"Collecting {group_size} rollouts for puzzle {puzzle}"):
                turns: list[Turn] = []
                input_ids = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt_text}],
                    tools=[calculator],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to("mps")["input_ids"]
                curr_exec = ExecutionResult(turn=None, cache=None, next_prompt="", tool_results=[])
                found_sol = False
                while not found_sol and input_ids.shape[1] < self.max_output_tokens:
                    print(f"\nNew turn, so far generated {input_ids.shape[1]}/{self.max_output_tokens} tokens")
                    try:
                        curr_exec = self.execute_turn(curr_exec.cache, input_ids, curr_exec.next_prompt, puzzle)
                    except Exception as e:
                        print(f"Error executing turn {curr_exec.next_prompt}: {e}")
                        turns = []
                        break
                    turns.append(curr_exec.turn)
                    input_ids = torch.cat([input_ids, curr_exec.turn.token_ids.to(input_ids.device)], dim=1)
                    for tool_result in curr_exec.tool_results:
                        found_sol = found_sol or verify_solution(tool_result, puzzle)
                print(f"Found solution: {found_sol}")
                ret[-1].append(Rollout(turns=turns, reward=found_sol))
        return ret

if __name__ == "__main__":
    model = Qwen3_5ForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B", dtype=torch.bfloat16).to("mps").eval()
    actor = Actor(model)
    puzzles = [[6, 6, 6, 6], [8, 1, 2, 9], [3, 5, 3, 7], [8, 3, 3, 8]]
    group_size = 10
    rollouts = actor.collect_rollouts(puzzles, group_size=group_size)
    for puzzle, group in zip(puzzles, rollouts):
        successes = sum(rollout.reward for rollout in group)
        print(f"Puzzle {puzzle}: {successes}/{group_size} successes")
        for i, rollout in enumerate(group, start=1):
            print(f"  Rollout {i}: {len(rollout.turns)} turns, reward {rollout.reward}")