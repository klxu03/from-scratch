import sys
from pathlib import Path

import torch
import re
from transformers import AutoTokenizer, Qwen3_5ForCausalLM
from tqdm import tqdm

import ast
import operator as ops
from fractions import Fraction
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.prompts import prompt

MODEL = "Qwen/Qwen3.5-0.8B"
MAX_OUTPUT_TOKENS = 4_096

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = Qwen3_5ForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("mps").eval()

def calculator(expression: str) -> str:
	# Must give docstring of this type with Args info
    """Evaluate an arithmetic expression using +, -, *, / and parentheses.

    Args:
        expression: Expression to evaluate, e.g. "(6 * 12) / (4 + 4)".
    """
    operators = {
        ast.Add: ops.add,
        ast.Sub: ops.sub,
        ast.Mult: ops.mul,
        ast.Div: ops.truediv,
    }

    if len(expression) > 128:
        raise ValueError("Expression too long")

    tree = ast.parse(expression, mode="eval")

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return Fraction(str(node.value))

        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            return operators[type(node.op)](
                evaluate(node.left), evaluate(node.right)
            )

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value

        raise ValueError("Only arithmetic expressions are allowed")

    return str(evaluate(tree.body))

def verify_solution(expression: str, puzzle: list[int]) -> bool:
	tree = ast.parse(expression, mode="eval")

	numbers_used = [
		node.value
		for node in ast.walk(tree)
		if isinstance(node, ast.Constant) and type(node.value) is int
	]

	return Counter(numbers_used) == Counter(puzzle)

def extend_cached_ids(prefix_ids: torch.Tensor, suffix: str) -> torch.Tensor | None:
	"""Append suffix tokens without changing the ids the cache was built from."""
	prefix = prefix_ids[0].tolist()
	prefix_text = tokenizer.decode(prefix, skip_special_tokens=False)
	extended = tokenizer(prefix_text + suffix, add_special_tokens=False)["input_ids"]
	if extended[:len(prefix)] != prefix:
		return None
	return torch.tensor([extended], dtype=prefix_ids.dtype, device=prefix_ids.device)

def continuation_suffix(prefix_ids: torch.Tensor, tool_results: list[str]) -> str:
	ids = prefix_ids[0].tolist()
	end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
	newline_id = tokenizer.encode("\n", add_special_tokens=False)[0]
	already_closed = len(ids) >= 2 and ids[-2] == end_id and ids[-1] == newline_id
	close = "" if already_closed else "\n" if ids[-1] == end_id else "<|im_end|>\n"
	tools = "".join(f"\n<tool_response>\n{result}\n</tool_response>" for result in tool_results)
	if tools:
		tools = f"<|im_start|>user{tools}<|im_end|>\n"
	return close + tools + "<|im_start|>assistant\n<think>\n"

def run_episode(puzzle: list[int]) -> str | None:
	prompt_text = prompt(MAX_OUTPUT_TOKENS, puzzle)
	TOKENS_USED = 0
	turn = 0
	messages = [{"role": "user", "content": prompt_text}]
	cache = None
	next_ids = None

	while TOKENS_USED < MAX_OUTPUT_TOKENS:
		turn += 1

		if next_ids is None:
			inputs = tokenizer.apply_chat_template(
				messages,
				tools=[calculator],
				tokenize=True,
				add_generation_prompt=True,
				enable_thinking=True,
				return_dict=True,
				return_tensors="pt",
			).to("mps")
		else:
			print(f"Cache hit at turn {turn}")
			inputs = {
				"input_ids": next_ids,
				"attention_mask": torch.ones_like(next_ids),
			}

		input_ids = inputs["input_ids"]

		with torch.inference_mode():
			outputs = model.generate(
				**inputs,
				past_key_values=cache,
				return_dict_in_generate=True,
				max_new_tokens=MAX_OUTPUT_TOKENS - TOKENS_USED,
				do_sample=True, temperature=1.0, top_k=20,
				top_p=0.95, repetition_penalty=1.1,
			)

		cache = outputs.past_key_values
		sequences = outputs.sequences
		raw_response = sequences[0, input_ids.shape[1]:]
		text = tokenizer.decode(raw_response, skip_special_tokens=True)
		TOKENS_USED += raw_response.numel()

		messages.append({
			"role": "assistant",
			"content": text
		})

		print(f"\nGenerated tokens: {TOKENS_USED} / {MAX_OUTPUT_TOKENS} on turn {turn}")
		print("Finished thinking:", "</think>" in text)
		print("Output:", text)

		matches = re.findall(
			r"<tool_call>\s*<function=calculator>\s*"
			r"<parameter=expression>(.*?)</parameter>\s*"
			r"</function>\s*</tool_call>",
			text,
			re.S,
		)

		tool_results = []
		for raw_expression in matches:
			expression = raw_expression.strip().strip("\"'")
			try:
				result = calculator(expression)
			except Exception as ignored:
				continue

			print(f"Calculator {expression} = {result}")
			if result == "24" and verify_solution(expression, puzzle):
				print(f"Found a valid solution: {expression}")
				return expression

			tool_results.append(f"{expression} = {result}")
			messages.append({
				"role": "tool",
				"content": f"{expression} = {result}"
			})

		next_ids = extend_cached_ids(sequences, continuation_suffix(sequences, tool_results))
		if next_ids is None:
			print(f"Cache mismatch after turn {turn}")
			cache = None

# (8*(9/(1+2))) # 0.8b got it about ~50% of the time
# (3*((3*5)-7)) # 0.8b got it 1/96 times took 44m, 4b got it 18/100 times took 3h
# 8/(3 - (8/3)) # 0.8b got it 1/42 times, 4b got it 9/23 times
INPUT_NUMBERS = [8, 1, 2, 9]

num_successes= []
progress = tqdm(range(100), desc="Running episodes")
for i in progress:
	progress.set_postfix(successes=len(num_successes), refresh=False)
	res = run_episode(INPUT_NUMBERS)
	if res:
		num_successes.append(res)

print(f"Number of successes: {len(num_successes)}")
print(f"Successes: {num_successes}")

"""
import csv
import random

DATA_DIR = Path(__file__).resolve().parents[1] / "24game/data"
RESULTS_PATH = Path(__file__).with_name("results.csv")

NUM_PUZZLES = 4
NUM_TRIALS = 10

selected = []

for difficulty in ["easy", "medium", "hard"]:
    path = DATA_DIR / f"{difficulty}.jsonl"
    puzzles = [
        list(map(int, line.split()))
        for line in path.read_text().splitlines()
        if line.strip()
    ]

    selected.extend(
        (numbers, difficulty)
        for numbers in random.sample(puzzles, NUM_PUZZLES)
    )

with RESULTS_PATH.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["problem", "difficulty", f"successes/{NUM_TRIALS}"])

    progress = tqdm(total=len(selected) * NUM_TRIALS, desc="Evaluating")

    for numbers, difficulty in selected:
        successes = 0

        for _ in range(NUM_TRIALS):
            successes += run_episode(numbers) is not None
            progress.update(1)
            progress.set_postfix(
                puzzle=str(numbers),
                successes=successes,
            )

        writer.writerow([" ".join(map(str, numbers)), difficulty, successes])
        f.flush()  # Save each puzzle's results immediately

    progress.close()

print(f"Results saved to {RESULTS_PATH}")
"""