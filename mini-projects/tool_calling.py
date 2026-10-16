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

def run_episode(puzzle: list[int]) -> str | None:
	prompt_text = prompt(MAX_OUTPUT_TOKENS, puzzle)
	TOKENS_USED = 0
	turn = 0
	messages = [{"role": "user", "content": prompt_text}]

	while TOKENS_USED < MAX_OUTPUT_TOKENS:
		turn += 1

		inputs = tokenizer.apply_chat_template(
			messages,
			tools=[calculator],
			tokenize=True,
			add_generation_prompt=True,
			enable_thinking=True,
			return_dict=True,
			return_tensors="pt",
		).to("mps")

		with torch.inference_mode():
			outputs = model.generate(
				**inputs, max_new_tokens=MAX_OUTPUT_TOKENS - TOKENS_USED,
				do_sample=True, temperature=1.0, top_k=20,
				top_p=0.95, repetition_penalty=1.1,
			)

		raw_response = outputs[0, inputs["input_ids"].shape[1]:]
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

			messages.append({
				"role": "tool",
				"content": f"{expression} = {result}"
			})

# (8*(9/(1+2))) # 0.8b got it about ~50% of the time
# (3*((3*5)-7)) # 0.8b got it 1/96 times took 44m, 4b got it 18/100 times took 3h
# 8/(3 - (8/3)) # 0.8b got it 1/42 times, 4b got it 9/23 times
INPUT_NUMBERS = [6, 6, 6, 6]

num_successes= []
progress = tqdm(range(1), desc="Running episodes")
for i in progress:
	progress.set_postfix(successes=len(num_successes), refresh=False)
	res = run_episode(INPUT_NUMBERS)
	if res:
		num_successes.append(res)

print(f"Number of successes: {len(num_successes)}")
print(f"Successes: {num_successes}")