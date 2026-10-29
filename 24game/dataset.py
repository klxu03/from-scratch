import operator as ops
from fractions import Fraction
from typing import NamedTuple
from itertools import combinations_with_replacement, combinations
import sympy
import re
from tqdm import tqdm
from pathlib import Path
from math import ceil
import random

SEED = 800
rng = random.Random(SEED)

class Term(NamedTuple):
    value: Fraction
    expr: str

OPERATORS = [
    (ops.add, "+"),
    (ops.sub, "-"),
    (lambda left, right: ops.sub(right, left), "r-"),
    (ops.mul, "*"),
    (ops.truediv, "/"),
    (lambda left, right: ops.truediv(right, left), "r/"),
]

def evaluate(terms: list[Term] | list[int]) -> list[Term]:
    """
    Given an initial starting number and a list of next numbers, return a list of all possible results.
    """
    if terms and isinstance(terms[0], int):
        terms = [Term(Fraction(n), str(n)) for n in terms]

    results = []

    if len(terms) == 1 and terms[0].value == 24:
        return [Term(24, terms[0].expr)]

    for left, right in combinations(terms, 2):
        for op in OPERATORS:
            try:
                result = op[0](left.value, right.value)
            except ZeroDivisionError:
                continue

            new_input = (next_inp := terms.copy(), next_inp.remove(left), next_inp.remove(right))[0]

            if op[1].startswith("r"):
                next_expr = f"({right.expr} {op[1].removeprefix("r")} {left.expr})"
            else:
                next_expr = f"({left.expr} {op[1]} {right.expr})"

            results.extend(evaluate([Term(result, next_expr)] + new_input))
    
    return results

def canonical(expr: str):
    return sympy.srepr(sympy.sympify(expr, evaluate=False))

def uses_fraction(expr: str) -> bool:
    while m := re.search(r"\(([^()]+)\)", expr):
        left, op, right = m.group(1).split()
        value = {"+": ops.add, "-": ops.sub, "*": ops.mul, "/": ops.truediv}[op](Fraction(left), Fraction(right))
        if value.denominator != 1:
            return True
        expr = expr.replace(m.group(0), str(value), 1)
    return False

def generate_dataset():
    puzzles = sorted(combinations_with_replacement(range(1, 14), 4))
    rng.shuffle(puzzles)
    puzzles = [rng.sample(puzzle, len(puzzle)) for puzzle in puzzles]

    easy = [] # remaining
    medium = [] # only one answer
    hard = [] # one answer that needs fractions

    for puzzle in tqdm(puzzles):
        input = [Term(Fraction(x), str(x)) for x in sorted(puzzle)]
        res = evaluate(input)
        unique = {}
        for r in res:
            unique[canonical(r.expr)] = r.expr

        if len(res) == 0 or len(unique.values()) > 1:
            if len(res) > 0: easy.append(puzzle)
            continue

        if uses_fraction(next(iter(unique.values()))):
            hard.append(puzzle)
        else:
            medium.append(puzzle)

    print(f"easy: {len(easy)}, medium: {len(medium)}, hard: {len(hard)}")

    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    for name, group in [("easy", easy), ("medium", medium), ("hard", hard)]:
        lines = []
        for puzzle in group:
            lines.append(" ".join(map(str, puzzle)) + "\n")
        (data_dir / f"{name}.jsonl").write_text("".join(lines), encoding="utf-8")

def split_dataset():
    val_percentage = 0.1
    test_percentage = 0.1

    data_dir = Path(__file__).resolve().parent / "data"
    for name in ["easy", "medium", "hard"]:
        path = data_dir / f"{name}.jsonl"
        puzzles = [
            list(map(int, line.split()))
            for line in path.read_text().splitlines()
            if line.strip()
        ]
        rng.shuffle(puzzles)
        val_count = ceil(len(puzzles) * val_percentage)
        test_count = ceil(len(puzzles) * test_percentage)
        val_puzzles = puzzles[:val_count]
        test_puzzles = puzzles[val_count:val_count + test_count]
        train_puzzles = puzzles[val_count + test_count:]

        for puzzles, split in [(train_puzzles, "train"), (val_puzzles, "val"), (test_puzzles, "test")]:
            with open(data_dir / f"{split}.jsonl", "a") as f:
                for puzzle in puzzles:
                    f.write(" ".join(map(str, puzzle)) + "\n")

def sort_puzzles():
    data_dir = Path(__file__).resolve().parent / "data"
    for name in ["train", "val", "test", "easy", "medium", "hard"]:
        path = data_dir / f"{name}.jsonl"
        puzzles = [
            sorted(int(n) for n in line.split())
            for line in path.read_text().splitlines()
            if line.strip()
        ]
        with open(data_dir / f"{name}.jsonl", "w") as f:
            for puzzle in puzzles:
                f.write(" ".join(map(str, puzzle)) + "\n")

if __name__ == "__main__":
    generate_dataset()
    split_dataset()
    sort_puzzles()