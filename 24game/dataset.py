import operator as ops
from fractions import Fraction
from typing import NamedTuple
from itertools import combinations_with_replacement, combinations
import sympy
import re
from tqdm import tqdm

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

def evaluate(terms: list[Term]) -> list[Term]:
    """
    Given an initial starting number and a list of next numbers, return a list of all possible results.
    """
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
    symbolic = re.sub(
        r"\d+",
        lambda m: f"x{m.group()}",
        expr
    )
    return sympy.cancel(sympy.sympify(symbolic))

input = [Term(Fraction(x), str(x)) for x in [8, 3, 8, 3]]
res = evaluate(input)
unique = {}

for r in res:
    unique[canonical(r.expr)] = r.expr
for v in unique.values():
    print(v)

puzzles = list(combinations_with_replacement(range(1, 14), 4))

num_solutions = 0
total_unique_num_solutions = 0

for puzzle in tqdm(puzzles):
    input = [Term(Fraction(x), str(x)) for x in puzzle]
    res = evaluate(input)
    num_solutions += (len(res) > 0)
    unique = {canonical(r.expr) for r in res}
    total_unique_num_solutions += len(unique)

print(f"num_solutions = {num_solutions} out of {len(puzzles)}, avg_unique_num_solutions = {total_unique_num_solutions / num_solutions}")