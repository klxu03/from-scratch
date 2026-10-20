import ast
import operator as ops
from fractions import Fraction
from collections import Counter
import re

def calculator(expression: str) -> str:
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
    expression, result = expression.split(" = ")
    tree = ast.parse(expression, mode="eval")

    if any(
        isinstance(node, ast.Constant) and type(node.value) is not int
        for node in ast.walk(tree)
    ):
        return False

    numbers_used = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and type(node.value) is int
    ]

    # return used all numbers and expression evaluates to 24
    return result == "24" and Counter(numbers_used) == Counter(puzzle) 

def execute_tools(text: str, puzzle: list[int]) -> list[str]:
    matches = re.findall(
        r"<tool_call>\s*<function=calculator>\s*"
        r"<parameter=expression>(.*?)</parameter>\s*"
        r"</function>\s*</tool_call>",
        text,
        re.S,
    )

    ret = []
    for raw_expression in matches:
        expression = raw_expression.strip().strip("\"'")
        try:
            result = calculator(expression)
        except Exception as ignored:
            continue

        print(f"Calculator {expression} = {result}")
        ret.append(f"{expression} = {result}")
    return ret