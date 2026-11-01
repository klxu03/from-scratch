import ast
import operator as ops
from fractions import Fraction
from collections import Counter
import re

def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression using +, -, *, / and parentheses.
    The calculator will return both the result as well as process the input numbers given and return numbers used with feedback about missing or extra numbers not in the original puzzle.

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
        return "ERROR: Expression too long"

    if "#" in expression or "=" in expression:
        return "ERROR: Comments and equality signs are not allowed"

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

        return "ERROR: Only arithmetic expressions are allowed"

    return str(evaluate(tree.body))

def numbers_used(expression: str) -> list[int]:
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

    return numbers_used

def verify_solution(expression: str, puzzle: list[int]) -> bool:
    parts = expression.split(" = ")
    if len(parts) != 2 or "ERROR" in expression:
        return False
    expression, result = parts

    nums_used = numbers_used(expression)
    if nums_used is False:
        return False

    # return used all numbers and expression evaluates to 24
    return result == "24" and Counter(nums_used) == Counter(puzzle) 

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
        except Exception as err:
            result = f"ERROR: expression got error {err}"

        print(f"Calculator {expression} has result: {result}")
        if "ERROR" in result:
            ret.append(result)
        else:
            ret.append(f"{expression} = {result}")
    return ret

def tool_response(tool_results: list[str], puzzle: list[int]) -> list[str]:
    responses = []
    for result in tool_results:
        if "ERROR" in result:
            responses.append(result)
            continue

        expression, _, value = result.partition(" = ")
        used = numbers_used(expression)
        if used is False:
            responses.append(f"Calculator result: {value}\nNumbers used: invalid\nInvalid card usage.")
            continue
        missing = list((Counter(puzzle) - Counter(used)).elements())
        extra = list((Counter(used) - Counter(puzzle)).elements())
        lines = [
            f"Expression checked: {expression}",
            f"Calculator result: {value}",
            f"Numbers used: {used}",
        ]
        if missing or extra:
            lines += ["Invalid card usage.", f"Missing: {missing}", f"Extra: {extra}"]
        responses.append("\n".join(lines))
    return responses