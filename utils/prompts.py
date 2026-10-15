def prompt(max_output_tokens: int, input_numbers: list[int]) -> str:
	return f"""
Solve the Game of 24.

Use each number exactly once.
Only +, -, *, / and parentheses are allowed. 
Only these operations allowed, no exp **, factorial !, floor divide //, equal =, etc.
Make sure to use parentheses to group operations you want to perform first.

You are given a token budget of {max_output_tokens} tokens across all turns.

You have access to a calculator tool that accepts a complete arithmetic expression as a string and returns its exact value.

Use the calculator to test and verify candidate expressions. You may send several calculator tool calls in one response. Every call is executed, and each result is returned before your next turn.

<tool_call>
<function=calculator>
<parameter=expression>
4 / (1 - 5 / 6)
</parameter>
</function>
</tool_call>

You end successfully when one of your attempted tool calls that use all the numbers once evaluates to 24.
You do not end by thinking through the right answer, only by returning a proper calculator tool call.

Input numbers: {input_numbers}
"""
