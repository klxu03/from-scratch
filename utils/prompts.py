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

Keep the search concise. Preserve each candidate as a full expression over the four original card occurrences; intermediate values are not extra cards.
When you think a complete candidate reaches 24, immediately submit that EXACT expression to the calculator. Do not replace it with its simplified numeric value, change its grouping, or keep reasoning instead of submitting it.
Check plausible complete candidates with the calculator rather than discarding them solely on mental arithmetic. Test one or a few distinct candidates per turn and use the feedback.

The expression argument must contain arithmetic only: no equality sign, question mark, explanation, or placeholder name.
Keep fractions exact and avoid repeating already tested expressions.

If no candidate is apparent, choose a final operation and work backward from 24. 
Explore different ways to divide the four card occurrences into two nonempty groups, including one-card versus three-card groups.

Keep each subexpression attached to the original cards it used, and combine only disjoint groups. 
Retain the full expression, not just its numeric value. 
Fractional intermediate results are allowed.

Backtrack to a different grouping when an approach fails rather than listing the same calculations again.

Input numbers: {input_numbers}
"""
