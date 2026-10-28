# RL Trainer of a LLM for 24 Game

This is the repo for my hand-coded project to RL a LLM to be good at the [24 Game](https://en.wikipedia.org/wiki/24_(puzzle)) where you are given 4 numbers (presumably cards) and you have to figure out a way using all the numbers and only elementary math operations to result in 24.

From a base model, this project applies LoRA finetuning on a custom generate dataset of 24-game puzzles. The RL algorithm implemented is a simple GRPO with one-step gradient updates per batch, with 0-1 reward if the puzzle was solved or not, and a KL penalty for the currently training finetuned model versus original base model (with LoRA adapters off).

## Installation

From the repo root, install Python 3.14 and dependencies into `.venv`:
```bash
uv sync --locked --python 3.14 --extra cuda
```
or on Mac without CUDA just run:
```bash
uv sync --locked
```

To verify CUDA and accelerated libraries
```bash
uv run --locked --extra cuda python -c "import torch, causal_conv1d, fla; print(torch.cuda.get_device_name())"
```

## Generate the dataset
```bash
uv run --locked 24game/dataset.py
```

This writes `24game/data/easy.jsonl`, `24game/data/medium.jsonl`, and `24game/data/hard.jsonl`, then splits them into `24game/data/train.jsonl`, `24game/data/val.jsonl`, and `24game/data/test.jsonl`. The split files are opened in append mode, so run this once. Inside the `split_dataset()` method you can configure the percentage of data to be in validation and test.

## Train
On CUDA:
```bash
uv run --locked --extra cuda python 24game/train.py
PYTORCH_ALLOC_CONF=expandable_segments:True nohup uv run --locked --extra cuda python -u 24game/train.py > run.run 2>&1 &
PYTORCH_ALLOC_CONF=expandable_segments:True nohup uv run --locked --extra cuda python -u 24game/eval.py > eval.run 2>&1 &
```

On Mac:
```bash
uv run 24game/train.py
```

The first run downloads `Qwen/Qwen3.5-0.8B`. Training uses CUDA when it is available, otherwise MPS, otherwise CPU. Checkpoints and the reward plot are written to `24game/checkpoints/`.

## Environment

Each puzzle is four integers. The model is given a fixed input found in `utils/prompts.py`. The model writes out text, and the environment will parse the outputted text for calculator tool calls. Tool calls look like:

```
<tool_call>
<function=calculator>
<parameter=expression>
(8 * 3) * (3 - 1)
</parameter>
</function>
</tool_call>
```

Every match in the response is executed. Quotes around the expression are stripped. The calculator parses the expression and evaluates it with exact fractions, so division does not become a float. It allows integer and float literals, `+`, `-`, `*`, `/`, parentheses, and unary plus and minus. 
Each accepted call is wrapped in a `<tool_response>` and shown to the model on the next turn. The response includes the calculator result and the numbers used. The expression has to use all four puzzle numbers exactly once, so when it does not, the response also lists the missing and extra digits:

```
Calculator result: 48
Numbers used: [8, 3, 3, 1]
Invalid card usage.
Missing: [8]
Extra: [1]
```

I treat that call as a success when the calculator result is `24`, every literal in the expression is an integer, and those integers are exactly the puzzle numbers, each used once.