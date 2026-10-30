import random
from pathlib import Path
from transformers import Qwen3_5ForCausalLM
import torch
from tqdm import tqdm
from actor import Actor
from dataset import evaluate
from learner import Learner
from lora_linear import apply_lora
from utils.memory import empty_device_cache

MODEL = "Qwen/Qwen3.5-0.8B"
MAX_OUTPUT_TOKENS = 4096
DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
if DEVICE.type == "cuda":
    torch.set_float32_matmul_precision("high")
model = Qwen3_5ForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE)
model = apply_lora(model)
actor = Actor(model, MAX_OUTPUT_TOKENS)
learner = Learner(model, MAX_OUTPUT_TOKENS)

EASY_PUZZLES = Path(__file__).resolve().parent / "data/easy.jsonl"
MEDIUM_PUZZLES = Path(__file__).resolve().parent / "data/medium.jsonl"
HARD_PUZZLES = Path(__file__).resolve().parent / "data/hard.jsonl"
PUZZLE_GROUPS = ("easy", "medium", "hard")

VAL_PATH = Path(__file__).resolve().parent / "data/val.jsonl"
TEST_PATH = Path(__file__).resolve().parent / "data/test.jsonl"

def load_puzzle_groups() -> dict[tuple[int, ...], str]:
    groups = {}
    for name, path in zip(PUZZLE_GROUPS, (EASY_PUZZLES, MEDIUM_PUZZLES, HARD_PUZZLES)):
        for line in path.read_text().splitlines():
            if line.strip():
                groups[tuple(int(n) for n in line.split())] = name
    return groups

def load_puzzle_splits() -> dict[tuple[int, ...], str]:
    splits = {}
    for name, path in (("val", VAL_PATH), ("test", TEST_PATH)):
        for line in path.read_text().splitlines():
            if line.strip():
                splits[tuple(int(n) for n in line.split())] = name
    return splits

puzzle_groups = load_puzzle_groups()
puzzle_splits = load_puzzle_splits()

traces_dir = Path(__file__).resolve().parent / "traces"
traces_dir.mkdir(parents=True, exist_ok=True)

def eval(checkpoint_path: Path | None, data_path: Path | None = None, runs_per_puzzle: int = 8) -> dict[tuple[int, ...], float]:
    """
    Runs our evaluation on on the data_path of puzzles, returns a dictionary of puzzle -> cumulative reward
    """
    if checkpoint_path is not None:
        learner.load_checkpoint(checkpoint_path)
        print(f"\nLoaded checkpoint from {checkpoint_path} for evaluation\n")
    model.eval()

    puzzles = []
    puzzle_keys = []

    data_paths = [VAL_PATH, TEST_PATH] if data_path is None else [data_path]

    for data_path in data_paths:
        for line in Path(data_path).read_text().splitlines():
            if line.strip():
                puzzle = [int(n) for n in line.split()]
                puzzle_keys.append(tuple(puzzle))
                puzzles.append(puzzle)

    RUNS_PER_PUZZLE = runs_per_puzzle
    group_counts = {name: 0 for name in PUZZLE_GROUPS}
    puzzle_success_rate: dict[tuple[int, ...], float] = {}
    for key in puzzle_keys:
        group_counts[puzzle_groups[key]] += 1
        puzzle_success_rate[key] = 0
    totals = {name: 0 for name in PUZZLE_GROUPS}

    with torch.inference_mode():
        for run in tqdm(range(RUNS_PER_PUZZLE), desc="Evaluating"):
            for puzzle in puzzles:
                random.shuffle(puzzle)
            rollouts = actor.collect_rollouts(puzzles, 1)
            run_rewards = {name: 0.0 for name in PUZZLE_GROUPS}
            for index, (key, group) in enumerate(zip(puzzle_keys, rollouts)):
                name = puzzle_groups[key]
                run_rewards[name] += group[0].reward
                puzzle_success_rate[key] += group[0].reward

                if run == 0:
                    history = torch.cat([
                        actor.rollout_managers[index].prompt_ids,
                        *(turn.token_ids for turn in group[0].turns),
                    ])
                    text = actor.tokenizer.decode(history, skip_special_tokens=False)
                    trace_path = traces_dir / f"{'_'.join(map(str, key))}.txt"
                    trace_path.write_text(text)

            for name in PUZZLE_GROUPS:
                totals[name] += run_rewards[name]
            print(
                f"Run {run + 1}: "
                + ", ".join(f"{name} {run_rewards[name]:.0f}/{group_counts[name]}" for name in PUZZLE_GROUPS)
            )

            unsolved = [key for key, success in puzzle_success_rate.items() if success == 0]
            print(f"Unsolved after {run + 1} runs: {len(unsolved)}/{len(puzzles)} {unsolved}")

            empty_device_cache(DEVICE)
    print(
        "Total: "
        + ", ".join(
            f"{name} {totals[name]:.0f}/{group_counts[name] * RUNS_PER_PUZZLE}"
            for name in PUZZLE_GROUPS
        )
    )
    return puzzle_success_rate

def solve(puzzle: list[int], checkpoint_path: Path | None = None) -> bool:
    if not evaluate(puzzle):
        print(f"No solution for {puzzle}")
        return False

    if checkpoint_path is not None:
        learner.load_checkpoint(checkpoint_path)
        print(f"Loaded checkpoint from {checkpoint_path} for evaluation")

    model.eval()
    ATTEMPTS = 20
    with torch.inference_mode():
        rollouts = actor.collect_rollouts([puzzle], ATTEMPTS)[0]
    success_count = sum(1 for rollout in rollouts if rollout.reward > 0)

    if success_count == 0:
        print(f"Unsuccessful in solving {puzzle}")
        return False
    
    print(f"Successfully solved {puzzle} {success_count} times")
    
    for rollout in rollouts:
        if rollout.reward > 0:
            history = torch.cat([
                actor.rollout_managers[0].prompt_ids,
                *(turn.token_ids for turn in rollout.turns),
            ])
            text = actor.tokenizer.decode(history, skip_special_tokens=False)
            trace_path = traces_dir / f"{'_'.join(map(str, puzzle))}.txt"
            trace_path.write_text(text)
            return True

def ask(prompt: str, checkpoint_path: Path | None = None) -> None:
    if checkpoint_path is not None:
        learner.load_checkpoint(checkpoint_path)
        print(f"Loaded checkpoint from {checkpoint_path} for evaluation")

    with torch.inference_mode():
        input_ids = actor.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
            return_dict=True,
            return_tensors="pt",
        )["input_ids"].to(model.device)
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=2048,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            use_cache=True,
        )
        text = actor.tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
        print(text)

if __name__ == "__main__":
    """
    Baseline Validation Results: 17/137
    Overfit on [6, 6, 6, 6]: 30/137
    A6000 Train on 35 steps of 12 batch size: 28/137
    Hero Train Chckpoint 32: 31/123 
    Hero Train Chckpoint 72: 56/137
    ** Checkpoint 120 - stopped A100 to A6000, swapped batch 20x12 -> 16x12
    Hero Train Checkpoint 297: 81/137 
    Hero Train Checkpoint 363: 80/137 
    Hero Train Checkpoint 398: 82/137 
    ** Checkpoint 411 - pruning epoch to just harder puzzles, now 8x24
    Hero Train Checkpoint 398: 72/137 
    Hero Train Checkpoint 507: 77/137 and 85/137 
    Hero Train Checkpoint 647: 89/137 and 85/137
    Hero Train Checkpoint 751: 92/137 and 87/137
    Hero Train Checkpoint 799: 87/137 and 86/137, with 4096 token limit: 108/137
    Hero Train Checkpoint 819 [4096 token limit]: 114/137, 107/137, 107/137
    Hero Train Checkpoint 918: 85/137 and 86/137, with 4096 token limit: 111/137, 115/137, 113/137
    ** Checkpoint 951 - further pruned with one forced hard puzzle, upped token limit 2048 -> 4096
    Hero Train Checkpoint 992 [4096 token limit]: 112/137, 115/137
    Hero Train Checkpoint 1066 [4096 token limit]: 117/137, 115/137, 112/137, 114/137
    ** Checkpoint 1071 - out of 6, at least 3 puzzles are forced hard. forced hards are evenly distributed
    ** Checkpoint 1077 - lower batch -> 5, added prefill_chunk_size, and pytorch alloc conf expandable memory segments for OOM. fixed a hard puzzle pool based on empirically low success rates
    Hero Train Checkpoint 1127: 109/137, New Inference Engine | 103/137, 105/137, 109/137 
    ** Checkpoint 1135 - batch to 6, 2 forced hards, new inference engine
    """

    """
    New Eval Scripts
    Checkpoint 1066:
    Run 1: easy 96/115, medium 15/20, hard 0/2
    Run 2: easy 96/115, medium 17/20, hard 1/2
    Run 3: easy 99/115, medium 14/20, hard 1/2
    Run 4: easy 98/115, medium 18/20, hard 0/2
    Success count: 133

    Checkpoint 1135:
    Run 1: easy 95/115, medium 17/20, hard 1/2 did (tuple) instead of [list] for puzzle
    Run 1: easy 90/115, medium 15/20, hard 0/2
    Run 2: easy 94/115, medium 15/20, hard 1/2
    Run 3: easy 93/115, medium 17/20, hard 1/2
    Run 4: easy 89/115, medium 14/20, hard 2/2
    Success count: 130

    Checkpoint 1210:
    Run 1: easy 90/115, medium 14/20, hard 1/2
    Run 2: easy 100/115, medium 17/20, hard 1/2
    Run 3: easy 90/115, medium 16/20, hard 0/2
    Run 4: easy 96/115, medium 14/20, hard 0/2
    Total: easy 376/460, medium 61/80, hard 2/8
    
    With (tuple) instead of [list] for puzzle
    Run 1: easy 90/115, medium 15/20, hard 1/2
    Run 2: easy 94/115, medium 17/20, hard 0/2

    Env now provides numbers used
    Run 1: easy 90/115, medium 15/20, hard 1/2
    Run 2: easy 94/115, medium 15/20, hard 1/2
    Run 3: easy 90/115, medium 16/20, hard 0/2
    Total: easy 366/460, medium 62/80, hard 2/8
    Success count: 127

    Checkpoint 1280 and env now provides missing/extra:
    Run 1: easy 92/115, medium 17/20, hard 1/2
    Run 2: easy 98/115, medium 13/20, hard 0/2
    Run 3: easy 96/115, medium 18/20, hard 0/2
    Run 4: easy 99/115, medium 14/20, hard 2/2
    Success count: 133

    Baseline when env doesn't give any extra info (just result):
    Run 1: easy 88/115, medium 16/20, hard 0/2
    Run 2: easy 91/115, medium 14/20, hard 0/2
    Run 3: easy 100/115, medium 13/20, hard 0/2
    Run 4: easy 93/115, medium 17/20, hard 0/2
    Success count: 130

    Checkpoint 1373:
    Run 1: easy 97/115, medium 16/20, hard 0/2
    Run 2: easy 94/115, medium 14/20, hard 1/2
    Run 3: easy 91/115, medium 16/20, hard 1/2
    Run 4: easy 92/115, medium 18/20, hard 0/2
    Success count: 133

    Run 1: easy 92/115, medium 17/20, hard 1/2
    ** Started randomly shuffling puzzles within each training and evaluation batch. And sorted puzzles (validation contamination)

    Run 1: easy 96/115, medium 16/20, hard 2/2
    Run 2: easy 98/115, medium 18/20, hard 2/2
    Run 3: easy 95/115, medium 18/20, hard 2/2
    Run 4: easy 91/115, medium 18/20, hard 1/2
    Unsolved after 4 runs: 7/137 [(10, 10, 2, 3), (10, 1, 8, 2), (3, 12, 6, 7), (8, 2, 9, 8), (6, 12, 12, 12), (12, 8, 5, 5), (4, 5, 6, 12)]

    Checkpoint 1454:
    Run 1: easy 90/115, medium 17/20, hard 1/2
    Run 2: easy 94/115, medium 14/20, hard 0/2
    Run 3: easy 95/115, medium 18/20, hard 1/2
    Unsolved after 3 runs: 7/137 [(4, 9, 10, 12), (5, 6, 8, 9), (2, 6, 12, 13), (3, 6, 7, 10), (1, 3, 13, 13), (3, 8, 9, 12), (1, 9, 11, 12)]
    Run 4: easy 100/115, medium 14/20, hard 0/2
    Unsolved after 4 runs: 2/137 [(4, 9, 10, 12), (3, 6, 7, 10)]
    Run 5: easy 100/115, medium 17/20, hard 1/2
    Run 6: easy 89/115, medium 15/20, hard 0/2
    Run 7: easy 87/115, medium 15/20, hard 0/2
    Run 8: easy 92/115, medium 15/20, hard 1/2
    Unsolved after 8 runs: 1/137 [(3, 6, 7, 10)]

    Run 1: easy 92/115, medium 16/20, hard 1/2
    Run 2: easy 96/115, medium 15/20, hard 0/2
    Run 3: easy 93/115, medium 15/20, hard 0/2
    Unsolved after 3 runs: 7/137 [(4, 10, 10, 12), (1, 5, 11, 12), (2, 3, 10, 10), (2, 2, 7, 12), (3, 6, 7, 10), (2, 2, 10, 13), (2, 4, 10, 10)]
    Run 4: easy 94/115, medium 17/20, hard 1/2
    Run 5: easy 88/115, medium 16/20, hard 1/2
    Unsolved after 5 runs: 2/137 [(2, 3, 10, 10), (3, 6, 7, 10)]
    Run 6: easy 96/115, medium 18/20, hard 2/2
    Run 7: easy 95/115, medium 19/20, hard 1/2
    Unsolved after 7 runs: 0/137 []
    Run 8: easy 99/115, medium 16/20, hard 2/2

    Checkpoint 1471:
    ** 1 hard, 2 medium, 4 random train split. Validation set back to original validation (not in training)
    Run 1: easy 90/115, medium 16/20, hard 1/2
    Run 2: easy 86/115, medium 17/20, hard 1/2
    Run 3: easy 92/115, medium 17/20, hard 1/2
    Run 4: easy 90/115, medium 19/20, hard 1/2
    Run 5: easy 92/115, medium 16/20, hard 1/2
    Unsolved after 5 runs: 5/137 [(1, 3, 7, 8), (2, 2, 7, 12), (2, 3, 10, 10), (5, 5, 8, 12), (5, 7, 10, 10)]
    Run 6: easy 87/115, medium 17/20, hard 2/2
    Unsolved after 6 runs: 2/137 [(2, 3, 10, 10), (5, 5, 8, 12)]
    Run 7: easy 93/115, medium 18/20, hard 2/2
    Run 8: easy 92/115, medium 14/20, hard 1/2
    Unsolved after 8 runs: 2/137 [(2, 3, 10, 10), (5, 5, 8, 12)]

    Checkpoint 1066:
    Run 1: easy 99/115, medium 14/20, hard 0/2
    Run 2: easy 98/115, medium 16/20, hard 1/2
    Run 3: easy 93/115, medium 16/20, hard 0/2
    Unsolved after 3 runs: 6/137 [(10, 10, 2, 3), (2, 7, 12, 4), (8, 1, 8, 10), (13, 8, 7, 8), (3, 3, 13, 7), (2, 11, 11, 2)]
    Run 4: easy 93/115, medium 16/20, hard 1/2
    Run 5: easy 99/115, medium 16/20, hard 2/2
    Unsolved after 5 runs: 4/137 [(10, 10, 2, 3), (8, 1, 8, 10), (13, 8, 7, 8), (3, 3, 13, 7)]
    Run 6: easy 98/115, medium 17/20, hard 2/2
    Unsolved after 6 runs: 3/137 [(10, 10, 2, 3), (13, 8, 7, 8), (3, 3, 13, 7)]
    Run 7: easy 100/115, medium 16/20, hard 1/2
    Run 8: easy 102/115, medium 16/20, hard 1/2
    Unsolved after 8 runs: 1/137 [(3, 3, 13, 7)]

    Eval on on val + test results
    Checkpoint 1066:
    Run 1: easy 188/230, medium 33/40, hard 0/4
    Run 2: easy 189/230, medium 35/40, hard 1/4
    Run 3: easy 189/230, medium 31/40, hard 2/4
    Unsolved after 3 runs: 9/274 [(2, 2, 11, 11), (5, 5, 8, 12), (6, 12, 12, 12), (7, 8, 8, 13), (3, 12, 12, 13), (5, 8, 10, 12), (6, 9, 9, 10), (6, 11, 12, 12), (8, 9, 9, 12)]
    Run 4: easy 198/230, medium 32/40, hard 1/4
    Unsolved after 4 runs: 6/274 [(2, 2, 11, 11), (5, 5, 8, 12), (6, 12, 12, 12), (7, 8, 8, 13), (3, 12, 12, 13), (6, 9, 9, 10)]
    Run 5: easy 202/230, medium 30/40, hard 3/4
    Run 6: easy 196/230, medium 32/40, hard 2/4
    Run 7: easy 192/230, medium 33/40, hard 1/4
    Unsolved after 7 runs: 2/274 [(5, 5, 8, 12), (6, 9, 9, 10)]
    Run 8: easy 192/230, medium 33/40, hard 3/4
    Run 9: easy 196/230, medium 32/40, hard 1/4
    Run 10: easy 187/230, medium 32/40, hard 1/4
    Run 11: easy 194/230, medium 32/40, hard 2/4
    Run 12: easy 191/230, medium 33/40, hard 2/4
    Unsolved after 12 runs: 1/274 [(5, 5, 8, 12)]
    Run 13: easy 193/230, medium 31/40, hard 0/4
    Unsolved after 13 runs: 0/274 []
    Run 14: easy 197/230, medium 32/40, hard 2/4
    Run 15: easy 198/230, medium 32/40, hard 1/4
    Run 16: easy 191/230, medium 30/40, hard 1/4

    Checkpoint 1565:
    Run 1: easy 184/230, medium 31/40, hard 1/4
    Run 2: easy 192/230, medium 34/40, hard 1/4
    Run 3: easy 184/230, medium 33/40, hard 0/4
    Run 4: easy 184/230, medium 29/40, hard 3/4
    Unsolved after 4 runs: 6/274 [(3, 3, 7, 13), (7, 8, 8, 13), (7, 8, 9, 12), (2, 9, 13, 13), (4, 6, 7, 9), (8, 9, 9, 12)]
    Run 5: easy 191/230, medium 33/40, hard 1/4
    Run 6: easy 194/230, medium 30/40, hard 1/4
    Run 7: easy 182/230, medium 33/40, hard 2/4
    Run 8: easy 188/230, medium 30/40, hard 1/4
    Unsolved after 8 runs: 2/274 [(7, 8, 8, 13), (2, 9, 13, 13)]
    Run 9: easy 184/230, medium 30/40, hard 1/4
    Run 10: easy 190/230, medium 30/40, hard 2/4
    Run 11: easy 188/230, medium 30/40, hard 1/4
    Run 12: easy 189/230, medium 30/40, hard 1/4
    Run 13: easy 192/230, medium 34/40, hard 2/4
    Run 14: easy 185/230, medium 28/40, hard 2/4
    Run 15: easy 191/230, medium 30/40, hard 2/4
    Run 16: easy 192/230, medium 27/40, hard 2/4

    Checkpoint 1638:
    """

    CHECKPOINT = 1638
    checkpoint_path = Path(__file__).resolve().parent / f"checkpoints/checkpoint_{CHECKPOINT}.pt"

    print(f"Evaluating checkpoint {CHECKPOINT}")
    RUNS_PER_PUZZLE = 16
    puzzle_success_rate = eval(checkpoint_path, runs_per_puzzle=16)

    success_count = sum(1 for success in puzzle_success_rate.values() if success > 0)
    print(f"Success count: {success_count}")

    for puzzle, success in puzzle_success_rate.items():
        print(f"Puzzle {puzzle} ({puzzle_splits[puzzle]}) succeeded {success} times")

    # solve([8, 1, 5, 4], checkpoint_path)