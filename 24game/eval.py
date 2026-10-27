from pathlib import Path
from transformers import Qwen3_5ForCausalLM
import torch
from tqdm import tqdm
from actor import Actor
from learner import Learner
from lora_linear import apply_lora

def empty_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

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

def load_puzzle_groups() -> dict[tuple[int, ...], str]:
    groups = {}
    val_set = {
        tuple(int(n) for n in line.split())
        for line in VAL_PATH.read_text().splitlines()
        if line.strip()
    }
    for name, path in zip(PUZZLE_GROUPS, (EASY_PUZZLES, MEDIUM_PUZZLES, HARD_PUZZLES)):
        for line in path.read_text().splitlines():
            if line.strip():
                puzzle = tuple(int(n) for n in line.split())
                if puzzle in val_set:
                    groups[puzzle] = name
    return groups

puzzle_groups = load_puzzle_groups()

traces_dir = Path(__file__).resolve().parent / "traces"
traces_dir.mkdir(parents=True, exist_ok=True)

def eval(checkpoint_path: Path | None, data_path: Path) -> dict[tuple[int, ...], float]:
    if checkpoint_path is not None:
        learner.load_checkpoint(checkpoint_path)
        print(f"Loaded checkpoint from {checkpoint_path} for evaluation")
    model.eval()

    puzzles = []
    for line in Path(data_path).read_text().splitlines():
        if line.strip():
            puzzle = tuple(int(n) for n in line.split())
            puzzles.append(list(puzzle))

    RUNS_PER_PUZZLE = 4
    group_counts = {name: 0 for name in PUZZLE_GROUPS}
    puzzle_success_rate: dict[tuple[int, ...], float] = {}
    for puzzle in puzzles:
        group_counts[puzzle_groups[tuple(puzzle)]] += 1
        puzzle_success_rate[tuple(puzzle)] = 0
    totals = {name: 0 for name in PUZZLE_GROUPS}

    with torch.inference_mode():
        for run in tqdm(range(RUNS_PER_PUZZLE), desc="Evaluating"):
            rollouts = actor.collect_rollouts(puzzles, 1)
            run_rewards = {name: 0.0 for name in PUZZLE_GROUPS}
            for index, (puzzle, group) in enumerate(zip(puzzles, rollouts)):
                name = puzzle_groups[tuple(puzzle)]
                run_rewards[name] += group[0].reward
                puzzle_success_rate[tuple(puzzle)] += group[0].reward

                if run == 0:
                    history = torch.cat([
                        actor.rollout_managers[index].prompt_ids,
                        *(turn.token_ids for turn in group[0].turns),
                    ])
                    text = actor.tokenizer.decode(history, skip_special_tokens=False)
                    trace_path = traces_dir / f"{'_'.join(map(str, puzzle))}.txt"
                    trace_path.write_text(text)

            for name in PUZZLE_GROUPS:
                totals[name] += run_rewards[name]
            print(
                f"Run {run + 1}: "
                + ", ".join(f"{name} {run_rewards[name]:.0f}/{group_counts[name]}" for name in PUZZLE_GROUPS)
            )
            empty_device_cache(DEVICE)
    print(
        "Total: "
        + ", ".join(
            f"{name} {totals[name]:.0f}/{group_counts[name] * RUNS_PER_PUZZLE}"
            for name in PUZZLE_GROUPS
        )
    )
    return puzzle_success_rate


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
    """

    CHECKPOINT = 1135
    checkpoint_path = Path(__file__).resolve().parent / f"checkpoints/checkpoint_{CHECKPOINT}.pt"
    print(f"Evaluating checkpoint {CHECKPOINT}")
    puzzle_success_rate = eval(checkpoint_path, VAL_PATH)

    success_count = sum(1 for success in puzzle_success_rate.values() if success > 0)
    print(f"Success count: {success_count}")