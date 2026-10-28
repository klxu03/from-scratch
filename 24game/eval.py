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
    """
    Runs our evaluation on on the data_path of puzzles, returns a dictionary of puzzle -> cumulative reward
    """
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

            num_puzzles_with_no_solves = sum(success == 0 for success in puzzle_success_rate.values())
            print(f"Unsolved after {run + 1} runs: {num_puzzles_with_no_solves}/{len(puzzles)}")

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
    """

    CHECKPOINT = 1373
    checkpoint_path = Path(__file__).resolve().parent / f"checkpoints/checkpoint_{CHECKPOINT}.pt"

    print(f"Evaluating checkpoint {CHECKPOINT}")
    puzzle_success_rate = eval(checkpoint_path, VAL_PATH)

    success_count = sum(1 for success in puzzle_success_rate.values() if success > 0)
    print(f"Success count: {success_count}")

    for puzzle, success in puzzle_success_rate.items():
        if success == 0:
            print(f"Puzzle {puzzle} failed")

    # solve([8, 1, 5, 4], checkpoint_path)