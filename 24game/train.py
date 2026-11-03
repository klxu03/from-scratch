from pathlib import Path
from transformers import Qwen3_5ForCausalLM
import torch
from tqdm import tqdm
from actor import Actor
from learner import Learner
from lora_linear import apply_lora
import time
import numpy as np
from utils.graph import create_rolling
from utils.memory import empty_device_cache, log_memory, print_memory, print_peak_memory, reset_peak_memory

MODEL = "Qwen/Qwen3.5-0.8B"
MAX_OUTPUT_TOKENS = 4096
# Only place the training device is chosen. Actor and learner follow model.device.
DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
if DEVICE.type == "cuda":
    torch.set_float32_matmul_precision("high")
model = Qwen3_5ForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE)
model = apply_lora(model)
actor = Actor(model, max_output_tokens=MAX_OUTPUT_TOKENS, tail_threshold=32)
learner = Learner(model, MAX_OUTPUT_TOKENS)

avg_reward_path = Path(__file__).resolve().with_name("average_reward.jpg")
avg_reward_csv_path = Path(__file__).resolve().with_name("average_rewards.csv")
rolling_avg_path = Path(__file__).resolve().with_name("rolling_avg.jpg")
_MEMORY_CSV = Path(__file__).resolve().with_name("memory_debug.csv")

def train(checkpoint_path: Path | None = None, initial_update_step: int = 0, initial_avg_rewards: list[float] = []):
    BATCH_SIZE = 14
    HARDS_PER_BATCH = 11
    GROUP_SIZE = 16

    if checkpoint_path is not None:
        print(f"Loading checkpoint from {checkpoint_path}")
        learner.load_checkpoint(checkpoint_path)

    for param_group in learner.optimizer.param_groups:
        param_group["lr"] = 5e-6

    print(
        "Learning rates:",
        [group["lr"] for group in learner.optimizer.param_groups],
    )

    data_path = Path(__file__).resolve().parent / "data/train.jsonl"
    train_data: list[list[int]] = []
    for line in data_path.read_text().splitlines():
        if line.strip():
            train_data.append(list(map(int, line.split()))) 

    hard_pool = [
        [1, 3, 4, 6], [1, 3, 8, 9],
        [1, 4, 5, 6], [1, 4, 7, 11],
        [1, 6, 11, 13], [1, 8, 12, 12],
        [2, 2, 10, 11], [2, 2, 12, 13],
        [2, 3, 5, 12], [2, 3, 9, 12],
        [2, 4, 10, 10], [2, 5, 5, 10],
        [2, 5, 6, 9], [2, 5, 7, 8],
        [2, 7, 7, 10], [2, 7, 12, 12],
        [2, 8, 8, 11], [3, 3, 7, 7],
        [3, 3, 8, 8], [3, 4, 6, 13],
        [3, 5, 7, 13], [3, 5, 8, 12],
        [3, 5, 12, 13], [3, 6, 7, 10],
        [3, 7, 9, 13], [3, 11, 12, 12],
        [4, 4, 7, 7], [4, 7, 8, 10],
        [4, 7, 9, 11], [5, 5, 7, 11],
        [5, 6, 10, 12], [5, 7, 7, 10],
        [5, 7, 7, 11], [5, 7, 9, 12],
        [5, 9, 10, 11], [5, 9, 12, 12],
        [6, 7, 8, 12], [7, 8, 10, 11],
        [7, 10, 12, 13], [9, 11, 12, 13],
    ]
    np.random.shuffle(hard_pool)
    avg_rewards = initial_avg_rewards

    log_memory(0, "initialized", model.device, _MEMORY_CSV)

    update_step = initial_update_step
    epoch = 0
    while True:
        puzzle_counts: dict[tuple[int, int, int, int], tuple[int, int]] = {}
        epoch += 1
        np.random.shuffle(train_data)
        train_per_batch = BATCH_SIZE - HARDS_PER_BATCH
        HARDS_THIS_BATCH = min(HARDS_PER_BATCH, len(hard_pool))
        for start in tqdm(
            range(0, len(train_data), train_per_batch),
            desc=f"Epoch {epoch} | Training batches",
            position=0,
            leave=True,
            dynamic_ncols=True,
        ):
            print("\n\n")
            training_batch = [puzzle.copy() for puzzle in train_data[start:start + train_per_batch]]
            if hard_pool:
                training_batch.extend(
                    hard_pool[(update_step * HARDS_THIS_BATCH + offset) % len(hard_pool)].copy()
                    for offset in range(HARDS_THIS_BATCH)
                )

            for puzzle in training_batch:
                np.random.shuffle(puzzle)

            learner.zero_grad()

            start_time = time.time()

            retries_before = reset_peak_memory(model.device)

            rollouts = actor.collect_rollouts(training_batch, GROUP_SIZE)
            print_peak_memory(model.device, "Actor")
            rollout_time = time.time()
            for i, rollout_group in enumerate(rollouts):
                learner.accumulate_group(rollout_group, training_batch[i], GROUP_SIZE * len(training_batch))
            avg_reward = learner.step() / (GROUP_SIZE * len(training_batch))
            log_memory(update_step + 1, device=model.device, csv_path=_MEMORY_CSV)
            step_time = time.time()

            for puzzle, group in zip(training_batch, rollouts):
                key = tuple(sorted(puzzle))
                successes, seen = puzzle_counts.get(key, (0, 0))
                puzzle_counts[key] = (successes + sum(rollout.reward for rollout in group), seen + len(group))

            avg_rewards.append(avg_reward)
            with avg_reward_csv_path.open("a") as f:
                f.write(f"{avg_reward:.6g}\n")

            for puzzle, group in zip(training_batch, rollouts):
                print(f"update_step {update_step + 1} | Puzzle: {sorted(puzzle)} had {sum(r.reward for r in group)} successes and generated {sum(turn.token_ids.numel() - turn.completion_start for rollout in group for turn in rollout.turns)} tokens")

            print_memory(model.device, retries_before)

            print(f"Average reward: {avg_reward}")
            print(f"Rollouts time: {rollout_time - start_time}, Step time: {step_time - rollout_time}\n")

            create_rolling(avg_reward_path, avg_rewards, [1], title="GRPO average reward")
            create_rolling(rolling_avg_path, avg_rewards, [5, 20, 40], title="GRPO rolling average")

            empty_device_cache(model.device)
            log_memory(update_step + 1, "after_empty_cache", model.device, _MEMORY_CSV)

            learner.create_checkpoint(Path(__file__).resolve().parent / "checkpoints" / f"checkpoint_{(update_step := update_step + 1)}.pt")

        # Puzzles under 50% join the pool. A pooled puzzle leaves only after it is seen
        # enough times and solved on at least 70% of those rollouts.
        for puzzle, (successes, seen) in puzzle_counts.items():
            if seen > 0 and successes / seen <= 0.5:
                puzzle_list = list(puzzle)
                if puzzle_list in hard_pool:
                    hard_pool.remove(puzzle_list)
                hard_pool.append(puzzle_list)
        kept = []
        for puzzle in hard_pool:
            successes, seen = puzzle_counts.get(tuple(sorted(puzzle)), (0, 0))
            if seen <= 2 * GROUP_SIZE or successes / seen <= 0.7:
                kept.append(puzzle)
                print(f"Puzzle {puzzle} added to hard pool")
        hard_pool = kept
        np.random.shuffle(hard_pool)


if __name__ == "__main__":
    starting_checkpoint = 2048
    checkpoint_path = Path(__file__).resolve().parent / f"checkpoints/notable/checkpoint_{starting_checkpoint}.pt"
    initial_avg_rewards = (
        [float(x) for x in avg_reward_csv_path.read_text().splitlines() if x.strip()]
        if avg_reward_csv_path.exists()
        else []
    )
    train(checkpoint_path, initial_update_step=starting_checkpoint, initial_avg_rewards=initial_avg_rewards)