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
from utils.memory import empty_device_cache, log_memory, print_memory, reset_peak_memory

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
actor = Actor(model, MAX_OUTPUT_TOKENS)
learner = Learner(model, MAX_OUTPUT_TOKENS)

avg_reward_path = Path(__file__).resolve().with_name("average_reward.jpg")
avg_reward_csv_path = Path(__file__).resolve().with_name("average_rewards.csv")
rolling_avg_path = Path(__file__).resolve().with_name("rolling_avg.jpg")
_MEMORY_CSV = Path(__file__).resolve().with_name("memory_debug.csv")

def train(checkpoint_path: Path | None = None, initial_update_step: int = 0, initial_avg_rewards: list[float] = []):
    BATCH_SIZE = 6
    HARD_PER_BATCH = 2
    GROUP_SIZE = 24

    if checkpoint_path is not None:
        print(f"Loading checkpoint from {checkpoint_path}")
        learner.load_checkpoint(checkpoint_path)

    data_path = Path(__file__).resolve().parent / "data/train.jsonl"
    train_data: list[list[int]] = []
    for line in data_path.read_text().splitlines():
        if line.strip():
            train_data.append(list(map(int, line.split()))) 

    # train_data = [[6, 6, 6, 6]] * 150 
    medium_hard_puzzles = train_data[-170:]
    hard_pool = [
        [1, 8, 9, 13],
        [1, 8, 12, 12],
        [2, 2, 7, 10],
        [2, 2, 10, 11],
        [2, 2, 11, 12],
        [2, 3, 5, 12],
        [2, 4, 10, 10],
        [2, 5, 5, 10],
        [2, 5, 7, 8],
        [3, 3, 5, 5],
        [3, 3, 5, 7],
        [3, 3, 7, 8],
        [3, 3, 8, 8],
        [3, 5, 7, 13],
        [3, 6, 6, 11],
        [4, 4, 7, 7],
        [5, 5, 8, 8],
        [5, 5, 13, 13],
        [5, 7, 10, 11],
        [7, 7, 8, 12],
        [8, 8, 8, 13],
        [8, 9, 11, 11],
    ]
    avg_rewards = initial_avg_rewards

    log_memory(0, "initialized", model.device, _MEMORY_CSV)

    update_step = initial_update_step
    # for epoch in range(EPOCHS):
    epoch = 0
    while True:
        epoch += 1
        random_indices = np.random.choice(len(train_data), size=40, replace=False)
        epoch_puzzles = medium_hard_puzzles + [train_data[i] for i in random_indices]
        np.random.shuffle(epoch_puzzles)
        for start in tqdm(
            range(0, len(epoch_puzzles), BATCH_SIZE - HARD_PER_BATCH),
            desc=f"Epoch {epoch} | Training batches",
            position=0,
            leave=True,
            dynamic_ncols=True,
        ):
            print("\n\n")
            training_batch = [puzzle.copy() for puzzle in epoch_puzzles[start:start + BATCH_SIZE - HARD_PER_BATCH]]
            # force HARD_PER_BATCH hard puzzles at the end
            hard_start = update_step % len(hard_pool)

            training_batch.extend(
                hard_pool[(hard_start + i) % len(hard_pool)].copy()
                for i in range(HARD_PER_BATCH)
            )

            for puzzle in training_batch:
                np.random.shuffle(puzzle)

            learner.zero_grad()

            start_time = time.time()

            retries_before = reset_peak_memory(model.device)

            rollouts = actor.collect_rollouts(training_batch, GROUP_SIZE)
            rollout_time = time.time()
            for i, rollout_group in enumerate(rollouts):
                learner.accumulate_group(rollout_group, training_batch[i], GROUP_SIZE * len(training_batch))
            avg_reward = learner.step() / (GROUP_SIZE * len(training_batch))
            log_memory(update_step + 1, device=model.device, csv_path=_MEMORY_CSV)
            step_time = time.time()
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

if __name__ == "__main__":
    starting_checkpoint = 1373
    checkpoint_path = Path(__file__).resolve().parent / f"checkpoints/notable/checkpoint_{starting_checkpoint}.pt"
    initial_avg_rewards = (
        [float(x) for x in avg_reward_csv_path.read_text().splitlines() if x.strip()]
        if avg_reward_csv_path.exists()
        else []
    )
    train(checkpoint_path, initial_update_step=starting_checkpoint, initial_avg_rewards=initial_avg_rewards)