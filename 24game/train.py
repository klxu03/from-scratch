from pathlib import Path
from transformers import Qwen3_5ForCausalLM
import torch
from tqdm import tqdm
from actor import Actor
from learner import Learner
from lora_linear import apply_lora
import time
import numpy as np
from graph import create_rolling

import psutil
import csv
import time

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
        [3, 13, 7, 5],
        [3, 8, 8, 3],
        [7, 7, 4, 4],
        [8, 1, 12, 12],
        [3, 5, 12, 2],
        [11, 11, 5, 1],
        [2, 10, 5, 5],
        [6, 6, 3, 10],
        [11, 11, 9, 8],
        [5, 7, 2, 8],
        [12, 8, 5, 12],
        [3, 7, 5, 3],
        [2, 11, 10, 2],
        [6, 3, 6, 11],
        [10, 11, 5, 7],
        [7, 3, 3, 8],
    ]
    avg_rewards = initial_avg_rewards

    log_memory(0, "initialized", model.device)

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
            training_batch = epoch_puzzles[start:start + BATCH_SIZE - HARD_PER_BATCH].copy()
            # force HARD_PER_BATCH hard puzzles at the end
            hard_start = update_step % len(hard_pool)

            training_batch.extend(
                hard_pool[(hard_start + i) % len(hard_pool)]
                for i in range(HARD_PER_BATCH)
            )

            learner.zero_grad()

            start_time = time.time()

            if model.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(model.device)
                retries_before = torch.cuda.memory_stats(model.device)["num_alloc_retries"]

            rollouts = actor.collect_rollouts(training_batch, GROUP_SIZE)
            rollout_time = time.time()
            for i, rollout_group in enumerate(rollouts):
                learner.accumulate_group(rollout_group, training_batch[i], GROUP_SIZE * len(training_batch))
            avg_reward = learner.step() / (GROUP_SIZE * len(training_batch))
            log_memory(update_step + 1, device=model.device)
            step_time = time.time()
            avg_rewards.append(avg_reward)
            with avg_reward_csv_path.open("a") as f:
                f.write(f"{avg_reward:.6g}\n")

            for puzzle, group in zip(training_batch, rollouts):
                print(f"update_step {update_step + 1} | Puzzle: {sorted(puzzle)} had {sum(r.reward for r in group)} successes and generated {sum(turn.token_ids.numel() - turn.completion_start for rollout in group for turn in rollout.turns)} tokens")

            if model.device.type == "cuda":
                torch.cuda.synchronize(model.device)
                memory_stats = torch.cuda.memory_stats(model.device)

                print(
                    f"Peak live: "
                    f"{torch.cuda.max_memory_allocated(model.device) / 2**30:.2f} GiB | "
                    f"Peak reserved: "
                    f"{torch.cuda.max_memory_reserved(model.device) / 2**30:.2f} GiB | "
                    f"Allocation retries this step: "
                    f"{memory_stats['num_alloc_retries'] - retries_before}",
                    flush=True,
                )

            print(f"Average reward: {avg_reward}")
            print(f"Rollouts time: {rollout_time - start_time}, Step time: {step_time - rollout_time}\n")

            create_rolling(avg_reward_path, avg_rewards, [1], title="GRPO average reward")
            create_rolling(rolling_avg_path, avg_rewards, [5, 20, 40], title="GRPO rolling average")

            empty_device_cache(model.device)
            log_memory(update_step + 1, "after_empty_cache", model.device)

            learner.create_checkpoint(Path(__file__).resolve().parent / "checkpoints" / f"checkpoint_{(update_step := update_step + 1)}.pt")

def empty_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

def _device_memory(device: torch.device) -> tuple[int, int]:
    """Bytes in live tensors, and bytes reserved from the driver by the caching allocator."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        return torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device)
    if device.type == "mps":
        torch.mps.synchronize()
        return torch.mps.current_allocated_memory(), torch.mps.driver_allocated_memory()
    return 0, 0

_PROCESS = psutil.Process()
def log_memory(step: int, phase: str = "after_step", device: torch.device | None = None) -> dict:
    """Record memory without clearing gradients, objects, or caches."""
    if device is None:
        device = model.device
    live, reserved = _device_memory(device)

    gib = 2**30
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()

    sizes = {
        "device": str(device),
        "device_allocated_gib": live / gib,
        "device_reserved_gib": reserved / gib,
        "device_gap_gib": (reserved - live) / gib,
        "process_rss_gib": _PROCESS.memory_info().rss / gib,
        "system_total_gib": vm.total / gib,
        "system_available_gib": vm.available / gib,
        "system_swap_gib": swap.used / gib,
        "disk_free_gib": psutil.disk_usage(_MEMORY_CSV.parent).free / gib,
        "process_uss_gib": _PROCESS.memory_full_info().uss / gib,
        "swap_in_gib": swap.sin / gib,
        "swap_out_gib": swap.sout / gib,
    }

    row = {
        "timestamp": time.time(),
        "pid": _PROCESS.pid,
        "step": step,
        "phase": phase,
        **sizes,
    }

    needs_header = (
        not _MEMORY_CSV.exists() or _MEMORY_CSV.stat().st_size == 0
    )
    with _MEMORY_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if needs_header:
            writer.writeheader()
        writer.writerow(row)

    return row

if __name__ == "__main__":
    starting_checkpoint = 1077
    checkpoint_path = Path(__file__).resolve().parent / f"checkpoints/notable/checkpoint_{starting_checkpoint}.pt"
    initial_avg_rewards = (
        [float(x) for x in avg_reward_csv_path.read_text().splitlines() if x.strip()]
        if avg_reward_csv_path.exists()
        else []
    )
    train(checkpoint_path, initial_update_step=starting_checkpoint, initial_avg_rewards=initial_avg_rewards)