from pathlib import Path
from transformers import Qwen3_5ForCausalLM
import torch
from tqdm import tqdm
from continuous_actor import Actor
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

def eval(checkpoint_path: Path | None, data_path: Path) -> tuple[int, float]:
    if checkpoint_path is not None:
        learner.load_checkpoint(checkpoint_path)
        print(f"Loaded checkpoint from {checkpoint_path} for evaluation")
    model.eval()

    puzzles = []
    for line in Path(data_path).read_text().splitlines():
        if line.strip():
            puzzles.append(list(map(int, line.split())))

    success_count = 0
    cumulative_reward = 0
    EVAL_BATCH_SIZE = 70
    GROUP_SIZE = 1
    with torch.inference_mode():
        ran = 0
        for start in tqdm(range(0, len(puzzles), EVAL_BATCH_SIZE), desc="Evaluating"):
            batch = puzzles[start:start + EVAL_BATCH_SIZE]
            rollouts = actor.collect_rollouts(batch, GROUP_SIZE)
            ran += len(batch)
            for results in rollouts:
                summed_reward = sum(r.reward for r in results)
                cumulative_reward += summed_reward
                if summed_reward > 0:
                    success_count += 1
            print(f"Successes so far: {success_count}/{ran} \n\n")
    print(f"\n Success count: {success_count}/{len(puzzles)}, Cumulative reward: {cumulative_reward} \n")
    return success_count, cumulative_reward

def train(checkpoint_path: Path | None = None, initial_update_step: int = 0, initial_avg_rewards: list[float] = []):
    BATCH_SIZE = 6
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
    harder_puzzles = train_data[-180:]
    avg_rewards = initial_avg_rewards

    log_memory(0, "initialized", model.device)

    update_step = initial_update_step
    # for epoch in range(EPOCHS):
    epoch = 0
    while True:
        epoch += 1
        random_indices = np.random.choice(len(train_data), size=30, replace=False)
        epoch_puzzles = harder_puzzles + [train_data[i] for i in random_indices]
        np.random.shuffle(epoch_puzzles)
        for start in tqdm(
            range(0, len(epoch_puzzles), BATCH_SIZE - 1),
            desc=f"Epoch {epoch} | Training batches",
            position=0,
            leave=True,
            dynamic_ncols=True,
        ):
            training_batch = epoch_puzzles[start:start + BATCH_SIZE - 1]
            # force one hard puzzle at the end, from train_data last 11 rows
            hard_puzzle = train_data[-11 + np.random.randint(11)]
            training_batch.append(hard_puzzle)
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
            log_memory((start // BATCH_SIZE) + 1, device=model.device)
            step_time = time.time()
            avg_rewards.append(avg_reward)
            with avg_reward_csv_path.open("a") as f:
                f.write(f"{avg_reward:.6g}\n")

            for puzzle, group in zip(training_batch, rollouts):
                print(f"update_step {update_step + 1} | Puzzle: {sorted(puzzle)} had {sum(r.reward for r in group)} successes and generated {sum(turn.token_ids.shape[-1] - turn.completion_start for rollout in group for turn in rollout.turns)} tokens")

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
            log_memory((start // BATCH_SIZE) + 1, "after_empty_cache", model.device)

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
    """

    """
    # sanity_path = Path(__file__).resolve().parent / "data/sanity.jsonl"
    val_path = Path(__file__).resolve().parent / "data/val.jsonl"
    # print("Evaluating base model")
    # base_success_count, base_reward = eval(None, sanity_path)
    checkpoint_path = Path(__file__).resolve().parent / "checkpoints/checkpoint_918.pt"
    print("Evaluating final checkpoint")
    final_success_count, final_reward = eval(checkpoint_path, val_path)
    # print(f"Base model: success count {base_success_count}, reward {base_reward}")
    print(f"Final checkpoint: success count {final_success_count}, reward {final_reward}")
    """

    starting_checkpoint = 951
    checkpoint_path = Path(__file__).resolve().parent / f"checkpoints/notable/checkpoint_{starting_checkpoint}.pt"
    initial_avg_rewards = (
        [float(x) for x in avg_reward_csv_path.read_text().splitlines() if x.strip()]
        if avg_reward_csv_path.exists()
        else []
    )
    train(checkpoint_path, initial_update_step=starting_checkpoint, initial_avg_rewards=initial_avg_rewards)