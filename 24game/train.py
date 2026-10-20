from pathlib import Path
from transformers import Qwen3_5ForCausalLM
import matplotlib
import torch
from tqdm import tqdm
from actor import Actor
from learner import Learner
from lora_linear import apply_lora

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import psutil
import csv
import time

MODEL = "Qwen/Qwen3.5-0.8B"
MAX_OUTPUT_TOKENS = 2048
# Only place the training device is chosen. Actor and learner follow model.device.
DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
model = Qwen3_5ForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE)
model = apply_lora(model)
actor = Actor(model, MAX_OUTPUT_TOKENS)
learner = Learner(model, MAX_OUTPUT_TOKENS)

def create_graph(path: Path, values: list[float], title: str, xlabel: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(values, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def eval(checkpoint_path: Path | None, data_path: Path) -> tuple[int, float]:
    if checkpoint_path is not None:
        learner.load_checkpoint(checkpoint_path)
    model.eval()

    puzzles = []
    for line in Path(data_path).read_text().splitlines():
        if line.strip():
            puzzles.append(list(map(int, line.split())))

    success_count = 0
    cumulative_reward = 0
    with torch.inference_mode():
        NUM_PUZZLES = 1
        for puzzle in tqdm(puzzles, desc="Evaluating"):
            results = actor.collect_rollouts([puzzle], NUM_PUZZLES)[0]
            summed_reward = sum(r.reward for r in results)
            print(f"Cumulative reward: {summed_reward} / {NUM_PUZZLES} puzzles for puzzle {puzzle}\n")
            cumulative_reward += summed_reward
            if summed_reward > 0:
                success_count += 1
    print(f"\nSuccess count: {success_count}/{len(puzzles)}, Cumulative reward: {cumulative_reward}")
    return success_count, cumulative_reward

def train(starting_index: int = 0, checkpoint_path: Path | None = None):
    BATCH_SIZE = 1
    GROUP_SIZE = 8
    EPOCHS = 2

    if checkpoint_path is not None:
        print(f"Loading checkpoint from {checkpoint_path}")
        learner.load_checkpoint(checkpoint_path)

    data_path = Path(__file__).resolve().parent / "data/train.jsonl"
    train_data: list[list[int]] = []
    for line in data_path.read_text().splitlines():
        if line.strip():
            train_data.append(list(map(int, line.split()))) 

    # train_data = [[6, 6, 6, 6]] * 150 
    cumulated_rewards = []
    cumulated_reward_path = Path(__file__).resolve().parent / "checkpoints" / "cumulated_reward.jpg"

    log_memory(0, "initialized", model.device)

    for epoch in tqdm(range(EPOCHS), desc="Epochs of GRPO Training"):
        for start in tqdm(range(starting_index, len(train_data), BATCH_SIZE), desc=f"Running epoch {epoch} of GRPO Training"):
            training_batch = train_data[start:start + BATCH_SIZE]
            learner.zero_grad()

            rollouts = actor.collect_rollouts(training_batch, GROUP_SIZE)
            for i, rollout_group in enumerate(rollouts):
                learner.accumulate_group(rollout_group, training_batch[i], GROUP_SIZE * BATCH_SIZE)
            cumulated_reward = learner.step()
            log_memory((start // BATCH_SIZE) + 1, device=model.device)
            cumulated_rewards.append(cumulated_reward)
            print(f"Cumulated reward: {cumulated_reward}")

            create_graph(cumulated_reward_path, cumulated_rewards, "GRPO cumulated reward", "step", "cumulated reward")
            empty_device_cache(model.device)
            log_memory((start // BATCH_SIZE) + 1, "after_empty_cache", model.device)

            if start % 25 == 0:
                learner.create_checkpoint(Path(__file__).resolve().parent / "checkpoints" / f"checkpoint_{epoch * len(train_data) + start}.pt")

    create_graph(cumulated_reward_path, cumulated_rewards, "GRPO cumulated reward", "step", "cumulated reward")
    log_memory(len(train_data), "finished", model.device)
    learner.create_checkpoint(Path(__file__).resolve().parent / "checkpoints" / "final.pt")

_MEMORY_CSV = Path(__file__).resolve().with_name("memory_debug.csv")
_PROCESS = psutil.Process()

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
    """

    """
    # sanity_path = Path(__file__).resolve().parent / "data/sanity.jsonl"
    sanity_path = Path(__file__).resolve().parent / "data/val.jsonl"
    print("Evaluating base model")
    base_success_count, base_reward = eval(None, sanity_path)
    checkpoint_path = Path(__file__).resolve().parent / "checkpoints/final.pt"
    print("Evaluating final checkpoint")
    final_success_count, final_reward = eval(checkpoint_path, sanity_path)
    print(f"Base model: success count {base_success_count}, reward {base_reward}")
    print(f"Final checkpoint: success count {final_success_count}, reward {final_reward}")
    """

    train()