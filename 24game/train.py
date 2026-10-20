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
model = Qwen3_5ForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("mps")
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

def eval(checkpoint_path: Path | None, data_path: Path) -> float:
    if checkpoint_path is not None:
        learner.load_checkpoint(checkpoint_path)
    model.eval()

    puzzles = []
    for line in Path(data_path).read_text().splitlines():
        if line.strip():
            puzzles.append(list(map(int, line.split())))

    with torch.inference_mode():
        for puzzle in tqdm(puzzles, desc="Evaluating"):
            results = actor.collect_rollouts([puzzle], 4)[0]
            summed_reward = sum(r.reward for r in results)
            print(f"Cumulative reward: {summed_reward} / 25 puzzles")
            return summed_reward

def train(starting_index: int = 0, checkpoint_path: Path | None = None):
    BATCH_SIZE = 1
    GROUP_SIZE = 8

    if checkpoint_path is not None:
        print(f"Loading checkpoint from {checkpoint_path}")
        learner.load_checkpoint(checkpoint_path)

    data_path = Path(__file__).resolve().parent / "data/train.jsonl"
    train_data: list[list[int]] = []
    for line in data_path.read_text().splitlines():
        if line.strip():
            train_data.append(list(map(int, line.split()))) 

    train_data = train_data[:16] + train_data[-16:]
    train_data = [[6, 6, 6, 6]] * 120 
    cumulated_rewards = []
    cumulated_reward_path = Path(__file__).resolve().parent / "checkpoints" / "cumulated_reward.jpg"

    log_memory(0, "initialized")

    for start in tqdm(range(starting_index, len(train_data), BATCH_SIZE), desc="GRPO Training"):
        training_batch = train_data[start:start + BATCH_SIZE]
        learner.zero_grad()

        rollouts = actor.collect_rollouts(training_batch, GROUP_SIZE)
        for i, rollout_group in enumerate(rollouts):
            learner.accumulate_group(rollout_group, training_batch[i], GROUP_SIZE * BATCH_SIZE)
        cumulated_reward = learner.step()
        log_memory((start // BATCH_SIZE) + 1)
        cumulated_rewards.append(cumulated_reward)
        print(f"Cumulated reward: {cumulated_reward}")

        create_graph(cumulated_reward_path, cumulated_rewards, "GRPO cumulated reward/8", "step", "cumulated reward")
        torch.mps.empty_cache()
        log_memory((start // BATCH_SIZE) + 1, "after_empty_cache")

        if start % 1 == 0:
            learner.create_checkpoint(Path(__file__).resolve().parent / "checkpoints" / f"checkpoint_{start}.pt")

    create_graph(cumulated_reward_path, cumulated_rewards, "GRPO cumulated reward/8", "step", "cumulated reward")
    log_memory(len(train_data), "finished")
    learner.create_checkpoint(Path(__file__).resolve().parent / "checkpoints" / "final.pt")

_MEMORY_CSV = Path(__file__).resolve().with_name("memory_debug.csv")
_PROCESS = psutil.Process()
def log_memory(step: int, phase: str = "after_step") -> dict:
    """Record memory without clearing gradients, objects, or caches."""
    torch.mps.synchronize()

    gib = 2**30
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    live = torch.mps.current_allocated_memory()
    driver = torch.mps.driver_allocated_memory()

    sizes = {
        "mps_live_gib": live / gib,
        "mps_driver_gib": driver / gib,
        "mps_gap_gib": (driver - live) / gib,
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
    Out of 25:
    Model  | 2048 Tokens | 4096 Tokens |
    Base   |     6       |      12     |
    T 72   |     12      |      17     |
    T 96   |     12      |      17     |
    """
    
    sanity_path = Path(__file__).resolve().parent / "data/sanity.jsonl"
    # print("Evaluating base model")
    # base_reward = eval(None, sanity_path)
    checkpoint_path = Path(__file__).resolve().parent / "checkpoints/checkpoint_4.pt"
    print("Evaluating final checkpoint")
    final_reward = eval(checkpoint_path, sanity_path)
    base_reward = 1
    print(f"Base reward: {base_reward}, Final reward: {final_reward}")

    # train(starting_index=98, checkpoint_path=Path(__file__).resolve().parent / "checkpoints/checkpoint_96.pt")
    # train()