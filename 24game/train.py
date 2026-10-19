from pathlib import Path
from transformers import Qwen3_5ForCausalLM
import matplotlib
import torch
from tqdm import tqdm
from actor import Actor
from learner import Learner

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL = "Qwen/Qwen3.5-0.8B"
MAX_OUTPUT_TOKENS = 2048
model = Qwen3_5ForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("mps")
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

def eval(checkpoint_path: Path | None, data_path: Path):
    if checkpoint_path is not None:
        learner.load_checkpoint(checkpoint_path)
    model.eval()

    puzzles = []
    for line in Path(data_path).read_text().splitlines():
        if line.strip():
            puzzles.append(list(map(int, line.split())))

    with torch.inference_mode():
        for puzzle in tqdm(puzzles, desc="Evaluating"):
            results = actor.collect_rollouts([puzzle], 25)[0]
            print(f"Cumulative reward: {sum(r.reward for r in results)}")

def train():
    BATCH_SIZE = 2
    GROUP_SIZE = 4

    data_path = Path(__file__).resolve().parent / "data/train.jsonl"
    train_data: list[list[int]] = []
    for line in data_path.read_text().splitlines():
        if line.strip():
            train_data.append(list(map(int, line.split()))) 

    train_data = train_data[:16] + train_data[-16:]
    train_data = [[6, 6, 6, 6]] * 306
    cumulated_rewards = []
    cumulated_reward_path = Path(__file__).resolve().parent / "checkpoints" / "cumulated_reward.jpg"

    for start in tqdm(range(0, len(train_data), BATCH_SIZE), desc="GRPO Training"):
        training_batch = train_data[start:start + BATCH_SIZE]
        rollouts = actor.collect_rollouts(training_batch, GROUP_SIZE)

        learner.zero_grad()
        for i, rollout_group in enumerate(rollouts):
            learner.accumulate_group(rollout_group, training_batch[i], GROUP_SIZE * BATCH_SIZE)
        cumulated_reward = learner.step()
        cumulated_rewards.append(cumulated_reward)
        print(f"Cumulated reward: {cumulated_reward}")
        create_graph(cumulated_reward_path, cumulated_rewards, "GRPO cumulated reward", "step", "cumulated reward")

        if start % 8 == 0:
            learner.create_checkpoint(Path(__file__).resolve().parent / "checkpoints" / f"checkpoint_{start}.pt")

    create_graph(cumulated_reward_path, cumulated_rewards, "GRPO cumulated reward", "step", "cumulated reward")
    learner.create_checkpoint(Path(__file__).resolve().parent / "checkpoints" / "final.pt")

if __name__ == "__main__":
    sanity_path = Path(__file__).resolve().parent / "data/sanity.jsonl"
    print("Evaluating base model")
    eval(None, sanity_path)
    checkpoint_path = Path(__file__).resolve().parent / "checkpoints/final.pt"
    print("Evaluating final checkpoint")
    eval(checkpoint_path, sanity_path)

    # train()