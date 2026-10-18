from pathlib import Path
from transformers import Qwen3_5ForCausalLM
import torch
from tqdm import tqdm
from actor import Actor

MODEL = "Qwen/Qwen3.5-0.8B"
MAX_OUTPUT_TOKENS = 4096
model = Qwen3_5ForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("mps")
actor = Actor(model, MAX_OUTPUT_TOKENS)

BATCH_SIZE = 4
GROUP_SIZE = 8

data_path = Path(__file__).resolve().parent / "data/train.jsonl"
train_data: list[list[int]] = []
for line in data_path.read_text().splitlines():
    if line.strip():
        train_data.append(list(map(int, line.split()))) 

train_data = train_data[:16] + train_data[-16:]

for start in tqdm(range(0, len(train_data), BATCH_SIZE), desc="GRPO Training"):
    training_batch = train_data[start:start + BATCH_SIZE]
    rollouts = actor.collect_rollouts(training_batch, GROUP_SIZE)