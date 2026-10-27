from pathlib import Path
from transformers import Qwen3_5ForCausalLM
import torch
from tqdm import tqdm
from actor import Actor
from learner import Learner
from lora_linear import apply_lora
from train import empty_device_cache

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
    EVAL_BATCH_SIZE = 144
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
    empty_device_cache(DEVICE)
    return success_count, cumulative_reward


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
    """

    val_path = Path(__file__).resolve().parent / "data/val.jsonl"
    CHECKPOINT = 1127
    checkpoint_path = Path(__file__).resolve().parent / f"checkpoints/checkpoint_{CHECKPOINT}.pt"
    print(f"Evaluating checkpoint {CHECKPOINT}")
    final_success_count, final_reward = eval(checkpoint_path, val_path)
    print(f"Final checkpoint: success count {final_success_count}, reward {final_reward}")