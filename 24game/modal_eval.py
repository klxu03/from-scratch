import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

import modal

GAME_DIR = Path(__file__).resolve().parent
REPO_ROOT = GAME_DIR.parent

app = modal.App("game24-eval")
cache_volume = modal.Volume.from_name("game24-eval", create_if_missing=True)

# Cache dependencies and datasets; mount changing Python code at runtime.
image = (
    modal.Image.debian_slim(python_version="3.14")
    .uv_sync(uv_project_dir=str(REPO_ROOT), extras=["cuda"])
    .add_local_dir(GAME_DIR / "data", "/root/24game/data", copy=True)
    .env({"HF_HOME": "/vol/hf", "PYTHONUNBUFFERED": "1"})
)

for source in sorted(GAME_DIR.glob("*.py")):
    if source.name != "modal_eval.py":
        image = image.add_local_file(source, f"/root/24game/{source.name}")
image = image.add_local_dir(REPO_ROOT / "utils", "/root/utils")


@app.function(
    image=image,
    gpu="A100-40GB",
    cpu=2,
    memory=4096,
    max_containers=1,
    timeout=12 * 3600,
    volumes={"/vol": cache_volume},
)
def run_eval(checkpoint_name: str, split: str, runs: int, job_id: str) -> int:
    checkpoint_path = Path("/vol/checkpoints") / f"{job_id}_{checkpoint_name}"

    try:
        # A reused container may have mounted the Volume before this upload.
        cache_volume.reload()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)

        sys.path[:0] = ["/root/24game", "/root"]
        import torch
        import eval as game_eval  # Initializes Qwen on the GPU.

        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        cache_volume.commit()  # Save downloaded Qwen weights immediately.
        game_eval.traces_dir = (
            Path("/vol/traces") / f"{Path(checkpoint_name).stem}_{job_id}" / split
        )
        game_eval.traces_dir.mkdir(parents=True, exist_ok=True)

        data_path = (
            None if split == "both" else Path(f"/root/24game/data/{split}.jsonl")
        )
        rewards = game_eval.eval(checkpoint_path, data_path, runs_per_puzzle=runs)
        solved = sum(score > 0 for score in rewards.values())

        report = {
            "checkpoint": checkpoint_name,
            "split": split,
            "runs": runs,
            "success_count": solved,
            "total_puzzles": len(rewards),
            "puzzles": {
                "_".join(map(str, puzzle)): float(score)
                for puzzle, score in rewards.items()
            },
        }
        report_path = (
            Path("/vol/reports")
            / f"{Path(checkpoint_name).stem}_{split}_{runs}_{job_id}.json"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2))

        print(f"Solved: {solved}/{len(rewards)}", flush=True)
        print(f"Peak GPU memory: {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB", flush=True)
        print(f"Report: {report_path}", flush=True) # uv run modal volume ls game24-eval reports
        return solved

    finally:
        checkpoint_path.unlink(missing_ok=True)
        cache_volume.commit()  # Persist traces/report and checkpoint deletion.


# Run from the repository root:
# uv run --with modal python 24game/modal_eval.py \
#     --checkpoint 24game/checkpoints/checkpoint_1911.pt \
#     --split both \
#     --runs 16
# Use --split val or --split test to evaluate only one split.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["both", "val", "test"], default="both")
    parser.add_argument("--runs", type=int, default=16)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint not found: {checkpoint}")
    if args.runs < 1:
        parser.error("--runs must be at least 1")

    # Unique staging name avoids collisions between simultaneous submissions.
    job_id = uuid4().hex[:12]
    with modal.enable_output(), app.run(detach=True):
        with cache_volume.batch_upload() as batch:
            batch.put_file(str(checkpoint), f"/checkpoints/{job_id}_{checkpoint.name}")

        call = run_eval.spawn(checkpoint.name, args.split, args.runs, job_id)
        print(f"Submitted: {checkpoint.name} (job {job_id})")
        print(f"Function call ID: {call.object_id}")
        print(f"Logs: {call.get_dashboard_url()}")
    
if __name__ == "__main__":
    main()
