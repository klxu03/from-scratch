import csv
import time
from pathlib import Path

import psutil
import torch

def reset_peak_memory(device: torch.device) -> int | None:
    """Reset the CUDA peak counters before a step. Returns the retry count to compare against later."""
    if device.type != "cuda":
        return None
    torch.cuda.reset_peak_memory_stats(device)
    return torch.cuda.memory_stats(device)["num_alloc_retries"]

def print_memory(device: torch.device, retries_before: int | None = None) -> None:
    if device.type != "cuda":
        return
    torch.cuda.synchronize(device)
    memory_stats = torch.cuda.memory_stats(device)
    retries = memory_stats["num_alloc_retries"]
    if retries_before is not None:
        retries -= retries_before
    print(
        f"Peak live: "
        f"{torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB | "
        f"Peak reserved: "
        f"{torch.cuda.max_memory_reserved(device) / 2**30:.2f} GiB | "
        f"Allocation retries this step: "
        f"{retries}",
        flush=True,
    )

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
def log_memory(step: int, phase: str = "after_step", device: torch.device | None = None, csv_path: Path | None = None) -> dict:
    """Record memory without clearing gradients, objects, or caches."""
    if device is None:
        raise ValueError("device is required")
    if csv_path is None:
        raise ValueError("csv_path is required")
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
        "disk_free_gib": psutil.disk_usage(csv_path.parent).free / gib,
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
        not csv_path.exists() or csv_path.stat().st_size == 0
    )
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if needs_header:
            writer.writeheader()
        writer.writerow(row)

    return row
