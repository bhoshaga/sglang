#!/usr/bin/env python3
"""Kill SGLang and related processes on CUDA_VISIBLE_DEVICES GPUs.

Called at the start of every CI job to clean up orphaned processes from
previous (possibly cancelled) runs.

Usage:
    python killall_sglang.py          # Kill on CUDA_VISIBLE_DEVICES GPUs (or all if unset)
    python killall_sglang.py --rocm   # ROCm mode (process-name kill only, no GPU query)

Exit codes:
    0 - Clean: all target GPUs have <10% memory usage after cleanup
    1 - Dirty: GPU memory still >10% after cleanup, indicating stuck processes
        or orphaned CUDA contexts that need a container restart
"""

import os
import re
import signal
import subprocess
import sys
import time

# Patterns for sglang server/worker processes
SGLANG_PATTERNS = [
    r"sglang::",
    r"sglang\.launch_server",
    r"sglang\.bench",
    r"sglang\.data_parallel",
    r"sglang\.srt",
    r"sgl_diffusion::",
]

# Patterns for test harness processes that orchestrate sglang servers.
# If these survive, they respawn new servers and re-occupy GPUs.
TEST_HARNESS_PATTERNS = [
    r"run_suite\.py",
    r"run_tests\.py",
]

MEMORY_THRESHOLD_PCT = 10


def get_target_gpus():
    """Return list of GPU indices to target, based on CUDA_VISIBLE_DEVICES."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip():
        return [int(g.strip()) for g in cvd.split(",") if g.strip().isdigit()]
    # If unset, target all GPUs
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
            timeout=10,
        )
        return [int(line.strip()) for line in out.strip().splitlines() if line.strip()]
    except (subprocess.SubprocessError, FileNotFoundError):
        return []


def get_gpu_pids(gpu_indices):
    """Return set of PIDs using the specified GPUs."""
    pids = set()
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=10,
        )
        # Get UUIDs for our target GPUs
        uuid_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            text=True,
            timeout=10,
        )
        target_uuids = set()
        for line in uuid_out.strip().splitlines():
            parts = line.split(",")
            if len(parts) == 2:
                idx, uuid = int(parts[0].strip()), parts[1].strip()
                if idx in gpu_indices:
                    target_uuids.add(uuid)

        for line in out.strip().splitlines():
            parts = line.split(",")
            if len(parts) == 2:
                uuid, pid = parts[0].strip(), parts[1].strip()
                if uuid in target_uuids and pid.isdigit():
                    pids.add(int(pid))
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return pids


def get_pids_by_pattern(patterns):
    """Return set of PIDs matching any of the given regex patterns."""
    combined = "|".join(patterns)
    pids = set()
    try:
        out = subprocess.check_output(["pgrep", "-f", combined], text=True, timeout=10)
        for line in out.strip().splitlines():
            if line.strip().isdigit():
                pids.add(int(line.strip()))
    except (subprocess.CalledProcessError, subprocess.SubprocessError):
        # pgrep returns exit 1 when no matches
        pass
    return pids


def kill_pids(pids, label=""):
    """Send SIGKILL to a set of PIDs. Skip our own PID."""
    my_pid = os.getpid()
    pids = {p for p in pids if p != my_pid and p > 1}
    if not pids:
        return
    if label:
        print(f"  Killing {label}: {sorted(pids)}")
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            print(f"  Warning: no permission to kill PID {pid}")


def get_gpu_memory_usage(gpu_indices):
    """Return dict of {gpu_index: (used_mib, total_mib)} for target GPUs."""
    usage = {}
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
        for line in out.strip().splitlines():
            parts = line.split(",")
            if len(parts) == 3:
                idx = int(parts[0].strip())
                if idx in gpu_indices:
                    used = int(parts[1].strip())
                    total = int(parts[2].strip())
                    usage[idx] = (used, total)
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return usage


def print_gpu_status(gpu_indices, header=""):
    """Print memory usage for target GPUs."""
    usage = get_gpu_memory_usage(gpu_indices)
    if header:
        print(f"\n{header}")
    for idx in sorted(usage):
        used, total = usage[idx]
        pct = (used / total * 100) if total > 0 else 0
        print(f"  GPU {idx}: {used} MiB / {total} MiB ({pct:.0f}%)")
    return usage


def check_memory_clean(gpu_indices):
    """Check all target GPUs have <MEMORY_THRESHOLD_PCT% memory usage.

    Returns True if clean, False if dirty.
    """
    usage = get_gpu_memory_usage(gpu_indices)
    for idx in sorted(usage):
        used, total = usage[idx]
        if total > 0 and (used / total * 100) >= MEMORY_THRESHOLD_PCT:
            return False
    return True


def run_nvidia_cleanup(gpu_indices):
    """Main NVIDIA cleanup flow."""
    target_set = set(gpu_indices)
    print(f"Target GPUs: {sorted(target_set)}")
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        print(f"CUDA_VISIBLE_DEVICES={cvd}")

    print_gpu_status(target_set, "Before cleanup:")

    # Step 1: Kill test harness processes (they respawn sglang servers)
    harness_pids = get_pids_by_pattern(TEST_HARNESS_PATTERNS)
    if harness_pids:
        kill_pids(harness_pids, "test harness processes")

    # Step 2: Kill sglang server/worker processes by name
    sglang_pids = get_pids_by_pattern(SGLANG_PATTERNS)
    if sglang_pids:
        kill_pids(sglang_pids, "sglang processes")

    # Step 3: Kill any remaining processes holding GPU memory on our GPUs
    gpu_pids = get_gpu_pids(target_set)
    if gpu_pids:
        kill_pids(gpu_pids, "remaining GPU processes")

    # Wait for processes to die and GPU memory to free
    time.sleep(3)

    # Step 4: Second pass — catch anything respawned or missed
    gpu_pids = get_gpu_pids(target_set)
    if gpu_pids:
        print("  Second pass: processes still on GPUs")
        kill_pids(gpu_pids, "stubborn GPU processes")
        time.sleep(3)

    usage = print_gpu_status(target_set, "After cleanup:")

    # Step 5: Check if GPUs are clean
    if not check_memory_clean(target_set):
        dirty_gpus = []
        for idx in sorted(usage):
            used, total = usage[idx]
            pct = (used / total * 100) if total > 0 else 0
            if pct >= MEMORY_THRESHOLD_PCT:
                dirty_gpus.append(f"GPU {idx} ({pct:.0f}%)")
        print(
            f"\nERROR: GPU memory still >={MEMORY_THRESHOLD_PCT}% after cleanup: "
            f"{', '.join(dirty_gpus)}"
        )
        print(
            "This indicates orphaned CUDA contexts that survive process kill. "
            "The container likely needs to be restarted."
        )
        print("Aborting CI job.")
        return 1

    print("\nGPUs clean.")
    return 0


def run_rocm_cleanup():
    """ROCm cleanup — process-name kill only."""
    print("Running in ROCm mode")
    harness_pids = get_pids_by_pattern(TEST_HARNESS_PATTERNS)
    if harness_pids:
        kill_pids(harness_pids, "test harness processes")
    sglang_pids = get_pids_by_pattern(SGLANG_PATTERNS)
    if sglang_pids:
        kill_pids(sglang_pids, "sglang processes")
    else:
        print("  No sglang processes found")
    return 0


def main():
    if "--rocm" in sys.argv:
        return run_rocm_cleanup()

    gpu_indices = get_target_gpus()
    if not gpu_indices:
        print("No GPUs detected, skipping cleanup")
        return 0

    return run_nvidia_cleanup(gpu_indices)


if __name__ == "__main__":
    sys.exit(main())
