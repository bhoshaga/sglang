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
import signal
import subprocess
import sys
import time

MEMORY_THRESHOLD_PCT = 10

# Process name patterns (ROCm fallback only — NVIDIA path uses GPU-scoped kills)
SGLANG_PATTERNS = [
    r"sglang::",
    r"sglang\.launch_server",
    r"sglang\.bench",
    r"sglang\.data_parallel",
    r"sglang\.srt",
    r"sgl_diffusion::",
]

# Orchestrator patterns — when found as ancestors of GPU processes, kill to prevent respawn
ORCHESTRATOR_PATTERNS = ["run_suite.py", "run_tests.py"]


# ---------------------------------------------------------------------------
# nvidia-smi helpers
# ---------------------------------------------------------------------------


def _run_smi(query, query_type="gpu"):
    """Run nvidia-smi query and return raw CSV lines."""
    flag = "--query-gpu" if query_type == "gpu" else "--query-compute-apps"
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"{flag}={query}", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        )
        return [l.strip() for l in out.strip().splitlines() if l.strip()]
    except (subprocess.SubprocessError, FileNotFoundError):
        return []


def get_target_gpus():
    """Return list of GPU indices to target, based on CUDA_VISIBLE_DEVICES."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip():
        return [int(g.strip()) for g in cvd.split(",") if g.strip().isdigit()]
    return [int(l) for l in _run_smi("index")]


def get_gpu_uuid_map(gpu_indices):
    """Return set of UUIDs for the given GPU indices."""
    target_uuids = set()
    for line in _run_smi("index,uuid"):
        parts = line.split(",", 1)
        if len(parts) == 2:
            idx, uuid = int(parts[0].strip()), parts[1].strip()
            if idx in gpu_indices:
                target_uuids.add(uuid)
    return target_uuids


def get_gpu_pids(target_uuids):
    """Return set of PIDs using GPUs with the given UUIDs."""
    pids = set()
    for line in _run_smi("gpu_uuid,pid", query_type="apps"):
        parts = line.split(",", 1)
        if len(parts) == 2:
            uuid, pid = parts[0].strip(), parts[1].strip()
            if uuid in target_uuids and pid.isdigit():
                pids.add(int(pid))
    return pids


def get_gpu_memory(gpu_indices):
    """Return dict of {gpu_index: (used_mib, total_mib)}."""
    usage = {}
    for line in _run_smi("index,memory.used,memory.total"):
        parts = line.split(",")
        if len(parts) == 3:
            idx = int(parts[0].strip())
            if idx in gpu_indices:
                usage[idx] = (int(parts[1].strip()), int(parts[2].strip()))
    return usage


# ---------------------------------------------------------------------------
# Process tree helpers
# ---------------------------------------------------------------------------


def get_orchestrator_ancestors(pids):
    """Walk process tree upward from PIDs, return ancestors matching ORCHESTRATOR_PATTERNS."""
    ancestors = set()
    visited = set()
    for pid in pids:
        current = pid
        while current and current > 1 and current not in visited:
            visited.add(current)
            try:
                cmdline = open(f"/proc/{current}/cmdline", "rb").read()
                cmdline = cmdline.decode("utf-8", errors="replace").replace("\x00", " ")
                if any(p in cmdline for p in ORCHESTRATOR_PATTERNS):
                    ancestors.add(current)
            except (FileNotFoundError, PermissionError):
                break
            try:
                stat = open(f"/proc/{current}/stat").read().split()
                current = int(stat[3])  # PPID
            except (FileNotFoundError, PermissionError, IndexError, ValueError):
                break
    return ancestors


def kill_pids(pids, label=""):
    """Send SIGKILL to PIDs, skipping self and init."""
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


def pgrep(patterns):
    """Return PIDs matching regex patterns (for ROCm fallback)."""
    pids = set()
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "|".join(patterns)], text=True, timeout=10
        )
        pids = {int(l) for l in out.strip().splitlines() if l.strip().isdigit()}
    except (subprocess.CalledProcessError, subprocess.SubprocessError):
        pass
    return pids


# ---------------------------------------------------------------------------
# Main flows
# ---------------------------------------------------------------------------


def print_gpu_status(gpu_indices, header=""):
    """Print and return memory usage for target GPUs."""
    usage = get_gpu_memory(gpu_indices)
    if header:
        print(f"\n{header}")
    for idx in sorted(usage):
        used, total = usage[idx]
        pct = (used / total * 100) if total > 0 else 0
        print(f"  GPU {idx}: {used} MiB / {total} MiB ({pct:.0f}%)")
    return usage


def run_nvidia_cleanup(gpu_indices):
    """GPU-scoped cleanup: find PIDs on target GPUs, kill their process trees."""
    target_set = set(gpu_indices)
    target_uuids = get_gpu_uuid_map(target_set)
    print(f"Target GPUs: {sorted(target_set)}")
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        print(f"CUDA_VISIBLE_DEVICES={cvd}")

    print_gpu_status(target_set, "Before cleanup:")

    # Step 1: Find PIDs on target GPUs, kill orchestrator ancestors first
    gpu_pids = get_gpu_pids(target_uuids)
    if not gpu_pids:
        print("  No processes found on target GPUs")
    else:
        ancestors = get_orchestrator_ancestors(gpu_pids)
        if ancestors:
            kill_pids(ancestors, "orchestrator ancestors")
            time.sleep(1)

        # Step 2: Kill remaining GPU processes
        gpu_pids = get_gpu_pids(target_uuids)
        if gpu_pids:
            kill_pids(gpu_pids, "GPU processes")

    time.sleep(3)

    # Step 3: Second pass for stragglers
    gpu_pids = get_gpu_pids(target_uuids)
    if gpu_pids:
        kill_pids(gpu_pids, "stubborn GPU processes (second pass)")
        time.sleep(3)

    # Step 4: Verify
    usage = print_gpu_status(target_set, "After cleanup:")
    dirty = [
        f"GPU {i} ({u / t * 100:.0f}%)"
        for i, (u, t) in sorted(usage.items())
        if t > 0 and (u / t * 100) >= MEMORY_THRESHOLD_PCT
    ]
    if dirty:
        print(
            f"\nERROR: GPU memory still >={MEMORY_THRESHOLD_PCT}% after cleanup: "
            f"{', '.join(dirty)}"
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
    """ROCm cleanup — process-name kill only (no GPU query available)."""
    print("Running in ROCm mode")
    kill_pids(pgrep(ORCHESTRATOR_PATTERNS), "orchestrator processes")
    sglang_pids = pgrep(SGLANG_PATTERNS)
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
