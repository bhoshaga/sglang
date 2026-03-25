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

# Orchestrator patterns — ancestors of GPU processes; kill to prevent respawn
ORCHESTRATOR_PATTERNS = ["run_suite.py", "run_tests.py"]


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
    """Return GPU indices from CUDA_VISIBLE_DEVICES, or all visible GPUs."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip():
        return [int(g.strip()) for g in cvd.split(",") if g.strip().isdigit()]
    return [int(l) for l in _run_smi("index")]


def get_gpu_pids(gpu_indices):
    """Return PIDs using the specified GPUs (by index)."""
    # Build index→UUID mapping
    target_uuids = set()
    for line in _run_smi("index,uuid"):
        parts = line.split(",", 1)
        if len(parts) == 2 and int(parts[0].strip()) in gpu_indices:
            target_uuids.add(parts[1].strip())
    # Find PIDs on those UUIDs
    pids = set()
    for line in _run_smi("gpu_uuid,pid", query_type="apps"):
        parts = line.split(",", 1)
        if len(parts) == 2 and parts[0].strip() in target_uuids:
            pid = parts[1].strip()
            if pid.isdigit():
                pids.add(int(pid))
    return pids


def get_gpu_memory(gpu_indices):
    """Return {gpu_index: (used_mib, total_mib)} for target GPUs."""
    usage = {}
    for line in _run_smi("index,memory.used,memory.total"):
        parts = line.split(",")
        if len(parts) == 3:
            idx = int(parts[0].strip())
            if idx in gpu_indices:
                usage[idx] = (int(parts[1].strip()), int(parts[2].strip()))
    return usage


def get_orchestrator_ancestors(pids):
    """Walk process tree upward from PIDs, return ancestors matching ORCHESTRATOR_PATTERNS."""
    ancestors, visited = set(), set()
    for pid in pids:
        current = pid
        while current > 1 and current not in visited:
            visited.add(current)
            try:
                cmdline = open(f"/proc/{current}/cmdline", "rb").read()
                cmdline = cmdline.decode("utf-8", errors="replace").replace("\x00", " ")
                if any(p in cmdline for p in ORCHESTRATOR_PATTERNS):
                    ancestors.add(current)
            except (FileNotFoundError, PermissionError):
                break
            try:
                current = int(open(f"/proc/{current}/stat").read().split()[3])
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
        except (ProcessLookupError, PermissionError):
            pass


def pgrep(patterns):
    """Return PIDs matching regex patterns (for ROCm fallback)."""
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "|".join(patterns)], text=True, timeout=10
        )
        return {int(l) for l in out.strip().splitlines() if l.strip().isdigit()}
    except (subprocess.CalledProcessError, subprocess.SubprocessError):
        return set()


def run_nvidia_cleanup(gpu_indices):
    """GPU-scoped cleanup: find PIDs on target GPUs, kill their process trees."""
    target_set = set(gpu_indices)
    print(f"Target GPUs: {sorted(target_set)}")
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        print(f"CUDA_VISIBLE_DEVICES={cvd}")

    # Show before state
    for idx, (used, total) in sorted(get_gpu_memory(target_set).items()):
        print(f"  GPU {idx}: {used} MiB / {total} MiB ({used / total * 100:.0f}%)")

    # Kill orchestrator ancestors first, then GPU processes (retry once)
    gpu_pids = get_gpu_pids(target_set)
    if not gpu_pids:
        print("  No processes found on target GPUs")
    else:
        kill_pids(get_orchestrator_ancestors(gpu_pids), "orchestrator ancestors")
        time.sleep(1)
        for attempt in range(2):
            gpu_pids = get_gpu_pids(target_set)
            if not gpu_pids:
                break
            label = "GPU processes" if attempt == 0 else "stubborn GPU processes"
            kill_pids(gpu_pids, label)
            time.sleep(3)

    # Verify
    usage = get_gpu_memory(target_set)
    print("\nAfter cleanup:")
    dirty = []
    for idx, (used, total) in sorted(usage.items()):
        pct = used / total * 100 if total > 0 else 0
        print(f"  GPU {idx}: {used} MiB / {total} MiB ({pct:.0f}%)")
        if pct >= MEMORY_THRESHOLD_PCT:
            dirty.append(f"GPU {idx} ({pct:.0f}%)")

    if dirty:
        print(
            f"\nERROR: GPU memory >={MEMORY_THRESHOLD_PCT}% after cleanup: {', '.join(dirty)}"
        )
        print("Orphaned CUDA contexts — container likely needs restart. Aborting CI.")
        return 1
    print("\nGPUs clean.")
    return 0


def run_rocm_cleanup():
    """ROCm cleanup — process-name kill only (no GPU query available)."""
    print("Running in ROCm mode")
    kill_pids(pgrep(ORCHESTRATOR_PATTERNS), "orchestrator processes")
    sglang_pids = pgrep(SGLANG_PATTERNS)
    (
        kill_pids(sglang_pids, "sglang processes")
        if sglang_pids
        else print("  No sglang processes found")
    )
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
