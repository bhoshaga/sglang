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

# Patterns for sglang server/worker processes (used only in ROCm fallback)
SGLANG_PATTERNS = [
    r"sglang::",
    r"sglang\.launch_server",
    r"sglang\.bench",
    r"sglang\.data_parallel",
    r"sglang\.srt",
    r"sgl_diffusion::",
]

# Patterns for test harness / orchestrator processes.
# When found as an ancestor of a GPU process, killing these prevents respawning.
TEST_HARNESS_PATTERNS = [
    "run_suite.py",
    "run_tests.py",
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


def get_ancestor_pids(pids):
    """Walk the process tree upward from the given PIDs.

    Returns a set of ancestor PIDs that match TEST_HARNESS_PATTERNS.
    These are orchestrator processes (run_suite.py, etc.) that would
    respawn sglang servers if left alive.
    """
    ancestors = set()
    visited = set()

    for pid in pids:
        current = pid
        while current and current > 1 and current not in visited:
            visited.add(current)
            try:
                with open(f"/proc/{current}/cmdline", "rb") as f:
                    cmdline = f.read().decode("utf-8", errors="replace")
                    cmdline = cmdline.replace("\x00", " ").strip()
            except (FileNotFoundError, PermissionError):
                break

            for pattern in TEST_HARNESS_PATTERNS:
                if pattern in cmdline:
                    ancestors.add(current)
                    break

            # Walk to parent
            try:
                with open(f"/proc/{current}/stat") as f:
                    stat = f.read().split()
                    # Field 4 (0-indexed 3) is PPID
                    ppid = int(stat[3])
                    current = ppid
            except (FileNotFoundError, PermissionError, (IndexError, ValueError)):
                break

    return ancestors


def get_child_pids(pids):
    """Get all descendant PIDs of the given PIDs."""
    children = set()
    # Build parent->children map from /proc
    parent_map = {}
    try:
        out = subprocess.check_output(
            ["ps", "-eo", "pid,ppid", "--no-headers"], text=True, timeout=10
        )
        for line in out.strip().splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                child, parent = int(parts[0]), int(parts[1])
                parent_map.setdefault(parent, set()).add(child)
    except (subprocess.SubprocessError, FileNotFoundError):
        return children

    # BFS from given PIDs
    queue = list(pids)
    while queue:
        pid = queue.pop(0)
        for child in parent_map.get(pid, set()):
            if child not in children:
                children.add(child)
                queue.append(child)
    return children


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
    """Check all target GPUs have <MEMORY_THRESHOLD_PCT% memory usage."""
    usage = get_gpu_memory_usage(gpu_indices)
    for idx in sorted(usage):
        used, total = usage[idx]
        if total > 0 and (used / total * 100) >= MEMORY_THRESHOLD_PCT:
            return False
    return True


def run_nvidia_cleanup(gpu_indices):
    """Main NVIDIA cleanup flow.

    All kills are GPU-scoped: we start from PIDs on target GPUs (via nvidia-smi),
    walk the process tree to find orchestrators, and kill the whole tree.
    This ensures processes on other GPUs are never touched.
    """
    target_set = set(gpu_indices)
    print(f"Target GPUs: {sorted(target_set)}")
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        print(f"CUDA_VISIBLE_DEVICES={cvd}")

    print_gpu_status(target_set, "Before cleanup:")

    # Step 1: Find PIDs on our target GPUs
    gpu_pids = get_gpu_pids(target_set)
    if not gpu_pids:
        print("  No processes found on target GPUs")
    else:
        # Step 2: Walk upward to find orchestrator ancestors (run_suite.py, etc.)
        # Killing these first prevents them from respawning sglang servers.
        ancestor_pids = get_ancestor_pids(gpu_pids)
        if ancestor_pids:
            # Also collect all descendants of ancestors (siblings on other tests, etc.)
            descendant_pids = get_child_pids(ancestor_pids)
            # Only kill descendants that are on our target GPUs or are the ancestors themselves
            scoped_descendants = descendant_pids & gpu_pids
            kill_pids(
                ancestor_pids, "orchestrator processes (ancestors of GPU processes)"
            )
            if scoped_descendants - ancestor_pids:
                kill_pids(scoped_descendants, "descendant processes on target GPUs")

        # Step 3: Kill GPU processes directly
        # Re-query in case ancestors already cleaned up children
        time.sleep(1)
        gpu_pids = get_gpu_pids(target_set)
        if gpu_pids:
            kill_pids(gpu_pids, "GPU processes")

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
    """ROCm cleanup — process-name kill only (no GPU query available)."""
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
