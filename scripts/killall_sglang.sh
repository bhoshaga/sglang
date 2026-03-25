#!/bin/bash
# Thin wrapper — real logic is in killall_sglang.py
# Kept for backwards compatibility with manual invocations.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARGS=()
if [ "$1" = "rocm" ]; then
    ARGS+=("--rocm")
fi

exec python3 "${SCRIPT_DIR}/killall_sglang.py" "${ARGS[@]}"
