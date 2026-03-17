#!/bin/bash
# Install comparison frameworks for diffusion benchmarks.
#
# IMPORTANT: vllm-omni, lightx2v, and sglang have conflicting torch/dep versions.
# They CANNOT coexist in the same pip environment. Instead, run_comparison.py
# handles this by running each framework sequentially — all sglang cases first
# (no extra install needed), then installing vllm-omni and running its cases,
# then installing lightx2v and running its cases.
#
# This script is called by run_comparison.py before each framework's cases.
# Usage: bash install_comparison_frameworks.sh <framework>

set -e

FRAMEWORK="${1:-all}"

install_vllm_omni() {
    echo "=== Installing vllm + vllm-omni ==="
    pip install --no-deps "vllm==0.16.0" 2>&1 | tail -3 || echo "WARNING: vllm install failed"
    pip install --force-reinstall --no-deps "vllm-omni==0.16.0" 2>&1 | tail -3 || echo "WARNING: vllm-omni install failed"
    # vllm may have downgraded flashinfer; restore sglang's pinned version
    SGLANG_FI_VER=$(pip show sglang 2>/dev/null | grep -i requires | grep -oP 'flashinfer.python==\K[0-9.]+' || true)
    if [ -n "$SGLANG_FI_VER" ]; then
        echo "Restoring flashinfer-python==$SGLANG_FI_VER..."
        pip install --no-deps "flashinfer-python==$SGLANG_FI_VER" 2>&1 | tail -2 || true
    fi
    echo "=== vllm-omni installation complete ==="
}

install_lightx2v() {
    echo "=== Installing LightX2V ==="
    pip install --no-deps "lightx2v @ git+https://github.com/ModelTC/LightX2V.git" 2>&1 | tail -5 || echo "WARNING: LightX2V install failed"
    echo "=== LightX2V installation complete ==="
}

restore_sglang() {
    echo "=== Restoring sglang entry points ==="
    # Re-install sglang editable to restore CLI entry points after vllm-omni overwrites /usr/local/bin/sglang
    pip install -e ".[diffusion]" --no-deps 2>/dev/null || true
    echo "=== sglang restored ==="
}

case "$FRAMEWORK" in
    vllm-omni)
        install_vllm_omni
        ;;
    lightx2v)
        install_lightx2v
        ;;
    restore-sglang)
        restore_sglang
        ;;
    all)
        install_vllm_omni
        install_lightx2v
        ;;
    *)
        echo "Unknown framework: $FRAMEWORK"
        echo "Usage: $0 {vllm-omni|lightx2v|restore-sglang|all}"
        exit 1
        ;;
esac
