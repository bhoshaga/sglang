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

    # Install vllm and vllm-omni without deps to avoid torch version conflicts
    # (torch version strings like 2.9.1+cu128 are not on PyPI)
    pip install --no-deps "vllm==0.16.0" 2>&1 | tail -3 || echo "WARNING: vllm install failed"
    pip install --force-reinstall --no-deps "vllm-omni==0.16.0" 2>&1 | tail -3 || echo "WARNING: vllm-omni install failed"

    # Install lightweight deps that vllm-omni needs at runtime
    pip install aenum cachetools cbor2 omegaconf prettytable torchsde 2>&1 | tail -3 || true

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
