#!/bin/bash
# =============================================================================
# Unsloth ROCm MI300X (gfx942) Install Script
# Target: AMD MI300X / MI325X, gfx942, ROCm 7.0+
# Tested environment: AMD Alola cluster, ROCm 7.0.2
# =============================================================================
set -euo pipefail

ROCM_VERSION="${ROCM_VERSION:-7.0}"
TORCH_INDEX="https://download.pytorch.org/whl/rocm${ROCM_VERSION}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================="
echo "  ROCm-Unsloth MI300X Installation"
echo "  ROCm version: ${ROCM_VERSION}"
echo "  PyTorch index: ${TORCH_INDEX}"
echo "============================================="

# 1. Detect ROCm installation
if [ ! -f /opt/rocm/.info/version ]; then
    echo "[WARN] /opt/rocm/.info/version not found — ensure ROCm ${ROCM_VERSION} is installed"
    echo "       Install guide: https://rocm.docs.amd.com/en/latest/deploy/linux/index.html"
fi
if command -v rocminfo >/dev/null 2>&1; then
    DETECTED_GPU=$(rocminfo 2>/dev/null | grep -oP "Name:\s+gfx\K[^\s]+" | head -1)
    echo "Detected AMD GPU: gfx${DETECTED_GPU:-unknown}"
fi

# 2. Install PyTorch with ROCm support
echo ""
echo "[1/6] Installing PyTorch for ROCm ${ROCM_VERSION}..."
pip install torch torchvision torchaudio --index-url "${TORCH_INDEX}" --upgrade

# 3. Install Triton (AMD backend included in ROCm PyTorch)
echo ""
echo "[2/6] Installing Triton (AMD ROCm backend)..."
pip install triton

# 4. Install bitsandbytes ROCm
# Pre-release required for PR #1887 fix (NaN at decode shape on AMD GPUs)
# Drop this pin once bitsandbytes >= 0.50 ships on PyPI with the fix
echo ""
echo "[3/6] Installing bitsandbytes for AMD ROCm..."
ARCH=$(uname -m)
PY_VER=$(python3 -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
BNB_PRE_URL="https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_main/bitsandbytes-0.0.0-${PY_VER}-${PY_VER}-linux_${ARCH}.whl"
pip install "${BNB_PRE_URL}" --extra-index-url https://pypi.org/simple/ 2>/dev/null || {
    echo "  [WARN] Pre-release bitsandbytes unavailable, falling back to PyPI (4-bit decode may have NaN issue)"
    pip install "bitsandbytes>=0.49.1"
}

# 5. Install cut-cross-entropy (AMD ROCm compatible via triton)
echo ""
echo "[4/6] Installing cut-cross-entropy (faster CE loss via triton)..."
pip install cut-cross-entropy 2>/dev/null || echo "  [INFO] cut-cross-entropy unavailable, will use fallback CE loss"

# 6. Install HuggingFace ecosystem
echo ""
echo "[5/6] Installing HuggingFace / training dependencies..."
pip install \
    transformers>=4.45.0 \
    accelerate>=0.34.0 \
    peft>=0.13.0 \
    trl>=0.12.0 \
    datasets>=2.19.0 \
    huggingface-hub \
    sentencepiece \
    protobuf

# 7. Install ROCm-unsloth packages from this repo
echo ""
echo "[6/6] Installing ROCm-unsloth packages..."
pip install -e "${SCRIPT_DIR}/unsloth-zoo/" --no-build-isolation
pip install -e "${SCRIPT_DIR}/unsloth-studio/" --no-deps --no-build-isolation

echo ""
echo "============================================="
echo "  Installation complete!"
echo "============================================="
echo ""
echo "Verify installation:"
echo "  python3 -c \"from unsloth_zoo.device_type import DEVICE_TYPE, is_hip; print(f\'Device: {DEVICE_TYPE}, is_hip: {is_hip()}\')""
echo ""
echo "Quick test (requires AMD GPU):"
echo "  HIP_VISIBLE_DEVICES=0 python3 -c \"
echo "    import torch; from unsloth_zoo.device_type import DEVICE_TYPE, is_hip"
echo "    print(f\'torch.version.hip={torch.version.hip}, DEVICE_TYPE={DEVICE_TYPE}\')"
echo "    t = torch.randn(4, 4, device=\'cuda\'); print(\'GPU tensor OK\', t.shape)"
echo "  \""
