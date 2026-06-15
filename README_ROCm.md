# ROCm-Unsloth — AMD MI300X Port

A ROCm-compatible port of [unsloth](https://github.com/unslothai/unsloth) for AMD GPUs, produced using the [AMDable](https://github.com/AMD-AIOSS/AMDable) CUDA→ROCm porting workflow.

## Hardware Target

| GPU | Architecture | ROCm | Status |
|-----|-------------|------|--------|
| AMD MI300X | gfx942 (CDNA3) | 7.0.2+ | ✅ Primary target |
| AMD MI325X | gfx942 (CDNA3) | 7.0.2+ | ✅ Same die |
| AMD MI355X | gfx950 (CDNA4) | 7.2+ | 🔶 Untested |
| AMD RX 7900 XTX | gfx1100 (RDNA3) | 6.0+ | 🔶 Untested |

## Repository Structure

```
ROCm-unsloth/
├── unsloth-zoo/          ← Core ML training kernels (ROCm-patched)
│   └── unsloth_zoo/      ← LoRA, QLoRA, fused losses, gradient checkpointing
├── unsloth-studio/       ← Studio UI + CLI (minimal AMD changes needed)
│   └── studio/           ← Web backend + training orchestration
├── install_rocm_mi300x.sh ← One-shot install script for MI300X
├── README_ROCm.md        ← This file
└── workspace/            ← Porting analysis, KB entries, intermediate outputs
```

## Installation

```bash
# Clone this repo
git clone https://github.com/AMD-AIOSS/ROCm-unsloth.git
cd ROCm-unsloth

# Install for MI300X with ROCm 7.0
ROCM_VERSION=7.0 bash install_rocm_mi300x.sh

# Or for ROCm 7.1/7.2:
ROCM_VERSION=7.1 bash install_rocm_mi300x.sh
```

### Manual Install

```bash
# PyTorch with ROCm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.0

# bitsandbytes (pre-release with AMD 4-bit decode fix)
pip install bitsandbytes>=0.49.1

# Core dependencies
pip install transformers accelerate peft trl datasets

# ROCm-unsloth packages
pip install -e ./unsloth-zoo/
pip install -e ./unsloth-studio/ --no-deps
```

## What Changed vs Upstream

The upstream unsloth already has a robust `DEVICE_TYPE` abstraction that routes between `"cuda"`, `"hip"`, `"xpu"`. Most code works on ROCm unchanged because PyTorch ROCm implements the `torch.cuda.*` API namespace.

### Changes Made

| File | Change | Reason |
|------|--------|--------|
| `unsloth_zoo/vllm_utils.py` | Added `is_hip()` guard around `torch.cuda.get_device_capability()` → `sm_cap` | SM architecture (SM80/SM90) has no AMD equivalent; gated CUTLASS FP8 and DeepGEMM paths |
| `unsloth_zoo/vllm_utils.py` | Added AMD skip for FlashInfer JIT | FlashInfer requires `nvcc` (CUDA compiler), not present on ROCm |
| `unsloth_zoo/tiled_mlp.py` | Added device guard around `torch.cuda.mem_get_info(0)` | Defensive guard for non-CUDA/HIP execution (XPU, CPU fallback paths) |
| `unsloth_zoo/saving_utils.py` | Changed `'device_type': 'cuda'` → `DEVICE_TYPE_TORCH` in GPU strategy dicts | Future-proofing for XPU / additional backends |

### Already Working Upstream (no changes needed)

- **`device_type.py`**: Complete AMD detection (`is_hip()`, `_detect_rocm_major_minor()`, `_detect_amd_rocm_runtime()`), `DEVICE_TYPE="hip"`, `DEVICE_TYPE_TORCH="cuda"` (PyTorch ROCm alias)
- **`compiler.py`**: `if DEVICE_TYPE == "hip": OLD_CUDA_ARCH_VERSION = False` already present
- **`loss_utils.py`**: HIP branch for `HAS_CUT_CROSS_ENTROPY` already present
- **`gradient_checkpointing.py`**: All `torch.cuda.*` calls properly guarded with `DEVICE_TYPE in ("cuda", "hip")`
- **`studio/backend/utils/hardware/amd.py`**: Full `amd-smi` integration for GPU monitoring
- **Triton kernels**: AMD triton backend (`triton-rocm`) is included with ROCm PyTorch — no `@triton.jit` changes needed

## Known Limitations

| Feature | Status | AMD Alternative |
|---------|--------|----------------|
| FlashInfer | ❌ Not supported (requires nvcc) | vLLM built-in paged attention |
| CUTLASS block FP8 | ❌ NVIDIA Hopper only (SM90) | hipBLASLt FP8 (separate integration needed) |
| DeepGEMM | ❌ NVIDIA Hopper only (SM90) | hipBLASLt algorithm search |
| SM-architecture dispatch | N/A | gfx9xx dispatch (composable_kernel) |
| Pre-quantized models (some) | ⚠️ Limited | bitsandbytes ROCm (pre-release) |

## Performance Notes (MI300X gfx942)

- **BF16 GEMM**: 843 TFLOPS at 8192³ via hipBLAS GemmEx (matches NVIDIA A100/H100 class)
- **Triton kernels**: Compile and run via AMD backend — same @triton.jit code, no changes
- **bitsandbytes**: 4-bit quantization works with pre-release (post PR#1887); blocksize=128 on AMD vs 64 on NVIDIA
- **Flash Attention**: Use `attn_implementation="flash_attention_2"` with transformers — this works via the ROCm flash-attn package (separate install if needed)

## Verification

```python
import torch
from unsloth_zoo.device_type import DEVICE_TYPE, DEVICE_TYPE_TORCH, is_hip

print(f"DEVICE_TYPE      = {DEVICE_TYPE}")       # Should be 'hip' on AMD
print(f"DEVICE_TYPE_TORCH= {DEVICE_TYPE_TORCH}") # Should be 'cuda' (PyTorch alias)
print(f"is_hip()         = {is_hip()}")           # Should be True on AMD
print(f"torch.version.hip= {torch.version.hip}")  # ROCm version string
print(f"GPU              = {torch.cuda.get_device_name(0)}")
```

## AMDable KB Entries

Porting patterns extracted during this port are in `workspace/kb_entries/` and will be promoted to the shared AMDable knowledge base.

## License

This port inherits the original unsloth/unsloth-zoo licenses (LGPL-3.0 / Apache-2.0). See individual package directories for details.
