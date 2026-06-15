# Unsloth → ROCm-Unsloth Porting Analysis

**Date**: 2026-06-15  
**Target**: AMD MI300X (gfx942), ROCm 7.0.2  
**Method**: AMDable Strategy B (PyTorch extension — keep CUDA spelling, guard non-portable paths)  
**Source repos**: unslothai/unsloth (studio), unslothai/unsloth-zoo (core kernels)

---

## Executive Summary

Unsloth already has a substantial `DEVICE_TYPE` abstraction layer in `unsloth_zoo/device_type.py` that routes between `"cuda"`, `"hip"`, `"xpu"`, and `"mlx"`. The PyTorch ROCm build implements `torch.cuda.*` as aliases to the HIP API, so the vast majority of CUDA Python code runs unchanged on ROCm.

**Only 4 specific patterns required changes** across 3 files. All other CUDA references were either already guarded or work via PyTorch's ROCm aliasing.

---

## Pattern Inventory

### ALREADY HANDLED (no changes needed)

| Pattern | Location | Why it works on ROCm |
|---------|----------|---------------------|
| `torch.cuda.is_available()` | multiple | Returns True on ROCm (PyTorch alias) |
| `torch.cuda.device_count()` | `device_type.py:259` | ROCm aliased ✓ |
| `torch.cuda.synchronize()` | `device_type.py:288` | Already in `DEVICE_TYPE in ("cuda","hip")` guard |
| `torch.cuda.empty_cache()` | `device_type.py:299`, `vllm_utils.py` multiple | ROCm aliased ✓ |
| `torch.cuda.mem_get_info()` | `saving_utils.py:3473`, `gradient_checkpointing.py:490` | ROCm aliased, inside existing guards ✓ |
| `torch.cuda.Stream()` | `gradient_checkpointing.py:399` | Inside `DEVICE_TYPE_TORCH=="cuda"` guard (HIP gets DEVICE_TYPE_TORCH="cuda") ✓ |
| `torch.cuda.Event` | `gradient_checkpointing.py:376` | Inside `DEVICE_TYPE in ("cuda","hip")` guard ✓ |
| `torch.cuda.get_device_name()` | `saving_utils.py:3479` | ROCm aliased, inside `torch.cuda.is_available()` guard ✓ |
| `@triton.jit` kernels | `compiler.py` | AMD triton backend handles these transparently ✓ |
| `from triton import ...` | `loss_utils.py`, `compiler.py` | AMD triton included in ROCm PyTorch ✓ |
| `torch.amp.custom_fwd(device_type=DEVICE_TYPE)` | `tiled_mlp.py:41-42` | Uses `DEVICE_TYPE` variable (gets "hip") ✓ |
| `is_hip()` detection | `device_type.py` | Reads `torch.version.hip` — fully implemented ✓ |
| AMD GPU detection | `device_type.py:_detect_amd_rocm_runtime()` | rocminfo, amd-smi, /dev/kfd, ROCR_VISIBLE_DEVICES ✓ |
| bitsandbytes HIP branch | `device_type.py:269-276` | Checks bnb blocksize (64 CUDA → 128 HIP), gates pre-quantized models ✓ |
| Flash attention (transformers) | `compiler.py:223-244` | Uses `is_flash_attn_available()` from transformers — works on ROCm flash-attn ✓ |
| `torch.cuda.get_device_capability()` (in CUDA guard) | `compiler.py:88`, `vllm_utils.py:1858` | Already inside `if DEVICE_TYPE == "cuda":` blocks ✓ |
| `amd-smi` GPU monitoring | `studio/backend/utils/hardware/amd.py` | Full implementation already exists ✓ |

### CHANGES APPLIED

#### Fix 1: `vllm_utils.py` — SM capability gate (line ~894)

**Pattern**: `torch.cuda.get_device_capability()` used to compute `sm_cap`, which then gates CUTLASS block FP8 (`sm_cap==90`) and DeepGEMM (NVIDIA Hopper SM90 only).

**Problem**: This code ran at module level before any `DEVICE_TYPE` check. On AMD, `sm_cap` would be computed from whatever `get_device_capability()` returns on gfx942, then incorrectly used for Hopper-specific dispatch.

**Fix applied**:
```python
# BEFORE:
capability = torch.cuda.get_device_capability()
sm_cap = capability[0] * 10 + capability[1]

# AFTER:
if not is_hip():
    capability = torch.cuda.get_device_capability()
    sm_cap = capability[0] * 10 + capability[1]
else:
    sm_cap = 0  # Not applicable on AMD; gates all SM90-specific CUDA paths
```

#### Fix 2: `vllm_utils.py` — FlashInfer AMD skip (line ~1972)

**Pattern**: FlashInfer availability check followed by nvcc/ninja detection. If FlashInfer is installed, it tries to JIT-compile CUDA kernels using nvcc.

**Problem**: nvcc (CUDA compiler) is not present on ROCm hosts. FlashInfer has no HIP backend. Installing FlashInfer on ROCm would fail or produce wrong results.

**Fix applied**: Added `if is_hip():` early-exit before FlashInfer check that clears any FlashInfer env vars and prints informational message, then redirects to vLLM's built-in paged attention.

#### Fix 3: `tiled_mlp.py` — `torch.cuda.mem_get_info(0)` defensive guard (line ~224)

**Pattern**: `torch.cuda.mem_get_info(0)` called to determine available GPU memory for tiled MLP computation.

**Assessment**: This actually works on ROCm (PyTorch aliases it). However, it was called without any device guard, which would fail on XPU or CPU-only execution paths. Added defensive guard.

**Fix applied**:
```python
# BEFORE:
free, total = torch.cuda.mem_get_info(0)

# AFTER:
if DEVICE_TYPE in ("cuda", "hip") and torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info(0)
    ...
else:
    target_gb = 4.0  # Conservative default for non-CUDA/HIP devices
```

#### Fix 4: `saving_utils.py` — hardcoded `'device_type': 'cuda'` in strategy dicts

**Pattern**: Internal data structures tracking memory strategy include `'device_type': 'cuda'` string literals.

**Assessment**: These are internal markers, not torch device strings. On ROCm, GPU access is still via `'cuda'` (aliased). Changed to use `DEVICE_TYPE_TORCH` for future-proofing (XPU compatibility).

### NOT APPLICABLE (guarded correctly or NVIDIA-only)

| Pattern | Location | Decision |
|---------|----------|----------|
| CUTLASS block FP8 | `vllm_utils.py` | Guarded by `sm_cap==90` (now correctly 0 on AMD) — no AMD path needed |
| DeepGEMM | `vllm_utils.py` | NVIDIA Hopper only; `is_deep_gemm_supported()` returns False on AMD |
| `nvcc` / `CUDA_HOME` checks | `vllm_utils.py` | Now skipped entirely for AMD via FlashInfer guard |
| `torch.ops._C.cutlass_scaled_mm_supports_block_fp8` | `vllm_utils.py` | sm_cap=0 prevents this from ever being called on AMD |
| SM90/Hopper features | various | Already gated by `sm_cap >= 90` checks that now evaluate to False on AMD |

---

## AMD-Specific Recommendations for MI300X

### Attention computation
- Use `attn_implementation="flash_attention_2"` with transformers flash-attn ROCm build
- Alternatively use `"sdpa"` (PyTorch scaled dot product attention — works on ROCm)
- Do NOT use FlashInfer (CUDA only)

### Quantization  
- bitsandbytes 4-bit: use pre-release with PR#1887 fix (NaN at decode shape)
- blocksize=128 on AMD vs 64 on NVIDIA (handled automatically by upstream unsloth_zoo)
- Pre-quantized model loading may be gated (`ALLOW_PREQUANTIZED_MODELS=False`) depending on bnb version

### GEMM performance
- hipBLAS GemmEx with COMPUTE_32F: 688-843 TFLOPS at various sizes (matches A100/H100)
- hipBLASLt algorithm search: up to 785 TFLOPS — see AMDable KB entry `hipblaslt_algorithm_search`
- Triton kernels: AMD backend handles @triton.jit transparently; same code compiles

### vLLM integration
- Use `VLLM_ATTENTION_BACKEND=FLASH_ATTN` (default) on AMD
- Do not set `VLLM_USE_FLASHINFER_SAMPLER=1` on AMD
- Float8 KV cache: AMD FP8 support differs from NVIDIA; test carefully

---

## Files Changed Summary

```
unsloth-zoo/unsloth_zoo/
├── vllm_utils.py          ← +is_hip import, SM cap guard, FlashInfer AMD skip
├── tiled_mlp.py           ← mem_get_info defensive guard
└── saving_utils.py        ← DEVICE_TYPE_TORCH in strategy dicts
```

**Files NOT changed (already AMD-compatible)**:
- `device_type.py` — complete AMD abstraction
- `compiler.py` — HIP branches present
- `loss_utils.py` — HIP branch present
- `gradient_checkpointing.py` — all properly guarded
- `fused_losses/` — pure PyTorch math, portable
- `patching_utils.py` — no GPU-specific code
- `studio/backend/utils/hardware/amd.py` — full AMD implementation

---

## Next Steps

1. **Validate on MI300X**: Submit a Slurm job to Alola cluster to verify import and basic training loop
2. **Benchmark**: Run LoRA fine-tuning on Llama-3-8B with bfloat16, measure throughput vs NVIDIA baseline
3. **Flash attention**: Install `flash-attn` ROCm build and test `flash_attention_2` path
4. **vLLM inference**: Test `get_vllm_llm()` with a Llama-3 model on MI300X
5. **Promote KB entries**: Promote workspace/kb_entries/ YAMLs to AMDable shared knowledge base
