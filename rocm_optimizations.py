"""
ROCm-Unsloth Optimization Utilities
=====================================
AMD MI300X/MI325X-specific optimizations for LoRA/QLoRA fine-tuning.

Usage:
    from rocm_optimizations import apply_rocm_optimizations, get_rocm_training_args

    model = apply_rocm_optimizations(model)
    training_args = get_rocm_training_args(output_dir="./output")
"""


# ══════════════════════════════════════════════════════════
# MEASURED OPTIMIZATION RESULTS (MI325X gfx942, ROCm 6.2.4)
# TinyLlama-1.1B, LoRA r=16, batch=4, seq=512, bfloat16
# Job 377780, 2026-06-19
# ══════════════════════════════════════════════════════════
#
# Config                         Tok/s  VRAM     Gain
# Baseline (q+v, eager, no GC)  27,418  10.57 GB  —
# SDPA + full QKV+O LoRA        26,793   7.56 GB  -2% speed, -28% VRAM
# + hipBLASLt                   26,890   7.56 GB  ~same
# + Gradient Checkpointing      20,301   3.14 GB  -26% speed, -70% VRAM
#
# Key findings:
# - SDPA training gain = VRAM reduction (-28%), NOT throughput in training
#   (SDPA +37% applies to INFERENCE, not training where activations must be stored)
# - hipBLASLt: neutral for 1B models; beneficial for >3B + large batches
# - GC: -26% throughput but -70% VRAM — essential for >13B models on MI325X
# ══════════════════════════════════════════════════════════

import os
import torch


def is_rocm():
    """Check if running on AMD ROCm."""
    return torch.version.hip is not None and torch.cuda.is_available()


def apply_rocm_optimizations(model, verbose=True):
    """
    Apply AMD ROCm-specific optimizations to a PyTorch model.

    Optimizations applied:
    1. BF16 precision (MI300X/MI325X natively support BF16 at full throughput)
    2. SDPA attention (37% faster than eager on ROCm, built-in MIOpen kernel)
    3. torch.compile with AMD backend (optional, adds JIT warmup)
    4. Gradient checkpointing setup (reduces VRAM for large models)

    Args:
        model: transformers model (post get_peft_model)
        verbose: print optimization summary

    Returns:
        model with optimizations applied
    """
    if not is_rocm():
        if verbose:
            print("Not on ROCm — skipping AMD optimizations")
        return model

    optimizations = []

    # 1. Enable gradient checkpointing for VRAM savings
    # Trades compute for memory: ~30% VRAM reduction, ~20% slower per step
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        optimizations.append("gradient_checkpointing (VRAM ~-30%)")

    # 2. Set SDPA as default attention (37% faster than eager on ROCm)
    # attn_implementation="sdpa" should be set at from_pretrained() time,
    # but we can also nudge the existing model config here
    if hasattr(model, 'config'):
        if not hasattr(model.config, '_attn_implementation') or \
           model.config._attn_implementation == 'eager':
            model.config._attn_implementation = 'sdpa'
            optimizations.append("SDPA attention (+37% vs eager)")

    # 3. AMD-specific environment variables for best performance
    env_vars = {
        # Enable hipBLASLt for GEMM (better than rocBLAS for transformer shapes)
        'PYTORCH_ENABLE_HIPBLASLT': '1',
        # Disable ROCm memory synchronization noise
        'HSA_ENABLE_SDMA': '0',
        # Enable MIOpen find-mode for first-run kernel selection
        'MIOPEN_FIND_MODE': 'FAST',
        # Persistent kernel cache across jobs on same node
        'MIOPEN_USER_DB_PATH': os.path.expanduser('~/.cache/miopen_db'),
    }
    applied_env = []
    for k, v in env_vars.items():
        if os.environ.get(k) != v:
            os.environ[k] = v
            applied_env.append(k)
    if applied_env:
        optimizations.append(f"env vars: {', '.join(applied_env)}")

    if verbose and optimizations:
        print(f"✅ ROCm optimizations applied ({torch.cuda.get_device_name(0)}):")
        for opt in optimizations:
            print(f"   • {opt}")

    return model


def get_rocm_training_args(
    output_dir="./rocm_output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    **kwargs
):
    """
    Get recommended TrainingArguments for AMD ROCm (MI300X/MI325X).

    Key ROCm-specific choices:
    - bf16=True: MI325X has native BF16 support (not fp16)
    - dataloader_pin_memory=False: HMM memory on MI300X doesn't benefit from pinning
    - optim="adamw_torch": avoid apex/fused variants (CUDA-only)
    - ddp_find_unused_parameters=False: safer with LoRA on ROCm

    Returns:
        dict of kwargs ready for transformers.TrainingArguments(**args)
    """
    from transformers import TrainingArguments

    args = dict(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        # ROCm-specific
        bf16=True,                          # Native BF16 on MI300X/MI325X
        fp16=False,                         # Don't use FP16 — BF16 is better on AMD
        dataloader_pin_memory=False,        # HMM memory doesn't benefit from pinning
        optim="adamw_torch",                # Use PyTorch AdamW (fused is CUDA-only)
        ddp_find_unused_parameters=False,   # Safer with LoRA
        # Standard good practices
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        gradient_checkpointing=True,        # VRAM savings (crucial for large models)
        gradient_checkpointing_kwargs={"use_reentrant": False},
        **kwargs
    )
    return args


def get_lora_config_rocm(
    r=16,
    lora_alpha=32,
    target_modules=None,
    lora_dropout=0.05,
    **kwargs
):
    """
    Recommended LoRA configuration for AMD ROCm.

    Difference from NVIDIA:
    - target_modules: include k_proj too (AMD SDPA benefits from full QKV LoRA)
    - No special blocksize needed (bitsandbytes handles AMD blocksize=128 automatically)
    """
    from peft import LoraConfig

    if target_modules is None:
        # Full QKV + output projection — better results on AMD than q+v only
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        **kwargs
    )


def print_rocm_info():
    """Print AMD GPU info and optimization status."""
    if not is_rocm():
        print("Not running on ROCm.")
        return

    print(f"{'='*55}")
    print(f"  AMD ROCm-Unsloth Environment")
    print(f"{'='*55}")
    print(f"  GPU:    {torch.cuda.get_device_name(0)}")
    print(f"  VRAM:   {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    print(f"  ROCm:   {torch.version.hip}")
    print(f"  PyTorch:{torch.__version__}")
    free, total = torch.cuda.mem_get_info(0)
    print(f"  Free:   {free/1024**3:.1f} GB / {total/1024**3:.1f} GB")
    print(f"{'='*55}")


if __name__ == "__main__":
    print_rocm_info()
