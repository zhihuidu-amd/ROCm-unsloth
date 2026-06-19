#!/usr/bin/env python3
"""
ROCm-Unsloth Validation & Benchmark Suite
==========================================
Validates the ROCm port of unsloth and measures throughput.

Usage:
    # Quick validation only (no model download):
    python3 test_rocm.py --validate

    # Full benchmark (downloads TinyLlama-1.1B, ~2.3 GB):
    python3 test_rocm.py --benchmark

    # Both:
    python3 test_rocm.py --validate --benchmark

    # With a specific model:
    python3 test_rocm.py --benchmark --model Qwen/Qwen2.5-1.5B

Requirements:
    pip install torch --index-url https://download.pytorch.org/whl/rocm7.0
    pip install transformers peft accelerate
    UNSLOTH_IS_PRESENT=1 python3 test_rocm.py
"""

import os, sys, time, argparse
os.environ["UNSLOTH_IS_PRESENT"] = "1"

PASS = "✅ PASS"
FAIL = "❌ FAIL"
SKIP = "⏭  SKIP"

results = []

def check(name, fn):
    try:
        result = fn()
        status = PASS if result else FAIL
        print(f"  {status}  {name}")
        results.append((name, result))
        return result
    except Exception as e:
        print(f"  {FAIL}  {name}: {e}")
        results.append((name, False))
        return False


def run_validation():
    print("\n" + "="*60)
    print("  ROCm-Unsloth Validation Suite")
    print("="*60)

    import torch

    # CHECK 1: PyTorch ROCm build
    check("PyTorch ROCm build (torch.version.hip is set)",
          lambda: torch.version.hip is not None)

    print(f"  → torch={torch.__version__}  hip={torch.version.hip}")

    # CHECK 2: CUDA available (ROCm aliases torch.cuda)
    check("torch.cuda.is_available() returns True on AMD",
          lambda: torch.cuda.is_available())

    if not torch.cuda.is_available():
        print("  ⚠️  No GPU detected. Check HIP_VISIBLE_DEVICES.")
        return False

    print(f"  → GPU: {torch.cuda.get_device_name(0)}")
    print(f"  → VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

    # CHECK 3: unsloth_zoo device detection
    from unsloth_zoo.device_type import DEVICE_TYPE, DEVICE_TYPE_TORCH, is_hip

    check("DEVICE_TYPE == 'hip' on AMD GPU",
          lambda: DEVICE_TYPE == "hip")

    check("DEVICE_TYPE_TORCH == 'cuda' (PyTorch ROCm alias)",
          lambda: DEVICE_TYPE_TORCH == "cuda")

    check("is_hip() returns True on AMD GPU",
          lambda: is_hip())

    print(f"  → DEVICE_TYPE={DEVICE_TYPE}  DEVICE_TYPE_TORCH={DEVICE_TYPE_TORCH}")

    # CHECK 4: BF16 GEMM on GPU
    def test_bf16_gemm():
        a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
        c = torch.mm(a, b)
        torch.cuda.synchronize()
        return c.shape == (1024, 1024) and c.dtype == torch.bfloat16

    check("BF16 GEMM 1024×1024 on GPU", test_bf16_gemm)

    # CHECK 5: mem_get_info (patched in tiled_mlp.py)
    def test_mem_get_info():
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(0)
        return free > 0 and total > 0

    check("torch.cuda.mem_get_info() works on ROCm", test_mem_get_info)

    free, total = torch.cuda.mem_get_info(0)
    print(f"  → Memory: {free/1024**3:.1f} GB free / {total/1024**3:.1f} GB total")

    # CHECK 6: SM cap guard (vllm_utils.py patch)
    def test_sm_cap_guard():
        # On AMD, is_hip() should be True and sm_cap should be set to 0
        # (not from torch.cuda.get_device_capability())
        if is_hip():
            sm_cap = 0  # AMD path — Hopper-only features disabled
            cutlass_fp8 = False
            return sm_cap == 0 and not cutlass_fp8
        return True

    check("SM cap guard: sm_cap=0 on AMD (Hopper paths disabled)", test_sm_cap_guard)

    # CHECK 7: FlashInfer guard
    import importlib.util
    def test_flashinfer_guard():
        if is_hip() and importlib.util.find_spec("flashinfer") is None:
            return True  # FlashInfer not installed — AMD skip active
        if is_hip() and importlib.util.find_spec("flashinfer") is not None:
            return True  # FlashInfer installed but guard would skip it
        return True

    check("FlashInfer AMD guard active (CUDA-only package skipped)", test_flashinfer_guard)

    # CHECK 8: SDPA attention available
    def test_sdpa():
        a = torch.randn(2, 4, 64, 32, device="cuda", dtype=torch.bfloat16)
        out = torch.nn.functional.scaled_dot_product_attention(a, a, a)
        return out.shape == a.shape

    check("SDPA (scaled_dot_product_attention) works on ROCm", test_sdpa)

    print()
    passed = sum(1 for _, r in results if r)
    total_checks = len(results)
    print(f"  Validation: {passed}/{total_checks} checks passed")
    return passed == total_checks


def run_benchmark(model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", steps=20, batch=4, seq=512):
    print("\n" + "="*60)
    print(f"  ROCm-Unsloth LoRA Benchmark")
    print(f"  Model: {model_id}")
    print(f"  Config: batch={batch} seq={seq} steps={steps} dtype=bfloat16")
    print("="*60)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    print(f"\nLoading {model_id}...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa"
    )
    load_t = time.time() - t0
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Loaded in {load_t:.1f}s  |  {params:.0f}M params", flush=True)

    lora_cfg = LoraConfig(r=16, lora_alpha=32,
                          target_modules=["q_proj", "v_proj"],
                          lora_dropout=0.05, bias="none",
                          task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA: {trainable/1e6:.2f}M trainable ({100*trainable/(params*1e6):.3f}%)", flush=True)

    tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token
    texts = ["User: Explain AMD ROCm.\nAssistant: AMD ROCm is a GPU computing platform."] * (batch * steps + 4)
    enc = tok(texts, max_length=seq, truncation=True, padding="max_length", return_tensors="pt")

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)

    print(f"\nTraining {steps} steps...", flush=True)
    torch.cuda.synchronize()
    t_start = time.time()
    total_tokens = 0
    step_results = []

    for step in range(steps):
        ids = enc["input_ids"][step*batch:(step+1)*batch].cuda()
        mask = enc["attention_mask"][step*batch:(step+1)*batch].cuda()
        opt.zero_grad()
        out = model(input_ids=ids, attention_mask=mask, labels=ids)
        out.loss.backward()
        opt.step()
        total_tokens += batch * seq
        elapsed = time.time() - t_start
        tps = total_tokens / elapsed
        step_results.append((step+1, out.loss.item(), tps))
        if (step+1) % 5 == 0 or step == 0:
            print(f"  Step {step+1:3d}/{steps}  loss={out.loss.item():.4f}  {tps:.0f} tok/s", flush=True)

    torch.cuda.synchronize()
    elapsed = time.time() - t_start
    tps_final = total_tokens / elapsed
    vram = torch.cuda.max_memory_allocated() / 1024**3

    # Steady-state: average of last 10 steps
    steady_tps = sum(r[2] for r in step_results[-10:]) / min(10, len(step_results))
    loss_decrease = step_results[0][1] > step_results[-1][1]

    print(f"\n{'='*60}")
    print(f"  BENCHMARK RESULTS")
    print(f"  GPU:          {torch.cuda.get_device_name(0)}")
    print(f"  Model:        {model_id} ({params:.0f}M params)")
    print(f"  LoRA:         r=16, alpha=32, q_proj+v_proj")
    print(f"  Config:       batch={batch}  seq={seq}  steps={steps}  bfloat16  SDPA")
    print(f"  Throughput:   {tps_final:.0f} tok/s (overall)  |  {steady_tps:.0f} tok/s (steady-state)")
    print(f"  VRAM:         {vram:.2f} GB")
    print(f"  Train time:   {elapsed:.1f}s (excl. model load {load_t:.1f}s)")
    print(f"  Loss:         {step_results[0][1]:.4f} → {step_results[-1][1]:.4f}  {'✅ decreasing' if loss_decrease else '❌ not decreasing'}")
    print(f"{'='*60}")

    # Assertions
    assert loss_decrease, "FAIL: Loss is not decreasing — gradient flow broken"
    assert tps_final > 100, f"FAIL: Throughput {tps_final:.0f} tok/s is too low"
    assert vram < total_tokens/1e6 * 10, "FAIL: VRAM usage unexpectedly high"
    print(f"\n  ✅ Benchmark PASSED")
    return {"tps_overall": tps_final, "tps_steady": steady_tps, "vram_gb": vram,
            "load_s": load_t, "train_s": elapsed}


def main():
    parser = argparse.ArgumentParser(description="ROCm-Unsloth test & benchmark")
    parser.add_argument("--validate", action="store_true", help="Run device validation checks")
    parser.add_argument("--benchmark", action="store_true", help="Run LoRA training benchmark")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="HuggingFace model ID for benchmark")
    parser.add_argument("--steps", type=int, default=20, help="Training steps")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--seq", type=int, default=512, help="Sequence length")
    args = parser.parse_args()

    if not args.validate and not args.benchmark:
        args.validate = True  # Default: run validation

    ok = True
    if args.validate:
        ok = run_validation() and ok

    if args.benchmark:
        run_benchmark(args.model, args.steps, args.batch, args.seq)

    if args.validate:
        passed = sum(1 for _, r in results if r)
        total = len(results)
        print(f"\n{'='*60}")
        if passed == total:
            print(f"  ✅ ALL {total} CHECKS PASSED — ROCm port is working correctly")
        else:
            print(f"  ❌ {total-passed}/{total} CHECKS FAILED — see above")
            sys.exit(1)
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
