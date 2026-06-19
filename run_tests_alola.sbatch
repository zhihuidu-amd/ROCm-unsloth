#!/bin/bash
#SBATCH --job-name=rocm-unsloth-test
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --nodelist=quanta-ccs-aus-e15-19
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/users/zhdu/Project/ROCm-unsloth/test-%j.out
#SBATCH --error=/scratch/users/zhdu/Project/ROCm-unsloth/test-%j.err

# ROCm-Unsloth Validation + Benchmark on Alola cluster (MI325X gfx942)
# Usage: sbatch run_tests_alola.sbatch

export PATH=$HOME/.local/bin:$PATH
export HF_HOME=/scratch/users/zhdu/.cache/huggingface
export HIP_VISIBLE_DEVICES=0
export UNSLOTH_IS_PRESENT=1
export PYTHONUNBUFFERED=1

PROJ="/scratch/users/zhdu/Project/ROCm-unsloth"
REPO="$PROJ/repo"
STAGE="$PROJ/stage"

echo "======================================"
echo " ROCm-Unsloth Test Suite"
echo " Node: $(hostname) | Job: $SLURM_JOB_ID | $(date)"
echo "======================================"

# Install PyTorch ROCm if needed
pip install --upgrade pip -q 2>&1 | tail -1
pip install -q torch==2.10.0 --index-url https://download.pytorch.org/whl/rocm7.0 2>&1 | tail -2
pip install -q transformers peft accelerate 2>&1 | tail -2

# Install unsloth_zoo from pre-staged files
SITE=$(python3 -c 'import site; print(site.getusersitepackages())')
[ -d "$SITE/unsloth_zoo" ] || cp -r $STAGE/unsloth_zoo $SITE/
[ -f "$SITE/unsloth/__init__.py" ] || (mkdir -p $SITE/unsloth && \
  printf '__version__="0.0.1-rocm"\nimport os\nos.environ["UNSLOTH_IS_PRESENT"]="1"\n' \
  > $SITE/unsloth/__init__.py)

# Run validation (fast, no model download)
echo ""
echo "=== Running Validation Checks ==="
UNSLOTH_IS_PRESENT=1 python3 $REPO/test_rocm.py --validate

# Run benchmark (downloads TinyLlama ~2.3GB if not cached)
echo ""
echo "=== Running LoRA Benchmark ==="
UNSLOTH_IS_PRESENT=1 python3 $REPO/test_rocm.py --benchmark --steps 20

echo ""
echo "Done — $(date)"
