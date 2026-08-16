#!/bin/bash
#SBATCH --job-name=tez-baseline
#SBATCH --partition=gpu2
#SBATCH --qos=gpu2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=logs/baseline-%j.out
#SBATCH --error=logs/baseline-%j.err

# Reproduce the full-corpus DG3 baseline on Goethe-NHR. Writes
# results/baselines/baseline.json (schema v2) with reproducibility SHAs
# and IG vs Kümmerer 2022. See scripts/slurm/README.md for the prefetch
# sequence.

set -euo pipefail

# Modules. nvidia/cuda/12.3.0 is the only cuda module currently on this
# cluster; pin the exact version so a future module reshuffle does not
# silently change the runtime CUDA.
module purge
module load nvidia/cuda/12.3.0

# Paths. WORKDIR is the repo on /work; HOME is too small for the venv +
# torch-cache + dataset.
WORKDIR=/work/dldevel/itez/Tez
export TORCH_HOME=/work/dldevel/itez/torch-cache
export TEZ_DATA_ROOT=$WORKDIR/data/mit1003

# Determinism: required by torch.use_deterministic_algorithms on CUDA.
# Without it, cublas-backed ops fall back to non-deterministic kernels.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# MIT1003 has 142 distinct image shapes, which fragments the caching
# allocator: at micro-batch 32 reserved memory ran 22 GB above allocated
# (82.2 GB of an 85.9 GB card) purely as unusable gaps. expandable_segments
# cuts that overhead to ~3 GB at identical speed and identical loss.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$WORKDIR"
mkdir -p logs results

.venv/bin/python -m tez_deepgaze.baseline \
    --device cuda \
    --out results/baselines/baseline.json
