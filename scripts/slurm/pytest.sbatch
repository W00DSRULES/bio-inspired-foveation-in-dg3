#!/bin/bash
#SBATCH --job-name=tez-pytest
#SBATCH --partition=gpu2
#SBATCH --qos=gpu2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/pytest-%j.out
#SBATCH --error=logs/pytest-%j.err

# Run the test suite on a real NVIDIA GPU.
#
# The suite is not run on the laptop: several tests load the whole MIT1003
# corpus and the pretrained DG3 weights. gpu2 is also the platform the reported numbers are produced on, so it is the
# one whose arithmetic matters.
#
# Env knobs:
#   TEZ_WORKDIR     : tree to test. Default /work/dldevel/itez/Tez. Point it at a
#                     copy to test code without disturbing the pinned checkout.
#   TEZ_VENV        : interpreter to use. Default $TEZ_WORKDIR/.venv, which may be
#                     a symlink to the main tree's environment.
#   TEZ_PYTEST_ARGS : pytest arguments. Default "tests -v".
#
#   TEZ_WORKDIR=/work/dldevel/itez/Tez-tests sbatch scripts/slurm/pytest.sbatch

set -euo pipefail

module purge
module load nvidia/cuda/12.3.0

WORKDIR=${TEZ_WORKDIR:-/work/dldevel/itez/Tez}
VENV=${TEZ_VENV:-$WORKDIR/.venv}
export TORCH_HOME=/work/dldevel/itez/torch-cache
export TEZ_DATA_ROOT=$WORKDIR/data/mit1003
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Test the source in WORKDIR, not whatever the venv has installed. The venv holds
# tez_deepgaze as an editable install pointing at the main checkout, so without
# this a copy would be tested against the main tree's code while appearing to
# test itself. The preflight below asserts the override actually took.
export PYTHONPATH=$WORKDIR/src${PYTHONPATH:+:$PYTHONPATH}

cd "$WORKDIR"
mkdir -p logs

"$VENV/bin/python" - <<'PY'
import os
import pathlib

import torch

import tez_deepgaze

assert torch.cuda.is_available(), "no CUDA device bound — need NVIDIA gpu2"
src = pathlib.Path(tez_deepgaze.__file__).resolve()
want = pathlib.Path(os.environ["PYTHONPATH"].split(":")[0]).resolve()
assert want in src.parents, f"importing {src}, not the tree under test ({want})"
print("preflight OK:", torch.cuda.get_device_name(0), "|", src)
PY

"$VENV/bin/python" -m pytest ${TEZ_PYTEST_ARGS:-tests -v}
