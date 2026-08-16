#!/bin/bash
#SBATCH --job-name=tez-fov-strat
#SBATCH --partition=gpu2
#SBATCH --qos=gpu2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/fov-strat-%j.out
#SBATCH --error=logs/fov-strat-%j.err

# Stratified two-arm analysis: where the likelihood cost falls as a function of
# saccade amplitude, fixation index, entropy and calibration. Forward-only —
# reads the trained checkpoints under TEZ_CKPT_ROOT, trains nothing. One job per
# contrast; each re-evaluates its reference arm, so the jobs are independent and
# can run concurrently (gpu2 allows 4).
#
# --dump-per-fixation is passed unconditionally, so every run keeps `amp_mass` —
# the model's predicted probability mass in annuli around the current gaze
# point, i.e. its implied saccade-amplitude distribution. That quantity
# integrates the density over regions, so unlike every other diagnostic it
# cannot be reconstructed from a scalar afterwards.
#
# Env knobs:
#   TEZ_CPD        : foveal_cpd of the foveated/centre arms. Required.
#   TEZ_FOLDS      : space-separated fold list. Default "0 1 2 3 4 5 6 7 8 9".
#   TEZ_SPLIT      : split to evaluate. Default "val" — test is reserved for a
#                    single final evaluation and this is a mechanism check.
#   TEZ_PPD        : pixels/degree. Default 35.
#   TEZ_ARM_REF    : reference arm, normal|foveated|center. Default normal.
#   TEZ_ARM_TEST   : test arm, normal|foveated|center. Default foveated.
#                    Deltas are test - ref. TEZ_ARM_REF=foveated TEZ_ARM_TEST=center
#                    is the contrast that isolates gaze-contingency: identical blur
#                    profile, fovea pinned to the image centre instead of tracking.
#   TEZ_SEED       : passed through so the scanpath subsample is reproducible.
#   TEZ_EPOCH      : checkpoint epoch. Default 7. NOT the resolver default (last
#                    epoch present).
#   TEZ_CKPT_ROOT  : per-(arm,fold) checkpoints. Default results/foveation_mit1003/ckpts.
#   TEZ_OUT        : output dir. Unset by default — the script derives a split- and
#                    pair-keyed directory so two runs cannot overwrite each other.

set -euo pipefail

module purge
module load nvidia/cuda/12.3.0

WORKDIR=/work/dldevel/itez/Tez
export TORCH_HOME=/work/dldevel/itez/torch-cache
export TEZ_DATA_ROOT=$WORKDIR/data/mit1003
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TEZ_REQUIRE_CUDA=1

cd "$WORKDIR"
mkdir -p logs results

.venv/bin/python -c "import torch; assert torch.cuda.is_available(), \
    'no CUDA device bound — need NVIDIA gpu2'; \
    print('preflight OK: CUDA device', torch.cuda.get_device_name(0))"

# No apostrophe in this message: bash treats a single quote inside ${var:?word}
# as a quote opener even within double quotes, and the script fails to parse.
: "${TEZ_CPD:?set TEZ_CPD to the foveal_cpd of the foveated arm, e.g. 40}"
TEZ_FOLDS=${TEZ_FOLDS:-"0 1 2 3 4 5 6 7 8 9"}
TEZ_SPLIT=${TEZ_SPLIT:-val}
TEZ_PPD=${TEZ_PPD:-35}
TEZ_CKPT_ROOT=${TEZ_CKPT_ROOT:-results/foveation_mit1003/ckpts}
TEZ_ARM_REF=${TEZ_ARM_REF:-normal}
TEZ_ARM_TEST=${TEZ_ARM_TEST:-foveated}
TEZ_SEED=${TEZ_SEED:-42}
# Default to the reporting epoch, not the checkpoint resolver's "last epoch
# present". Leaving it unset would analyse different checkpoints from the ones
# every number in the results chapter comes from.
TEZ_EPOCH=${TEZ_EPOCH:-7}
# Left to the script by default: it derives a split- and pair-keyed directory,
# which is what stops a val run and a test run (or two different arm pairs at
# one cpd) from silently overwriting each other.
TEZ_OUT_FLAG=""
[[ -n "${TEZ_OUT:-}" ]] && TEZ_OUT_FLAG="--out ${TEZ_OUT}"

echo "cpd=${TEZ_CPD} folds=${TEZ_FOLDS} split=${TEZ_SPLIT} ckpt_root=${TEZ_CKPT_ROOT}"
echo "contrast=${TEZ_ARM_TEST}-${TEZ_ARM_REF} seed=${TEZ_SEED}"

.venv/bin/python -u scripts/foveation_stratified.py \
    --folds ${TEZ_FOLDS} \
    --split "${TEZ_SPLIT}" \
    --cpd "${TEZ_CPD}" \
    --ppd "${TEZ_PPD}" \
    --arm-ref "${TEZ_ARM_REF}" \
    --arm-test "${TEZ_ARM_TEST}" \
    --seed "${TEZ_SEED}" \
    --epoch "${TEZ_EPOCH}" \
    --dump-per-fixation \
    --dataset-variant "${TEZ_DATASET_VARIANT:-plain}" \
    --ckpt-root "${TEZ_CKPT_ROOT}" \
    ${TEZ_OUT_FLAG}
