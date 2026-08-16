#!/bin/bash
#SBATCH --job-name=tez-fov-eval
#SBATCH --partition=gpu2
#SBATCH --qos=gpu2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/fov-eval-%j.out
#SBATCH --error=logs/fov-eval-%j.err

# Evaluate the trained normal / foveated@{cpd} / center@{cpd} checkpoints across
# folds and aggregate into results/foveation_mit1003/table.{md,json} + figure.
# Run AFTER the foveation_train_fold jobs have produced all checkpoints under
# TEZ_CKPT_ROOT. Eval is forward-only.
#
# Env knobs:
#   TEZ_FOLDS      : space-separated fold list. Default "0 1 2 3 4 5 6 7 8 9".
#   TEZ_SPLIT      : held-out split to report. Default "test".
#   TEZ_CPDS       : foveal_cpd sweep. Default "40 20 10" (the protocol).
#   TEZ_CENTER_CPDS: foveal_cpd values to also evaluate as the fixed-centre
#                    ablation. Default "40 20 10" — the protocol pairs every
#                    centre arm with its gaze-contingent counterpart. Set to an
#                    explicitly empty string ("") to skip the centre arms; the
#                    default applies only when the variable is unset.
#   TEZ_DUMP_PER_FIXATION : 1 (default) also writes per-fixation records to
#                    <out>/per_fixation/, one npz per (arm, fold) — required for
#                    the per-fixation analyses, and unrecoverable from the
#                    aggregates afterwards. 0 disables.
#   TEZ_FAST_FORWARD : 1 (default) evaluates through the fast DG3 forward
#                    (see fast_dg3). 0 uses the vendor code path — the control
#                    for the shared-image numeric asymmetry between arms.
#   TEZ_PPD        : pixels/degree. Default 35.
#   TEZ_DATASET_VARIANT : "plain" (default) or "initial"
#                    (MIT1003_initial_fix_consistent: every free fixation scored,
#                    centre conditions the early histories — the DG3 paper's
#                    protocol). The initial variant must already exist under
#                    $TEZ_DATA_ROOT (rsynced from the laptop Octave build).
#   TEZ_CKPT_ROOT  : per-(arm,fold) checkpoints. Default results/foveation_mit1003/ckpts.
#   TEZ_EPOCH      : checkpoint epoch to eval. Default: last epoch present.
#   TEZ_OUT        : output dir. Default results/foveation_mit1003.
#   TEZ_WORKDIR    : tree to evaluate from. Default /work/<project>/<user>/Tez. Point
#                    it at a copy to run an export from a known-good checkout
#                    without disturbing the main tree — same knob as pytest.sbatch.
#                    Pair it with an absolute TEZ_CKPT_ROOT,
#                    since checkpoints are gitignored and live only in the main tree.
#   TEZ_VENV       : interpreter. Default $TEZ_WORKDIR/.venv.

set -euo pipefail

module purge
module load nvidia/cuda/12.3.0

WORKDIR=${TEZ_WORKDIR:-"/work/<project>/<user>/Tez"}
VENV=${TEZ_VENV:-$WORKDIR/.venv}
export TORCH_HOME="/work/<project>/<user>/torch-cache"
export TEZ_DATA_ROOT=$WORKDIR/data/mit1003
# Evaluate the source in WORKDIR, not whatever the venv has installed — the venv
# holds tez_deepgaze as an editable install pointing at the main checkout.
export PYTHONPATH=$WORKDIR/src${PYTHONPATH:+:$PYTHONPATH}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# MIT1003 has 142 distinct image shapes, which fragments the caching
# allocator: at micro-batch 32 reserved memory ran 22 GB above allocated
# (82.2 GB of an 85.9 GB card) purely as unusable gaps. expandable_segments
# cuts that overhead to ~3 GB at identical speed and identical loss.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TEZ_REQUIRE_CUDA=1

cd "$WORKDIR"
mkdir -p logs results

"$VENV/bin/python" -c "import torch; assert torch.cuda.is_available(), \
    'no CUDA device bound — need NVIDIA gpu2'; \
    print('preflight OK: CUDA device', torch.cuda.get_device_name(0))"

TEZ_FOLDS=${TEZ_FOLDS:-"0 1 2 3 4 5 6 7 8 9"}
TEZ_SPLIT=${TEZ_SPLIT:-test}
TEZ_CPDS=${TEZ_CPDS:-"40 20 10"}
TEZ_PPD=${TEZ_PPD:-35}
TEZ_DATASET_VARIANT=${TEZ_DATASET_VARIANT:-plain}
if [[ "$TEZ_DATASET_VARIANT" != "plain" && "$TEZ_DATASET_VARIANT" != "initial" ]]; then
    echo "ERROR: TEZ_DATASET_VARIANT must be plain|initial, got '$TEZ_DATASET_VARIANT'" >&2
    exit 1
fi
# A non-plain variant reroutes the *defaults*: a bare variant submission would
# otherwise evaluate plain-trained checkpoints under the other protocol and
# overwrite the plain-protocol tables in place. An explicit
# TEZ_CKPT_ROOT / TEZ_OUT still wins (that is how a deliberate cross-protocol
# robustness eval is requested).
if [[ "$TEZ_DATASET_VARIANT" != "plain" ]]; then
    TEZ_CKPT_ROOT=${TEZ_CKPT_ROOT:-results/foveation_mit1003/ckpts_${TEZ_DATASET_VARIANT}}
    TEZ_OUT=${TEZ_OUT:-results/foveation_mit1003_${TEZ_DATASET_VARIANT}}
else
    TEZ_CKPT_ROOT=${TEZ_CKPT_ROOT:-results/foveation_mit1003/ckpts}
    TEZ_OUT=${TEZ_OUT:-results/foveation_mit1003}
fi
# ${VAR-default}, not ${VAR:-default}: an explicitly empty TEZ_CENTER_CPDS must
# disable the centre arms, while an unset one gets the protocol default.
TEZ_CENTER_CPDS=${TEZ_CENTER_CPDS-"40 20 10"}
TEZ_DUMP_PER_FIXATION=${TEZ_DUMP_PER_FIXATION:-1}
TEZ_FAST_FORWARD=${TEZ_FAST_FORWARD:-1}
EPOCH_FLAG=""
if [[ -n "${TEZ_EPOCH:-}" ]]; then EPOCH_FLAG="--epoch ${TEZ_EPOCH}"; fi
CENTER_FLAG=""
if [[ -n "$TEZ_CENTER_CPDS" ]]; then CENTER_FLAG="--center-cpds ${TEZ_CENTER_CPDS}"; fi
DUMP_FLAG=""
if [[ "$TEZ_DUMP_PER_FIXATION" == "1" ]]; then DUMP_FLAG="--dump-per-fixation"; fi
FF_FLAG=""
if [[ "$TEZ_FAST_FORWARD" == "0" ]]; then FF_FLAG="--no-fast-forward"; fi

echo "folds=${TEZ_FOLDS} split=${TEZ_SPLIT} cpds=${TEZ_CPDS} \
center_cpds=${TEZ_CENTER_CPDS:-none} ckpt_root=${TEZ_CKPT_ROOT} \
dump_per_fixation=${TEZ_DUMP_PER_FIXATION} fast_forward=${TEZ_FAST_FORWARD}"

"$VENV/bin/python" -u scripts/foveation_sweep_table.py \
    --device cuda \
    --folds ${TEZ_FOLDS} \
    --split "${TEZ_SPLIT}" \
    --cpds ${TEZ_CPDS} \
    ${CENTER_FLAG} \
    ${DUMP_FLAG} \
    ${FF_FLAG} \
    --ppd "${TEZ_PPD}" \
    --dataset-variant "${TEZ_DATASET_VARIANT}" \
    --ckpt-root "${TEZ_CKPT_ROOT}" \
    ${EPOCH_FLAG} \
    --out "${TEZ_OUT}"
