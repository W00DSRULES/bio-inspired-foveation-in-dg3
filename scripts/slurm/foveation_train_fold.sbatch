#!/bin/bash
#SBATCH --job-name=tez-fov-train
#SBATCH --partition=gpu2
#SBATCH --qos=gpu2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/fov-train-%j.out
#SBATCH --error=logs/fov-train-%j.err

# Train ONE (arm, fold, foveal_cpd) of the normal-vs-foveated
# comparison: DeepGaze III readout heads fine-tuned on MIT1003, backbone frozen,
# input optionally foveated gaze-contingently. Calls tez_deepgaze.foveated_train.
#
# Env knobs (set before sbatch, or via --export):
#   TEZ_ARM          : "normal" (sharp input), "foveated" (gaze-contingent) or
#                      "center" (foveated with the fovea pinned to the image
#                      centre — the ablation that isolates gaze-contingency, and
#                      what prior pixel-level foveated networks do). Required.
#   TEZ_FOLD         : CV fold 0..9. Required.
#   TEZ_FOVEAL_CPD   : foveal cutoff cyc/deg for the foveated arms. Default 40.
#                      Sweep uses {40, 20, 10}; ignored for the normal arm.
#   TEZ_PPD          : pixels/degree. Default 35 (MIT1003).
#   TEZ_EPOCHS       : head fine-tuning epochs. Default 12 — see the note below.
#   TEZ_START_EPOCH  : first epoch to run. Default: auto — continue after the
#                      newest COMPLETE bundle (weights.pt, metrics.json AND
#                      optimizer.pt) under TEZ_OUT, or start at 1 if there is
#                      none. Set explicitly to override (see below).
#   TEZ_FAST_FORWARD : 1 (default) trains with the fast DG3 forward (see
#                      fast_dg3). 0 uses the vendor code path — the control for
#                      the shared-image numeric asymmetry between arms.
#   TEZ_LR           : base Adam LR for the heads. Default 3e-4, chosen by a
#                      fold-0 LR probe. Decayed 10x at epochs 5, 8 and 11 — see
#                      the note below.
#   TEZ_MICRO_BATCH  : per-image gradient-accumulation chunk. Default 36 (CUDA),
#                      per 768x1024; scaled by image area in foveated_train.
#   TEZ_DATASET_VARIANT : "plain" (default) trains from the second free fixation.
#                      "initial" trains on
#                      MIT1003_initial_fix_consistent — every free fixation is a
#                      target, the forced central fixation conditions the early
#                      histories (the DG3 paper's protocol). Any non-plain value
#                      suffixes the checkpoint root (ckpts_<variant>) so the two
#                      protocols cannot share a tree, and the run refuses to
#                      write where bundles of another variant already live.
#   TEZ_SEED         : Default 42. Any other value writes to a separate checkpoint
#                      root (see TEZ_OUT), so a multi-seed probe cannot overwrite
#                      itself; the run also refuses to write into a bundle made
#                      with a different seed.
#   TEZ_OUT          : checkpoint dir. Default results/foveation_mit1003/ckpts/<tag>/fold<fold>,
#                      or ckpts_seed<seed>/<tag>/fold<fold> when TEZ_SEED is not 42,
#                      where <tag> is "normal", "fov_cpd<cpd>" or
#                      "fov_cpd<cpd>_center" — the layout
#                      scripts/foveation_sweep_table.py reads. The centre arm gets
#                      its own tag: the path is keyed only by (tag, fold, epoch),
#                      so sharing one with the gaze-contingent arm at the same cpd
#                      would have it overwrite that arm's checkpoints. Evaluate a
#                      seed root by passing it as --ckpt-root / TEZ_CKPT_ROOT;
#                      nothing else changes.
#
# Epochs and the decaying rate
# ----------------------------
# The rate is stepped down 10x at epochs 5, 8 and 11 (3e-4 -> 3e-5 -> 3e-6 ->
# 3e-7; see lr_for_epoch in foveated_train.py). The milestones are absolute epoch
# numbers, so a shorter TEZ_EPOCHS fires only the decays it reaches (at 7 epochs,
# the epoch-5 one). At a flat rate the arms do not plateau within a few epochs;
# the decaying rate lets the last epochs settle. One epoch runs ~30 min on an
# A100 with the fast forward, so a 10-fold arm at 12 epochs costs ~60 GPU-h.
#
# foveated_train.py supports resume (--init-weights / --start-epoch, wired up via
# TEZ_START_EPOCH below), so raising this continues an existing arm instead of
# rerunning it — but only bit-exactly when the previous run wrote optimizer.pt.
# Resuming a bundle without optimizer.pt is a warm restart (Adam moments zeroed),
# valid for a paired comparison only if every arm is resumed identically. The
# learning rate is safe across a resume either way: it is computed from the epoch
# number, not from scheduler state.
#
# Defaults are aligned deliberately: this file, foveation_pump.sh (EPOCHS=12)
# and any new submission must agree, or a new arm is
# silently incomparable with the existing checkpoints. The CLI default in
# foveated_train.py stays at 1 because that is a smoke-test entry point, not the
# experiment protocol.

set -euo pipefail

module purge
module load nvidia/cuda/12.3.0

WORKDIR=/work/dldevel/itez/Tez
export TORCH_HOME=/work/dldevel/itez/torch-cache
export TEZ_DATA_ROOT=$WORKDIR/data/mit1003
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# MIT1003 has 142 distinct image shapes, which fragments the caching
# allocator: at micro-batch 32 reserved memory ran 22 GB above allocated
# (82.2 GB of an 85.9 GB card) purely as unusable gaps. expandable_segments
# cuts that overhead to ~3 GB at identical speed and identical loss.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TEZ_REQUIRE_CUDA=1

cd "$WORKDIR"
mkdir -p logs results

.venv/bin/python -c "import torch; assert torch.cuda.is_available(), \
    'no CUDA device bound — need NVIDIA gpu2'; \
    print('preflight OK: CUDA device', torch.cuda.get_device_name(0))"

TEZ_ARM=${TEZ_ARM:?TEZ_ARM is required (normal|foveated|center)}
TEZ_FOLD=${TEZ_FOLD:?TEZ_FOLD is required (0..9)}
TEZ_FOVEAL_CPD=${TEZ_FOVEAL_CPD:-40}
TEZ_PPD=${TEZ_PPD:-35}
TEZ_EPOCHS=${TEZ_EPOCHS:-12}  # must match foveation_pump.sh EPOCHS — see header
TEZ_LR=${TEZ_LR:-3e-4}
TEZ_MICRO_BATCH=${TEZ_MICRO_BATCH:-36}   # per 768x1024; scaled by image area in foveated_train
TEZ_SEED=${TEZ_SEED:-42}
TEZ_DATASET_VARIANT=${TEZ_DATASET_VARIANT:-plain}
if [[ "$TEZ_DATASET_VARIANT" != "plain" && "$TEZ_DATASET_VARIANT" != "initial" ]]; then
    echo "ERROR: TEZ_DATASET_VARIANT must be plain|initial, got '$TEZ_DATASET_VARIANT'" >&2
    exit 1
fi
TEZ_FAST_FORWARD=${TEZ_FAST_FORWARD:-1}
FF_FLAG=""
if [[ "$TEZ_FAST_FORWARD" == "0" ]]; then FF_FLAG="--no-fast-forward"; fi
# Optional smoke/debug knobs: train/eval on only N stimuli. Empty = full fold.
TEZ_SUBSAMPLE_TRAIN=${TEZ_SUBSAMPLE_TRAIN:-}
TEZ_SUBSAMPLE_VAL=${TEZ_SUBSAMPLE_VAL:-}
SUB_FLAGS=""
[[ -n "$TEZ_SUBSAMPLE_TRAIN" ]] && SUB_FLAGS="$SUB_FLAGS --subsample-train $TEZ_SUBSAMPLE_TRAIN"
[[ -n "$TEZ_SUBSAMPLE_VAL" ]] && SUB_FLAGS="$SUB_FLAGS --subsample-val $TEZ_SUBSAMPLE_VAL"

case "$TEZ_ARM" in
    normal)   FOV_FLAG="--no-foveate"; TAG="normal" ;;
    foveated) FOV_FLAG="--foveate";    TAG="fov_cpd${TEZ_FOVEAL_CPD%.*}" ;;
    center)   FOV_FLAG="--foveate --center-fovea"
              TAG="fov_cpd${TEZ_FOVEAL_CPD%.*}_center" ;;
    *) echo "ERROR: TEZ_ARM must be normal|foveated|center, got '$TEZ_ARM'" >&2; exit 1 ;;
esac

# Subsampled runs are smoke tests and must not write where the real checkpoints
# live — the path is keyed only by (arm, fold, epoch), so a smoke run of the same
# arm and fold silently overwrites a fully trained epoch. See the pair script.
CKPT_ROOT=results/foveation_mit1003/ckpts
if [[ -n "$TEZ_SUBSAMPLE_TRAIN" || -n "$TEZ_SUBSAMPLE_VAL" ]]; then
    CKPT_ROOT=results/foveation_mit1003/ckpts_smoke
    echo "NOTE: subsampled run — checkpoints go to $CKPT_ROOT, not the real tree"
fi
# Seed probe: a non-default seed gets its own checkpoint root. The path is keyed
# only by (arm, fold, epoch) — the same reason a smoke run can clobber a trained
# epoch above — so four seeds of one (arm, fold) would each overwrite the last and
# the seed-to-seed spread would be unrecoverable without retraining. Varying the
# ROOT rather than the fold directory keeps every eval entry point working
# unchanged, since they all take --ckpt-root / TEZ_CKPT_ROOT and append
# <tag>/fold<k>. Seed 42 keeps the existing path so the trained arms stay put.
if [[ "$TEZ_SEED" != "42" ]]; then
    CKPT_ROOT="${CKPT_ROOT}_seed${TEZ_SEED}"
    echo "NOTE: non-default seed — checkpoints go to $CKPT_ROOT"
fi
# A non-plain dataset variant gets its own root for the same reason as a
# non-default seed: the path is keyed only by (tag, fold, epoch), so a
# variant retrain into the default root would OVERWRITE the plain-protocol
# checkpoints.
if [[ "$TEZ_DATASET_VARIANT" != "plain" ]]; then
    CKPT_ROOT="${CKPT_ROOT}_${TEZ_DATASET_VARIANT}"
    echo "NOTE: dataset variant ${TEZ_DATASET_VARIANT} — checkpoints go to $CKPT_ROOT"
fi
TEZ_OUT=${TEZ_OUT:-${CKPT_ROOT}/${TAG}/fold${TEZ_FOLD}}

# Backstop for the case the seed root cannot catch: an explicit TEZ_OUT, or a
# default path reused after the convention changes. Refuse rather than silently
# destroy another seed's result, which is only detectable long afterwards.
EXISTING_BUNDLE=$(ls -d "${TEZ_OUT}"/epoch_* 2>/dev/null | sort | head -1 || true)
if [[ -n "$EXISTING_BUNDLE" && -f "${EXISTING_BUNDLE}/metrics.json" ]]; then
    PREV_SEED=$(.venv/bin/python -c \
        "import json,sys; print(json.load(open(sys.argv[1])).get('reproducibility',{}).get('seed',''))" \
        "${EXISTING_BUNDLE}/metrics.json" 2>/dev/null || echo "")
    if [[ -n "$PREV_SEED" && "$PREV_SEED" != "$TEZ_SEED" ]]; then
        echo "ERROR: ${TEZ_OUT} already holds a bundle trained with seed ${PREV_SEED}," >&2
        echo "       but this run uses seed ${TEZ_SEED}. Writing here would destroy the" >&2
        echo "       other seed's result. Point TEZ_OUT somewhere else to proceed." >&2
        exit 1
    fi
    # Same backstop for the dataset variant. A bundle that does not record
    # dataset_variant is plain-variant, so absent defaults to "plain".
    PREV_VARIANT=$(.venv/bin/python -c \
        "import json,sys; print(json.load(open(sys.argv[1])).get('dataset_variant','plain'))" \
        "${EXISTING_BUNDLE}/metrics.json" 2>/dev/null || echo "")
    if [[ -n "$PREV_VARIANT" && "$PREV_VARIANT" != "$TEZ_DATASET_VARIANT" ]]; then
        echo "ERROR: ${TEZ_OUT} already holds a bundle trained on dataset variant" >&2
        echo "       '${PREV_VARIANT}', but this run uses '${TEZ_DATASET_VARIANT}'. Writing here" >&2
        echo "       would destroy the other protocol's result. Point TEZ_OUT elsewhere." >&2
        exit 1
    fi
fi

# Continuation: an explicit TEZ_START_EPOCH=N+1 carries an existing arm on from
# epoch N. Left unset, the script auto-resumes after the newest COMPLETE bundle —
# weights.pt, metrics.json AND optimizer.pt, because without the Adam moments a
# resume is a WARM RESTART, not a continuation. Bundles without optimizer.pt are
# deliberately NOT resumed: the fold retrains from epoch 1 and overwrites them.
# The weights always come from the arm's own bundle, so a continuation never
# crosses arms.
RESUME_FLAGS=""
TEZ_START_EPOCH=${TEZ_START_EPOCH:-}
if [[ -z "$TEZ_START_EPOCH" ]]; then
    TEZ_START_EPOCH=1
    LAST_COMPLETE=""
    for d in $(ls -d "${TEZ_OUT}"/epoch_* 2>/dev/null | sort || true); do
        [[ -f "$d/weights.pt" && -f "$d/metrics.json" && -f "$d/optimizer.pt" ]] \
            && LAST_COMPLETE="$d"
    done
    # Only continue a run of the SAME protocol: a bundle whose run targeted a
    # different --epochs (a 1-epoch smoke, a shorter schedule) must be
    # retrained, not extended — the LR schedule and stopping point differ.
    if [[ -n "$LAST_COMPLETE" ]]; then
        PREV_PROTO=$(.venv/bin/python -c \
            "import json,sys; print(json.load(open(sys.argv[1])).get('reproducibility',{}).get('args',{}).get('epochs',''))" \
            "${LAST_COMPLETE}/metrics.json" 2>/dev/null || echo "")
        if [[ "$PREV_PROTO" != "$TEZ_EPOCHS" ]]; then
            # A COMPLETE run of another protocol (its last bundle reached its
            # own epochs target) is a finished result; a pump relaunch that
            # forgot TEZ_EPOCHS must not silently destroy it. Only an
            # incomplete other-protocol run (a smoke, an aborted attempt) is
            # overwritten as before.
            LAST_DONE=$((10#$(basename "$LAST_COMPLETE" | cut -d_ -f2)))
            if [[ -n "$PREV_PROTO" && "$LAST_DONE" -ge "$PREV_PROTO" \
                  && "${TEZ_FORCE_RETRAIN:-0}" != "1" ]]; then
                echo "ERROR: ${TEZ_OUT} holds a COMPLETE epochs=${PREV_PROTO} run" >&2
                echo "       (last bundle ${LAST_COMPLETE}); this run targets epochs=${TEZ_EPOCHS}" >&2
                echo "       and retraining from epoch 1 would destroy it. Set TEZ_FORCE_RETRAIN=1" >&2
                echo "       to overwrite deliberately, or point TEZ_OUT elsewhere." >&2
                exit 1
            fi
            echo "NOTE: ${LAST_COMPLETE} comes from a run that targeted epochs=${PREV_PROTO:-unknown},"
            echo "      not ${TEZ_EPOCHS} — different protocol, not resuming; retraining"
            echo "      from epoch 1 will overwrite it."
            LAST_COMPLETE=""
        fi
    fi
    if [[ -n "$LAST_COMPLETE" ]]; then
        LAST_EPOCH=$((10#$(basename "$LAST_COMPLETE" | cut -d_ -f2)))
        if [[ "$LAST_EPOCH" -ge "$TEZ_EPOCHS" ]]; then
            echo "nothing to do: ${LAST_COMPLETE} already reaches TEZ_EPOCHS=${TEZ_EPOCHS}"
            exit 0
        fi
        TEZ_START_EPOCH=$((LAST_EPOCH + 1))
        echo "auto-resume: ${LAST_COMPLETE} is complete -> starting at epoch ${TEZ_START_EPOCH}"
    elif [[ -n "$EXISTING_BUNDLE" ]]; then
        echo "NOTE: bundle(s) under ${TEZ_OUT} lack optimizer.pt — cannot continue"
        echo "      them; retraining from epoch 1 will overwrite them."
    fi
fi
if [[ "$TEZ_START_EPOCH" -gt 1 ]]; then
    PREV=$(printf '%s/epoch_%03d/weights.pt' "$TEZ_OUT" $((TEZ_START_EPOCH - 1)))
    [[ -f "$PREV" ]] || { echo "ERROR: cannot resume, missing $PREV" >&2; exit 1; }
    RESUME_FLAGS="--start-epoch ${TEZ_START_EPOCH} --init-weights ${PREV}"
    echo "resuming from ${PREV} -> epochs ${TEZ_START_EPOCH}..${TEZ_EPOCHS}"
fi

echo "arm=${TEZ_ARM} fold=${TEZ_FOLD} foveal_cpd=${TEZ_FOVEAL_CPD} ppd=${TEZ_PPD}"
echo "epochs=${TEZ_START_EPOCH}..${TEZ_EPOCHS} lr=${TEZ_LR} micro_batch=${TEZ_MICRO_BATCH}"
echo "tag=${TAG} dataset_variant=${TEZ_DATASET_VARIANT} out=${TEZ_OUT}"

.venv/bin/python -u -m tez_deepgaze.foveated_train \
    --device cuda \
    --fold "${TEZ_FOLD}" \
    --epochs "${TEZ_EPOCHS}" \
    ${RESUME_FLAGS} \
    ${FOV_FLAG} \
    ${FF_FLAG} \
    --ppd "${TEZ_PPD}" \
    --foveal-cpd "${TEZ_FOVEAL_CPD}" \
    --lr "${TEZ_LR}" \
    --micro-batch "${TEZ_MICRO_BATCH}" \
    ${SUB_FLAGS} \
    --dataset-variant "${TEZ_DATASET_VARIANT}" \
    --seed "${TEZ_SEED}" \
    --out "${TEZ_OUT}"
