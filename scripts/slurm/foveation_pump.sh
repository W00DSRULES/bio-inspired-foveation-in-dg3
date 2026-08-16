#!/bin/bash
# Submission pump for the Goethe-NHR gpu2 QOS (12 submitted / 4 running max).
#
# Submits the normal-vs-foveated matrix in waves under the QOS limit, skipping
# any spec whose final checkpoint already exists, then submits the eval job once
# all training is done. Restart-safe at epoch granularity: a resubmitted fold is
# auto-resumed by foveation_train_fold.sbatch from its newest complete epoch
# bundle, not retrained from scratch.
#
# Matrix (70 training jobs):
#   normal                × folds 0..9   (sharp-input control)
#   foveated@{40,20,10}   × folds 0..9   (gaze-contingent, dose-response)
#   center@{40,20,10}     × folds 0..9   (fixed-centre ablation)
#
# Every centre arm runs at the same cpd and folds as its gaze-contingent
# counterpart: same blur profile, fovea pinned to the image centre instead of
# tracking gaze, so the per-cpd difference isolates the gaze-contingency and
# stays paired on all ten folds.
#
# Launch detached from the login node:
#   cd /work/<project>/<user>/Tez
#   nohup bash scripts/slurm/foveation_pump.sh > logs/pump.log 2>&1 < /dev/null &
set -euo pipefail
cd "/work/<project>/<user>/Tez" || { echo "ERROR: cannot cd to the cluster checkout" >&2; exit 1; }
LIMIT=12    # gpu2 QOS: MaxSubmitPU=12, MaxJobsPU=4 (only 4 ever run at once)
# Protocol length, with the 10x rate decay — see foveation_train_fold.sbatch
# header. The default 12 matches that file's TEZ_EPOCHS default; an override
# (e.g. TEZ_EPOCHS=7 for the initial-variant run) is passed to every training
# job explicitly, so the arms cannot diverge.
EPOCHS=${TEZ_EPOCHS:-12}
# Dataset variant, passed to every training job and to the eval jobs. A
# non-plain variant reroutes checkpoints (ckpts_<variant>) and eval output
# (results/foveation_mit1003_<variant>) so the two protocols cannot overwrite
# each other. See foveation_train_fold.sbatch header.
DATASET_VARIANT=${TEZ_DATASET_VARIANT:-plain}
if [[ "$DATASET_VARIANT" != "plain" && "$DATASET_VARIANT" != "initial" ]]; then
  echo "ERROR: TEZ_DATASET_VARIANT must be plain|initial, got '$DATASET_VARIANT'" >&2; exit 1
fi
CKPT_ROOT=results/foveation_mit1003/ckpts
EVAL_OUT=results/foveation_mit1003
if [[ "$DATASET_VARIANT" != "plain" ]]; then
  CKPT_ROOT="${CKPT_ROOT}_${DATASET_VARIANT}"
  EVAL_OUT="${EVAL_OUT}_${DATASET_VARIANT}"
fi
SB=scripts/slurm/foveation_train_fold.sbatch

SPECS=()   # the cpd column is ignored for the normal arm
for k in 0 1 2 3 4 5 6 7 8 9; do SPECS+=("normal 20 $k"); done
for cpd in 40 20 10; do
  for k in 0 1 2 3 4 5 6 7 8 9; do SPECS+=("foveated $cpd $k"); done
  for k in 0 1 2 3 4 5 6 7 8 9; do SPECS+=("center $cpd $k"); done
done

arm_tag() {     # arm cpd -> checkpoint tag; must match foveation_train_fold.sbatch
  case "$1" in
    normal)   printf 'normal' ;;
    foveated) printf 'fov_cpd%s' "$2" ;;
    center)   printf 'fov_cpd%s_center' "$2" ;;
    *) echo "ERROR: unknown arm '$1'" >&2; exit 1 ;;
  esac
}
ckpt_final() {  # arm cpd fold -> final-epoch checkpoint path
  printf '%s/%s/fold%s/epoch_%03d/weights.pt' \
    "$CKPT_ROOT" "$(arm_tag "$1" "$2")" "$3" "$EPOCHS"
}
qcount() { squeue -h -u "$USER" -n tez-fov-train 2>/dev/null | wc -l; }

i=0; N=${#SPECS[@]}
echo "pump start $(date -u): $N specs, EPOCHS=$EPOCHS, LIMIT=$LIMIT, VARIANT=$DATASET_VARIANT"
while [ "$i" -lt "$N" ]; do
  read -r ARM CPD FOLD <<< "${SPECS[$i]}"
  if [ -f "$(ckpt_final "$ARM" "$CPD" "$FOLD")" ]; then
    echo "skip (done): $ARM cpd$CPD fold$FOLD"; i=$((i + 1)); continue
  fi
  if [ "$(qcount)" -lt "$LIMIT" ]; then
    if TEZ_ARM="$ARM" TEZ_FOVEAL_CPD="$CPD" TEZ_FOLD="$FOLD" TEZ_EPOCHS="$EPOCHS" \
        TEZ_DATASET_VARIANT="$DATASET_VARIANT" \
        sbatch --parsable "$SB" >/dev/null 2>&1; then
      echo "submitted $ARM cpd$CPD fold$FOLD ($((i + 1))/$N) $(date -u +%H:%M)"
      i=$((i + 1)); sleep 3
    else
      echo "submit blocked (QOS), waiting $(date -u +%H:%M)"; sleep 120
    fi
  else
    sleep 120
  fi
done

echo "all $N training jobs submitted; waiting for completion $(date -u)"
while [ "$(squeue -h -u "$USER" -n tez-fov-train 2>/dev/null | wc -l)" -gt 0 ]; do sleep 180; done

# An empty queue does not mean success — FAILED/TIMEOUT/CANCELLED jobs leave it too.
# Gate eval on every spec's final checkpoint actually existing (same path as the skip check).
echo "training queue empty; verifying final checkpoints $(date -u)"
missing=()
for spec in "${SPECS[@]}"; do
  read -r ARM CPD FOLD <<< "$spec"
  ck="$(ckpt_final "$ARM" "$CPD" "$FOLD")"
  [ -f "$ck" ] || missing+=("$ck")
done
if [ "${#missing[@]}" -gt 0 ]; then
  echo "ERROR: ${#missing[@]} final checkpoint(s) missing — not submitting eval:" >&2
  printf '  %s\n' "${missing[@]}" >&2
  exit 1
fi

# Two eval jobs, not one.
#
# --epoch is PINNED to the reporting epoch (EVAL_EPOCH below) rather than left to
# default. The default is "last epoch present per (arm, fold)", and
# foveation_sweep_table only *warns* when folds resolve to different epochs; a
# cross-fold mean silently mixing training lengths is close to undetectable after
# the fact. The reporting epoch is the first at which every arm has plateaued
# (scripts/summarize_epoch.py --stop-epoch); set it once, here, for every arm at
# once.
#
# The val pass exists because test is evaluated once. Every exploratory and
# secondary analysis — stratification, calibration, entropy, the amp_mass
# comparison, anything suggested by looking at the data — has to run on val or
# it is a second look at the test set. So val is dumped and read freely; test is
# dumped once and read for the primary contrast only.
EVAL_EPOCH=${TEZ_EVAL_EPOCH:-7}
if [ "$EPOCHS" -lt "$EVAL_EPOCH" ]; then
  echo "ERROR: EPOCHS=$EPOCHS < EVAL_EPOCH=$EVAL_EPOCH — the eval would demand an epoch" >&2
  echo "       training never produced. Set TEZ_EVAL_EPOCH to match." >&2
  exit 1
fi
echo "training complete; submitting eval jobs at epoch ${EVAL_EPOCH} $(date -u)"
for split in val test; do
  TEZ_FOLDS="0 1 2 3 4 5 6 7 8 9" TEZ_CPDS="40 20 10" TEZ_CENTER_CPDS="40 20 10" \
    TEZ_SPLIT="$split" TEZ_EPOCH="$EVAL_EPOCH" \
    TEZ_DATASET_VARIANT="$DATASET_VARIANT" TEZ_CKPT_ROOT="$CKPT_ROOT" TEZ_OUT="$EVAL_OUT" \
    sbatch scripts/slurm/foveation_sweep_eval.sbatch
  echo "  submitted ${split} eval (ckpt_root=$CKPT_ROOT out=$EVAL_OUT)"
done
echo "PUMP DONE $(date -u)"
