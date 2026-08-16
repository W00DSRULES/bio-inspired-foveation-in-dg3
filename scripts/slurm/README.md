# SLURM scripts — Goethe-NHR

Submission templates for running this repo on the CSC Goethe-NHR cluster
(`ssh <user>@goethe.hhlr-gu.de`).

## Cluster layout (as of bootstrap)

| Path | Use |
|------|-----|
| `/home/<project>/<user>` | dotfiles, small configs. 30 GB quota — do NOT put `.venv/`, datasets, or weights here. |
| `/work/<project>/<user>` | repo (`Tez/`), `.venv/`, dataset, weights cache, results. 94 TB group quota. |

GPU partitions (`sinfo -o "%P %l %G %f"`):

| Partition | Time limit | GPUs | Vendor | Use |
|-----------|-----------|------|--------|-----|
| `gpu_test`| 8h        | gpu:8 (2 nodes) | **AMD/ROCm** | ❌ not usable — CUDA torch falls back to CPU |
| `gpu`     | 21d       | gpu:8 (28 nodes) | **AMD/ROCm** | ❌ not usable — CUDA torch falls back to CPU |
| `sgpu`    | 45min     | gpu:8           | **AMD/ROCm** | ❌ not usable |
| `gpu2`    | 5d        | gpu:8 or gpu:2  | **NVIDIA** | ✅ the only partition this env can use |

> **Critical:** this environment ships a CUDA build of torch (`2.5.1+cu121`).
> The `gpu`/`gpu_test`/`sgpu` partitions are **AMD GPUs** (ROCm) — `ActiveFeatures=GPU`,
> no `nvidia-smi`, `CUDA_VISIBLE_DEVICES` unset. A CUDA torch cannot see them and
> silently runs on CPU (~7h/epoch). Only `gpu2` is tagged `ActiveFeatures=NVIDIA`.
> Submitting there requires the `gpu2` **QOS** (`AllowQos=gpu2`) — request it from
> CSC if `sacctmgr show assoc user=$USER` does not list it. The sbatch template
> sets `TEZ_REQUIRE_CUDA=1` and runs a CUDA preflight, so a misrouted job
> fails in seconds instead of burning hours on CPU.

CUDA module: `nvidia/cuda/12.3.0` (only version available).

MATLAB is **not** installed on this cluster. `scripts/fetch_mit1003.py`
therefore cannot run here — the dataset is prefetched locally and `scp`-ed
to `$WORKDIR/data/mit1003/`.

## One-time bootstrap (login node)

```bash
ssh <user>@goethe.hhlr-gu.de
cd /work/<project>/<user>

# 1. Repo
git clone <repo-url> Tez && cd Tez

# 2. uv (install once, into $HOME — small footprint)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 3. Environment (multi-GB; logs to terminal)
uv sync

# 4. Confirm torch built against CUDA (CUDA not visible on login node,
#    but torch.version.cuda should be non-None).
.venv/bin/python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

## Prefetch artefacts (login node has internet; compute nodes do NOT)

```bash
export TORCH_HOME=/work/<project>/<user>/torch-cache
export TEZ_DATA_ROOT=/work/<project>/<user>/Tez/data/mit1003

# DG3 pretrained weights → $TORCH_HOME/hub/checkpoints/deepgaze3.pth
.venv/bin/python -c "import deepgaze_pytorch; deepgaze_pytorch.DeepGazeIII(pretrained=True)"

# MIT1003 centerbias
.venv/bin/python -c "from tez_deepgaze.centerbias import ensure_centerbias; ensure_centerbias()"
```

## Upload MIT1003 dataset from laptop

From the local machine (skip `fetch_mit1003.py` on the cluster — no MATLAB
there):

```bash
# On laptop, in the repo root:
rsync -avh --progress data/mit1003/ \
    <user>@goethe.hhlr-gu.de:/work/<project>/<user>/Tez/data/mit1003/
```

After upload, on the cluster, confirm:

```bash
ls /work/<project>/<user>/Tez/data/mit1003/MIT1003/
# expected: stimuli/  stimuli.hdf5  fixations.hdf5
ls -la /work/<project>/<user>/torch-cache/hub/checkpoints/deepgaze3.pth
ls -la /work/<project>/<user>/Tez/data/centerbias_mit1003.npy
```

## Interactive smoke (GPU node)

Before submitting a batch job, prove the env works end-to-end on a real GPU:

```bash
srun --partition=gpu2 --qos=gpu2 --gres=gpu:1 --time=00:30:00 --pty bash
module load nvidia/cuda/12.3.0
export TORCH_HOME=/work/<project>/<user>/torch-cache
export TEZ_DATA_ROOT=/work/<project>/<user>/Tez/data/mit1003
cd /work/<project>/<user>/Tez

# 1 forward pass on CUDA
.venv/bin/python scripts/smoke_test.py
exit
```

Expected: `device: cuda`, the forward pass finishes in seconds.

## Environment variables consumed by the codebase

| Variable | Read by | Effect |
|----------|---------|--------|
| `TEZ_DATA_ROOT` | `paths.py`, `cv_split.py`, `centerbias.py` | Overrides MIT1003 dataset dir + centerbias cache location. Default: `<repo>/data/mit1003`. |
| `TORCH_HOME` | torch + `paths.py` | Pretrained weights cache. Default: `~/.cache/torch`. Set to `/work/<project>/<user>/torch-cache` on the cluster so SHA-256 is recorded correctly. |
| `CUBLAS_WORKSPACE_CONFIG` | CUDA | Required by `torch.use_deterministic_algorithms(True)`. Set to `:4096:8`. |

## Gaze-contingent foveation vs normal (10-fold CV + foveal_cpd sweep)

Trains DeepGaze III readout heads (backbone frozen) on MIT1003, comparing normal
input against gaze-contingent Geisler–Perry foveated input at three strengths
(`foveal_cpd ∈ {40, 20, 10}`, lower = stronger). MPS cannot train this model (backward gives
non-finite gradients — `foveated_train.py` skips MPS by default); CUDA `gpu2` is
required.

**Arms** (`TEZ_ARM`): `normal` (sharp input), `foveated` (gaze-contingent), `center`
(foveated with the fovea pinned to the image centre). The centre arm is the ablation
that isolates gaze-contingency: same blur profile as `foveated` at the same
`foveal_cpd`, so `foveated@C − center@C` is the gaze-contingency term. It writes to
its own tag `fov_cpd<C>_center`, because the checkpoint path is keyed only by
(tag, fold, epoch) and sharing a tag would overwrite the gaze-contingent arm.

**Matrix:** `normal` + `foveated@{40,20,10}` + `center@{40,20,10}`, 10 folds each
(70 training jobs), then one eval/aggregation job per split. `foveation_pump.sh`
submits the whole matrix under the QOS limit and the fold script auto-resumes an
incomplete fold from its newest complete epoch bundle. Prefer the pump to the
loops below; the loops show what each job needs.

```bash
cd /work/<project>/<user>/Tez && mkdir -p logs

# normal arm, 10 folds (foveal_cpd irrelevant for this arm)
for k in 0 1 2 3 4 5 6 7 8 9; do
  TEZ_ARM=normal TEZ_FOLD=$k sbatch scripts/slurm/foveation_train_fold.sbatch
done

# foveated arm, 10 folds × sweep {40,20,10}
for C in 40 20 10; do
  for k in 0 1 2 3 4 5 6 7 8 9; do
    TEZ_ARM=foveated TEZ_FOVEAL_CPD=$C TEZ_FOLD=$k \
      sbatch scripts/slurm/foveation_train_fold.sbatch
  done
done
squeue -u $USER

# after all 70 checkpoints exist, evaluate on the held-out split and aggregate
# into results/foveation_mit1003/<split>/table.{md,json} + figure
sbatch scripts/slurm/foveation_sweep_eval.sbatch
```

Checkpoints land at `results/foveation_mit1003/ckpts/<tag>/fold<k>/epoch_<E>/`
(`<tag>` = `normal`, `fov_cpd<C>` or `fov_cpd<C>_center`) — the layout
`foveation_sweep_table.py` reads. Evaluate a centre arm by passing its cpd to
`--center-cpds` (`TEZ_CENTER_CPDS` in `foveation_sweep_eval.sbatch`); the
fixed-centre setting itself is read back from each checkpoint's `metrics.json`,
so it cannot be evaluated gaze-contingently by mistake.

```bash
# fixed-centre ablation, 10 folds × each cutoff {40,20,10}
for C in 40 20 10; do
  for k in 0 1 2 3 4 5 6 7 8 9; do
    TEZ_ARM=center TEZ_FOVEAL_CPD=$C TEZ_FOLD=$k \
      sbatch scripts/slurm/foveation_train_fold.sbatch
  done
done
```

**Exporting a table by hand.** The eval job takes the cutoff list, the split, the
epoch and the output directory from the environment, so an export needs no code
change. `foveation_sweep_table.py` appends the split to `--out`, so the default
`TEZ_OUT` lands in `results/foveation_mit1003/{val,test}/`. This is the command
that reproduces the thesis test table (`results/foveation_mit1003_initial/test/`,
epoch 5 of the seven-epoch paper-protocol run):

```bash
TEZ_CPDS="40 20 10" TEZ_CENTER_CPDS="40 20 10" \
TEZ_SPLIT=test TEZ_EPOCH=5 TEZ_DATASET_VARIANT=initial \
sbatch scripts/slurm/foveation_sweep_eval.sbatch
```

With `TEZ_DATASET_VARIANT=initial` the sbatch derives `TEZ_CKPT_ROOT`
(`results/foveation_mit1003/ckpts_initial`) and `TEZ_OUT`
(`results/foveation_mit1003_initial`) itself. It writes `table.{md,json}`,
`per_image_ig.json` and the sweep figure; `TEZ_SPLIT=val` writes the val counterpart. The `normal` control
arm is always evaluated alongside whatever cutoffs are named, so the ΔIG in the
output is self-contained. Requires the matching
`results/foveation_mit1003/ckpts/<tag>/fold{0..9}/` on the cluster. Narrowing
`TEZ_CPDS`/`TEZ_CENTER_CPDS` exports a single arm, but writes to the same path, so
point `TEZ_OUT` at a scratch directory when you do. `TEZ_EPOCH` pins the
checkpoint epoch; left unset, the last complete epoch per fold is used, which
need not be the reporting epoch — pass it explicitly.

**Interactive smoke first** (prove CUDA training works before the 70-job batch):

```bash
srun --partition=gpu2 --qos=gpu2 --gres=gpu:1 --time=00:30:00 --pty bash
module load nvidia/cuda/12.3.0
export TORCH_HOME=/work/<project>/<user>/torch-cache TEZ_DATA_ROOT=/work/<project>/<user>/Tez/data/mit1003
cd /work/<project>/<user>/Tez
.venv/bin/python -m tez_deepgaze.foveated_train --device cuda --fold 0 --epochs 1 \
    --subsample-train 20 --subsample-val 10 --foveate --foveal-cpd 20
exit
```

Invariants to check in each checkpoint `metrics.json`:

- `reproducibility.args.device == "cuda"`.
- `train.nonfinite_steps == 0` (no diverged/skipped steps; CUDA should be clean,
  unlike MPS).
- `val.ig_bits_per_fix` is well above 0 (not collapsed to the centerbias floor).
- `arm` matches the intended arm; `foveation.foveal_cpd` matches the intended sweep point.

Tunables (defaults in `foveation_train_fold.sbatch`): `TEZ_EPOCHS=12` (must match
`foveation_pump.sh`; arms are only comparable at equal epochs), `TEZ_LR=3e-4`
(base rate, decayed 10x at epochs 5, 8 and 11; chosen by a fold-0 LR probe),
`TEZ_MICRO_BATCH=36`.

The eval job writes its report to `$TEZ_OUT/<split>/table.{md,json}` — one row per
arm, LL/IG/NSS/AUC as mean ± SE across the requested folds, plus a
normal-vs-foveated gap table and the per-image IGs in `per_image_ig.json`.
`foveation_sweep_table.py` appends the split itself, so the default `TEZ_OUT`
(`results/foveation_mit1003/`) resolves to `…/val/` and `…/test/`.
