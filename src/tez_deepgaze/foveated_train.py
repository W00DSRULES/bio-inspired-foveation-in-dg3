"""Fine-tune DeepGaze III's readout heads on MIT1003, with the input optionally
foveated gaze-contingently.

The mechanism:

  * the model is a plain :class:`deepgaze_pytorch.DeepGazeIII`;
  * the DenseNet backbone (``model.features``) is frozen; the readout heads
    (saliency / scanpath / fixation-selection / finalizer) are trained;
  * with ``--foveate`` the input image is re-foveated around the current
    fixation before every forward pass (gaze-contingent input, via
    :class:`foveate_input.Foveation`); ``--no-foveate`` is the normal-input
    control arm.

Two arms, one flag: ``--foveate`` / ``--no-foveate``. Same protocol (epochs,
LR, data, nearest-pixel target sampling) so the arms differ only in the input.

Viewing geometry is config, not hard-coded: ``--ppd`` (MIT1003 ≈ 35)
and ``--foveal-cpd`` set the Geisler–Perry falloff.

Smoke test (CPU, a few minutes — MPS is rejected for training):

    python -m tez_deepgaze.foveated_train --fold 0 --epochs 1 \
        --subsample-train 20 --subsample-val 12 --foveate
"""
from __future__ import annotations

import argparse
import hashlib
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import deepgaze_pytorch

from .datasets import MIT1003TrainingDataset
from .device import pick_device, to_device
from .fast_dg3 import apply_fast_forward, apply_fast_layernorm
from .foveate_input import FOVEAL_CPD, MIT1003_PPD, Foveation
from .ig import LOG2
from .instrument import (
    FROZEN_SUBMODULES,
    _log_density_batch_core,
    image_tensor,
    load_checkpoint_weights,
    save_checkpoint_bundle,
)
from .paths import CENTERBIAS_CACHE, REPO_ROOT, RESULTS, deepgaze3_weights_path
from .script_utils import foveation_record

START_FIXATION = 1  # score from the 2nd fixation; the 1st is history, not a target
MIN_PATH_LEN_FOR_TRAINING = 2  # need at least 2 fixations to form one prefix

# Learning-rate decay. The DG3 authors step the rate down by a factor of 10 at
# fixed epoch milestones; base 3e-4 was chosen by a 4-epoch flat-LR probe over
# {3e-5, 1e-4, 3e-4, 1e-3} x {normal, foveated@40, foveated@10} on fold 0:
# 3e-4 gave the best epoch-4 val IG on every arm with zero nonfinite steps;
# 1e-3 was equally stable but val IG turned over within the four epochs. All
# 12 probe cells had zero nonfinite steps and strictly falling train NLL. The
# rate is base for epochs 1-4, base/10 for 5-7, base/100 for 8-10, base/1000
# from 11.
LR_MILESTONES = (5, 8, 11)
LR_GAMMA = 0.1

# --micro-batch is calibrated for an image of this size; other sizes scale to
# keep activation memory roughly constant (see _adaptive_micro_batch).
REF_PIXELS = 768 * 1024


def lr_for_epoch(epoch: int, base_lr: float) -> float:
    """Scheduled learning rate for a 1-based ``epoch`` index.

    Computed from the epoch number, not from stepped scheduler state. A
    ``torch.optim.lr_scheduler`` holds its position in an internal counter, so a
    ``--start-epoch`` resume would either have to replay the skipped steps or
    persist and reload that counter — and getting it wrong is silent, the run
    just trains at the wrong rate. The closed form has nothing to carry across a
    restart: epoch N gets the same rate whether or not the run was interrupted.
    """
    return base_lr * LR_GAMMA ** sum(epoch >= m for m in LR_MILESTONES)


def set_epoch_lr(optimizer: torch.optim.Optimizer, epoch: int, base_lr: float) -> float:
    """Apply :func:`lr_for_epoch` to every param group; return the rate set."""
    lr = lr_for_epoch(epoch, base_lr)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def _adaptive_micro_batch(micro_batch: int, height: int, width: int) -> int:
    """Scale the micro-batch by image area so peak memory stays roughly fixed.

    Activation memory grows about linearly in (batch x pixels) — measured on an
    A100 at 768x1024 it is ~1.8 GB per sample — so a batch tuned on one image
    size overflows on a larger one. MIT1003 spans 414,720 to 1,048,576 pixels
    (142 distinct shapes), a 2.5x range, and the largest images are 1.33x the
    768x1024 reference: a micro-batch of 32 fits there in ~60 GB but needs
    ~80 GB on the biggest image, which is the whole card.

    Scaling by area keeps every stimulus near the same footprint instead of
    sizing everything for the worst case. This does not change results: the
    micro-batch is only how gradient accumulation is chunked, and chunk size was
    measured to leave gradients identical (differences ~1e-5 relative, and none
    at all between different multi-chunk sizes).
    """
    return max(1, int(micro_batch * REF_PIXELS / max(height * width, 1)))


# ----- reproducibility metadata + scanpath-pair helpers -----


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for buf in iter(lambda: fh.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


def _torch_hub_deepgaze3_path() -> Path | None:
    """Best-effort lookup for the cached pretrained DG3 weights file.

    Uses :func:`paths.deepgaze3_weights_path` (``TORCH_HOME``-aware). Returns
    ``None`` if the file is not present.
    """
    cand = deepgaze3_weights_path()
    return cand if cand.exists() else None


def code_provenance() -> dict:
    """Which code actually ran: commit, and a hash of any uncommitted changes.

    The other hashes pin the *inputs* (lockfile, pretrained weights,
    centerbias) but not the source, and a checkout can run a commit with
    uncommitted changes laid over it. ``git_dirty_sha256`` is a hash of
    ``git diff HEAD`` — empty tree and dirty tree are then distinguishable, and
    two dirty runs can be compared for equality without storing the diff itself.

    Best-effort: a missing git, or a non-repository working directory, records
    nulls rather than failing a training run.
    """
    import subprocess

    def _git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", "-C", str(REPO_ROOT), *args],
                capture_output=True, text=True, timeout=10, check=True,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return None

    head = _git("rev-parse", "HEAD")
    diff = _git("diff", "HEAD")
    out: dict = {"git_commit": head.strip() if head else None}
    if diff is not None:
        out["git_dirty"] = bool(diff.strip())
        out["git_dirty_sha256"] = (
            hashlib.sha256(diff.encode()).hexdigest() if diff.strip() else None
        )
    return out


def reproducibility_metadata(seed: int) -> dict:
    """Collect SHAs, seeds, and the cudnn flags for the run's metrics JSON."""
    meta: dict = {
        "seed": int(seed),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(torch.backends.mps.is_available()),
        **code_provenance(),
    }
    cudnn = getattr(torch.backends, "cudnn", None)
    if cudnn is not None:
        meta["cudnn_deterministic"] = bool(cudnn.deterministic)
        meta["cudnn_benchmark"] = bool(cudnn.benchmark)

    uv_lock = REPO_ROOT / "uv.lock"
    if uv_lock.exists():
        meta["uv_lock_sha256"] = file_sha256(uv_lock)
    dg3_pth = _torch_hub_deepgaze3_path()
    if dg3_pth is not None:
        meta["deepgaze3_pth_sha256"] = file_sha256(dg3_pth)
        meta["deepgaze3_pth_path"] = str(dg3_pth)
    # Same path centerbias.py actually loads from (TEZ_DATA_ROOT-aware).
    if CENTERBIAS_CACHE.exists():
        meta["centerbias_mit1003_npy_sha256"] = file_sha256(CENTERBIAS_CACHE)
    # The CV partition is load-bearing — the ten released DG3 ensemble members
    # correspond to it — so a silent change there would move every number.
    from .datasets import DEFAULT_CV_SPLIT

    if Path(DEFAULT_CV_SPLIT).exists():
        meta["cv_split_sha256"] = file_sha256(DEFAULT_CV_SPLIT)
        meta["cv_split_path"] = str(DEFAULT_CV_SPLIT)
    return meta


def _gather_targets(
    log_d: torch.Tensor,
    target_x: list[float],
    target_y: list[float],
) -> torch.Tensor:
    """Sample the log-density at each target fixation via nearest-pixel lookup.

    Matches :func:`baseline._nearest_sample` (round-half-even; pysaliency's
    readers truncate instead — see that docstring), so training NLL and eval
    LL use the same interpolation. Gradients flow through the
    indexed values.

    Indexed in one shot rather than in a Python loop over the batch. A loop
    issues one ``index_put`` per sample; at micro-batch 36 that is ~40 %
    of the step's CPU time spent launching kernels, which is fine while the GPU
    is the slower side and becomes the wall as soon as it is not. ``np.round``
    is round-half-to-even, the same rule Python's ``round`` follows, so the
    chosen pixel is unchanged.
    """
    H, W = log_d.shape[-2:]
    B = log_d.size(0)
    xi = np.clip(np.round(np.asarray(target_x, dtype=np.float64)), 0, W - 1).astype(np.int64)
    yi = np.clip(np.round(np.asarray(target_y, dtype=np.float64)), 0, H - 1).astype(np.int64)
    idx = torch.arange(B, device=log_d.device)
    return log_d[idx,
                 torch.from_numpy(yi).to(log_d.device),
                 torch.from_numpy(xi).to(log_d.device)]


def _build_pairs_for_image(paths, start_fixation: int = START_FIXATION):
    """Walk each subject's scanpath, yielding (hist_xs, hist_ys, target_x, target_y).

    A scanpath of length N produces N-start_fixation training examples:
    predicting fix[start_fixation], fix[start_fixation+1], ..., fix[N-1]
    each given the history of all prior fixations.
    """
    hist_x_batch, hist_y_batch, tx, ty = [], [], [], []
    for xs, ys, _ in paths:
        valid_end = len(xs)
        if valid_end < max(MIN_PATH_LEN_FOR_TRAINING, start_fixation + 1):
            continue
        for i in range(start_fixation, valid_end):
            hist_x_batch.append(list(xs[:i]))
            hist_y_batch.append(list(ys[:i]))
            tx.append(float(xs[i]))
            ty.append(float(ys[i]))
    return hist_x_batch, hist_y_batch, tx, ty


def _stim_pairs(dataset, sidx: int, max_pairs_per_stim: int | None):
    """Image, centerbias and truncated (history, target) pairs for one stimulus.

    Shared by the train and eval epoch loops so the pair construction and the
    ``max_pairs_per_stim`` truncation cannot drift between them.
    """
    image, cb = dataset.image_and_centerbias(sidx)
    hist_x, hist_y, tx, ty = _build_pairs_for_image(dataset.scanpaths_for_stim(sidx))
    if max_pairs_per_stim:
        hist_x, hist_y = hist_x[:max_pairs_per_stim], hist_y[:max_pairs_per_stim]
        tx, ty = tx[:max_pairs_per_stim], ty[:max_pairs_per_stim]
    return image, cb, hist_x, hist_y, tx, ty


# ----- freeze + audit -----

# The readout networks that adapt to (foveated) backbone features. The DenseNet
# backbone is frozen. The finalizers — a fixed Gaussian
# smoothing kernel + a center-bias weight — are also frozen: they are output
# shaping, not content readout, so adapting them would let the arms differ in
# something other than the input transform. (Freezing the smoothing sigma is
# also required on MPS: a
# grad-carrying gaussian kernel trips an internal conv1d assert.)
# FROZEN_SUBMODULES lives in instrument.py because the checkpoint format
# (what weights.pt omits, what the loader may demand back) is defined by it.
TRAINABLE_SUBMODULES = ("saliency_networks", "scanpath_networks", "fixation_selection_networks")


def freeze_for_head_training(model: torch.nn.Module) -> None:
    """Train the readout networks; freeze the backbone and the finalizers.

    Frozen submodules go to ``requires_grad=False`` and ``.eval()`` — the
    backbone's BatchNorm running stats stay fixed and the finalizer's Gaussian
    smoothing kernel stays constant. The whole model is kept in ``.eval()``
    (the trainable readout networks use LayerNorm only, so train/eval mode is
    numerically irrelevant); ``requires_grad`` alone selects what updates.
    """
    model.eval()
    for name in FROZEN_SUBMODULES:
        for p in getattr(model, name).parameters():
            p.requires_grad_(False)


def audit_trainable(model: torch.nn.Module) -> dict:
    """Assert the frozen submodules are frozen + in eval mode; report counts."""
    frozen_train = 0
    for name in FROZEN_SUBMODULES:
        sub = getattr(model, name)
        frozen_train += sum(p.numel() for p in sub.parameters() if p.requires_grad)
        if sub.training:
            raise AssertionError(f"{name} must be in eval() mode (running stats/kernels would shift)")
    if frozen_train != 0:
        raise AssertionError(f"frozen submodules have {frozen_train} trainable params")
    head_train = sum(
        p.numel() for name in TRAINABLE_SUBMODULES
        for p in getattr(model, name).parameters() if p.requires_grad
    )
    total_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if head_train == 0 or head_train != total_train:
        raise AssertionError(
            f"expected all {total_train} trainable params in {TRAINABLE_SUBMODULES}; "
            f"got {head_train} there"
        )
    return {
        "backbone_params": int(sum(p.numel() for p in model.features.parameters())),
        "frozen_trainable": int(frozen_train),
        "head_trainable": int(head_train),
    }


# ----- train / eval loops -----


def train_one_epoch_foveated(
    model,
    dataset: MIT1003TrainingDataset,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    foveation: Foveation,
    *,
    foveate: bool,
    grad_clip: float | None = 1.0,
    max_pairs_per_stim: int | None = None,
    subsample: int | None = None,
    seed: int = 0,
    micro_batch: int = 4,
    profile_timing: bool = False,
) -> dict:
    """One epoch of head fine-tuning; per-fixation NLL at human targets.

    With ``foveate`` the log-density is computed on gaze-contingent foveated
    input; otherwise on the sharp image (both via
    :func:`instrument._log_density_batch_core`). Targets are sampled
    nearest-pixel (:func:`_gather_targets`), matching
    ``baseline._nearest_sample`` so train-time and eval-time LL agree.
    """
    model.eval()  # backbone BN fixed; heads have no mode-sensitive layers
    stim_indices = list(dataset.stim_indices)
    rng = np.random.RandomState(seed)
    if subsample is not None and subsample < len(stim_indices):
        stim_indices = rng.choice(stim_indices, size=subsample, replace=False).tolist()
    rng.shuffle(stim_indices)

    total_nll_nats = 0.0
    total_fix = 0
    nonfinite_steps = 0
    # Diagnostics. Grad norm is free (clip_grad_norm_ returns it anyway); peak
    # memory is a counter read, no sync. `profile_timing` adds per-phase timers
    # that each need a cuda synchronize, so it is opt-in — leaving it on would
    # itself serialise the pipeline it is trying to measure.
    grad_norms: list[float] = []
    n_clipped = 0
    n_steps = 0
    phase = {k: 0.0 for k in ("data", "foveation", "forward", "backward", "optimizer")}

    def _tick() -> float:
        if profile_timing and device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    trainable = [p for p in model.parameters() if p.requires_grad]

    pbar = tqdm(stim_indices, desc="train")
    for sidx in pbar:
        t0 = _tick()
        image, cb, hist_x, hist_y, tx, ty = _stim_pairs(dataset, sidx, max_pairs_per_stim)
        phase["data"] += _tick() - t0
        if not hist_x:
            continue
        n_total = len(hist_x)
        mb = _adaptive_micro_batch(micro_batch, image.shape[0], image.shape[1])

        # The blur pyramid depends only on the image, so build it once per
        # stimulus and reuse it across the micro-batches.
        t0 = _tick()
        fov_stack = (foveation.blur_stack(image_tensor(image, device))
                     if foveate else None)
        phase["foveation"] += _tick() - t0

        optimizer.zero_grad()
        # Accumulate the loss and the finiteness flag on the GPU. Reading either
        # with .item() inside the micro-batch loop forces a synchronise, which
        # stalls the CPU until every queued kernel has finished and stops it
        # running ahead to enqueue the next batch. One read per stimulus instead
        # of two per micro-batch is the same arithmetic with far fewer stalls.
        step_nll_t = torch.zeros((), device=device, dtype=torch.float64)
        finite_t = torch.ones((), device=device, dtype=torch.bool)
        for s in range(0, n_total, mb):
            e = min(s + mb, n_total)
            t0 = _tick()
            log_d = _log_density_batch_core(
                model, image, cb, hist_x[s:e], hist_y[s:e], device,
                foveation=foveation if foveate else None, foveation_stack=fov_stack,
            )
            ll = _gather_targets(log_d, tx[s:e], ty[s:e])  # nearest-pixel
            phase["forward"] += _tick() - t0

            finite_t &= torch.isfinite(ll).all()
            t0 = _tick()
            (-ll.sum() / n_total).backward()
            phase["backward"] += _tick() - t0
            step_nll_t += (-ll.detach()).sum().to(torch.float64)
        t0 = _tick()
        # Skip the update if this stimulus produced a non-finite loss OR a
        # non-finite accumulated gradient — one bad step corrupts every later
        # forward (head activations overflow), so never apply it.
        #
        # The finiteness of the gradient is read off clip_grad_norm_'s return
        # value: the total gradient norm is non-finite exactly when some
        # gradient is. Checking each of the 380 trainable tensors with
        # `torch.isfinite(p.grad).all()` in Python would be 380 CUDA
        # synchronises per stimulus (~300k per epoch), each draining the queue
        # and stopping the CPU running ahead to enqueue the next micro-batch;
        # this way loss, flag and norm come back in a single transfer.
        #
        # max_norm=inf when clipping is off: torch clamps the scale factor to 1,
        # so the gradients are multiplied by exactly 1.0 and left unchanged.
        gn_t = torch.nn.utils.clip_grad_norm_(
            trainable, grad_clip if grad_clip is not None else float("inf")
        )
        stats = torch.stack([
            step_nll_t,
            (finite_t & torch.isfinite(gn_t)).to(torch.float64),
            gn_t.to(torch.float64),
        ]).cpu()                       # the one sync per stimulus
        step_nll, step_ok, gn = float(stats[0]), bool(stats[1]), float(stats[2])
        if not step_ok:
            nonfinite_steps += 1
            optimizer.zero_grad()
            phase["optimizer"] += _tick() - t0
            continue
        if grad_clip is not None:
            grad_norms.append(gn)
            if gn > grad_clip:
                n_clipped += 1
        optimizer.step()
        n_steps += 1
        phase["optimizer"] += _tick() - t0

        total_nll_nats += step_nll
        total_fix += n_total
        pbar.set_description(
            f"train  NLL={total_nll_nats / max(total_fix, 1):.3f} nats/fix"
        )

    out = {
        "n_stim_seen": len(stim_indices),
        "n_fix_seen": int(total_fix),
        "nonfinite_steps": int(nonfinite_steps),
        "mean_nll_nats_per_fix": float(total_nll_nats / max(total_fix, 1)),
        "mean_ll_bits_per_fix": float(-total_nll_nats / max(total_fix, 1) / LOG2),
        "n_optimizer_steps": n_steps,
    }
    if grad_norms:
        arr = np.asarray(grad_norms)
        out["grad_norm"] = {
            "mean": float(arr.mean()), "median": float(np.median(arr)),
            "max": float(arr.max()), "p95": float(np.percentile(arr, 95)),
            "n_clipped": n_clipped,
            "frac_clipped": float(n_clipped / max(n_steps, 1)),
        }
    if device.type == "cuda":
        out["gpu_memory_gb"] = {
            "peak_allocated": torch.cuda.max_memory_allocated() / 1e9,
            "peak_reserved": torch.cuda.max_memory_reserved() / 1e9,
        }
    if profile_timing:
        total = sum(phase.values())
        out["phase_seconds"] = {k: round(v, 2) for k, v in phase.items()}
        out["phase_fraction"] = {k: round(v / total, 4) for k, v in phase.items()} if total else {}
    return out


@torch.no_grad()
def eval_one_epoch_foveated(
    model,
    dataset: MIT1003TrainingDataset,
    device: torch.device,
    foveation: Foveation,
    *,
    foveate: bool,
    max_pairs_per_stim: int | None = None,
    subsample: int | None = None,
    seed: int = 0,
    micro_batch: int = 8,
) -> dict:
    """Pooled per-fixation LL and IG-over-centerbias (bits) on the val fold.

    IG is the sentinel: a collapsed fit sits at the
    centerbias floor (IG ≈ 0). Same nearest-pixel convention and same
    input (foveated iff ``foveate``) as training.
    """
    model.eval()
    ll_nats_sum = 0.0
    cb_nats_sum = 0.0
    n_fix = 0

    stim_indices = list(dataset.stim_indices)
    if subsample is not None and subsample < len(stim_indices):
        rng = np.random.RandomState(seed)
        stim_indices = sorted(rng.choice(stim_indices, size=subsample, replace=False).tolist())

    # Per-image IG, kept alongside the aggregate. The aggregate alone cannot say
    # whether a common stopping epoch was a clear call or a borderline one, and
    # it discards the image-level spread that dominates the variance.
    per_image: list[dict] = []

    pbar = tqdm(stim_indices, desc="eval")
    for sidx in pbar:
        image, cb, hist_x, hist_y, tx, ty = _stim_pairs(dataset, sidx, max_pairs_per_stim)
        if not hist_x:
            continue
        n_total = len(hist_x)
        mb = _adaptive_micro_batch(micro_batch, image.shape[0], image.shape[1])
        fov_stack = (foveation.blur_stack(image_tensor(image, device))
                     if foveate else None)
        cb_t_full = torch.from_numpy(cb.astype(np.float32)).to(device)
        img_ll = img_cb = 0.0
        for s in range(0, n_total, mb):
            e = min(s + mb, n_total)
            log_d = _log_density_batch_core(
                model, image, cb, hist_x[s:e], hist_y[s:e], device,
                foveation=foveation if foveate else None, foveation_stack=fov_stack,
            )
            ll = _gather_targets(log_d, tx[s:e], ty[s:e]).detach().cpu().numpy()
            cb_expanded = cb_t_full.unsqueeze(0).expand(e - s, -1, -1)
            cb_ll = _gather_targets(cb_expanded, tx[s:e], ty[s:e]).detach().cpu().numpy()
            ll_nats_sum += float(ll.sum())
            cb_nats_sum += float(cb_ll.sum())
            img_ll += float(ll.sum())
            img_cb += float(cb_ll.sum())
        n_fix += n_total
        per_image.append({
            "stim": int(sidx),
            "n_fix": int(n_total),
            "ig_bits_per_fix": float(((img_ll - img_cb) / n_total) / LOG2),
        })
        if n_fix > 0:
            pbar.set_description(
                f"eval   LL={(ll_nats_sum / n_fix) / LOG2:+.3f} b/fix  "
                f"IG={((ll_nats_sum - cb_nats_sum) / n_fix) / LOG2:+.3f}"
            )

    if n_fix == 0:
        return {"n_stim_seen": len(stim_indices), "n_fix_seen": 0}
    return {
        "n_stim_seen": len(stim_indices),
        "n_fix_seen": int(n_fix),
        "ll_bits_per_fix": float((ll_nats_sum / n_fix) / LOG2),
        "cb_ll_bits_per_fix": float((cb_nats_sum / n_fix) / LOG2),
        "ig_bits_per_fix": float(((ll_nats_sum - cb_nats_sum) / n_fix) / LOG2),
        "per_image": per_image,
    }


# ----- main -----


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0, help="CV fold (0..9)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4,
                    help="base Adam LR for the readout heads, chosen by the fold-0 "
                         f"LR probe. Decayed by {LR_GAMMA:g} at epochs {LR_MILESTONES} "
                         "— see lr_for_epoch")
    ap.add_argument("--grad-clip", type=float, default=1.0,
                    help="max grad norm; steps with non-finite grads are skipped")
    ap.add_argument("--foveate", action=argparse.BooleanOptionalAction, default=True,
                    help="gaze-contingent foveated input; --no-foveate is the "
                         "normal-input control arm")
    ap.add_argument("--ppd", type=float, default=MIT1003_PPD,
                    help="pixels per degree (viewing geometry; MIT1003 ≈ 35)")
    ap.add_argument("--foveal-cpd", type=float, default=FOVEAL_CPD,
                    help=f"foveal cutoff cyc/deg; human ~{FOVEAL_CPD:.0f} (faithful), "
                         "lower = stronger/coarser-than-human")
    ap.add_argument("--micro-batch", type=int, default=4,
                    help="per-image gradient-accumulation chunk (raise on CUDA)")
    ap.add_argument("--max-pairs-per-stim", type=int, default=None,
                    help="cap (history, target) pairs per stimulus — smoke/debug only")
    ap.add_argument("--init-weights", type=Path, default=None,
                    help="resume from this epoch bundle's weights.pt instead of the "
                         "pretrained read-out. Pair with --start-epoch. If a sibling "
                         "optimizer.pt exists it is loaded too, making the resume a true "
                         "continuation; without it Adam restarts from zeroed moments "
                         "(a warm restart), which is only a fair comparison if every "
                         "arm is resumed the same way.")
    ap.add_argument("--center-fovea", action="store_true",
                    help="pin the fovea to the image centre instead of tracking gaze. The "
                         "fixed-centre ablation: same Geisler-Perry blur, but the input no "
                         "longer changes with the model's fixation, which is what prior "
                         "pixel-level foveated networks do. Isolates the gaze-contingency.")
    ap.add_argument("--fast-forward", action=argparse.BooleanOptionalAction, default=True,
                    help="hoist the loop-invariant scanpath encoding out of the 10-member\n"
                         "ensemble loop. Bit-identical (tests/test_fast_dg3.py asserts atol=0); "
                         "--no-fast-forward restores the vendor forward for comparison.")
    ap.add_argument("--fast-layernorm", action=argparse.BooleanOptionalAction, default=True,
                    help="replace the vendor DG3 LayerNorm forward, which is 58%% of a "
                         "training step's GPU time because it materialises its per-channel "
                         "affine parameters to full (C, H, W) tensors and reduces with only B "
                         "rows of parallelism. Same algebra, ~1e-6 different in the last bits "
                         "of the log-density; --no-fast-layernorm restores the vendor version.")
    ap.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True,
                    help="cudnn deterministic algorithms (default on; the project requires "
                         "reproducible runs). --no-deterministic also enables cudnn autotuning "
                         "and is for PROFILING ONLY — its results are not reproducible.")
    ap.add_argument("--profile-timing", action="store_true",
                    help="record a per-phase timing breakdown (data / foveation / forward / "
                         "backward / optimizer). Each phase boundary needs a cuda synchronize, "
                         "which itself serialises the pipeline, so this SLOWS the run and is for "
                         "diagnosis only — never for a reported run.")
    ap.add_argument("--start-epoch", type=int, default=1,
                    help="first epoch index to run (default 1). Set to N+1 when resuming "
                         "from epoch N so the per-epoch data shuffle (seeded seed+epoch) "
                         "continues the original sequence instead of repeating it, and so "
                         "checkpoints land beside the existing ones rather than "
                         "overwriting them.")
    ap.add_argument("--device", choices=["cuda", "cpu"], default=None,
                    help="override device. MPS is rejected for training: its "
                         "backward gives non-finite gradients for DG3 heads. "
                         "Use cpu locally, cuda on the cluster.")
    ap.add_argument("--dataset-variant", choices=["plain", "initial"], default="plain",
                    help="'plain' trains from the second free fixation (the committed "
                         "protocol). 'initial' loads MIT1003_initial_fix_consistent, "
                         "where index 0 is the forced central fixation, so the same "
                         "START_FIXATION=1 trains on every free fixation with the "
                         "centre as history — the DG3 paper's training protocol. "
                         "Recorded in every checkpoint's metrics.json; never mix "
                         "variants under one checkpoint root.")
    ap.add_argument("--subsample-train", type=int, default=None)
    ap.add_argument("--subsample-val", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=RESULTS / "foveation_mit1003" / "train_smoke")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.backends.cudnn is not None:
        # Determinism is a deliberate project choice and the default. It costs
        # speed twice over: deterministic convolution algorithms are slower than
        # the fastest available, and it rules out cudnn autotuning (benchmark),
        # because autotuning may select a different algorithm between runs.
        # --no-deterministic exists ONLY to price that choice in a profiling
        # run; results produced with it must not be reported.
        torch.backends.cudnn.deterministic = args.deterministic
        torch.backends.cudnn.benchmark = not args.deterministic
        if not args.deterministic:
            print("WARNING: cudnn determinism DISABLED and benchmark enabled. "
                  "This run is for timing only — its numbers are not reproducible "
                  "and must not be reported.")

    if args.device is not None:
        device = torch.device(args.device)
    else:
        # pick_device honors the TEZ_REQUIRE_CUDA cluster guard. MPS backward
        # produces non-finite gradients for DG3 head training, so downgrade an
        # MPS pick to CPU — every step would be skipped and the checkpoint
        # would just repeat the pretrained heads.
        device = pick_device()
        if device.type == "mps":
            device = torch.device("cpu")
    # Three arms, not two: the fixed-centre ablation is also foveated input, so
    # labelling it "foveated" would make two different arms indistinguishable in
    # the metrics. (Eval does not rely on this string — it reads center_fovea
    # from the foveation record — but a checkpoint should say what it is.)
    if not args.foveate:
        arm = "normal"
    else:
        arm = "foveated_center" if args.center_fovea else "foveated"
    print(f"device: {device}   arm: {arm}")

    print("loading pretrained DeepGaze III...")
    model = to_device(deepgaze_pytorch.DeepGazeIII(pretrained=True), device)
    freeze_for_head_training(model)
    audit = audit_trainable(model)
    print(f"trainable readout params: {audit['head_trainable']:,}  "
          f"(backbone frozen: {audit['backbone_params']:,})")

    if args.fast_forward:
        n_patched = apply_fast_forward(model)
        print(f"fast forward applied to {n_patched} mixture module(s)")
    if args.fast_layernorm:
        n_ln = apply_fast_layernorm(model)
        print(f"fast layernorm applied to {n_ln} module(s)")

    foveation = Foveation(ppd=args.ppd, foveal_cpd=args.foveal_cpd,
                          center_fovea=args.center_fovea)
    train_ds = MIT1003TrainingDataset(fold_no=args.fold, split="train",
                                      dataset_variant=args.dataset_variant)
    val_ds = MIT1003TrainingDataset(fold_no=args.fold, split="val",
                                    dataset_variant=args.dataset_variant)
    print(f"{train_ds}\n{val_ds}")

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)

    resume: dict | None = None
    if args.init_weights is not None:
        fmt = load_checkpoint_weights(model, args.init_weights, device)
        opt_path = args.init_weights.parent / "optimizer.pt"
        opt_loaded = opt_path.exists()
        if opt_loaded:
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))
        resume = {
            "init_weights": str(args.init_weights),
            "init_weights_sha256": file_sha256(args.init_weights),
            "init_weights_format": fmt,
            "optimizer_state_restored": opt_loaded,
            "continuation": "true" if opt_loaded else "warm-restart",
            "start_epoch": args.start_epoch,
        }
        print(f"resuming from {args.init_weights} ({fmt} format); "
              f"optimizer state {'restored' if opt_loaded else 'NOT found — warm restart'}")
        if not opt_loaded:
            print("  warning: Adam moments restart from zero. Only comparable across arms "
                  "if every arm is resumed the same way; do not describe this as "
                  "uninterrupted training.")
    if args.start_epoch > args.epochs:
        raise SystemExit(
            f"--start-epoch {args.start_epoch} exceeds --epochs {args.epochs}: nothing to do"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    repro = reproducibility_metadata(args.seed)
    repro["args"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    repro["arm"] = arm
    repro["audit"] = audit
    if resume is not None:
        repro["resume"] = resume

    for epoch in range(args.start_epoch, args.epochs + 1):
        t0 = time.time()
        lr = set_epoch_lr(optimizer, epoch, args.lr)
        print(f"\n=== epoch {epoch} / {args.epochs} ({arm})  lr={lr:.1e} ===")
        train_stats = train_one_epoch_foveated(
            model, train_ds, optimizer, device, foveation,
            foveate=args.foveate,
            grad_clip=args.grad_clip, max_pairs_per_stim=args.max_pairs_per_stim,
            subsample=args.subsample_train, seed=args.seed + epoch,
            micro_batch=args.micro_batch, profile_timing=args.profile_timing,
        )
        if train_stats["n_fix_seen"] == 0 and train_stats["nonfinite_steps"] > 0:
            raise SystemExit(
                f"epoch {epoch}: every optimizer step was skipped as non-finite "
                f"({train_stats['nonfinite_steps']} steps). A checkpoint would just "
                "repeat the pretrained heads — aborting instead of writing one."
            )
        val_stats = eval_one_epoch_foveated(
            model, val_ds, device, foveation,
            foveate=args.foveate, max_pairs_per_stim=args.max_pairs_per_stim,
            subsample=args.subsample_val, seed=args.seed + epoch,
            micro_batch=args.micro_batch * 2,
        )
        epoch_secs = time.time() - t0

        metrics: OrderedDict[str, object] = OrderedDict()
        metrics["epoch"] = epoch
        metrics["epoch_seconds"] = float(epoch_secs)
        metrics["arm"] = arm
        # Which dataset variant the pairs came from; readers treat an absent
        # key as plain-variant. The train sbatch refuses to write into a root
        # whose bundles record a different variant.
        metrics["dataset_variant"] = args.dataset_variant
        metrics["lr"] = float(lr)
        # Recorded from the Foveation itself, so every setting eval has to
        # reproduce travels with the checkpoint (script_utils.foveation_record).
        metrics["foveation"] = foveation_record(foveation)
        metrics["train"] = train_stats
        metrics["val"] = val_stats
        metrics["reproducibility"] = repro
        bundle = save_checkpoint_bundle(args.out, epoch=epoch, model=model,
                                        metrics=dict(metrics), optimizer=optimizer)
        print(f"\nepoch {epoch}: train LL={train_stats['mean_ll_bits_per_fix']:+.3f} b/fix  "
              f"val LL={val_stats.get('ll_bits_per_fix', float('nan')):+.3f}  "
              f"val IG={val_stats.get('ig_bits_per_fix', float('nan')):+.3f}  "
              f"({epoch_secs:.0f}s)  →  {bundle}")


if __name__ == "__main__":
    main()
