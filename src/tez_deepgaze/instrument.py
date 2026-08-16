"""Instrumentation: forward-pass helpers, scanpath sampler, checkpoint bundles.

This module is the shared low-level interface to DG3:

- ``compute_log_density`` / ``compute_log_density_batch`` wrap a single
  forward pass and handle NaN-padding of the scanpath history to
  ``model.included_fixations`` length.
- ``sample_scanpath`` is an autoregressive sampler used for demo figures.
- ``save_checkpoint_bundle`` defines the on-disk layout that training
  writes per epoch.

Everything downstream — baseline evaluation, foveated vs. non-foveated
comparisons, training checkpoints — goes through these helpers, so the
batching and padding logic lives in exactly one place.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from pysaliency.models import sample_from_logdensity


@dataclass
class SampledScanpath:
    """A single sampled scanpath on one image."""
    seed: int
    x: list[float]
    y: list[float]


def _fixation_history(x: Sequence[float], y: Sequence[float], included):
    hist_x, hist_y = [], []
    for idx in included:
        try:
            hist_x.append(x[idx])
            hist_y.append(y[idx])
        except IndexError:
            hist_x.append(np.nan)
            hist_y.append(np.nan)
    return hist_x, hist_y


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    """Promote a grayscale HxW array to HxWx3 (MIT1003 has a few gray stimuli)."""
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    return image


def image_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    """(1, C, H, W) float32 tensor from an HxWx3 image array."""
    return torch.tensor(image.transpose(2, 0, 1)[None], dtype=torch.float32, device=device)


@torch.no_grad()
def compute_log_density(
    model,
    image: np.ndarray,            # HxWx3 uint8
    centerbias: np.ndarray,       # HxW log-density
    fixation_x: Sequence[float],
    fixation_y: Sequence[float],
    device: torch.device,
    foveation=None,               # optional foveate_input.Foveation
    foveation_stack: torch.Tensor | None = None,
) -> np.ndarray:
    """Return HxW log-density prediction conditioned on the given scanpath history.

    With ``foveation`` the input image is foveated around the current (most
    recent) fixation before the forward pass — gaze-contingent input: the model
    sees the scene as an eye fixating there would (sharp at gaze, lower
    resolution toward the periphery, periphery never zeroed). The scanpath
    history passed to the model is unchanged; only the pixels the backbone
    sees are re-centred. ``foveation_stack`` optionally reuses a blur pyramid
    from ``foveation.blur_stack`` built from this same image.
    """
    return compute_log_density_batch(
        model, image, centerbias, [list(fixation_x)], [list(fixation_y)], device,
        foveation=foveation, foveation_stack=foveation_stack,
    )[0]


def _log_density_batch_core(
    model,
    image: np.ndarray,            # HxWx3 uint8
    centerbias: np.ndarray,       # HxW log-density
    hist_x_batch: list[Sequence[float]],
    hist_y_batch: list[Sequence[float]],
    device: torch.device,
    foveation=None,               # optional foveate_input.Foveation
    foveation_stack: torch.Tensor | None = None,
) -> torch.Tensor:
    """Forward pass shared by the no-grad and gradient-enabled batch callers.

    Evaluates many histories on the SAME image in one forward pass. Pads
    histories with NaN so they share length. Returns a (B, H, W) tensor
    (sliced from the model's (B, 1, H, W) output) with gradients following
    whatever context the caller is in — this function has no ``no_grad``
    of its own, so :func:`compute_log_density_batch` wraps it for inference
    and :mod:`foveated_train` calls it directly to get gradients for the
    readout heads.

    With ``foveation`` every history is scored on the SAME source image
    foveated around its OWN current fixation (the most recent history entry),
    so the backbone sees a sharp-at-gaze, low-resolution-periphery view
    centred where that scanpath currently is. The blur pyramid is built once
    and shared across the batch (:meth:`Foveation.foveate_shared_image`);
    ``foveation_stack`` optionally reuses a pyramid built earlier from this
    same image via ``foveation.blur_stack``.
    """
    B = len(hist_x_batch)
    image_t = image_tensor(image, device)
    # Every history is scored on the SAME picture whenever the input does not
    # depend on the history, so B copies of it go through the frozen DenseNet-201
    # to produce B identical feature maps. The hoisted forward accepts a single
    # shared image and runs the backbone once instead (see fast_dg3), which is
    # only safe when that forward is installed — hence the capability check
    # rather than an unconditional shortcut. The result is unchanged either way.
    #
    # Two arms qualify: the sharp one, and the fixed-centre foveated ablation,
    # where the fovea is pinned to the image centre and so every fixation on a
    # stimulus sees the identical foveated picture. Only gaze-contingent
    # foveation genuinely differs per history.
    same_image_for_every_history = foveation is None or foveation.center_fovea
    shared = (same_image_for_every_history and B > 1
              and getattr(model, "accepts_shared_image", False))
    if foveation is None:
        image_b = image_t if shared else image_t.expand(B, -1, -1, -1).contiguous()
    else:
        # Fovea centre = current (most recent) fixation. An empty history (e.g.
        # start_fixation=0, predicting the first fixation with no prior gaze)
        # has no current fixation, so fall back to image centre — where each
        # trial started the observer's gaze — instead of raising on hx[-1].
        H, W = image.shape[:2]
        cx, cy = W / 2.0, H / 2.0
        if foveation.center_fovea:
            # Ablation arm: pin the fovea to the image centre for every step, so
            # the input no longer changes with gaze. Same blur profile, same
            # everything else — the only difference is gaze-contingency. All B
            # centres coincide, so on the shared path one is enough and the
            # blend, like the backbone, runs once.
            n = 1 if shared else B
            fx = torch.full((n,), cx, dtype=torch.float32, device=device)
            fy = torch.full((n,), cy, dtype=torch.float32, device=device)
        else:
            fx = torch.tensor([float(hx[-1]) if len(hx) else cx for hx in hist_x_batch],
                              dtype=torch.float32, device=device)
            fy = torch.tensor([float(hy[-1]) if len(hy) else cy for hy in hist_y_batch],
                              dtype=torch.float32, device=device)
        image_b = foveation.foveate_shared_image(
            image_t, fx, fy, stack=foveation_stack
        )  # (B, C, H, W)
    cb_t = torch.tensor(centerbias[None], dtype=torch.float32, device=device)
    if not shared:
        cb_t = cb_t.expand(B, -1, -1).contiguous()

    included = model.included_fixations
    L = len(included)
    xh = np.full((B, L), np.nan, dtype=np.float32)
    yh = np.full((B, L), np.nan, dtype=np.float32)
    for b in range(B):
        hx, hy = _fixation_history(hist_x_batch[b], hist_y_batch[b], included)
        xh[b] = hx
        yh[b] = hy
    xh_t = torch.from_numpy(xh).to(device)
    yh_t = torch.from_numpy(yh).to(device)

    log_density = model(image_b, cb_t, xh_t, yh_t)
    return log_density[:, 0]


@torch.no_grad()
def compute_log_density_batch(
    model,
    image: np.ndarray,            # HxWx3 uint8
    centerbias: np.ndarray,       # HxW log-density
    hist_x_batch: list[Sequence[float]],
    hist_y_batch: list[Sequence[float]],
    device: torch.device,
    foveation=None,               # optional foveate_input.Foveation
    foveation_stack: torch.Tensor | None = None,
) -> np.ndarray:
    """Evaluate many histories on the SAME image in one forward pass.

    Returns BxHxW log-density. Pads histories with NaN so they share length.
    With ``foveation`` each history is scored on the image foveated
    gaze-contingently around its own current fixation; see
    :func:`_log_density_batch_core`.
    """
    log_density = _log_density_batch_core(
        model, image, centerbias, hist_x_batch, hist_y_batch, device,
        foveation=foveation, foveation_stack=foveation_stack,
    )
    return log_density.detach().cpu().numpy()


def sample_scanpath(
    model,
    image: np.ndarray,
    centerbias: np.ndarray,
    start_xy: tuple[float, float],
    n_fixations: int,
    device: torch.device,
    seed: int = 0,
    foveation=None,               # optional foveate_input.Foveation
    foveation_stack=None,
) -> SampledScanpath:
    """Autoregressively sample a scanpath from a pretrained DeepGaze III model.

    With ``foveation`` the input image is re-foveated around the current
    fixation before every forward pass, so the model chooses its next saccade
    from a sharp-at-gaze, low-resolution-but-present peripheral view. The blur
    pyramid is built once and reused across steps. The sampled path differs
    from the unfoveated one because the log-density differs.

    ``foveation_stack`` accepts a pyramid already built for this image, which
    matters when drawing several samples from the same stimulus: the stack
    depends only on the image, and building it is ~1.7 s at full resolution, so
    rebuilding it per sample would dominate the cost of a sampling pass.
    """
    rst = np.random.RandomState(seed)
    xs = [float(start_xy[0])]
    ys = [float(start_xy[1])]
    stack = foveation_stack
    if foveation is not None and stack is None:
        stack = foveation.blur_stack(image_tensor(image, device))
    for _ in range(n_fixations):
        log_d = compute_log_density(
            model, image, centerbias, xs, ys, device,
            foveation=foveation, foveation_stack=stack,
        )
        nx, ny = sample_from_logdensity(log_d, rst=rst)
        xs.append(float(nx))
        ys.append(float(ny))
    return SampledScanpath(seed=seed, x=xs, y=ys)


# Submodules that head training freezes (see foveated_train.freeze_for_head_training
# for the rationale). The checkpoint format is defined by this list: weights.pt
# omits everything under these names, so the loader must not demand those tensors
# back — even from a model that was never frozen, like the one eval builds.
FROZEN_SUBMODULES = ("features", "finalizers")


def trainable_state_dict(model: torch.nn.Module) -> dict:
    """The ``state_dict`` entries that training can actually change.

    Only the read-out heads are unfrozen, and the backbone runs in ``eval`` mode
    so its BatchNorm buffers do not move either. Everything else in a full
    ``state_dict`` is therefore a byte-identical copy of the pretrained weights.
    """
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v for k, v in model.state_dict().items() if k in trainable}


def load_checkpoint_weights(
    model: torch.nn.Module, path: Path, device=None
) -> str:
    """Load a ``weights.pt`` into ``model``, accepting either checkpoint format.

    Returns ``"full"`` or ``"trainable"`` to say which was found.

    A checkpoint holds either a complete ``state_dict`` (~79 MB, 98.5 % of it a
    copy of the frozen backbone) or only the trainable read-out tensors (~1 MB).
    A partial dict is applied on top of whatever the model already holds, so
    the caller must have loaded the pretrained weights first — which is what
    makes the two formats equivalent.

    The completeness check exempts tensors under :data:`FROZEN_SUBMODULES`:
    training freezes those, so new-format checkpoints legitimately omit them,
    and ``model`` here may be a fresh (never frozen) instance whose
    ``requires_grad`` flags say nothing about what training could change.
    """
    sd = torch.load(path, map_location=device or "cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        raise ValueError(f"{path}: unexpected keys in checkpoint: {sorted(unexpected)[:5]}")
    required = {
        n for n, p in model.named_parameters()
        if p.requires_grad and n.split(".", 1)[0] not in FROZEN_SUBMODULES
    }
    if not required.issubset(set(sd)):
        raise ValueError(
            f"{path}: checkpoint is missing trainable tensors "
            f"{sorted(required - set(sd))[:5]} — not a usable read-out checkpoint"
        )
    return "trainable" if missing else "full"


def save_checkpoint_bundle(
    outdir: Path,
    *,
    epoch: int,
    model: torch.nn.Module,
    metrics: dict,
    optimizer: torch.optim.Optimizer | None = None,
) -> Path:
    """Save read-out weights + metrics, and optimizer state when given.

    Layout:
        outdir/epoch_XXX/
            weights.pt      trainable read-out tensors (see trainable_state_dict)
            metrics.json
            optimizer.pt    Adam moment estimates + step count, if `optimizer`

    ``weights.pt`` holds only the trainable tensors. Read it back with
    :func:`load_checkpoint_weights`, which also accepts a full-model
    ``state_dict``.

    ``optimizer.pt`` is what makes a later ``--start-epoch`` resume a true
    continuation rather than a warm restart: Adam's per-weight moment estimates
    are built up over training and are not recoverable from the weights alone.
    It costs ~2 x the trainable parameter count (a couple of MB), so there is no
    reason not to write it.
    """
    bundle = outdir / f"epoch_{epoch:03d}"
    bundle.mkdir(parents=True, exist_ok=True)

    torch.save(trainable_state_dict(model), bundle / "weights.pt")
    (bundle / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if optimizer is not None:
        torch.save(optimizer.state_dict(), bundle / "optimizer.pt")

    return bundle
