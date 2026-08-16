"""A drop-in faster forward for ``DeepGazeIIIMixture``, mathematically unchanged.

``deepgaze_pytorch.DeepGazeIII(pretrained=True)`` is a **10-member mixture**: one
shared frozen DenseNet backbone, then ten independent read-out stacks whose
log-densities are averaged in log space. The vendor forward loops over those ten
members and, inside the loop, calls::

    scanpath_features = encode_scanpath_features(x_hist, y_hist, size=(H, W), ...)
    scanpath_features = F.interpolate(scanpath_features, readout_shape)

Neither call depends on the loop variable. ``encode_scanpath_features`` is a
function of ``x_hist``, ``y_hist`` and the image size alone, so the identical
tensor is rebuilt **ten times per forward** — and it is not a cheap tensor: it
materialises ``(B, 4, H, W)`` grids for XS and YS, squares both, adds them, takes
a square root, and concatenates to ``(B, 12, H, W)`` — all at *full* image
resolution, only to be immediately downsampled to ``readout_shape`` (a 64x
reduction in pixels at 1024x768).

Hoisting both calls above the loop computes them once. The arithmetic is
untouched: same inputs, same function, same result, reused instead of recomputed.
:func:`assert_forward_matches` checks that against the vendor implementation.

Applied explicitly by the caller, never on import, so a run that does not ask for
it behaves exactly as before::

    model = deepgaze_pytorch.DeepGazeIII(pretrained=True)
    n = apply_fast_forward(model)      # returns how many modules were patched
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from deepgaze_pytorch.layers import LayerNorm as DGLayerNorm
from deepgaze_pytorch.modules import DeepGazeIIIMixture, encode_scanpath_features


def _fast_mixture_forward(self, x, centerbias, x_hist=None, y_hist=None, durations=None):
    """``DeepGazeIIIMixture.forward`` with the loop-invariant work hoisted out.

    Also accepts a **single shared image** with many histories: pass ``x`` with
    batch 1 alongside ``x_hist``/``y_hist`` of batch B and the frozen backbone
    runs once instead of B times, its output broadcast across the batch. That is
    exactly the sharp (non-foveated) arm, where every sample of a micro-batch is
    the same picture — the vendor path materialises B identical copies and pushes
    all of them through DenseNet-201. Passing B copies still works and gives the
    same answer; this is an opt-in shortcut, not a behaviour change.
    """
    orig_shape = x.shape
    n_hist = x_hist.shape[0] if x_hist is not None else orig_shape[0]
    shared_image = orig_shape[0] == 1 and n_hist > 1
    if not shared_image and x_hist is not None and orig_shape[0] != n_hist:
        raise ValueError(
            f"image batch {orig_shape[0]} and history batch {n_hist} disagree; "
            "pass either matching batches or a single shared image"
        )

    x = F.interpolate(x, scale_factor=1 / self.downsample, recompute_scale_factor=False)
    x = self.features(x)

    readout_shape = [
        math.ceil(orig_shape[2] / self.downsample / self.readout_factor),
        math.ceil(orig_shape[3] / self.downsample / self.readout_factor),
    ]
    x = [F.interpolate(item, readout_shape) for item in x]
    x = torch.cat(x, dim=1)
    if shared_image:
        # .contiguous() is deliberate. A bare .expand() is a free stride-0 view,
        # but feeding that into the read-out convolutions takes a different
        # summation path than a dense tensor and shifts the result by ~5e-6 —
        # small, yet enough to make the shared-image path non-reproducible
        # against the replicated one. Materialising the copy costs memory at the
        # read-out resolution (~1/64 of the image) and still skips the B-1
        # redundant DenseNet-201 passes, which is where the time actually goes.
        x = x.expand(n_hist, -1, -1, -1).contiguous()
        if centerbias.shape[0] == 1:
            centerbias = centerbias.expand(n_hist, -1, -1).contiguous()
    readout_input = x

    # --- the whole point: computed once, not once per ensemble member ---
    scanpath_features = None
    if any(sn is not None for sn in self.scanpath_networks):
        scanpath_features = encode_scanpath_features(
            x_hist, y_hist, size=(orig_shape[2], orig_shape[3]), device=readout_input.device
        )
        scanpath_features = F.interpolate(scanpath_features, readout_shape)

    predictions = []
    for saliency_network, scanpath_network, fixation_selection_network, finalizer in zip(
        self.saliency_networks, self.scanpath_networks,
        self.fixation_selection_networks, self.finalizers,
    ):
        x = saliency_network(readout_input)
        y = scanpath_network(scanpath_features) if scanpath_network is not None else None
        x = fixation_selection_network((x, y))
        x = finalizer(x, centerbias)
        predictions.append(x[:, np.newaxis, :, :])

    predictions = torch.cat(predictions, dim=1) - np.log(len(self.saliency_networks))
    return predictions.logsumexp(dim=(1), keepdim=True)


def apply_fast_forward(model: torch.nn.Module) -> int:
    """Bind the hoisted forward to every ``DeepGazeIIIMixture`` in ``model``.

    Returns the number of modules patched — 0 means the model was not a mixture
    and nothing changed, which is not an error but is worth logging.
    """
    n = 0
    targets = [model] if isinstance(model, DeepGazeIIIMixture) else []
    targets += [m for m in model.modules() if isinstance(m, DeepGazeIIIMixture) and m is not model]
    for m in targets:
        m.forward = _fast_mixture_forward.__get__(m, type(m))
        n += 1
    # Marker for callers deciding whether to hand over one shared image or B
    # copies: only the hoisted forward understands the batch-1 shorthand.
    model.accepts_shared_image = n > 0
    return n


def _fast_layer_norm_forward(self, input):
    """``deepgaze_pytorch.layers.LayerNorm.forward`` without the materialisation.

    The vendor version is the single largest cost in a training step — 58 % of
    GPU time in an operator-level profile. Two reasons, both incidental to the
    mathematics:

    1. It broadcasts the per-channel ``weight`` and ``bias`` by hand, with two
       nested ``repeat_interleave`` calls that build a full ``(C, H, W)`` copy of
       each on every forward. At the read-out's 2048x96x128 that is 100 MB per
       parameter, so 200 MB written and read back per layer per ensemble member,
       purely to express a broadcast.
    2. It then calls ``F.layer_norm`` with ``normalized_shape = (C, H, W)``, i.e.
       one statistic per sample. cuDNN's layer-norm kernel parallelises over
       rows, and there are only ``B`` of them — 36 blocks on a 108-SM A100, each
       reducing 25 M elements. Occupancy is a third of the card at best.

    Writing the same formula out — ``(x - mean) * rsqrt(var + eps) * w + b`` with
    the parameters left as broadcast views — reduces with torch's ordinary
    tree-reduction kernels, which use the whole card, and never materialises
    anything. Identical algebra, and ``layer_norm``'s biased variance.

    Not bit-identical: the reductions sum in a different order, so results differ
    in the last few significant bits. :func:`layer_norm_max_diff` measures that
    on real shapes so the size of the change is a number, not a hope.

    Written as a single ``addcmul`` on purpose. The obvious spelling,
    ``(x - mean) * rstd * w + b``, is three chained elementwise ops, so autograd
    keeps three full-size intermediates and the step runs out of memory at the
    micro-batch the card otherwise fits. Folding the constants into a per-sample
    per-channel ``scale`` and ``shift`` — both ``(B, C, 1, 1)``, a few kilobytes —
    leaves exactly one full-size tensor, the output, which is what the fused
    kernel also keeps.
    """
    C = self.features
    # unbiased=False is what F.layer_norm uses.
    var, mean = torch.var_mean(input, dim=(1, 2, 3), keepdim=True, unbiased=False)
    scale = torch.rsqrt(var + self.eps)
    if self.weight is not None:
        scale = scale * self.weight.view(1, C, 1, 1)
    shift = -mean * scale
    if self.bias is not None:
        shift = shift + self.bias.view(1, C, 1, 1)
    return torch.addcmul(shift, input, scale)


def apply_fast_layernorm(model: torch.nn.Module) -> int:
    """Bind :func:`_fast_layer_norm_forward` to every DG3 ``LayerNorm`` in ``model``.

    Returns how many modules were patched. Covers ``LayerNormMultiInput`` too,
    which owns ordinary ``LayerNorm`` children and delegates to them.
    """
    n = 0
    for m in model.modules():
        if isinstance(m, DGLayerNorm):
            m.forward = _fast_layer_norm_forward.__get__(m, type(m))
            n += 1
    return n


@torch.no_grad()
def layer_norm_max_diff(features: int, shape: tuple[int, int, int], device) -> float:
    """Largest absolute disagreement between the two layer-norm forwards.

    ``shape`` is ``(B, H, W)``. Uses a random input with a realistic scale rather
    than a toy one, because the disagreement is relative to the magnitudes
    involved.
    """
    B, H, W = shape
    ln = DGLayerNorm(features).to(device)
    torch.nn.init.normal_(ln.weight, mean=1.0, std=0.1)
    torch.nn.init.normal_(ln.bias, mean=0.0, std=0.1)
    x = torch.randn(B, features, H, W, device=device)

    ref = DGLayerNorm.forward(ln, x)
    got = _fast_layer_norm_forward(ln, x)
    return float((ref - got).abs().max())


@torch.no_grad()
def assert_forward_matches(model, image, centerbias, x_hist, y_hist, atol: float = 0.0) -> float:
    """Compare vendor and hoisted forwards on one input; return the max abs diff.

    ``atol=0.0`` demands bit-identical output, which is what hoisting a pure
    function out of a loop should give. Raises if the difference exceeds ``atol``.
    """
    mixtures = [m for m in ([model] + list(model.modules())) if isinstance(m, DeepGazeIIIMixture)]
    if not mixtures:
        raise TypeError("model contains no DeepGazeIIIMixture to compare")

    originals = [getattr(m, "forward", None) for m in mixtures]
    had_own = ["forward" in m.__dict__ for m in mixtures]
    for m, own in zip(mixtures, had_own):          # fall back to the class forward
        if own:
            del m.__dict__["forward"]
    ref = model(image, centerbias, x_hist, y_hist)

    for m in mixtures:
        m.forward = _fast_mixture_forward.__get__(m, type(m))
    got = model(image, centerbias, x_hist, y_hist)

    for m, orig, own in zip(mixtures, originals, had_own):   # restore what was there
        if own:
            m.forward = orig
        else:
            m.__dict__.pop("forward", None)

    diff = float((ref - got).abs().max())
    if diff > atol:
        raise AssertionError(
            f"hoisted forward differs from vendor forward by {diff:.3e} (atol={atol}); "
            "the two must agree — do not use the fast path"
        )
    return diff
