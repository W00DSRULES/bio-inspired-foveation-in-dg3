"""Unit tests for the gaze-contingent foveation blur (foveate_input).

These exercise the space-variant blur only — no DeepGaze model — so they run
fast and without loading weights. They codify the invariants the mechanism
relies on: shape preservation, a low-resolution-but-present (never black)
periphery, a partition-of-unity blend, the batched-single-image path matching
per-sample forward, monotonic acuity falloff, and viewing-geometry (ppd) control.
"""
from __future__ import annotations

import numpy as np
import torch

from tez_deepgaze.foveate_input import Foveation


def _rand_image(H=192, W=256, lo=40.0, hi=210.0, seed=0):
    # Larger than the deepest pyramid kernel (~145 px for σ=24) — the module
    # targets screen-sized images (MIT1003 ~768×1024); tiny frames can't hold
    # the reflect padding for the coarsest level.
    rng = np.random.RandomState(seed)
    img = lo + (hi - lo) * rng.rand(1, 3, H, W).astype(np.float32)
    return torch.from_numpy(img)


def test_forward_shape_preserved():
    img = _rand_image()
    out = Foveation(ppd=35.0, foveal_cpd=20.0)(img, torch.tensor([48.0]), torch.tensor([32.0]))
    assert out.shape == img.shape
    assert torch.isfinite(out).all()


def test_periphery_low_resolution_but_not_black():
    # Foveation is a convex blend of blurred copies (weights sum to 1) and each
    # blur preserves the local mean with reflect padding, so a strictly-positive
    # image stays strictly positive everywhere — the periphery is dimmer in
    # detail, never zeroed.
    img = _rand_image(lo=40.0, hi=210.0)
    out = Foveation(ppd=35.0, foveal_cpd=8.0)(img, torch.tensor([48.0]), torch.tensor([32.0]))
    assert float(out.min()) > 30.0            # nowhere near black
    assert float(out.max()) <= float(img.max()) + 1e-3
    assert float(out.min()) >= float(img.min()) - 1e-3


def test_blend_weights_are_partition_of_unity():
    fov = Foveation(ppd=35.0, foveal_cpd=12.0)
    e_px = torch.linspace(0, 600, 200)
    level = fov.fractional_level(e_px)
    wsum = sum((1.0 - (level - k).abs()).clamp(min=0.0) for k in range(fov.n_levels))
    assert torch.allclose(wsum, torch.ones_like(wsum), atol=1e-6)


def test_foveate_shared_image_matches_forward():
    # The blur-once batched path must equal stacking forward over each centre.
    img = _rand_image()
    fov = Foveation(ppd=35.0, foveal_cpd=12.0)
    fx = torch.tensor([10.0, 48.0, 90.0])
    fy = torch.tensor([12.0, 32.0, 60.0])
    out_forward = fov(img.expand(3, -1, -1, -1).contiguous(), fx, fy)
    out_shared = fov.foveate_shared_image(img, fx, fy)
    assert torch.allclose(out_forward, out_shared, atol=1e-5)


def test_acuity_falloff_is_monotonic():
    # High-frequency texture: local detail (std) must decrease with eccentricity.
    rng = np.random.RandomState(1)
    H, W = 200, 200
    img = torch.from_numpy((rng.rand(1, 3, H, W) * 255).astype(np.float32))
    out = Foveation(ppd=35.0, foveal_cpd=10.0)(img, torch.tensor([100.0]), torch.tensor([100.0]))[0, 0]

    def local_std(cy, cx, r=8):
        return float(out[cy - r:cy + r, cx - r:cx + r].std())

    fovea = local_std(100, 100)
    mid = local_std(100, 150)
    periph = local_std(100, 195)
    assert fovea > mid > periph


def test_ppd_controls_blur_strength():
    # Higher ppd → a fixed pixel offset is fewer degrees but a lower cyc/px
    # cutoff, i.e. a higher pyramid level (more blur). foveal_cpd fixed.
    e_px = torch.tensor([150.0])
    lvl_35 = float(Foveation(ppd=35.0, foveal_cpd=20.0).fractional_level(e_px))
    lvl_70 = float(Foveation(ppd=70.0, foveal_cpd=20.0).fractional_level(e_px))
    assert lvl_70 > lvl_35 > 0.0


def test_fovea_sharp_above_display_nyquist():
    # At e=0 the target cutoff is foveal_cpd cyc/deg. If foveal_cpd >= 0.5*ppd
    # (display Nyquist) the fovea is fully sharp (level 0); below it, mildly
    # limited (small positive level).
    zero = torch.tensor([0.0])
    sharp = Foveation(ppd=35.0, foveal_cpd=20.0).fractional_level(zero)  # 20 > 17.5
    limited = Foveation(ppd=70.0, foveal_cpd=20.0).fractional_level(zero)  # 20 < 35
    assert float(sharp) == 0.0
    assert float(limited) > 0.0


def test_center_fovea_defaults_off_and_is_recorded():
    """Gaze-contingent is the default; the ablation must be opted into."""
    assert Foveation().center_fovea is False
    assert Foveation(center_fovea=True).center_fovea is True


def test_center_fovea_ignores_gaze():
    """With center_fovea the foveated input must not depend on the fixation.

    That is the whole definition of the fixed-centre ablation: identical blur
    profile, but the picture the backbone sees stops changing as gaze moves.
    """
    import numpy as np

    from tez_deepgaze.instrument import _log_density_batch_core

    class _Echo(torch.nn.Module):
        """Stands in for DG3: returns the foveated input so we can inspect it."""
        included_fixations = [-1, -2, -3, -4]

        def forward(self, x, centerbias, x_hist=None, y_hist=None, durations=None):
            self.seen = x.detach().clone()
            return x.mean(dim=1, keepdim=True)

    rng = np.random.RandomState(0)
    image = rng.randint(0, 256, (96, 128, 3)).astype(np.uint8)
    cb = np.zeros((96, 128), dtype=np.float32)
    hx = [[10.0, 12.0], [110.0, 100.0]]      # two very different gaze points
    hy = [[10.0, 12.0], [80.0, 70.0]]
    dev = torch.device("cpu")

    m = _Echo()
    _log_density_batch_core(m, image, cb, hx, hy, dev,
                            foveation=Foveation(ppd=35.0, foveal_cpd=10.0, center_fovea=True))
    centred = m.seen
    assert torch.equal(centred[0], centred[1]), \
        "center_fovea must give every sample the same image regardless of gaze"

    _log_density_batch_core(m, image, cb, hx, hy, dev,
                            foveation=Foveation(ppd=35.0, foveal_cpd=10.0, center_fovea=False))
    gazed = m.seen
    assert not torch.equal(gazed[0], gazed[1]), \
        "gaze-contingent foveation must differ between distinct fixations"
