"""The two premises the per-stimulus gradient check rests on.

`train_one_epoch_foveated` decides whether to apply an optimiser step from the
total gradient norm returned by ``clip_grad_norm_`` rather than by running
``all(torch.isfinite(p.grad).all() for p in trainable)`` (380 CUDA synchronises
per stimulus). The two are only equivalent if:

1. the norm is non-finite exactly when some gradient is, and
2. passing ``max_norm=inf`` (the no-clipping case) leaves gradients untouched.

Both are properties of torch, not of this codebase, so they are worth pinning:
if a future torch changes either, training would silently start applying steps
it should skip, or silently start scaling gradients it should leave alone.
"""
from __future__ import annotations

import math

import pytest
import torch


def _params(values):
    ps = []
    for v in values:
        p = torch.nn.Parameter(torch.zeros(2))
        p.grad = torch.tensor(v, dtype=torch.float32)
        ps.append(p)
    return ps


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_norm_is_nonfinite_when_any_gradient_is(bad):
    """One bad entry among many good ones must poison the norm."""
    ps = _params([[1.0, 2.0], [3.0, 4.0], [bad, 0.0]])
    norm = torch.nn.utils.clip_grad_norm_(ps, 1.0)
    assert not torch.isfinite(norm), f"{bad} in a gradient left the norm finite"


def test_norm_is_finite_when_all_gradients_are():
    ps = _params([[3.0, 4.0], [0.0, 0.0]])
    norm = torch.nn.utils.clip_grad_norm_(ps, float("inf"))
    assert torch.isfinite(norm)
    assert norm.item() == pytest.approx(5.0)


def test_infinite_max_norm_does_not_touch_gradients():
    """The no-clipping path must stay bit-exact, not merely close."""
    before = [[0.5, -2.0], [1e-7, 3e4]]
    ps = _params(before)
    torch.nn.utils.clip_grad_norm_(ps, float("inf"))
    for p, orig in zip(ps, before):
        assert torch.equal(p.grad, torch.tensor(orig)), "inf max_norm rescaled a gradient"


def test_finite_max_norm_still_clips():
    """Guard the other direction: the change must not have disabled clipping."""
    ps = _params([[3.0, 4.0]])          # norm 5
    norm = torch.nn.utils.clip_grad_norm_(ps, 1.0)
    assert norm.item() == pytest.approx(5.0)          # returns the PRE-clip norm
    assert float(ps[0].grad.norm()) == pytest.approx(1.0, abs=1e-5)


def _gather_targets_loop(log_d, target_x, target_y):
    """The original per-sample Python loop, kept here as the reference."""
    import numpy as np

    H, W = log_d.shape[-2:]
    return torch.stack([
        log_d[b,
              int(np.clip(round(float(target_y[b])), 0, H - 1)),
              int(np.clip(round(float(target_x[b])), 0, W - 1))]
        for b in range(log_d.size(0))
    ])


def test_vectorised_gather_picks_the_same_pixels():
    """Batched indexing must select exactly what the loop selected, ties included.

    Out-of-frame targets (clipping) and exact .5 coordinates (rounding rule) are
    the two places a rewrite like this silently drifts, so both are covered.
    """
    import numpy as np

    from tez_deepgaze.foveated_train import _gather_targets

    H, W, B = 37, 53, 12
    rng = np.random.RandomState(0)
    torch.manual_seed(0)
    log_d = torch.randn(B, H, W)

    tx = list(rng.uniform(-5, W + 5, B))          # deliberately out of frame
    ty = list(rng.uniform(-5, H + 5, B))
    assert torch.equal(_gather_targets(log_d, tx, ty), _gather_targets_loop(log_d, tx, ty))

    half = [0.5, 1.5, 2.5, 3.5]                   # round-half-to-even, both ways
    assert torch.equal(_gather_targets(log_d[:4], half, half),
                       _gather_targets_loop(log_d[:4], half, half))


def test_vectorised_gather_still_carries_gradients():
    """One gradient per selected pixel — the loss depends on nothing else."""
    import numpy as np

    from tez_deepgaze.foveated_train import _gather_targets

    H, W, B = 20, 24, 6
    rng = np.random.RandomState(1)
    log_d = torch.randn(B, H, W, requires_grad=True)
    _gather_targets(log_d, list(rng.uniform(0, W, B)), list(rng.uniform(0, H, B))).sum().backward()
    assert int((log_d.grad != 0).sum()) == B


def test_stacked_readback_carries_all_three_values():
    """The single-transfer read used per stimulus must round-trip loss/flag/norm."""
    nll = torch.tensor(12.25, dtype=torch.float64)
    finite = torch.tensor(True)
    gn = torch.tensor(0.75)
    stats = torch.stack([nll, (finite & torch.isfinite(gn)).to(torch.float64),
                         gn.to(torch.float64)]).cpu()
    assert float(stats[0]) == 12.25
    assert bool(stats[1]) is True
    assert float(stats[2]) == 0.75

    stats = torch.stack([nll, (finite & torch.isfinite(torch.tensor(math.nan))).to(torch.float64),
                         torch.tensor(math.nan, dtype=torch.float64)]).cpu()
    assert bool(stats[1]) is False
