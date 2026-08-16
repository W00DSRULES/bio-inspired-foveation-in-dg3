"""The learning-rate schedule, and that a resume lands on the right rate.

`--start-epoch` resumes are part of the protocol, so the rate for epoch N has to
be the same whether the run reached N in one go or was restarted at N. That is
why the schedule is a function of the epoch number rather than stepped scheduler
state — a scheduler's position lives in a counter that a restart would have to
replay or reload, and getting it wrong is silent.
"""
from __future__ import annotations

import pytest
import torch

from tez_deepgaze.foveated_train import LR_MILESTONES, lr_for_epoch, set_epoch_lr
from tez_deepgaze.script_utils import common_stop_epoch

BASE = 3e-4  # protocol base rate

# LR_MILESTONES are absolute epochs: decay by 10x at each. Production trains 7
# epochs, so only the epoch-5 decay fires; the later milestones are checked too.
EXPECTED = {
    1: 3e-4, 2: 3e-4, 3: 3e-4, 4: 3e-4,
    5: 3e-5, 6: 3e-5, 7: 3e-5,
    8: 3e-6, 9: 3e-6, 10: 3e-6,
    11: 3e-7, 12: 3e-7,
}


@pytest.mark.parametrize("epoch,want", sorted(EXPECTED.items()))
def test_schedule_matches_the_protocol(epoch, want):
    assert lr_for_epoch(epoch, BASE) == pytest.approx(want, rel=1e-12)


def test_rate_never_rises():
    rates = [lr_for_epoch(e, BASE) for e in range(1, 13)]
    assert all(b <= a for a, b in zip(rates, rates[1:]))
    assert len(set(rates)) == len(LR_MILESTONES) + 1


def test_resume_gets_the_same_rate_as_an_uninterrupted_run():
    """Epoch-for-epoch equality between one 1..12 run and a 1..6 + 7..12 pair.

    Checked on the optimizer itself, not just on `lr_for_epoch`, because what
    matters is the rate the param groups actually carry into the step.
    """
    def rates(start: int, end: int) -> dict[int, float]:
        # A resumed run builds a fresh Adam at the base LR, exactly as main()
        # does before optionally loading optimizer.pt.
        opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=BASE)
        out = {}
        for epoch in range(start, end + 1):
            out[epoch] = set_epoch_lr(opt, epoch, BASE)
            assert opt.param_groups[0]["lr"] == out[epoch]
        return out

    uninterrupted = rates(1, 12)
    resumed = rates(7, 12)
    assert resumed == {e: uninterrupted[e] for e in range(7, 13)}
    # A resume that starts on a milestone is the case a replayed scheduler gets
    # wrong most easily.
    assert rates(5, 5)[5] == uninterrupted[5]
    assert rates(11, 11)[11] == uninterrupted[11]


def test_every_param_group_is_updated():
    p, q = torch.nn.Parameter(torch.zeros(1)), torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([{"params": [p]}, {"params": [q]}], lr=BASE)
    lr = set_epoch_lr(opt, 8, BASE)
    assert [g["lr"] for g in opt.param_groups] == [lr, lr]


# --- common stopping epoch ------------------------------------------------


def _rising(start: float, deltas: list[float]) -> dict[int, float]:
    """Epoch -> val IG, built from a start value and per-epoch improvements."""
    out, v = {1: start}, start
    for i, d in enumerate(deltas, start=2):
        v += d
        out[i] = v
    return out


def test_stop_epoch_is_the_first_where_every_arm_has_plateaued():
    # arm A flattens after epoch 3; arm B not until epoch 5. The rule stops all
    # arms together, so the answer is B's epoch, not A's.
    curves = {
        "a": _rising(1.50, [0.020, 0.0002, 0.0002, 0.0001, 0.0001]),
        "b": _rising(1.50, [0.020, 0.0200, 0.0200, 0.0001, 0.0001]),
    }
    assert common_stop_epoch(curves) == 6


def test_stop_epoch_is_none_when_an_arm_is_still_rising():
    """No arm has plateaued: report the final epoch and say so."""
    curves = {
        "a": _rising(1.50, [0.002, 0.002, 0.002, 0.002]),
        "b": _rising(1.50, [0.002, 0.002, 0.002, 0.002]),
    }
    assert common_stop_epoch(curves) is None


def test_stop_epoch_ignores_epochs_an_arm_is_missing():
    """Arms are only comparable where all of them have a value."""
    curves = {
        "a": _rising(1.50, [0.0001, 0.0001, 0.0001]),
        "b": {1: 1.50, 2: 1.5001, 3: 1.5002},        # no epoch 4
    }
    assert common_stop_epoch(curves) == 3


def test_stop_epoch_needs_two_consecutive_flat_gaps():
    """One flat epoch is not a plateau."""
    curves = {"a": _rising(1.50, [0.0001, 0.020, 0.0001, 0.0001])}
    assert common_stop_epoch(curves) == 5           # not epoch 2
