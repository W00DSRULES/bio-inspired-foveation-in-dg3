"""paired_stats: the fold-/image-paired arm comparison behind table.json's
`paired_vs_normal` and the standalone `--paired-from` recompute."""
from __future__ import annotations

import math

import numpy as np
import pytest

from tez_deepgaze.script_utils import paired_stats


def _arms(offsets_by_fold):
    """Two arms over shared stimuli; arm B = arm A + a per-image offset."""
    arms = {"normal": [], "foveated@40": []}
    for fold, offsets in enumerate(offsets_by_fold):
        stims = [f"s{fold}_{i}" for i in range(len(offsets))]
        base = [1.0 + 0.1 * i for i in range(len(offsets))]
        arms["normal"].append({"fold": fold, "stim": stims, "IG_bits": base})
        arms["foveated@40"].append(
            {"fold": fold, "stim": stims,
             "IG_bits": [b + o for b, o in zip(base, offsets)]})
    return arms


def test_fold_and_image_paired_stats_match_hand_computation():
    offsets = [[0.1, 0.3], [-0.2, 0.0], [0.2, 0.4]]  # fold means 0.2, -0.1, 0.3
    out = paired_stats(_arms(offsets))["foveated@40"]

    fold_means = [0.2, -0.1, 0.3]
    mean = float(np.mean(fold_means))
    se = float(np.std(fold_means, ddof=1) / math.sqrt(3))
    fp = out["fold_paired"]
    assert fp["mean"] == pytest.approx(mean)
    assert fp["se"] == pytest.approx(se)
    # The interval is mean +/- 2 SE, the one convention ch03 stats-plan sets.
    assert fp["interval_2se"][0] == pytest.approx(mean - 2 * se)
    assert fp["interval_2se"][1] == pytest.approx(mean + 2 * se)
    assert fp["n_folds"] == 3
    assert fp["negative_folds"] == 1

    flat = [o for fold in offsets for o in fold]
    ip = out["image_paired"]
    assert ip["mean"] == pytest.approx(np.mean(flat))
    assert ip["t"] == pytest.approx(
        np.mean(flat) / (np.std(flat, ddof=1) / math.sqrt(len(flat))))
    assert ip["n_images"] == len(flat)


def test_single_fold_has_no_interval():
    out = paired_stats(_arms([[0.1, 0.2, 0.3]]))["foveated@40"]
    assert out["fold_paired"]["se"] is None
    assert out["fold_paired"]["interval_2se"] is None
    assert out["image_paired"]["n_images"] == 3


def test_unpaired_stimuli_raise():
    arms = _arms([[0.1, 0.2]])
    arms["foveated@40"][0]["stim"] = ["other_a", "other_b"]
    with pytest.raises(ValueError, match="not paired"):
        paired_stats(arms)

