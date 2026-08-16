"""Evaluation-pipeline invariants pinned on synthetic data — no corpus, no DG3.

Three things share one synthetic corpus here because they share one risk: code
that feeds reported numbers but was only ever exercised by full cluster runs.

1. `eval_one_epoch_foveated` (the per-epoch val loop arms are selected on) must
   agree with `evaluate.run` (the loop the reported tables come from). The two
   are separate implementations over the same forward core; drift between them
   means model selection and the reported table disagree silently.
2. `evaluate.run`'s foveated path sorts scanpaths by stimulus and reuses one
   blur pyramid per image. If the caching broke, results would stay right but
   runs would slow ~15x; if the sort broke, the cache would silently rebuild.
3. `baseline.py`'s schema-v2 writer produces the artefact the thesis quotes;
   before this file its key set was only checked on gpu2.

The stub models are content-dependent (their log-density is a function of the
input pixels), so the foveated comparisons genuinely test that both paths blur
the same picture the same way — a content-blind stub would pass with the
foveation broken.
"""
from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from tez_deepgaze.centerbias import load_centerbias_for_image
from tez_deepgaze.evaluate import run as evaluate_run
from tez_deepgaze.foveate_input import Foveation
from tez_deepgaze.foveated_train import eval_one_epoch_foveated

LOG2 = math.log(2.0)

# Big enough for the deepest pyramid level's reflect padding (~145 px kernel),
# small enough that a CPU foveation pass stays fast.
H, W = 192, 256

# Two stimuli, five scanpaths. Lengths 5/2/1 on stim 0 and 4/3 on stim 1: the
# length-1 path must be skipped by BOTH loops (evaluate.run's valid_end check
# and _build_pairs_for_image's MIN_PATH_LEN), and 100.5 pins that the two
# nearest-pixel lookups round half-to-even identically.
_SCANPATHS = [
    (0, [10.0, 100.5, 30.0, 200.0, 255.6], [12.0, 50.0, 100.0, 191.4, 60.0]),
    (0, [40.0, 80.0], [40.0, 80.0]),
    (0, [128.0], [96.0]),
    (1, [220.0, 30.0, 128.5, 5.0], [20.0, 170.0, 96.5, 5.0]),
    (1, [60.0, 60.4, 61.0], [60.0, 60.4, 61.0]),
]


def _images():
    rng = np.random.RandomState(0)
    return [rng.randint(0, 256, (H, W, 3)).astype(np.uint8) for _ in range(2)]


def _fake_corpus():
    """The same scanpaths exposed both ways: pysaliency-style for evaluate.run
    and adapter-style for eval_one_epoch_foveated."""
    images = _images()
    stimuli = SimpleNamespace(stimuli=images)
    fixations = SimpleNamespace(
        train_xs=[np.asarray(xs, dtype=float) for _, xs, _ in _SCANPATHS],
        train_ys=[np.asarray(ys, dtype=float) for _, _, ys in _SCANPATHS],
        train_ns=[n for n, _, _ in _SCANPATHS],
    )
    dataset = SimpleNamespace(
        stim_indices=[0, 1],
        scanpaths_for_stim=lambda sidx: [
            (xs, ys, 0) for n, xs, ys in _SCANPATHS if n == sidx
        ],
        image_and_centerbias=lambda sidx: (
            images[sidx], load_centerbias_for_image(H, W).astype(np.float32)
        ),
    )
    return stimuli, fixations, dataset


class ContentStub(nn.Module):
    """Log-density = per-sample log-softmax of the (possibly foveated) input's
    first channel. History-blind, content-dependent: blurring the input moves
    the output, so foveation parity is actually tested."""

    included_fixations = (-1, -2, -3, -4)

    def forward(self, image, centerbias, x_hist, y_hist, durations=None):
        logits = image[:, 0, :, :] * 0.02
        flat = logits.reshape(logits.shape[0], -1)
        logd = flat - torch.logsumexp(flat, dim=1, keepdim=True)
        return logd.reshape(logits.shape[0], 1, *logits.shape[1:])


@pytest.mark.parametrize("foveate", [False, True],
                         ids=["sharp", "foveated"])
def test_val_loop_agrees_with_the_reporting_evaluator(foveate):
    """`eval_one_epoch_foveated` and `evaluate.run` must agree per image.

    Arms are selected on the first loop's val LL and reported from the second;
    this pins pooled LL, pooled IG, the fixation count, and every per-image IG
    to the same values through both, on sharp and on foveated input.
    micro_batch=2 forces the val loop to chunk differently from evaluate.run's
    one-shot batch, so the pooling equivalence is exercised too.
    """
    stimuli, fixations, dataset = _fake_corpus()
    model = ContentStub().eval()
    device = torch.device("cpu")
    fov = Foveation(ppd=35.0, foveal_cpd=10.0)

    res_val = eval_one_epoch_foveated(
        model, dataset, device, fov, foveate=foveate, micro_batch=2)
    res_rep = evaluate_run(
        model=model, stimuli=stimuli, fixations=fixations,
        indices=list(range(len(_SCANPATHS))), device=device,
        model_name="stub", fold_no=-1,
        foveation=fov if foveate else None)

    # 4 + 1 + 3 + 2 pairs from the four evaluable scanpaths.
    assert res_val["n_fix_seen"] == res_rep["n_fixations"] == 10

    # Sharp input is exact. Foveated input agrees to fp32 round-off only: the
    # two loops batch differently (per scanpath vs per stimulus), and per-sample
    # float32 results shift in the last bits with batch composition — the same
    # platform effect test_fast_dg3.FP32_ROUNDING documents. IG additionally
    # carries a ~1e-7-bit offset either way, because the val loop gathers the
    # centerbias as float32 (datasets.image_and_centerbias hands it over as
    # float32) while evaluate.run samples its own float64 load. Reported IGs
    # live at 1e-3 bits, four orders above both.
    tol = 1e-9 if not foveate else 5e-6
    rep_ll_bits = res_rep["ll_per_fixation_nats"] / LOG2
    assert res_val["ll_bits_per_fix"] == pytest.approx(rep_ll_bits, abs=tol)
    rep_ig_bits = rep_ll_bits - res_rep["ll_centerbias_bits_pooled"]
    assert res_val["ig_bits_per_fix"] == pytest.approx(rep_ig_bits, abs=5e-6)

    rep_per_image = dict(zip(res_rep["per_image"]["stim"],
                             res_rep["per_image"]["IG_bits"]))
    val_per_image = {r["stim"]: r["ig_bits_per_fix"] for r in res_val["per_image"]}
    assert set(val_per_image) == set(rep_per_image) == {0, 1}
    for stim, ig in val_per_image.items():
        assert ig == pytest.approx(rep_per_image[stim], abs=5e-6)


class CountingFoveation(Foveation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stack_builds = 0

    def blur_stack(self, image):
        self.stack_builds += 1
        return super().blur_stack(image)


def test_foveated_eval_builds_one_pyramid_per_stimulus():
    """Interleaved scanpath indices must not defeat the blur-stack cache.

    evaluate.run sorts indices by stimulus so the pyramid is built once per
    image rather than once per scanpath, and the reordering must not change
    any aggregate (the aggregation is order-independent by construction).
    """
    stimuli, fixations, _ = _fake_corpus()
    model = ContentStub().eval()
    device = torch.device("cpu")
    interleaved = [0, 3, 1, 4]          # stim 0, 1, 0, 1

    fov = CountingFoveation(ppd=35.0, foveal_cpd=10.0)
    res_a = evaluate_run(model=model, stimuli=stimuli, fixations=fixations,
                         indices=interleaved, device=device,
                         model_name="stub", fold_no=-1, foveation=fov)
    assert fov.stack_builds == 2, \
        f"expected one pyramid per stimulus, built {fov.stack_builds}"

    fov_b = CountingFoveation(ppd=35.0, foveal_cpd=10.0)
    res_b = evaluate_run(model=model, stimuli=stimuli, fixations=fixations,
                         indices=[0, 1, 3, 4], device=device,
                         model_name="stub", fold_no=-1, foveation=fov_b)
    for key in ("ll_per_fixation_nats", "IG_bits_mean", "NSS_mean", "AUC_mean"):
        assert res_a[key] == pytest.approx(res_b[key], abs=1e-12), \
            f"{key} depends on scanpath index order"


class PeakStub(nn.Module):
    """Nearly all mass on one fixed pixel — known-answer anchor for NSS/AUC."""

    included_fixations = (-1, -2, -3, -4)
    PEAK_YX = (96, 128)

    def forward(self, image, centerbias, x_hist, y_hist, durations=None):
        B = image.shape[0]
        h, w = image.shape[-2], image.shape[-1]
        logits = torch.full((B, h, w), -20.0, dtype=torch.float32,
                            device=image.device)
        logits[:, self.PEAK_YX[0], self.PEAK_YX[1]] = 0.0
        flat = logits.reshape(B, -1)
        logd = flat - torch.logsumexp(flat, dim=1, keepdim=True)
        return logd.reshape(B, 1, h, w)


def test_nss_and_auc_have_known_anchors():
    """Value checks, not just schema: uniform → AUC exactly 0.5; a density
    peaked on the fixated pixel → AUC ~1, NSS large and positive. The existing
    suite only asserted SE > 0 and key presence for these two metrics."""
    stimuli, fixations, _ = _fake_corpus()
    device = torch.device("cpu")

    class UniformStub(nn.Module):
        included_fixations = (-1, -2, -3, -4)

        def forward(self, image, centerbias, x_hist, y_hist, durations=None):
            B = image.shape[0]
            h, w = image.shape[-2], image.shape[-1]
            return torch.full((B, 1, h, w), -math.log(h * w),
                              dtype=torch.float32, device=image.device)

    res_u = evaluate_run(model=UniformStub().eval(), stimuli=stimuli,
                         fixations=fixations, indices=[0, 3], device=device,
                         model_name="uniform", fold_no=-1)
    # Every pixel ties every fixation: rank-based AUC is exactly 1/2. (NSS is
    # undefined here — the density has zero variance — so it is not asserted.)
    assert res_u["AUC_mean"] == pytest.approx(0.5, abs=1e-6)

    peak = PeakStub()
    py, px = peak.PEAK_YX
    fixations_on_peak = SimpleNamespace(
        train_xs=[np.array([10.0, float(px), float(px)])],
        train_ys=[np.array([10.0, float(py), float(py)])],
        train_ns=[0],
    )
    res_p = evaluate_run(model=peak.eval(), stimuli=stimuli,
                         fixations=fixations_on_peak, indices=[0],
                         device=device, model_name="peak", fold_no=-1)
    assert res_p["AUC_mean"] > 0.99
    # Nearly all mass on one of the H*W pixels puts the z-scored density at the
    # fixation at sqrt(H*W) — mean 1/N and standard deviation 1/sqrt(N-1). This
    # is the value through `evaluate.run`, so it pins the wiring and not just
    # `nss` itself; the swapped-unpack formula gives N instead, 221x larger, so
    # unlike a `> 10` bound this fails if the vendor routine comes back.
    assert res_p["NSS_mean"] == pytest.approx(math.sqrt(H * W), rel=0.01)
    assert res_p["LL_bits_mean"] > 0.0


def test_nss_z_scores_the_density():
    """NSS is `(p - mean) / sd` read at the fixation. `deepgaze_pytorch`'s copy
    unpacks `torch.std_mean` in the wrong order and computes `(p - sd) / mean`,
    so `evaluate.nss` overrides it; this pins the definition against a value
    computed by hand."""
    from tez_deepgaze.evaluate import nss

    torch.manual_seed(0)
    logits = torch.randn(1, 8, 8)
    log_density = logits - torch.logsumexp(logits.reshape(1, -1), 1).reshape(1, 1, 1)
    mask = torch.zeros(1, 8, 8, dtype=torch.int32)
    mask[0, 3, 5] = 1

    density = torch.exp(log_density)[0]
    expected = (density[3, 5] - density.mean()) / density.std()

    # Absolute, not relative: the fused `torch.std_mean` and the separate
    # `.mean()`/`.std()` reduce in different orders, so the two float32 results
    # differ in the last bits. That is invisible at the NSS values this project
    # reports and unbounded as a ratio near zero, where some seeds land. The
    # worst absolute gap over 3,000 seeds is 5e-7. The swap this pins moves the
    # value by 0.27 on the seed below, five orders above the tolerance.
    got = nss(log_density, mask, weights=torch.ones(1))
    assert float(got) == pytest.approx(float(expected), abs=1e-5)


def test_baseline_schema_writer_populates_every_required_key(tmp_path):
    """`_run_full_corpus` writes the artefact the thesis quotes; its key set
    and internal consistency were previously only checkable on a cluster run.
    Uses the committed centerbias.json and a stub model — no DG3, no corpus."""
    from tez_deepgaze.baseline import BASELINE_JSON_REQUIRED_KEYS, _run_full_corpus

    stimuli, fixations, _ = _fake_corpus()
    out = tmp_path / "baseline.json"
    args = SimpleNamespace(seed=0, subsample=0, out=out)
    _run_full_corpus(args, torch.device("cpu"), stimuli, fixations,
                     ContentStub().eval())

    result = json.loads(out.read_text())
    missing = BASELINE_JSON_REQUIRED_KEYS - set(result)
    assert not missing, f"schema-v2 artefact missing keys: {missing}"
    assert result["$schema_version"] == 2
    assert result["n_fixations_evaluated"] == 10
    assert result["ll_per_fixation_bits"] == pytest.approx(
        result["ll_per_fixation_nats"] / LOG2)
    # IG must be model LL minus the committed external centerbias LL — the
    # subtraction happening twice (or not at all) is the classic IG bug.
    assert result["ig_bits_corpus_pooled"] == pytest.approx(
        result["ll_per_fixation_bits"] - result["ll_centerbias_bits_corpus_pooled"])


def test_baseline_schema_writer_honours_subsample(tmp_path):
    """--subsample must reduce the scanpath count and still write a complete
    schema — it is the documented smoke path for the full artefact."""
    from tez_deepgaze.baseline import BASELINE_JSON_REQUIRED_KEYS, _run_full_corpus

    stimuli, fixations, _ = _fake_corpus()
    out = tmp_path / "smoke.json"
    args = SimpleNamespace(seed=0, subsample=2, out=out)
    _run_full_corpus(args, torch.device("cpu"), stimuli, fixations,
                     ContentStub().eval())

    result = json.loads(out.read_text())
    assert not (BASELINE_JSON_REQUIRED_KEYS - set(result))
    assert result["n_scanpaths_evaluated"] <= 2


def test_centerbias_is_a_proper_log_density_at_any_size():
    """The resize + renormalisation must hand back log-densities summing to 1
    — `compute_ig_bits`'s single-subtraction contract silently breaks if the
    zoom step stops being renormalised."""
    for h, w in ((H, W), (123, 217)):
        cb = load_centerbias_for_image(h, w)
        assert cb.shape == (h, w)
        assert np.isfinite(cb).all()
        total = np.exp(cb.astype(np.float64)).sum()
        assert total == pytest.approx(1.0, abs=1e-6)
