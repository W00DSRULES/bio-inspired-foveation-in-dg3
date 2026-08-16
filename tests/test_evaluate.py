"""Tests for `tez_deepgaze.evaluate.run`."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from tez_deepgaze.evaluate import run as evaluate_run

# Loads the corpus (mit1003_subsample) and pretrained DG3 (dg3_model).
pytestmark = pytest.mark.heavy


# The result contract. `ll_per_fixation_nats` is the baseline.py-parity sidecar;
# `LL_bits_mean` is the vendor-metric-path bits aggregate. Both are required.
EXPECTED_KEYS = {
    "LL_bits_mean", "LL_bits_se",
    "IG_bits_mean", "IG_bits_se",
    "NSS_mean", "NSS_se",
    "AUC_mean", "AUC_se",
    "n_fixations", "n_images",
    "model", "fold_no", "device", "seed",
    "ll_per_fixation_nats",
}


class StubUniform(nn.Module):
    """Uniform log-density with the DG3 forward signature."""

    included_fixations = (-1, -2, -3, -4)

    def forward(self, image, centerbias, x_hist, y_hist, durations=None):
        B = image.shape[0]
        H, W = image.shape[-2], image.shape[-1]
        return torch.full((B, 1, H, W), -math.log(H * W),
                          dtype=torch.float32, device=image.device)


# The DG3 evaluations below are deterministic (fixed seed, eval mode, fixed
# subsample), so tests that assert different properties of the same result
# share one run instead of repeating it — the same trade test_cv_split makes
# for its split artifact. Tests only read these dicts.

@pytest.fixture(scope="module")
def dg3_result(device, dg3_model, mit1003_subsample):
    """One default-argument evaluate.run over the shared subsample."""
    stimuli, fixations, indices = mit1003_subsample
    return evaluate_run(
        model=dg3_model, stimuli=stimuli, fixations=fixations,
        indices=indices, device=device,
        model_name="DG3", fold_no=-1,
    )


@pytest.fixture(scope="module")
def dg3_per_fixation_result(device, dg3_model, mit1003_subsample):
    """The same evaluation with the per-fixation records retained."""
    stimuli, fixations, indices = mit1003_subsample
    return evaluate_run(
        model=dg3_model, stimuli=stimuli, fixations=fixations,
        indices=indices, device=device, model_name="DG3", fold_no=-1,
        per_fixation=True,
    )


def test_smoke_parity_with_baseline(device, dg3_model, mit1003_subsample):
    """Harness reproduces baseline.py per-fix LL nats on the SAME subsample to <1e-6.

    NOTE: harness_nats reads the `ll_per_fixation_nats` sidecar key (baseline.py
    parity). LL_bits_mean uses the vendor-metric path; the
    two diverge by per-image-uniform weighting which is intentional.

    Compare against a FRESH in-process baseline run, never against the stored
    results/baselines/baseline.json. The stored artefact is a full-corpus cluster
    run over a different fixation set than any subsample, so equality with it is
    not the property under test.
    """
    from tez_deepgaze.baseline import _nearest_sample  # noqa: F401 — sentinel: nearest-pixel
    from tez_deepgaze.centerbias import load_centerbias_for_image
    from tez_deepgaze.instrument import compute_log_density_batch

    stimuli, fixations, indices = mit1003_subsample

    # Reference: the plain per-fixation LL loop, inline on the same indices.
    ref_ll_sum, ref_n = 0.0, 0
    for sp_idx in indices:
        xs = np.asarray(fixations.train_xs[sp_idx], dtype=float)
        ys = np.asarray(fixations.train_ys[sp_idx], dtype=float)
        stim_idx = int(fixations.train_ns[sp_idx])
        image = np.asarray(stimuli.stimuli[stim_idx])
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        cb = load_centerbias_for_image(image.shape[0], image.shape[1])
        valid_end = int(np.sum(~(np.isnan(xs) | np.isnan(ys))))
        if valid_end <= 1:
            continue
        hist_x = [list(xs[:i]) for i in range(1, valid_end)]
        hist_y = [list(ys[:i]) for i in range(1, valid_end)]
        log_d = compute_log_density_batch(
            dg3_model, image, cb, hist_x, hist_y, device,
        )
        for k, i in enumerate(range(1, valid_end)):
            ref_ll_sum += _nearest_sample(log_d[k], xs[i], ys[i])
            ref_n += 1
    ref_ll_per_fix_nats = ref_ll_sum / ref_n

    # Harness path on the same indices.
    result = evaluate_run(
        model=dg3_model,
        stimuli=stimuli,
        fixations=fixations,
        indices=indices,
        device=device,
        model_name="DeepGazeIII-pretrained",
        fold_no=-1,
    )
    # The harness exposes `ll_per_fixation_nats` as the baseline.py-parity
    # sidecar (raw nats). The bit-exact contract is what matters.
    harness_nats = result["ll_per_fixation_nats"]
    assert abs(harness_nats - ref_ll_per_fix_nats) < 1e-6


def test_output_schema(dg3_result):
    """Harness returns dict with the expected keys (incl. ll_per_fixation_nats)."""
    assert EXPECTED_KEYS.issubset(set(dg3_result.keys())), \
        f"missing keys: {EXPECTED_KEYS - set(dg3_result.keys())}"


def test_se_positive(dg3_result):
    """Standard error reported (and > 0 when n_images > 1)."""
    if dg3_result["n_images"] > 1:
        assert dg3_result["LL_bits_se"] > 0.0
        assert dg3_result["NSS_se"] > 0.0


def test_central_fixation_excluded(dg3_result, mit1003_subsample):
    """Harness fixation count == sum(fixations.lengths[indices] > 0)."""
    _, fixations, indices = mit1003_subsample
    # Per scanpath sp_idx, baseline.py counts (valid_end - 1) non-initial fixations.
    # That equals (fix.lengths[sp_idx]) where len = number of NON-initial fixations
    # IF lengths convention follows pysaliency's "trailing fix count after start".
    # The test uses the canonical baseline.py recipe to match exactly.
    expected = 0
    for sp_idx in indices:
        xs = np.asarray(fixations.train_xs[sp_idx], dtype=float)
        ys = np.asarray(fixations.train_ys[sp_idx], dtype=float)
        valid_end = int(np.sum(~(np.isnan(xs) | np.isnan(ys))))
        if valid_end > 1:
            expected += valid_end - 1
    assert dg3_result["n_fixations"] == expected


def test_model_agnostic_signature(device, mit1003_subsample):
    """Any nn.Module with DG3 forward signature works, no per-config branches."""
    stimuli, fixations, indices = mit1003_subsample

    model = StubUniform().to(device).eval()
    # Smoke: this MUST run end-to-end; the evaluator has no isinstance branches.
    result = evaluate_run(
        model=model, stimuli=stimuli, fixations=fixations,
        indices=indices, device=device,
        model_name="stub-uniform", fold_no=-1,
    )
    assert "LL_bits_mean" in result


def test_units_are_bits(device, mit1003_subsample):
    """LL_bits_mean is in bits (vendor metric path; uniform → 0 bits/fix)."""
    stimuli, fixations, indices = mit1003_subsample

    model = StubUniform().to(device).eval()
    result = evaluate_run(
        model=model, stimuli=stimuli, fixations=fixations,
        indices=indices, device=device,
        model_name="stub-uniform", fold_no=-1,
    )
    # Uniform log-density of shape (H, W) gives LL = 0 bits/fix exactly via the
    # vendor metric: (sum(log_d * mask) / fix_count) → -log(H*W) (nats),
    # then + log(H*W) and / log(2) → 0. (See vendor metrics.py line 28.)
    assert abs(result["LL_bits_mean"]) < 1e-5


def test_start_fixation_changes_count(device, dg3_model, mit1003_subsample):
    """start_fixation differential.

    Calling with start_fixation=0 vs 1 must produce different fixation counts;
    the difference is exactly `n_scanpaths_evaluated` (one extra fixation per
    scanpath when including the central/initial fixation). Defends against
    silent reversion to a literal `range(1, valid_end)` in evaluate.py.
    """
    stimuli, fixations, indices = mit1003_subsample

    # A stub model rather than dg3_model keeps this fast.
    model = StubUniform().to(device).eval()

    res0 = evaluate_run(
        model=model, stimuli=stimuli, fixations=fixations,
        indices=indices, device=device,
        model_name="stub-uniform", fold_no=-1,
        start_fixation=0,
    )
    res1 = evaluate_run(
        model=model, stimuli=stimuli, fixations=fixations,
        indices=indices, device=device,
        model_name="stub-uniform", fold_no=-1,
        start_fixation=1,
    )
    assert res0["n_fixations"] != res1["n_fixations"], \
        "start_fixation=0 vs 1 produced identical n_fixations — differential broken"
    # Strong form: passing start_fixation=0 adds exactly one fixation per scanpath.
    # `n_scanpaths_evaluated` is the count of scanpaths with valid_end > start_fixation
    # for the higher start (i.e. those that contributed when start_fixation=1).
    assert res0["n_fixations"] - res1["n_fixations"] == res1["n_scanpaths_evaluated"], \
        "start_fixation=0 should add exactly one fixation per evaluated scanpath"


def test_per_image_ig_is_what_the_aggregate_is_taken_over(dg3_result):
    """The returned per-image array must BE the aggregate's input, not a copy of it.

    The reported result rests on ~10 fold-level numbers while ~1,000 image-level values
    exist. Returning them is only useful if they are the same values the reported
    mean and SE come from — if the array were recomputed separately, a paired
    per-image analysis would not be analysing the reported number. Equality is
    asserted exactly, which also pins that adding the array changed no aggregate.
    """
    result = dg3_result
    per_img = np.asarray(result["per_image"]["IG_bits"], dtype=np.float64)
    assert len(per_img) == result["n_images"]
    assert len(result["per_image"]["stim"]) == result["n_images"]
    assert sorted(set(result["per_image"]["stim"])) == list(result["per_image"]["stim"]), \
        "stimulus ids must be unique and aligned with the IG values"

    assert float(per_img.mean()) == result["IG_bits_mean"]
    if result["n_images"] > 1:
        se = float(per_img.std(ddof=1) / math.sqrt(len(per_img)))
        assert se == result["IG_bits_se"]


def test_per_image_ig_agrees_with_the_per_fixation_records(dg3_per_fixation_result):
    """The two paths that can produce per-image IG must not drift apart.

    `per_fixation=True` already returns the model and centerbias log-densities at
    every fixation, so per-image IG is derivable from them by grouping on `stim`.
    The aggregate path keeps its own running sums instead — cheaper, and it is
    what the reported number has always used. Rather than collapse one into the
    other, this pins that they agree.
    """
    result = dg3_per_fixation_result
    pf = result["per_fixation"]
    ig_per_fix = pf["ll_bits"] - pf["cb_bits"]
    derived = {
        int(s): float(ig_per_fix[pf["stim"] == s].mean())
        for s in np.unique(pf["stim"])
    }
    got = dict(zip(result["per_image"]["stim"], result["per_image"]["IG_bits"]))
    assert set(got) == set(derived)
    for stim, value in got.items():
        assert value == pytest.approx(derived[stim], rel=0, abs=1e-9)


def test_per_fixation_diagnostics_math():
    """Entropy, spatial PIT and mode distance on hand-checkable densities."""
    from tez_deepgaze.evaluate import _per_fixation_diagnostics

    H = W = 4
    n_pix = H * W

    # Row 0: exactly uniform. Entropy = log2(16) = 4 bits; every pixel ties the
    # fixated one, so the mass at density >= threshold is the whole image, 1.0.
    # Row 1: all mass on pixel (2, 3). Entropy -> 0; the fixation is placed at
    # (2, 3) too, so PIT = 1.0 there and the mode distance is 0.
    logd = torch.empty(2, H, W)
    logd[0] = math.log(1.0 / n_pix)
    peak = torch.full((H, W), -30.0)
    peak[2, 3] = 0.0
    logd[1] = peak - torch.logsumexp(peak.flatten(), 0)

    d = _per_fixation_diagnostics(logd, np.array([0.0, 3.0]), np.array([0.0, 2.0]))

    assert d["entropy_bits"][0] == pytest.approx(4.0, abs=1e-4)
    assert d["pit"][0] == pytest.approx(1.0, abs=1e-5)
    assert d["entropy_bits"][1] < 0.02
    assert d["pit"][1] == pytest.approx(1.0, abs=1e-4)
    assert d["mode_dist_px"][1] == pytest.approx(0.0, abs=1e-6)

    # Mode distance is Euclidean from the argmax to the fixated pixel: put the
    # fixation at (0, 0) with the peak still at (2, 3) -> sqrt(9 + 4).
    d2 = _per_fixation_diagnostics(logd[1:], np.array([0.0]), np.array([0.0]))
    assert d2["mode_dist_px"][0] == pytest.approx(math.sqrt(3**2 + 2**2), abs=1e-5)
    # Almost all mass sits above the near-zero density at (0, 0), so PIT ~ 0
    # only if the fixated pixel itself is excluded; it is included, so PIT is
    # 1.0 minus nothing measurable below it -> stays ~1 for a two-level map.
    assert 0.0 <= d2["pit"][0] <= 1.0 + 1e-6


def test_per_fixation_pit_is_uniform_for_a_calibrated_sampler():
    """PIT values are U(0,1) when fixations are drawn from the model's own density.

    This is the property the calibration diagnostic relies on, so it is worth
    pinning: sample locations from a known density and the resulting PIT
    population should be close to uniform.
    """
    from tez_deepgaze.evaluate import _per_fixation_diagnostics

    rng = np.random.RandomState(0)
    H = W = 24
    logits = torch.from_numpy(rng.randn(H, W).astype(np.float32)) * 1.5
    logd = logits - torch.logsumexp(logits.flatten(), 0)
    p = logd.exp().numpy().ravel()

    n = 4000
    flat = rng.choice(np.arange(H * W), size=n, p=p / p.sum())
    ys, xs = (flat // W).astype(float), (flat % W).astype(float)

    d = _per_fixation_diagnostics(logd.unsqueeze(0).expand(n, H, W).contiguous(), xs, ys)
    pit = np.sort(d["pit"])
    i = np.arange(1, n + 1)
    ks = float(np.max(np.abs(np.maximum(i / n - pit, pit - (i - 1) / n))))
    # 1.63/sqrt(n) is the ~99% KS critical value; a discrete density makes the
    # PIT mildly lumpy, so allow generous slack while still rejecting a broken
    # implementation (which lands near 0.5).
    assert ks < 0.10, f"PIT not close to uniform for a calibrated sampler: KS={ks}"


def test_amp_mass_is_a_distribution_over_the_declared_annuli():
    """`amp_mass` partitions the density into the pre-registered amplitude bins.

    Every pixel falls in exactly one annulus, so the rows must sum to 1 and the
    column count must match the declared edges. Both are load-bearing: the whole
    point of the field is to be the model's implied distribution over saccade
    amplitude, comparable bin-for-bin against the human one.
    """
    from tez_deepgaze.evaluate import AMP_EDGES_PX, _per_fixation_diagnostics

    H, W = 40, 60
    rng = np.random.RandomState(1)
    logits = torch.from_numpy(rng.randn(3, H, W).astype(np.float32))
    logd = logits - torch.logsumexp(logits.flatten(1), 1).view(-1, 1, 1)

    gx = np.array([0.0, W / 2, W - 1.0])
    gy = np.array([0.0, H / 2, H - 1.0])
    d = _per_fixation_diagnostics(logd, gx, gy, gx, gy)

    assert d["amp_mass"].shape == (3, len(AMP_EDGES_PX) - 1)
    assert np.allclose(d["amp_mass"].sum(axis=1), 1.0, atol=1e-6)
    assert (d["amp_mass"] >= 0).all()

    # A density concentrated on the gaze pixel must put its mass in bin 0, and a
    # gaze point in the far corner must push mass into the outer annuli — this is
    # what pins the geometry rather than merely the normalisation.
    peak = torch.full((1, H, W), -30.0)
    peak[0, 5, 7] = 0.0
    peak = peak - torch.logsumexp(peak.flatten(1), 1).view(-1, 1, 1)
    at_gaze = _per_fixation_diagnostics(
        peak, np.array([7.0]), np.array([5.0]), np.array([7.0]), np.array([5.0]))
    assert at_gaze["amp_mass"][0, 0] > 0.99
    far = _per_fixation_diagnostics(
        peak, np.array([7.0]), np.array([5.0]), np.array([59.0]), np.array([39.0]))
    # gaze (59, 39) to the peak at (7, 5) is ~62 px, which is the 50-70 bin.
    assert far["amp_mass"][0, AMP_EDGES_PX.index(50.0)] > 0.99


def test_per_fixation_geometry_and_clustering_keys(dg3_per_fixation_result):
    """The columns a clustered analysis depends on must be present and consistent.

    `subject` and `stimulus_id` cannot be recovered from any summary, and
    `sacc_px` / `target_ecc_px` must be distinct quantities — conflating them
    would silently merge "distance from gaze" with "distance from image centre",
    which is exactly the confound the eccentricity column exists to separate.
    """
    pf = dg3_per_fixation_result["per_fixation"]

    # sacc_px is the distance from the fovea centre to the target.
    recomputed = np.hypot(pf["target_x"] - pf["gaze_x"], pf["target_y"] - pf["gaze_y"])
    assert np.allclose(recomputed, pf["sacc_px"], atol=1e-6, equal_nan=True)
    # target_ecc_px is the distance from the image centre to the target.
    ecc = np.hypot(pf["target_x"] - pf["img_w"] / 2, pf["target_y"] - pf["img_h"] / 2)
    assert np.allclose(ecc, pf["target_ecc_px"], atol=1e-6)
    assert not np.allclose(pf["sacc_px"], pf["target_ecc_px"])

    assert pf["subject"].min() >= 0
    assert all(len(s) == 40 for s in pf["stimulus_id"])
    # One stimulus index must map to exactly one stimulus id, or grouping on the
    # id would silently split or merge images.
    for stim in np.unique(pf["stim"]):
        assert len(set(pf["stimulus_id"][pf["stim"] == stim])) == 1


def test_save_per_fixation_round_trips(tmp_path, dg3_per_fixation_result):
    """The npz keeps every per-fixation column plus enough metadata to stand alone."""
    from tez_deepgaze.evaluate import save_per_fixation

    result = dg3_per_fixation_result
    path = tmp_path / "normal_fold3.npz"
    save_per_fixation(result, path, arm="normal", fold=3, split="val")

    z = np.load(path, allow_pickle=False)
    for key, arr in result["per_fixation"].items():
        assert key in z.files
        if arr.dtype.kind == "f":
            assert np.allclose(z[key], arr, equal_nan=True)
        else:
            assert (z[key] == arr).all()
    meta = json.loads(str(z["meta_json"]))
    assert meta["arm"] == "normal" and meta["fold"] == 3 and meta["split"] == "val"
    assert meta["n_fixations"] == result["n_fixations"]
    assert len(z["amp_edges_px"]) == result["per_fixation"]["amp_mass"].shape[1] + 1
    # `fold` is the caller's label, `fold_no` comes from the evaluation itself.
    # They have different provenances and the npz must not conflate them.
    assert meta["fold_no"] == result["fold_no"] != meta["fold"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
