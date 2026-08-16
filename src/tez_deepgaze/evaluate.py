"""Single model-agnostic evaluator: per-fixation LL/IG/NSS/AUC mean ± SE.

Works on any ``nn.Module`` with the DG3 forward signature
``(image, centerbias, x_hist, y_hist) -> (B, 1, H, W)``. The caller passes
the model — this function never introspects its type, so the same code
path serves the unmodified DG3 baseline and the gaze-contingent
input-foveation arm (``foveation=`` below).

The per-image loop is structured so that the returned ``ll_per_fixation_nats``
matches a plain per-fixation nearest-pixel loop over the same
``(model, stimuli, fixations, indices)`` to within 1e-6; a parity test asserts
that equality.

``start_fixation`` controls the first scanpath index evaluated. Default 1
scores from the second fixation on, using the first as history — index 0 has
none. (``pysaliency`` already drops MIT1003's central forced-start fixation,
so index 0 is the first free fixation.) Passing 0 is untested: DG3's scanpath
encoder is not built for empty histories.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from deepgaze_pytorch.metrics import auc, log_likelihood

from .baseline import _nearest_sample
from .centerbias import load_centerbias_for_image
from .ig import LOG2
from .instrument import compute_log_density_batch, ensure_rgb, image_tensor


def nss(
    log_density: torch.Tensor,
    fixation_mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Normalized Scanpath Saliency: the z-scored density read at the fixations.

    A local copy of ``deepgaze_pytorch.metrics.nss`` with one line corrected.
    The released version unpacks ``mean, std = torch.std_mean(density, ...)``,
    but ``torch.std_mean`` returns ``(std, mean)``, so each name holds the
    other's value and the function computes ``(p - sigma) / mu`` where NSS is
    ``(p - mu) / sigma``. ``log_likelihood`` and ``auc`` were read and are
    correct, so they are still imported.

    Everything else — the weight normalisation, the sparse-mask branch and the
    reduction — is the vendor implementation unchanged, so the returned value
    keeps the same shape and scanpath-mean semantics its callers assume. The
    vendor signature defaults ``weights`` to ``None``, which its own first line
    then raises on; the argument is required here instead.

    A map of exactly zero variance gives ``nan``, the value of ``0 / 0``. The
    swapped version returned ``1.0`` there by dividing by the mean instead, and
    nothing depends on that, since a DG3 map is a log-softmax over pixels and
    the case needs every pixel bit-identical.
    """
    weights = len(weights) * weights.view(-1, 1, 1) / weights.sum()
    if isinstance(fixation_mask, torch.sparse.IntTensor):
        dense_mask = fixation_mask.to_dense()
    else:
        dense_mask = fixation_mask

    fixation_count = dense_mask.sum(dim=(-1, -2), keepdim=True)

    density = torch.exp(log_density)
    std, mean = torch.std_mean(density, dim=(-1, -2), keepdim=True)
    saliency_map = (density - mean) / std

    return torch.mean(
        weights * torch.sum(saliency_map * dense_mask, dim=(-1, -2), keepdim=True)
        / fixation_count
    )


def _build_dense_fixation_mask(
    eval_xs: np.ndarray,
    eval_ys: np.ndarray,
    H: int,
    W: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a dense (B, H, W) one-hot fixation mask.

    A sparse representation would be more memory-efficient, but the Apple
    MPS backend does not support the sparse ops the metrics use,
    so we materialise the dense mask unconditionally.
    """
    B = len(eval_xs)
    mask = torch.zeros((B, H, W), dtype=torch.int32, device=device)
    xs_int = np.clip(np.round(eval_xs).astype(int), 0, W - 1)
    ys_int = np.clip(np.round(eval_ys).astype(int), 0, H - 1)
    for k in range(B):
        mask[k, ys_int[k], xs_int[k]] = 1
    return mask


# Saccade-amplitude bin edges in pixels, also used as the annulus edges for
# `amp_mass` below. Chosen from the measured MIT1003 amplitude distribution
# (89,255 saccades, median 160.2 px) so that:
#   * the sharp-disc radii land exactly on an edge — 11.5 (cpd 20), 103.5
#     (cpd 40), 149.5 (cpd 50, not part of the design) and 0 (cpd 10,
#     trivially) — so a regime change at any arm's identity disc shows as a
#     step between bins, never inside one;
#   * every bin holds >= 500 fixations, so no cell is decided by a handful of
#     saccades. The 0-11.5 bin is the thinnest at 503 (0.56 %) and exists
#     because it is exactly "inside cpd 20's disc";
#   * resolution is roughly logarithmic, which matches how saccade amplitudes
#     are distributed.
# One set for every analysis: with per-fixation records saved, re-binning is
# always possible after the fact, so this is the *reported* set and any other
# binning is a robustness check against it.
AMP_EDGES_PX = (0.0, 11.5, 18.0, 25.0, 35.0, 50.0, 70.0, 103.5, 125.0, 149.5,
                175.0, 210.0, 260.0, 320.0, 400.0, 500.0, float("inf"))


@torch.no_grad()
def _per_fixation_diagnostics(
    log_d_t: torch.Tensor,
    eval_xs: np.ndarray,
    eval_ys: np.ndarray,
    gaze_xs: np.ndarray | None = None,
    gaze_ys: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Shape diagnostics of the predicted density, one value per fixation.

    ``log_d_t`` is ``(B, H, W)`` log-density normalised over pixels
    (``logsumexp == 0``), so ``exp`` is a proper probability per pixel.

    The likelihood metrics read the density only at the fixated pixel and say
    nothing about the rest of the distribution; two models can score the same
    while being diffuse-and-uncertain versus sharp-and-mislocated. These
    quantities separate those cases:

    - ``entropy_bits``: $-\\sum p \\log_2 p$ over the image. High = the model
      spread its mass; low = it committed to a peak.
    - ``pit``: the probability mass at density *at least* that of the fixated
      pixel, i.e. the spatial probability integral transform. Small means the
      fixation landed in a high-ranked region. A calibrated model produces PIT
      values uniform on [0, 1] — deviation from uniform is miscalibration, and
      is testable against uniform directly.
    - ``mode_dist_px``: distance from the density's argmax to the true
      fixation. Interpretable in pixels where bits are not.
    - ``mode_x`` / ``mode_y``: where that argmax actually is. The distance says
      how far the peak sits from the fixation; the coordinates say in which
      direction it moved, which is what distinguishes "drifted toward the image
      centre" from "drifted away from gaze".
    - ``amp_mass``: ``(B, len(AMP_EDGES_PX) - 1)`` — the predicted probability
      mass in annuli around the *current gaze point*, i.e. the model's implied
      distribution over how far the next saccade will go. Requires ``gaze_xs`` /
      ``gaze_ys``; omitted without them. This is the only quantity here that
      cannot be reconstructed after the fact from a scalar summary, because it
      integrates the density over regions rather than reading one pixel — and it
      is what makes a claim about the model's amplitude prior testable against
      the human amplitude distribution.

    Nearest-pixel lookup matches ``baseline._nearest_sample`` and
    :func:`_build_dense_fixation_mask`, so these align with the reported LL.
    """
    B, H, W = log_d_t.shape
    dev = log_d_t.device
    xs_int = torch.from_numpy(np.clip(np.round(eval_xs).astype(int), 0, W - 1)).to(dev)
    ys_int = torch.from_numpy(np.clip(np.round(eval_ys).astype(int), 0, H - 1)).to(dev)

    p = log_d_t.exp()
    entropy_bits = -(p * log_d_t).sum(dim=(1, 2)) / LOG2

    rows = torch.arange(B, device=dev)
    thr = log_d_t[rows, ys_int, xs_int]                      # density at the fixation
    pit = (p * (log_d_t >= thr.view(-1, 1, 1))).sum(dim=(1, 2))

    flat_mode = log_d_t.flatten(1).argmax(dim=1)
    mode_y, mode_x = flat_mode // W, flat_mode % W
    mode_dist = torch.sqrt(
        (mode_x - xs_int).float() ** 2 + (mode_y - ys_int).float() ** 2
    )

    out = {
        "entropy_bits": entropy_bits.cpu().numpy(),
        "pit": pit.cpu().numpy(),
        "mode_dist_px": mode_dist.cpu().numpy(),
        "mode_x": mode_x.float().cpu().numpy(),
        "mode_y": mode_y.float().cpu().numpy(),
    }

    if gaze_xs is not None and gaze_ys is not None:
        gx = torch.as_tensor(gaze_xs, dtype=torch.float32, device=dev).view(-1, 1, 1)
        gy = torch.as_tensor(gaze_ys, dtype=torch.float32, device=dev).view(-1, 1, 1)
        yy = torch.arange(H, dtype=torch.float32, device=dev).view(1, -1, 1)
        xx = torch.arange(W, dtype=torch.float32, device=dev).view(1, 1, -1)
        # Squared distance against squared edges: squaring is monotonic on
        # non-negatives, so the bucket assignment is identical and a sqrt over
        # B*H*W (~12M elements per scanpath) is avoided. inf**2 is inf, so the
        # open outer edge still behaves.
        dist2 = (xx - gx) ** 2 + (yy - gy) ** 2                # (B, H, W)
        # Interior edges only: bucketize returns 0 below edges[0] and len(edges)
        # at or above edges[-1], so passing the interior gives exactly one bucket
        # per bin with no clamping needed.
        edges = torch.tensor([e * e for e in AMP_EDGES_PX[1:-1]],
                             dtype=torch.float32, device=dev)
        idx = torch.bucketize(dist2, edges).view(B, -1)
        mass = torch.zeros(B, len(AMP_EDGES_PX) - 1, dtype=p.dtype, device=dev)
        mass.scatter_add_(1, idx, p.view(B, -1))
        # Every pixel lands in exactly one bucket, so the rows are a complete
        # partition of the density and sum to 1 in exact arithmetic. In float32
        # over ~800k pixels they drift by a few times 1e-4, and MPS has no
        # float64 to accumulate in, so renormalise in float64 on the host. The
        # correction is pure summation noise, two orders below any effect being
        # measured, but leaving it in would make the rows not-quite-a-distribution.
        # .cpu() before .double(): MPS cannot cast to float64 on device.
        m64 = mass.cpu().double().numpy()
        out["amp_mass"] = (m64 / m64.sum(axis=1, keepdims=True)).astype(np.float32)

    return out


@torch.no_grad()
def run(
    model: torch.nn.Module,
    stimuli: Any,
    fixations: Any,
    indices: list[int],
    device: torch.device,
    model_name: str = "model",
    fold_no: int = -1,
    seed: int = 42,
    start_fixation: int = 1,
    foveation=None,
    per_fixation: bool = False,
    save_maps_for: set[int] | None = None,
) -> dict:
    """Per-fixation LL/IG/NSS/AUC mean ± SE on the supplied scanpath indices.

    A single forward pass per image produces a (B, H, W) log-density tensor
    that feeds both consumers — the per-fixation nearest-pixel lookup
    (the corpus-pooled LL number) and the saliency metrics
    (intra-image LL/NSS/AUC).

    Args:
        start_fixation: first scanpath index to evaluate. Default 1 scores
            from the second fixation on, using the first as history.
        foveation: optional :class:`foveate_input.Foveation`. When given, each
            history is scored on the image foveated gaze-contingently around
            its current fixation (the foveated arm). ``None`` (default)
            scores the sharp image.
        per_fixation: also return one record per evaluated fixation, keyed
            ``per_fixation`` (see below). Off by default: the aggregate path
            does not depend on it. Needed for
            any stratified analysis, because the aggregates below pool into
            per-image sums and discard the individual fixations.

    Aggregation is image-stratified (Kümmerer 2022): every scanpath's
    fixations are pooled into their stimulus, and the reported mean ± SE is
    taken across images (~1003 for full MIT1003), not across scanpaths
    (~15045). Pooling by fixation count within each image recovers the
    per-fixation mean for that image.

    Returns dict with keys:
      - LL_bits_mean / LL_bits_se: vendor log_likelihood, bits/fix *over the
        uniform baseline* (uniform model → 0), per-image mean ± SE.
      - LL_raw_bits_mean / LL_raw_bits_se: raw model log-density at fixations
        in bits/fix (same convention as the corpus-pooled
        ll_per_fixation_bits), per-image mean ± SE.
      - IG_bits_mean / IG_bits_se: information gain over the centerbias,
        bits/fix (raw model LL − raw centerbias LL, both nearest-pixel),
        per-image mean ± SE.
      - NSS_mean / NSS_se, AUC_mean / AUC_se: per-image mean ± SE. AUC is
        the vendor routine; NSS is the corrected local one (see ``nss``).
      - n_fixations, n_images, n_scanpaths_evaluated, model, fold_no,
        device, seed.
      - ll_per_fixation_nats: corpus-pooled raw model LL (parity check
        against a plain per-fixation loop; not a reported metric).
      - ll_centerbias_bits_pooled: corpus-pooled raw centerbias LL in bits,
        computed in-loop for cross-checking against results/baselines/
        centerbias.json.
      - per_image: ``{"stim": [...], "IG_bits": [...]}`` — the image-stratified
        information gains the IG mean and SE are taken over, with the stimulus
        index each one belongs to, in the same order. These are what make a
        *paired* per-image comparison of two arms possible: pairing on the
        stimulus removes image difficulty, which dominates the spread (fold-level
        IG spans 1.477-1.561 while the effects being measured are ~0.005). Aggregates
        are computed from this same list, so returning it adds no second path.
      - per_fixation (only when ``per_fixation=True``): dict of equal-length
        1-D arrays, one entry per evaluated fixation, in evaluation order —
        ``stim``, ``scanpath``, ``fix_index`` (index within the scanpath),
        ``sacc_px`` (amplitude of the saccade that landed here; NaN for the
        first evaluated fixation of a scanpath, which has no predecessor),
        ``ll_bits`` and ``cb_bits`` (model and centerbias log-density at the
        fixation, so IG is their difference), plus ``entropy_bits``, ``pit``
        and ``mode_dist_px`` from :func:`_per_fixation_diagnostics`.
    """
    train_x = fixations.train_xs
    train_y = fixations.train_ys
    train_n = fixations.train_ns

    # Per-image accumulators, keyed by stimulus index. Each holds a running
    # sum over all fixations of every scanpath that lands on that stimulus.
    from collections import defaultdict

    img_ll_over_uniform_sum: dict[int, float] = defaultdict(float)  # vendor bits
    img_nss_sum: dict[int, float] = defaultdict(float)
    img_auc_sum: dict[int, float] = defaultdict(float)
    img_model_nats_sum: dict[int, float] = defaultdict(float)  # raw model log-density
    img_cb_nats_sum: dict[int, float] = defaultdict(float)     # raw centerbias log-density
    img_fix_count: dict[int, int] = defaultdict(int)

    ll_nats_sum = 0.0   # corpus-pooled raw model nats (parity sidecar)
    cb_nats_sum = 0.0   # corpus-pooled raw centerbias nats (cross-check)
    n_fix_total = 0
    n_scanpaths_evaluated = 0

    pf: dict[str, list] = (
        {k: [] for k in (
            # identity / grouping — what a clustered or mixed-effects analysis
            # needs: subject is a random effect crossed with image, and
            # `stimulus_id` is a stable hash where `stim` is only a positional
            # index.
            "stim", "stimulus_id", "subject", "scanpath", "fix_index",
            # predictors
            "sacc_px", "sacc_dir_deg", "target_ecc_px", "target_x", "target_y",
            "gaze_x", "gaze_y", "img_h", "img_w", "fix_t",
            # outcomes
            "ll_bits", "cb_bits",
            # density shape
            "entropy_bits", "pit", "mode_dist_px", "mode_x", "mode_y", "amp_mass")}
        if per_fixation else {}
    )
    train_subjects = getattr(fixations, "train_subjects", None)
    train_ts = getattr(fixations, "train_ts", None)
    stimulus_ids = getattr(stimuli, "stimulus_ids", None)

    # Foveated runs group scanpaths by stimulus so the blur pyramid is built
    # once per image, not once per each of the ~15 scanpaths sharing it. The
    # aggregation below is order-independent, so reordering is safe.
    if foveation is not None:
        indices = sorted(indices, key=lambda i: int(train_n[i]))
    fov_stack = None
    fov_stack_stim = -1
    saved_maps: list[dict] = []

    for sp_idx in tqdm(indices, desc="evaluate"):
        xs = np.asarray(train_x[sp_idx], dtype=float)
        ys = np.asarray(train_y[sp_idx], dtype=float)
        stim_idx = int(train_n[sp_idx])
        image = ensure_rgb(np.asarray(stimuli.stimuli[stim_idx]))
        cb = load_centerbias_for_image(image.shape[0], image.shape[1])

        valid_end = int(np.sum(~(np.isnan(xs) | np.isnan(ys))))
        if valid_end <= start_fixation:
            continue
        n_scanpaths_evaluated += 1

        hist_x_batch = [list(xs[:i]) for i in range(start_fixation, valid_end)]
        hist_y_batch = [list(ys[:i]) for i in range(start_fixation, valid_end)]

        if per_fixation:
            # The fovea centre for each step, and the origin the amp_mass
            # annuli are measured from. Image centre when there is no
            # predecessor, matching instrument.compute_log_density_batch.
            gaze_xs = np.array([xs[i - 1] if i >= 1 else image.shape[1] / 2.0
                                for i in range(start_fixation, valid_end)], dtype=float)
            gaze_ys = np.array([ys[i - 1] if i >= 1 else image.shape[0] / 2.0
                                for i in range(start_fixation, valid_end)], dtype=float)

        # One forward pass; both consumers below read this same tensor.
        if foveation is not None and stim_idx != fov_stack_stim:
            fov_stack = foveation.blur_stack(image_tensor(image, device))
            fov_stack_stim = stim_idx
        log_d_np = compute_log_density_batch(
            model, image, cb, hist_x_batch, hist_y_batch, device,
            foveation=foveation, foveation_stack=fov_stack,
        )

        # Consumer A: per-fixation raw LL via nearest-pixel lookup, for both
        # the model and the centerbias floor. The model sum is the corpus-pooled
        # LL the parity test checks; the centerbias sum
        # gives the IG denominator per image with no extra forward pass
        # (the centerbias-only LL is just cb evaluated at each fixation).
        n_k = 0
        for k, i in enumerate(range(start_fixation, valid_end)):
            ll_model = _nearest_sample(log_d_np[k], xs[i], ys[i])
            ll_cb = _nearest_sample(cb, xs[i], ys[i])
            ll_nats_sum += ll_model
            cb_nats_sum += ll_cb
            img_model_nats_sum[stim_idx] += ll_model
            img_cb_nats_sum[stim_idx] += ll_cb
            if per_fixation:
                H_i, W_i = image.shape[0], image.shape[1]
                pf["stim"].append(stim_idx)
                pf["stimulus_id"].append(
                    str(stimulus_ids[stim_idx]) if stimulus_ids is not None else ""
                )
                pf["subject"].append(
                    int(train_subjects[sp_idx]) if train_subjects is not None else -1
                )
                pf["scanpath"].append(int(sp_idx))
                pf["fix_index"].append(i)
                # Amplitude of the saccade that produced this fixation. The
                # first evaluated fixation has a predecessor only when
                # start_fixation > 0; otherwise there is no incoming saccade.
                pf["sacc_px"].append(
                    float(math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]))
                    if i >= 1 else float("nan")
                )
                # Direction of that same saccade, degrees in (-180, 180].
                # Image y grows downward, so the y term is negated to make this
                # the ordinary counter-clockwise convention: 0 = rightward,
                # +90 = upward. NaN where sacc_px is NaN, so the two align.
                # This is a property of the human scanpath and is therefore
                # identical across arms — a stratifier (does the cost depend on
                # saccade direction?), not an outcome. The direction statistics
                # of DG3's own figures are about *sampled* paths, which is a
                # different measurement.
                pf["sacc_dir_deg"].append(
                    math.degrees(math.atan2(-(ys[i] - ys[i - 1]), xs[i] - xs[i - 1]))
                    if i >= 1 else float("nan")
                )
                # Distance of the target from the image centre. Correlated with
                # sacc_px but not the same thing, and the two must be separable:
                # without it, "the model lost its peripheral target" cannot be
                # told apart from "the centerbias prior did the work".
                pf["target_ecc_px"].append(
                    float(math.hypot(xs[i] - W_i / 2.0, ys[i] - H_i / 2.0))
                )
                pf["target_x"].append(float(xs[i]))
                pf["target_y"].append(float(ys[i]))
                pf["gaze_x"].append(float(gaze_xs[k]))
                pf["gaze_y"].append(float(gaze_ys[k]))
                pf["img_h"].append(int(H_i))
                pf["img_w"].append(int(W_i))
                pf["fix_t"].append(
                    float(train_ts[sp_idx][i])
                    if train_ts is not None and i < len(train_ts[sp_idx])
                    else float("nan")
                )
                pf["ll_bits"].append(ll_model / LOG2)
                pf["cb_bits"].append(ll_cb / LOG2)
            n_k += 1
        n_fix_total += n_k

        # Consumer B: the saliency metrics on the same tensor. Force
        # float32 because on CUDA the model may emit float64, while the
        # vendor metrics and the fix_mask weights below assume float32.
        log_d_t = torch.from_numpy(log_d_np).to(device).float()
        H, W = log_d_np.shape[1], log_d_np.shape[2]
        eval_xs = xs[start_fixation:valid_end]
        eval_ys = ys[start_fixation:valid_end]
        fix_mask = _build_dense_fixation_mask(eval_xs, eval_ys, H, W, device)
        weights = torch.ones(len(eval_xs), dtype=torch.float32, device=device)

        # These three return the mean over the scanpath's fixations.
        # Weighting by n_k before pooling into the image bucket recovers the
        # per-fixation mean within each image.
        ll_b = float(log_likelihood(log_d_t, fix_mask, weights=weights).item())
        nss_b = float(nss(log_d_t, fix_mask, weights=weights).item())
        auc_b = float(auc(log_d_t, fix_mask, weights=weights).item())

        img_ll_over_uniform_sum[stim_idx] += ll_b * n_k
        img_nss_sum[stim_idx] += nss_b * n_k
        img_auc_sum[stim_idx] += auc_b * n_k
        img_fix_count[stim_idx] += n_k

        if per_fixation:
            # The annuli are centred on each step's current gaze, so amp_mass is
            # directly comparable to sacc_px: same origin, same edges.
            diag = _per_fixation_diagnostics(log_d_t, eval_xs, eval_ys, gaze_xs, gaze_ys)
            for key, arr in diag.items():
                # amp_mass is (B, n_bins); every other diagnostic is (B,).
                pf[key].extend(list(arr) if arr.ndim > 1 else arr.tolist())

        if per_fixation and save_maps_for and stim_idx in save_maps_for:
            # Half-resolution copies for a small pre-chosen stimulus set. Full
            # maps for every fixation would be hundreds of terabytes; these
            # answer "what did the density actually look like" for figures and
            # post-hoc questions without another GPU pass.
            #
            # Stored as LOG-density, not probability. A typical per-pixel
            # probability on a 768x1024 frame is ~3e-7, which is subnormal in
            # float16: ~7% relative error and about five representable levels,
            # i.e. the map is destroyed for anything quantitative. Log-density
            # lands in ~[-20, -5], where float16 resolves ~0.01-0.02 nats.
            #
            # Sum-pooled (avg_pool2d * 4), so a 2x2 block carries the block's
            # total mass and the half-resolution map is still a distribution
            # summing to 1. Plain averaging integrated to 0.25.
            m = torch.nn.functional.avg_pool2d(log_d_t.exp().unsqueeze(1), 2).squeeze(1) * 4.0
            saved_maps.append({
                "stim": stim_idx, "scanpath": int(sp_idx),
                "fix_index": np.arange(start_fixation, valid_end),
                "log_density_half": m.clamp_min(1e-30).log().cpu().numpy().astype(np.float16),
            })

    stim_ids = sorted(img_fix_count.keys())
    n_images = len(stim_ids)

    per_img_ll_bits = [img_ll_over_uniform_sum[s] / img_fix_count[s] for s in stim_ids]
    per_img_nss = [img_nss_sum[s] / img_fix_count[s] for s in stim_ids]
    per_img_auc = [img_auc_sum[s] / img_fix_count[s] for s in stim_ids]
    per_img_ll_raw_bits = [
        (img_model_nats_sum[s] / img_fix_count[s]) / LOG2 for s in stim_ids
    ]
    per_img_ig_bits = [
        ((img_model_nats_sum[s] - img_cb_nats_sum[s]) / img_fix_count[s]) / LOG2
        for s in stim_ids
    ]

    def _mean_se(vals: list[float]) -> tuple[float, float]:
        if not vals:
            return float("nan"), float("nan")
        arr = np.asarray(vals, dtype=np.float64)
        mean = float(arr.mean())
        se = float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
        return mean, se

    ll_bits_mean, ll_bits_se = _mean_se(per_img_ll_bits)
    ll_raw_bits_mean, ll_raw_bits_se = _mean_se(per_img_ll_raw_bits)
    ig_bits_mean, ig_bits_se = _mean_se(per_img_ig_bits)
    nss_mean, nss_se = _mean_se(per_img_nss)
    auc_mean, auc_se = _mean_se(per_img_auc)

    ll_per_fix_nats = ll_nats_sum / n_fix_total if n_fix_total > 0 else float("nan")
    ll_cb_bits_pooled = (
        (cb_nats_sum / n_fix_total) / LOG2 if n_fix_total > 0 else float("nan")
    )

    out_per_fixation = (
        {k: np.asarray(v) for k, v in pf.items()} if per_fixation else None
    )
    if per_fixation:
        lengths = {k: len(v) for k, v in out_per_fixation.items()}
        if len(set(lengths.values())) != 1 or next(iter(lengths.values())) != n_fix_total:
            raise RuntimeError(
                f"per-fixation records are ragged or miscounted: {lengths}, "
                f"expected {n_fix_total} of each"
            )

    return {
        **({"per_fixation": out_per_fixation} if per_fixation else {}),
        **({"density_maps": saved_maps} if saved_maps else {}),
        **({"amp_edges_px": list(AMP_EDGES_PX)} if per_fixation else {}),
        # The values every mean and SE below are taken over, not a recomputation
        # of them — kept so a paired per-image analysis does not need a second
        # run. All four metrics, not just IG: NSS and AUC exist nowhere else at
        # any granularity finer than the cross-fold mean (the per-fixation dump
        # carries ll_bits/cb_bits but no saliency metrics), so without these the
        # standing claim that the cost falls on localization rather than ranking
        # could not be given an error bar without a full re-eval pass.
        "per_image": {"stim": stim_ids, "IG_bits": per_img_ig_bits,
                      "NSS": per_img_nss, "AUC": per_img_auc,
                      "LL_bits": per_img_ll_bits,
                      "LL_raw_bits": per_img_ll_raw_bits},
        "LL_bits_mean": ll_bits_mean,
        "LL_bits_se": ll_bits_se,
        "LL_raw_bits_mean": ll_raw_bits_mean,
        "LL_raw_bits_se": ll_raw_bits_se,
        "IG_bits_mean": ig_bits_mean,
        "IG_bits_se": ig_bits_se,
        "NSS_mean": nss_mean,
        "NSS_se": nss_se,
        "AUC_mean": auc_mean,
        "AUC_se": auc_se,
        "n_fixations": n_fix_total,
        "n_images": n_images,
        "n_scanpaths_evaluated": n_scanpaths_evaluated,
        "model": model_name,
        "fold_no": fold_no,
        "device": str(device),
        "seed": seed,
        "ll_per_fixation_nats": ll_per_fix_nats,
        "ll_centerbias_bits_pooled": ll_cb_bits_pooled,
    }


def save_per_fixation(result: dict, path, **meta) -> None:
    """Write one arm-fold's per-fixation records to ``path`` as a compressed npz.

    Raw ``ll_bits`` and ``cb_bits`` are stored rather than any difference, so
    every contrast — dIG against another arm, IG against the centerbias floor —
    is formed downstream from the same file, and IG stays auditable instead of
    trusted. ``stimulus_id`` and ``subject`` travel with the rows because they
    are the clustering units: a cluster-robust or mixed-effects analysis is
    impossible without them, and neither can be recovered from a summary.

    Roughly 1.5 MB per (arm, fold) at MIT1003 scale, so the whole 7-arm 10-fold
    design costs ~100 MB — small enough that no epoch needs to be discarded.

    ``density_maps`` (present only when ``run`` was given ``save_maps_for``) go
    to a sibling ``*_maps.npz``: they are per-scanpath and ragged, so they cannot
    share the flat per-fixation table. Budget these separately and deliberately —
    ~15 scanpaths per stimulus x ~6 evaluated fixations x 384x512 x 2 B is about
    35 MB per stimulus per arm, so a two-stimulus set across 7 arms is ~0.5 GB.
    """
    import json
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pf = result.get("per_fixation")
    if pf is None:
        raise ValueError("result has no per_fixation records; call run(per_fixation=True)")

    payload = {k: np.asarray(v) for k, v in pf.items()}
    payload["amp_edges_px"] = np.asarray(result.get("amp_edges_px", AMP_EDGES_PX))
    # Scalars the rows do not carry, so a concatenation of many files stays
    # self-describing. JSON rather than separate keys to keep arbitrary metadata
    # (arm, epoch, git sha, foveation geometry) without a schema migration.
    payload["meta_json"] = np.asarray(json.dumps({
        "model": result.get("model"), "fold_no": result.get("fold_no"),
        "seed": result.get("seed"), "device": result.get("device"),
        "n_fixations": result.get("n_fixations"), "n_images": result.get("n_images"),
        **meta,
    }))
    np.savez_compressed(path, **payload)

    maps = result.get("density_maps")
    if maps:
        flat = {}
        for j, m in enumerate(maps):
            flat[f"{j}_stim"] = np.asarray(m["stim"])
            flat[f"{j}_scanpath"] = np.asarray(m["scanpath"])
            flat[f"{j}_fix_index"] = np.asarray(m["fix_index"])
            flat[f"{j}_log_density_half"] = m["log_density_half"]
        flat["n_groups"] = np.asarray(len(maps))
        np.savez_compressed(path.with_name(path.stem + "_maps.npz"), **flat)
