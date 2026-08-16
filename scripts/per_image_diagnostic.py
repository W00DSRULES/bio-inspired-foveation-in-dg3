"""Per-image diagnostic: human consensus difficulty vs DG3 performance.

Bridges the qualitative consensus gallery to the headline quantitative
metrics. For each sampled MIT1003 image, we compute:

  Human "difficulty" metrics — capturing how dispersed / disagreed-on the
  population's gaze is on this image:

    - centroid_spread_norm    mean pairwise distance between per-subject
                              fixation centroids, divided by image diagonal.
                              Higher = subjects' attention centres are more
                              spread out across the image.
    - fixation_entropy_norm   Shannon entropy of the population fixation
                              density (16×16 grid), divided by log2(256) so
                              it lives in [0, 1]. Higher = fixations are
                              more uniformly distributed across the image.
    - consensus_area_75_pct   fraction of image area inside the ≥ 75 %
                              subject-overlap consensus region. Higher =
                              broader consensus. Note: this is bounded
                              above by the rest of the image; many abstract
                              stimuli have 0 % at the 75 % threshold.

  DG3 per-image performance (averaged across all subjects' fixations on
  this image, pooled; scored from the second fixation on, the first being
  history):

    - ll_bits      mean log-density at human fixations, in bits
    - ig_bits      ll_bits − centerbias_bits (information gain)
    - nss          mean normalized scanpath saliency
    - auc          mean AUC vs uniform negatives

Outputs:

  results/per_image_diagnostic/diagnostic.json   per-image rows + correlations
  results/per_image_diagnostic/diagnostic.md     readable summary + corr table
  results/per_image_diagnostic/scatter.png       2×2 scatter grid: fixation
                                                 entropy against all four
                                                 performance metrics

Usage:
    python scripts/per_image_diagnostic.py
    python scripts/per_image_diagnostic.py --n-stim 50
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import deepgaze_pytorch
import pysaliency
from deepgaze_pytorch.metrics import auc as vendor_auc
from deepgaze_pytorch.metrics import log_likelihood as vendor_ll

from tez_deepgaze.baseline import _nearest_sample
from tez_deepgaze.centerbias import load_centerbias_for_image
from tez_deepgaze.device import pick_device, to_device
# nss is the corrected local one, not deepgaze_pytorch.metrics.nss.
from tez_deepgaze.evaluate import _build_dense_fixation_mask, nss
from tez_deepgaze.figstyle import canvas
from tez_deepgaze.human_scanpaths import (
    CONSENSUS_RADIUS_PX,
    ENTROPY_BINS,
    all_human_scanpaths,
    consensus_count,
    fixation_entropy,
    inter_subject_dispersion,
)
from tez_deepgaze.ig import LOG2
from tez_deepgaze.instrument import compute_log_density_batch
from tez_deepgaze.paths import DATA_ROOT
from tez_deepgaze.script_utils import image_rgb, sample_indices

OUT = Path(__file__).resolve().parents[1] / "results" / "per_image_diagnostic"
MIN_PATH_LEN = 3


def difficulty_metrics(paths, H: int, W: int) -> dict[str, float]:
    """Three per-image difficulty measures (higher = more dispersed / harder)."""
    # A subject with no recorded fixation on this image contributes to none of
    # the three measures, but would still count towards the consensus
    # thresholds below, which are fractions of the subject count. MIT1003 holds
    # 129 such empty trains, and at ceil(0.90 N) each one makes the 90 % mask
    # unreachable on the image that carries it. Empty is missing data, not
    # disagreement, so it is dropped before N is taken.
    paths = [p for p in paths if len(p[0]) > 0]

    # 1. Centroid spread across subjects
    centroid_spread_norm = inter_subject_dispersion(
        paths, H, W)["centroid_pairwise_dist_mean_norm"]

    # 2. Population fixation density entropy
    _, entropy_bits, entropy_norm = fixation_entropy(paths, H, W)

    # 3. 75% consensus area (this is *inverse* difficulty — bigger = more
    #    agreement). We keep it as a raw percentage; the correlation is what
    #    we read.
    count = consensus_count(paths, H, W, CONSENSUS_RADIUS_PX)
    n_subj = len(paths)
    thr75 = int(np.ceil(0.75 * n_subj))
    thr90 = int(np.ceil(0.90 * n_subj))
    consensus75_pct = float(100.0 * (count >= thr75).sum() / (H * W))
    consensus90_pct = float(100.0 * (count >= thr90).sum() / (H * W))

    return {
        "centroid_spread_norm": centroid_spread_norm,
        "fixation_entropy_bits": entropy_bits,
        "fixation_entropy_norm": entropy_norm,
        "consensus_area_75_pct": consensus75_pct,
        "consensus_area_90_pct": consensus90_pct,
        "n_subjects": int(n_subj),
    }


@torch.no_grad()
def dg3_perf_one_image(model, image, cb, paths, device,
                       *, start_fixation: int = 1) -> dict[str, float]:
    """Mean LL (bits) / IG / NSS / AUC per image, pooled over all subject fixations.

    Matches ``evaluate.run``: ``start_fixation=1`` scores from the second
    fixation on, using the first as history.
    """
    H, W = image.shape[:2]
    ll_nats_sum, cb_nats_sum, n_fix = 0.0, 0.0, 0
    nss_vals, auc_vals, ll_b_vals = [], [], []

    for xs, ys, _ in paths:
        valid_end = len(xs)
        if valid_end <= start_fixation:
            continue
        hist_x = [list(xs[:i]) for i in range(start_fixation, valid_end)]
        hist_y = [list(ys[:i]) for i in range(start_fixation, valid_end)]
        log_d_np = compute_log_density_batch(model, image, cb, hist_x, hist_y, device)

        # Per-fixation LL via the same nearest-pixel sample as baseline.py.
        for k, i in enumerate(range(start_fixation, valid_end)):
            ll_nats_sum += _nearest_sample(log_d_np[k], xs[i], ys[i])
            cb_nats_sum += _nearest_sample(cb, xs[i], ys[i])
            n_fix += 1

        log_d_t = torch.from_numpy(log_d_np).to(device).float()
        eval_xs = xs[start_fixation:valid_end]
        eval_ys = ys[start_fixation:valid_end]
        fix_mask = _build_dense_fixation_mask(eval_xs, eval_ys, H, W, device)
        weights = torch.ones(len(eval_xs), dtype=torch.float32, device=device)
        nss_vals.append(float(nss(log_d_t, fix_mask, weights=weights).item()))
        auc_vals.append(float(vendor_auc(log_d_t, fix_mask, weights=weights).item()))
        ll_b_vals.append(float(vendor_ll(log_d_t, fix_mask, weights=weights).item()))

    if n_fix == 0:
        nan = float("nan")
        return {"ll_bits": nan, "cb_bits": nan, "ig_bits": nan,
                "nss": nan, "auc": nan, "n_fix": 0, "n_scanpaths": 0}
    return {
        "ll_bits": float((ll_nats_sum / n_fix) / LOG2),
        "cb_bits": float((cb_nats_sum / n_fix) / LOG2),
        "ig_bits": float(((ll_nats_sum - cb_nats_sum) / n_fix) / LOG2),
        "ll_bits_vendor_mean": float(np.mean(ll_b_vals)),
        "nss": float(np.mean(nss_vals)),
        "auc": float(np.mean(auc_vals)),
        "n_fix": int(n_fix),
        "n_scanpaths": int(len(nss_vals)),
    }


def _pearson_spearman(x, y):
    """Return (pearson_r, spearman_r, n) after dropping nan pairs."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 3:
        return float("nan"), float("nan"), n
    xs, ys = x[m], y[m]
    pearson = float(np.corrcoef(xs, ys)[0, 1])
    # Spearman = Pearson on ranks. Use argsort+argsort for tiebreaks.
    rx = np.argsort(np.argsort(xs))
    ry = np.argsort(np.argsort(ys))
    spearman = float(np.corrcoef(rx, ry)[0, 1])
    return pearson, spearman, n


# Centroid spread is recorded per image but not correlated: at n = 50 its
# interval spans nearly zero to 0.5. The thesis reports the two scalars below.
DIFFICULTY_KEYS = (
    "fixation_entropy_norm",
    "consensus_area_75_pct",
)
PERF_KEYS = ("ig_bits", "nss", "auc", "ll_bits", "ll_uniform_bits")


def _perf_value(row: dict, pk: str) -> float:
    """One per-image metric. ``ll_bits`` is the raw log-density in bits; the thesis
    defines log-likelihood against a uniform map (ch03 eq. ll), which adds
    log2(H*W) of that image, so ``ll_uniform_bits`` is derived here rather than
    stored — the JSON keeps the raw value it was written with."""
    if pk == "ll_uniform_bits":
        H, W = row["image_shape_hw"]
        return float(row["dg3"]["ll_bits"] + np.log2(H * W))
    return row["dg3"][pk]


def correlations(rows: list[dict]) -> dict:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for dk in DIFFICULTY_KEYS:
        out[dk] = {}
        for pk in PERF_KEYS:
            xs = [r["difficulty"][dk] for r in rows]
            ys = [_perf_value(r, pk) for r in rows]
            p, s, n = _pearson_spearman(xs, ys)
            out[dk][pk] = {"pearson": p, "spearman": s, "n": n}
    return out


def _scatter_grid(rows: list[dict], corr: dict, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, **canvas((12, 10)))

    # The scalar the chapter argues from, against all four metrics. Consensus
    # area is computed and recorded in diagnostic.json but not plotted: the two
    # measure the same property and the entropy is the stronger of them on every
    # metric, so plotting both would show one relation twice. The consensus
    # regions remain the figures' way of drawing agreement.
    plots = [
        ("fixation_entropy_norm", "ig_bits",
         "Fixation entropy (norm)", "IG over centerbias (bits/fix)"),
        ("fixation_entropy_norm", "ll_uniform_bits",
         "Fixation entropy (norm)", "LL over uniform (bits/fix)"),
        ("fixation_entropy_norm", "nss",
         "Fixation entropy (norm)", "NSS"),
        ("fixation_entropy_norm", "auc",
         "Fixation entropy (norm)", "AUC"),
    ]
    for ax, (dk, pk, xlbl, ylbl) in zip(axes.ravel(), plots):
        xs = np.array([r["difficulty"][dk] for r in rows], dtype=float)
        ys = np.array([_perf_value(r, pk) for r in rows], dtype=float)
        m = np.isfinite(xs) & np.isfinite(ys)
        xs, ys = xs[m], ys[m]
        ax.scatter(xs, ys, alpha=0.7, s=36, color="#1d6fb8", edgecolor="white")
        if len(xs) >= 2:
            coef = np.polyfit(xs, ys, 1)
            xline = np.linspace(xs.min(), xs.max(), 200)
            yline = np.polyval(coef, xline)
            if pk == "auc":
                # AUC cannot exceed one.
                keep = yline <= 1.0
                xline, yline = xline[keep], yline[keep]
            ax.plot(xline, yline, "-", color="#e63946", linewidth=1.4, alpha=0.85)
        c = corr[dk][pk]
        # Both coefficients go in the panel and the axis labels name the pair, so
        # the title does not repeat the x variable. Three decimals, the precision
        # the chapter quotes.
        ax.set_title(
            f"$r = {c['pearson']:+.3f}$   "
            f"$\\rho = {c['spearman']:+.3f}$   $n = {c['n']}$",
            fontsize=10,
        )
        ax.set_xlabel(xlbl)
        ax.set_ylabel(ylbl)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi="figure", bbox_inches="tight")
    plt.close(fig)


def _render_markdown(meta: dict, rows: list[dict], corr: dict) -> str:
    lines = [
        "# Per-image diagnostic — human-consensus difficulty vs DG3 performance",
        "",
        f"- n_stim evaluated: **{meta['n_stim_used']}** "
        f"(of {meta['n_stim_requested']} sampled)",
        f"- consensus radius: {CONSENSUS_RADIUS_PX} px",
        f"- entropy bin grid: {ENTROPY_BINS} × {ENTROPY_BINS}",
        f"- start_fixation: {meta['start_fixation']} "
        f"(scored from the second fixation; the first is history, not a target)",
        f"- device: {meta['device']}",
        "",
        "## Correlations (per-image)",
        "",
        "Pearson r — linear correlation. Spearman rho — rank correlation, "
        "robust to outliers. A negative r between *difficulty* and "
        "*performance* means the model does worse on harder images.",
        "",
        "| difficulty \\ performance | "
        + " | ".join(PERF_KEYS) + " |",
        "|---|" + "|".join("---" for _ in PERF_KEYS) + "|",
    ]
    for dk in DIFFICULTY_KEYS:
        cells = []
        for pk in PERF_KEYS:
            c = corr[dk][pk]
            cells.append(f"r = {c['pearson']:+.2f} (ρ = {c['spearman']:+.2f})")
        lines.append(f"| **{dk}** | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Reading the table",
        "",
        "Sign convention: `fixation_entropy_norm` goes up as humans spread out "
        "/ disagree (harder images), while `consensus_area_75_pct` goes up when "
        "subjects converge (easier images). So the *expected* directions of "
        "correlation with DG3 performance are:",
        "",
        "- fixation_entropy vs perf → **negative** (harder → lower perf)",
        "- consensus_area_75 vs perf → **positive**",
        "",
        "If the observed sign is opposite, that means DG3 happens to perform "
        "*better* exactly where the population is most dispersed — usually "
        "because the centerbias baseline performs even worse on those images "
        "(no concentrated target → centerbias is the wrong prior).",
        "",
        "## Scatter plots",
        "",
        "![scatter](scatter.png)",
        "",
        "## Per-image rows",
        "",
        "| stim | n_subj | centroid_spread | entropy_norm | consensus_75_% | "
        "IG bits | NSS | AUC |",
        "|---|---|---|---|---|---|---|---|",
    ]
    # Sort by IG ascending so the worst-performing images are at the top
    for r in sorted(rows, key=lambda r: r["dg3"]["ig_bits"]):
        d = r["difficulty"]
        p = r["dg3"]
        lines.append(
            f"| {r['stim_idx']}"
            f" | {d['n_subjects']}"
            f" | {d['centroid_spread_norm']:.3f}"
            f" | {d['fixation_entropy_norm']:.3f}"
            f" | {d['consensus_area_75_pct']:.2f}"
            f" | {p['ig_bits']:.3f}"
            f" | {p['nss']:.3f}"
            f" | {p['auc']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def run(args) -> None:
    device = pick_device()
    print(f"device: {device}")

    print("loading MIT1003...")
    from tez_deepgaze.paths import load_mit1003_variant

    stimuli, fixations = load_mit1003_variant(args.dataset_variant)
    total = len(stimuli)
    stim_idxs = sample_indices(args.n_stim, total, args.seed)
    print(f"stimuli: {len(stim_idxs)} (of {total})")

    print("loading pretrained DeepGaze III...")
    model = to_device(deepgaze_pytorch.DeepGazeIII(pretrained=True), device).eval()

    rows: list[dict] = []
    t0 = time.time()
    for k, sidx in enumerate(stim_idxs):
        image = image_rgb(stimuli, sidx)
        H, W = image.shape[:2]
        cb = load_centerbias_for_image(H, W)
        paths = all_human_scanpaths(fixations, sidx)
        # Human dispersion is always computed on free fixations. Under the
        # initial variant the recorded paths carry the procedural central
        # fixation at index 0; stripping it here reconstructs the plain paths
        # (validated to 3e-14 px by fetch_mit1003 --with-initial), so the
        # scalars are identical across variants while the model eval below
        # still sees the loaded variant's full paths as history.
        strip = 1 if args.dataset_variant == "initial" else 0
        pairs = [((xs[strip:], ys[strip:], s), (xs, ys, s)) for xs, ys, s in paths]
        usable = [(h, f) for h, f in pairs if len(h[0]) >= MIN_PATH_LEN]
        if len(usable) < 2:
            continue
        # The dispersion scalars take every subject who fixated the image; the
        # model metrics take only those with MIN_PATH_LEN fixations to score.
        # eq:consensus-masks thresholds at ceil(0.75 N), and N is meant to count
        # who looked -- how long a scanpath ran is a property of the scoring.
        diff = difficulty_metrics([h for h, _ in pairs], H, W)
        perf = dg3_perf_one_image(
            model, image, cb, [f for _, f in usable], device,
            start_fixation=args.start_fixation,
        )
        if perf["n_fix"] == 0:
            continue
        rows.append({
            "stim_idx": int(sidx),
            "image_shape_hw": [int(H), int(W)],
            "difficulty": diff,
            "dg3": perf,
        })
        elapsed = time.time() - t0
        eta = (len(stim_idxs) - (k + 1)) * elapsed / (k + 1)
        print(f"  [{k+1}/{len(stim_idxs)}] stim {sidx} "
              f"n_subj={diff['n_subjects']:>2} "
              f"IG = {perf['ig_bits']:+.3f} bits  NSS = {perf['nss']:+.3f}  "
              f"AUC = {perf['auc']:.3f}  "
              f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

    corr = correlations(rows)

    meta = {
        "n_stim_requested": args.n_stim,
        "n_stim_used": len(rows),
        "seed": args.seed,
        "start_fixation": args.start_fixation,
        "dataset_variant": args.dataset_variant,
        "consensus_radius_px": CONSENSUS_RADIUS_PX,
        "entropy_bins": ENTROPY_BINS,
        # Gates which scanpaths DG3 is scored on; the difficulty scalars beside
        # them take every subject who fixated the image.
        "model_min_path_len": MIN_PATH_LEN,
        "scalars_refreshed": False,
        "device": str(device),
        "model": "DeepGazeIII(pretrained=True)",
    }

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "diagnostic.json").write_text(json.dumps({
        "meta": meta, "rows": rows, "correlations": corr,
    }, indent=2))
    (out / "diagnostic.md").write_text(_render_markdown(meta, rows, corr))
    _scatter_grid(rows, corr, out / "scatter.png")
    print(f"\nwrote {out/'diagnostic.json'}, {out/'diagnostic.md'}, "
          f"{out/'scatter.png'}")


def run_refresh_scalars(args) -> None:
    """Recompute the difficulty scalars of a committed diagnostic.json in place.

    The scalars are a property of the human fixations and the constants at the
    top of this file; the DG3 metrics beside them are a property of the model
    and the stimulus. Changing a constant like ``CONSENSUS_RADIUS_PX`` therefore
    invalidates half of each row, and re-running :func:`run` to recover it would
    spend a GPU pass recomputing the half that cannot have moved. This reloads
    the fixations for the stimuli already in the file, rewrites their
    ``difficulty`` block, and redraws — no model, no image pixels.
    """
    from tez_deepgaze.paths import load_mit1003_variant

    path = args.out / "diagnostic.json"
    d = json.loads(path.read_text())
    # The variant the file was produced under, not the CLI default: refreshing a
    # row must load the same fixations run() did. Files without the field are
    # plain.
    variant = d["meta"].get("dataset_variant", "plain")
    print(f"loading MIT1003 fixations ({variant})...")
    _, fix = load_mit1003_variant(variant)
    strip = 1 if variant == "initial" else 0  # see run(): scalars are free-fixation only
    for row in d["rows"]:
        H, W = row["image_shape_hw"]
        paths = all_human_scanpaths(fix, row["stim_idx"])
        row["difficulty"] = difficulty_metrics(
            [(xs[strip:], ys[strip:], s) for xs, ys, s in paths], H, W)
    d["correlations"] = correlations(d["rows"])
    # Every constant the rewritten block depends on, so the file cannot claim a
    # filter it was not computed under, plus the flag saying it was rewritten.
    d["meta"]["consensus_radius_px"] = CONSENSUS_RADIUS_PX
    d["meta"]["entropy_bins"] = ENTROPY_BINS
    d["meta"].pop("min_path_len", None)
    d["meta"]["model_min_path_len"] = MIN_PATH_LEN
    d["meta"]["scalars_refreshed"] = True
    path.write_text(json.dumps(d, indent=2))
    (args.out / "diagnostic.md").write_text(
        _render_markdown(d["meta"], d["rows"], d["correlations"]))
    _scatter_grid(d["rows"], d["correlations"], args.out / "scatter.png")
    print(f"refreshed scalars at R = {CONSENSUS_RADIUS_PX} px for "
          f"{len(d['rows'])} stimuli; rewrote {path}, diagnostic.md, scatter.png")


def run_human_only(args) -> None:
    """Dataset-level human dispersion scalars, no model, no image pixels.

    Deliberately loads the plain MIT1003 variant regardless of which protocol
    the model evaluations use: the forced central start fixation is procedural,
    and including it would inflate consensus and depress spread at the image
    centre for every stimulus alike. Human-behaviour statistics are computed on
    free fixations only.
    """
    print("loading MIT1003...")
    stimuli, fixations = pysaliency.get_mit1003(location=str(DATA_ROOT))
    total = len(stimuli)
    sizes = list(stimuli.sizes)
    rows: list[dict] = []
    t0 = time.time()
    for sidx in range(total):
        H, W = sizes[sidx]
        paths = all_human_scanpaths(fixations, sidx)
        if len(paths) < 2:
            continue
        # Every subject who fixated the image, for the reason given in run():
        # the consensus threshold is a fraction of N, so applying the scoring
        # filter to the human side would move it per image.
        diff = difficulty_metrics(paths, int(H), int(W))
        rows.append({"stim_idx": int(sidx),
                     "image_shape_hw": [int(H), int(W)], **diff})
        if (sidx + 1) % 200 == 0:
            print(f"  [{sidx + 1}/{total}] ({time.time() - t0:.0f}s)")

    def _q(key: str) -> dict[str, float]:
        v = np.array([r[key] for r in rows], dtype=float)
        v = v[np.isfinite(v)]
        qs = np.percentile(v, [0, 25, 50, 75, 100])
        return {"min": float(qs[0]), "q25": float(qs[1]), "median": float(qs[2]),
                "q75": float(qs[3]), "max": float(qs[4]),
                "mean": float(v.mean()),
                "zero_fraction": float((v == 0).mean())}

    payload = {
        "meta": {
            "n_stim": len(rows),
            "consensus_radius_px": CONSENSUS_RADIUS_PX,
            "entropy_bins": ENTROPY_BINS,
            "dataset_variant": "plain",
            "note": "free fixations only; the procedural central start fixation "
                    "is excluded by construction. Every subject who fixated an "
                    "image counts towards N; no scanpath-length filter applies",
        },
        "summary": {k: _q(k) for k in (
            "centroid_spread_norm", "fixation_entropy_norm",
            "consensus_area_75_pct", "consensus_area_90_pct")},
        "rows": rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "human_scalars.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {path} ({len(rows)} stimuli)")


def run_join(args) -> None:
    """Correlate the dataset-level human scalars with per-image arm metrics.

    ``--join`` points at a sweep-table output directory (its
    ``per_image_ig.json`` carries per-image IG/NSS/AUC/LL per arm, folds
    concatenated so every stimulus appears once). Reports, per arm, the
    scalar-vs-metric correlations at full-dataset n, and additionally the
    correlation of each scalar with the per-image ΔIG of every non-normal arm
    against the sharp control — the question the 50-stimulus subsample could
    not ask. Run it on val output; the test split is read once, for the
    primary contrast only.
    """
    scal_path = args.out / "human_scalars.json"
    scalars = {r["stim_idx"]: r for r in json.loads(scal_path.read_text())["rows"]}
    src = json.loads((args.join / "per_image_ig.json").read_text())

    def _arm_by_stim(label: str) -> dict[int, dict[str, float]]:
        by_stim: dict[int, dict[str, float]] = {}
        for entry in src["arms"][label]:
            for i, s in enumerate(entry["stim"]):
                by_stim[int(s)] = {k: float(entry[k][i])
                                   for k in ("IG_bits", "NSS", "AUC", "LL_bits")}
        return by_stim

    scalar_keys = ("centroid_spread_norm", "fixation_entropy_norm",
                   "consensus_area_75_pct")
    out: dict = {"source": str(args.join), "split": src.get("split"),
                 "n_scalar_stimuli": len(scalars), "arms": {}, "delta_ig_vs_normal": {}}
    normal = _arm_by_stim("normal")
    for label in src["arms"]:
        by_stim = _arm_by_stim(label)
        common = sorted(set(by_stim) & set(scalars))
        out["arms"][label] = {
            sk: {mk: dict(zip(("pearson", "spearman", "n"), _pearson_spearman(
                [scalars[s][sk] for s in common],
                [by_stim[s][mk] for s in common])))
                 for mk in ("IG_bits", "NSS", "AUC", "LL_bits")}
            for sk in scalar_keys
        }
        if label != "normal":
            common_d = sorted(set(by_stim) & set(normal) & set(scalars))
            deltas = [by_stim[s]["IG_bits"] - normal[s]["IG_bits"] for s in common_d]
            out["delta_ig_vs_normal"][label] = {
                sk: dict(zip(("pearson", "spearman", "n"), _pearson_spearman(
                    [scalars[s][sk] for s in common_d], deltas)))
                for sk in scalar_keys
            }
    path = args.out / f"join_{args.join.parent.name}_{args.join.name}.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("source", "split")}, indent=2))
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-stim", type=int, default=50,
                    help="how many MIT1003 stimuli to evaluate (default 50; "
                         "use --n-stim 1003 for the full corpus)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start-fixation", type=int, default=1,
                    help="first scanpath index to evaluate (default 1, which "
                         "uses fixation 0 as history rather than scoring it)")
    ap.add_argument("--dataset-variant", choices=["plain", "initial"], default="plain",
                    help="model-eval path only ('initial' scores every free "
                         "fixation with the centre as history); the human "
                         "scalars of --human-only always use free fixations")
    # The four alternatives to a full run. They are alternatives, not flags that
    # compose: main() dispatches to exactly one.
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--human-only", action="store_true",
                      help="compute the human dispersion scalars for every "
                           "stimulus (no model, no GPU) and write "
                           "human_scalars.json; ignores --n-stim/--seed")
    mode.add_argument("--join", type=Path, default=None, metavar="EVAL_DIR",
                      help="correlate human_scalars.json with EVAL_DIR/"
                           "per_image_ig.json (a sweep-table output directory); "
                           "use a val directory — test is read once, for the "
                           "primary contrast")
    ap.add_argument("--out", type=Path, default=OUT,
                    help="output directory (default results/per_image_diagnostic); "
                         "any run overwrites it in place, so point a reduced-n "
                         "smoke run elsewhere")
    mode.add_argument("--replot", action="store_true",
                      help="redraw scatter.png from the committed diagnostic.json "
                           "without loading the model or the corpus")
    mode.add_argument("--refresh-scalars", action="store_true",
                      help="recompute the human difficulty scalars of the "
                           "committed diagnostic.json in place after a constant "
                           "such as CONSENSUS_RADIUS_PX changes, reusing its DG3 "
                           "metrics (no model, no GPU)")
    args = ap.parse_args()
    if args.refresh_scalars:
        run_refresh_scalars(args)
    elif args.replot:
        # The correlations block is recomputed from the rows so that the panel
        # titles, diagnostic.json and diagnostic.md carry the same values; the
        # rows themselves are not touched.
        path = args.out / "diagnostic.json"
        d = json.loads(path.read_text())
        d["correlations"] = correlations(d["rows"])
        path.write_text(json.dumps(d, indent=2))
        (args.out / "diagnostic.md").write_text(
            _render_markdown(d["meta"], d["rows"], d["correlations"]))
        _scatter_grid(d["rows"], d["correlations"], args.out / "scatter.png")
        print(f"wrote {args.out/'scatter.png'} from {path}; correlations block refreshed")
    elif args.human_only:
        run_human_only(args)
    elif args.join is not None:
        run_join(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
