"""Human-scanpath retrieval and summary statistics for MIT1003.

Pulls scanpaths from a ``pysaliency`` ``FixationTrains``-style object (the
``train_xs / train_ys / train_ns / train_subjects`` API surface) and exposes:

- :func:`all_human_scanpaths` — every subject's scanpath for one stimulus.
- :func:`pick_human_scanpath` — single scanpath by subject (or first).
- :func:`human_length_for_stim` — median/mean fixation count across subjects,
  intended as the per-stimulus length target for AI-side normalization.
- :func:`inter_subject_dispersion` — scanpath-length + centroid-spread summary.

Everything is ``numpy``-only and side-effect free; this module deliberately
does not import torch / matplotlib so it can be used inside evaluation
scripts without paying the model-load cost.
"""
from __future__ import annotations

import numpy as np

# The one home for the consensus radius: ch03 §Consensus regions states
# "R = 70 pixels throughout", and every figure script and notebook 02 import
# it from here rather than carrying a copy.
#
# 70 px is 2 degrees at MIT1003's viewing geometry (ppd 35), the foveal radius
# of ch02, so a pixel within R of a fixation is one the subject saw foveally.
# Written as a literal rather than derived from foveate_input.MIT1003_PPD
# because this module imports numpy only, which is what lets evaluation scripts
# use it without paying for torch.
CONSENSUS_RADIUS_PX = 70


def all_human_scanpaths(fixations, stim_idx: int) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Return ``[(xs, ys, subject_idx), ...]`` for every scanpath on ``stim_idx``.

    NaN-padded fixations are filtered out per scanpath. Empty scanpaths are
    included so callers can decide whether to drop them.
    """
    ns = np.asarray(fixations.train_ns)
    mask = np.where(ns == stim_idx)[0]
    if len(mask) == 0:
        raise RuntimeError(f"no human scanpaths for stim {stim_idx}")
    subjects = (np.asarray(fixations.train_subjects)
                if hasattr(fixations, "train_subjects") else None)
    out: list[tuple[np.ndarray, np.ndarray, int]] = []
    for sp in mask:
        xs = np.asarray(fixations.train_xs[sp], dtype=float)
        ys = np.asarray(fixations.train_ys[sp], dtype=float)
        valid = ~(np.isnan(xs) | np.isnan(ys))
        subj = int(subjects[sp]) if subjects is not None else -1
        out.append((xs[valid], ys[valid], subj))
    return out


def pick_human_scanpath(
    fixations,
    stim_idx: int,
    subject_idx: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return one human scanpath for ``stim_idx``.

    If ``subject_idx`` is provided and present, returns that subject's scanpath;
    otherwise returns the first available.
    """
    paths = all_human_scanpaths(fixations, stim_idx)
    if subject_idx is not None:
        for xs, ys, subj in paths:
            if subj == subject_idx:
                return xs, ys, subj
    return paths[0]


def human_length_for_stim(
    fixations,
    stim_idx: int,
    agg: str = "median",
    *,
    min_len: int = 2,
) -> int:
    """Per-stimulus human scanpath length aggregate, used to length-match AI scans.

    ``agg`` is one of ``"median"``, ``"mean"``, ``"max"``, ``"min"``. Empty/short
    scanpaths below ``min_len`` are discarded before aggregation.
    """
    paths = all_human_scanpaths(fixations, stim_idx)
    lengths = np.array([len(xs) for xs, _, _ in paths if len(xs) >= min_len],
                       dtype=float)
    if len(lengths) == 0:
        raise RuntimeError(
            f"stim {stim_idx}: no scanpaths with length >= {min_len}"
        )
    if agg == "median":
        return int(round(float(np.median(lengths))))
    if agg == "mean":
        return int(round(float(lengths.mean())))
    if agg == "max":
        return int(lengths.max())
    if agg == "min":
        return int(lengths.min())
    raise ValueError(f"unknown agg={agg!r}; expected median|mean|max|min")


def consensus_count(
    paths: list[tuple[np.ndarray, np.ndarray, int]],
    H: int,
    W: int,
    radius: int,
) -> np.ndarray:
    """Per-pixel count of *distinct subjects* with any fixation within ``radius``.

    Returns int array of shape (H, W). Within-subject duplicates do not
    inflate the count: each subject's fixation disks are OR-ed first, then
    summed across subjects.
    """
    count = np.zeros((H, W), dtype=np.int32)
    for xs, ys, _ in paths:
        mask = np.zeros((H, W), dtype=bool)
        for x, y in zip(xs, ys):
            cy, cx = int(round(float(y))), int(round(float(x)))
            y0, y1 = max(0, cy - radius), min(H, cy + radius + 1)
            x0, x1 = max(0, cx - radius), min(W, cx + radius + 1)
            if y1 <= y0 or x1 <= x0:
                continue
            yg, xg = np.ogrid[y0:y1, x0:x1]
            disk = (yg - cy) ** 2 + (xg - cx) ** 2 <= radius ** 2
            mask[y0:y1, x0:x1] |= disk
        count += mask.astype(np.int32)
    return count


def inter_subject_dispersion(
    paths: list[tuple[np.ndarray, np.ndarray, int]],
    H: int,
    W: int,
) -> dict:
    """Compact dispersion summary across subjects for one stimulus.

    Returns scanpath-length stats and the mean pairwise distance between
    per-subject fixation centroids, normalized by the image diagonal.
    """
    lengths = np.array([len(xs) for xs, _, _ in paths], dtype=float)
    centroids = np.array([(xs.mean(), ys.mean()) for xs, ys, _ in paths
                          if len(xs) > 0])
    diag = float(np.hypot(H, W))
    if len(centroids) < 2:
        # A mean pairwise distance needs a pair. Zero would read as perfect
        # agreement, which is the opposite of what one centroid supports.
        centroid_spread = float("nan")
    else:
        d = np.sqrt(((centroids[:, None] - centroids[None, :]) ** 2).sum(-1))
        iu = np.triu_indices(len(centroids), k=1)
        centroid_spread = float(d[iu].mean())
    return {
        "n_subjects": int(len(paths)),
        "scanpath_length_mean": float(lengths.mean()),
        "scanpath_length_median": float(np.median(lengths)),
        "scanpath_length_std": float(lengths.std(ddof=0)),
        "scanpath_length_min": int(lengths.min()),
        "scanpath_length_max": int(lengths.max()),
        "centroid_pairwise_dist_mean_px": centroid_spread,
        "centroid_pairwise_dist_mean_norm": centroid_spread / diag,
    }


# The one home for the fixation-entropy grid, for the same reason as
# CONSENSUS_RADIUS_PX above. ch03 states the bin count as a claim ("a 16x16
# grid ... log2 K = 8"), and the figure that draws the grid and the diagnostic
# that reports the scalar have to agree by construction, not by two matching
# literals. Callers pass paths already stripped of the procedural central
# fixation, as per_image_diagnostic does under the `initial` variant.
ENTROPY_BINS = 16


def fixation_entropy(
    paths: list[tuple[np.ndarray, np.ndarray, int]], height: int, width: int
) -> tuple[np.ndarray, float, float]:
    """Population fixation histogram, its entropy in bits, and that normalised.

    Subjects with no recorded fixation on the stimulus are dropped: empty is
    missing data, not agreement. The normaliser is log2(ENTROPY_BINS**2), so the
    third return value lies in [0, 1] and rises as the population's gaze spreads.
    """
    paths = [p for p in paths if len(p[0]) > 0]
    if not paths:
        return np.zeros((ENTROPY_BINS, ENTROPY_BINS)), float("nan"), float("nan")
    all_xs = np.concatenate([xs for xs, _, _ in paths])
    all_ys = np.concatenate([ys for _, ys, _ in paths])
    hist, _, _ = np.histogram2d(
        all_xs, all_ys, bins=ENTROPY_BINS, range=[[0, width], [0, height]],
    )
    total = hist.sum()
    if total <= 0:
        return hist, float("nan"), float("nan")
    p = hist / total
    nz = p[p > 0]
    bits = float(-np.sum(nz * np.log2(nz)))
    return hist, bits, bits / np.log2(ENTROPY_BINS**2)
