"""What else changes along the scanpath, beside the paired likelihood difference.

The cutoff-10 cost shrinks from the first scored fixation to the ninth
(\\subsecref{results-diagnostics}). Two explanations for that are checkable
against the per-fixation records rather than arguable, and this checks them.

  1. Saccade amplitude. The cost also deepens with amplitude, so if early
     fixations carried the longer saccades the index trend would be the
     amplitude trend under another name. Mean amplitude per index bin settles
     it -- and it rises along the scanpath, which is the opposite of what that
     explanation needs.

  2. The level. The control's own information gain falls along the scanpath, so
     part of the shrinking cost is simply that there is less left to lose. The
     difference expressed as a share of the level separates the two.

Reads the per-fixation dumps ``foveation_stratified.py`` writes, which are far
too large to commit (25 MB per cutoff), and aggregates them into a small JSON
that is committed, the same arrangement ``collect_training_curves.py`` uses:

    python scripts/index_profile.py

Both arms of a contrast are scored on identical fixations, so the difference is
paired per fixation; the reported uncertainty is across the ten folds, matching
\\subsecref{stats-plan}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STRAT = ROOT / "results" / "foveation_mit1003_initial" / "stratified"
# One bin per fixation out to the eighth, then a tail. foveation_stratified
# pooled 4-5 and 6-8 to keep its per-fold counts up, which is not needed here.
# The tail starts at 9 because that is where it stops thinning out the figure:
# 9+ holds 7,126 fixations against the eighth bin's 6,566, so every bin is
# comparable. Splitting further would put 825 in the eleventh and 111 across
# the last three, since scanpaths reach index 15.
FIX_INDEX_BINS = [(i, i) for i in range(1, 9)] + [(9, 100)]
N_FOLDS = 10


def bin_label(lo: int, hi: int) -> str:
    return f"{lo}" if lo == hi else (f"{lo}+" if hi >= 100 else f"{lo}-{hi}")


def profile_one(cpd: int, folds: int = N_FOLDS) -> dict:
    """Per-index-bin amplitude, control level and paired difference for one cutoff."""
    d = STRAT / f"cpd{cpd}_val" / "per_fixation"
    rows = []
    for lo, hi in FIX_INDEX_BINS:
        per_fold: list[dict] = []
        for k in range(folds):
            a = np.load(d / f"normal_fold{k}.npz")
            b = np.load(d / f"fov_cpd{cpd}_fold{k}.npz")
            if not np.array_equal(a["fix_index"], b["fix_index"]):
                raise SystemExit(f"cpd{cpd} fold{k}: arms are not on the same fixations")
            m = (a["fix_index"] >= lo) & (a["fix_index"] <= hi)
            if not m.any():
                continue
            per_fold.append({
                "n": int(m.sum()),
                "d_ll": float((b["ll_bits"][m] - a["ll_bits"][m]).mean()),
                "ig_normal": float((a["ll_bits"][m] - a["cb_bits"][m]).mean()),
                "amp_px": float(a["sacc_px"][m].mean()),
            })
        v = np.array([f["d_ll"] for f in per_fold])
        se = float(v.std(ddof=1) / np.sqrt(len(v)))
        lvl = float(np.mean([f["ig_normal"] for f in per_fold]))
        rows.append({
            "fix_index_lo": lo, "fix_index_hi": hi, "label": bin_label(lo, hi),
            "n_fixations": int(sum(f["n"] for f in per_fold)),
            "mean_amplitude_px": float(np.mean([f["amp_px"] for f in per_fold])),
            "ig_normal_bits": lvl,
            "d_ll_bits": {"mean": float(v.mean()), "se": se,
                          "t": float(v.mean() / se), "n_folds": len(v)},
            # The difference as a share of what the control knew there. If the
            # index trend were only the level falling, this column would be flat.
            "d_ll_as_share_of_level": float(v.mean() / lvl),
        })
    return {"cpd": cpd, "n_folds": folds, "bins": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cpds", type=int, nargs="+", default=[40, 20, 10])
    ap.add_argument("--out", type=Path, default=STRAT / "index_profile.json")
    args = ap.parse_args()

    out = {"source": "results/foveation_mit1003_initial/stratified/cpd*_val/per_fixation/",
           "split": "val", "bins": [list(b) for b in FIX_INDEX_BINS], "arms": {}}
    for cpd in args.cpds:
        prof = profile_one(cpd)
        out["arms"][f"foveated@{cpd}"] = prof
        print(f"=== cutoff {cpd}")
        print(f"{'index':>6} {'n':>8} {'mean amp':>9} {'IG level':>9} "
              f"{'d LL':>9} {'SE':>8} {'share':>8}")
        for r in prof["bins"]:
            print(f"{r['label']:>6} {r['n_fixations']:8d} {r['mean_amplitude_px']:9.1f} "
                  f"{r['ig_normal_bits']:9.3f} {r['d_ll_bits']['mean']:+9.4f} "
                  f"{r['d_ll_bits']['se']:8.4f} {r['d_ll_as_share_of_level']*100:7.2f}%")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
