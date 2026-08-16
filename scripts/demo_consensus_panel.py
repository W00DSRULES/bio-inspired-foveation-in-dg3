"""3-panel consensus-and-comparison figure per MIT1003 stimulus (CLI).

Per stimulus, renders:

    A. All human scanpaths (one color per subject, small dots, thin lines —
       no numbered labels so the picture stays readable when 15 subjects
       overlap).
    B. Consensus regions: pixels within `--radius` px of a fixation, scored
       across subjects. Light = visited by >=75% of subjects, dark = >=90%.
    C. DG3's probability map conditioned on the start fixation (inferno overlay).

The figure itself is built by :func:`tez_deepgaze.consensus_panel.build_consensus_panel`,
shared with ``notebooks/02_consensus_panel.ipynb`` so both render identically.

Output: results/consensus_panels/consensus_stimXXXX.png and sidecar .json.

Usage:
    python scripts/demo_consensus_panel.py --stim-indices 71 77 91 455 522 876 884
    python scripts/demo_consensus_panel.py --stim-indices 91 --radius 60

The six composites ch04 stacks (the extremes of Figure "The extreme stimuli by
information gain") carry the stimulus's per-image IG, NSS and AUC in the title,
read from the per-image diagnostic, and only the first carries the A/B/C panel
headings:

    python scripts/demo_consensus_panel.py --stim-indices 128 704 528 750 639 489 \
        --metrics-from results/per_image_diagnostic_initial_1003/diagnostic.json \
        --titles-first-only
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt

import deepgaze_pytorch
import pysaliency

from tez_deepgaze.consensus_panel import DEFAULT_OUT, build_consensus_panel
from tez_deepgaze.device import pick_device, to_device
from tez_deepgaze.human_scanpaths import CONSENSUS_RADIUS_PX
from tez_deepgaze.paths import DATA_ROOT


def run(args) -> None:
    device = pick_device()
    print(f"device: {device}")
    print("loading MIT1003...")
    stimuli, fixations = pysaliency.get_mit1003(location=str(DATA_ROOT))
    print("loading pretrained DeepGaze III...")
    model = to_device(deepgaze_pytorch.DeepGazeIII(pretrained=True), device).eval()

    metrics = {}
    if args.metrics_from:
        rows = json.loads(args.metrics_from.read_text())["rows"]
        metrics = {r["stim_idx"]: r["dg3"] for r in rows}

    t0 = time.time()
    for k, sidx in enumerate(args.stim_indices):
        suffix = ""
        if sidx in metrics:
            m = metrics[sidx]
            suffix = (f"   ·   IG {m['ig_bits']:.2f} bits   ·   NSS {m['nss']:.1f}"
                      f"   ·   AUC {m['auc']:.3f}")
        fig, summary = build_consensus_panel(
            sidx, stimuli=stimuli, fixations=fixations, model=model, device=device,
            radius=args.radius, out_dir=DEFAULT_OUT, title_suffix=suffix,
            panel_titles=(k == 0) or not args.titles_first_only,
            page_frac=args.page_frac,
        )
        plt.close(fig)
        elapsed = time.time() - t0
        print(f"  [{k+1}/{len(args.stim_indices)}] stim {sidx} "
              f"→ {summary['out_png']}  ({elapsed:.1f}s elapsed)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stim-indices", type=int, nargs="+", required=True)
    ap.add_argument("--radius", type=int, default=CONSENSUS_RADIUS_PX,
                    help=f"consensus radius in pixels (default {CONSENSUS_RADIUS_PX}, "
                         "2° at MIT1003's original viewing geometry, the foveal radius)")
    ap.add_argument("--metrics-from", type=Path, default=None,
                    help="per-image diagnostic JSON; puts each stimulus's IG, NSS and "
                         "AUC in its title")
    ap.add_argument("--titles-first-only", action="store_true",
                    help="A/B/C panel headings on the first stimulus only")
    ap.add_argument("--page-frac", type=float, default=0.89,
                    help="fraction of the text width the composite prints at (the ch04 "
                         "stack of six uses 0.5)")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
