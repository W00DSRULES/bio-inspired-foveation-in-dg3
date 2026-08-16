"""Single-panel human-scanpath teaser figure for the thesis introduction.

One human fixation sequence over one image, no model output — the point being
made in ch01 is about eye movements, not about DG3 yet.

    python scripts/demo_intro_scanpath.py --stim-idx 91

Outputs land in results/foveation_mit1003/ (same stimulus as the later
gaze-contingent demo figure, so the thesis reuses one running example).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pysaliency

from tez_deepgaze.figstyle import canvas
from tez_deepgaze.human_scanpaths import pick_human_scanpath
from tez_deepgaze.paths import DATA_ROOT
from tez_deepgaze.script_utils import draw_scanpath, image_rgb, stim_label

OUT = Path(__file__).resolve().parents[1] / "results" / "foveation_mit1003"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stim-idx", type=int, default=91)
    ap.add_argument("--subject-idx", type=int, default=None,
                    help="pick a specific human subject; default: first available")
    args = ap.parse_args()

    stimuli, fixations = pysaliency.get_mit1003(location=str(DATA_ROOT))
    image = image_rgb(stimuli, args.stim_idx)
    H, W = image.shape[:2]
    xs, ys, subj = pick_human_scanpath(fixations, args.stim_idx, args.subject_idx)

    fig, ax = plt.subplots(**canvas((W, H), page_frac=0.72))
    ax.imshow(image)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    # Printed at 0.72 of the text block, where the default markers put the digits
    # at about 5 pt.
    draw_scanpath(ax, xs, ys, color="#1d6fb8", marker_scale=1.5)
    fig.tight_layout(pad=0)

    OUT.mkdir(parents=True, exist_ok=True)
    out_png = OUT / f"intro_scanpath_stim{args.stim_idx:04d}.png"
    fig.savefig(out_png, dpi="figure", bbox_inches="tight")
    plt.close(fig)

    config = {
        "stim_idx": args.stim_idx,
        "stim_label": stim_label(stimuli, args.stim_idx),
        "human_subject_idx": subj,
        "n_fixations": int(len(xs)),
        "image_shape_hw": [H, W],
    }
    (OUT / f"intro_scanpath_stim{args.stim_idx:04d}.json").write_text(
        json.dumps(config, indent=2)
    )
    print(f"saved figure: {out_png}")
    print(f"human subject: {subj}, {len(xs)} fixations")


if __name__ == "__main__":
    main()
