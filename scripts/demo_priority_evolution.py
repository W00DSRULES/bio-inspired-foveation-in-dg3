"""Show how DG3's probability map changes as the scanpath history grows.

In DG3 the "saliency map" and "priority map" are not separate outputs — the
probability map *is* the saliency network's contribution combined with the
scanpath-history embedding and the centerbias, all run through the fixation
selection head. With one fixation in history, the map is close to a classical
saliency map; after several fixations it shifts toward unvisited regions. This
script renders that progression on a single image, feeding the model one
subject's own recorded scanpath (the same subject Figures 1.1 and 2.3 draw),
so each panel is where the model expected that subject to go next. Nothing is
sampled from the model.

Usage:
    python scripts/demo_priority_evolution.py --stim-idx 91

Outputs land in results/demos/priority_evolution/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import deepgaze_pytorch
import pysaliency

from tez_deepgaze.centerbias import load_centerbias_for_image
from tez_deepgaze.device import pick_device, to_device
from tez_deepgaze.human_scanpaths import pick_human_scanpath
from tez_deepgaze.figstyle import canvas
from tez_deepgaze.instrument import compute_log_density
from tez_deepgaze.paths import DATA_ROOT
from tez_deepgaze.script_utils import draw_scanpath, image_rgb, stim_label

OUT = Path(__file__).resolve().parents[1] / "results" / "demos" / "priority_evolution"


def evolution(stim_idx: int, subject: int, ranks: list[int]) -> None:
    device = pick_device()
    print(f"device: {device}")

    stimuli, fixations = pysaliency.get_mit1003(location=str(DATA_ROOT))
    image = image_rgb(stimuli, stim_idx)
    H, W = image.shape[:2]
    cb = load_centerbias_for_image(H, W)

    print("loading pretrained DeepGaze III...")
    model = to_device(deepgaze_pytorch.DeepGazeIII(pretrained=True), device).eval()

    # The subject's recorded scanpath is the history; the model is never
    # asked to produce a fixation.
    sx, sy, subj = pick_human_scanpath(fixations, stim_idx, subject_idx=subject)
    xs = [float(v) for v in sx]
    ys = [float(v) for v in sy]
    n_needed = max(ranks)
    if len(xs) < n_needed:
        raise SystemExit(f"subject {subj} made {len(xs)} fixations, fewer than {n_needed}")

    priority_at_rank: dict[int, np.ndarray] = {}
    # rank == number of the subject's fixations in the history when the map is read
    for rank in ranks:
        priority_at_rank[rank] = compute_log_density(model, image, cb,
                                                     xs[:rank], ys[:rank], device)

    n_panels = len(ranks)
    fig, axes = plt.subplots(1, n_panels, **canvas((4.5 * n_panels, 6.0)))
    for ax in axes:
        ax.imshow(image)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.axis("off")

    for ax, rank in zip(axes, ranks):
        log_d = priority_at_rank[rank]
        prio = np.exp(log_d)
        prio_disp = prio / (prio.max() + 1e-12)
        ax.imshow(prio_disp, cmap="inferno", alpha=0.55)
        # draw_scanpath prints the fixation numbers in white on the marker face, so a
        # white colour here would hide them; blue matches the scanpath overlays elsewhere.
        draw_scanpath(ax, xs[:rank], ys[:rank], color="#1d6fb8", marker_scale=3.0)
        ax.set_title(f"after {rank} fixation{'s' if rank > 1 else ''}", fontsize=10)

    label = stim_label(stimuli, stim_idx)
    # No suptitle: the caption names the figure.
    fig.tight_layout()

    OUT.mkdir(parents=True, exist_ok=True)
    out_png = OUT / f"priority_evolution_stim{stim_idx:04d}.png"
    fig.savefig(out_png, dpi="figure", bbox_inches="tight")
    plt.close(fig)

    config = {
        "stim_idx": stim_idx,
        "stim_label": label,
        "ranks": ranks,
        "subject": int(subj),
        "scanpath_xs": xs,
        "scanpath_ys": ys,
        "image_shape_hw": [H, W],
        "device": str(device),
        "model": "DeepGazeIII(pretrained=True)",
        "note": (
            "rank = number of the subject's own fixations in the history when "
            "the probability map is read; the path is recorded, not sampled"
        ),
    }
    (OUT / f"priority_evolution_stim{stim_idx:04d}.json").write_text(
        json.dumps(config, indent=2)
    )

    print(f"saved figure: {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stim-idx", type=int, required=True,
                    help="MIT1003 stimulus index (0..1002)")
    ap.add_argument("--subject", type=int, default=0,
                    help="which subject's scanpath is the history (default 0, "
                         "as in the intro and background scanpath figures)")
    ap.add_argument("--ranks", type=int, nargs="+",
                    default=[1, 3, 4],
                    help="fixation counts at which to render the probability map")
    args = ap.parse_args()
    evolution(args.stim_idx, args.subject, sorted(args.ranks))


if __name__ == "__main__":
    main()
