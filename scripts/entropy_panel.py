"""What fixation entropy measures, on the dataset's least and most dispersed stimulus.

The consensus construction has a figure showing how it is built
(``demo_consensus_panel.py``); the entropy of \\eqnref{fix-entropy} has none, only
the scatter plots, which show its correlation rather than the quantity itself.
This draws the quantity: every subject's fixations, the 16x16 grid the
entropy is actually computed over, and where the resulting value sits in the
dataset.

The two stimuli default to the dataset minimum and maximum of
``fixation_entropy_norm`` as recorded by ``per_image_diagnostic.py``, so the
contrast is the widest the dataset offers rather than a chosen pair. The
consensus composites remain the thesis's visual language for agreement; this
figure only defines the scalar the numbers use.

Reads the committed diagnostic JSON and the fixation data only -- no model, no
checkpoints, no GPU:

    python scripts/entropy_panel.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tez_deepgaze.figstyle import canvas  # noqa: E402
from tez_deepgaze.human_scanpaths import (  # noqa: E402
    ENTROPY_BINS,
    all_human_scanpaths,
    fixation_entropy,
)
from tez_deepgaze.paths import load_mit1003_variant  # noqa: E402
from tez_deepgaze.script_utils import image_rgb  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "results" / "per_image_diagnostic_initial_1003" / "diagnostic.json"
OUT = ROOT / "results" / "foveation_mit1003_initial" / "figs"
# Same grid as per_image_diagnostic.difficulty_metrics; log2(16*16) = 8 puts the
# normalised entropy in [0, 1].
def entropy_grid(paths, H: int, W: int, strip: int = 1):
    """The population histogram and its normalised entropy, from the one home.

    ``strip`` drops the leading forced central fixation, as
    ``per_image_diagnostic`` does under the ``initial`` variant: it sits at the
    exact centre of every image and would add the same spurious cluster to all
    1003 of them.
    """
    hist, _, norm = fixation_entropy(
        [(xs[strip:], ys[strip:], s) for xs, ys, s in paths], H, W)
    return hist, norm


def pick_stimuli(rows) -> list[tuple[int, str]]:
    """The dataset's lowest and highest entropy stimulus.

    Two rather than a spread: the figure exists to show what the scalar
    separates, and the extremes do that with the least page.
    """
    h = np.array([r["difficulty"]["fixation_entropy_norm"] for r in rows])
    order = np.argsort(h)
    return [(rows[order[0]]["stim_idx"], "lowest in the dataset"),
            (rows[order[-1]]["stim_idx"], "highest in the dataset")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stim-indices", type=int, nargs=2, default=None,
                    help="override the two stimuli (low entropy, high entropy)")
    ap.add_argument("--dataset-variant", default="initial", choices=["plain", "initial"])
    ap.add_argument("--out", type=Path, default=OUT / "entropy_panel.png")
    args = ap.parse_args()

    rows = json.loads(DIAG.read_text())["rows"]
    all_h = np.array([r["difficulty"]["fixation_entropy_norm"] for r in rows])
    if args.stim_indices:
        picks = [(s, "") for s in args.stim_indices]
    else:
        picks = pick_stimuli(rows)

    stimuli, fixations = load_mit1003_variant(args.dataset_variant)

    fig = plt.figure(**canvas((9.6, 7.4)))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.0, 1.0, 0.40],
                  hspace=0.24, wspace=0.05)
    cols = []

    for c, (stim, label) in enumerate(picks):
        image = image_rgb(stimuli, stim)
        H, W = image.shape[:2]
        paths = list(all_human_scanpaths(fixations, stim))
        strip = 1 if args.dataset_variant == "initial" else 0
        hist, h_norm = entropy_grid(paths, H, W, strip=strip)
        pct = float((all_h < h_norm).mean() * 100)
        cols.append({"stim": stim, "h": h_norm, "pct": pct, "label": label})

        ax = fig.add_subplot(gs[0, c])
        ax.imshow(image)
        cmap = plt.get_cmap("turbo")
        for i, (xs, ys, _subj) in enumerate(paths):
            xs, ys = xs[strip:], ys[strip:]
            if len(xs) == 0:
                continue
            ax.plot(xs, ys, "o", ms=4.2, mfc=cmap(i / max(len(paths) - 1, 1)),
                    mec="white", mew=0.7, alpha=0.95)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"stimulus {stim}", fontsize=11, pad=6)
        if c == 0:
            ax.set_ylabel("all subjects'\nfixations", fontsize=10)

        # The cells go back on the photograph rather than onto a bare grid: the
        # point is which parts of *this scene* the looking landed in, and an
        # abstract heat grid reads as a game board instead.
        ax = fig.add_subplot(gs[1, c])
        ax.imshow(image)
        ax.add_patch(plt.Rectangle((0, 0), W, H, color="white", alpha=0.32))
        # histogram2d indexes [x, y]; transpose to draw it the way up the image is.
        share = hist.T / hist.max()
        # A colourmap whose low end is pale rather than black, so a lightly
        # occupied cell still reads over the photograph, and alpha exactly zero
        # where nobody looked so the scene shows through untouched.
        rgba = plt.get_cmap("YlOrRd")(0.15 + 0.85 * share)
        rgba[..., 3] = np.where(share > 0, 0.40 + 0.55 * share ** 0.6, 0.0)
        ax.imshow(rgba, extent=[0, W, H, 0], interpolation="nearest")
        for k in range(1, ENTROPY_BINS):
            ax.axvline(k * W / ENTROPY_BINS, color="#333333", lw=0.35, alpha=0.28)
            ax.axhline(k * H / ENTROPY_BINS, color="#333333", lw=0.35, alpha=0.28)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        # No value or rank here: the strip below places both stimuli on the
        # distribution with their values, and the caption says which is which.
        if c == 0:
            ax.set_ylabel("the same fixations counted\nin a $16 \\times 16$ grid", fontsize=10)

    # Where the two sit in the dataset, so the numbers have a scale.
    ax = fig.add_subplot(gs[2, :])
    ax.hist(all_h, bins=60, color="#B8C2CC", edgecolor="none")
    for col, colour in zip(cols, ("#009E73", "#E69F00")):
        ax.axvline(col["h"], color=colour, lw=2.0)
        ax.annotate(f"stim {col['stim']}\n{col['h']:.3f}",
                    xy=(col["h"], ax.get_ylim()[1] * 0.98), ha="center", va="top",
                    fontsize=9, color=colour, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    ax.axvline(float(np.median(all_h)), color="#555555", lw=1.0, ls="--")
    ax.set_xlabel("normalised fixation entropy $H_{\\mathrm{fix}}$ over the 1003 stimuli "
                  "(dashed line: the median)", fontsize=10)
    ax.set_yticks([])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)

    # No suptitle: the caption already says which two stimuli these are and why.
    fig.subplots_adjust(left=0.075, right=0.985, top=0.965, bottom=0.085)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)

    args.out.with_suffix(".json").write_text(json.dumps({
        "source": str(DIAG.relative_to(ROOT)),
        "dataset_variant": args.dataset_variant,
        "entropy_bins": ENTROPY_BINS,
        "dataset_min": float(all_h.min()),
        "dataset_median": float(np.median(all_h)),
        "dataset_max": float(all_h.max()),
        "stimuli": cols,
    }, indent=2) + "\n")
    print(f"wrote {args.out}")
    for col in cols:
        print(f"  stim {col['stim']:4d}  H={col['h']:.3f}  percentile {col['pct']:.1f}")


if __name__ == "__main__":
    main()
