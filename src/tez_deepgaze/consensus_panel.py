"""Three-panel consensus-and-comparison figure for one MIT1003 stimulus.

Per stimulus, renders:

    A. All human scanpaths (one colour per subject).
    B. Consensus regions: pixels within ``radius`` px of a fixation, scored
       across subjects (light = >=75%, dark = >=90% with a contour).
    C. DG3's probability map conditioned on one synthetic fixation at the image
       centre. The recorded central start is dropped by the loader, so this is
       a constructed starting point, not a fixation read from the data.

No panel shows a scanpath sampled from DG3: nothing in the thesis is measured
in the sampling mode (Methods, distribution-level metrics), so such a panel
would show a mode no number comes from.

This is the single source of truth for the figure: both
``scripts/demo_consensus_panel.py`` (CLI, batch) and
``notebooks/02_consensus_panel.ipynb`` (interactive) call
:func:`build_consensus_panel`, so they render identically to the figures used
in the thesis.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from .centerbias import load_centerbias_for_image
from .figstyle import canvas
from .human_scanpaths import (
    CONSENSUS_RADIUS_PX,
    all_human_scanpaths,
    consensus_count,
)
from .instrument import compute_log_density
from .paths import REPO_ROOT
from .script_utils import image_rgb, stim_label

DEFAULT_OUT = REPO_ROOT / "results" / "consensus_panels"


def _draw_all_subjects(ax, paths, H, dot_radius_frac: float = 0.011) -> None:
    cmap = plt.get_cmap("tab20")
    n = len(paths)
    dot_r = max(4, int(dot_radius_frac * H))
    for i, (xs, ys, _) in enumerate(paths):
        color = cmap(i / max(n - 1, 1))
        if len(xs) >= 2:
            ax.plot(xs, ys, "-", color=color, linewidth=0.8, alpha=0.5, zorder=2)
        for x, y in zip(xs, ys):
            ax.add_patch(Circle((x, y), radius=dot_r,
                                facecolor=color, edgecolor="white",
                                linewidth=0.6, alpha=0.9, zorder=3))


def build_consensus_panel(
    stim_idx: int,
    *,
    stimuli,
    fixations,
    model,
    device,
    radius: int = CONSENSUS_RADIUS_PX,
    out_dir: Path | str | None = None,
    title_suffix: str = "",
    panel_titles: bool = True,
    page_frac: float = 0.89,
):
    """Build the 3-panel consensus figure for one stimulus.

    Returns ``(fig, summary)``. The figure is left open so an interactive
    caller (a notebook) can display it; a batch caller should ``plt.close(fig)``
    after saving. When ``out_dir`` is given, writes
    ``consensus_stimXXXX.png`` + a sidecar ``.json`` there.

    ``title_suffix`` is appended to the "Stimulus N" title (ch04 puts the
    stimulus's per-image metrics there); ``panel_titles=False`` drops the
    A/B/C panel headings, for the rows below the first when several of these
    composites stack in one figure; ``page_frac`` is the fraction of the text
    width the composite is printed at, so its text prints at the same size
    whether it stands alone (0.89) or is one of six stacked rows (0.5).
    """
    image = image_rgb(stimuli, stim_idx)
    H, W = image.shape[:2]
    cb = load_centerbias_for_image(H, W)

    paths = all_human_scanpaths(fixations, stim_idx)
    # A subject with no recorded fixation draws nothing and joins no consensus,
    # but would still raise the thresholds below, which are fractions of the
    # subject count. Dropped here so the panel and the per-image diagnostic
    # threshold the same population.
    paths = [p for p in paths if len(p[0]) > 0]
    n_subj = len(paths)

    start_xy = (W / 2.0, H / 2.0)
    log_d = compute_log_density(model, image, cb, [start_xy[0]], [start_xy[1]], device)
    prio = np.exp(log_d)
    prio_disp = prio / (prio.max() + 1e-12)

    count = consensus_count(paths, H, W, radius)
    thr75 = int(np.ceil(0.75 * n_subj))
    thr90 = int(np.ceil(0.90 * n_subj))
    mask75 = count >= thr75
    mask90 = count >= thr90
    pct75 = 100.0 * mask75.sum() / (H * W)
    pct90 = 100.0 * mask90.sum() / (H * W)

    # Three panels across ``page_frac`` of the text block, so the titles are set
    # for that width.
    fig, axes = plt.subplots(1, 3, **canvas((16.5, 5.2), page_frac=page_frac))
    for ax in axes:
        ax.imshow(image)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.axis("off")

    _draw_all_subjects(axes[0], paths, H)
    if panel_titles:
        axes[0].set_title("A. Human scanpaths", fontsize=9)

    # Colourblind-safe overlay: lightness ordering (light -> dark) is the primary
    # signal; the 90% peak also gets a white-haloed black contour so it reads on
    # any background regardless of hue.
    overlay = np.zeros((H, W, 4), dtype=float)
    overlay[mask75] = [0.55, 0.80, 1.00, 0.40]  # light sky-blue
    overlay[mask90] = [0.06, 0.16, 0.40, 0.70]  # dark navy
    axes[1].imshow(overlay)
    axes[1].contour(mask90.astype(float), levels=[0.5], colors="white",
                    linewidths=2.6, alpha=0.95)
    axes[1].contour(mask90.astype(float), levels=[0.5], colors="black",
                    linewidths=1.1, alpha=0.95)
    # One line each. Six of these composites stack in one figure of ch04, so a
    # second title line is six lines of page spent restating what the caption
    # says once — the radius and the subject count.
    if panel_titles:
        axes[1].set_title("B. Consensus regions", fontsize=9)

    axes[2].imshow(prio_disp, cmap="inferno", alpha=0.55)
    if panel_titles:
        axes[2].set_title("C. Probability map, first fixation", fontsize=9)

    # The stimulus number, not the MIT filename: the number is what the chapter
    # and every other figure refer to, and the filename identifies nothing a
    # reader of the thesis can look up.
    fig.suptitle(f"Stimulus {stim_idx}{title_suffix}", fontsize=11)
    fig.tight_layout()

    summary = {
        "stim_idx": stim_idx,
        "stim_label": stim_label(stimuli, stim_idx),
        "image_shape_hw": [H, W],
        "n_subjects": n_subj,
        "consensus_radius_px": radius,
        "thresholds": {"th75_subjects": thr75, "th90_subjects": thr90},
        "consensus_area_pct": {"th75": pct75, "th90": pct90},
        "map_start_xy": [float(start_xy[0]), float(start_xy[1])],
        "device": str(device),
        "model": "DeepGazeIII(pretrained=True)",
    }

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_png = out_dir / f"consensus_stim{stim_idx:04d}.png"
        fig.savefig(out_png, dpi="figure", bbox_inches="tight")
        summary["out_png"] = str(out_png.relative_to(REPO_ROOT))
        (out_dir / f"consensus_stim{stim_idx:04d}.json").write_text(
            json.dumps(summary, indent=2)
        )

    return fig, summary
