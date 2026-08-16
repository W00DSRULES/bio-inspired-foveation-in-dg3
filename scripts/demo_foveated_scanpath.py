"""Demo: gaze-contingent foveated scanpath on one MIT1003 stimulus.

Samples two scanpaths from the pretrained DeepGaze III for the same seed and
start point:

  * normal input  — the sharp image, as the baseline sampler sees it;
  * gaze-contingent foveated input — at every step the image is re-foveated
    around the current fixation before the forward pass, so the model picks
    its next saccade from a sharp-at-gaze, low-resolution-but-present
    peripheral view.

The figure is one row per step and two columns, sharp on the left and
gaze-contingent on the right. Each cell carries the input that arm's backbone
receives at that step, the probability map it predicts from it, the path so
far, and the fixation it goes to next, so the divergence can be read as a
difference between the two maps rather than only as a difference between two
finished paths. ``--steps`` selects which fixations become rows.

A JSON sidecar records everything needed to reproduce the figure.

Usage:
    .venv/bin/python scripts/demo_foveated_scanpath.py --stim-idx 91
    .venv/bin/python scripts/demo_foveated_scanpath.py --stim-idx 91 --foveal-cpd 12
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.patheffects as pe
from matplotlib.patches import Circle

import deepgaze_pytorch

from tez_deepgaze.centerbias import load_centerbias_for_image
from tez_deepgaze.device import pick_device, to_device
from tez_deepgaze.figstyle import canvas
from tez_deepgaze.foveate_input import FOVEAL_CPD, MIT1003_PPD, Foveation
from tez_deepgaze.instrument import compute_log_density, sample_scanpath
from tez_deepgaze.paths import load_mit1003
from tez_deepgaze.script_utils import (
    image_rgb,
    saccade_amplitudes,
    stim_label,
)

OUT = Path(__file__).resolve().parents[1] / "results" / "foveation_mit1003"

# Okabe-Ito: lightness-separated, colourblind-safe (no red-green pairing).
NORMAL_COLOR = "#E69F00"    # orange
FOVEATED_COLOR = "#0072B2"  # blue


def _foveate_display(foveation: Foveation, img_t: torch.Tensor, stack: torch.Tensor,
                     fx: float, fy: float) -> np.ndarray:
    """Foveate ``img_t`` (1, 3, H, W) around (fx, fy); return uint8 for display.

    Same operation the sampler feeds the model (linear blend on [0,255]), only
    cast back to uint8 for imshow.

    ``img_t`` and ``stack`` are prepared once by the caller and reused for every
    filmstrip frame. That is not a micro-optimisation: the deepest pyramid level
    is a 145x145 kernel over a full-resolution frame, so rebuilding the pyramid
    per frame on CPU costs minutes per frame.
    """
    out = foveation.foveate_shared_image(
        img_t, torch.tensor([fx]), torch.tensor([fy]), stack=stack)
    return out[0].permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()


def _first_divergence(a: list[float], b: list[float], tol: float = 1e-6) -> int | None:
    """Index of the first fixation where the two paths differ (None if identical)."""
    for i, (u, v) in enumerate(zip(a, b)):
        if abs(u - v) > tol:
            return i
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stim-idx", type=int, default=91,
                    help="MIT1003 stimulus index")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-fix", type=int, default=10,
                    help="number of fixations to sample")
    ap.add_argument("--steps", type=int, nargs="*", default=None,
                    help="0-based fixation indices to draw as rows; default all")
    ap.add_argument("--contact-sheet", action="store_true",
                    help="also write a thumbnail sheet of every step, to pick --steps from")
    ap.add_argument("--ppd", type=float, default=MIT1003_PPD,
                    help="pixels per degree (MIT1003 ~= 35)")
    ap.add_argument("--foveal-cpd", type=float, default=20.0,
                    help=f"foveal cutoff cyc/deg; human acuity is ~{FOVEAL_CPD:.0f} "
                         "(mild on screen-sized images), lower = stronger")
    args = ap.parse_args()

    device = pick_device()
    print(f"device: {device}")

    stimuli, _ = load_mit1003()
    image = image_rgb(stimuli, args.stim_idx)
    H, W = image.shape[:2]
    cb = load_centerbias_for_image(H, W)
    start_xy = (W / 2.0, H / 2.0)

    print("loading pretrained DeepGaze III...")
    model = to_device(deepgaze_pytorch.DeepGazeIII(pretrained=True), device).eval()
    foveation = Foveation(ppd=args.ppd, foveal_cpd=args.foveal_cpd)

    print(f"sampling normal-input path (seed={args.seed})...")
    normal = sample_scanpath(model, image, cb, start_xy, args.n_fix, device,
                             seed=args.seed)
    print(f"sampling gaze-contingent foveated path (seed={args.seed})...")
    fov = sample_scanpath(model, image, cb, start_xy, args.n_fix, device,
                          seed=args.seed, foveation=foveation)

    div_x = _first_divergence(normal.x, fov.x)
    div_y = _first_divergence(normal.y, fov.y)
    first_div = min([d for d in (div_x, div_y) if d is not None], default=None)
    per_fix_dist = np.hypot(np.array(fov.x) - np.array(normal.x),
                            np.array(fov.y) - np.array(normal.y))
    paths_differ = bool(first_div is not None)
    print(f"paths differ: {paths_differ} "
          f"(first divergence at fixation {first_div}, "
          f"mean per-fixation distance {per_fix_dist.mean():.1f} px)")

    # ----- figure -----
    # One row per step, two columns: the sharp control on the left and the
    # gaze-contingent arm on the right. Each cell carries the input that arm's
    # backbone receives at that step, the probability map it produces from it,
    # and the path so far — so the reader can see the two maps agree, then
    # separate once the fovea has moved off the first target, then send the
    # paths to different places.
    steps = ([s for s in args.steps if 0 <= s < len(fov.x)] if args.steps
             else list(range(len(fov.x))))
    disp_t = torch.from_numpy(image.transpose(2, 0, 1)[None]).float().to(device)
    disp_stack = foveation.blur_stack(disp_t)

    print(f"computing probability maps for {len(steps)} steps x 2 arms...")
    panels = []
    for s in steps:
        sharp_ld = compute_log_density(
            model, image, cb, normal.x[: s + 1], normal.y[: s + 1], device)
        fov_ld = compute_log_density(
            model, image, cb, fov.x[: s + 1], fov.y[: s + 1], device,
            foveation=foveation, foveation_stack=disp_stack)
        fov_frame = _foveate_display(foveation, disp_t, disp_stack,
                                     fov.x[s], fov.y[s])
        panels.append((s, sharp_ld, fov_ld, fov_frame))

    # One colour scale across every panel, so a difference between two cells is
    # a difference in the prediction and not in the normalisation.
    allv = np.concatenate([p[1].ravel() for p in panels]
                          + [p[2].ravel() for p in panels])
    vmin, vmax = np.percentile(allv, [55.0, 99.9])

    def cell(ax, bg, ld, sp, s, color, scale: float = 1.0) -> None:
        """One panel: the arm's input, its probability map, and the path so far."""
        ax.imshow(bg)
        ax.imshow(np.clip(ld, vmin, vmax), cmap="magma", alpha=0.55,
                  vmin=vmin, vmax=vmax)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.plot(sp.x[: s + 1], sp.y[: s + 1], "-", color=color,
                linewidth=2.0 * scale, zorder=3,
                path_effects=[pe.withStroke(linewidth=3.6 * scale,
                                            foreground="white")])
        ax.add_patch(Circle((sp.x[s], sp.y[s]),
                            radius=max(9, int(0.014 * H)) * (1.0 if scale > 0.7 else 1.6),
                            facecolor="#FFFFFF", edgecolor="#000000",
                            linewidth=1.6 * scale, zorder=5))
        if s + 1 < len(sp.x):
            ax.plot(sp.x[s + 1], sp.y[s + 1], "*", color=color,
                    markersize=17 * scale, markeredgecolor="white",
                    markeredgewidth=1.3 * scale, zorder=6)

    def mark_divergence(ax, s) -> None:
        if first_div is not None and s + 1 == first_div:
            for spine in ax.spines.values():
                spine.set_edgecolor("#C81E1E")
                spine.set_linewidth(3.0)

    # Two step-pairs per row rather than one. With one pair per row the figure
    # is about 0.375 as tall as it is wide per step, so past four steps it runs
    # off a printed page at \textwidth; at two pairs per row that halves and the
    # whole sampled scanpath fits on one page.
    PAIRS_PER_ROW = 2
    nrow = math.ceil(len(panels) / PAIRS_PER_ROW)
    row_h = (11.0 / (2 * PAIRS_PER_ROW)) * H / W
    # ch03 gives this one a height as well as a width, so the canvas is
    # capped the same way and the panels print at the size they are drawn for.
    fig, axes = plt.subplots(nrow, 2 * PAIRS_PER_ROW, squeeze=False,
                             **canvas((11.0, row_h * nrow + 0.5),
                                      page_height_frac=0.78))
    for k, (s, sharp_ld, fov_ld, fov_frame) in enumerate(panels):
        row, base = divmod(k, PAIRS_PER_ROW)
        for off, (bg, ld, sp, color, name) in enumerate((
            (image, sharp_ld, normal, NORMAL_COLOR, "sharp"),
            (fov_frame, fov_ld, fov, FOVEATED_COLOR, "gaze-contingent foveated"),
        )):
            ax = axes[row][2 * base + off]
            cell(ax, bg, ld, sp, s, color, scale=0.6)
            ax.set_title(f"after fixation {s + 1} · {name}" if off == 0
                         else name, fontsize=9.5)
            mark_divergence(ax, s)
    for k in range(len(panels), nrow * PAIRS_PER_ROW):   # blank any unused pair
        row, base = divmod(k, PAIRS_PER_ROW)
        for off in (0, 1):
            axes[row][2 * base + off].axis("off")

    # The suptitle sits in a fixed band at the top, so it does not collide with
    # the first row's headings however many rows the figure ends up with. The
    # band is measured against the canvas the figure is drawn on, which is set
    # by the page rather than by the row count.
    band = 0.75 / fig.get_figheight()
    fig.suptitle(
        f"Same seed, same start: where the two arms separate   "
        f"(ppd {args.ppd:.0f}, $f_{{c0}}$ = {args.foveal_cpd:.0f} cyc/deg)",
        fontsize=11, y=1.0 - 0.22 * band,
    )
    # hspace has to clear a row heading: the rows are short on a canvas sized
    # for the page, so a gap in fractions of a row is a small gap in points.
    fig.subplots_adjust(top=1.0 - band, bottom=0.005, left=0.008, right=0.992,
                        hspace=0.30, wspace=0.03)
    n_frames = len(panels)

    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"gaze_contingent_demo_stim{args.stim_idx:04d}"
    out_png = OUT / f"{stem}.png"
    fig.savefig(out_png, dpi="figure", bbox_inches="tight")
    plt.close(fig)

    # Contact sheet: every step at thumbnail size, pairs laid out across the
    # page instead of down it. The tall one-row-per-step figure runs off a
    # printed page past about four steps, so this is the picker for --steps.
    if args.contact_sheet:
        per_row = 3                                   # step pairs per row
        nrow = math.ceil(len(panels) / per_row)
        cs, cax = plt.subplots(nrow, 2 * per_row,
                               figsize=(1.85 * 2 * per_row,
                                        1.85 * nrow * H / W + 0.7),
                               squeeze=False)
        for k, (s, sharp_ld, fov_ld, fov_frame) in enumerate(panels):
            r, base = divmod(k, per_row)
            for off, (bg, ld, sp, color, tag) in enumerate((
                (image, sharp_ld, normal, NORMAL_COLOR, "sharp"),
                (fov_frame, fov_ld, fov, FOVEATED_COLOR, "fov"),
            )):
                ax = cax[r][2 * base + off]
                cell(ax, bg, ld, sp, s, color, scale=0.45)
                ax.set_title(f"{s}: {tag}" if off == 0 else tag, fontsize=8)
                mark_divergence(ax, s)
        for k in range(len(panels), nrow * per_row):   # blank the unused pairs
            r, base = divmod(k, per_row)
            for off in (0, 1):
                cax[r][2 * base + off].axis("off")
        cs.suptitle(f"stim {args.stim_idx}, $f_{{c0}}$ = {args.foveal_cpd:.0f} "
                    f"cyc/deg — pass the step numbers you want to --steps",
                    fontsize=11)
        cs.subplots_adjust(top=0.90, bottom=0.01, left=0.01, right=0.99,
                           hspace=0.22, wspace=0.04)
        sheet_png = OUT / f"{stem}_contact_sheet.png"
        cs.savefig(sheet_png, dpi=110, bbox_inches="tight")
        plt.close(cs)
        print(f"saved contact sheet: {sheet_png}")

    sidecar = {
        "stim_idx": args.stim_idx,
        "stim_label": stim_label(stimuli, args.stim_idx),
        "image_shape_hw": [H, W],
        "start_xy": [float(start_xy[0]), float(start_xy[1])],
        "seed": args.seed,
        "n_fixations_sampled": args.n_fix,
        "n_frames_shown": n_frames,
        "foveation": {"ppd": args.ppd, "foveal_cpd": args.foveal_cpd,
                      "e2_deg": foveation.e2_deg, "n_levels": foveation.n_levels},
        "device": str(device),
        "model": "DeepGazeIII(pretrained=True)",
        "normal_path": {"x": normal.x, "y": normal.y},
        "foveated_path": {"x": fov.x, "y": fov.y},
        "paths_differ": paths_differ,
        "first_divergence_fixation": first_div,
        "mean_per_fixation_distance_px": float(per_fix_dist.mean()),
        "max_per_fixation_distance_px": float(per_fix_dist.max()),
        "normal_mean_saccade_px": float(saccade_amplitudes(normal.x, normal.y).mean()),
        "foveated_mean_saccade_px": float(saccade_amplitudes(fov.x, fov.y).mean()),
    }
    (OUT / f"{stem}.json").write_text(json.dumps(sidecar, indent=2))

    print(f"saved figure:  {out_png}")
    print(f"saved sidecar: {OUT / f'{stem}.json'}")


if __name__ == "__main__":
    main()
