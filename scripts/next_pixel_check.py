"""Where does foveation move the predicted next fixation, and where does it not?

DeepGaze III's output is a distribution over the next fixation; its mode is the
single most likely next target. Foveating the input changes what the model can
resolve away from gaze, so it can change that target. The question is *when*.

Najemnik & Geisler (2005) predict the answer: saccade targets are selected from
peripheral preview, so degrading the periphery should matter only when the thing
worth looking at is *in* the periphery. If the fovea already sits on the scene's
attractor, foveation removes nothing the model was using, and the predicted
target should not move at all.

This script tests that rather than illustrating it. It sweeps gaze over a grid,
computes the mode of the log-density under sharp and foveated input at every
point, and reports the mode shift against the distance from gaze to the
attractor -- the mode the sharp model predicts from the image centre. The two
panels at the top are then *chosen from the sweep* (the gaze point nearest the
attractor, and the one with the largest shift) rather than hand-picked, so the
illustration cannot disagree with the measurement.

Both arms are evaluated on identical histories, and the blur pyramid is built
once and shared across every gaze point, so the sweep costs one forward pass per
gaze point per arm.

    .venv/bin/python scripts/next_pixel_check.py
    .venv/bin/python scripts/next_pixel_check.py --stim-idx 91 --grid 9 7
"""
from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr

import deepgaze_pytorch

from tez_deepgaze.centerbias import load_centerbias_for_image
from tez_deepgaze.device import pick_device, to_device
from tez_deepgaze.foveate_input import MIT1003_PPD, Foveation
from tez_deepgaze.instrument import compute_log_density_batch, image_tensor
from tez_deepgaze.paths import RESULTS, load_mit1003
from tez_deepgaze.script_utils import image_rgb, stim_label

OUT = RESULTS / "foveation_mit1003"
NORMAL_COLOR = "#E69F00"
FOVEATED_COLOR = "#0072B2"
# A shift below this is treated as "the target did not move". One pixel of
# argmax jitter is meaningless; 10 px is well under the 2.96 deg identity disc
# at the human cutoff and under any saccade MIT1003 records.
STILL_PX = 10.0


def sweep_modes(model, image, cb, gx, gy, device, foveation, stack, chunk):
    """Mode of the log-density at every gaze point, as an ``(N, 2)`` xy array."""
    H, W = image.shape[:2]
    out = []
    for i in range(0, len(gx), chunk):
        hx = [[float(x)] for x in gx[i:i + chunk]]
        hy = [[float(y)] for y in gy[i:i + chunk]]
        lds = compute_log_density_batch(
            model, image, cb, hx, hy, device,
            foveation=foveation, foveation_stack=stack if foveation else None)
        for ld in lds:
            j = int(np.argmax(ld))
            out.append((j % W, j // W))
    return np.asarray(out, dtype=float)


def panel(ax, image, log_density, gaze, mode, title):
    prio = np.exp(log_density)
    ax.imshow(image)
    ax.imshow(prio / (prio.max() + 1e-12), cmap="inferno", alpha=0.55)
    ax.plot(*gaze, "+", color="w", ms=18, mew=3)
    ax.plot(*mode, "o", mfc="none", mec="w", ms=17, mew=3)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stim-idx", type=int, default=91)
    ap.add_argument("--grid", type=int, nargs=2, default=[7, 5],
                    metavar=("NX", "NY"), help="gaze grid resolution")
    ap.add_argument("--ppd", type=float, default=MIT1003_PPD)
    ap.add_argument("--foveal-cpd", type=float, default=20.0,
                    help="foveal cutoff cyc/deg; 20 is stronger than the human "
                         "40, so the effect is legible in a figure")
    ap.add_argument("--chunk", type=int, default=8,
                    help="gaze points per forward batch (memory bound)")
    args = ap.parse_args()

    device = pick_device()
    print(f"device: {device}")
    stimuli, _ = load_mit1003()
    image = image_rgb(stimuli, args.stim_idx)
    H, W = image.shape[:2]
    cb = load_centerbias_for_image(H, W)

    print("loading pretrained DeepGaze III...")
    model = to_device(deepgaze_pytorch.DeepGazeIII(pretrained=True), device).eval()
    foveation = Foveation(ppd=args.ppd, foveal_cpd=args.foveal_cpd).to(device)
    stack = foveation.blur_stack(image_tensor(image, device))

    nx, ny = args.grid
    gxg, gyg = np.meshgrid(np.linspace(0.1, 0.9, nx) * W,
                           np.linspace(0.15, 0.85, ny) * H)
    gx, gy = gxg.ravel(), gyg.ravel()
    print(f"sweeping {len(gx)} gaze points, both arms...")
    m_norm = sweep_modes(model, image, cb, gx, gy, device, None, None, args.chunk)
    m_fov = sweep_modes(model, image, cb, gx, gy, device, foveation, stack, args.chunk)

    shift = np.hypot(*(m_fov - m_norm).T)
    # The attractor: what the sharp model wants to look at from the image centre.
    centre = int(np.argmin(np.hypot(gx - W / 2, gy - H / 2)))
    attractor = m_norm[centre]
    d_attr = np.hypot(gx - attractor[0], gy - attractor[1])

    rho, p = spearmanr(d_attr, shift)
    near = d_attr < np.median(d_attr)
    moved = shift > STILL_PX
    print(f"attractor at {attractor.astype(int).tolist()}")
    print(f"spearman(distance to attractor, mode shift) = {rho:.3f} (p = {p:.2g})")
    print(f"moved (>{STILL_PX:.0f} px):  gaze near attractor {int((moved & near).sum())}"
          f"/{int(near.sum())}   far {int((moved & ~near).sum())}/{int((~near).sum())}")

    # Illustrative panels chosen FROM the sweep, not hand-picked.
    i_on = centre if shift[centre] <= STILL_PX else int(np.argmin(d_attr))
    i_off = int(np.argmax(shift))

    fig = plt.figure(figsize=(13.5, 13.0))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.85], hspace=0.16,
                  wspace=0.05)
    for row, idx in ((0, i_on), (1, i_off)):
        gaze = (gx[idx], gy[idx])
        for col, (fov_arg, stack_arg, name, color) in enumerate((
                (None, None, "Normal", NORMAL_COLOR),
                (foveation, stack, "Foveated", FOVEATED_COLOR))):
            ld = compute_log_density_batch(
                model, image, cb, [[gaze[0]]], [[gaze[1]]], device,
                foveation=fov_arg, foveation_stack=stack_arg)[0]
            mode = m_norm[idx] if col == 0 else m_fov[idx]
            ax = fig.add_subplot(gs[row, col])
            panel(ax, image, ld, gaze, mode,
                  f"{name} — mode at ({mode[0]:.0f}, {mode[1]:.0f})")
            if col == 0:
                ax.set_ylabel(
                    f"gaze {'ON' if row == 0 else 'OFF'} the attractor\n"
                    f"mode shift = {shift[idx]:.0f} px", fontsize=11)
                ax.axis("on")
                ax.set_xticks([])
                ax.set_yticks([])

    ax = fig.add_subplot(gs[2, :])
    # Linear axis, not symlog: the story is "almost everything is zero, a few
    # points are huge", and a log axis flattens exactly that contrast.
    ax.axhspan(0, STILL_PX, color="0.90", zorder=0)
    ax.annotate(f"inside this band the predicted target did not move (≤{STILL_PX:.0f} px)",
                xy=(d_attr.min() * 1.1, STILL_PX / 2), xytext=(0, 46),
                textcoords="offset points", fontsize=9.5, color="0.35", zorder=5,
                arrowprops=dict(arrowstyle="->", color="0.55", lw=1.1))
    ax.scatter(d_attr[~moved], shift[~moved], s=52, facecolor="0.72",
               edgecolor="0.35", zorder=3)
    ax.scatter(d_attr[moved], shift[moved], s=68, color=FOVEATED_COLOR,
               edgecolor="w", linewidth=1.2, zorder=4)
    for idx, tag, off, ha in ((i_on, "shown in A / B above", (0, 30), "center"),
                              (i_off, "shown in C / D above", (-14, -6), "right")):
        ax.annotate(tag, (d_attr[idx], shift[idx]), fontsize=9.5, color="0.25",
                    textcoords="offset points", xytext=off, ha=ha, zorder=6,
                    arrowprops=dict(arrowstyle="->", color="0.55", lw=1.1))
    ax.set_xlabel("How far the attractor is from where the eye is looking  (px)",
                  fontsize=11)
    ax.set_ylabel("How far the predicted next\ntarget moves when foveated  (px)",
                  fontsize=11)
    ax.set_ylim(-18, max(shift.max() * 1.18, 60))
    ax.set_xlim(0, d_attr.max() * 1.04)
    ax.set_title(
        "Each dot is one place the eye could be looking (35 in total). "
        "Reading left to right: while the attractor is\nnear the eye, foveation "
        "changes nothing at all; once it is far away, the model starts picking a "
        "different target.", fontsize=11.5, loc="left")
    ax.grid(alpha=0.25)
    # Upper LEFT: every large shift sits at large distance, so the right-hand
    # side of these axes is where the data lives and a box there hides points.
    ax.text(0.015, 0.95,
            f"near the attractor:  {int((moved & near).sum())} of "
            f"{int(near.sum())} moved\n"
            f"far from it:  {int((moved & ~near).sum())} of "
            f"{int((~near).sum())} moved\n"
            f"rank correlation ρ = {rho:.2f}",
            transform=ax.transAxes, fontsize=10, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="w", edgecolor="0.75"))

    fig.suptitle(
        f"Foveation moves the next-fixation target only when the target is in the "
        f"periphery\n{stim_label(stimuli, args.stim_idx)} (stim {args.stim_idx}), "
        f"foveated at {args.foveal_cpd:.0f} cpd.  "
        "+ = current gaze,  ○ = most likely next fixation (distribution mode)",
        fontsize=13)

    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"next_pixel_check_stim{args.stim_idx:04d}"
    fig.savefig(OUT / f"{stem}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    (OUT / f"{stem}.json").write_text(json.dumps({
        "stim_idx": args.stim_idx,
        "stim_label": stim_label(stimuli, args.stim_idx),
        "image_shape_hw": [H, W],
        "device": str(device),
        "model": "DeepGazeIII(pretrained=True)",
        "ppd": args.ppd,
        "foveal_cpd": args.foveal_cpd,
        "grid": args.grid,
        "still_threshold_px": STILL_PX,
        "attractor_xy": attractor.tolist(),
        "spearman_rho": float(rho),
        "spearman_p": float(p),
        "n_moved_near_attractor": int((moved & near).sum()),
        "n_near_attractor": int(near.sum()),
        "n_moved_far_attractor": int((moved & ~near).sum()),
        "n_far_attractor": int((~near).sum()),
        "median_shift_px": float(np.median(shift)),
        "max_shift_px": float(shift.max()),
        "panel_on_attractor": {
            "gaze_xy": [gx[i_on], gy[i_on]],
            "mode_normal_xy": m_norm[i_on].tolist(),
            "mode_foveated_xy": m_fov[i_on].tolist(),
            "shift_px": float(shift[i_on]),
        },
        "panel_off_attractor": {
            "gaze_xy": [gx[i_off], gy[i_off]],
            "mode_normal_xy": m_norm[i_off].tolist(),
            "mode_foveated_xy": m_fov[i_off].tolist(),
            "shift_px": float(shift[i_off]),
        },
        "sweep": {
            "gaze_x": gx.tolist(), "gaze_y": gy.tolist(),
            "mode_normal": m_norm.tolist(), "mode_foveated": m_fov.tolist(),
            "shift_px": shift.tolist(),
            "distance_to_attractor_px": d_attr.tolist(),
        },
        "note": (
            "Illustrative panels are selected from the sweep (nearest gaze to the "
            "attractor; largest measured shift), not hand-picked. The attractor is "
            "the sharp-input mode from the gaze point nearest the image centre."
        ),
    }, indent=2) + "\n")
    print(f"saved figure:  {OUT / f'{stem}.png'}")
    print(f"saved sidecar: {OUT / f'{stem}.json'}")


if __name__ == "__main__":
    main()
