"""How two arms score the same human scanpath, step by step.

One *scanpath* under two arms, in the mode the evaluator runs: the history is
always the human's own fixations, never the model's, and at every step both arms
are asked for the probability of the pixel that subject moved to next. That is
the quantity ``evaluate.run`` averages, drawn one term at a time.

The distinction matters because the other scanpath figure in this repo
(``demo_foveated_scanpath.py``) *samples* -- it feeds the model's own draw back as
history, so its two paths diverge for a reason that has nothing to do with how
either arm is scored. Nothing is measured that way.

Needs the trained per-(arm, fold) checkpoints, which are too large to commit:

    python scripts/scanpath_density_panel.py --ckpt-root results/foveation_mit1003/ckpts_initial

The stimulus must be in the *test* half of the fold whose checkpoints are loaded,
or the read-out is being shown an image it trained on. The default pair (stimulus
91, fold 9) satisfies that; subject 7 is the running example of ch01 and ch03.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import deepgaze_pytorch  # noqa: E402

from tez_deepgaze.centerbias import load_centerbias_for_image  # noqa: E402
from tez_deepgaze.device import pick_device, to_device  # noqa: E402
from tez_deepgaze.figstyle import canvas  # noqa: E402
from tez_deepgaze.foveate_input import (  # noqa: E402
    MIT1003_PPD,
    Foveation,
    identity_radius_px,
)
from tez_deepgaze.human_scanpaths import all_human_scanpaths  # noqa: E402
from tez_deepgaze.instrument import compute_log_density, image_tensor  # noqa: E402
from tez_deepgaze.paths import load_mit1003_variant  # noqa: E402
from tez_deepgaze.script_utils import image_rgb, set_heads, stim_label  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "foveation_mit1003_initial" / "figs"
LN2 = float(np.log(2.0))

# Same colour key as the other results figures.
COLOUR = {"normal": "#555555", "fov_cpd40": "#E69F00",
          "fov_cpd20": "#C2185B", "fov_cpd10": "#009E73"}
CPD = {"normal": None, "fov_cpd40": 40.0, "fov_cpd20": 20.0, "fov_cpd10": 10.0}
LABEL = {"normal": "sharp control", "fov_cpd40": "gaze-contingent @ 40",
         "fov_cpd20": "gaze-contingent @ 20", "fov_cpd10": "gaze-contingent @ 10"}


def scanpath_for(fixations, stim_idx: int, subject: int):
    for xs, ys, subj in all_human_scanpaths(fixations, stim_idx):
        if int(subj) == subject:
            return xs, ys
    raise SystemExit(f"stim {stim_idx}: no scanpath for subject {subject}")


def walk_arm(model, pretrained_state, ckpt: Path, cpd, image, cb, xs, ys, ppd, device):
    """Per-step log-densities for one arm, teacher-forced on the human history.

    Step ``i`` conditions on fixations ``0..i`` and is scored at fixation
    ``i + 1``, the same indexing ``evaluate.run`` uses with ``start_fixation=1``.
    """
    set_heads(model, pretrained_state, ckpt, device)
    fov = None if cpd is None else Foveation(ppd=ppd, foveal_cpd=cpd).to(device)
    steps = []
    for i in range(len(xs) - 1):
        hx, hy = xs[:i + 1], ys[:i + 1]
        logd = compute_log_density(model, image, cb, hx, hy, device, foveation=fov)
        ti, tj = int(round(float(ys[i + 1]))), int(round(float(xs[i + 1])))
        shown = image
        if fov is not None:
            t = image_tensor(image, device)
            f = fov.foveate_shared_image(t, torch.tensor([float(hx[-1])], device=device),
                                         torch.tensor([float(hy[-1])], device=device))
            shown = f[0].permute(1, 2, 0).cpu().numpy().clip(0, 255).astype(np.uint8)
        steps.append({
            "i": i,
            "logd": logd,
            "shown": shown,
            # Information gain in bits at the pixel the subject actually chose --
            # one term of the average the tables report.
            "ig_bits": float((logd[ti, tj] - cb[ti, tj]) / LN2),
            "amp_px": float(np.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i])),
        })
        print(f"  step {i} -> {i+1}  {steps[-1]['amp_px']:6.0f} px  "
              f"{steps[-1]['ig_bits']:+.3f} bits")
    return steps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt-root", type=Path, required=True,
                    help="the trained matrix: {tag}/fold{k}/epoch_{E:03d}/weights.pt")
    ap.add_argument("--stim", type=int, default=91)
    ap.add_argument("--subject", type=int, default=7)
    ap.add_argument("--arm", default="fov_cpd20", choices=[a for a in CPD if a != "normal"],
                    help="the foveated arm to set against the sharp control")
    ap.add_argument("--fold", type=int, default=9,
                    help="the fold whose checkpoints to load; --stim must be in its test half")
    ap.add_argument("--epoch", type=int, default=5, help="the reporting epoch")
    ap.add_argument("--ppd", type=float, default=MIT1003_PPD)
    ap.add_argument("--dataset-variant", default="initial", choices=["plain", "initial"],
                    help="'initial' is the thesis protocol: index 0 is the central fixation")
    # Five rows, not the whole scanpath. The figure shows how one step is scored,
    # and it makes that point by the third row; nine rows cost a full float page
    # and shrank every panel to make it three more times.
    ap.add_argument("--max-steps", type=int, default=5,
                    help="draw only the first N steps (default 5; pass 0 for all)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or OUT / f"scanpath_density_{args.arm}_stim{args.stim:04d}.png"

    device = pick_device()
    print(f"device: {device}")
    stimuli, fixations = load_mit1003_variant(args.dataset_variant)
    image = image_rgb(stimuli, args.stim)
    H, W = image.shape[:2]
    cb = load_centerbias_for_image(H, W)
    xs, ys = scanpath_for(fixations, args.stim, args.subject)
    print(f"stim {args.stim} ({stim_label(stimuli, args.stim)}), subject {args.subject}, "
          f"{len(xs)} fixations -> {len(xs) - 1} scored steps")

    model = to_device(deepgaze_pytorch.DeepGazeIII(pretrained=True), device).eval()
    pretrained_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    walks = {}
    for tag in ("normal", args.arm):
        ckpt = args.ckpt_root / tag / f"fold{args.fold}" / f"epoch_{args.epoch:03d}" / "weights.pt"
        if not ckpt.exists():
            raise SystemExit(f"{ckpt} missing — copy the trained matrix or fix --ckpt-root")
        print(f"{tag}:")
        walks[tag] = walk_arm(model, pretrained_state, ckpt, CPD[tag], image, cb,
                              xs, ys, args.ppd, device)

    ctl, fvd = walks["normal"], walks[args.arm]
    n = len(ctl) if not args.max_steps else min(args.max_steps, len(ctl))
    radius = identity_radius_px(CPD[args.arm], args.ppd)
    colour = COLOUR[args.arm]

    # One shared scale per row-type, so a panel's brightness is comparable down
    # the column as well as across it.
    vmin = min(min(s["logd"].min() for s in ctl), min(s["logd"].min() for s in fvd))
    vmax = max(max(s["logd"].max() for s in ctl), max(s["logd"].max() for s in fvd))
    diffs = [(f["logd"] - c["logd"]) / LN2 for c, f in zip(ctl[:n], fvd[:n])]
    dlim = max(float(np.abs(d).max()) for d in diffs)

    cw = 2.75
    # Sized for the width ch03 gives the figure, so the panel headings print
    # at about 8 pt; hspace grows with them so a heading clears the row above.
    fig, axes = plt.subplots(n, 4, **canvas((4 * cw, n * cw * H / W + 0.9),
                                            page_height_frac=0.86),
                             gridspec_kw={"hspace": 0.24, "wspace": 0.02})
    axes = np.atleast_2d(axes)

    def frame(ax):
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    for r in range(n):
        c, f = ctl[r], fvd[r]
        hx, hy = xs[:r + 1], ys[:r + 1]
        tx, ty = float(xs[r + 1]), float(ys[r + 1])

        ax = axes[r, 0]
        ax.imshow(f["shown"])
        ax.plot(hx, hy, "-o", color="white", ms=3.0, lw=1.1, alpha=0.85)
        ax.plot([hx[-1]], [hy[-1]], "o", ms=10, mfc="none", mec="white", mew=2.2)
        if radius > 1:
            ax.add_patch(plt.Circle((hx[-1], hy[-1]), radius, fill=False,
                                    color="white", lw=1.2, ls="--", alpha=0.9))
        ax.annotate("", xy=(tx, ty), xytext=(hx[-1], hy[-1]),
                    arrowprops=dict(arrowstyle="->", color="#B22222", lw=2.0))
        ax.plot([tx], [ty], "*", ms=14, color="#B22222", mec="white", mew=0.7)
        frame(ax)
        ax.set_ylabel(f"step {r} $\\rightarrow$ {r + 1}\n{f['amp_px']:.0f} px",
                      fontsize=9.5, labelpad=6)
        if r == 0:
            # The frame is what the foveated arm receives; the caption says so.
            ax.set_title("input", fontsize=10.5, color=colour, pad=8)

        short = f"gaze-contingent @ {CPD[args.arm]:g}"
        for col, (rec, lab, col_colour) in enumerate(
                ((c, "sharp control", COLOUR["normal"]),
                 (f, short, colour)), start=1):
            ax = axes[r, col]
            ax.imshow(rec["logd"], cmap="magma", vmin=vmin, vmax=vmax)
            ax.plot([tx], [ty], "*", ms=14, color="#39FF14", mec="black", mew=0.7)
            frame(ax)
            # No per-panel value: one scanpath is an illustration, and a number
            # on it read as an effect size. The values stay in the JSON sidecar.
            if r == 0:
                ax.set_title(lab, fontsize=10.5, color=col_colour, pad=8)

        ax = axes[r, 3]
        im = ax.imshow(diffs[r], cmap="RdBu_r", vmin=-dlim, vmax=dlim)
        ax.plot([tx], [ty], "*", ms=14, color="#111111", mec="white", mew=0.7)
        frame(ax)
        if r == 0:
            ax.set_title("difference", fontsize=10.5, color="0.3", pad=8)

    fig.colorbar(im, cax=fig.add_axes([0.915, 0.06, 0.008, 0.12]),
                 label="$\\Delta$ log-density (bits)")
    total_c = float(np.mean([s["ig_bits"] for s in ctl]))
    total_f = float(np.mean([s["ig_bits"] for s in fvd]))
    # One line; the numbers are the caption's job.
    fig.suptitle(f"Stimulus {args.stim}, subject {args.subject}",
                 fontsize=12.5, y=0.995)
    # The header band (suptitle plus one row of column headings) is a fixed height
    # in inches, so its share of the canvas depends on how many steps are drawn.
    # Hardcoding a fraction put the title through the headings at five rows.
    top = 1.0 - 0.72 / float(fig.get_size_inches()[1])
    fig.subplots_adjust(left=0.055, right=0.905, top=top, bottom=0.012)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi="figure", bbox_inches="tight")
    plt.close(fig)

    out.with_suffix(".json").write_text(json.dumps({
        "stim": args.stim, "stim_label": stim_label(stimuli, args.stim),
        "subject": args.subject, "arm": args.arm, "fold": args.fold, "epoch": args.epoch,
        "ppd": args.ppd, "dataset_variant": args.dataset_variant,
        "sharp_radius_px": radius, "image_shape": [H, W],
        "n_steps_scored": len(ctl), "n_steps_drawn": n,
        "ckpt_root": str(args.ckpt_root), "device": str(device),
        "scanpath_x": [float(v) for v in xs], "scanpath_y": [float(v) for v in ys],
        "amp_px": [s["amp_px"] for s in ctl],
        "ig_bits_normal": [s["ig_bits"] for s in ctl],
        f"ig_bits_{args.arm}": [s["ig_bits"] for s in fvd],
        "scanpath_mean_ig_normal": total_c,
        f"scanpath_mean_ig_{args.arm}": total_f,
    }, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
