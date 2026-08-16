"""The Chapter 2 (Background) figures.

The first is a pure formula plot; the others read MIT1003 stimuli and human
fixations, so they need the corpus. Only ``dg3_composition`` needs the
pretrained weights, and it runs two forward passes on one image.

1. ``retinal_falloff.png``     what the Geisler-Perry cutoff simplifies away —
                               the relative resolution implied by Watson (2014)
                               midget-RGC density along each of the four
                               meridians, against the one radial cutoff the
                               thesis uses.
2. ``saliency_vs_scanpath.png`` the same image as a fixation *distribution* and
                               as an ordered *sequence* — the two prediction
                               problems Chapter 2 distinguishes.
3. ``dg3_composition.png``     DG3's read-out plus the weighted centerbias, and
                               the probability map they sum to — the addition
                               the Finalizer performs, on one image.
4. ``one_fixation.png``        one scene as a single fixation delivers it —
                               sharp, at human acuity, and at a coarser-than-
                               human cutoff. Runs the Foveation of Chapter 3
                               forward on one stimulus, so it needs a device
                               but no weights.

All write to ``results/background/``.

    .venv/bin/python scripts/make_background_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tez_deepgaze.figstyle import canvas  # noqa: E402
from tez_deepgaze.paths import DATA_ROOT, RESULTS  # noqa: E402

OUT = RESULTS / "background"

# --- Watson (2014), Table 1: midget-RGC receptive-field density -------------
# d(r) = 2 * d_c(0) * [a (1 + r/r_2)^-2 + (1 - a) exp(-r/r_e)]  per meridian.
# The leading 2 is Watson's constraint that each foveal cone drives two midget
# RGCs; d_c(0) is the Curcio et al. (1990) foveal cone density.
DC0 = 14804.6  # cones/deg^2
MERIDIANS = {
    "temporal": dict(a=0.9851, r2=1.058, re=22.14),
    "superior": dict(a=0.9935, r2=1.035, re=16.35),
    "nasal":    dict(a=0.9729, r2=1.084, re=7.633),
    "inferior": dict(a=0.9960, r2=0.9932, re=12.13),
}
COLORS = {"temporal": "#1f77b4", "superior": "#d62728",
          "nasal": "#2ca02c", "inferior": "#9467bd"}

# Geisler & Perry (1998) acuity cutoff: f_c(e) = f_c0 * e_2 / (e + e_2).
E2_GP = 2.3  # degrees

# Peripheral patch magnified in one_fixation.png (x, y, w, h in stim-91
# pixels): the group of people on the left, ~10 deg from the chosen
# fixation and the highest-frequency region at that eccentricity.
CROP = (240, 555, 240, 170)


def watson_density(r: np.ndarray, meridian: str) -> np.ndarray:
    p = MERIDIANS[meridian]
    return 2.0 * DC0 * (p["a"] * (1.0 + r / p["r2"]) ** -2
                        + (1.0 - p["a"]) * np.exp(-r / p["re"]))


def fig_retinal_falloff() -> None:
    r = np.linspace(0.0, 30.0, 600)
    fig, ax = plt.subplots(1, 1, **canvas((6.2, 4.2), page_frac=0.72))

    # Relative *linear* resolution: sampling density is per unit area, so the
    # resolution it supports goes as its square root. Both curves are
    # normalised to the fovea, which is the only way to compare an anatomical
    # count with a fitted psychophysical cutoff.
    for name, colour in COLORS.items():
        rel = np.sqrt(watson_density(r, name) / watson_density(np.array([0.0]), name)[0])
        ax.plot(r, rel, color=colour, lw=1.4, alpha=0.75, label=name)
    ax.plot(r, E2_GP / (r + E2_GP), color="#000000", lw=2.4, ls="--",
            label="Geisler–Perry  $e_2/(e+e_2)$")
    ax.set_xlabel("eccentricity $r$ (deg)")
    ax.set_ylabel("resolution, relative to the fovea")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 1.02)

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "retinal_falloff.png", dpi="figure", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "retinal_falloff.png")


def fig_saliency_vs_scanpath(stim_idx: int = 91, subject: int = 0) -> None:
    import pysaliency
    from scipy.ndimage import gaussian_filter

    from tez_deepgaze.human_scanpaths import all_human_scanpaths, pick_human_scanpath
    from tez_deepgaze.script_utils import draw_scanpath, image_rgb, stim_label

    stimuli, fixations = pysaliency.get_mit1003(location=str(DATA_ROOT))
    img = image_rgb(stimuli, stim_idx)
    h, w = img.shape[:2]

    paths = all_human_scanpaths(fixations, stim_idx)
    xs_all = np.concatenate([p[0] for p in paths])
    ys_all = np.concatenate([p[1] for p in paths])

    # Fixation density over every observer: the "where", with order discarded.
    heat = np.zeros((h, w), dtype=float)
    xi = np.clip(xs_all.astype(int), 0, w - 1)
    yi = np.clip(ys_all.astype(int), 0, h - 1)
    np.add.at(heat, (yi, xi), 1.0)
    heat = gaussian_filter(heat, sigma=0.035 * max(h, w))
    heat /= heat.max()

    fig, axes = plt.subplots(1, 2, **canvas((11.0, 4.4)))
    for ax in axes:
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])

    # Per-pixel alpha, so unfixated regions keep the photograph legible instead
    # of being dimmed by a flat overlay.
    axes[0].imshow(heat, cmap="inferno", alpha=np.clip(heat * 1.35, 0, 0.88))
    # What each panel discards, and that they hold the same fixations, is in the
    # caption; the title carries only whose gaze it is. MIT1003 is fifteen
    # subjects for every image, spelled out as the thesis spells it — the count
    # stays a number in the sidecar.
    axes[0].set_title("A. Fixation distribution, all subjects", fontsize=10)

    xs, ys, subj = pick_human_scanpath(fixations, stim_idx, subject_idx=subject)
    # Half a page wide, so the default markers put the digits under 4 pt.
    # Same blue as the intro figure, so the two scanpaths read as one thing.
    draw_scanpath(axes[1], xs, ys, color="#1d6fb8", marker_scale=2.2)
    axes[1].set_title(f"B. Subject {subj}'s scanpath", fontsize=10)

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "saliency_vs_scanpath.png", dpi="figure", bbox_inches="tight")
    plt.close(fig)

    (OUT / "saliency_vs_scanpath.json").write_text(json.dumps({
        "stim_idx": stim_idx, "stim": stim_label(stimuli, stim_idx),
        "subject": subj, "n_observers": len(paths),
        "n_fixations_total": int(len(xs_all)), "image_shape": [h, w],
        "kde_sigma_frac": 0.035,
    }, indent=2) + "\n")
    print("wrote", OUT / "saliency_vs_scanpath.png")


def fig_dg3_composition(stim_idx: int = 91, subject: int = 0, n_hist: int = 4) -> None:
    """The one place DG3's terms combine by addition: read-out + prior.

    The Finalizer adds the weighted log-centerbias to the blurred read-out and
    normalises, so running the model twice — once with the real prior, once with
    a flat one — recovers the term it adds, to the small residual the sidecar
    records. (The released model averages ten read-outs, so the recovered term is
    close to the mean weight times the centerbias rather than equal to it.) This
    isolates a term for display; it is not an information-gain accounting, which
    ``ig.py`` rules ``cb=zeros`` out for. The image and the fixation
    history do not separate this way: they enter a learned network together, so
    they are shown as one read-out rather than as two summands.
    """
    import deepgaze_pytorch
    import pysaliency

    from tez_deepgaze.centerbias import load_centerbias_for_image
    from tez_deepgaze.device import pick_device, to_device
    from tez_deepgaze.human_scanpaths import pick_human_scanpath
    from tez_deepgaze.instrument import compute_log_density
    from tez_deepgaze.script_utils import image_rgb, stim_label

    stimuli, fixations = pysaliency.get_mit1003(location=str(DATA_ROOT))
    img = image_rgb(stimuli, stim_idx)
    h, w = img.shape[:2]
    cb = load_centerbias_for_image(h, w)
    xs, ys, subj = pick_human_scanpath(fixations, stim_idx, subject_idx=subject)
    hx = [float(v) for v in xs[:n_hist]]
    hy = [float(v) for v in ys[:n_hist]]

    device = pick_device()
    model = to_device(deepgaze_pytorch.DeepGazeIII(pretrained=True), device).eval()
    full = compute_log_density(model, img, cb, hx, hy, device)
    flat = compute_log_density(model, img, np.zeros_like(cb), hx, hy, device)
    prior = full - flat          # what the Finalizer added, up to the normalisation

    weights = [float(f.center_bias_weight.detach().cpu()) for f in model.finalizers]
    w_mean = float(np.mean(weights))
    # The prior panel is close to the centerbias itself. The residual is small
    # and covers both the Finalizer's down- and upsampling around the addition
    # and the ten read-outs the released model averages.
    resid_sd = float((prior - w_mean * cb).std())

    # A and C are normalised log densities, so one scale serves both and the
    # prior's pull on the corners is visible; B is a difference of two of them,
    # a different quantity, so it carries its own scale.
    lo = min(np.percentile(flat, 1), np.percentile(full, 1))
    hi = max(np.percentile(flat, 99.9), np.percentile(full, 99.9))

    from tez_deepgaze.script_utils import draw_scanpath

    fig, axes = plt.subplots(1, 3, **canvas((12.6, 3.9)))
    # What each panel is made of is in the caption; the titles name them.
    panels = [
        (flat, "A. Read-out", (lo, hi), True),
        (prior, "B. Centerbias prior", (None, None), False),
        (full, "C. Probability map", (lo, hi), True),
    ]
    # Every panel is the same kind of object — a map over the photograph, same
    # colormap, same alpha. Drawing B bare made it read as a different sort of
    # thing standing between two overlays. vmin=vmax=None autoscales B.
    for ax, (arr, title, (vmin, vmax), with_history) in zip(axes, panels):
        ax.imshow(img)
        ax.imshow(arr, cmap="inferno", alpha=0.62, vmin=vmin, vmax=vmax)
        if with_history:
            # The four fixations the read-out is conditioned on, in the blue
            # of the other scanpath figures.
            draw_scanpath(ax, hx, hy, color="#1d6fb8", marker_scale=2.2)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    # Read the gutters off the laid-out axes rather than measuring them once:
    # the panel widths follow the stimulus aspect, which stim_idx can change.
    box = [ax.get_position() for ax in axes]
    mid = (box[1].y0 + box[1].y1) / 2
    for (left, right), glyph in zip(((box[0].x1, box[1].x0),
                                     (box[1].x1, box[2].x0)), "+="):
        fig.text((left + right) / 2, mid, glyph, fontsize=22,
                 ha="center", va="center")

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "dg3_composition.png", dpi="figure", bbox_inches="tight")
    plt.close(fig)

    (OUT / "dg3_composition.json").write_text(json.dumps({
        "stim_idx": stim_idx, "stim": stim_label(stimuli, stim_idx),
        "subject": subj, "history_x": hx, "history_y": hy,
        "image_shape": [h, w], "device": str(device),
        "model": "DeepGazeIII(pretrained=True)",
        "center_bias_weights": [round(v, 4) for v in weights],
        "center_bias_weight_mean": round(w_mean, 4),
        "prior_residual_sd_log_units": round(resid_sd, 4),
        "prior_span_log_units": round(float(prior.max() - prior.min()), 4),
        "shared_scale_log_density": [round(float(lo), 3), round(float(hi), 3)],
        "note": (
            "panel B is full - flat, the term the model adds; it is close to "
            "center_bias_weight_mean * centerbias, to prior_residual_sd_log_units"
        ),
    }, indent=2) + "\n")
    print("wrote", OUT / "dg3_composition.png")


def fig_one_fixation(stim_idx: int = 91, subject: int = 0, fix_idx: int = 3) -> None:
    import pysaliency
    import torch

    from tez_deepgaze.device import pick_device, to_device
    from tez_deepgaze.foveate_input import MIT1003_PPD, Foveation, identity_radius_px
    from tez_deepgaze.human_scanpaths import pick_human_scanpath
    from tez_deepgaze.script_utils import image_rgb, stim_label

    stimuli, fixations = pysaliency.get_mit1003(location=str(DATA_ROOT))
    img = image_rgb(stimuli, stim_idx)
    xs, ys, subj = pick_human_scanpath(fixations, stim_idx, subject_idx=subject)
    fx, fy = float(xs[fix_idx]), float(ys[fix_idx])

    device = pick_device()
    img_t = to_device(torch.from_numpy(img).permute(2, 0, 1)[None].float(), device)
    fx_t = to_device(torch.tensor([fx]), device)
    fy_t = to_device(torch.tensor([fy]), device)

    # The pyramid depends only on the level count, so one stack serves both
    # strengths; only the per-pixel level map differs between them.
    strengths = [40.0, 10.0]
    fov = Foveation(ppd=MIT1003_PPD, foveal_cpd=strengths[0]).to(device)
    stack = fov.blur_stack(img_t)
    out = {}
    for cpd in strengths:
        f = Foveation(ppd=MIT1003_PPD, foveal_cpd=cpd).to(device)
        y = f.foveate_shared_image(img_t, fx_t, fy_t, stack=stack)
        out[cpd] = y[0].permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()

    titles = [
        "A. The image (sharp)",
        "B. A human eye ($f_{c0}=40$)",
        "C. A coarser eye ($f_{c0}=10$)",
    ]
    panels = [img, out[40.0], out[10.0]]
    radii = [None, identity_radius_px(40.0, MIT1003_PPD),
             identity_radius_px(10.0, MIT1003_PPD)]

    # Faithful foveation is mild at this image size, so the full frames alone
    # show nothing. The second row magnifies one peripheral patch, chosen for
    # high-frequency content well away from the fixation.
    cx0, cy0, cw, ch = CROP
    crop_ecc = float(np.hypot(cx0 + cw / 2 - fx, cy0 + ch / 2 - fy) / MIT1003_PPD)

    fig, axes = plt.subplots(2, 3, **canvas((12.4, 6.6)),
                             gridspec_kw={"height_ratios": [1.0, 0.92]})
    for col, (panel, title, rad) in enumerate(zip(panels, titles, radii)):
        top = axes[0, col]
        top.imshow(panel)
        top.plot(fx, fy, marker="+", ms=13, mew=2.0, color="#E8000B")
        if rad:
            top.add_patch(plt.Circle((fx, fy), rad, fill=False, lw=1.3,
                                     ls="--", color="#E8000B", alpha=0.95))
        top.add_patch(plt.Rectangle((cx0, cy0), cw, ch, fill=False, lw=1.2,
                                    color="#FFD200"))
        top.set_title(title, fontsize=9.5)

        # The box on the frame says which region this is; the magnified patch
        # needs no frame of its own.
        bot = axes[1, col]
        bot.imshow(panel[cy0:cy0 + ch, cx0:cx0 + cw])
        bot.axis("off")

    axes[1, 0].set_ylabel(f"magnified, {crop_ecc:.0f}$^\\circ$ out", fontsize=9)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "one_fixation.png", dpi="figure", bbox_inches="tight")
    plt.close(fig)

    (OUT / "one_fixation.json").write_text(json.dumps({
        "stim_idx": stim_idx, "stim": stim_label(stimuli, stim_idx),
        "subject": subj, "fixation_index": fix_idx, "fixation_px": [fx, fy],
        "ppd": MIT1003_PPD, "foveal_cpd": strengths,
        "identity_radius_px": {str(c): identity_radius_px(c, MIT1003_PPD)
                               for c in strengths},
        "crop_xywh": list(CROP), "crop_eccentricity_deg": crop_ecc,
        "image_shape": list(img.shape[:2]), "device": str(device),
    }, indent=2) + "\n")
    print("wrote", OUT / "one_fixation.png")


def fixation_entropy(xs, ys, height: int, width: int, grid: int = 16) -> float:
    """Normalised Shannon entropy of pooled fixations over a ``grid`` x ``grid``
    partition of the frame, in [0, 1]. Zero means every fixation in one cell;
    one means a uniform spread. Same quantity as Chapter 3's H_fix."""
    k = grid * grid
    gx = np.clip((np.asarray(xs) / width * grid).astype(int), 0, grid - 1)
    gy = np.clip((np.asarray(ys) / height * grid).astype(int), 0, grid - 1)
    counts = np.bincount(gy * grid + gx, minlength=k).astype(float)
    p = counts / counts.sum()
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum() / np.log2(k))


def rank_by_dispersion(stimuli, fixations, min_fixations: int = 10):
    """Every stimulus ranked by :func:`fixation_entropy`, most dispersed first.

    Returns ``[(stim_idx, H_fix), ...]``. This is what backs the figure's claim
    that a chosen stimulus is among the corpus's most or least dispersed, so
    the claim is reproducible rather than asserted.
    """
    from tez_deepgaze.human_scanpaths import all_human_scanpaths

    sizes = stimuli.sizes if hasattr(stimuli, "sizes") else stimuli.shapes
    out = []
    for i in range(len(sizes)):
        try:
            paths = all_human_scanpaths(fixations, i)
        except RuntimeError:
            continue
        xs = np.concatenate([p[0] for p in paths]) if paths else np.empty(0)
        ys = np.concatenate([p[1] for p in paths]) if paths else np.empty(0)
        if len(xs) < min_fixations:
            continue
        out.append((i, fixation_entropy(xs, ys, sizes[i][0], sizes[i][1])))
    out.sort(key=lambda r: -r[1])
    return out


def fig_scanpath_variability(
    stims: tuple[int, int] = (62, 121),
    n_show: int = 5,
) -> None:
    """Individual observers' scanpaths on two stimuli, one panel each.

    Two rows, because they make different halves of the same argument. The
    first stimulus has high spatial agreement -- every observer looks at the
    same object -- yet the order and the number of fixations differ sharply.
    The second is near the top of the corpus by fixation entropy, where
    observers land in different regions altogether.

    Both stimuli's entropy and their rank among all 1003 are recomputed here by
    :func:`rank_by_dispersion` and written to the JSON sidecar, so the caption's
    claim about dispersion is regenerable from this file.

    The observers shown are the first ``n_show`` in subject order, not a
    selection: with 15 per stimulus, showing all of them at page width would
    leave each panel too small to read a path in.
    """
    import pysaliency

    from tez_deepgaze.human_scanpaths import all_human_scanpaths
    from tez_deepgaze.script_utils import image_rgb, stim_label

    stimuli, fixations = pysaliency.get_mit1003(location=str(DATA_ROOT))

    ranking = rank_by_dispersion(stimuli, fixations)
    rank_of = {s: (k + 1, h) for k, (s, h) in enumerate(ranking)}

    # tab10, not tab20: each panel holds one observer, so the colour only has
    # to separate panels. tab20's pale alternates disappear against the white
    # outline the paths are drawn with.
    palette = [plt.cm.tab10(i) for i in range(10)]

    rows = []
    for s in stims:
        img = image_rgb(stimuli, s)
        paths = [p for p in all_human_scanpaths(fixations, s) if len(p[0]) > 0]
        rows.append((s, img, paths))

    # Size each row by its own aspect ratio, so a portrait stimulus does not
    # get letterboxed into a landscape row.
    ratios = [r[1].shape[0] / r[1].shape[1] for r in rows]
    fig = plt.figure(**canvas((2.25 * n_show, sum(2.25 * r for r in ratios) + 0.9)))
    gs = fig.add_gridspec(len(rows), n_show, height_ratios=ratios,
                          hspace=0.14, wspace=0.06)

    # The photograph is drawn at full opacity: dimming it to make the path
    # legible washes the stimulus out against the page. The path carries its
    # own white outline instead, which reads over light and dark content alike.
    halo = [pe.Stroke(linewidth=3.2, foreground="white"), pe.Normal()]

    for r, (s, img, paths) in enumerate(rows):
        for k in range(n_show):
            ax = fig.add_subplot(gs[r, k])
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            xs, ys, subj = paths[k]
            c = palette[k % len(palette)]
            ax.plot(xs, ys, "-", color=c, linewidth=1.6, zorder=3,
                    path_effects=halo)
            ax.scatter(xs, ys, s=16, color=c, edgecolor="white",
                       linewidth=0.8, zorder=4)
            ax.plot(xs[0], ys[0], "o", color="white", markersize=6,
                    markeredgecolor="black", markeredgewidth=1.1, zorder=5)
            # Which subjects these are is in the caption and the sidecar; the
            # count is what the caption's "four to nine" refers to.
            ax.set_title(f"{len(xs)} fixations", fontsize=9, pad=2)
            if k == 0:
                ax.set_ylabel(f"stimulus {s}", fontsize=9)

    fig.savefig(OUT / "scanpath_variability.png", dpi="figure", bbox_inches="tight")
    plt.close(fig)

    (OUT / "scanpath_variability.json").write_text(json.dumps({
        "stims": list(stims),
        "stim_labels": [stim_label(stimuli, s) for s in stims],
        "n_shown_per_stim": n_show,
        "observers_shown": [[int(p[2]) for p in r[2][:n_show]] for r in rows],
        "n_observers_total": [len(r[2]) for r in rows],
        "fixation_counts_shown": [[int(len(p[0])) for p in r[2][:n_show]]
                                  for r in rows],
        "selection": "first n_show in subject order, not curated",
        "fixation_entropy": {str(s): round(rank_of[s][1], 4) for s in stims},
        "dispersion_rank": {str(s): rank_of[s][0] for s in stims},
        "n_stimuli_ranked": len(ranking),
        "rank_note": "1 = most dispersed; see rank_by_dispersion()",
    }, indent=2) + "\n")
    print("wrote", OUT / "scanpath_variability.png")


if __name__ == "__main__":
    fig_retinal_falloff()
    fig_saliency_vs_scanpath()
    fig_dg3_composition()
    fig_one_fixation()
    fig_scanpath_variability()
