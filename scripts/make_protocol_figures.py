"""Every thesis figure that does not need the trained model.

None of these needs a forward pass — they are properties of the transform, the
corpus and the already-measured results. The five ``gp_*`` figures and
``foveation_strength`` load MIT1003 stimuli and build a Gaussian pyramid at full
resolution, so run them on ``pick_device()``: that stage is about 1.7 s on MPS or
CUDA and minutes on CPU. The rest run on a laptop in seconds.

Mechanism (ch03 §Space-variant blur — what the transform is):

1. ``gp_eccentricity.png`` the map every later quantity is a function of: each
                           pixel's distance from the current fixation.
2. ``gp_pyramid.png``      the octave-spaced pyramid, the per-pixel fractional
                           level, and the blend of the two that makes a frame.
3. ``gp_weights.png``      the blend written out: per-level weight maps summing
                           to 1 at every pixel, at the human arm.
4. ``gp_stepvsinterp.png`` why the blend is continuous: nearest-level rounding
                           bands and rings, interpolation does not.
5. ``gp_strength_grid.png`` foveation strength across content, fovea at centre.

Protocol (ch03 §Training protocol / ch04 §Dose-response — what it does):

6. ``aliasing_demo.png``   what DeepGaze III's nearest-neighbour halving does to
                           a sharp periphery, and why our blur accidentally
                           fixes it.
7. ``foveation_strength.png``  the sharp radius: the acuity cutoff crossing the
                           display Nyquist limit, and the disc that crossing
                           implies drawn on a stimulus at true scale.
8. ``saccade_coverage.png``  the human saccade amplitudes, with each cutoff's
                           sharp radius on them and the share of targets inside.
9. ``dose_response.png``  measured dIG against foveal cutoff, for the
                           gaze-contingent and the fixed-centre arm, next to the
                           share of saccade targets inside the sharp disc.
10. ``stratified_amplitude.png``  dIG against saccade amplitude for three arms with
                           their sharp radii marked. At cpd 40 the profile is flat
                           on both sides of the disc — see ``disc_contrast.png``,
                           which tests the two regions against each other.
11. ``stratified_fixation_index.png``  dIG against fixation index. No comparable
                           trend at cpd 40 or 20; at cpd 10 the cost is deepest
                           over the first two scored fixations and shrinks along
                           the scanpath.

Figures 7 and 8 are separate because the methodology chapter asks two questions
of them a page apart — what the radius is, then what it covers.

The stratified pairing is fixation-for-fixation, so the centerbias term is the
same on both sides and cancels: the ``d_LL`` these files record is identical to
the information-gain difference the thesis reports, and the figures carry the
chapter's name for it.

Figures 10 and 11 read
``results/foveation_mit1003_initial/stratified/cpd*_val/stratified.json`` and need that
analysis to have been run (``scripts/foveation_stratified.py``).

Every ``gp_*`` figure is drawn with :class:`tez_deepgaze.foveate_input.Foveation`
— the same object the training and evaluation arms use — so what the thesis shows
is what the experiment ran.

    .venv/bin/python scripts/make_protocol_figures.py
    .venv/bin/python scripts/make_protocol_figures.py --only gp_pyramid gp_strength_grid
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Circle, Rectangle
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tez_deepgaze.foveate_input import (  # noqa: E402
    E2_DEG,
    MIT1003_PPD,
    Foveation,
    identity_radius_px,
)
from tez_deepgaze.device import pick_device  # noqa: E402
from tez_deepgaze.figstyle import canvas  # noqa: E402
from tez_deepgaze.human_scanpaths import (  # noqa: E402
    CONSENSUS_RADIUS_PX,
    all_human_scanpaths,
    consensus_count,
)
from tez_deepgaze.paths import load_mit1003  # noqa: E402
from tez_deepgaze.script_utils import (  # noqa: E402
    image_rgb,
    sample_indices,
    stim_label,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "foveation_mit1003" / "figs"
CONSENSUS_OUT = ROOT / "results" / "consensus_panels"
PPD = MIT1003_PPD
NYQ_DISPLAY = PPD / 2          # 17.5 cyc/deg, what the monitor can show

# The dose-response points are read from the primary run's committed table at
# figure time rather than held here as literals, so the figure cannot drift
# from the prose and tables that quote the same artefact. dose_response.png is
# written into that tree's figs/ directory alongside the other chapter-4 figures.
DOSE_TREE = ROOT / "results" / "foveation_mit1003_initial"


def _dose_rows() -> list[tuple[float, float, float]]:
    """(cpd, dIG gaze-contingent, dIG fixed-centre), fold-paired means."""
    t = json.loads((DOSE_TREE / "test" / "table.json").read_text())["paired_vs_normal"]
    return [(float(c), t[f"foveated@{c}"]["fold_paired"]["mean"],
             t[f"center@{c}"]["fold_paired"]["mean"]) for c in (40, 20, 10)]

# The running example carried through ch01, ch03 and ch04 (i132306037.jpeg).
GP_STIM = 91
# Content spread for the strength grid. Fixed rather than sampled so the figure
# is stable across runs.
GRID_STIMS = [200, 300, 450, 700]
# Stim 91 is not in the grid: ch02's one_fixation figure already shows it.
#
# Where the fovea sits, and which patch is magnified, per stimulus. Both default
# to the image centre and ``_peripheral_patch``; an entry overrides either as a
# fraction of (W, H). Two rows need it:
#   200 — fovea and patch swapped, so the shop signage falls in the periphery
#         and the figure shows what foveation does to text rather than to brick.
#   700 — fovea in the top-right corner, putting the cat ~23° out, which is the
#         largest eccentricity in the grid and the clearest degradation.
GRID_FOVEA: dict[int, tuple[float, float]] = {200: (0.256, 0.233), 700: (0.88, 0.08)}
GRID_PATCH_CENTER: dict[int, tuple[float, float]] = {200: (0.5, 0.5)}
# Filled by fig_gp_strength_grid: per-cutoff mean |foveated - sharp| over the four
# grid stimuli, in 0-255 units. Written to gp_figures.json on regeneration; the
# ch03 strength-grid caption quotes these numbers.
GRID_MEAN_ABS_CHANGE: dict[str, float] = {}
# The two shallow arm cutoffs, drawn 40 over 20 so the level maps show how
# little of the stack a faithful strength uses.
PYRAMID_CPDS = (40.0, 20.0)
# Filled by fig_gp_pyramid: per-cutoff max of the fractional level L(e) on the
# pyramid stimulus. Written to gp_figures.json; the ch03 caption quotes these.
PYRAMID_LEVEL_MAX: dict[str, float] = {}
STEP_CPD = 10.0
# The human arm, and the one the primary comparison trained on. Used where the
# figure is a readout of numbers rather than a picture of blur — the weight maps
# and the eccentricity map stay legible at 40 because neither shows image detail.
ARM_CPD = 40.0

_STIMULI = None
_FIXATIONS = None
_DEVICE = None


def _corpus():
    global _STIMULI, _FIXATIONS
    if _STIMULI is None:
        _STIMULI, _FIXATIONS = load_mit1003()
    return _STIMULI, _FIXATIONS


def stimuli():
    """MIT1003 stimuli, loaded once. Indices match every other figure in the repo."""
    return _corpus()[0]


def fixations():
    """MIT1003 fixations, loaded once alongside the stimuli."""
    return _corpus()[1]


def device() -> torch.device:
    """Where the foveation runs. Forward-only, so MPS is fine.

    Not a presentation detail: the deepest pyramid level uses a 145x145 kernel
    at full resolution, which is ~50 GFLOP of dense convolution. That is 1.7 s on
    MPS or CUDA and several minutes on CPU, so the ``gp_*`` figures are only
    comfortable to regenerate on a GPU.
    """
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = pick_device()
    return _DEVICE


def image_chw(idx: int) -> torch.Tensor:
    """One stimulus as a ``(1, 3, H, W)`` float tensor in [0, 255], on ``device()``."""
    arr = image_rgb(stimuli(), idx).astype(np.float32)
    return torch.from_numpy(arr.transpose(2, 0, 1))[None].to(device())


def foveation(cpd: float) -> Foveation:
    return Foveation(ppd=PPD, foveal_cpd=cpd).to(device())


def show(ax, t: torch.Tensor) -> None:
    """Draw a ``(1, 3, H, W)`` tensor, axes off."""
    ax.imshow(t[0].permute(1, 2, 0).clamp(0, 255).cpu().numpy().astype(np.uint8))
    ax.set_xticks([])
    ax.set_yticks([])


def level_sigma(fov: Foveation, level: np.ndarray) -> np.ndarray:
    """Effective blur std for a fractional pyramid ``level``.

    The blend weights integer levels with triangular hats, so the effective
    radius is the linear interpolation of the level sigmas — this is a readout
    of what :meth:`Foveation._blend` does, not a separate model of it.
    """
    return np.interp(level, np.arange(fov.n_levels), fov.level_sigmas.cpu().numpy())


def blur_fft(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian blur of std ``sigma`` px via FFT (exact, and O(N log N))."""
    if sigma <= 0:
        return img
    _, _, H, W = img.shape
    fy = torch.fft.fftfreq(H).view(-1, 1)
    fx = torch.fft.rfftfreq(W).view(1, -1)
    h = torch.exp(-2.0 * (np.pi**2) * (sigma**2) * (fy**2 + fx**2))
    return torch.fft.irfft2(torch.fft.rfft2(img) * h, s=(H, W))


def half_nn(t: torch.Tensor) -> torch.Tensor:
    """Nearest-neighbour halving — what DG3's forward does to its input."""
    return F.interpolate(t, scale_factor=0.5, recompute_scale_factor=False)


def half_area(t: torch.Tensor) -> torch.Tensor:
    """Area (correctly prefiltered) halving — the aliasing-free reference."""
    return F.interpolate(t, scale_factor=0.5, mode="area",
                         recompute_scale_factor=False)


def save(fig, name: str, out_dir: Path = OUT) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / name, dpi="figure", bbox_inches="tight")
    plt.close(fig)


def _busiest_patch(score: torch.Tensor, side: int, stride: int,
                   accept=None) -> tuple[int, int]:
    """Top-left of the ``side``×``side`` window with the highest mean score.

    ``accept(y, x)`` optionally rejects window positions (e.g. outside an
    eccentricity band).
    """
    H, W = score.shape
    best, by, bx = -1.0, 0, 0
    for y in range(0, H - side, stride):
        for x in range(0, W - side, stride):
            if accept is not None and not accept(y, x):
                continue
            v = float(score[y:y + side, x:x + side].mean())
            if v > best:
                best, by, bx = v, y, x
    return by, bx


def sharp_radius_deg(cpd: float) -> float:
    """The sharp-disc radius (:func:`identity_radius_px`), in degrees."""
    return identity_radius_px(cpd, PPD) / PPD


def saccade_amplitudes_deg() -> np.ndarray:
    # The initial-fixation variant: index 0 of each scanpath is the recorded
    # central fixation, so the diffs below are exactly the 104,171 scored
    # transitions of the ch03 protocol (every free fixation a target).
    variant = ROOT / "data/mit1003/MIT1003_initial_fix_consistent"
    if not (variant / "fixations.hdf5").exists():
        raise SystemExit(
            f"{variant} not found — the saccade panels need the initial-fixation "
            "dataset variant. Build it once with "
            "`.venv/bin/python scripts/fetch_mit1003.py --with-initial` (Octave required)."
        )
    with h5py.File(variant / "fixations.hdf5", "r") as f:
        xs, ys = f["train_xs"][:], f["train_ys"][:]
    out = []
    for x, y in zip(xs, ys):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        if len(x) >= 2:
            out.append(np.hypot(np.diff(x), np.diff(y)) / PPD)
    return np.concatenate(out)


ALIAS_BAND_DEG = (3.0, 8.0)   # eccentricity annulus the ratio is measured over
ALIAS_CPD = 40.0              # the arm the ratio is quoted for


def aliasing_band_ratio(cpd: float = ALIAS_CPD,
                        band_deg: tuple[float, float] = ALIAS_BAND_DEG,
                        n_stim: int = 12) -> dict:
    """How much of the arms' input difference is artefact rather than information.

    DeepGaze III halves its input by discarding every second pixel with no
    prefilter. Blurring before that discard therefore does two things at once:
    it removes peripheral detail (intended) and it suppresses the aliasing the
    unprefiltered halving would have produced (not intended). Both land in the
    difference the backbone sees between the two arms, and the second is an
    artefact of DG3's own downsampling rather than a property of foveation.

    Three versions of the same image, all after halving — what the backbone
    actually receives:

    ``sharp``     nearest-neighbour halving of the original (the normal arm),
    ``foveated``  nearest-neighbour halving of the foveated original,
    ``reference`` *area* halving of the original — the correctly prefiltered
                  version, i.e. the sharp arm as it would be with no aliasing.

    Then, within the eccentricity annulus:

        artefact    = RMS(sharp - reference)     what the blur incidentally fixes
        information = RMS(reference - foveated)  what the blur genuinely removes
        ratio       = artefact / information

    A ratio near 1 means half the apparent "foveation difference" is artefact
    suppression. Over 12 stimuli it is 1.32.

    The antialiasing control this quantifies is out of the thesis's scope; the
    ratio is not quoted in the thesis. The function and the JSON it writes are
    the record of the measurement.

    Averaged over ``n_stim`` stimuli because a single image's high-frequency
    content is not representative.
    """
    lo, hi = band_deg
    fov = foveation(cpd)
    num, den, per_stim = [], [], []
    for idx in sample_indices(n_stim, len(stimuli()), seed=0):
        img = image_chw(idx)
        # Crop to even dimensions before halving: MIT1003 has odd-sized stimuli
        # (e.g. 1023 px) and MPS area-pooling requires the input to divide the
        # output exactly. One row/column costs nothing here.
        _, _, H0, W0 = img.shape
        img = img[:, :, : H0 - H0 % 2, : W0 - W0 % 2]
        _, _, H, W = img.shape
        fx = torch.tensor([W / 2.0])
        fy = torch.tensor([H / 2.0])
        foveated_full = fov(img, fx, fy)
        sharp_h, fov_h, ref_h = half_nn(img), half_nn(foveated_full), half_area(img)

        # Eccentricity in the HALVED grid: one halved pixel spans two originals,
        # so the effective ppd halves too.
        h, w = sharp_h.shape[-2:]
        yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32),
                                torch.arange(w, dtype=torch.float32), indexing="ij")
        ecc_deg = torch.hypot(xx.to(img.device) - w / 2,
                              yy.to(img.device) - h / 2) / (PPD / 2)
        band = ((ecc_deg >= lo) & (ecc_deg <= hi)).unsqueeze(0).unsqueeze(0)
        if band.sum() == 0:
            continue

        def rms(a, b):
            d = (a - b) ** 2 * band
            return float(torch.sqrt(d.sum() / (band.sum() * a.shape[1])))

        artefact, information = rms(sharp_h, ref_h), rms(ref_h, fov_h)
        num.append(artefact)
        den.append(information)
        per_stim.append({"stim": int(idx), "artefact_rms": artefact,
                         "information_rms": information,
                         "ratio": artefact / information if information else None})

    a, i = float(np.mean(num)), float(np.mean(den))
    return {
        "cpd": cpd, "band_deg": list(band_deg), "n_stim": len(per_stim),
        "artefact_rms_mean": a, "information_rms_mean": i,
        "ratio_of_means": a / i,
        "ratio_mean_per_stim": float(np.mean([p["ratio"] for p in per_stim])),
        "artefact_share_of_total": a / (a + i),
        "per_stim": per_stim,
        "definition": ("artefact = RMS(nn-halved sharp - area-halved sharp); "
                       "information = RMS(area-halved sharp - nn-halved foveated); "
                       "both over the eccentricity annulus, after DG3's halving"),
    }


def fig_aliasing() -> None:
    """The confound, on a real stimulus, with the band ratio measured.

    A peripheral crop is shown four ways. The pair that matters is the middle
    two: they are what the backbone actually receives in the two arms, and they
    differ by more than the information the blur removed. The ratio quantifying
    that is computed by :func:`aliasing_band_ratio` and written alongside.
    """
    stim = sorted((ROOT / "data/mit1003/MIT1003/stimuli").glob("*.jpeg"))
    arr = np.asarray(Image.open(stim[3]).convert("RGB"), dtype=np.float32)
    x = torch.from_numpy(arr.transpose(2, 0, 1))[None]
    _, _, H, W = x.shape

    # Blur matching cpd 40 at ~15 deg eccentricity: cutoff 40*2.3/(15+2.3)
    # = 5.3 cyc/deg = 0.152 cyc/px, so sigma = 0.1874 / 0.152.
    sigma = 0.1874 / (40.0 * E2_DEG / (15.0 + E2_DEG) / PPD)
    xf = blur_fft(x, sigma)

    # Pick the 128x128 crop with the most high-frequency energy — aliasing is
    # invisible on smooth content, so choosing the busiest patch is the honest
    # way to show what the operation does rather than a lucky one.
    lap = (x - blur_fft(x, 1.0)).abs().mean(1)[0]
    S = 128
    by, bx = _busiest_patch(lap, S, stride=64)

    def crop(t, s):
        return (t[0, :, by // s:by // s + S // s, bx // s:bx // s + S // s]
                .permute(1, 2, 0).clamp(0, 255).numpy().astype(np.uint8))

    # Only the halved versions are shown: they are what the backbone actually
    # receives, and the full-resolution original would triple the figure's file
    # size for context the caption can carry.
    panels = [
        (crop(half_nn(x), 2), "SHARP arm\nwhat the backbone gets", "#c0392b"),
        (crop(half_nn(xf), 2), "FOVEATED arm\nwhat the backbone gets", "#27ae60"),
        (crop(half_area(x), 2), "sharp, halved correctly\n(the missing prefilter)", "0.4"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.0), dpi=150)
    for ax, (im, title, c) in zip(axes, panels):
        ax.imshow(im, interpolation="nearest")
        ax.set_title(title, fontsize=10, color=c,
                     fontweight="bold" if c != "0.4" else "normal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(c)
            sp.set_linewidth(2.2 if c != "0.4" else 0.8)
    fig.suptitle(
        "A peripheral crop, ~15° from gaze. The first two panels are what the "
        "backbone receives in each arm.\nThey differ in peripheral detail "
        "(intended) and in speckle that is not in the scene (not intended) — "
        "compare panel 1 with panel 3.", fontsize=10.5, y=1.04)
    ratio = aliasing_band_ratio()
    fig.text(0.5, -0.045,
             f"Measured over {ratio['n_stim']} stimuli in the "
             f"{ALIAS_BAND_DEG[0]:g}\u2013{ALIAS_BAND_DEG[1]:g}\u00b0 annulus at cpd "
             f"{ALIAS_CPD:g}: artefact/information = "
             f"{ratio['ratio_of_means']:.2f}, i.e. "
             f"{100 * ratio['artefact_share_of_total']:.0f}% of the difference the "
             "backbone sees between the arms is aliasing the blur suppressed, not "
             "detail it removed.",
             ha="center", fontsize=9.5, color="0.25", wrap=True)
    fig.tight_layout()
    save(fig, "aliasing_demo.png")
    (OUT / "aliasing_ratio.json").write_text(json.dumps(ratio, indent=2) + "\n")
    print(f"  aliasing artefact/information ratio = {ratio['ratio_of_means']:.2f} "
          f"({100 * ratio['artefact_share_of_total']:.0f}% artefact)")


# The three arm cutoffs. One palette for them across the whole thesis: the same three colours
# make_results_figures.CPD_COLOUR uses, so a cutoff does not change colour
# between chapters. cpd 20 is not the Okabe-Ito blue: the sharp-radius figures
# draw its disc on stimulus 91's sky, where that blue vanished.
CPD_COLOUR = {40.0: "#E69F00", 20.0: "#C2185B", 10.0: "#009E73"}


def fig_strength() -> None:
    """The sharp radius: where the curve crosses, and how big that disc is.

    Two panels because the chapter derives the radius twice over — once in closed
    form from the crossing, once as a claim about the frame ("3 degrees inside a
    29-degree stimulus"). The crossing alone plots a radius as an x-coordinate,
    which is the part readers could not picture, so the right panel puts the same
    number back on an image at the scale it is actually applied.
    """
    e = np.linspace(0, 10, 600)
    # constrained_layout, not tight_layout: the right panel is an image with a
    # fixed aspect, which tight_layout cannot size around.
    fig, (ax, ax2) = plt.subplots(
        1, 2, **canvas((11.0, 4.1), page_frac=0.98), layout="constrained",
        gridspec_kw={"width_ratios": [1.0, 1.15]})

    for cpd, c in CPD_COLOUR.items():
        ax.plot(e, cpd * E2_DEG / (e + E2_DEG), color=c, lw=2.2,
                label=rf"$f_{{c0}} = {cpd:g}$ cyc/deg")
    ax.axhline(NYQ_DISPLAY, color="k", ls="--", lw=1.3)
    # Below the line, so the strip above it is free for the two radius labels.
    ax.text(9.8, NYQ_DISPLAY - 1.3, f"display Nyquist = {NYQ_DISPLAY:g}",
            ha="right", va="top", fontsize=9)
    for cpd, c in CPD_COLOUR.items():
        r = sharp_radius_deg(cpd)
        if r <= 0:
            continue
        # Only the widest disc is shaded; the e_sharp label and the dotted drop
        # line say what the shade is, so it carries no legend entry.
        if cpd == 40.0:
            ax.axvspan(0, r, color=c, alpha=0.09)
        ax.plot([r], [NYQ_DISPLAY], "o", color=c, ms=7, zorder=5)
        ax.plot([r, r], [0, NYQ_DISPLAY], color=c, ls=":", lw=1.4)
        # Just above the Nyquist line is the only strip of this panel both curves
        # leave empty, so it is where the two radii can be labelled without one
        # sitting on a curve or on the other. The symbol is spelled out once.
        label = rf"$e_\mathrm{{sharp}} = {r:.1f}$°" if cpd == 40.0 else f"{r:.1f}°"
        ax.text(r + 0.16, NYQ_DISPLAY + 1.3, label, ha="left", va="bottom",
                fontsize=9.5, color=c, zorder=6,
                path_effects=[pe.withStroke(linewidth=3.2, foreground="w")])
    ax.set_xlim(0, 10)
    # The tallest curve is f_c0 = 40 at the fovea, so the axis stops just above it.
    ax.set_ylim(0, 42)
    # Same wording as the right panel: both axes measure degrees away from gaze,
    # and one of them naming that "eccentricity" made the pair read as two
    # different quantities.
    ax.set_xlabel("degrees from the point of gaze")
    ax.set_ylabel("resolvable cutoff (cycles/degree)")
    ax.legend(frameon=False, fontsize=9, loc="upper right",
              bbox_to_anchor=(1.0, 1.0))

    # The right panel is the f_c0 = 40 arm's own input, so the outer circle is a
    # boundary the reader can check: inside it the frame is bit-identical to the
    # sharp original, outside it is not.
    img = image_chw(GP_STIM)
    _, _, H, W = img.shape
    cx, cy = W / 2.0, H / 2.0
    frame = foveation(40.0)(img, torch.tensor([cx]), torch.tensor([cy]))
    half_w, half_h = cx / PPD, cy / PPD
    ax2.imshow(frame[0].permute(1, 2, 0).clamp(0, 255).cpu().numpy().astype(np.uint8),
               extent=(-half_w, half_w, -half_h, half_h))
    for cpd in (40.0, 20.0):
        r = sharp_radius_deg(cpd)
        ax2.add_patch(Circle((0, 0), r, fill=False, ec=CPD_COLOUR[cpd], lw=2.2,
                             zorder=5))
    # Smaller than the cpd-20 disc (4.3 pt across) and behind it, so the marker
    # cannot hide the disc.
    ax2.plot([0], [0], "o", color="w", mec="k", ms=2.5, zorder=3)
    r40, r20 = sharp_radius_deg(40.0), sharp_radius_deg(20.0)
    # Just the radii: the colours name the cutoffs (legend on the left). Both
    # labels sit in the empty sky above the aircraft, each with a leader down
    # to its own circle.
    ax2.annotate(f"{r40:.1f}°", (-r40 * 0.5, r40 * 0.87),
                 textcoords="offset points", xytext=(-26, 32), ha="center",
                 fontsize=10, color=CPD_COLOUR[40.0],
                 path_effects=[pe.withStroke(linewidth=2.5, foreground="w")],
                 arrowprops=dict(arrowstyle="-", color=CPD_COLOUR[40.0], lw=1.2))
    # The cpd-20 disc is 10 px across on a 1024 px frame. Drawn without a leader
    # it reads as a smudge, and its being nearly invisible is the point.
    ax2.annotate(f"{r20:.1f}°", (r20 * 0.5, r20 * 0.87),
                 textcoords="offset points", xytext=(30, 40), ha="center",
                 fontsize=10, color=CPD_COLOUR[20.0],
                 path_effects=[pe.withStroke(linewidth=2.5, foreground="w")],
                 arrowprops=dict(arrowstyle="-", color=CPD_COLOUR[20.0], lw=1.2))
    ax2.set_xlabel("degrees from the point of gaze")
    ax2.tick_params(labelsize=8.5)
    # Ticks out to the frame edge rather than a span quoted in the title: the
    # reader measures the stimulus off the axis instead of being told its size.
    ax2.set_xticks([-14, -7, 0, 7, 14])
    ax2.set_yticks([-10, 0, 10])
    save(fig, "foveation_strength.png")


def fig_saccade_coverage() -> None:
    """How much of the eye's own movement the sharp disc covers."""
    amps = saccade_amplitudes_deg()
    # The axis ends at the longest saccade there is, rounded up, so no saccade
    # is binned away and the axis reaches exactly as far as the data.
    xmax = float(np.ceil(amps.max()))
    fig, ax = plt.subplots(**canvas((9.0, 2.9), page_frac=0.94))
    ax.hist(amps, bins=np.arange(0, xmax + 0.25, 0.25), color="0.62", edgecolor="none")
    top = ax.get_ylim()[1]
    for cpd, c in CPD_COLOUR.items():
        r = sharp_radius_deg(cpd)
        if r <= 0:
            continue
        ax.axvline(r, color=c, lw=2.2, zorder=3)
        # Both radii sit under 3 degrees, so stacked horizontal labels to the
        # right of the peak clear each other; rotated in-place labels would
        # overlap at print size.
        y = 0.90 * top if cpd == 40.0 else 0.66 * top
        ax.annotate(rf"$f_{{c0}} = {cpd:g}$:  {r:.1f}°, "
                    f"{100 * (amps < r).mean():.1f}% of targets inside",
                    (r, y), textcoords="offset points", xytext=(46, 0),
                    va="center", fontsize=9.5, color=c,
                    arrowprops=dict(arrowstyle="-", color=c, lw=1.0))
    med = float(np.median(amps))
    ax.axvline(med, color="0.3", ls=":", lw=1.4)
    ax.annotate(f"median {med:.1f}°", (med, 0.30 * top),
                textcoords="offset points", xytext=(8, 0), fontsize=9.5,
                color="0.3", va="center")
    ax.set_xlim(0, xmax)
    ax.set_xticks([0, 7, 14, 21, xmax])
    ax.set_xlabel("saccade amplitude (degrees from the point of gaze)")
    ax.set_ylabel("saccades")
    save(fig, "saccade_coverage.png")


def fig_dose() -> None:
    """The measured effect for both foveation arms, next to the disc geometry."""
    amps = saccade_amplitudes_deg()
    dose = _dose_rows()
    cpds = np.array([d[0] for d in dose])
    fov = np.array([d[1] for d in dose])
    cen = np.array([d[2] for d in dose])
    cover = np.array([100 * (amps < sharp_radius_deg(c)).mean() for c in cpds])

    fig, ax = plt.subplots(**canvas((8.2, 4.8), page_frac=0.9))
    ax.axhline(0, color="0.7", lw=1)
    ax.plot(cpds, fov, "-o", color="#2980b9", lw=2.2, ms=9, zorder=3,
            label="gaze-contingent")
    ax.plot(cpds, cen, "--s", color="#E69F00", lw=2.0, ms=8, zorder=3,
            label="fixed-centre")
    for c, d in zip(cpds, fov):
        # Offset left of the marker at the last point: directly above it the
        # label sits on the descending line, which overdraws the minus sign.
        offset = (-30, 4) if c == cpds.min() else (0, 10)
        ax.annotate(f"{d:+.4f}", (c, d), textcoords="offset points",
                    xytext=offset, ha="center", fontsize=9, color="#1f5f8b")
    for c, d in zip(cpds, cen):
        # Same reason as above, and below the last point the label would also
        # land on the coverage bar's percentage.
        offset = (-34, -6) if c == cpds.min() else (0, -18)
        ax.annotate(f"{d:+.4f}", (c, d), textcoords="offset points",
                    xytext=offset, ha="center", fontsize=9, color="#9a6a00")
    ax.set_xlabel("foveal cutoff (cycles/degree) — lower means stronger foveation")
    ax.set_ylabel("ΔIG vs sharp input (bits/fixation)")
    ax.invert_xaxis()
    ax.set_ylim(-0.058, 0.014)
    ax.set_title("ΔIG against foveal cutoff, with sharp-disc coverage "
                 "(test split, 10 folds per arm)", fontsize=11)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    ax2 = ax.twinx()
    ax2.bar(cpds, cover, width=3.2, color="0.85", zorder=0)
    ax2.set_ylabel("% of saccade targets inside the sharp disc", color="0.45")
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis="y", colors="0.45")
    ax2.set_zorder(0)
    ax.set_zorder(1)
    ax.patch.set_visible(False)
    for c, v in zip(cpds, cover):
        # One decimal everywhere: at .0f the cpd-40 bar rounds to a whole
        # percent and the cpd-20 bar to "1%", against the values prose quotes.
        ax2.text(c, v + 2, f"{v:.1f}%",
                 ha="center", fontsize=9, color="0.45")
    fig.tight_layout()
    save(fig, "dose_response.png", out_dir=DOSE_TREE / "figs")


# Okabe-Ito, fixed order, validated for CVD separation against a light surface.
STRAT_ARMS = [(40, "#E69F00"), (20, "#C2185B"), (10, "#009E73")]
# The stratified artefacts of the primary run.
STRAT = ROOT / "results" / "foveation_mit1003_initial" / "stratified"


def strat_path(cpd: int) -> Path:
    """The reporting-epoch stratified artefact for one arm (validation split)."""
    return STRAT / f"cpd{cpd}_val" / "stratified.json"


def stratified(cpd: int) -> dict:
    """Pooled stratification for one arm, or a clear error if it was not run."""
    p = strat_path(cpd)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing — run scripts/foveation_stratified.py for cpd {cpd} first"
        )
    return json.loads(p.read_text())["pooled"]


def fold_clustered(cpd: int, section: str, key: str) -> dict[float, tuple[float, float]]:
    """Per-bin mean and SE with *fold* as the unit of independence.

    The per-fixation SE the analysis reports treats 104,171 fixations as
    independent. They are not — fixations nest inside scanpaths, subjects and
    images — so that SE is anti-conservative. Taking the spread across folds
    instead absorbs every within-fold correlation at once. It costs power (10
    observations per bin) and is the conservative choice, so it is what the
    figures draw. Point estimates are unchanged by this; only the bars are.

    This is why the figures mark fewer bins as resolved than the per-bin ``t`` in
    ``stratified.json``: at cpd 40 the 400-500 px bin is t = +4.9 per fixation and
    t = +2.7 across folds. Quote the fold-clustered value.
    """
    folds = json.loads(strat_path(cpd).read_text())["per_fold"]
    per_bin: dict[float, list[float]] = {}
    for f in folds:
        for r in f[section]:
            if r["d_LL"]["n"] >= 2:
                per_bin.setdefault(r[key], []).append(r["d_LL"]["mean"])
    out = {}
    for k, vals in per_bin.items():
        a = np.asarray(vals, dtype=float)
        if len(a) < 3:
            out[k] = (float(a.mean()), float("nan"))
        else:
            out[k] = (float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a))))
    return out


def fig_stratified_amplitude() -> None:
    """Does the sharp disc locate the effect? Fold-clustered, two standard errors.

    At cutoff 40 the profile is flat on both sides of the 103.5 px radius, and the
    positive bins sit among the long saccades beyond it. That is not the disc
    account, which is a claim about the two regions differing. Pooling each fold's
    bins into the two regions and pairing across folds gives inside +0.0019,
    outside +0.0050 and a difference of +0.0031 +- 0.0094 -- they do not differ.
    That pooled contrast is the appendix's, in one sentence; do not read this
    figure bin by bin.

    At cutoff 10, where the disc has no radius at all, the cost deepens with
    amplitude. Marker area gives the bin counts, because the shortest bins are thin
    enough that a reader is entitled to ask.
    """
    fig, ax = plt.subplots(**canvas((9.5, 5.2)))
    for cpd, col in STRAT_ARMS:
        p = stratified(cpd)
        rows = p["by_saccade_amplitude"]
        clust = fold_clustered(cpd, "by_saccade_amplitude", "lo_px")
        mid = np.array([(r["lo_px"] + (r["hi_px"] or 700)) / 2 for r in rows])
        mean = np.array([clust[r["lo_px"]][0] for r in rows])
        se = np.array([clust[r["lo_px"]][1] for r in rows])
        radius = p["sharp_radius_px"]
        # Two dashed lines against three curves is a question the figure should
        # answer itself: at cutoff 10 nothing is left untouched, not even the
        # point of gaze.
        ax.errorbar(mid, mean, yerr=2 * se, lw=2, color=col, capsize=3, zorder=3,
                    label=f"gaze-contingent @ {cpd}"
                          + ("" if radius > 0 else " (no sharp disc)"))
        # Marker area tracks how many fixations the bin holds. The bins are far
        # from equal -- 580 in the shortest, 15,337 around 70-103 px -- and on a
        # log axis the thin ones take as much width as the full ones, so without
        # this the sparsest points look as solid as the rest.
        n = np.array([r["d_LL"]["n"] for r in rows], dtype=float)
        ax.scatter(mid, mean, s=18 + 170 * np.sqrt(n / n.max()), color=col,
                   zorder=4, edgecolor="white", linewidth=0.6)
        if radius > 0:
            ax.axvline(radius, color=col, ls="--", lw=1.6, alpha=0.8, zorder=1)
            # Pinned to the top of the axes rather than to a data value, so the
            # label cannot land on a curve when the y-range shifts. The arrow
            # carries the side: a saccade shorter than the radius landed inside
            # the disc that cutoff leaves sharp, and a line on its own says
            # nothing about which of its two sides that is.
            ax.annotate(f"$\\leftarrow$ sharp disc, {cpd:g} cyc/deg",
                        xy=(radius, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(4, -6), textcoords="offset points", color=col,
                        fontsize=9, ha="left", va="top", fontweight="bold")
    ax.axhline(0, color="#444444", lw=1.0, zorder=2)
    # Headroom so the two dashed-line labels clear the tallest error bar.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.07 * (hi - lo))
    ax.set_xscale("log")
    # Explicit ticks: a log axis labels decades by default, which on a 12-600 px
    # range leaves two numbers on the axis and no way to read a bin off it.
    ticks = [12, 25, 50, 100, 200, 400, 600]
    ax.set_xlim(4.6, 800)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks[:-1]] + ["500+"])
    ax.tick_params(axis="x", which="minor", length=0)
    ax.set_xlabel("saccade amplitude (px, log scale)")
    # The same amplitudes in degrees along the top. Every quantity the figure is
    # read against -- the sharp radii, e_2, the cutoffs themselves -- is set in
    # degrees, and a pixel axis alone leaves the reader dividing by ppd.
    deg = ax.secondary_xaxis("top", functions=(lambda x: x / PPD, lambda d: d * PPD))
    deg.set_xticks([0.25, 0.5, 1, 2, 5, 10, 20])
    deg.set_xticklabels(["0.25", "0.5", "1", "2", "5", "10", "20"])
    deg.minorticks_off()
    deg.set_xlabel(f"saccade amplitude (degrees of visual angle, ppd = {PPD:g})")
    ax.set_ylabel(r"$\Delta$ IG (bits/fixation)" "\n"
                  r"gaze-contingent $-$ sharp control")
    # The y-label says which arm is subtracted from which; the sign needs no
    # further gloss.
    # No title: the caption names the quantity, the split and the bars, and the
    # axis labels carry the rest.
    ax.grid(alpha=0.18, lw=0.6)
    h, lab = ax.get_legend_handles_labels()
    h += [plt.Line2D([], [], marker="o", ls="", ms=5, color="#888888"),
          plt.Line2D([], [], marker="o", ls="", ms=11, color="#888888")]
    # The bin edges are the same for all three arms, so the last arm's counts are
    # every arm's. Numbers rather than "fewest" and "most", so the caption does
    # not have to carry the range.
    lab += [f"{n.min():,.0f} fixations", f"{n.max():,.0f} fixations"]
    # Opaque: the cutoff-40 dashed line runs behind the box, and at the default
    # alpha it shows through as a stray column of dashes inside the legend.
    ax.legend(h, lab, fontsize=9, loc="lower left", ncol=2, framealpha=1.0)
    fig.tight_layout()
    save(fig, "stratified_amplitude.png", out_dir=DOSE_TREE / "figs")


def fig_stratified_fixation_index() -> None:
    """The cost was predicted to grow along the scanpath. Where it moves, it shrinks.

    At cpd 10 the cost is deepest over the first two fixations and a fifth of that
    by the tail; cpd 40 and cpd 20 are flat. Reads ``index_profile.json`` rather
    than ``stratified.json`` so it can put one bin on each fixation out to the
    ninth: the cluster run pooled 4-5 and 6-8 to keep its per-fold counts up,
    which the re-aggregation from the per-fixation dumps does not need.
    """
    prof = json.loads((STRAT / "index_profile.json").read_text())["arms"]
    fig, ax = plt.subplots(**canvas((7.8, 4.6), page_frac=0.85))
    for cpd, col in STRAT_ARMS:
        rows = prof[f"foveated@{cpd}"]["bins"]
        x = np.arange(len(rows))
        mean = np.array([r["d_ll_bits"]["mean"] for r in rows])
        se = np.array([r["d_ll_bits"]["se"] for r in rows])
        ax.errorbar(x, mean, yerr=2 * se, lw=2, color=col, capsize=3, marker="o", ms=6,
                    zorder=3, label=f"gaze-contingent @ {cpd}")
    # Bin counts under the tick, so "how many fixations is that point?" is
    # answered on the axis rather than in a second panel or a lookup.
    labels = [f"{r['label']}\n{r['n_fixations'] / 1000:.1f}k"
              for r in prof["foveated@10"]["bins"]]
    ax.axhline(0, color="#444444", lw=1.0, zorder=2)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("fixation index in scanpath, and the fixations in each bin")
    ax.set_ylabel(r"$\Delta$ IG (bits/fixation)" "\n"
                  r"gaze-contingent $-$ sharp control")
    # No title: the caption names the quantity, the split and the bars, and the
    # axis labels carry the rest.
    ax.grid(alpha=0.18, lw=0.6)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    save(fig, "stratified_fixation_index.png", out_dir=DOSE_TREE / "figs")


def fig_gp_eccentricity() -> None:
    """The one array the transform is driven by.

    Nothing about the scene is in it — only the geometry. Cutoff, level and
    blend weight are each a pure function of this map, so the blur at a pixel is
    fixed once you know its value here.
    """
    img = image_chw(GP_STIM)
    _, _, H, W = img.shape
    fov = foveation(ARM_CPD)
    fx = torch.tensor([W / 2.0])
    fy = torch.tensor([H / 2.0])
    e_deg = (fov._eccentricity_px(H, W, fx, fy, img.device)[0] / PPD).cpu().numpy()

    fig, (ax_img, ax) = plt.subplots(1, 2, figsize=(13.0, 4.8), dpi=150)

    show(ax_img, img)
    ax_img.plot([W / 2], [H / 2], "o", mfc="w", mec="k", ms=7)
    ax_img.set_title(f"MIT1003 stimulus {GP_STIM}, fixation at centre", fontsize=11)

    im = ax.imshow(e_deg, cmap="viridis")
    cs = ax.contour(e_deg, levels=[2, 5, 10, 15], colors="w", linewidths=0.9)
    ax.clabel(cs, fmt="%g°", fontsize=9)
    r_deg = sharp_radius_deg(ARM_CPD)
    ax.add_patch(plt.Circle((W / 2, H / 2), r_deg * PPD, fill=False,
                            color="#d62728", lw=1.8, ls="--"))
    ax.annotate(f"sharp disc at {ARM_CPD:g} cyc/deg\n{r_deg:.2f}° = {r_deg * PPD:.0f} px",
                (W / 2, H / 2 - r_deg * PPD), color="#d62728", fontsize=9,
                ha="center", va="bottom",
                textcoords="offset points", xytext=(0, 4),
                path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("per-pixel eccentricity map", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, label="degrees from the fixation")

    fig.suptitle("The map behind every later quantity — "
                 "$e_{\\mathrm{px}} \\to e_{\\mathrm{deg}} \\to f_c(e) \\to L(e) \\to$ "
                 "blend weights", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "gp_eccentricity.png")


def fig_gp_pyramid() -> None:
    """The algorithm in one picture: stack, per-pixel level, blend.

    Drawn at the two shallow arm cutoffs, 40 over 20. The stack itself is
    cutoff-independent; only the levels either cutoff reaches are shown, which
    is the point — the deep levels exist for the coarse probes.
    """
    img = image_chw(GP_STIM)
    _, _, H, W = img.shape
    fx = torch.tensor([W / 2.0])
    fy = torch.tensor([H / 2.0])

    rows = []
    for cpd in PYRAMID_CPDS:
        fov = foveation(cpd)
        stack = fov.blur_stack(img)
        lvl = fov.fractional_level(fov._eccentricity_px(H, W, fx, fy, img.device))
        blended = fov.foveate_shared_image(img, fx, fy, stack=stack)
        rows.append((fov, lvl[0].cpu().numpy(), blended, stack))
        PYRAMID_LEVEL_MAX[f"{cpd:g}"] = round(float(lvl.max()), 2)

    lvl_max = max(lvl.max() for _, lvl, _, _ in rows)
    n_show = int(np.ceil(lvl_max)) + 1  # levels either cutoff touches
    stack0 = rows[0][3]

    # The levels differ by well under one grey level per pixel on smooth content,
    # so four whole frames printed at a quarter of the text block are four copies
    # of the same photograph. The row shows the busiest patch instead, magnified.
    by, bx, side = _peripheral_patch(img)

    # Three rows: the pyramid levels the two cutoffs reach, then for each cutoff
    # the per-pixel level map beside the blended frame. The frame carries no
    # box and no column heading; the level map's heading and the row label say
    # what it is.
    fig = plt.figure(**canvas((12.6, 11.2)))
    gs = fig.add_gridspec(3, 2 * n_show, height_ratios=(1.0, 2.0, 2.0),
                          hspace=0.14, wspace=0.08)
    for i in range(n_show):
        ax = fig.add_subplot(gs[0, 2 * i:2 * i + 2])
        show(ax, stack0[i][:, :, by:by + side, bx:bx + side])
        ax.set_title(f"pyramid level {i}", fontsize=10)

    axs_l, im = [], None
    for r, (cpd, (fov, lvl, blended, _)) in enumerate(zip(PYRAMID_CPDS, rows),
                                                     start=1):
        ax_l = fig.add_subplot(gs[r, 0:n_show])
        im = ax_l.imshow(lvl, cmap="viridis", vmin=0.0, vmax=float(lvl_max))
        ax_l.set_xticks([])
        ax_l.set_yticks([])
        ax_l.set_ylabel(f"$f_{{c0}}$ = {cpd:g} cyc/deg", fontsize=12)
        axs_l.append(ax_l)

        ax_b = fig.add_subplot(gs[r, n_show:2 * n_show])
        show(ax_b, blended)
        ax_b.plot([W / 2], [H / 2], "o", mfc="w", mec="k", ms=6)
        if r == 1:
            ax_l.set_title("per-pixel level  $L(e)$", fontsize=11)

    fig.colorbar(im, ax=axs_l, orientation="horizontal", fraction=0.04,
                 pad=0.03, label="pyramid level")
    fig.suptitle(f"MIT1003 stimulus {GP_STIM}, fixation at centre", fontsize=13)
    save(fig, "gp_pyramid.png")


def fig_gp_weights() -> None:
    """The blend written out, at the arm the primary comparison trained on.

    ``gp_pyramid`` shows the stack and the level map; this shows the weights
    that turn one into the other. Only the levels that carry weight somewhere in
    the frame are drawn — at 40 cpd the falloff never reaches the deep levels,
    which is what a faithful strength looks like.
    """
    img = image_chw(GP_STIM)
    _, _, H, W = img.shape
    fov = foveation(ARM_CPD)
    stack = fov.blur_stack(img)
    fx = torch.tensor([W / 2.0])
    fy = torch.tensor([H / 2.0])
    e_px = fov._eccentricity_px(H, W, fx, fy, img.device)
    lvl = fov.fractional_level(e_px)[0].cpu().numpy()
    blended = fov.foveate_shared_image(img, fx, fy, stack=stack)

    n_used = min(fov.n_levels, int(np.ceil(lvl.max())) + 1)
    ncol = n_used + 1
    fig, axes = plt.subplots(2, ncol, figsize=(3.9 * ncol, 6.6), dpi=150)

    for k in range(n_used):
        show(axes[0, k], stack[k])
        axes[0, k].set_title(f"level {k}  ($\\sigma$ = {fov.level_sigmas[k]:.2f} px)",
                             fontsize=11)
        w = np.clip(1.0 - np.abs(lvl - k), 0.0, None)
        im = axes[1, k].imshow(w, cmap="viridis", vmin=0, vmax=1)
        axes[1, k].set_title(f"weight  $w_{k}(e)$", fontsize=11)
        axes[1, k].set_xticks([])
        axes[1, k].set_yticks([])
    fig.colorbar(im, ax=axes[1, n_used - 1], fraction=0.046,
                 label="contribution 0→1")

    show(axes[0, -1], blended)
    axes[0, -1].plot([W / 2], [H / 2], "o", mfc="w", mec="k", ms=6)
    axes[0, -1].set_title(r"blend = $\sum_k w_k \cdot$ level$_k$", fontsize=11)
    im_lvl = axes[1, -1].imshow(lvl, cmap="magma")
    axes[1, -1].set_title("fractional level  $L(e)$", fontsize=11)
    axes[1, -1].set_xticks([])
    axes[1, -1].set_yticks([])
    fig.colorbar(im_lvl, ax=axes[1, -1], fraction=0.046)

    fig.suptitle(
        f"The blend, made explicit — {stim_label(stimuli(), GP_STIM)} at "
        f"{ARM_CPD:g} cpd, ppd {PPD:g}. Each pixel is a weighted mix of the two "
        "levels bracketing its $L(e)$; the weight maps sum to 1 everywhere.",
        fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "gp_weights.png")


def fig_gp_stepvsinterp() -> None:
    """Nearest-level selection bands; the interpolated blend does not."""
    img = image_chw(GP_STIM)
    _, _, H, W = img.shape
    fov = foveation(STEP_CPD)
    stack = fov.blur_stack(img)
    fx = torch.tensor([W / 2.0])
    fy = torch.tensor([H / 2.0])
    e_px = fov._eccentricity_px(H, W, fx, fy, img.device)
    level = fov.fractional_level(e_px)

    # Nearest-level selection: gather the single closest pyramid level per pixel.
    idx = level.round().long().clamp(0, fov.n_levels - 1)          # (1, H, W)
    gather = idx[:, None].expand(-1, 3, -1, -1)[None]              # (1, 1, 3, H, W)
    stepped = torch.gather(stack, 0, gather)[0]

    fig, axes = plt.subplots(1, 3, figsize=(19.0, 5.4), dpi=150)
    for ax, t, title in (
        (axes[0], stepped, "Nearest level (round $L$)\n→ uniform bands, visible rings"),
        (axes[1], fov.foveate_shared_image(img, fx, fy, stack=stack),
         "Interpolated (the method used)\n→ blur varies continuously"),
    ):
        show(ax, t)
        ax.set_title(title, fontsize=12)
        ax.plot([W / 2], [H / 2], "o", mfc="w", mec="k", ms=6)

    r = np.linspace(0, float(np.hypot(H / 2, W / 2)), 700)
    lvl = fov.fractional_level(torch.from_numpy(r).float()).numpy()
    axes[2].plot(r, level_sigma(fov, np.round(lvl)), color="#b8860b", lw=2,
                 label="nearest level (staircase)")
    axes[2].plot(r, level_sigma(fov, lvl), color="#1f77b4", lw=2,
                 label="interpolated (used)")
    axes[2].set_xlabel(f"eccentricity  (pixels,  ppd = {PPD:g})")
    axes[2].set_ylabel(r"effective blur radius  $\sigma$  (px)")
    axes[2].set_title("Blur vs eccentricity\ncontinuous, not step-like", fontsize=12)
    axes[2].grid(alpha=0.3)
    axes[2].legend(frameon=True, fontsize=10)

    fig.suptitle(
        f"Why the blend is continuous — {stim_label(stimuli(), GP_STIM)} at "
        f"{STEP_CPD:g} cpd. The pyramid levels are discrete, but each pixel is a "
        "weighted blend of its\ntwo bracketing levels, so the applied blur rises "
        "smoothly with eccentricity. Rounding to the nearest level (left) is what "
        "would produce rings.", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "gp_stepvsinterp.png")


CROP_FRAC = 0.34         # patch side as a fraction of the frame's short edge
CROP_BAND = (0.45, 0.8)  # where in the eccentricity range the patch is taken


def _peripheral_patch(img: torch.Tensor) -> tuple[int, int, int]:
    """``(top, left, side)`` of the busiest square patch in the mid-periphery.

    Blur is invisible on smooth content, so the patch is chosen by high-frequency
    energy rather than by position — the same rationale as ``fig_aliasing``.
    Restricting it to an eccentricity band keeps it out of the sharp fovea and
    off the frame edge.

    The side scales with the frame so the patch stays a recognisable fraction of
    the picture. A patch too small to place at a glance defeats the figure: the
    reader has to be able to see that the magnified crop and the marked square
    are the same thing.
    """
    _, _, H, W = img.shape
    side = int(CROP_FRAC * min(H, W))
    detail = (img - foveation(50.0).blur_stack(img)[2]).abs().mean(1)[0]
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                            torch.arange(W, dtype=torch.float32), indexing="ij")
    ecc = torch.hypot(xx.to(img.device) - W / 2, yy.to(img.device) - H / 2)
    lo, hi = (CROP_BAND[0] * ecc.max(), CROP_BAND[1] * ecc.max())
    by, bx = _busiest_patch(
        detail, side, stride=16,
        accept=lambda y, x: lo <= float(ecc[y + side // 2, x + side // 2]) <= hi)
    return by, bx, side


def fig_gp_strength_grid() -> None:
    """Strength across content, with the fovea pinned to the centre.

    Shown as magnified peripheral patches rather than whole frames. That is not
    a presentation preference: averaged over the grid stimuli, the mean
    absolute difference from the sharp original is 1.5/255 at cpd 40 and
    4.8/255 at cpd 10 (recomputed on every run and written to gp_figures.json),
    so at full-frame print size the four columns are indistinguishable and the
    figure cannot show what its caption claims. The mildness is itself the
    finding (see the module docstring of ``foveate_input``) — but a figure that
    means to show a progression has to show one, so each row carries a context
    thumbnail with the patch marked, then that patch enlarged.

    Holding the fovea still isolates strength from gaze-contingency: the trained
    model re-centres on its own fixation every step (ch03 §Gaze-contingent
    application), and fixing it here is a presentation choice, not the protocol.
    Its position is per-row (``GRID_FOVEA``) rather than always the image centre,
    so each row can put the magnified patch at a useful eccentricity.
    """
    cpds = [40.0, 20.0, 10.0]
    mean_abs: dict[float, list[float]] = {c: [] for c in cpds}
    ncol = len(cpds) + 2                       # context + sharp + cutoffs
    BOX = "#e34948"
    # Row height equals the width of one patch column, so the square patches
    # fill their axes instead of sitting in a tall box with padding either side.
    unit = 2.7
    # No suptitle and tight gutters: the caption says what the columns are, and
    # every point of margin taken back goes to the patches, which are the
    # content. The context column is a little wider than a patch so the whole
    # stimulus stays readable at its own aspect.
    fig, axes = plt.subplots(
        len(GRID_STIMS), ncol,
        **canvas((unit * (ncol + 0.9), unit * len(GRID_STIMS))),
        gridspec_kw={"width_ratios": [1.9] + [1.0] * (ncol - 1),
                     "wspace": 0.05, "hspace": 0.12})
    for row, stim in enumerate(GRID_STIMS):
        img = image_chw(stim)
        _, _, H, W = img.shape
        fov_fx, fov_fy = GRID_FOVEA.get(stim, (0.5, 0.5))
        cx, cy = fov_fx * W, fov_fy * H
        fx = torch.tensor([cx])
        fy = torch.tensor([cy])
        # The pyramid depends only on n_levels, not on foveal_cpd, so one stack
        # per image serves every strength — one stack per row, not one per cell.
        stack = foveation(cpds[0]).blur_stack(img)
        by, bx, side = _peripheral_patch(img)
        if stim in GRID_PATCH_CENTER:
            pfx, pfy = GRID_PATCH_CENTER[stim]
            bx = int(np.clip(pfx * W - side / 2, 0, W - side))
            by = int(np.clip(pfy * H - side / 2, 0, H - side))

        ax = axes[row, 0]
        show(ax, img)
        ax.add_patch(Rectangle((bx, by), side, side, fill=False,
                               edgecolor=BOX, lw=3.0))
        # The white dot is the fovea; the caption says so.
        ax.plot([cx], [cy], "o", mfc="w", mec="k", ms=9, mew=1.6)
        ax.set_ylabel(f"stim {stim}", fontsize=11, fontweight="bold")

        panels = [("sharp", img)]
        for cpd in cpds:
            fov_img = foveation(cpd).foveate_shared_image(img, fx, fy, stack=stack)
            mean_abs[cpd].append(float((fov_img - img).abs().mean()))
            panels.append((rf"$f_{{c0}} = {cpd:g}$", fov_img))
        # The box on the stimulus says which region the patches are; the
        # patches themselves carry no frame and no eccentricity label.
        for col, (title, t) in enumerate(panels, start=1):
            ax = axes[row, col]
            show(ax, t[:, :, by:by + side, bx:bx + side])
            ax.axis("off")
            if row == 0:
                ax.set_title(title, fontsize=11.5)

    GRID_MEAN_ABS_CHANGE.clear()
    GRID_MEAN_ABS_CHANGE.update(
        {f"{cpd:g}": round(float(np.mean(v)), 2) for cpd, v in mean_abs.items()})
    fig.tight_layout(pad=0.4)
    save(fig, "gp_strength_grid.png")


def fig_consensus_method() -> None:
    """How a consensus region is built, on one stimulus (appendix).

    A shows the raw disagreement between subjects; B the two consensus regions,
    thresholds of the count map. Human fixation data only — no model, no forward
    pass.
    """
    idx = GP_STIM
    img = image_rgb(stimuli(), idx)
    H, W = img.shape[:2]
    paths = all_human_scanpaths(fixations(), idx)
    # Same population the equation thresholds: a subject with no fixation on this
    # image is missing data, and counting them would raise ceil(0.90 N) past what
    # the subjects who did look can reach.
    paths = [p for p in paths if len(p[0]) > 0]
    n = len(paths)
    cmap = plt.get_cmap("tab20")
    colours = [cmap(i / max(n - 1, 1)) for i in range(n)]

    fig, axes = plt.subplots(1, 2, **canvas((10.0, 4.3)))
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)

    # A -- every subject's scanpath, one colour each.
    ax = axes[0]
    ax.imshow(img)
    for (xs, ys, _), col in zip(paths, colours):
        ax.plot(xs, ys, "-", color=col, lw=0.9, alpha=0.55, zorder=2)
        ax.plot(xs, ys, "o", color=col, ms=4.2, mec="white", mew=0.6, zorder=3)
    # "fifteen", not "15": the thesis spells the MIT1003 subject count out.
    spelled = {15: "fifteen"}.get(n, str(n))
    ax.set_title(f"A. {spelled} subjects, {sum(len(p[0]) for p in paths)} fixations",
                 fontsize=11)

    # B -- the two masks of eq:consensus-masks, thresholds of the count map.
    ax = axes[1]
    count = consensus_count(paths, H, W, CONSENSUS_RADIUS_PX)
    ax.imshow(img)
    for frac, col in ((0.75, "#7ec8e3"), (0.90, "#10214b")):
        ax.contour(count >= int(np.ceil(frac * n)), levels=[0.5],
                   colors=[col], linewidths=2.2, zorder=4)
    ax.set_title(r"B. consensus regions $M_{0.75}$ / $M_{0.90}$",
                 fontsize=11)

    fig.suptitle(f"MIT1003 stimulus {idx}", fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "consensus_method.png", out_dir=CONSENSUS_OUT)


FIGURES = {
    "consensus_method": fig_consensus_method,
    "gp_eccentricity": fig_gp_eccentricity,
    "gp_pyramid": fig_gp_pyramid,
    "gp_weights": fig_gp_weights,
    "gp_stepvsinterp": fig_gp_stepvsinterp,
    "gp_strength_grid": fig_gp_strength_grid,
    "aliasing_demo": fig_aliasing,
    "foveation_strength": fig_strength,
    "saccade_coverage": fig_saccade_coverage,
    "dose_response": fig_dose,
    "stratified_amplitude": fig_stratified_amplitude,
    "stratified_fixation_index": fig_stratified_fixation_index,
}

# Figures that do not live in OUT, by topic directory.
FIGURE_DIR = {"consensus_method": CONSENSUS_OUT, "dose_response": DOSE_TREE / "figs",
              "stratified_amplitude": DOSE_TREE / "figs",
              "stratified_fixation_index": DOSE_TREE / "figs"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="+", choices=sorted(FIGURES),
                    help="regenerate a subset (default: all)")
    args = ap.parse_args()

    names = args.only or list(FIGURES)
    OUT.mkdir(parents=True, exist_ok=True)
    for name in names:
        FIGURES[name]()
        print(f"wrote {FIGURE_DIR.get(name, OUT) / (name + '.png')}")

    # Sidecar: the gp_* figures embed choices (which stimuli, which cutoffs) that
    # the PNGs cannot carry. Same reproducibility contract as the demo figures.
    # Only rewritten when a gp_* figure was actually regenerated — otherwise
    # `--only dose_response` would overwrite a good record with a stale one.
    gp_names = sorted(n for n in names if n.startswith("gp_"))
    if gp_names:
        # The two measurement dicts are only computed when their figure runs; a
        # partial --only run must carry the previous run's values forward, not
        # drop them from the committed record.
        sidecar = OUT / "gp_figures.json"
        prev = json.loads(sidecar.read_text()) if sidecar.exists() else {}
        carried = {k: prev[k] for k in ("grid_mean_abs_change", "pyramid_level_max")
                   if k in prev}
        sidecar.write_text(json.dumps({
            "ppd": PPD,
            "e2_deg": E2_DEG,
            "gp_stim": GP_STIM,
            "gp_stim_label": stim_label(stimuli(), GP_STIM),
            "grid_stims": GRID_STIMS,
            "acuity_cpds": [40.0, 20.0, 10.0],
            "pyramid_cpds": list(PYRAMID_CPDS),
            "stepvsinterp_cpd": STEP_CPD,
            "arm_cpd": ARM_CPD,
            "foveation": "tez_deepgaze.foveate_input.Foveation",
            "device": str(device()),
            "regenerated": gp_names,
            **carried,
            **({"grid_mean_abs_change": GRID_MEAN_ABS_CHANGE}
               if GRID_MEAN_ABS_CHANGE else {}),
            **({"pyramid_level_max": PYRAMID_LEVEL_MAX}
               if PYRAMID_LEVEL_MAX else {}),
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
