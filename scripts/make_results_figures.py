"""The results figures and the thesis result tables.

Companion to ``make_protocol_figures.py``, which draws what the transform *is*.
This draws what the production run *measured*. Everything here reads a committed
JSON under ``results/foveation_mit1003_initial/`` — no model and no forward pass —
so it runs on a laptop in seconds and every figure is regenerable from the repo.
The one dataset read is ``training_curves``, which needs each stimulus's scored
fixation count to put epoch 0 on the same scale as the rest of the curve.

Results (ch04):

1. ``fold_paired.png``         all six arm-versus-control contrasts with their ten
                               fold-paired values drawn individually. Shows how
                               much of each interval is carried by how many folds.
2. ``gaze_contingency.png``    gaze-contingent minus fixed-centre against cutoff:
                               blur held identical, only gaze-tracking removed.
3. ``training_curves.png``     validation IG per epoch for all seven arms, from the
                               pretrained read-out at epoch 0, with the reporting
                               epoch and the rate decays marked.
4. ``disc_contrast.png``       the sharp-disc account tested as a contrast rather
                               than read off the per-bin plot. Carried in the
                               appendix, not the chapter.

Tables: ``results_tables.tex`` holds the two chapter-4 tables as booktabs LaTeX,
generated from the same JSONs the figures read so the thesis cannot drift from
the artefacts.

``per_image_dispersion.json`` correlates the per-image contrast against the
fixation entropy of ch04 4.2. It is the one place the chapter's two results
sections meet, and ch05 disc-per-image quotes the bound it writes.

``grouping_intervals.json`` re-derives the by-image and by-subject intervals ch04
compares the fold pairing against, so those two numbers come out of code rather
than out of prose.

    .venv/bin/python scripts/make_results_figures.py
    .venv/bin/python scripts/make_results_figures.py --only training_curves

``training_curves.png`` needs ``training_curves.json``, which is aggregated from
the 840 per-epoch ``metrics.json`` files in the cluster's checkpoint tree. That
tree is far too large to commit, so the aggregate is committed instead; see the
module docstring of ``scripts/collect_training_curves.py`` to rebuild it.
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tez_deepgaze.figstyle import canvas  # noqa: E402  (needs the path above)

ROOT = Path(__file__).resolve().parents[1]
# The tree the thesis reads; --results-root overrides it.
RES = ROOT / "results" / "foveation_mit1003_initial"
OUT = RES / "figs"

# Okabe-Ito, and the same cutoff-to-colour mapping make_protocol_figures.py uses,
# so a reader carries one colour key across every figure in the chapter.
# cpd 20 is not the Okabe-Ito blue: on stimulus 91's sky, where the sharp-radius
# figures draw its disc, that blue vanished, and one cutoff keeps one colour
# through the whole thesis.
CPD_COLOUR = {40: "#E69F00", 20: "#C2185B", 10: "#009E73"}
CONTROL = "#555555"
N_FOLDS = 10
# Chosen on validation and applied to all seven arms at once. The decays are the
# protocol's, not a tuned schedule.
#
# Read from the artefact rather than written down, so the figure and the tables
# cannot disagree about it.
def report_epoch() -> int:
    """The epoch the committed tables were scored at."""
    return int(load("test", "table.json")["epochs_used"][0].rsplit("_", 1)[-1])


LR_DECAYS = (5, 8, 11)

# table.json / per_image_ig.json name arms one way, the checkpoint tree and the
# mixed-effects artefacts another. One place to translate.
TABLE_ARMS = ["normal", "foveated@40", "foveated@20", "foveated@10",
              "center@40", "center@20", "center@10"]
CKPT_TAG = {"normal": "normal",
            "foveated@40": "fov_cpd40", "foveated@20": "fov_cpd20",
            "foveated@10": "fov_cpd10", "center@40": "fov_cpd40_center",
            "center@20": "fov_cpd20_center", "center@10": "fov_cpd10_center"}
LABEL = {"normal": "sharp control",
         "foveated@40": "gaze-contingent @ 40", "foveated@20": "gaze-contingent @ 20",
         "foveated@10": "gaze-contingent @ 10", "center@40": "fixed-centre @ 40",
         "center@20": "fixed-centre @ 20", "center@10": "fixed-centre @ 10"}
CONTRASTS = ["foveated@40", "center@40", "foveated@20", "center@20",
             "foveated@10", "center@10"]
# The read-out every arm starts from, scored on the same validation folds under
# the same protocol. It is the arms' common origin, so both training-curve panels
# start there rather than at epoch 1. The upper panel re-weights it (``epoch0_ig``);
# the lower one pairs arms within it, which is the form ch04 §4.1.2 quotes.
EPOCH0 = "pretrained_epoch0/val"


# Cached: the callers treat a parsed artefact as immutable and several of them
# ask for the same one repeatedly — fig_gaze_contingency alone wants the 1.1 MB
# test dump twelve times.
@functools.lru_cache(maxsize=None)
def load(*parts: str) -> dict:
    """A committed result artefact, or a message naming what has to run first."""
    p = RES.joinpath(*parts)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing — produced by the cluster evaluation "
                                "(scripts/slurm/foveation_sweep_eval.sbatch)")
    return json.loads(p.read_text())


def save(fig, name: str) -> None:
    fig.savefig(OUT / name, dpi="figure", bbox_inches="tight")
    plt.close(fig)


def per_image(split: str) -> dict[str, dict[int, tuple[int, float]]]:
    """``arm -> {stimulus: (fold, IG bits)}`` for one split.

    Every arm is scored on the same 1,003 stimuli in the same fold assignment, so
    keying by stimulus is what makes the difference paired.
    """
    j = load(split, "per_image_ig.json")
    out: dict[str, dict[int, tuple[int, float]]] = {}
    for arm, folds in j["arms"].items():
        d: dict[int, tuple[int, float]] = {}
        for f in folds:
            for s, ig in zip(f["stim"], f["IG_bits"]):
                d[s] = (f["fold"], ig)
        out[arm] = d
    return out


def paired(split: str, test_arm: str, ref_arm: str = "normal"):
    """Per-image difference and its fold label, aligned on stimulus.

    Reproduces ``table.json``'s ``fold_paired`` and ``image_paired`` blocks
    exactly, which is the check that this reader agrees with the evaluator.
    """
    p = per_image(split)
    ref, tst = p[ref_arm], p[test_arm]
    stims = sorted(ref)
    delta = np.array([tst[s][1] - ref[s][1] for s in stims])
    fold = np.array([ref[s][0] for s in stims])
    base = np.array([ref[s][1] for s in stims])
    return np.array(stims), delta, fold, base


def fold_stats(delta: np.ndarray, fold: np.ndarray) -> dict:
    """Fold-paired mean, SE and the two-standard-error interval around it.

    ``interval`` is mean ± 2 SE, the one convention ch03 stats-plan sets for the
    thesis.
    """
    fm = np.array([delta[fold == k].mean() for k in range(N_FOLDS)])
    se = float(fm.std(ddof=1) / np.sqrt(N_FOLDS))
    return {"per_fold": fm, "mean": float(fm.mean()), "se": se,
            "interval": (fm.mean() - 2 * se, fm.mean() + 2 * se),
            "negative": int((fm < 0).sum())}


# --------------------------------------------------------------------------- 1

def fig_fold_paired() -> None:
    """All six contrasts with their ten fold values drawn, not just summarised.

    A mean and an interval hide whether ten folds agree or two outliers carry the
    result, so each fold is drawn. One fold's point is that fold's foveated
    checkpoint minus that fold's sharp checkpoint, averaged over the images held
    out from both -- verified against foveation_sweep_table, which evaluates every
    (arm, fold) on that fold's own index list.

    The count of folds either side of zero is not printed: the reader can see the
    spread, and a count invites reading the figure as a sign test.

    The bar is two standard errors of the mean, the one convention ch03
    stats-plan sets for every plus-minus and every bar in the thesis.
    """
    fig, ax = plt.subplots(**canvas((10.6, 5.6)))
    rng = np.random.default_rng(0)  # jitter only, so overlapping folds stay readable
    # Data occupies [-0.062, +0.015]; the numeric column sits beyond it so labels
    # align with each other instead of tracking each interval's right end.
    lo, hi = -0.068, 0.020
    yticks, ylabels = [], []

    for i, arm in enumerate(CONTRASTS):
        y = len(CONTRASTS) - i
        cpd = int(arm.split("@")[1])
        colour = CPD_COLOUR[cpd]
        gaze = arm.startswith("foveated")
        _, delta, fold, _ = paired("test", arm)
        s = fold_stats(delta, fold)
        ax.plot([s["mean"] - 2 * s["se"], s["mean"] + 2 * s["se"]], [y, y], lw=2.6,
                color=colour, zorder=3, solid_capstyle="round")
        ax.plot(s["per_fold"], y + rng.uniform(-0.13, 0.13, N_FOLDS), "o", ms=5,
                mfc=colour if gaze else "white", mec=colour, mew=1.4, alpha=0.85, zorder=4)
        ax.plot([s["mean"]], [y], "D" if gaze else "s", ms=10, color=colour,
                mec="white", mew=1.2, zorder=5)
        yticks.append(y)
        ylabels.append(LABEL[arm])

    ax.axvline(0, color="#333333", lw=1.4, ls="--", zorder=2)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=10)
    ax.set_ylim(-0.62, len(CONTRASTS) + 0.75)
    # The legend carries what the bar is, so the label stays one line and fits.
    ax.set_xlabel(r"$\Delta$IG vs sharp control (bits/fixation)")
    # No title: the caption names the figure and the split.
    ax.grid(axis="x", alpha=0.18, lw=0.6)
    ax.set_xlim(lo, hi)
    # Every 0.02 rather than every 0.01: at the printed size the tighter ticks
    # ran their labels together.
    ax.set_xticks([-0.06, -0.04, -0.02, 0.0])
    ax.legend(handles=[
        plt.Line2D([], [], marker="D", ls="", ms=9, color=CONTROL, label="gaze-contingent (mean)"),
        plt.Line2D([], [], marker="s", ls="", ms=9, color=CONTROL, label="fixed-centre (mean)"),
        plt.Line2D([], [], marker="o", ls="", ms=6, mfc="white", mec=CONTROL,
                   label="individual fold"),
        plt.Line2D([], [], ls="-", lw=2.6, color=CONTROL,
                   label="±2 standard errors of the mean"),
    ], fontsize=9, loc="lower left", ncol=2, framealpha=0.95,
        bbox_to_anchor=(0.0, -0.005))
    fig.tight_layout()
    save(fig, "fold_paired.png")


# --------------------------------------------------------------------------- 2

def fig_gaze_contingency() -> None:
    """Blur held identical, gaze-tracking removed — and it costs more the worse it gets.

    This is the contrast that separates this work from pixel-level foveated
    networks that blur to a fixed centre. Both arms at a cutoff see the same
    amount of blur; only the fovea's placement differs.

    Cutoff on the horizontal axis and the effect on the vertical, the way a
    dose-response reads.
    """
    fig, ax = plt.subplots(**canvas((8.0, 5.0), page_frac=0.9))
    cpds = [40, 20, 10]
    # Evenly spaced columns rather than a numeric cutoff axis: three points do
    # not make a curve, and spacing them by value invites reading a slope.
    xs = list(range(len(cpds)))
    rng = np.random.default_rng(1)  # jitter only, so overlapping folds stay readable
    means = []
    for cpd, x in zip(cpds, xs):
        _, delta, fold, _ = paired("test", f"foveated@{cpd}", f"center@{cpd}")
        st = fold_stats(delta, fold)
        means.append(st["mean"])
        col = CPD_COLOUR[cpd]
        ax.plot(x + rng.uniform(-0.09, 0.09, N_FOLDS), st["per_fold"], "o", ms=5,
                mfc="white", mec=col, mew=1.3, alpha=0.8, zorder=3)
        ax.plot([x, x], [st["mean"] - 2 * st["se"], st["mean"] + 2 * st["se"]],
                lw=3.0, color=col, zorder=4, solid_capstyle="round")
        ax.plot([x], [st["mean"]], "D", ms=11, color=col, mec="white", mew=1.2, zorder=5)

    ax.plot(xs, means, "-", color="0.6", lw=1.4, zorder=2)
    ax.axhline(0, color="#333333", lw=1.4, ls="--", zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{c}\n{'(human)' if c == 40 else 'probe'}" for c in cpds],
                       fontsize=9.5)
    ax.set_xlim(-0.45, len(cpds) - 0.55)
    ax.set_xlabel("foveal cutoff (cyc/deg) — lower means stronger foveation")
    ax.set_ylabel(r"$\Delta$IG, gaze-contingent $-$ fixed-centre (bits/fix)")
    # No title and no in-plot note: the caption states the contrast, the split and
    # what the bars are, and repeating it here gave the reader the same sentence
    # twice on one page.
    ax.grid(axis="y", alpha=0.18, lw=0.6)
    fig.tight_layout()
    save(fig, "gaze_contingency.png")


# --------------------------------------------------------------------------- 3

@functools.lru_cache(maxsize=1)
def fixations_per_stimulus() -> dict[int, int]:
    """``stimulus -> scored fixations``, from the dataset variant the thesis uses.

    Index 0 of every scanpath is the forced central start, which conditions the
    first prediction and is never a target, so a scanpath of length k contributes
    k - 1. Summing these gives 104,171 over 14,916 scanpaths, the count every
    committed table records.
    """
    p = ROOT / "data/mit1003/MIT1003_initial_fix_consistent/fixations.hdf5"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing — build the variant with "
            "scripts/fetch_mit1003.py --with-initial (needs Octave or MATLAB)")
    import h5py
    with h5py.File(p, "r") as f:
        ns, xs = f["train_ns"][:], f["train_xs"][:]
    out: dict[int, int] = {}
    for n, x in zip(ns, xs):
        k = int(np.sum(~np.isnan(np.asarray(x))))
        if k >= 2:
            out[int(n)] = out.get(int(n), 0) + k - 1
    return out


def epoch0_ig() -> dict[str, np.ndarray]:
    """``arm -> per-fold validation IG`` for the read-out before any fine-tuning.

    Only the four arms scored in ``pretrained_epoch0/val`` are in it: the
    fixed-centre arms were never run un-adapted, so the figure draws nothing for
    them at epoch 0. Empty for a tree that has no such artefact, whose curves then
    start at epoch 1.

    Weighted by fixations, not by images, so that it is on the same scale as
    epochs 1-7. ``table.json`` averages a fold's per-image IG while the training
    loop pools over fixations, and on identical data those disagree by about
    0.003 bits (1.5740 against 1.5773 for the control at epoch 5) — the size of
    the arm differences the thesis reports, and it would have sat inside the
    epoch-0-to-1 rise reading as adaptation. Re-weighting closes it: this
    reproduces the training loop at epoch 5 to four decimals on every arm.
    """
    if not (RES / EPOCH0 / "per_image_ig.json").exists():
        return {}
    n_fix = fixations_per_stimulus()
    j = load(EPOCH0, "per_image_ig.json")["arms"]
    return {arm: np.array([float(np.average(f["IG_bits"],
                                            weights=[n_fix[int(s)] for s in f["stim"]]))
                           for f in sorted(j[arm], key=lambda f: f["fold"])])
            for arm in j}


def fig_training_curves() -> None:
    """Where the read-out starts and how fast it adapts: every arm is flat by epoch 5.

    Training stops at the first epoch where every arm has gained under 0.001 bits
    on each of the two preceding epochs (``script_utils.common_stop_epoch``),
    which is epoch 7. The reporting epoch is the highest point
    inside that run, epoch 5 in all seven arms, so no arm gets its own best epoch.

    Epoch 0 is the pretrained read-out, on the same folds and weighted to the same
    scale as the rest of the curve (``epoch0_ig``).
    """
    curves = load("training_curves.json")["arms"]
    n_epochs = max(int(e) for tag in curves for f in curves[tag] for e in curves[tag][f])
    epochs = np.arange(1, n_epochs + 1)

    def arm_matrix(tag: str) -> np.ndarray:
        """(folds, epochs) validation IG for one arm."""
        return np.array([[curves[tag][str(f)][str(e)]["val_ig"] for e in epochs]
                         for f in range(N_FOLDS)])

    fig, axes = plt.subplots(2, 1, **canvas((9.4, 8.0)), sharex=True,
                            gridspec_kw={"height_ratios": [1.25, 1], "hspace": 0.12})

    ax = axes[0]
    e0 = epoch0_ig()
    for arm in TABLE_ARMS:
        m = arm_matrix(CKPT_TAG[arm])
        mean, se = m.mean(0), m.std(0, ddof=1) / np.sqrt(N_FOLDS)
        col = CONTROL if arm == "normal" else CPD_COLOUR[int(arm.split("@")[1])]
        ls = "-" if arm.startswith(("normal", "foveated")) else "--"
        x = epochs
        if arm in e0:
            a0 = e0[arm]
            x = np.concatenate([[0], epochs])
            mean = np.concatenate([[a0.mean()], mean])
            se = np.concatenate([[a0.std(ddof=1) / np.sqrt(N_FOLDS)], se])
        ax.plot(x, mean, ls, color=col, lw=2.4 if arm == "normal" else 1.9,
                label=LABEL[arm], zorder=3)
        ax.fill_between(x, mean - 2 * se, mean + 2 * se, color=col, alpha=0.12, lw=0, zorder=2)
        if arm in e0:
            ax.plot([0], [mean[0]], "o", ms=6, color=col, mec="white", mew=1.1, zorder=5)
    # Headroom at the bottom for a legend that would otherwise sit on the cpd-10
    # arms, which are the lowest curves in the panel.
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo - 0.46 * (yhi - ylo), yhi)
    # Only the decays the seven-epoch run reaches. The schedule's milestones are
    # absolute at 5, 8 and 11, and drawing the two that never fire cost a third of
    # the axis width to show nothing happening. The one that does fire is at the
    # reporting epoch, so the red line below already marks it and a dotted line
    # underneath it would be invisible.
    fired = [e for e in LR_DECAYS if e <= n_epochs and e != report_epoch()]
    for e in fired:
        ax.axvline(e, color="0.75", lw=1.0, ls=":", zorder=1)
    # The rate in effect over each segment of the schedule (base 3e-4, divided by
    # ten at each decay), printed at the segment midpoints. The point at epoch e
    # is the validation after training epoch e, and epoch 5 already trains at the
    # decayed rate, so the base-rate segment ends at the epoch-4 point.
    for x, rate in ((2.5, r"lr $3{\times}10^{-4}$"), (6.0, r"$3{\times}10^{-5}$")):
        ax.text(x, 0.995, rate, transform=ax.get_xaxis_transform(), ha="center",
                va="top", fontsize=9, color="0.45",
                bbox=dict(fc="white", ec="none", pad=1.2))
    rep = report_epoch()
    ax.axvline(rep, color="#333333", lw=1.8, zorder=4)
    ax.text(rep, 0.92, " reporting epoch", transform=ax.get_xaxis_transform(),
            ha="left", va="top", fontsize=9, color="#333333", fontweight="bold")
    ax.set_ylabel("validation IG (bits/fixation)\nmean ± 2 SE over 10 folds")
    ax.grid(alpha=0.18, lw=0.6)
    ax.legend(fontsize=9, ncol=3, loc="lower center", framealpha=0.95)

    ax = axes[1]
    ref = arm_matrix("normal")
    for arm in CONTRASTS:
        m = arm_matrix(CKPT_TAG[arm]) - ref  # fold-paired at every epoch
        mean, se = m.mean(0), m.std(0, ddof=1) / np.sqrt(N_FOLDS)
        col = CPD_COLOUR[int(arm.split("@")[1])]
        ls = "-" if arm.startswith("foveated") else "--"
        x = epochs
        if arm in e0:
            s0 = fold_stats(*paired(EPOCH0, arm)[1:3])
            x = np.concatenate([[0], epochs])
            mean = np.concatenate([[s0["mean"]], mean])
            se = np.concatenate([[s0["se"]], se])
        ax.plot(x, mean, ls, color=col, lw=1.9, label=LABEL[arm], zorder=3)
        ax.fill_between(x, mean - 2 * se, mean + 2 * se, color=col, alpha=0.10, lw=0, zorder=2)
        if arm in e0:
            ax.plot([0], [mean[0]], "o", ms=6, color=col, mec="white", mew=1.1, zorder=5)
    for e in fired:
        ax.axvline(e, color="0.75", lw=1.0, ls=":", zorder=1)
    ax.axvline(report_epoch(), color="#333333", lw=1.8, zorder=4)
    ax.axhline(0, color="#333333", lw=1.0, zorder=2)
    ax.set_xlabel("epoch")
    if e0:
        ax.set_xticks(np.concatenate([[0], epochs]))
        ax.set_xticklabels(["0\npretrained\nread-out"] + [str(e) for e in epochs])
    else:
        ax.set_xticks(epochs)
    ax.set_ylabel(r"$\Delta$IG vs control" "\n" "(bits/fixation, fold-paired)")
    ax.grid(alpha=0.18, lw=0.6)
    # No second legend: this panel's six arms and both line styles are already
    # keyed in the panel above, which shares the axis.
    save(fig, "training_curves.png")


# --------------------------------------------------------------------------- tables

def write_tables() -> Path:
    """Both result tables as booktabs LaTeX, from the JSONs the figures read.

    Generated rather than hand-typed so a re-run cannot leave the thesis quoting
    numbers no artefact contains.
    """
    test = load("test", "table.json")
    def num(v: float, places: int = 4, sign: bool = False) -> str:
        """A number in math mode, so a negative renders as a minus and not a hyphen."""
        return f"${v:+.{places}f}$" if sign else f"${v:.{places}f}$"

    rows = []
    for arm in TABLE_ARMS:
        a = test["arms"][arm]
        # The three arm-level columns are printed as means: their spread across
        # folds is image difficulty, which every arm shares, and printing it made
        # the reader ask why AUC had none. The comparison's own interval is in
        # the last column.
        cells = [LABEL[arm],
                 num(a["IG"]["mean"]),
                 num(a["NSS"]["mean"], 3),
                 num(a["AUC"]["mean"])]
        # The delta-IG column carries its own interval, so the table can be read
        # without the prose. It is a fold-paired quantity, which is why the
        # column heading says "fold-paired".
        if arm == "normal":
            cells += ["---"]
        else:
            fp = test["paired_vs_normal"][arm]["fold_paired"]
            cells += [f"${fp['mean']:+.4f} \\pm {2 * fp['se']:.4f}$"]
        rows.append(" & ".join(cells) + r" \\")

    epoch = test["epochs_used"][0].replace("epoch_", "").lstrip("0")
    tex = f"""% Generated by scripts/make_results_figures.py — do not edit by hand.
% Source: {RES.relative_to(ROOT)}/{{test/table.json,
% test/per_image_ig.json}}. Regenerate after any re-evaluation.
% Requires \\usepackage{{booktabs}}.

\\begin{{table}}[tbp]
\\centering
\\caption[The production matrix: seven arms, ten folds]{{Seven arms, ten image-stratified folds,
test split at epoch {epoch}. The first three columns are means over the ten folds; their spread
across folds is dominated by image difficulty, which every arm shares. The last column pairs the
folds, so that shared component cancels, and carries two standard errors of its mean.}}
\\label{{tab:phase17-arms}}
\\small
\\setlength{{\\tabcolsep}}{{4.5pt}}
\\begin{{tabular}}{{lcccr}}
\\toprule
arm & IG over centerbias & NSS & AUC & $\\Delta$IG vs control, fold-paired \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    p = RES / "results_tables.tex"
    p.write_text(tex)
    return p


# The per-image diagnostic is its own run — pretrained read-out, sharp input, all
# 1003 stimuli — so it is not part of the arm tree and does not move with
# --results-root; a tree scored under another protocol would not pair with it.
PER_IMAGE_DIAG = ROOT / "results" / "per_image_diagnostic_initial_1003" / "diagnostic.json"
DISPERSION_SCALARS = ("fixation_entropy_norm", "consensus_area_75_pct")
# Captured at import, before --results-root can rebind RES, so main can tell
# whether it is pointed at the tree the diagnostic was scored under.
DISPERSION_RES = RES


def write_dispersion_link() -> Path:
    """Does the foveation cost land on the images the subjects agreed about?

    ch04 4.2 measures how far the fifteen subjects spread on each stimulus and
    4.1 measures what foveation costs, and nothing joins the two unless the
    per-image cost tracks the per-image spread. A cost concentrated on
    high-agreement images would say the foveated arm loses the obvious targets;
    one concentrated on scattered images would say it cannot sort among
    candidates. This correlates them on stimulus id so that ch05
    disc-per-image can quote a measured bound rather than argue from the design.
    """
    scal = {int(r["stim_idx"]): r["difficulty"]
            for r in json.loads(PER_IMAGE_DIAG.read_text())["rows"]}
    arms: dict[str, dict] = {}
    for arm in ("foveated@40", "foveated@20", "foveated@10"):
        stims, delta, _, _ = paired("test", arm)
        rows = [(scal[int(s)], dv) for s, dv in zip(stims, delta) if int(s) in scal]
        d = np.array([dv for _, dv in rows])
        entry: dict = {"n_stimuli": len(rows), "mean_delta_ig": float(d.mean())}
        for key in DISPERSION_SCALARS:
            x = np.array([sc[key] for sc, _ in rows])
            entry[key] = {"pearson_r": float(stats.pearsonr(x, d)[0]),
                          "spearman_rho": float(stats.spearmanr(x, d)[0])}
        arms[arm] = entry
    largest = max(abs(arms[a][k][c])
                  for a in arms for k in DISPERSION_SCALARS
                  for c in ("pearson_r", "spearman_rho"))
    p = RES / "per_image_dispersion.json"
    p.write_text(json.dumps(
        {"split": "test",
         "source": {"delta_ig": str((RES / "test" / "per_image_ig.json").relative_to(ROOT)),
                    "dispersion": str(PER_IMAGE_DIAG.relative_to(ROOT))},
         "arms": arms,
         "largest_abs_correlation": largest}, indent=2) + "\n")
    return p


def _per_fixation(tag: str) -> dict[str, np.ndarray]:
    """One arm's per-fixation test records, concatenated over the ten folds."""
    keys = ("stimulus_id", "subject", "ll_bits", "cb_bits")
    parts: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    for fold in range(N_FOLDS):
        p = RES / "test" / "per_fixation" / f"{tag}_fold{fold}.npz"
        if not p.exists():
            raise FileNotFoundError(
                f"{p} missing — the sweep eval writes it with --dump-per-fixation")
        with np.load(p, allow_pickle=True) as z:
            for k in keys:
                parts[k].append(z[k])
    return {k: np.concatenate(v) for k, v in parts.items()}


def write_grouping_intervals(test_arm: str = "fov_cpd40", ref_arm: str = "normal") -> Path:
    """The appendix's by-image and by-subject intervals for the main comparison.

    The fold pairing ch04 reports collapses each fold to one number. This takes
    the same contrast one fixation at a time and lets the grouping vary instead,
    which is the appendix's point: pairing by fold is the widest of the three.

    The standard errors are the usual clustered form for a mean: sum the centred
    differences within each group, then take the root of the sum of squares over
    the count.
    """
    a, b = _per_fixation(test_arm), _per_fixation(ref_arm)
    if not np.array_equal(a["stimulus_id"], b["stimulus_id"]):
        raise ValueError(f"{test_arm} and {ref_arm} are not scored on the same fixations")
    d = (a["ll_bits"] - a["cb_bits"]) - (b["ll_bits"] - b["cb_bits"])
    n, mean = d.size, float(d.mean())

    out = {}
    for name, groups in (("image", a["stimulus_id"]), ("subject", a["subject"])):
        _, inv = np.unique(groups, return_inverse=True)
        n_groups = int(inv.max()) + 1
        se = float(np.sqrt((np.bincount(inv, weights=d - mean) ** 2).sum()) / n)
        # Two standard errors, the interval convention of ch03 stats-plan.
        out[name] = {"n_groups": n_groups, "se": se,
                     "interval_2se": [mean - 2 * se, mean + 2 * se]}

    p = RES / "grouping_intervals.json"
    p.write_text(json.dumps(
        {"contrast": f"{test_arm} vs {ref_arm}", "split": "test", "n_fixations": n,
         "mean_dig_bits": mean,
         "note": "Two standard errors, matching ch03 stats-plan. Fold pairing is "
                 "not here: it collapses each fold to one number, and its "
                 "standard error is test/table.json's, which ch04 quotes.",
         "groupings": out}, indent=2) + "\n")
    return p


# --------------------------------------------------------------------------- 4

STRAT = RES / "stratified"


def disc_split(cpd: int) -> dict:
    """Pool the amplitude bins into inside-disc and outside-disc, per fold.

    The per-bin plot invites reading sixteen bins at once and concluding that the
    difference sits outside the identity disc. That conclusion is a contrast, so
    it has to be tested as one: pool each fold's bins into the two regions, pair
    across folds, and difference them. Weighting is by fixation count, which makes
    the all-bins aggregate reproduce the recorded per-fold overall exactly.
    """
    j = json.loads((STRAT / f"cpd{cpd}_val" / "stratified.json").read_text())
    # The radius is itself a bin edge, so "inside" and "outside" are complements
    # and one predicate defines both, so no bin can fall out of either.
    # The rounding is because the radius arrives from JSON as 103.49999999999999.
    r0 = round(j["pooled"]["sharp_radius_px"], 6)

    def inside(r) -> bool:
        return (r["hi_px"] or np.inf) <= r0

    ins, out = [], []
    for f in j["per_fold"]:
        rows = f["by_saccade_amplitude"]

        def agg(keep):
            n = np.array([r["d_LL"]["n"] for r in rows if keep(r)], float)
            m = np.array([r["d_LL"]["mean"] for r in rows if keep(r)], float)
            return float((n * m).sum() / n.sum()) if n.sum() else np.nan

        ins.append(agg(inside))
        out.append(agg(lambda r: not inside(r)))

    return {"radius": j["pooled"]["sharp_radius_px"],
            "inside": np.array(ins), "outside": np.array(out)}


def _interval(a: np.ndarray) -> tuple[float, float, float]:
    a = a[~np.isnan(a)]
    se = float(a.std(ddof=1) / np.sqrt(len(a)))
    return float(a.mean()), se, float(a.mean() / se)


def fig_disc_contrast() -> None:
    """Does the sharp disc locate the difference? Tested as a contrast.

    The per-bin figure invites the conclusion that it does: at cutoff 40 the
    positive bins are the long saccades, beyond the 103.5 px disc. Pooling each
    fold's bins into the two regions gives +0.0019 inside and +0.0050 outside, a
    difference of +0.0031 +- 0.0094 (two standard errors) -- they do not differ.
    The right panel shows why: the sign of outside-minus-inside flips from fold to
    fold, four of the ten. At cutoff 20 the same contrast is +0.0215, t = +2.98,
    and its inside region is the single 0-11.5 px bin the 11.5 px disc leaves.
    """
    # Stacked rather than side by side: the row labels and the numeric column of
    # the upper panel need the full width of the page to stay legible.
    fig, axes = plt.subplots(2, 1, **canvas((9.0, 8.0)),
                             gridspec_kw={"height_ratios": [1.35, 1], "hspace": 0.45})

    ax = axes[0]
    rows, y = [], 0
    for cpd in (10, 20, 40):
        d = disc_split(cpd)
        col = CPD_COLOUR[cpd]
        parts = [("outside disc", d["outside"], "o")]
        if not np.all(np.isnan(d["inside"])):
            parts = [("inside disc", d["inside"], "s"),
                     ("outside disc", d["outside"], "o"),
                     ("outside − inside", d["outside"] - d["inside"], "D")]
        for lab, a, mk in parts:
            m, se, t = _interval(a)
            lo, hi = m - 2 * se, m + 2 * se
            contrast = lab.startswith("outside −")
            ax.plot([lo, hi], [y, y], lw=3.0 if contrast else 2.2, color=col,
                    alpha=1.0 if contrast else 0.75, zorder=3, solid_capstyle="round")
            ax.plot([m], [y], mk, ms=11 if contrast else 8, color=col,
                    mec="white", mew=1.2, zorder=4)
            ax.text(0.088, y, f"{m:+.4f}   t={t:+.2f}", fontsize=9, va="center",
                    color=col, fontweight="bold" if contrast else "normal")
            rows.append((y, f"{lab}" + (f"   (cutoff {cpd})" if lab.startswith("inside")
                                        or len(parts) == 1 else "")))
            y -= 1
        y -= 0.55

    ax.axvline(0, color="#333333", lw=1.4, ls="--", zorder=2)
    ax.set_yticks([r[0] for r in rows])
    ax.set_yticklabels([r[1] for r in rows], fontsize=9)
    ax.set_xlabel(r"$\Delta$IG (bits/fixation), fold-paired, $\pm 2$ SE")
    ax.set_xlim(-0.055, 0.135)
    ax.set_xticks([-0.04, -0.02, 0.0, 0.02, 0.04])
    ax.set_title("pooled inside-disc and outside-disc regions\n"
                 "(fixation-weighted, fold-paired)", fontsize=11)
    ax.grid(axis="x", alpha=0.18, lw=0.6)

    ax = axes[1]
    for cpd in (40, 20):
        d = disc_split(cpd)
        col = CPD_COLOUR[cpd]
        for i, (a, b) in enumerate(zip(d["inside"], d["outside"])):
            ax.plot([0, 1], [a, b], "-", color=col, lw=1.3, alpha=0.55,
                    zorder=2, label=f"cutoff {cpd}, disc {d['radius']:g} px"
                    if i == 0 else None)
            ax.plot([0, 1], [a, b], "o", ms=4, color=col, alpha=0.7, zorder=3)
        mi, mo = np.nanmean(d["inside"]), np.nanmean(d["outside"])
        ax.plot([0, 1], [mi, mo], "-", color=col, lw=3.4, zorder=4)
        ax.plot([0, 1], [mi, mo], "D", ms=10, color=col, mec="white", mew=1.3, zorder=5)
    ax.axhline(0, color="#333333", lw=1.4, ls="--", zorder=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["inside the disc", "outside the disc"], fontsize=10)
    ax.set_xlim(-0.22, 1.22)
    ax.set_ylabel(r"$\Delta$IG (bits/fixation)")
    ax.set_title("per-fold inside-to-outside step\n(one line per fold)", fontsize=11)
    ax.grid(axis="y", alpha=0.18, lw=0.6)
    ax.legend(fontsize=9, loc="lower left", framealpha=0.95)

    fig.suptitle("the sharp-disc account tested as a contrast — validation split",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    save(fig, "disc_contrast.png")


FIGURES = {
    "disc_contrast": fig_disc_contrast,
    "fold_paired": fig_fold_paired,
    "gaze_contingency": fig_gaze_contingency,
    "training_curves": fig_training_curves,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="+", choices=sorted(FIGURES) + ["tables", "dispersion", "groupings"],
                    help="regenerate a subset (default: all figures, the tables and the "
                         "dispersion link)")
    ap.add_argument("--results-root", type=Path, default=None,
                    help="read artefacts from (and write figs/ + tables into) this "
                         "results tree instead of the default "
                         "results/foveation_mit1003_initial")
    args = ap.parse_args()

    if args.results_root is not None:
        global RES, OUT, STRAT
        RES = args.results_root.resolve()
        OUT = RES / "figs"
        STRAT = RES / "stratified"
        load.cache_clear()   # keyed on the path parts, not on RES

    names = args.only or list(FIGURES) + ["tables", "dispersion", "groupings"]
    OUT.mkdir(parents=True, exist_ok=True)
    for name in names:
        if name == "tables":
            print(f"wrote {write_tables()}")
        elif name == "groupings":
            print(f"wrote {write_grouping_intervals()}")
        elif name == "dispersion":
            if RES != DISPERSION_RES:
                print(f"skipped dispersion: {PER_IMAGE_DIAG.name} is scored under the "
                      f"protocol of {DISPERSION_RES.name}, not {RES.name}")
            else:
                print(f"wrote {write_dispersion_link()}")
        else:
            FIGURES[name]()
            print(f"wrote {OUT / (name + '.png')}")


if __name__ == "__main__":
    main()
