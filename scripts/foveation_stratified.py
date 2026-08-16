"""Stratified analysis of the normal-vs-foveated arms: where the difference falls.

The aggregate tables report one number per arm. That number cannot say *why* the
arms differ, or where in a scanpath the difference sits. This script re-evaluates
two trained arms with `evaluate.run(per_fixation=True)`, pairs the resulting
per-fixation records fixation-for-fixation, and reports four things the aggregate
hides:

1. **Likelihood by saccade amplitude.** The mechanistic claim is that foveation
   hurts because the model cannot see peripheral saccade targets. That predicts
   the loss concentrates in *long* saccades and vanishes for short ones. If the
   loss is flat in amplitude, the explanation is wrong and something more
   generic (feature degradation) is doing the work. The sharp-fovea radius
   `e2 * (2 * foveal_cpd - ppd)` marks where the prediction changes regime: a
   saccade shorter than that lands on pixels the foveation left bit-identical.
   That is a prediction, not a tautology — the target pixel is untouched, but
   the backbone's receptive field still reaches blurred periphery, so a dLL of
   ~0 inside the disc is an empirical result about how local the readout is.
2. **Likelihood by fixation index.** Early fixations are centre-biased and
   gist-driven, later ones content-driven, so foveation should cost more later.
   Also checks the gap is not an artefact of the first one or two fixations.
3. **Entropy.** Separates "diffuse and uncertain" (mass spread thinner, higher
   entropy) from "confidently mislocated" (still peaked, peak in the wrong
   place). Both lose likelihood; they are different failures.
4. **Calibration.** The spatial PIT — probability mass at density at least that
   of the fixated pixel — is uniform on [0, 1] for a calibrated model. A
   one-sample KS statistic against uniform quantifies the deviation, and the
   two arms can be compared on it.

Both arms are evaluated on identical scanpath indices, so every comparison is
paired and the fold-difficulty variation that dominates the across-fold error
bars cancels.

    .venv/bin/python scripts/foveation_stratified.py --folds 0 1 2 --cpd 20
    .venv/bin/python scripts/foveation_stratified.py --folds 0 --cpd 40 \
        --subsample-val 40 --split val          # quick local check
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import deepgaze_pytorch

from tez_deepgaze.device import pick_device, to_device
from tez_deepgaze.evaluate import AMP_EDGES_PX, save_per_fixation
from tez_deepgaze.foveated_train import code_provenance
from tez_deepgaze.foveate_input import MIT1003_PPD, Foveation, identity_radius_px
from tez_deepgaze.paths import RESULTS
from tez_deepgaze.script_utils import eval_arm_checkpoint, fold_scanpath_indices

OUT = RESULTS / "foveation_mit1003" / "stratified"
# One amplitude grid, shared with the annuli `evaluate.amp_mass` integrates the
# predicted density over (see AMP_EDGES_PX for how the edges were chosen). It has
# to be the same grid in both places: the measured dLL per bin and the model's
# predicted mass per bin are only comparable bin-for-bin if they are binned
# identically, and comparing them is the whole point of recording amp_mass.
AMP_EDGES = list(AMP_EDGES_PX)
FIX_INDEX_BINS = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 8), (9, 100)]


def paired_stats(delta: np.ndarray) -> dict:
    """Mean, SE and a two-sided one-sample t on a paired difference."""
    d = np.asarray(delta, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2:
        return {"n": n, "mean": float(d.mean()) if n else float("nan"), "se": float("nan"),
                "t": float("nan")}
    mean = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(n))
    return {"n": n, "mean": mean, "se": se, "t": mean / se if se > 0 else float("nan")}


def ks_vs_uniform(pit: np.ndarray) -> float:
    """One-sample Kolmogorov-Smirnov statistic of PIT values against U(0, 1).

    0 = perfectly calibrated. Computed directly rather than via scipy so the
    dependency set stays as declared.
    """
    x = np.sort(np.asarray(pit, dtype=float))
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return float("nan")
    i = np.arange(1, n + 1)
    return float(np.max(np.abs(np.maximum(i / n - x, x - (i - 1) / n))))


def collect_arm(model, pretrained_state, stimuli, fixations, indices, device,
                ckpt_root: Path, tag: str, fold: int, fov, seed: int,
                pretrained: bool = False, dump_dir: Path | None = None,
                split: str = "val", save_maps_for: set[int] | None = None,
                epoch: int | None = None,
                expect_dataset_variant: str | None = None) -> dict:
    """Evaluate one arm, retaining per-fixation records.

    ``pretrained=True`` uses the pretrained DG3 heads for both arms instead of
    the per-(arm, fold) trained checkpoints. That measures the raw input-
    foveation gap with no read-out adaptation, and is the only way to exercise
    this script where the trained checkpoints are absent (they live on the
    cluster). Not the reported configuration.
    """
    res, ckpt, fov = eval_arm_checkpoint(
        model, pretrained_state, stimuli, fixations, indices, device,
        ckpt_root=ckpt_root, tag=tag, fold=fold, fov=fov, seed=seed,
        pretrained=pretrained, fold_no=fold, per_fixation=True,
        save_maps_for=save_maps_for, epoch=epoch,
        expect_dataset_variant=expect_dataset_variant)
    if dump_dir is not None:
        save_per_fixation(
            res, dump_dir / f"{tag}_fold{fold}.npz",
            arm=tag, fold=fold, split=split,
            epoch=None if ckpt is None else ckpt.parent.name,
            ppd=None if fov is None else fov.ppd,
            foveal_cpd=None if fov is None else fov.foveal_cpd,
            heads="pretrained" if pretrained else "trained",
            **code_provenance(),
        )
    return res


def _key(pf: dict) -> np.ndarray:
    """Stable per-fixation identity, so the two arms can be aligned safely."""
    return np.stack([pf["scanpath"], pf["fix_index"]], axis=1)


def analyse(rn: dict, rf: dict, foveal_cpd: float, ppd: float) -> dict:
    """Pair the two arms' per-fixation records and stratify the difference."""
    a, b = rn["per_fixation"], rf["per_fixation"]
    if not np.array_equal(_key(a), _key(b)):
        raise RuntimeError(
            "arms did not produce identically-ordered per-fixation records; "
            "they must be evaluated on the same indices with the same start_fixation"
        )

    ig_n = a["ll_bits"] - a["cb_bits"]
    ig_f = b["ll_bits"] - b["cb_bits"]
    d_ll = b["ll_bits"] - a["ll_bits"]        # == d_IG, the centerbias cancels
    amp = a["sacc_px"]
    r_sharp = identity_radius_px(foveal_cpd, ppd)

    by_amp = []
    for lo, hi in zip(AMP_EDGES[:-1], AMP_EDGES[1:]):
        m = np.isfinite(amp) & (amp >= lo) & (amp < hi)
        if not m.any():
            continue
        by_amp.append({
            "lo_px": lo, "hi_px": None if hi == np.inf else hi,
            "lo_deg": lo / ppd, "hi_deg": None if hi == np.inf else hi / ppd,
            "inside_sharp_disc": bool(hi <= r_sharp) if hi != np.inf else False,
            "frac_of_fixations": float(m.mean()),
            "d_LL": paired_stats(d_ll[m]),
            "IG_normal": float(ig_n[m].mean()),
            "IG_foveated": float(ig_f[m].mean()),
        })

    by_fix = []
    for lo, hi in FIX_INDEX_BINS:
        m = (a["fix_index"] >= lo) & (a["fix_index"] <= hi)
        if not m.any():
            continue
        by_fix.append({"fix_index_lo": lo, "fix_index_hi": hi,
                       "frac_of_fixations": float(m.mean()),
                       "d_LL": paired_stats(d_ll[m])})

    return {
        "sharp_radius_px": r_sharp,
        "sharp_radius_deg": r_sharp / ppd,
        "frac_saccades_inside_sharp_disc": float(np.mean(amp[np.isfinite(amp)] <= r_sharp)),
        "overall": {
            "d_LL": paired_stats(d_ll),
            "d_entropy_bits": paired_stats(b["entropy_bits"] - a["entropy_bits"]),
            "d_mode_dist_px": paired_stats(b["mode_dist_px"] - a["mode_dist_px"]),
            "entropy_bits_normal": float(np.mean(a["entropy_bits"])),
            "entropy_bits_foveated": float(np.mean(b["entropy_bits"])),
            "mode_dist_px_normal": float(np.mean(a["mode_dist_px"])),
            "mode_dist_px_foveated": float(np.mean(b["mode_dist_px"])),
        },
        "calibration": {
            "ks_vs_uniform_normal": ks_vs_uniform(a["pit"]),
            "ks_vs_uniform_foveated": ks_vs_uniform(b["pit"]),
            "pit_mean_normal": float(np.mean(a["pit"])),
            "pit_mean_foveated": float(np.mean(b["pit"])),
            "pit_deciles_normal": np.histogram(a["pit"], bins=10, range=(0, 1))[0].tolist(),
            "pit_deciles_foveated": np.histogram(b["pit"], bins=10, range=(0, 1))[0].tolist(),
        },
        "by_saccade_amplitude": by_amp,
        "by_fixation_index": by_fix,
        "n_fixations": int(len(d_ll)),
    }


def _fmt(st: dict) -> str:
    """Render a paired-stats dict; bins with n < 2 have no SE, so say so."""
    if st["n"] < 2:
        return f"n={st['n']} | — |"
    return f"{st['mean']:+.4f} ± {st['se']:.4f} | {st['t']:+.2f} |"


def render(res: dict, cpd: float, folds: list[int], split: str, out: Path) -> str:
    o, cal = res["overall"], res["calibration"]
    L = [
        f"# Stratified analysis — normal vs foveated@{cpd:g} on MIT1003",
        "",
        f"- Folds {folds}, {split} split, {res['n_fixations']} fixations, paired per fixation.",
        f"- Sharp-fovea radius {res['sharp_radius_px']:.1f} px "
        f"({res['sharp_radius_deg']:.2f} deg); "
        f"{100 * res['frac_saccades_inside_sharp_disc']:.1f} % of saccades land inside it.",
        "",
        "## Overall (paired)",
        "",
        "| quantity | normal | foveated | Δ (foveated − normal) | t |",
        "|---|---|---|---|---|",
        f"| LL (bits/fix) | — | — | {o['d_LL']['mean']:+.4f} ± {o['d_LL']['se']:.4f} "
        f"| {o['d_LL']['t']:+.2f} |",
        f"| entropy (bits) | {o['entropy_bits_normal']:.3f} | {o['entropy_bits_foveated']:.3f} "
        f"| {o['d_entropy_bits']['mean']:+.4f} ± {o['d_entropy_bits']['se']:.4f} "
        f"| {o['d_entropy_bits']['t']:+.2f} |",
        f"| mode distance (px) | {o['mode_dist_px_normal']:.1f} "
        f"| {o['mode_dist_px_foveated']:.1f} "
        f"| {o['d_mode_dist_px']['mean']:+.2f} ± {o['d_mode_dist_px']['se']:.2f} "
        f"| {o['d_mode_dist_px']['t']:+.2f} |",
        "",
        "Entropy rising alongside a likelihood drop means the model spread its mass "
        "(less certain). Entropy flat or falling means it stayed committed and was "
        "mislocated instead — a different failure.",
        "",
        "## Calibration (spatial PIT vs uniform; 0 = calibrated)",
        "",
        f"- KS statistic: normal **{cal['ks_vs_uniform_normal']:.4f}**, "
        f"foveated **{cal['ks_vs_uniform_foveated']:.4f}**",
        f"- PIT mean: normal {cal['pit_mean_normal']:.4f}, "
        f"foveated {cal['pit_mean_foveated']:.4f} (0.5 = uniform)",
        "",
        "## Δ LL by saccade amplitude",
        "",
        "| amplitude (px) | (deg) | inside sharp disc | share of fix. | Δ LL (bits/fix) | t |",
        "|---|---|---|---|---|---|",
    ]
    for r in res["by_saccade_amplitude"]:
        hi = "∞" if r["hi_px"] is None else f"{r['hi_px']:g}"
        hid = "∞" if r["hi_deg"] is None else f"{r['hi_deg']:.1f}"
        L.append(
            f"| {r['lo_px']:g}–{hi} | {r['lo_deg']:.1f}–{hid} "
            f"| {'yes' if r['inside_sharp_disc'] else 'no'} "
            f"| {100 * r['frac_of_fixations']:.1f} % "
            f"| {_fmt(r['d_LL'])}"
        )
    L += [
        "",
        "If the peripheral-preview account is right, Δ LL should be ~0 in the bins inside "
        "the sharp disc and grow with amplitude beyond it. A flat profile refutes it.",
        "",
        "## Δ LL by fixation index",
        "",
        "| fixation index | share of fix. | Δ LL (bits/fix) | t |",
        "|---|---|---|---|",
    ]
    for r in res["by_fixation_index"]:
        lab = (f"{r['fix_index_lo']}" if r["fix_index_lo"] == r["fix_index_hi"]
               else f"{r['fix_index_lo']}–{r['fix_index_hi']}")
        L.append(f"| {lab} | {100 * r['frac_of_fixations']:.1f} % | {_fmt(r['d_LL'])}")
    md = "\n".join(L) + "\n"
    (out / "stratified.md").write_text(md)
    return md


def figure(res: dict, cpd: float, out: Path) -> None:
    """The three diagnostic panels for one arm pair.

    Bars are two standard errors, the one convention ch03 stats-plan sets.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    rows = res["by_saccade_amplitude"]
    xs = np.arange(len(rows))
    means = [r["d_LL"]["mean"] for r in rows]
    errs = [2 * r["d_LL"]["se"] for r in rows]
    cols = ["#56B4E9" if r["inside_sharp_disc"] else "#0072B2" for r in rows]
    ax.bar(xs, means, yerr=errs, color=cols, capsize=3)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"{r['lo_px']:g}–" + ("∞" if r["hi_px"] is None else f"{r['hi_px']:g}") for r in rows],
        rotation=45, ha="right", fontsize=8)
    ax.axvline(sum(1 for r in rows if r["inside_sharp_disc"]) - 0.5,
               color="#E69F00", ls="--", lw=1.2, label="sharp-disc radius")
    ax.set_xlabel("saccade amplitude (px)")
    ax.set_ylabel("Δ LL (bits/fix), test − reference")
    ax.set_title("Cost by saccade amplitude  (bars: ±2 SE)")
    ax.legend(fontsize=8)

    ax = axes[1]
    rows = res["by_fixation_index"]
    xs = np.arange(len(rows))
    ax.bar(xs, [r["d_LL"]["mean"] for r in rows],
           yerr=[2 * r["d_LL"]["se"] for r in rows], color="#0072B2", capsize=3)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"{r['fix_index_lo']}" if r["fix_index_lo"] == r["fix_index_hi"]
         else f"{r['fix_index_lo']}–{r['fix_index_hi']}" for r in rows], fontsize=8)
    ax.set_xlabel("fixation index in scanpath")
    ax.set_ylabel("Δ LL (bits/fix)")
    ax.set_title("Cost by fixation index  (bars: ±2 SE)")

    ax = axes[2]
    cal = res["calibration"]
    centres = np.arange(10) * 0.1 + 0.05
    for key, lab, col in (("pit_deciles_normal", "normal", "#000000"),
                          ("pit_deciles_foveated", f"foveated@{cpd:g}", "#0072B2")):
        counts = np.asarray(cal[key], dtype=float)
        ax.plot(centres, counts / counts.sum(), marker="o", color=col, label=lab)
    ax.axhline(0.1, color="#E69F00", ls="--", lw=1.2, label="calibrated (uniform)")
    ax.set_xlabel("spatial PIT (mass at density ≥ fixation's)")
    ax.set_ylabel("fraction of fixations")
    ax.set_title("Calibration")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out / "stratified.png", dpi=140)
    plt.close(fig)


# arm name -> (checkpoint tag template, center_fovea flag or None for sharp).
# None means "no Foveation at all", which is what the sharp control is.
ARM_SPECS = {
    "normal":   ("normal", None),
    "foveated": ("fov_cpd{c}", False),
    "center":   ("fov_cpd{c}_center", True),
}


def _arm(name: str, cpd: float, ppd: float, device):
    """(checkpoint tag, Foveation or None) for an arm name from ARM_SPECS."""
    tag_tmpl, center = ARM_SPECS[name]
    tag = tag_tmpl.format(c=int(cpd))
    if center is None:
        return tag, None
    return tag, Foveation(ppd=ppd, foveal_cpd=cpd, center_fovea=center).to(device)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Redraws stratified.png from a committed stratified.json: no model, no
    # checkpoints, no corpus.
    ap.add_argument("--replot", type=Path, default=None, metavar="STRATIFIED_DIR",
                    help="redraw stratified.png from that directory's "
                         "stratified.json and exit")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--cpd", type=float, default=20.0,
                    help="foveal_cpd of the foveated/centre arms named below")
    # `--arm-test center --arm-ref foveated` is the contrast that isolates
    # gaze-contingency: same blur profile, fovea pinned instead of tracking.
    # Defaults give normal-vs-fov_cpd{C}.
    ap.add_argument("--arm-ref", default="normal", choices=list(ARM_SPECS),
                    help="reference arm (the 'a' side; deltas are test - ref)")
    ap.add_argument("--arm-test", default="foveated", choices=list(ARM_SPECS),
                    help="test arm (the 'b' side)")
    ap.add_argument("--ppd", type=float, default=MIT1003_PPD)
    ap.add_argument("--ckpt-root", type=Path,
                    default=RESULTS / "foveation_mit1003" / "ckpts")
    ap.add_argument("--subsample-val", type=int, default=None,
                    help="evaluate only N scanpaths per fold (smoke runs)")
    ap.add_argument("--pretrained", action="store_true",
                    help="use pretrained DG3 heads for both arms instead of trained "
                         "checkpoints (raw input-foveation gap; also the only way to run "
                         "this locally, where the cluster checkpoints are absent)")
    ap.add_argument("--dump-per-fixation", action="store_true",
                    help="write per-fixation npz per (arm, fold) into <out>/per_fixation/. "
                         "Required for any per-fixation analysis (index_profile.py) — "
                         "the clustering keys (subject, stimulus_id) exist only there.")
    ap.add_argument("--save-maps-for", type=int, nargs="*", default=None,
                    help="stimulus indices whose half-resolution predicted densities are "
                         "retained (a few MB; for figures and post-hoc questions)")
    # Without this the checkpoint resolver falls through to "last epoch present",
    # which is not the reporting epoch. A run left to the default would silently
    # analyse different checkpoints from the ones every reported number comes from.
    ap.add_argument("--epoch", type=int, default=None,
                    help="checkpoint epoch to analyse. Default: last epoch present, "
                         "which is NOT the reported epoch — pass the reporting epoch "
                         "(scripts/summarize_epoch.py --stop-epoch) to match chapter 4.")
    ap.add_argument("--dataset-variant", choices=["plain", "initial"], default="plain",
                    help="'initial' loads MIT1003_initial_fix_consistent so scoring "
                         "covers every free fixation with the centre as history (the "
                         "DG3 paper's protocol); pair with the matching --ckpt-root")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.replot is not None:
        payload = json.loads((args.replot / "stratified.json").read_text())
        figure(payload["pooled"], float(payload["cpd"]), args.replot)
        print(f"redrew {args.replot / 'stratified.png'}")
        return

    if args.arm_ref == args.arm_test:
        raise SystemExit(f"--arm-ref and --arm-test are both {args.arm_ref!r}: "
                         "there is no contrast to compute")
    # The split and the arm pair are in the directory name, so a val run and a
    # test run (or two arm pairs at one cpd) cannot overwrite each other.
    pair = "" if (args.arm_ref, args.arm_test) == ("normal", "foveated") \
        else f"_{args.arm_test}_vs_{args.arm_ref}"
    # A non-plain variant gets its own results tree, matching the sweep evals
    # and what make_results_figures/make_protocol_figures read.
    variant_root = (
        OUT if args.dataset_variant == "plain"
        else RESULTS / f"foveation_mit1003_{args.dataset_variant}" / "stratified"
    )
    out = args.out or (variant_root / f"cpd{args.cpd:g}_{args.split}{pair}")
    out.mkdir(parents=True, exist_ok=True)
    dump_dir = (out / "per_fixation") if args.dump_per_fixation else None
    maps_for = set(args.save_maps_for) if args.save_maps_for else None

    device = pick_device()
    from tez_deepgaze.paths import load_mit1003_variant

    stimuli, fixations = load_mit1003_variant(args.dataset_variant)
    print("loading DeepGaze III...")
    model = to_device(deepgaze_pytorch.DeepGazeIII(pretrained=True), device).eval()
    pretrained_state = {k: v.clone() for k, v in model.state_dict().items()}
    tag_ref, fov_ref = _arm(args.arm_ref, args.cpd, args.ppd, device)
    tag_test, fov_test = _arm(args.arm_test, args.cpd, args.ppd, device)
    print(f"contrast: {args.arm_test} ({tag_test}) - {args.arm_ref} ({tag_ref})")

    per_fold = []
    for fold in args.folds:
        idx = fold_scanpath_indices(stimuli, fixations, fold, args.split,
                                    args.subsample_val, args.seed)
        print(f"fold {fold} {args.split}: {len(idx)} scanpaths")
        rn = collect_arm(model, pretrained_state, stimuli, fixations, idx, device,
                         args.ckpt_root, tag_ref, fold, fov_ref, args.seed,
                         args.pretrained, dump_dir=dump_dir, split=args.split,
                         save_maps_for=maps_for, epoch=args.epoch,
                         expect_dataset_variant=args.dataset_variant)
        rf = collect_arm(model, pretrained_state, stimuli, fixations, idx, device,
                         args.ckpt_root, tag_test, fold, fov_test, args.seed,
                         args.pretrained, dump_dir=dump_dir, split=args.split,
                         save_maps_for=maps_for, epoch=args.epoch,
                         expect_dataset_variant=args.dataset_variant)
        res = analyse(rn, rf, args.cpd, args.ppd)
        res["fold"] = fold
        per_fold.append(res)
        print(f"  dLL {res['overall']['d_LL']['mean']:+.4f}  "
              f"dEntropy {res['overall']['d_entropy_bits']['mean']:+.4f}  "
              f"dMode {res['overall']['d_mode_dist_px']['mean']:+.2f} px")

    # Pool folds by concatenating their per-fixation populations: every fold
    # contributes disjoint stimuli, so the pooled set is each fixation once.
    pooled = per_fold[0] if len(per_fold) == 1 else _pool(per_fold)
    payload = {"cpd": args.cpd, "ppd": args.ppd, "folds": args.folds,
               "dataset_variant": args.dataset_variant,
               "split": args.split, "seed": args.seed, "device": str(device),
               "heads": "pretrained" if args.pretrained else "trained per-(arm,fold)",
               # Which arms the deltas are actually between. The nested keys carry
               # `_normal` / `_foveated` suffixes that read as reference / test
               # rather than as arm names — the pair below is what says who they
               # were.
               "arm_ref": args.arm_ref, "arm_test": args.arm_test,
               "tag_ref": tag_ref, "tag_test": tag_test,
               "delta_is": f"{args.arm_test} - {args.arm_ref}",
               "pooled": pooled, "per_fold": per_fold}
    (out / "stratified.json").write_text(json.dumps(payload, indent=2))
    print(render(pooled, args.cpd, args.folds, args.split, out))
    figure(pooled, args.cpd, out)
    print(f"wrote {out}/stratified.{{md,json,png}}")


def _pool(per_fold: list[dict]) -> dict:
    """Fixation-count-weighted pool of the per-fold stratifications.

    Means combine by weight; standard errors combine as independent estimates
    of the same population mean, which is valid because folds hold disjoint
    stimuli. Bins absent from a fold simply do not contribute.
    """
    def comb(entries: list[tuple[int, dict]]) -> dict:
        n = sum(e[1]["n"] for e in entries)
        if n == 0:
            return {"n": 0, "mean": float("nan"), "se": float("nan"), "t": float("nan")}
        mean = sum(e[1]["n"] * e[1]["mean"] for e in entries) / n
        var = sum((e[1]["n"] * e[1]["se"]) ** 2 for e in entries if np.isfinite(e[1]["se"]))
        se = math.sqrt(var) / n if var > 0 else float("nan")
        return {"n": n, "mean": mean, "se": se,
                "t": mean / se if se and np.isfinite(se) and se > 0 else float("nan")}

    out = {k: per_fold[0][k] for k in ("sharp_radius_px", "sharp_radius_deg")}
    out["n_fixations"] = sum(f["n_fixations"] for f in per_fold)
    # Coverage is a property of the fixation population, not of the geometry,
    # so it must be pooled rather than copied from the first fold.
    out["frac_saccades_inside_sharp_disc"] = sum(
        f["frac_saccades_inside_sharp_disc"] * f["n_fixations"] for f in per_fold
    ) / out["n_fixations"]
    out["overall"] = {}
    for k in per_fold[0]["overall"]:
        if isinstance(per_fold[0]["overall"][k], dict):
            out["overall"][k] = comb([(0, f["overall"][k]) for f in per_fold])
        else:
            tot = out["n_fixations"]
            out["overall"][k] = sum(
                f["overall"][k] * f["n_fixations"] for f in per_fold) / tot
    cal_keys = per_fold[0]["calibration"]
    out["calibration"] = {}
    for k in cal_keys:
        if k.startswith("pit_deciles"):
            out["calibration"][k] = np.sum(
                [f["calibration"][k] for f in per_fold], axis=0).tolist()
        else:
            tot = out["n_fixations"]
            out["calibration"][k] = sum(
                f["calibration"][k] * f["n_fixations"] for f in per_fold) / tot
    for section, key in (("by_saccade_amplitude", "lo_px"), ("by_fixation_index", "fix_index_lo")):
        merged = {}
        for f in per_fold:
            for row in f[section]:
                merged.setdefault(row[key], []).append(row)
        rows = []
        for k in sorted(merged):
            group = merged[k]
            row = dict(group[0])
            n_tot = sum(g["d_LL"]["n"] for g in group)
            row["d_LL"] = comb([(0, g["d_LL"]) for g in group])
            row["frac_of_fixations"] = n_tot / out["n_fixations"]
            for extra in ("IG_normal", "IG_foveated"):
                if extra in row:
                    row[extra] = sum(
                        g[extra] * g["d_LL"]["n"] for g in group) / max(n_tot, 1)
            rows.append(row)
        out[section] = rows
    return out


if __name__ == "__main__":
    main()
