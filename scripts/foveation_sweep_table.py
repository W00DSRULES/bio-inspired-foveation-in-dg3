"""Normal vs gaze-contingent foveated DG3 on MIT1003, with a foveation-strength
sweep and 10-fold cross-validation.

For each requested fold and arm it evaluates the held-out split with the shared
evaluator (:func:`evaluate.run`, foveation-aware) — LL / IG / NSS / AUC — then
aggregates across folds into `results/foveation_mit1003/table.{md,json}` with
one row per arm (normal, foveated@each foveal_cpd) and a side-by-side sampled-
scanpath figure across the sweep. The per-image IGs behind those aggregates are
written to `per_image_ig.json`, so the arms can also be compared paired on the
stimulus.

Arms and heads
--------------
- normal: sharp input.
- foveated@C: gaze-contingent foveated input at foveal_cpd = C.
- center@C (`--center-cpds`): the same blur profile with the fovea pinned to the
  image centre. foveated@C minus center@C is the gaze-contingency term.
Heads come from per-(arm, fold) trained checkpoints under `--ckpt-root`
(the cluster deliverable), or, with `--pretrained`, the pretrained DG3 heads
for every cell (fast local preliminary — the pure input-foveation gap, no
fine-tuning). Checkpoint layout:
    {ckpt-root}/{tag}/fold{k}/epoch_{E:03d}/weights.pt
    tag = "normal", "fov_cpd{C}" or "fov_cpd{C}_center"

Eval is forward-only, so MPS is fine (only DG3 training backward is broken on
MPS). Table rows carry mean ± 2 SE, the ch03 convention; the SE itself is
cross-fold when >1 fold, within-fold per-image for 1 fold.

Local preliminary sweep (pretrained heads, fold 0):
    .venv/bin/python scripts/foveation_sweep_table.py --pretrained --folds 0 \
        --split val --subsample-val 40 --cpds 40 20 10
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import deepgaze_pytorch

from tez_deepgaze.centerbias import load_centerbias_for_image
from tez_deepgaze.device import pick_device, to_device
from tez_deepgaze.evaluate import save_per_fixation
from tez_deepgaze.foveated_train import code_provenance
from tez_deepgaze.fast_dg3 import apply_fast_forward, apply_fast_layernorm
from tez_deepgaze.foveate_input import MIT1003_PPD, MIT1003_PPD_REPORT, Foveation
from tez_deepgaze.instrument import image_tensor, sample_scanpath
from tez_deepgaze.paths import RESULTS
from tez_deepgaze.script_utils import (
    draw_scanpath,
    eval_arm_checkpoint,
    fold_scanpath_indices,
    image_rgb,
    paired_stats,
    set_heads,
    stim_label,
)

OUT = RESULTS / "foveation_mit1003"
# Okabe-Ito, lightness-separated: normal (dark) then foveated by strength.
# fov40 is the primary arm and needs its own step, distinct from fov20.
ARM_COLORS = {"normal": "#000000", "fov40": "#009E73",
              "fov20": "#0072B2", "fov10": "#E69F00"}


def _arm_metrics(r: dict, fov, seed, dump: Path | None = None, **meta):
    """Condense one (arm, fold) eval result. IG is evaluate.run's
    image-stratified IG over the centerbias (computed in-loop, no separate pass).

    With ``dump`` set, the per-fixation records are written there.
    That is what any per-fixation analysis (stratification, clustered errors)
    needs: the clustering keys (subject, stimulus_id) exist nowhere else, and
    cannot be recovered from the aggregates. Costs some eval time for the density-shape
    diagnostics, so it is opt-in rather than always on.
    """
    if dump is not None:
        save_per_fixation(r, dump, seed=seed,
                          ppd=None if fov is None else fov.ppd,
                          foveal_cpd=None if fov is None else fov.foveal_cpd,
                          center_fovea=None if fov is None else fov.center_fovea,
                          **meta, **code_provenance())
    return {
        "LL": (r["LL_bits_mean"], r["LL_bits_se"]),
        "IG": (r["IG_bits_mean"], r["IG_bits_se"]),
        "NSS": (r["NSS_mean"], r["NSS_se"]),
        "AUC": (r["AUC_mean"], r["AUC_se"]),
        "cb_floor": r["ll_centerbias_bits_pooled"],
        "n_images": r["n_images"], "n_fixations": r["n_fixations"],
        # Kept per fold so the arms can be compared paired on the stimulus. The
        # aggregates above average away image difficulty, which is the largest
        # variance component by far.
        "per_image": r["per_image"],
    }


def _sample_arm_paths(model, stimuli, fixations, stim_indices, device, fov,
                      n_samples: int, seed: int, path: Path, **meta) -> None:
    """Draw ``n_samples`` scanpaths per stimulus and write them as one npz.

    Sampled *stochastically*, not greedily. HAT (Yang 2024) evaluates with a
    single greedy argmax path per image to stop best-of-N from favouring
    over-deterministic models, and that is the right call for its similarity
    metrics — but DeepGaze III is a distribution model, so its greedy path is
    the mode trajectory rather than a draw, and would systematically
    under-disperse exactly the amplitude and direction statistics this pass
    exists to measure. The HAT discipline that does transfer is declaring the
    count up front, which ``--sample-paths`` does.

    Paths start at the image centre, matching the forced central first fixation
    MIT1003 presentation used and the convention ``compute_log_density_batch``
    applies when there is no history. Length is matched per image to the human
    median, so the sampled and human amplitude distributions are drawn from
    scanpaths of the same length and a length difference cannot masquerade as a
    behavioural one.

    ``border_dist_px`` travels with every fixation because the density is
    normalised over the frame, so amplitudes are bounded by it. Analyses that
    exclude near-border fixations (Acik et al. drop within 3 deg) need the
    distance rather than a flag.
    """
    from tez_deepgaze.human_scanpaths import human_length_for_stim

    cols: dict[str, list] = {k: [] for k in (
        "stim", "sample", "seed", "fix_index", "x", "y", "border_dist_px")}
    for sidx in tqdm(stim_indices, desc="sample"):
        image = image_rgb(stimuli, sidx)
        H, W = image.shape[:2]
        cb = load_centerbias_for_image(H, W)
        try:
            human_len = human_length_for_stim(fixations, sidx, "median")
        except RuntimeError:      # no scanpath long enough to aggregate
            continue
        # sample_scanpath returns the start point plus n_fixations draws, so a
        # human path of length L is matched by L-1 draws, not L.
        n_draws = human_len - 1
        if n_draws < 1:
            continue
        stack = fov.blur_stack(image_tensor(image, device)) if fov is not None else None
        for s in range(n_samples):
            # Distinct per (stimulus, sample) so no two draws share a stream,
            # and reproducible from the recorded seed alone.
            sp_seed = seed * 1_000_003 + sidx * 101 + s
            sp = sample_scanpath(model, image, cb, (W / 2.0, H / 2.0), n_draws,
                                 device, seed=sp_seed, foveation=fov,
                                 foveation_stack=stack)
            for j, (x, y) in enumerate(zip(sp.x, sp.y)):
                cols["stim"].append(sidx)
                cols["sample"].append(s)
                cols["seed"].append(sp_seed)
                cols["fix_index"].append(j)
                cols["x"].append(float(x))
                cols["y"].append(float(y))
                cols["border_dist_px"].append(
                    float(min(x, y, W - 1 - x, H - 1 - y)))
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: np.asarray(v) for k, v in cols.items()}
    payload["meta_json"] = np.asarray(json.dumps({
        "n_samples": n_samples, "seed": seed, "start": "image_centre",
        "length": "human median per image", "sampling": "stochastic",
        "ppd_blur": None if fov is None else fov.ppd,
        "ppd_report": MIT1003_PPD_REPORT,
        **meta, **code_provenance(),
    }))
    np.savez_compressed(path, **payload)


def _aggregate(per_fold: list[dict]) -> dict:
    """Cross-fold mean ± SE (SE = std/sqrt(K)); within-fold SE when K == 1."""
    out = {}
    for metric in ("LL", "IG", "NSS", "AUC"):
        vals = [f[metric][0] for f in per_fold]
        if len(vals) > 1:
            mean = float(np.mean(vals))
            se = float(np.std(vals, ddof=1) / math.sqrt(len(vals)))
        else:
            mean, se = vals[0], per_fold[0][metric][1]
        out[metric] = {"mean": mean, "se": se}
    out["n_folds"] = len(per_fold)
    out["n_fixations"] = sum(f["n_fixations"] for f in per_fold)
    return out


def _sweep_figure(model, stimuli, stim_ids, cpds, ppd, device, seed, n_fix, out_png):
    """Illustrative: sampled scanpaths (pretrained heads) across the sweep."""
    cols = 1 + len(cpds)
    fig, axes = plt.subplots(len(stim_ids), cols, figsize=(4.0 * cols, 3.6 * len(stim_ids)),
                             squeeze=False)
    for r, sidx in enumerate(stim_ids):
        image = image_rgb(stimuli, sidx)
        H, W = image.shape[:2]
        cb = load_centerbias_for_image(H, W)
        start = (W / 2.0, H / 2.0)
        panels = [("normal", ARM_COLORS["normal"],
                   sample_scanpath(model, image, cb, start, n_fix, device, seed=seed))]
        for c in cpds:
            fov = Foveation(ppd=ppd, foveal_cpd=c)
            panels.append((f"foveated@{c:g}", ARM_COLORS.get(f"fov{int(c)}", "#0072B2"),
                           sample_scanpath(model, image, cb, start, n_fix, device,
                                           seed=seed, foveation=fov)))
        for cc, (title, color, sp) in enumerate(panels):
            ax = axes[r, cc]
            ax.imshow(image)
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)
            ax.axis("off")
            draw_scanpath(ax, sp.x, sp.y, color=color)
            if r == 0:
                ax.set_title(title, fontsize=11)
            if cc == 0:
                ax.set_ylabel(stim_label(stimuli, sidx), fontsize=9)
    fig.suptitle("Sampled scanpaths (same seed) across the foveation sweep "
                 "(pretrained heads)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _fmt(m: dict) -> str:
    """mean ± 2 SE, the one convention ch03 stats-plan sets for the thesis."""
    return f"{m['mean']:+.3f} ± {2 * m['se']:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--dataset-variant", choices=["plain", "initial"], default="plain",
                    help="'plain' is the committed protocol (central start fixation "
                         "dropped on load, scoring starts at the second free fixation). "
                         "'initial' loads MIT1003_initial_fix_consistent, where index 0 "
                         "is the forced central fixation, so the same start_fixation=1 "
                         "scores every free fixation with the centre as history — the "
                         "DG3 paper's protocol. Folds and scanpath order are identical "
                         "across variants (validated by fetch_mit1003.py --with-initial).")
    ap.add_argument("--cpds", type=float, nargs="+", default=[40.0, 20.0, 10.0],
                    help="foveal_cpd sweep for the gaze-contingent foveated arm")
    ap.add_argument("--center-cpds", type=float, nargs="*", default=[],
                    help="foveal_cpd values to also evaluate as the fixed-centre "
                         "ablation (fovea pinned to the image centre). Differencing "
                         "foveated@C against center@C isolates the gaze-contingency, "
                         "since the blur profile is identical.")
    ap.add_argument("--ppd", type=float, default=MIT1003_PPD)
    ap.add_argument("--pretrained", action="store_true",
                    help="use pretrained heads for every cell (local preliminary; "
                         "no checkpoints needed)")
    ap.add_argument("--ckpt-root", type=Path, default=OUT / "ckpts",
                    help="per-(arm,fold) trained checkpoints (ignored with --pretrained)")
    ap.add_argument("--epoch", type=int, default=None, help="checkpoint epoch (default: last)")
    ap.add_argument("--subsample-val", type=int, default=None,
                    help="cap scanpaths per fold (local smoke)")
    ap.add_argument("--fig-stims", type=int, nargs="*", default=[91, 3],
                    help="stimuli drawn in the sampled-scanpath sweep figure")
    ap.add_argument("--sample-paths", type=int, default=0, metavar="N",
                    help="draw N scanpaths per stimulus per (arm, fold) and write them "
                         "to <out>/scanpaths/. 0 (default) skips sampling. Needed for the "
                         "saccade-amplitude / direction comparison against the human "
                         "envelope, which no committed artefact currently supports — "
                         "sampled paths were previously drawn straight to a PNG and the "
                         "coordinates discarded. Costs roughly N x (stimuli) x (path "
                         "length) forward passes per arm-fold, so do it in the eval pass "
                         "rather than as a separate job.")
    ap.add_argument("--sample-stims", type=int, default=None, metavar="M",
                    help="sample on only M stimuli per fold (deterministic subsample). "
                         "Caps the cost of --sample-paths; default is every stimulus.")
    ap.add_argument("--save-maps-for", type=int, nargs="*", default=None,
                    help="stimulus indices whose half-resolution predicted log-densities "
                         "are retained alongside the per-fixation dump. Roughly 35 MB per "
                         "stimulus per arm, so name them deliberately; needs "
                         "--dump-per-fixation")
    ap.add_argument("--n-fix", type=int, default=10)
    ap.add_argument("--dump-per-fixation", action="store_true",
                    help="retain and write per-fixation records to <out>/per_fixation/ "
                         "as one npz per (arm, fold). Required for any per-fixation "
                         "analysis: the clustering keys (subject, stimulus_id) exist "
                         "only there and cannot be recovered from the aggregates "
                         "afterwards.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--fast-forward", action=argparse.BooleanOptionalAction, default=True,
                    help="use the faster DG3 forward and layer norm (see fast_dg3). "
                         "--no-fast-forward evaluates through the vendor code instead.")
    ap.add_argument("--paired-from", type=Path, default=None, metavar="DIR",
                    help="no eval: recompute the paired arm comparison from an "
                         "existing DIR/per_image_ig.json and write DIR/paired.json. "
                         "Makes a quoted paired CI regenerable from the committed "
                         "artefact without re-running the sweep.")
    args = ap.parse_args()

    if args.paired_from is not None:
        per_image = json.loads((args.paired_from / "per_image_ig.json").read_text())
        paired = paired_stats(per_image["arms"])
        payload = {"source": "per_image_ig.json", "ref": "normal",
                   "arms": paired, **code_provenance()}
        (args.paired_from / "paired.json").write_text(json.dumps(payload, indent=2))
        for label, p in paired.items():
            fp = p["fold_paired"]
            ci = ("n/a (single fold)" if fp["se"] is None else
                  f"± {2 * fp['se']:.4f} (2 SE)")
            print(f"{label}: fold-paired dIG {fp['mean']:+.4f} {ci} "
                  f"({fp['negative_folds']}/{fp['n_folds']} folds negative); "
                  f"image-paired t={p['image_paired']['t']:+.2f} "
                  f"over {p['image_paired']['n_images']} images")
        print(f"wrote {args.paired_from / 'paired.json'}")
        return

    device = torch.device(args.device) if args.device else pick_device()
    print(f"device: {device}  |  {'pretrained heads' if args.pretrained else 'trained checkpoints'}")
    # The split goes in the output path, so a val run and a test run to the same
    # --out cannot overwrite each other's table.json, per_image_ig.json,
    # paired.json and per-fixation npz.
    args.out = args.out / args.split
    dump_dir = (args.out / "per_fixation") if args.dump_per_fixation else None
    # Density maps are their own decision, not a side effect of asking for
    # per-fixation records: at ~0.5 GB of half-resolution maps across 7 arms they
    # must be asked for.
    maps_for = set(args.save_maps_for) if args.save_maps_for else None
    if dump_dir is not None:
        print(f"per-fixation records -> {dump_dir}")
    if maps_for:
        print(f"density maps -> {dump_dir} for stimuli {sorted(maps_for)}")
    from tez_deepgaze.paths import load_mit1003_variant

    stimuli, fixations = load_mit1003_variant(args.dataset_variant)

    arms = [("normal", "normal", None)]  # (label, ckpt-tag, foveation)
    for c in args.cpds:
        arms.append((f"foveated@{c:g}", f"fov_cpd{int(c)}", Foveation(ppd=args.ppd, foveal_cpd=c)))
    # Tags must match foveation_train_fold.sbatch, which gives the centre arm its
    # own directory so it cannot overwrite the gaze-contingent arm at the same cpd.
    for c in args.center_cpds:
        arms.append((f"center@{c:g}", f"fov_cpd{int(c)}_center",
                     Foveation(ppd=args.ppd, foveal_cpd=c, center_fovea=True)))

    print("loading DeepGaze III...")
    model = to_device(deepgaze_pytorch.DeepGazeIII(pretrained=True), device).eval()
    pretrained_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    # Same two forward-path changes training uses, for the same reason — this
    # evaluation is the slowest job in the project. The hoist is bit-identical;
    # the layer norm moves the log-density by ~1e-6 nats, five orders below the
    # third decimal these tables report. Both are applied AFTER the pretrained
    # state is captured, so head loading is unaffected.
    if args.fast_forward:
        print(f"fast forward applied to {apply_fast_forward(model)} mixture module(s)")
        print(f"fast layernorm applied to {apply_fast_layernorm(model)} module(s)")

    per_fold = {label: [] for label, _, _ in arms}
    used_epochs: set[str] = set()
    # Checkpoints trained under a different dataset variant than this eval are
    # legitimate (that cross is exactly the protocol-robustness measurement)
    # but must be visible in the artefact, not silent.
    train_variant_mismatches: dict[str, set[str]] = {}
    for fold in args.folds:
        indices = fold_scanpath_indices(stimuli, fixations, fold, args.split,
                                         args.subsample_val, args.seed)
        print(f"fold {fold} {args.split}: {len(indices)} scanpaths")
        for label, tag, fov in arms:
            r, ckpt, eval_fov = eval_arm_checkpoint(
                model, pretrained_state, stimuli, fixations, indices, device,
                ckpt_root=args.ckpt_root, tag=tag, fold=fold, fov=fov,
                seed=args.seed, pretrained=args.pretrained, epoch=args.epoch,
                fold_no=fold,
                per_fixation=dump_dir is not None, save_maps_for=maps_for)
            if ckpt is not None:
                used_epochs.add(ckpt.parent.name)
                try:
                    train_variant = json.loads(
                        (ckpt.parent / "metrics.json").read_text()
                    ).get("dataset_variant", "plain")
                except (OSError, json.JSONDecodeError):
                    train_variant = "unknown"
                if train_variant != args.dataset_variant:
                    train_variant_mismatches.setdefault(label, set()).add(train_variant)
            m = _arm_metrics(
                r, eval_fov, args.seed,
                dump=(dump_dir / f"{tag}_fold{fold}.npz") if dump_dir else None,
                arm=label, tag=tag, fold=fold, split=args.split,
                dataset_variant=args.dataset_variant,
                epoch=None if ckpt is None else ckpt.parent.name)
            per_fold[label].append(m)
            print(f"  {label:14s} LL={m['LL'][0]:+.3f} IG={m['IG'][0]:+.3f} "
                  f"NSS={m['NSS'][0]:+.3f} AUC={m['AUC'][0]:.3f} (cb floor {m['cb_floor']:+.3f})")
            if args.sample_paths > 0:
                # Same heads and same eval_fov as the metrics above, so the
                # sampled behaviour and the reported likelihood describe one model.
                stims = sorted({int(fixations.train_ns[i]) for i in indices})
                if args.sample_stims is not None and args.sample_stims < len(stims):
                    rng = np.random.RandomState(args.seed + fold)
                    stims = sorted(rng.choice(stims, size=args.sample_stims,
                                              replace=False).tolist())
                _sample_arm_paths(
                    model, stimuli, fixations, stims, device, eval_fov,
                    args.sample_paths, args.seed,
                    args.out / "scanpaths" / f"{tag}_fold{fold}.npz",
                    arm=label, tag=tag, fold=fold, split=args.split,
                    epoch=None if ckpt is None else ckpt.parent.name)

    if train_variant_mismatches:
        print("NOTE: checkpoints were trained under a different dataset variant than "
              f"this eval ({args.dataset_variant}): "
              f"{ {k: sorted(v) for k, v in train_variant_mismatches.items()} } — "
              "recorded in table.json as train_dataset_variant_mismatches")
    if len(used_epochs) > 1:
        print(f"WARNING: folds resolved to different checkpoint epochs {sorted(used_epochs)} "
              "— the cross-fold mean mixes training lengths; pass --epoch to pin one")

    agg = {label: _aggregate(per_fold[label]) for label, _, _ in arms}
    normal_agg = agg["normal"]

    args.out.mkdir(parents=True, exist_ok=True)
    set_heads(model, pretrained_state, None, device)  # figure uses pretrained heads
    print("rendering sweep figure...")
    _sweep_figure(model, stimuli, args.fig_stims, args.cpds, args.ppd, device,
                  args.seed, args.n_fix, args.out / "scanpath_sweep.png")

    heads = "pretrained heads (no fine-tuning)" if args.pretrained else "trained per-(arm,fold) checkpoints"

    # Per-image IG, one record per (arm, fold). Every arm sees the same stimuli
    # in the same fold, so these pair on `stim` — which removes image difficulty,
    # the dominant variance component, from the arm comparison. paired_stats
    # raises if the arms are NOT paired, so it doubles as the pairing assert.
    per_image_arms = {
        label: [{"fold": fold, **m["per_image"]}
                for fold, m in zip(args.folds, per_fold[label])]
        for label, _, _ in arms
    }
    paired = paired_stats(per_image_arms)

    # Fold-paired gap SE per metric: SE of the per-fold (arm − normal) means.
    # The unpaired cross-fold SEs in `arms` are dominated by image difficulty,
    # which is common to both arms and cancels in the difference.
    def gap_se(label: str, metric: str) -> float | None:
        diffs = [a[metric][0] - n[metric][0]
                 for a, n in zip(per_fold[label], per_fold["normal"])]
        if len(diffs) < 2:
            return None
        return float(np.std(diffs, ddof=1) / math.sqrt(len(diffs)))

    table_json = {
        "heads": heads, "pretrained": args.pretrained,
        "folds": args.folds, "split": args.split, "ppd": args.ppd, "cpds": args.cpds,
        "center_cpds": args.center_cpds,
        "arms": {label: agg[label] for label, _, _ in arms},
        "gaps_vs_normal": {
            label: {k: agg[label][k]["mean"] - normal_agg[k]["mean"] for k in ("LL", "IG", "NSS", "AUC")}
            for label, _, _ in arms if label != "normal"
        },
        "gaps_se_vs_normal": {
            label: {k: gap_se(label, k) for k in ("LL", "IG", "NSS", "AUC")}
            for label, _, _ in arms if label != "normal"
        },
        "paired_vs_normal": paired,
        "ckpt_root": None if args.pretrained else str(args.ckpt_root),
        "dataset_variant": args.dataset_variant,
        **({"train_dataset_variant_mismatches":
            {k: sorted(v) for k, v in train_variant_mismatches.items()}}
           if train_variant_mismatches else {}),
        "epoch_requested": args.epoch,
        "epochs_used": sorted(used_epochs),
        "fast_forward": args.fast_forward,
        "dump_per_fixation": args.dump_per_fixation,
        "device": str(device), "seed": args.seed,
        **code_provenance(),
    }
    (args.out / "table.json").write_text(json.dumps(table_json, indent=2))

    (args.out / "per_image_ig.json").write_text(json.dumps({
        "heads": heads, "folds": args.folds, "split": args.split,
        "arms": per_image_arms,
    }, indent=2))

    n_folds = agg["normal"]["n_folds"]
    md = [
        "# Normal vs gaze-contingent foveated DG3 on MIT1003 (foveation sweep)",
        "",
        f"- Heads: **{heads}**",
        f"- {n_folds}-fold CV, {args.split} split, folds {args.folds}, "
        f"{agg['normal']['n_fixations']} fixations (normal arm)",
        f"- Foveation: ppd={args.ppd:.0f}; sweep foveal_cpd ∈ "
        f"{{{', '.join(f'{c:g}' for c in args.cpds)}}} (lower = stronger)",
        *([f"- Fixed-centre ablation at foveal_cpd ∈ "
           f"{{{', '.join(f'{c:g}' for c in args.center_cpds)}}}: same blur profile, "
           "fovea pinned to the image centre. foveated@C − center@C is the "
           "gaze-contingency term."] if args.center_cpds else []),
        "",
        "| arm | LL (bits/fix over uniform) | IG (bits/fix over centerbias) | NSS | AUC |",
        "|---|---|---|---|---|",
    ]
    for label, _, _ in arms:
        a = agg[label]
        md.append(f"| {label} | {_fmt(a['LL'])} | {_fmt(a['IG'])} | {_fmt(a['NSS'])} | {_fmt(a['AUC'])} |")
    md += ["", "**Gap vs normal (foveated − normal), fold-paired Δ ± 2 SE:**", "",
           "| arm | Δ LL | Δ IG | Δ NSS | Δ AUC |", "|---|---|---|---|---|"]

    def _fmt_gap(label: str, k: str) -> str:
        g = table_json["gaps_vs_normal"][label][k]
        se = table_json["gaps_se_vs_normal"][label][k]
        return f"{g:+.4f}" if se is None else f"{g:+.4f} ± {2 * se:.4f}"

    for label in [a[0] for a in arms if a[0] != "normal"]:
        md.append(f"| {label} | " + " | ".join(_fmt_gap(label, k)
                                               for k in ("LL", "IG", "NSS", "AUC")) + " |")
    for label, p in paired.items():
        fp = p["fold_paired"]
        if fp["se"] is not None:
            # Computed from mean and se rather than read from the file, so an
            # artefact carrying the older Student-t ``ci95`` key prints the same
            # interval as one carrying ``interval_2se``.
            md += ["", f"ΔIG {label}: fold-paired {fp['mean']:+.4f} ± "
                       f"{2 * fp['se']:.4f} (2 SE), "
                       f"{fp['negative_folds']}/{fp['n_folds']} folds negative; "
                       f"image-paired t = {p['image_paired']['t']:+.2f} over "
                       f"{p['image_paired']['n_images']} images."]
    md += ["",
           f"Arm rows: mean ± 2 SE {'across folds' if n_folds > 1 else '(within-fold, per image)'}, "
           "dominated by image difficulty, which is common to all arms. Gap rows are "
           "fold-paired, so that component cancels — gap SEs are not comparable to arm SEs.",
           "Sweep scanpaths: `scanpath_sweep.png`. Per-image IG behind the pairing: "
           "`per_image_ig.json` (recompute with `--paired-from`).",
           ""]
    (args.out / "table.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nwrote {args.out / 'table.md'} and table.json")


if __name__ == "__main__":
    main()
