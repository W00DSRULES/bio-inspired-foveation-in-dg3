"""Helpers shared by the scripts/ entry points.

Small, dependency-light utilities shared by the demo/analysis scripts under
``scripts/``, so there is one definition to read and fix.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def fold_scanpath_indices(stimuli, fixations, fold: int, split: str,
                          subsample: int | None = None, seed: int = 42) -> list[int]:
    """Scanpath indices belonging to one CV fold's split, optionally subsampled."""
    from .datasets import MIT1003TrainingDataset

    stims = set(MIT1003TrainingDataset(fold_no=fold, split=split,
                                       stimuli=stimuli, fixations=fixations).stim_indices)
    train_ns = np.asarray(fixations.train_ns)
    idx = [i for i in range(len(train_ns)) if int(train_ns[i]) in stims]
    if subsample is not None and subsample < len(idx):
        rng = np.random.RandomState(seed)
        idx = sorted(rng.choice(idx, size=subsample, replace=False).tolist())
    return idx


def ckpt_path(ckpt_root: Path, tag: str, fold: int, epoch: int | None = None) -> Path:
    """Resolve ``{ckpt_root}/{tag}/fold{k}/epoch_{E:03d}/weights.pt``."""
    fold_dir = ckpt_root / tag / f"fold{fold}"
    if epoch is not None:
        return fold_dir / f"epoch_{epoch:03d}" / "weights.pt"
    # Only consider complete checkpoints — an epoch dir whose weights.pt was
    # never written (interrupted job) must not shadow the last good one.
    epochs = sorted(p for p in fold_dir.glob("epoch_*") if (p / "weights.pt").exists())
    if not epochs:
        raise FileNotFoundError(f"no complete epoch_*/weights.pt under {fold_dir}")
    return epochs[-1] / "weights.pt"


def foveation_record(fov) -> dict:
    """Everything a checkpoint has to store about the input its heads were trained on.

    Written by training into the bundle's ``metrics.json`` and read back by
    :func:`resolve_ckpt_foveation`. The two live next to each other on purpose:
    a setting that is written but never read back is the same as not recording it.

    ``e2_deg`` and ``n_levels`` are constructor arguments of
    :class:`~tez_deepgaze.foveate_input.Foveation` and are recorded because
    ``resolve_ckpt_foveation`` validates only the keys it finds: an unrecorded
    parameter would be rebuilt from the *eval request* instead of the
    checkpoint. e2 sets the whole falloff shape
    (and the sharp-disc radius that carries the geometric argument); n_levels
    sets the coarsest available blur, which binds at cpd 10.

    ``level_sigmas`` / ``sharp_radius_px`` are derived, not inputs. They are here
    so an artefact describes the transform it used without re-running code.
    """
    return {"ppd": fov.ppd, "foveal_cpd": fov.foveal_cpd,
            "center_fovea": fov.center_fovea,
            "e2_deg": fov.e2_deg, "n_levels": fov.n_levels,
            "level_sigmas": [round(float(s), 6) for s in fov.level_sigmas],
            "sharp_radius_px": round(fov.sharp_radius_px, 4)}


def resolve_ckpt_foveation(ckpt: Path, fov, allow_arm_mismatch: bool = False):
    """The foveation to evaluate a checkpoint with, reconciled with its own record.

    ``allow_arm_mismatch`` opts into the off-diagonal of the train-condition x
    eval-condition matrix: sharp-trained heads scored on foveated input, or
    foveated-trained heads scored sharp. Those two cells are what separate "the
    transform removed no information the model uses" from "the transform removed
    information and the heads relearned around it" -- the diagonal alone cannot
    tell them apart. It is off by default because an *accidental* mismatch
    produces a large unexplained loss that reads as a finding; callers that set
    it must record the mismatch alongside the number (see ``arm_eval_mismatch``
    in the return of :func:`ckpt_arm_mismatch`).

    The checkpoint tag encodes only foveal_cpd, so the rest of the input
    transform has to come from the bundle's ``metrics.json``.

    ``ppd`` and ``foveal_cpd`` are *validated*: a disagreement there is a caller
    error with no safe default, so it raises. ``center_fovea`` is instead *taken
    from the checkpoint*, because nothing in the eval request identifies it —
    heads trained with the fovea pinned to the image centre and then scored
    gaze-contingently are tested on an input they never saw, and would lose for a
    reason that has nothing to do with the experiment. It cannot travel into an
    eval that asks for no foveation at all, so that combination raises.

    Bundles without a ``center_fovea`` key are left alone.

    The bundle's top-level ``arm`` is checked first: the foveation record alone
    cannot distinguish a sharp-trained checkpoint from a foveated-trained one,
    because training writes the record for both arms. Bundles without the key
    are left alone.
    """
    meta_path = ckpt.parent / "metrics.json"
    if not meta_path.exists():
        return fov
    meta = json.loads(meta_path.read_text())
    rec = meta.get("foveation") or {}

    arm = meta.get("arm")
    if arm is not None and not allow_arm_mismatch:
        if arm != "normal" and fov is None:
            raise ValueError(
                f"{ckpt}: heads were trained on foveated input (arm={arm}), but eval "
                "requested no foveation; pass a matching Foveation, point "
                "--ckpt-root at the sharp arm, or set allow_arm_mismatch to run "
                "this deliberately as a transfer cell"
            )
        if arm == "normal" and fov is not None:
            raise ValueError(
                f"{ckpt}: heads were trained on sharp input (arm=normal), but eval "
                "requested foveation; evaluate this arm sharp, point --ckpt-root "
                "at a foveated arm, or set allow_arm_mismatch to run this "
                "deliberately as a transfer cell"
            )
    if arm is not None and allow_arm_mismatch:
        mismatch = ckpt_arm_mismatch(ckpt, fov)
        if mismatch is not None:
            print(f"{ckpt}: TRANSFER CELL — {mismatch}. This is not a matched "
                  "arm; do not compare it against the diagonal without saying so.")
            # An explicit transfer cell keeps the requested transform: the whole
            # point is to score these heads on input they were not trained on, so
            # the center_fovea reconciliation below must not pull the checkpoint's
            # own setting back in.
            return fov

    got_center = rec.get("center_fovea")
    if got_center is not None and bool(got_center) != bool(getattr(fov, "center_fovea", False)):
        if fov is None:
            raise ValueError(
                f"{ckpt}: heads were trained on fixed-centre foveated input, but eval "
                "requested no foveation; evaluate this arm with a Foveation or point "
                "--ckpt-root at the sharp arm"
            )
        print(f"{ckpt}: heads trained with center_fovea={bool(got_center)} — "
              f"evaluating that way (request said {fov.center_fovea})")
        fov = type(fov)(ppd=fov.ppd, foveal_cpd=fov.foveal_cpd, e2_deg=fov.e2_deg,
                        n_levels=fov.n_levels,
                        center_fovea=bool(got_center)).to(fov.level_sigmas.device)

    if fov is None:
        return None
    # e2_deg and n_levels are validated on the same footing as ppd/foveal_cpd:
    # without them a default change between training and eval would silently
    # apply a different transform. A key the bundle does not carry is skipped.
    for key, want in (("ppd", fov.ppd), ("foveal_cpd", fov.foveal_cpd),
                      ("e2_deg", fov.e2_deg), ("n_levels", fov.n_levels)):
        got = rec.get(key)
        if got is not None and abs(float(got) - float(want)) > 1e-6:
            raise ValueError(
                f"{ckpt}: heads trained with {key}={got} but eval requested "
                f"{key}={want}; pass matching --ppd/--cpds or another --ckpt-root"
            )
    return fov


def ckpt_arm_mismatch(ckpt: Path, fov) -> str | None:
    """Describe a train-condition / eval-condition mismatch, or None if matched.

    The string is meant to travel into results metadata beside the number, so a
    transfer cell can never later be mistaken for a matched arm. Returns None
    when the bundle has no ``arm`` key or the conditions agree.
    """
    meta_path = ckpt.parent / "metrics.json"
    if not meta_path.exists():
        return None
    arm = json.loads(meta_path.read_text()).get("arm")
    if arm is None:
        return None
    eval_cond = "sharp" if fov is None else (
        "foveated_center" if getattr(fov, "center_fovea", False) else "foveated")
    # Training labels the sharp arm "normal"; eval calls the same condition
    # "sharp". Same thing, two vocabularies — fold them before comparing.
    train_cond = "sharp" if arm == "normal" else arm
    if train_cond == eval_cond:
        return None
    return f"trained={train_cond} eval={eval_cond}"


def set_heads(model, pretrained_state: dict, ckpt: Path | None, device) -> None:
    """Reset to pretrained, then overlay a checkpoint if given (well-defined None).

    Resetting first is what lets a trainable-only checkpoint be equivalent to a
    full-model one: the frozen tensors come from `pretrained_state`, the
    read-out from the checkpoint.
    """
    from .instrument import load_checkpoint_weights

    model.load_state_dict(pretrained_state)
    if ckpt is not None:
        load_checkpoint_weights(model, ckpt, device)


def ckpt_dataset_variant(ckpt: Path) -> str:
    """Dataset variant recorded in a checkpoint bundle's ``metrics.json``.

    An absent key reads as ``"plain"``.
    """
    import json

    try:
        return json.loads(
            (Path(ckpt).parent / "metrics.json").read_text()
        ).get("dataset_variant", "plain")
    except (OSError, ValueError):
        return "unknown"


def eval_arm_checkpoint(model, pretrained_state, stimuli, fixations, indices, device, *,
                        ckpt_root: Path, tag: str, fold: int, fov, seed: int,
                        pretrained: bool = False, epoch: int | None = None,
                        fold_no: int = -1, per_fixation: bool = False,
                        save_maps_for: set[int] | None = None,
                        allow_arm_mismatch: bool = False,
                        expect_dataset_variant: str | None = None):
    """Resolve one (arm, fold) checkpoint, load its heads, and evaluate.

    The ``ckpt_path`` → ``resolve_ckpt_foveation`` → ``set_heads`` →
    ``evaluate.run`` sequence is the contract every checkpoint evaluation
    (sweep table, stratified analysis) relies on; keeping it here means it
    cannot drift between callers. ``pretrained=True`` skips the checkpoint and
    evaluates the pretrained heads. Returns ``(result, ckpt, fov)`` — ``fov``
    as reconciled with the checkpoint's own foveation record.
    """
    from .evaluate import run as evaluate_run

    ckpt = None if pretrained else ckpt_path(ckpt_root, tag, fold, epoch)
    # Protocol guard: with expect_dataset_variant set, a checkpoint trained
    # under another dataset variant is refused rather than silently producing
    # protocol-crossed numbers. Callers that cross deliberately (the
    # robustness matrix via foveation_sweep_table) pass None and record the
    # cross themselves.
    if ckpt is not None and expect_dataset_variant is not None:
        recorded = ckpt_dataset_variant(ckpt)
        if recorded != expect_dataset_variant:
            raise RuntimeError(
                f"{ckpt} was trained on dataset variant '{recorded}' but this "
                f"eval expects '{expect_dataset_variant}'. Point --ckpt-root at "
                "the matching tree, or run the cross deliberately via "
                "foveation_sweep_table, which records it in table.json."
            )
    mismatch = None
    if ckpt is not None:
        # The checkpoint's own record decides how the input is foveated —
        # notably whether the fovea tracks gaze or is pinned to the centre.
        # With allow_arm_mismatch the *request* wins instead, because that is
        # the point of a transfer cell; the mismatch is returned so the caller
        # can stamp it beside the number.
        mismatch = ckpt_arm_mismatch(ckpt, fov) if allow_arm_mismatch else None
        fov = resolve_ckpt_foveation(ckpt, fov, allow_arm_mismatch=allow_arm_mismatch)
    set_heads(model, pretrained_state, ckpt, device)
    result = evaluate_run(model, stimuli, fixations, indices, device, model_name="dg3",
                          fold_no=fold_no, seed=seed, foveation=fov,
                          per_fixation=per_fixation, save_maps_for=save_maps_for)
    if mismatch is not None:
        result["arm_eval_mismatch"] = mismatch
    return result, ckpt, fov


def common_stop_epoch(curves: dict[str, dict[int, float]], *,
                      min_delta: float = 0.001, patience: int = 2) -> int | None:
    """First epoch at which *every* arm has plateaued.

    ``curves`` maps arm tag -> epoch -> mean validation IG across folds. Returns
    the earliest epoch E such that, for every arm, validation IG improved by less
    than ``min_delta`` on each of the last ``patience`` epochs; ``None`` if no
    such epoch exists in the data.

    This is deliberately a *post-hoc reporting-epoch selection*, not in-loop
    early stopping, and the distinction is forced rather than chosen. The rule
    quantifies over all arms simultaneously, but each (arm, fold) trains as an
    independent Slurm job that never sees the other 69 and does not even run at
    the same time — the pump submits in waves under a 12-submitted / 4-running
    QOS cap. There is therefore no moment inside a training job at which the
    condition could be evaluated, so in-loop early stopping is not possible.

    Running every epoch and selecting afterwards reports exactly what the rule
    asks for, keeps the full curves as evidence that the plateau happened, and
    cannot be gamed by one arm stopping earlier than another. What it costs is
    the GPU time an early stop would have saved.

    Applying this to the *test* split would be selection on the reported set, so
    feed it validation curves only.
    """
    if not curves:
        return None
    epochs = sorted(set.intersection(*(set(c) for c in curves.values())))
    for i, e in enumerate(epochs):
        if i < patience:            # need `patience` preceding gaps to judge
            continue
        window = epochs[i - patience:i + 1]
        if all(
            all(c[b] - c[a] < min_delta for a, b in zip(window, window[1:]))
            for c in curves.values()
        ):
            return e
    return None


def val_ig_curves(ckpt_root: Path) -> dict[str, dict[int, float]]:
    """Mean validation IG per (arm tag, epoch), averaged over whatever folds ran.

    Reads ``{ckpt_root}/{tag}/fold*/epoch_*/metrics.json``. Shaped for
    :func:`common_stop_epoch`.
    """
    curves: dict[str, dict[int, list[float]]] = {}
    for tag_dir in sorted(p for p in ckpt_root.glob("*") if p.is_dir()):
        for meta_path in sorted(tag_dir.glob("fold*/epoch_*/metrics.json")):
            meta = json.loads(meta_path.read_text())
            ig = (meta.get("val") or {}).get("ig_bits_per_fix")
            if ig is None:
                continue
            curves.setdefault(tag_dir.name, {}).setdefault(
                int(meta["epoch"]), []).append(float(ig))
    return {tag: {e: float(np.mean(v)) for e, v in sorted(by_epoch.items())}
            for tag, by_epoch in curves.items()}


def image_rgb(stimuli, idx: int) -> np.ndarray:
    from .instrument import ensure_rgb

    return ensure_rgb(np.asarray(stimuli.stimuli[idx]))


def stim_label(stimuli, idx: int) -> str:
    for attr in ("filenames", "stimulus_ids"):
        vals = getattr(stimuli, attr, None)
        if vals is not None:
            try:
                return Path(str(vals[idx])).name
            except Exception:
                pass
    return f"stim {idx}"


def sample_indices(n_stim: int, total: int, seed: int) -> list[int]:
    """Uniformly sample stim indices spanning the corpus, deterministic on seed."""
    if n_stim >= total:
        return list(range(total))
    rng = np.random.RandomState(seed)
    return sorted(rng.choice(total, size=n_stim, replace=False).tolist())


def _points_per_pixel(ax) -> float:
    """Points of drawing per pixel of image, for an equal-aspect image axes.

    The marker circles below are sized in image pixels and their labels in
    points, so the two only stay proportionate if the label is measured off the
    axes. Read straight from the figure geometry rather than from a renderer,
    which is not available until the figure is drawn.
    """
    w_in, h_in = ax.figure.get_size_inches()
    box = ax.get_position()
    x0, x1 = sorted(ax.get_xlim())
    y0, y1 = sorted(ax.get_ylim())
    return 72.0 * min(box.width * w_in / (x1 - x0), box.height * h_in / (y1 - y0))


def draw_scanpath(ax, xs, ys, *, color, marker_scale: float = 1.0) -> None:
    """Numbered scanpath: bright line, white-faced first marker with dark edge
    so lightness (not hue) carries the ordering — readable on blurred frames.

    ``marker_scale`` widens the markers, and with them the digits inside, for a
    panel that prints at a fraction of the page width."""
    from matplotlib.patches import Circle

    ax.plot(xs, ys, "-", color=color, linewidth=1.8, alpha=0.9, zorder=3)
    H = ax.get_ylim()[0]
    radius = max(11, int(0.016 * H)) * marker_scale
    # 1.5 puts the digit at about six tenths of the marker's diameter.
    label_pt = max(2.0, 1.5 * radius * _points_per_pixel(ax))
    for i, (x, y) in enumerate(zip(xs, ys), start=1):
        face = "#FFFFFF" if i == 1 else color
        edge = "#000000" if i == 1 else "#FFFFFF"
        ax.add_patch(Circle((x, y), radius=radius, facecolor=face,
                            edgecolor=edge, linewidth=1.5, zorder=4))
        ax.text(x, y, str(i), ha="center", va="center",
                fontsize=label_pt, fontweight="bold",
                color="#000000" if i == 1 else "#FFFFFF", zorder=5)


def saccade_amplitudes(xs, ys) -> np.ndarray:
    """Euclidean distances between consecutive fixations. Shape (n-1,)."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if len(xs) < 2:
        return np.empty(0)
    return np.hypot(np.diff(xs), np.diff(ys))


def paired_stats(arms: dict, ref: str = "normal") -> dict:
    """Fold- and image-paired arm comparisons from per-image IG records.

    ``arms`` has the shape ``per_image_ig.json`` stores: label -> list of
    per-fold records ``{"fold": k, "stim": [...], "IG_bits": [...]}``. Every
    arm sees the same stimuli in the same fold, so the differences pair
    exactly and image difficulty — the dominant variance component — cancels.
    Raises if an arm's stimuli do not match the reference's (that pairing is
    the whole premise, so a mismatch is an error, not a warning).

    Per non-reference arm:
    - ``fold_paired``: mean ΔIG over per-fold mean differences, with SE and the
      two-standard-error interval around it (``se``/``interval_2se`` are None
      for a single fold), plus the count of negative folds.
    - ``image_paired``: mean ΔIG over per-image differences pooled across
      folds, with its one-sample t statistic.

    Both intervals are mean ± 2 SE, the one convention ch03 stats-plan sets.
    Artefacts may carry the interval under ``interval_2se`` or, as a Student-t
    95 % interval, under ``ci95``; consumers compute the interval from ``mean``
    and ``se`` rather than reading it, so both give the same answer.
    """
    ref_by_fold = {r["fold"]: dict(zip(r["stim"], r["IG_bits"])) for r in arms[ref]}
    out = {}
    for label, records in arms.items():
        if label == ref:
            continue
        fold_diffs, image_diffs = [], []
        for r in records:
            ref_ig = ref_by_fold[r["fold"]]
            if set(r["stim"]) != set(ref_ig):
                raise ValueError(
                    f"{label} fold {r['fold']}: stimuli do not match {ref}'s — "
                    "arms are not paired")
            diffs = [ig - ref_ig[s] for s, ig in zip(r["stim"], r["IG_bits"])]
            fold_diffs.append(float(np.mean(diffs)))
            image_diffs.extend(diffs)
        k = len(fold_diffs)
        fold_mean = float(np.mean(fold_diffs))
        if k > 1:
            se = float(np.std(fold_diffs, ddof=1) / np.sqrt(k))
            interval = [fold_mean - 2 * se, fold_mean + 2 * se]
        else:
            se, interval = None, None
        n = len(image_diffs)
        img_mean = float(np.mean(image_diffs))
        img_se = float(np.std(image_diffs, ddof=1) / np.sqrt(n))
        img_t = float(img_mean / img_se)
        out[label] = {
            "fold_paired": {"mean": fold_mean, "se": se, "interval_2se": interval,
                            "n_folds": k,
                            "negative_folds": int(sum(d < 0 for d in fold_diffs))},
            # Reported with an interval, not just a t: a bare t invites reading
            # significance off a number whose scale is unstated. Treats the 1003
            # image differences as independent, which they are not — subject is a
            # random effect crossed with image and does not appear here at all —
            # so this interval is anti-conservative by construction. `fold_paired`
            # absorbs every within-fold correlation and is the one claims use.
            "image_paired": {"mean": img_mean, "se": img_se, "t": img_t,
                             "interval_2se": [img_mean - 2 * img_se,
                                              img_mean + 2 * img_se],
                             "n_images": n,
                             "independence": "images assumed independent; "
                                             "subject clustering ignored"},
        }
    return out


def turn_angles_deg(xs, ys) -> np.ndarray:
    """Angles between consecutive saccade vectors in degrees. Shape (n-2,).

    Saccades with zero length are skipped (an exactly-on-prev refixation is
    ill-defined as a vector).
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if len(xs) < 3:
        return np.empty(0)
    dx = np.diff(xs)
    dy = np.diff(ys)
    norm = np.hypot(dx, dy)
    pairs = []
    for i in range(len(norm) - 1):
        n1, n2 = norm[i], norm[i + 1]
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        cos = (dx[i] * dx[i + 1] + dy[i] * dy[i + 1]) / (n1 * n2)
        pairs.append(float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))))
    return np.array(pairs, dtype=float)
