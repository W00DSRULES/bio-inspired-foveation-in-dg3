"""Reproduce DeepGaze III baseline log-likelihood on MIT1003.

Computes per-fixation log-likelihood (in bits) averaged over the
full MIT1003 scanpath corpus.
Saves results to results/baselines/baseline.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from importlib.metadata import version as _pkg_version
from pathlib import Path

import numpy as np
import torch

import deepgaze_pytorch

from .device import pick_device, to_device
from .ig import LOG2, compute_ig_bits
from .paths import CENTERBIAS_CACHE, REPO_ROOT, RESULTS, deepgaze3_weights_path, load_mit1003

BASELINES = RESULTS / "baselines"

# Pretrained DG3 checkpoint path used for the reproducibility SHA-256.
# ``deepgaze3_weights_path`` honours ``TORCH_HOME`` so the SHA stays accurate
# when the cache is moved off ``$HOME`` (the default cluster setup writes to
# ``$WORK/torch-cache``).
WEIGHTS_PATH = deepgaze3_weights_path()
UV_LOCK_PATH = REPO_ROOT / "uv.lock"

# Schema contract for results/baselines/baseline.json. We assert this key set is fully
# populated before writing, so callers that depend on specific fields
# cannot silently break if someone forgets a field. The "notes" key carries
# the per-image IG approximation caveat described inside _run_full_corpus.
BASELINE_JSON_REQUIRED_KEYS = frozenset({
    "n_fixations_evaluated", "ll_per_fixation_nats", "ll_per_fixation_bits", "device",
    "n_images", "ll_bits_mean_per_image", "ll_bits_se_per_image",
    "ig_bits_corpus_pooled", "ig_bits_per_image_mean", "ig_bits_per_image_se",
    "nss_mean", "nss_se", "auc_mean", "auc_se",
    "seed", "cudnn_deterministic",
    "weights_sha256", "centerbias_sha256", "uv_lock_sha256",
    "pysaliency_version", "deepgaze_pytorch_commit",
    "notes",
})


def _nearest_sample(log_density: np.ndarray, x: float, y: float) -> float:
    """Return log_density(y, x) via round-half-even nearest-pixel lookup.

    pysaliency's own readers truncate the sub-pixel coordinate instead of
    rounding; the two conventions pick a different pixel on ~14% of MIT1003
    fixations (mean difference +0.003 bits/fix). Training and every evaluator
    in this repo share THIS lookup, so all internal comparisons are
    convention-consistent.
    """
    h, w = log_density.shape
    xi = int(np.clip(round(x), 0, w - 1))
    yi = int(np.clip(round(y), 0, h - 1))
    return float(log_density[yi, xi])


def _run_full_corpus(args, device, stimuli, fixations, model) -> None:
    """Full-corpus evaluation path: delegate to evaluate.run, subtract the
    centerbias LL to get IG, attach reproducibility metadata, and write the
    full baseline.json schema."""
    # Lazy import breaks a circular import: evaluate.py imports
    # `_nearest_sample` from this module at module-load time, so a
    # top-level `from .evaluate import run` here would deadlock.
    from .evaluate import run as evaluate_run

    # Determinism flags must be set BEFORE the eval forward passes run.
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except RuntimeError:
        # Older torch versions do not accept warn_only; the cudnn flag above
        # is the primary determinism guarantee, so we tolerate this failure.
        pass
    torch.manual_seed(args.seed)

    # IG = LL_model - LL_centerbias requires a precomputed centerbias LL
    # number on the same corpus. Fail loudly if that file is missing rather
    # than silently reporting IG = 0.
    cb_json_path = BASELINES / "centerbias.json"
    if not cb_json_path.exists():
        raise FileNotFoundError(
            f"{cb_json_path} not found. This file is checked into the repo and "
            "should not normally be missing; restore it from git if it was "
            "deleted by mistake."
        )
    ll_centerbias_bits = float(json.loads(cb_json_path.read_text())["ll_per_fixation_bits"])

    # Full-corpus eval through evaluate.run.
    train_x = fixations.train_xs if hasattr(fixations, "train_xs") else None
    if train_x is None:
        raise RuntimeError("pysaliency fixations did not expose train_xs; check version")
    indices = list(range(len(train_x)))
    if args.subsample > 0:
        # --subsample is honoured even on the full-corpus path so we can do
        # quick smoke runs that still exercise the full output schema.
        rng = random.Random(args.seed)
        rng.shuffle(indices)
        indices = indices[: args.subsample]
    print(f"evaluating {len(indices)} scanpaths via evaluate.run")

    result_eval = evaluate_run(
        model=model,
        stimuli=stimuli,
        fixations=fixations,
        indices=indices,
        device=device,
        model_name="DeepGazeIII-pretrained",
        fold_no=-1,
        seed=args.seed,
        start_fixation=1,
    )

    # Corpus-pooled IG (the convention used for paper comparisons,
    # following the Kümmerer 2022 Table 2 column layout).
    ll_dg3_corpus_bits = result_eval["ll_per_fixation_nats"] / LOG2
    ig_bits_corpus_pooled = compute_ig_bits(ll_dg3_corpus_bits, ll_centerbias_bits)

    # Cross-check: the centerbias LL that evaluate.run computes in-loop
    # (per-fixation, nearest-pixel, per stimulus) against the external
    # centerbias.json used for the headline IG above. They are computed the
    # same way and must agree; a drift signals that the two centerbias
    # sources have diverged.
    ll_cb_bits_inloop = result_eval["ll_centerbias_bits_pooled"]
    cb_drift = abs(ll_cb_bits_inloop - ll_centerbias_bits)
    print(
        f"centerbias LL cross-check: external {ll_centerbias_bits:.4f} bits/fix, "
        f"in-loop {ll_cb_bits_inloop:.4f} bits/fix (drift {cb_drift:.4f})"
    )
    if cb_drift > 0.01:
        print(
            "WARNING: in-loop and external centerbias LL differ by more than "
            "0.01 bits/fix — per-image IG and corpus-pooled IG use different "
            "centerbias references."
        )

    # Per-image IG over the centerbias, image-stratified with per-image SE.
    # evaluate.run computes this exactly: for each stimulus it pools all its
    # fixations and takes (raw model LL − raw centerbias LL) in bits, then the
    # mean ± SE is over stimuli. Unlike the corpus-pooled figure this carries
    # a real per-image standard error for ablation tables.
    ig_bits_per_image_mean = result_eval["IG_bits_mean"]
    ig_bits_per_image_se = result_eval["IG_bits_se"]

    # Reproducibility SHAs.
    def _sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING"

    weights_sha = _sha(WEIGHTS_PATH)
    centerbias_sha = _sha(CENTERBIAS_CACHE)
    uv_lock_sha = _sha(UV_LOCK_PATH)

    # Dependency versions.
    try:
        pysaliency_version = _pkg_version("pysaliency")
    except Exception:
        pysaliency_version = "unknown"
    try:
        deepgaze_pytorch_commit = _pkg_version("deepgaze-pytorch")
    except Exception:
        deepgaze_pytorch_commit = "unknown"

    # Output schema. The first block is the flat schema other tooling reads
    # — these keys must not be renamed or removed. The rest is schema
    # version 2 with per-image stats, IG, NSS, AUC, and reproducibility
    # metadata.
    result = {
        # Flat keys read by other tooling — do not rename.
        "n_fixations_evaluated": result_eval["n_fixations"],
        "ll_per_fixation_nats": result_eval["ll_per_fixation_nats"],
        "ll_per_fixation_bits": ll_dg3_corpus_bits,
        "device": str(device),
        # Current schema (version 2):
        "$schema_version": 2,
        "n_scanpaths_evaluated": result_eval["n_scanpaths_evaluated"],
        "n_images": result_eval["n_images"],
        "ll_bits_mean_per_image": result_eval["LL_raw_bits_mean"],
        "ll_bits_se_per_image": result_eval["LL_raw_bits_se"],
        "ig_bits_corpus_pooled": ig_bits_corpus_pooled,
        "ig_bits_per_image_mean": ig_bits_per_image_mean,
        "ig_bits_per_image_se": ig_bits_per_image_se,
        "nss_mean": result_eval["NSS_mean"],
        "nss_se": result_eval["NSS_se"],
        "auc_mean": result_eval["AUC_mean"],
        "auc_se": result_eval["AUC_se"],
        "seed": args.seed,
        "cudnn_deterministic": True,
        "weights_sha256": weights_sha,
        "centerbias_sha256": centerbias_sha,
        "uv_lock_sha256": uv_lock_sha,
        "pysaliency_version": pysaliency_version,
        "deepgaze_pytorch_commit": deepgaze_pytorch_commit,
        "created": datetime.now(timezone.utc).isoformat(),
        "ll_centerbias_bits_corpus_pooled": ll_centerbias_bits,
        "kumerer_2022_target_ig_bits": 1.536,
        "model": result_eval["model"],
        "fold_no": result_eval["fold_no"],
        "ll_centerbias_bits_inloop": ll_cb_bits_inloop,
        "notes": {
            "aggregation": (
                "All *_per_image / nss / auc statistics are image-stratified: "
                "fixations are pooled per stimulus (n_images stimuli) and the "
                "mean ± SE is taken across stimuli, following Kümmerer 2022. "
                "ll_bits_mean_per_image and ig_bits_per_image_* are raw "
                "log-density bits (model, and model − centerbias), matching "
                "the corpus-pooled ll_per_fixation_bits convention. "
                "ig_bits_corpus_pooled is the fixation-pooled figure used for "
                "paper comparisons; ig_bits_per_image_mean is the stimulus "
                "mean with a real per-stimulus SE and differs slightly because "
                "stimuli carry unequal fixation counts."
            ),
            "nss_convention": (
                "nss_mean is NSS on the history-conditioned log-density, so it "
                "is a scanpath-conditional NSS and is not directly comparable "
                "to single-map spatial NSS reported elsewhere. It comes from "
                "evaluate.nss, not deepgaze_pytorch.metrics.nss: the released "
                "routine unpacks torch.std_mean the wrong way round and "
                "returns about 3x the correct value."
            ),
        },
    }

    # Fail fast if any required key was omitted, before the file is written.
    missing = BASELINE_JSON_REQUIRED_KEYS - set(result.keys())
    assert not missing, f"result dict missing required keys: {missing}"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")
    print(f"IG (corpus-pooled): {ig_bits_corpus_pooled:.4f} bits/fix vs Kümmerer 2022 1.536 ± 0.010")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsample", type=int, default=0,
                    help="evaluate only N scanpaths (0 = all); quick smoke runs "
                         "still exercise the full output schema")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=BASELINES / "baseline.json",
                    help="output JSON path (default: results/baselines/baseline.json)")
    ap.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None,
                    help="override the auto-selected device "
                         "(use 'cuda' on GPU servers)")
    args = ap.parse_args()

    device = (
        torch.device(args.device) if args.device is not None else pick_device()
    )
    print(f"device: {device}")

    stimuli, fixations = load_mit1003()
    print(f"stimuli: {len(stimuli)}, fixations: {len(fixations.x)}")

    model = to_device(deepgaze_pytorch.DeepGazeIII(pretrained=True), device).eval()
    _run_full_corpus(args, device, stimuli, fixations, model)


if __name__ == "__main__":
    main()
