"""Image-stratified 10-fold CV split for MIT1003.

Wraps ``pysaliency.dataset_config.{train,validation,test}_split`` for all 10
folds, asserts disjointness at split time, and writes a JSON sidecar keyed
by stimulus_id (not the subset-relative integer index) so the artefact
survives reload across processes.

These are the same pysaliency calls used by the published DG3 training
protocol, so the partition matches the one Kümmerer 2022 reports against.

Caveat on the seed argument: ``pysaliency`` internally hardcodes
``np.random.RandomState(42)`` when building the cross-validation split,
so passing a different ``--seed`` here does NOT change the partition. We
still record the requested seed in the sidecar for traceability.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from importlib.metadata import version as _pkg_version
from pathlib import Path

from pysaliency.dataset_config import test_split, train_split, validation_split

from .paths import RESULTS, load_mit1003


def build_mit1003_10fold(
    seed: int = 42,
    out: Path | None = None,
    corpus: tuple | None = None,
) -> dict:
    """Build all 10 image-stratified folds; assert disjointness; persist JSON.

    Returns the spec dict (same content as the written JSON sidecar).

    ``corpus`` optionally supplies an already-loaded ``(stimuli, fixations)``
    pair. Without it every call re-reads the whole of MIT1003 from HDF5, which
    is pure waste for a caller that already holds the data — and enough of it
    in one process to exhaust memory (four repeated calls is what killed the
    cv-split test suite).
    """
    if out is None:
        out = RESULTS / "cv_splits" / f"mit1003-10fold-seed{seed}.json"

    stimuli, fixations = corpus if corpus is not None else load_mit1003()

    folds: list[dict] = []
    for k in range(10):
        train_s, _ = train_split(
            stimuli, fixations,
            crossval_folds=10, fold_no=k,
            val_folds=1, test_folds=1, random=True,
        )
        val_s, _ = validation_split(
            stimuli, fixations,
            crossval_folds=10, fold_no=k,
            val_folds=1, test_folds=1, random=True,
        )
        test_s, _ = test_split(
            stimuli, fixations,
            crossval_folds=10, fold_no=k,
            val_folds=1, test_folds=1, random=True,
        )

        train_ids = list(train_s.stimulus_ids)
        val_ids = list(val_s.stimulus_ids)
        test_ids = list(test_s.stimulus_ids)

        train_set = set(train_ids)
        val_set = set(val_ids)
        test_set = set(test_ids)

        # Assert disjointness at split time so any future change in
        # pysaliency's partitioning logic fails here rather than silently
        # leaking test images into training.
        assert train_set.isdisjoint(test_set), (
            f"fold {k}: train ∩ test stimulus_ids non-empty"
        )
        assert train_set.isdisjoint(val_set), (
            f"fold {k}: train ∩ val stimulus_ids non-empty"
        )
        assert val_set.isdisjoint(test_set), (
            f"fold {k}: val ∩ test stimulus_ids non-empty"
        )

        folds.append({
            "fold_no": k,
            "n_train_stimuli": len(train_ids),
            "n_val_stimuli": len(val_ids),
            "n_test_stimuli": len(test_ids),
            "train_stimulus_ids": train_ids,
            "val_stimulus_ids": val_ids,
            "test_stimulus_ids": test_ids,
        })

    spec = {
        "$schema_version": 1,
        "dataset": "MIT1003",
        "n_stimuli": len(stimuli),
        "crossval_folds": 10,
        "val_folds": 1,
        "test_folds": 1,
        "random": True,
        "seed": int(seed),
        "seed_origin": (
            "pysaliency.filter_datasets._get_crossval_split hardcoded "
            "RandomState(42)"
        ),
        "pysaliency_version": _pkg_version("pysaliency"),
        "created": datetime.now(timezone.utc).isoformat(),
        "folds": folds,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, indent=2))
    return spec


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build image-stratified 10-fold CV split for MIT1003 (BASE-04).",
    )
    ap.add_argument("--seed", type=int, default=42,
                    help="seed value recorded in the sidecar (note: pysaliency "
                         "hardcodes RandomState(42) regardless)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output JSON path (default: results/cv_splits/...)")
    args = ap.parse_args()

    spec = build_mit1003_10fold(seed=args.seed, out=args.out)
    out_path = args.out if args.out else (
        RESULTS / "cv_splits" / f"mit1003-10fold-seed{args.seed}.json"
    )
    print(f"wrote {out_path}")
    for f in spec["folds"]:
        print(
            f"  fold {f['fold_no']:2d}: "
            f"train={f['n_train_stimuli']:4d}  "
            f"val={f['n_val_stimuli']:4d}  "
            f"test={f['n_test_stimuli']:4d}"
        )


if __name__ == "__main__":
    main()
