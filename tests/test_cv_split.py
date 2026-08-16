"""Tests for `tez_deepgaze.cv_split.build_mit1003_10fold`.

One call to ``build_mit1003_10fold`` peaks at ~9 GB: it runs 30 pysaliency split
operations (10 folds x train/val/test), each materialising subsets of the
1003-image corpus. Loading the corpus is cheap by comparison (~0.1 GB, lazy
HDF5) — it is the splitting that costs. Building once per test would mean five
builds in one process, more than a 17 GB machine holds.

These tests assert different properties of the *same* artifact, so the build is
a session fixture and happens once. Only the determinism check needs a second,
independent build, and it compares against the fixture rather than doing two of
its own.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tez_deepgaze.cv_split import build_mit1003_10fold

# Loads the corpus and peaks at ~9 GB building the split (see module docstring).
pytestmark = pytest.mark.heavy


@pytest.fixture(scope="session")
def split_spec(tmp_path_factory, mit1003_corpus) -> tuple[dict, Path]:
    """Build the 10-fold split once; return (returned spec, written path)."""
    out = tmp_path_factory.mktemp("cv") / "split.json"
    spec = build_mit1003_10fold(seed=42, out=out, corpus=mit1003_corpus)
    return spec, out


def test_disjoint_stimulus_ids(split_spec) -> None:
    """assert set(train) & set(test) == set() for every fold."""
    _, out = split_spec
    spec = json.loads(out.read_text())
    for fold in spec["folds"]:
        train_ids = set(fold["train_stimulus_ids"])
        val_ids = set(fold["val_stimulus_ids"])
        test_ids = set(fold["test_stimulus_ids"])
        assert train_ids.isdisjoint(test_ids), f"fold {fold['fold_no']} train ∩ test"
        assert train_ids.isdisjoint(val_ids), f"fold {fold['fold_no']} train ∩ val"
        assert val_ids.isdisjoint(test_ids), f"fold {fold['fold_no']} val ∩ test"


def test_determinism_seed42(tmp_path: Path, split_spec, mit1003_corpus) -> None:
    """Same seed → same fold-0 train_stimulus_ids across independent builds."""
    spec_a, _ = split_spec
    spec_b = build_mit1003_10fold(seed=42, out=tmp_path / "b.json", corpus=mit1003_corpus)
    assert spec_a["folds"][0]["train_stimulus_ids"] == spec_b["folds"][0]["train_stimulus_ids"]


def test_persistence_roundtrip(split_spec) -> None:
    """Write JSON → re-parse → identical fold list."""
    spec_returned, out = split_spec
    spec_loaded = json.loads(out.read_text())
    assert spec_returned["folds"] == spec_loaded["folds"]


def test_fold_count_is_ten(split_spec) -> None:
    """10-fold CV produces exactly 10 folds."""
    _, out = split_spec
    spec = json.loads(out.read_text())
    assert len(spec["folds"]) == 10
    assert spec["crossval_folds"] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
