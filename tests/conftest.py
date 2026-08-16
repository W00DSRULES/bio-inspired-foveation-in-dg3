"""Shared pytest fixtures for the evaluator + CV split tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import deepgaze_pytorch
import pysaliency

from tez_deepgaze.device import pick_device, to_device

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "mit1003"


@pytest.fixture(scope="session")
def device() -> torch.device:
    return pick_device()


@pytest.fixture(scope="session")
def dg3_model(device):
    """Pretrained DeepGaze III, loaded once per pytest session.

    Uses the same loader as baseline.py:51 — ensures the parity test
    compares two invocations of identical weights.
    """
    return to_device(deepgaze_pytorch.DeepGazeIII(pretrained=True), device).eval()


@pytest.fixture(scope="session")
def mit1003_corpus():
    """The whole corpus, loaded once per session.

    Each load reads stimuli.hdf5 + fixations.hdf5 in full; repeating it inside
    one process exhausts memory. Anything needing the corpus takes this.
    """
    return pysaliency.get_mit1003(location=str(DATA_ROOT))


@pytest.fixture(scope="session")
def mit1003_subsample(mit1003_corpus):
    """Stimuli + fixations + a fixed list of N scanpath indices (seed=0).

    Derived from ``mit1003_corpus`` rather than loading again — one corpus
    load per process, as that fixture's docstring requires.
    """
    stimuli, fixations = mit1003_corpus
    rng = np.random.RandomState(0)
    indices = list(range(len(fixations.train_xs)))
    rng.shuffle(indices)
    return stimuli, fixations, indices[:10]  # 10 scanpaths is enough for parity smoke
