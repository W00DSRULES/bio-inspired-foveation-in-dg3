"""Shared env-aware paths for the package.

Single source of truth for the repo root, the MIT1003 data root
(``TEZ_DATA_ROOT``), the centerbias cache, and the torch-hub DG3
checkpoint path (``TORCH_HOME``). Import from here instead of
re-deriving paths per module, so training and evaluation cannot
disagree about where data lives.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# MIT1003 dataset directory. Defaults to ``<repo>/data/mit1003``; set
# ``TEZ_DATA_ROOT`` to point at a different filesystem (e.g.
# ``/work/dldevel/itez/Tez/data/mit1003`` on Goethe-NHR).
DATA_ROOT = Path(os.environ.get("TEZ_DATA_ROOT", REPO_ROOT / "data" / "mit1003"))

RESULTS = REPO_ROOT / "results"

# Cache location for the centerbias .npy — a sibling of the MIT1003
# directory: ``<repo>/data/centerbias_mit1003.npy`` by default, or
# ``$TEZ_DATA_ROOT/../centerbias_mit1003.npy`` when the env var is set.
CENTERBIAS_CACHE = DATA_ROOT.parent / "centerbias_mit1003.npy"


def deepgaze3_weights_path(filename: str = "deepgaze3.pth") -> Path:
    """Torch-hub cache path for the pretrained DG3 checkpoint.

    Honours ``TORCH_HOME`` (the cluster setup writes to
    ``$WORK/torch-cache``); falls back to ``~/.cache/torch``, matching
    ``torch.utils.model_zoo.load_url``. Existence is not checked.
    """
    return (
        Path(os.environ.get("TORCH_HOME", "~/.cache/torch")).expanduser()
        / "hub" / "checkpoints" / filename
    )


def load_mit1003(location: Path | str = DATA_ROOT):
    """Load MIT1003 (stimuli, fixations) from ``location`` (default DATA_ROOT)."""
    import pysaliency  # deferred: keep this module cheap to import

    return pysaliency.get_mit1003(location=str(location))


def load_mit1003_variant(dataset_variant: str, location: Path | str = DATA_ROOT):
    """Load a MIT1003 dataset variant: ``"plain"`` or ``"initial"``.

    ``"plain"`` is :func:`load_mit1003` — the forced central start fixation was
    dropped at build time, so scanpath index 0 is the first free fixation.
    ``"initial"`` loads ``MIT1003_initial_fix_consistent`` (built by
    ``fetch_mit1003.py --with-initial``), where index 0 is the central
    fixation, so a ``start_fixation=1`` consumer targets every free fixation
    with the centre as history — the protocol of the DG3 paper. This is the
    single loader every entry point's ``--dataset-variant`` flag routes
    through, so the variants cannot drift apart between scripts.
    """
    if dataset_variant == "plain":
        return load_mit1003(location)
    if dataset_variant == "initial":
        import pysaliency.external_datasets.mit as psmit

        return psmit.get_mit1003_with_initial_fixation(
            location=str(location), replace_initial_invalid_fixations=True,
        )
    raise ValueError(f"dataset_variant must be plain|initial, got {dataset_variant!r}")
