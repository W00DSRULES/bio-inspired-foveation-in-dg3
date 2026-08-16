"""Dataset adapter for MIT1003 training / evaluation.

Wraps ``pysaliency.get_mit1003`` and the 10-fold CV split so a future
corpus swap is a *class change* on the caller side, not a rewrite of
``foveated_train.py``. The adapter exposes the small set of primitives
``foveated_train.py`` needs:

  - ``stim_indices`` — the stimulus indices for the given (fold, split)
  - ``scanpaths_for_stim(stim_idx)`` returning a list of ``(xs, ys, subj)``
    tuples (the same shape as :func:`human_scanpaths.all_human_scanpaths`)
  - ``image_and_centerbias(stim_idx)`` returning the ``(H × W × 3 uint8,
    H × W log-density float32)`` pair used by every forward pass in the
    project

The class deliberately does **not** mimic the upstream
``deepgaze_pytorch.data`` API — that one carries an ``lmdb`` /
``IPython`` / ``tensorboard`` dependency chain we do not want. The
primitives above are sufficient for the per-image foveated training
loop in ``foveated_train.py``.

Swapping to a different corpus is a class change: subclass
:class:`MIT1003TrainingDataset` (or write a sibling), override
``_load_corpus`` and the stimulus-id mapping, and the rest of the
foveated training loop is unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .centerbias import load_centerbias_for_image
from .human_scanpaths import all_human_scanpaths
from .paths import DATA_ROOT as DEFAULT_DATA_ROOT
from .paths import RESULTS, load_mit1003_variant


DEFAULT_CV_SPLIT = RESULTS / "cv_splits" / "mit1003-10fold-seed42.json"


class MIT1003TrainingDataset:
    """MIT1003 + 10-fold CV adapter for the training loop.

    Construction is cheap (paths + JSON load); the actual stimuli / fixation
    objects are loaded lazily on first access and then cached.

    Parameters
    ----------
    fold_no:
        Which of the 10 CV folds to use (0..9).
    split:
        One of ``"train"``, ``"val"``, ``"test"`` — picks which
        stimulus-id list from that fold to expose as
        :attr:`stim_indices`.
    dataset_variant:
        ``"plain"`` (default) loads the standard MIT1003, where the forced
        central start fixation was dropped at build time, so scanpath index 0
        is the first free fixation. ``"initial"`` loads
        ``MIT1003_initial_fix_consistent`` (built by
        ``fetch_mit1003.py --with-initial``), where index 0 is the central
        fixation — a ``start_fixation=1`` consumer then targets every free
        fixation with the centre as history, the protocol of the DG3 paper.
        Fold membership is identical across variants: the split maps stimulus
        ids, and both variants carry the same 1003 stimuli in the same order.
    data_root:
        MIT1003 corpus root (the directory under which ``MIT1003/`` lives).
        Defaults to :data:`tez_deepgaze.paths.DATA_ROOT`
        (``<repo>/data/mit1003``, overridable via ``TEZ_DATA_ROOT``).
    cv_split_path:
        Path to the JSON produced by
        :mod:`tez_deepgaze.cv_split`. Defaults to the canonical 10-fold
        split shipped with the repo.
    stimuli / fixations:
        Optional already-loaded corpus objects. When given, the lazy
        ``load_mit1003`` is skipped — callers that iterate several folds
        can load the corpus once and share it.
    """

    def __init__(
        self,
        fold_no: int,
        split: str = "train",
        *,
        dataset_variant: str = "plain",
        data_root: Path | str = DEFAULT_DATA_ROOT,
        cv_split_path: Path | str = DEFAULT_CV_SPLIT,
        stimuli=None,
        fixations=None,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train|val|test, got {split!r}")
        if dataset_variant not in ("plain", "initial"):
            raise ValueError(
                f"dataset_variant must be plain|initial, got {dataset_variant!r}"
            )
        self.fold_no = int(fold_no)
        self.split = split
        self.dataset_variant = dataset_variant
        self.data_root = Path(data_root)
        self.cv_split_path = Path(cv_split_path)

        self._stimuli = stimuli
        self._fixations = fixations
        self._stim_indices: list[int] | None = None
        self._id_to_index: dict[str, int] | None = None

    # ----- corpus access (lazy) -----

    def _load_corpus(self):
        if self._stimuli is None or self._fixations is None:
            self._stimuli, self._fixations = load_mit1003_variant(
                self.dataset_variant, self.data_root)
        return self._stimuli, self._fixations

    @property
    def stimuli(self):
        return self._load_corpus()[0]

    @property
    def fixations(self):
        return self._load_corpus()[1]

    # ----- CV split → list of stimulus indices -----

    def _build_id_index(self) -> dict[str, int]:
        if self._id_to_index is None:
            stimuli = self.stimuli
            ids = getattr(stimuli, "stimulus_ids", None)
            if ids is None:
                raise RuntimeError(
                    "stimuli object exposes no `stimulus_ids`; cannot map CV "
                    "split hashes to stimulus indices"
                )
            self._id_to_index = {str(s): i for i, s in enumerate(ids)}
        return self._id_to_index

    @property
    def stim_indices(self) -> list[int]:
        """Stimulus indices for this (fold, split)."""
        if self._stim_indices is None:
            spec = json.loads(self.cv_split_path.read_text())
            # The CV split file uses `crossval_folds` as a *count* (e.g. 10)
            # and exposes the list of fold dicts under `folds`.
            folds = spec.get("folds")
            if not isinstance(folds, list):
                raise RuntimeError(
                    f"CV split file {self.cv_split_path} has no 'folds' list"
                )
            fold = next((f for f in folds if int(f["fold_no"]) == self.fold_no), None)
            if fold is None:
                raise RuntimeError(
                    f"fold {self.fold_no} not found in {self.cv_split_path}"
                )
            id_key = f"{self.split}_stimulus_ids"
            if id_key not in fold:
                raise RuntimeError(
                    f"fold {self.fold_no} has no {id_key!r} entry"
                )
            id_to_idx = self._build_id_index()
            indices: list[int] = []
            for sid in fold[id_key]:
                if sid in id_to_idx:
                    indices.append(id_to_idx[sid])
                else:
                    raise RuntimeError(
                        f"stimulus id {sid!r} from fold not found in MIT1003"
                    )
            self._stim_indices = sorted(indices)
        return list(self._stim_indices)

    # ----- per-stimulus accessors -----

    def scanpaths_for_stim(
        self, stim_idx: int,
    ) -> list[tuple[np.ndarray, np.ndarray, int]]:
        """All subjects' scanpaths on this stimulus, NaN-stripped."""
        return all_human_scanpaths(self.fixations, stim_idx)

    def image_and_centerbias(self, stim_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(image, centerbias)`` for ``stim_idx``.

        Image is ``H × W × 3 uint8``; centerbias is ``H × W float32``
        log-density (proper, summed to 1).
        """
        from .instrument import ensure_rgb

        img = ensure_rgb(np.asarray(self.stimuli.stimuli[stim_idx]))
        H, W = img.shape[:2]
        cb = load_centerbias_for_image(H, W)
        return img, cb

    def __len__(self) -> int:
        return len(self.stim_indices)

    def __repr__(self) -> str:
        return (
            f"MIT1003TrainingDataset(fold_no={self.fold_no}, "
            f"split={self.split!r}, variant={self.dataset_variant!r}, "
            f"n_stim={len(self)})"
        )
