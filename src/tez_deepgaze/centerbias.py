"""Download and load the MIT1003 centerbias used by DeepGaze III."""
import os
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
from scipy.ndimage import zoom
from scipy.special import logsumexp

# Cache location (TEZ_DATA_ROOT-aware) is defined in paths.py so every
# module records the SHA of the same file this module loads.
from .paths import CENTERBIAS_CACHE as CACHE

URL = "https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/centerbias_mit1003.npy"


def ensure_centerbias() -> Path:
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading centerbias to {CACHE}")
        # Download to a per-process temp name and rename into place: rename is
        # atomic, so concurrent cold-cache processes (e.g. per-fold Slurm jobs
        # starting together) can never observe a half-written file.
        tmp = CACHE.with_name(f"{CACHE.name}.tmp.{os.getpid()}")
        try:
            urlretrieve(URL, tmp)
            os.replace(tmp, CACHE)
        finally:
            tmp.unlink(missing_ok=True)
    return CACHE


def load_centerbias_for_image(height: int, width: int) -> np.ndarray:
    """Return a log-density centerbias matched to the given image size."""
    template = np.load(ensure_centerbias())
    cb = zoom(
        template,
        (height / template.shape[0], width / template.shape[1]),
        order=0,
        mode="nearest",
    )
    cb -= logsumexp(cb)
    return cb
