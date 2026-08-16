"""Quick smoke test: load DG3, load MIT1003, run 1 forward pass, time it."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import deepgaze_pytorch
import pysaliency

from tez_deepgaze.centerbias import load_centerbias_for_image
from tez_deepgaze.device import pick_device, to_device
from tez_deepgaze.instrument import compute_log_density, ensure_rgb

DATA = Path(__file__).resolve().parents[1] / "data" / "mit1003"


def main() -> None:
    device = pick_device()
    print(f"device: {device}")

    stimuli, fixations = pysaliency.get_mit1003(location=str(DATA))
    print(f"stimuli: {len(stimuli)}, fixations: {len(fixations.x)}")
    print(f"FixationTrains attrs: train_xs={hasattr(fixations, 'train_xs')} "
          f"scanpaths={hasattr(fixations, 'scanpaths')}")

    # pick first scanpath
    xs = fixations.train_xs[0]
    ys = fixations.train_ys[0]
    stim_idx = int(fixations.train_ns[0])
    image = np.asarray(stimuli.stimuli[stim_idx])
    print(f"first scanpath: stim={stim_idx}, n_fix={len(xs)}, img shape={image.shape}")

    print("loading DeepGazeIII ...")
    t0 = time.time()
    model = to_device(deepgaze_pytorch.DeepGazeIII(pretrained=True), device).eval()
    print(f"  model load: {time.time()-t0:.1f}s")
    print(f"  included_fixations: {model.included_fixations}")

    image = ensure_rgb(image)
    cb = load_centerbias_for_image(image.shape[0], image.shape[1])
    print(f"centerbias shape: {cb.shape}")

    hist_x = list(xs[:3])
    hist_y = list(ys[:3])
    t0 = time.time()
    log_d = compute_log_density(model, image, cb, hist_x, hist_y, device)
    print(f"  forward pass 1: {time.time()-t0:.2f}s, log_d shape={log_d.shape}, "
          f"logsumexp={float(np.logaddexp.reduce(log_d.ravel())):.3f}")

    # second pass to see warm-cache time
    t0 = time.time()
    log_d = compute_log_density(model, image, cb, hist_x, hist_y, device)
    print(f"  forward pass 2: {time.time()-t0:.2f}s (warm)")


if __name__ == "__main__":
    main()
