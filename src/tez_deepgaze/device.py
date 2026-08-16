"""Device selection: prefer CUDA, then MPS (Apple Silicon), else CPU."""
import os

import torch


def pick_device() -> torch.device:
    """Pick the best available device.

    If ``TEZ_REQUIRE_CUDA=1`` is set (cluster GPU jobs do this), raise instead
    of silently falling back to CPU when CUDA is unavailable. On Goethe-NHR the
    ``gpu``/``gpu_test``/``sgpu`` partitions are AMD/ROCm nodes that a CUDA-built
    torch cannot use — without this guard a misrouted job runs ~7h/epoch on CPU
    before anyone notices. Only the NVIDIA ``gpu2`` partition satisfies this.
    """
    cuda_ok = torch.cuda.is_available()
    if os.environ.get("TEZ_REQUIRE_CUDA") == "1" and not cuda_ok:
        raise RuntimeError(
            "TEZ_REQUIRE_CUDA=1 but torch.cuda.is_available() is False. "
            "This GPU job was placed on a node with no usable NVIDIA GPU "
            "(on Goethe-NHR, use the NVIDIA 'gpu2' partition — 'gpu'/'gpu_test' "
            "are AMD/ROCm and incompatible with this CUDA torch build). "
            "Refusing to fall back to CPU."
        )
    if cuda_ok:
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_device(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """Move a model to device, casting to float32 (required by MPS)."""
    return model.float().to(device)
