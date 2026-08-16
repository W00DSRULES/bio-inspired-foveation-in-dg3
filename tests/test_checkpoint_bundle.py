"""Checkpoint round-trip: trainable-only weights, both formats, optimizer state.

``weights.pt`` holds only the trainable read-out tensors, not a full model
``state_dict``. A full ``state_dict`` must still load, so the format detection is
load-bearing and gets its own tests.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from tez_deepgaze.instrument import (
    load_checkpoint_weights,
    save_checkpoint_bundle,
    trainable_state_dict,
)


class _TwoPart(nn.Module):
    """A frozen 'backbone' and a trainable 'head', mirroring the real setup."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)
        for p in self.features.parameters():
            p.requires_grad_(False)


def test_trainable_state_dict_excludes_frozen():
    m = _TwoPart()
    keys = set(trainable_state_dict(m))
    assert keys == {"head.weight", "head.bias"}
    assert not any(k.startswith("features.") for k in keys)


def test_bundle_writes_trainable_weights_and_optimizer(tmp_path):
    m = _TwoPart()
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-3)
    # One real step so Adam has non-trivial moment estimates to persist.
    m.head(torch.randn(3, 4)).sum().backward()
    opt.step()

    bundle = save_checkpoint_bundle(tmp_path, epoch=5, model=m, metrics={"a": 1},
                                    optimizer=opt)
    assert (bundle / "weights.pt").exists()
    assert (bundle / "metrics.json").exists()
    assert (bundle / "optimizer.pt").exists()
    assert set(torch.load(bundle / "weights.pt")) == {"head.weight", "head.bias"}

    restored = torch.optim.Adam([p for p in _TwoPart().parameters() if p.requires_grad], lr=1e-3)
    restored.load_state_dict(torch.load(bundle / "optimizer.pt"))
    assert restored.state_dict()["state"][0]["step"] == opt.state_dict()["state"][0]["step"]


def test_optimizer_file_absent_when_not_passed(tmp_path):
    m = _TwoPart()
    bundle = save_checkpoint_bundle(tmp_path, epoch=1, model=m, metrics={})
    assert not (bundle / "optimizer.pt").exists()


def test_load_trainable_checkpoint_restores_head_and_keeps_frozen(tmp_path):
    """A partial checkpoint must overlay the head without disturbing the backbone."""
    trained = _TwoPart()
    with torch.no_grad():
        trained.head.weight.fill_(0.5)
        trained.features.weight.fill_(9.0)   # pretend the backbone differs
    bundle = save_checkpoint_bundle(tmp_path, epoch=1, model=trained, metrics={})

    fresh = _TwoPart()
    with torch.no_grad():
        fresh.features.weight.fill_(3.0)     # stands in for the pretrained backbone
    fmt = load_checkpoint_weights(fresh, bundle / "weights.pt")

    assert fmt == "trainable"
    assert torch.allclose(fresh.head.weight, torch.full_like(fresh.head.weight, 0.5))
    # The frozen tensors come from whatever was loaded first, NOT the checkpoint.
    assert torch.allclose(fresh.features.weight, torch.full_like(fresh.features.weight, 3.0))


def test_load_full_legacy_checkpoint_still_works(tmp_path):
    """A checkpoint holding a complete state_dict is still loadable."""
    trained = _TwoPart()
    with torch.no_grad():
        trained.head.weight.fill_(0.25)
        trained.features.weight.fill_(7.0)
    legacy = tmp_path / "weights.pt"
    torch.save(trained.state_dict(), legacy)      # the full format

    fresh = _TwoPart()
    fmt = load_checkpoint_weights(fresh, legacy)

    assert fmt == "full"
    assert torch.allclose(fresh.head.weight, torch.full_like(fresh.head.weight, 0.25))
    assert torch.allclose(fresh.features.weight, torch.full_like(fresh.features.weight, 7.0))


def test_load_rejects_checkpoint_missing_trainable_tensors(tmp_path):
    """A checkpoint without the read-out is not silently accepted."""
    bad = tmp_path / "weights.pt"
    torch.save({"features.weight": torch.zeros(4, 4)}, bad)
    try:
        load_checkpoint_weights(_TwoPart(), bad)
    except ValueError as exc:
        assert "missing trainable tensors" in str(exc)
    else:
        raise AssertionError("expected ValueError for a checkpoint with no head tensors")


def test_two_formats_are_equivalent_after_pretrained_reset(tmp_path):
    """The whole point: full and trainable-only format land on identical parameters.

    The equivalence holds *because the backbone is frozen* — training leaves it
    byte-identical to the pretrained weights, so omitting it from the checkpoint
    loses nothing. The test therefore starts the trained model from the same
    frozen tensors, which is what a real run guarantees.
    """
    pretrained_state = {k: v.clone() for k, v in _TwoPart().state_dict().items()}
    trained = _TwoPart()
    trained.load_state_dict(pretrained_state)     # frozen part matches, as in a real run
    with torch.no_grad():
        trained.head.weight.normal_()             # only the head moves
        trained.head.bias.normal_()

    torch.save(trained.state_dict(), tmp_path / "full.pt")
    torch.save(trainable_state_dict(trained), tmp_path / "trainable.pt")

    via_full = _TwoPart()
    via_full.load_state_dict(pretrained_state)
    load_checkpoint_weights(via_full, tmp_path / "full.pt")

    via_partial = _TwoPart()
    via_partial.load_state_dict(pretrained_state)
    load_checkpoint_weights(via_partial, tmp_path / "trainable.pt")

    for (ka, va), (kb, vb) in zip(via_full.state_dict().items(),
                                  via_partial.state_dict().items()):
        assert ka == kb
        assert torch.allclose(va, vb), f"{ka} differs between checkpoint formats"


class _WithFinalizers(nn.Module):
    """Backbone + head + finalizers, none frozen — as eval builds the model."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)
        self.finalizers = nn.Linear(2, 2)


def test_partial_checkpoint_loads_into_never_frozen_model(tmp_path):
    """Eval never freezes anything, so its model has requires_grad=True on the
    finalizers — but new-format checkpoints legitimately omit them (training
    freezes FROZEN_SUBMODULES). The loader must not demand them back, or it
    rejects every production checkpoint with "missing trainable tensors
    ['finalizers...']".
    """
    trained = _WithFinalizers()
    for name in ("features", "finalizers"):          # as freeze_for_head_training does
        for p in getattr(trained, name).parameters():
            p.requires_grad_(False)
    with torch.no_grad():
        trained.head.weight.fill_(0.5)
    bundle = save_checkpoint_bundle(tmp_path, epoch=1, model=trained, metrics={})
    assert not any("finalizers" in k for k in torch.load(bundle / "weights.pt"))

    fresh = _WithFinalizers()                        # nothing frozen, like eval
    with torch.no_grad():
        fresh.finalizers.weight.fill_(3.0)           # stands in for pretrained values
    fmt = load_checkpoint_weights(fresh, bundle / "weights.pt")

    assert fmt == "trainable"
    assert torch.allclose(fresh.head.weight, torch.full_like(fresh.head.weight, 0.5))
    # The finalizers keep what the caller loaded first (the pretrained state).
    assert torch.allclose(fresh.finalizers.weight,
                          torch.full_like(fresh.finalizers.weight, 3.0))


def test_unfreezing_a_module_puts_it_back_in_the_checkpoint():
    """The format follows requires_grad, so it self-adjusts if the backbone is unfrozen.

    Unfreezing the backbone is listed as future work. If that happens, its
    updated weights must not be silently dropped from the checkpoint — they are
    not, because the key set is derived from requires_grad rather than hardcoded.
    """
    m = _TwoPart()
    assert not any(k.startswith("features.") for k in trainable_state_dict(m))
    for p in m.features.parameters():
        p.requires_grad_(True)
    keys = set(trainable_state_dict(m))
    assert {"features.weight", "features.bias"} <= keys
