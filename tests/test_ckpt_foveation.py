"""The input transform a checkpoint was trained with must travel with it.

The checkpoint path encodes only the arm tag and foveal_cpd. Everything else
about the input — viewing geometry, and above all whether the fovea tracks gaze
or is pinned to the image centre — lives in the bundle's metrics.json. If eval
does not read it back, a fixed-centre arm gets trained on one input and scored on
another, and loses for a reason that has nothing to do with the experiment.
"""
from __future__ import annotations

import inspect
import json

import pytest

from tez_deepgaze.foveate_input import Foveation
from tez_deepgaze.script_utils import foveation_record, resolve_ckpt_foveation


def _bundle(tmp_path, fov=None, arm=None, **overrides):
    """An epoch bundle holding the metrics.json training would have written."""
    d = tmp_path / "epoch_001"
    d.mkdir(parents=True, exist_ok=True)
    rec = foveation_record(fov) if fov is not None else {}
    rec.update(overrides)
    meta = {"epoch": 1, "foveation": rec}
    if arm is not None:
        meta["arm"] = arm
    (d / "metrics.json").write_text(json.dumps(meta))
    return d / "weights.pt"


def test_center_fovea_checkpoint_is_not_evaluated_gaze_contingently(tmp_path):
    """The fixed-centre flag must reach eval on its own."""
    trained = Foveation(ppd=35.0, foveal_cpd=20.0, center_fovea=True)
    ckpt = _bundle(tmp_path, trained)

    # The caller asks for the default, gaze-contingent foveation.
    requested = Foveation(ppd=35.0, foveal_cpd=20.0)
    assert requested.center_fovea is False

    resolved = resolve_ckpt_foveation(ckpt, requested)
    assert resolved.center_fovea is True, \
        "a center-fovea checkpoint was silently evaluated gaze-contingently"
    assert (resolved.ppd, resolved.foveal_cpd) == (35.0, 20.0)
    assert resolved.e2_deg == requested.e2_deg
    assert resolved.n_levels == requested.n_levels


def test_gaze_contingent_checkpoint_is_not_evaluated_fixed_centre(tmp_path):
    """The same guard the other way round."""
    ckpt = _bundle(tmp_path, Foveation(ppd=35.0, foveal_cpd=20.0))
    resolved = resolve_ckpt_foveation(
        ckpt, Foveation(ppd=35.0, foveal_cpd=20.0, center_fovea=True))
    assert resolved.center_fovea is False


def test_center_fovea_checkpoint_cannot_be_evaluated_sharp(tmp_path):
    """`fov=None` is the sharp arm; a fixed-centre record cannot travel into it."""
    ckpt = _bundle(tmp_path, Foveation(ppd=35.0, foveal_cpd=20.0, center_fovea=True))
    with pytest.raises(ValueError, match="fixed-centre"):
        resolve_ckpt_foveation(ckpt, None)


def test_matching_request_is_returned_unchanged(tmp_path):
    fov = Foveation(ppd=35.0, foveal_cpd=20.0, center_fovea=True)
    assert resolve_ckpt_foveation(_bundle(tmp_path, fov), fov) is fov

    sharp = _bundle(tmp_path / "sharp", Foveation(ppd=35.0, foveal_cpd=20.0))
    assert resolve_ckpt_foveation(sharp, None) is None


def test_sharp_trained_checkpoint_cannot_be_evaluated_foveated(tmp_path):
    """Training writes the foveation record for BOTH arms, so the record alone
    cannot catch a wrong --ckpt-root; the top-level arm can."""
    fov = Foveation(ppd=35.0, foveal_cpd=40.0)
    ckpt = _bundle(tmp_path, fov, arm="normal")
    with pytest.raises(ValueError, match="sharp input"):
        resolve_ckpt_foveation(ckpt, fov)


def test_foveated_trained_checkpoint_cannot_be_evaluated_sharp(tmp_path):
    ckpt = _bundle(tmp_path, Foveation(ppd=35.0, foveal_cpd=20.0), arm="foveated")
    with pytest.raises(ValueError, match="foveated input"):
        resolve_ckpt_foveation(ckpt, None)


def test_matching_arm_passes(tmp_path):
    fov = Foveation(ppd=35.0, foveal_cpd=20.0)
    assert resolve_ckpt_foveation(_bundle(tmp_path, fov, arm="foveated"), fov) is fov

    sharp = _bundle(tmp_path / "sharp", Foveation(ppd=35.0, foveal_cpd=20.0), arm="normal")
    assert resolve_ckpt_foveation(sharp, None) is None

    center = Foveation(ppd=35.0, foveal_cpd=20.0, center_fovea=True)
    assert resolve_ckpt_foveation(
        _bundle(tmp_path / "center", center, arm="foveated_center"), center) is center


@pytest.mark.parametrize("key,value", [("ppd", 70.0), ("foveal_cpd", 10.0)])
def test_geometry_mismatch_still_raises(tmp_path, key, value):
    """Unchanged behaviour: ppd/cpd have no safe default, so they are checked."""
    ckpt = _bundle(tmp_path, Foveation(ppd=35.0, foveal_cpd=20.0), **{key: value})
    with pytest.raises(ValueError, match=key):
        resolve_ckpt_foveation(ckpt, Foveation(ppd=35.0, foveal_cpd=20.0))


def test_checkpoints_predating_the_record_are_left_alone(tmp_path):
    """Bundles written before center_fovea was recorded must keep evaluating."""
    ckpt = _bundle(tmp_path, None, ppd=35.0, foveal_cpd=20.0)   # no center_fovea key
    fov = Foveation(ppd=35.0, foveal_cpd=20.0)
    assert resolve_ckpt_foveation(ckpt, fov) is fov

    (ckpt.parent / "metrics.json").unlink()                     # no metrics at all
    assert resolve_ckpt_foveation(ckpt, fov) is fov


def test_record_round_trips_every_setting_eval_needs():
    """What training writes is what resolve reads — one source, not two.

    Every constructor argument of Foveation must appear, because resolve only
    validates keys it finds: an unrecorded argument is silently taken from the
    eval request instead of the checkpoint.
    """
    fov = Foveation(ppd=35.0, foveal_cpd=20.0, center_fovea=True)
    rec = foveation_record(fov)
    ctor = set(inspect.signature(Foveation.__init__).parameters) - {"self"}
    assert ctor <= set(rec), f"unrecorded constructor args: {ctor - set(rec)}"
    assert rec["ppd"] == 35.0
    assert rec["foveal_cpd"] == 20.0
    assert rec["center_fovea"] is True
    assert rec["e2_deg"] == 2.3
    assert rec["n_levels"] == 7
    # Derived, so an artefact describes its transform without re-running code.
    assert rec["sharp_radius_px"] == 11.5           # e2*(2*cpd - ppd)
    assert rec["level_sigmas"][0] == 0.0
    assert json.loads(json.dumps(rec)) == rec       # must survive metrics.json
