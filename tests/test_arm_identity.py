"""Three arms must stay distinguishable, in the metrics and on disk.

The fixed-centre ablation is foveated input too, so it is one careless string
away from being indistinguishable from the gaze-contingent arm. Two things go
wrong if that happens: the checkpoint path is keyed only by (tag, fold, epoch),
so one arm silently overwrites the other's weights; and a bundle that does not
say which arm it is cannot be audited after the fact.
"""
from __future__ import annotations

import re
from pathlib import Path

from tez_deepgaze.foveate_input import Foveation
from tez_deepgaze.script_utils import foveation_record

SLURM = Path(__file__).resolve().parents[1] / "scripts" / "slurm"


def _tags_from(path: Path) -> dict[str, str]:
    """The (arm -> tag) mapping a submitter script's case statement encodes."""
    text = path.read_text()
    out = {}
    for arm in ("normal", "foveated", "center"):
        m = re.search(rf'^\s*{arm}\)(.*?)(?=^\s*\w+\)|^\s*\*\))', text,
                      re.MULTILINE | re.DOTALL)
        assert m, f"{path.name}: no case branch for arm '{arm}'"
        tag = re.search(r'TAG="([^"]+)"', m.group(1))
        assert tag, f"{path.name}: arm '{arm}' sets no TAG"
        out[arm] = tag.group(1)
    return out


def test_every_arm_gets_its_own_checkpoint_tag():
    script = "foveation_train_fold.sbatch"
    tags = _tags_from(SLURM / script)
    assert len(set(tags.values())) == 3, f"{script}: arms share a tag: {tags}"
    # The centre arm must not collide with the gaze-contingent arm at the same
    # cpd — that is the collision that would overwrite trained weights.
    assert tags["center"] != tags["foveated"]
    assert tags["center"].startswith(tags["foveated"])


def test_the_centre_arm_actually_passes_the_flag():
    text = (SLURM / "foveation_train_fold.sbatch").read_text()
    m = re.search(r'^\s*center\)(.*?)(?=^\s*\*\))', text, re.MULTILINE | re.DOTALL)
    assert "--center-fovea" in m.group(1), "centre arm does not pass --center-fovea"
    assert "--foveate" in m.group(1), "centre arm must still foveate"


def test_the_pump_uses_the_same_tags_as_the_submitters():
    """The pump's skip check reads the path the submitter writes; they must match."""
    pump = (SLURM / "foveation_pump.sh").read_text()
    assert "fov_cpd%s_center" in pump, "pump cannot build the centre arm's tag"
    for arm in ("normal", "foveated", "center"):
        assert re.search(rf'SPECS\+=\("{arm} ', pump), f"pump never submits arm '{arm}'"


def test_a_centre_checkpoint_is_self_describing():
    """center_fovea reaches the bundle, so an arm can be identified from it alone."""
    assert foveation_record(Foveation(foveal_cpd=20.0, center_fovea=True))["center_fovea"]
    assert not foveation_record(Foveation(foveal_cpd=20.0))["center_fovea"]
