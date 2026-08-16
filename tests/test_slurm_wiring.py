"""Every env knob a SLURM script documents must actually be read by it.

A documented-but-unread knob is silent by construction — the caller sets it, the
script ignores it, and the output looks well-formed (a fixed-centre arm that is
trained but never evaluated leaves only a missing table column). These tests are
cheap (pure text, no GPU, no model) and turn that into a failure.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

SLURM_DIR = Path(__file__).resolve().parents[1] / "scripts" / "slurm"
JOB_SCRIPTS = sorted(SLURM_DIR.glob("*.sbatch")) + sorted(SLURM_DIR.glob("*.sh"))

# Knobs consumed by the code rather than by the shell, or exported for a child
# process, so they legitimately never appear as ${VAR} in the script body.
EXEMPT = {"TEZ_DATA_ROOT", "TEZ_REQUIRE_CUDA"}

# "#   TEZ_FOO  : description" in a header comment block.
DOC_KNOB = re.compile(r"^#\s+(TEZ_[A-Z0-9_]+)\s*:", re.M)


def _documented(text: str) -> set[str]:
    return set(DOC_KNOB.findall(text)) - EXEMPT


def _read(text: str, knob: str) -> bool:
    """Is the knob actually dereferenced somewhere in the script?"""
    return bool(re.search(rf"\$\{{{knob}[:}}]|\${knob}\b", text))


@pytest.mark.parametrize("script", JOB_SCRIPTS, ids=lambda p: p.name)
def test_documented_knobs_are_read(script: Path):
    text = script.read_text()
    unread = sorted(k for k in _documented(text) if not _read(text, k))
    assert not unread, (
        f"{script.name} documents {unread} but never dereferences them. "
        "A caller setting one would be silently ignored."
    )


def test_center_cpds_reaches_the_eval_script():
    """The wiring TEZ_CENTER_CPDS -> --center-cpds, asserted end to end.

    Presence of the variable is not enough: the documentation can be in place
    while the plumbing that turns it into the flag is missing.
    """
    text = (SLURM_DIR / "foveation_sweep_eval.sbatch").read_text()
    assert "TEZ_CENTER_CPDS" in text, "eval job does not read TEZ_CENTER_CPDS"
    assert "--center-cpds" in text, (
        "eval job never passes --center-cpds, so a trained fixed-centre arm "
        "cannot appear in the table"
    )
    # The flag must be on the command line, not merely assigned to a variable.
    invocation = text.split("foveation_sweep_table.py", 1)[1]
    assert "CENTER_FLAG" in invocation or "--center-cpds" in invocation, (
        "--center-cpds is built but never reaches foveation_sweep_table.py"
    )


def test_eval_script_accepts_the_flag_the_job_passes():
    """`--center-cpds` must exist in the script the job invokes."""
    sweep = SLURM_DIR.parent / "foveation_sweep_table.py"
    assert '"--center-cpds"' in sweep.read_text(), (
        "foveation_sweep_table.py does not define --center-cpds, which "
        "foveation_sweep_eval.sbatch passes"
    )
