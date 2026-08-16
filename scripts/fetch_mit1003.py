"""Fetch MIT1003 robustly.

pysaliency's downloader is not resumable and the MIT server is flaky.
This script:
  1. Downloads the three required zip files via curl (resumable, retrying)
     into a persistent cache at data/mit1003-raw/.
  2. Monkeypatches pysaliency.utils.download_file to copy from that cache.
  3. Runs pysaliency.get_mit1003, which will extract, run MATLAB (or Octave),
     and write the final HDF5 into data/mit1003/MIT1003/.

With ``--with-initial`` it instead builds ``MIT1003_initial_fix_consistent/``
via ``get_mit1003_with_initial_fixation(replace_initial_invalid_fixations=True)``
— the dataset variant the DeepGaze III training code uses, where each scanpath
keeps the forced central start fixation as its first entry. Afterwards the new
variant is validated against the existing plain dataset: pysaliency guarantees
``with_initial[lengths > 0]`` equals the plain fixations, so every shared row
is compared and the run fails loudly on a structural mismatch.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pysaliency
import pysaliency.utils as psu
import pysaliency.external_datasets.mit as psmit

ROOT = Path(__file__).resolve().parents[1] / "data"
RAW = ROOT / "mit1003-raw"
DEST = ROOT / "mit1003"

FILES = {
    "http://people.csail.mit.edu/tjudd/WherePeopleLook/ALLSTIMULI.zip": "ALLSTIMULI.zip",
    "http://people.csail.mit.edu/tjudd/WherePeopleLook/DATA.zip": "DATA.zip",
    "http://people.csail.mit.edu/tjudd/WherePeopleLook/Code/DatabaseCode.zip": "DatabaseCode.zip",
}


def curl_download(url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "curl", "-L", "--retry", "10", "--retry-delay", "5",
        "--retry-all-errors", "--continue-at", "-",
        "--fail", "--show-error",
        "-o", str(out), url,
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_raw() -> None:
    for url, name in FILES.items():
        out = RAW / name
        if out.exists() and out.stat().st_size > 1024:
            print(f"[cache] {name} ({out.stat().st_size / 1e6:.1f} MB)")
            continue
        curl_download(url, out)


NAN_SHIMS = {
    "nanmedian.m": (
        "function y = nanmedian(x, dim)\n"
        "if nargin < 2\n"
        "    y = median(x, 'omitnan');\n"
        "else\n"
        "    y = median(x, dim, 'omitnan');\n"
        "end\n"
        "end\n"
    ),
    "nanmean.m": (
        "function y = nanmean(x, dim)\n"
        "if nargin < 2\n"
        "    y = mean(x, 'omitnan');\n"
        "else\n"
        "    y = mean(x, dim, 'omitnan');\n"
        "end\n"
        "end\n"
    ),
    "nanstd.m": (
        "function y = nanstd(x, flag, dim)\n"
        "if nargin < 2, flag = 0; end\n"
        "if nargin < 3\n"
        "    y = std(x, flag, 'omitnan');\n"
        "else\n"
        "    y = std(x, flag, dim, 'omitnan');\n"
        "end\n"
        "end\n"
    ),
    "nansum.m": (
        "function y = nansum(x, dim)\n"
        "if nargin < 2\n"
        "    y = sum(x, 'omitnan');\n"
        "else\n"
        "    y = sum(x, dim, 'omitnan');\n"
        "end\n"
        "end\n"
    ),
    "nanvar.m": (
        "function y = nanvar(x, w, dim)\n"
        "if nargin < 2, w = 0; end\n"
        "if nargin < 3\n"
        "    y = var(x, w, 'omitnan');\n"
        "else\n"
        "    y = var(x, w, dim, 'omitnan');\n"
        "end\n"
        "end\n"
    ),
    "nanmax.m": (
        "function y = nanmax(x, y2, dim)\n"
        "if nargin == 1\n"
        "    y = max(x, [], 'omitnan');\n"
        "elseif nargin == 2\n"
        "    y = max(x, y2, 'omitnan');\n"
        "else\n"
        "    y = max(x, y2, dim, 'omitnan');\n"
        "end\n"
        "end\n"
    ),
    "nanmin.m": (
        "function y = nanmin(x, y2, dim)\n"
        "if nargin == 1\n"
        "    y = min(x, [], 'omitnan');\n"
        "elseif nargin == 2\n"
        "    y = min(x, y2, 'omitnan');\n"
        "else\n"
        "    y = min(x, y2, dim, 'omitnan');\n"
        "end\n"
        "end\n"
    ),
}


def install_patch() -> None:
    """Patch pysaliency: (1) copy zips from RAW cache, (2) inject nanX shims
    into MATLAB cwd before extract_all_fixations runs."""
    _orig_download = psu.download_file

    def patched_download(url: str, target: str, verify_ssl: bool = True) -> None:
        name = FILES.get(url)
        if name is not None:
            src = RAW / name
            if src.exists():
                print(f"[patch] copy {src} -> {target}")
                shutil.copyfile(src, target)
                return
        _orig_download(url, target, verify_ssl=verify_ssl)

    psu.download_file = patched_download

    _orig_matlab = psu.run_matlab_cmd

    def patched_matlab(cmd: str, cwd: str | None = None) -> None:
        if cwd is not None:
            for fname, body in NAN_SHIMS.items():
                p = Path(cwd) / fname
                if not p.exists():
                    p.write_text(body)
                    print(f"[patch] wrote shim {p}")
        _orig_matlab(cmd, cwd=cwd)

    psu.run_matlab_cmd = patched_matlab
    # pysaliency.external_datasets.mit imports run_matlab_cmd by name;
    # rebind there too.
    psmit.run_matlab_cmd = patched_matlab


def validate_with_initial(fixations) -> None:
    """Check the with-initial build against the existing plain dataset.

    pysaliency's contract: ``with_initial[lengths > 0]`` equals the plain
    fixations row for row. A coordinate drift here would mean the (MATLAB vs
    Octave) extraction runs disagree, and the new variant could not be compared
    against results computed on the plain data. Structural mismatch exits 1.
    """
    import numpy as np

    # The plain reference must already exist: if it were missing,
    # get_mit1003 would rebuild it with the same patched Octave toolchain
    # that just built the with-initial variant, and the comparison below
    # would check Octave against itself instead of against the build the
    # committed results were computed from.
    plain_hdf5 = DEST / "MIT1003" / "fixations.hdf5"
    if not plain_hdf5.exists():
        print(f"FAIL: plain reference {plain_hdf5} does not exist — restore the "
              "plain dataset first, then re-run the validation")
        sys.exit(1)
    plain_stim, plain = pysaliency.get_mit1003(location=str(DEST))
    del plain_stim
    sel = fixations.lengths > 0
    n_shared, n_plain = int(sel.sum()), len(plain.x)
    print(f"validate: shared rows {n_shared} vs plain rows {n_plain}")
    if n_shared != n_plain:
        print("FAIL: row counts differ — extraction is not consistent with the plain build")
        sys.exit(1)
    same_stim = bool((fixations.n[sel] == plain.n).all())
    same_pos = bool((fixations.lengths[sel] == plain.lengths + 1).all())
    dx = float(np.abs(fixations.x[sel] - plain.x).max())
    dy = float(np.abs(fixations.y[sel] - plain.y).max())
    print(f"validate: stimulus alignment {same_stim}, position offset +1 {same_pos}")
    print(f"validate: max |dx| {dx:.2e} px, max |dy| {dy:.2e} px on {n_plain} shared fixations")
    if not (same_stim and same_pos):
        print("FAIL: rows are not aligned with the plain dataset")
        sys.exit(1)
    if max(dx, dy) > 0.5:
        print("FAIL: shared coordinates differ by more than half a pixel")
        sys.exit(1)
    print(f"validate: {int((fixations.lengths == 0).sum())} initial fixations added")
    print("validate: PASS")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-initial", action="store_true",
                    help="build MIT1003_initial_fix_consistent (forced central start "
                         "fixation kept, invalid initials replaced by exact center) "
                         "and validate it against the plain dataset")
    args = ap.parse_args()

    ensure_raw()
    install_patch()
    DEST.mkdir(parents=True, exist_ok=True)
    if args.with_initial:
        print(f"Loading MIT1003_initial_fix_consistent into {DEST} ...")
        stimuli, fixations = psmit.get_mit1003_with_initial_fixation(
            location=str(DEST), replace_initial_invalid_fixations=True,
        )
        print(f"  stimuli:   {len(stimuli)}")
        print(f"  fixations: {len(fixations.x)}")
        validate_with_initial(fixations)
    else:
        print(f"Loading MIT1003 into {DEST} ...")
        stimuli, fixations = pysaliency.get_mit1003(location=str(DEST))
        print(f"  stimuli:   {len(stimuli)}")
        print(f"  fixations: {len(fixations.x)}")


if __name__ == "__main__":
    main()
