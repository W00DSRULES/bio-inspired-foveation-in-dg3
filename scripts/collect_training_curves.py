"""Aggregate the per-epoch training metrics into one committable JSON.

The production checkpoint tree holds one ``metrics.json`` per (arm, fold,
epoch), each carrying a per-image validation breakdown that makes it far too
large to commit, and it lives beside multi-gigabyte weight bundles on the
cluster anyway. ``training_curves.png`` needs four scalars per file. This pulls
exactly those out and writes a single ~130 KB artefact that is committed, so the
figure stays regenerable from the repo once the checkpoint tree is gone.

Run it where the checkpoints are (the cluster), then copy the result back:

    python scripts/collect_training_curves.py
    scp goethe:/work/dldevel/itez/Tez/results/foveation_mit1003/training_curves.json \\
        results/foveation_mit1003/

It reads nothing but JSON, so it needs no GPU, no model and no corpus, and it is
safe to run on a login node.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "foveation_mit1003"
PATTERN = "*/fold*/epoch_*/metrics.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt-root", type=Path, default=RES / "ckpts",
                    help="checkpoint tree to walk (default: the production matrix)")
    ap.add_argument("--out", type=Path, default=RES / "training_curves.json")
    args = ap.parse_args()

    files = sorted(args.ckpt_root.glob(PATTERN))
    if not files:
        raise SystemExit(f"no {PATTERN} under {args.ckpt_root} — is this the cluster?")

    arms: dict[str, dict[str, dict[str, dict]]] = {}
    for p in files:
        m = re.search(r"([^/]+)/fold(\d+)/epoch_(\d+)/metrics\.json$", p.as_posix())
        arm, fold, epoch = m.group(1), str(int(m.group(2))), str(int(m.group(3)))
        j = json.loads(p.read_text())
        arms.setdefault(arm, {}).setdefault(fold, {})[epoch] = {
            "val_ig": j["val"]["ig_bits_per_fix"],
            "val_ll": j["val"]["ll_bits_per_fix"],
            "train_ll": j["train"]["mean_ll_bits_per_fix"],
            "lr": j["lr"],
        }

    args.out.write_text(json.dumps({
        "source": f"{args.ckpt_root.relative_to(ROOT)}/{PATTERN}",
        "n_files": len(files),
        "fields": "per (arm, fold, epoch): validation IG and LL, mean train LL, learning rate",
        "arms": arms,
    }, indent=1) + "\n")
    print(f"wrote {args.out} — {len(files)} files, {len(arms)} arms, "
          f"{sorted(len(v) for v in arms.values())[-1]} folds")


if __name__ == "__main__":
    main()
