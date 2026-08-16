"""Print the headline numbers from one or more epoch bundles.

Exists so submission scripts stop embedding Python one-liners inside heredocs
inside ssh — quoting f-strings through three layers is how you get
``SyntaxError: f-string expression part cannot include a backslash`` instead of
a benchmark result.

    python scripts/summarize_epoch.py results/profile/mb32 results/profile/mb40

With ``--stop-epoch`` it instead answers the protocol question: given every
arm's validation curve under one checkpoint root, which epoch is the reporting
epoch (the first at which every arm has plateaued)? That rule quantifies over all
arms at once, so no single training job can evaluate it — see
``script_utils.common_stop_epoch``.

    python scripts/summarize_epoch.py --stop-epoch results/foveation_mit1003/ckpts
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from tez_deepgaze.script_utils import common_stop_epoch, val_ig_curves


def summarize(bundle_dir: Path) -> None:
    epochs = sorted(bundle_dir.glob("epoch_*/metrics.json"))
    if not epochs:
        print(f"{bundle_dir.name:22} (no epoch bundles)")
        return
    for path in epochs:
        j = json.loads(path.read_text())
        t = j["train"]
        g = t.get("gpu_memory_gb") or {}
        args = j.get("reproducibility", {}).get("args", {})
        parts = [
            f"{bundle_dir.name:22}",
            f"ep{j['epoch']}",
            f"mb={args.get('micro_batch', '?')}",
            f"{j['epoch_seconds']:7.1f}s",
            f"alloc={g.get('peak_allocated', float('nan')):5.1f}GB",
            f"resv={g.get('peak_reserved', float('nan')):5.1f}GB",
            f"trainLL={t['mean_ll_bits_per_fix']:.6f}",
            f"valIG={j.get('val', {}).get('ig_bits_per_fix', float('nan')):.4f}",
            f"nonfin={t['nonfinite_steps']}",
        ]
        gn = t.get("grad_norm")
        if gn:
            parts.append(f"gradnorm={gn['mean']:.3f}/{gn['max']:.3f}")
        print("  ".join(parts))


def report_stop_epoch(ckpt_root: Path, min_delta: float = 0.001) -> None:
    """Print each arm's validation-IG curve and the common reporting epoch."""
    curves = val_ig_curves(ckpt_root)
    if not curves:
        raise SystemExit(f"no epoch bundles with a val IG under {ckpt_root}")
    for tag, by_epoch in curves.items():
        deltas = " ".join(
            f"{by_epoch[b] - by_epoch[a]:+.4f}"
            for a, b in zip(sorted(by_epoch), sorted(by_epoch)[1:])
        )
        print(f"{tag:22} IG {by_epoch[max(by_epoch)]:.4f} @ep{max(by_epoch)}  "
              f"deltas {deltas}")
    stop = common_stop_epoch(curves, min_delta=min_delta)
    if stop is None:
        print(f"\nNo epoch where all {len(curves)} arms improved by <{min_delta} "
              "for two consecutive epochs. Report the final epoch and say the "
              "arms had not plateaued.")
    else:
        print(f"\nCommon stopping epoch: {stop} "
              f"(all {len(curves)} arms below {min_delta} bits/epoch for 2 epochs). "
              f"Pass it to the eval as TEZ_EVAL_EPOCH={stop}.")


def main() -> None:
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    if args[0] == "--stop-epoch":
        if len(args) != 2:
            raise SystemExit("--stop-epoch takes exactly one checkpoint root")
        report_stop_epoch(Path(args[1]))
        return
    for d in args:
        summarize(Path(d))


if __name__ == "__main__":
    main()
