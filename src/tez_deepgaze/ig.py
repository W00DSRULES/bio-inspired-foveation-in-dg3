"""Information-gain helpers (canonical definition)."""
from __future__ import annotations

import math

# nats → bits conversion factor, shared by every module that reports bits.
LOG2 = math.log(2.0)


def compute_ig_bits(ll_model_bits: float, ll_baseline_bits: float) -> float:
    """Information gain in bits/fix: IG = LL_model − LL_baseline.

    Both operands must be evaluated on the same fixation set and must already
    include the centerbias contribution. DG3 adds the centerbias *inside*
    ``Finalizer.forward``, so the model's reported LL is the post-centerbias
    LL — do NOT try to "subtract centerbias" by passing ``cb=zeros`` to the
    model. Compare model-LL to a separately-evaluated centerbias-only LL
    instead.
    """
    return float(ll_model_bits) - float(ll_baseline_bits)
