"""The hoisted DeepGazeIIIMixture forward must be bit-identical to the vendor one.

It only moves two loop-invariant calls out of the ensemble loop, so anything
other than an exact match means the hoist is wrong and must not be used.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from deepgaze_pytorch.modules import DeepGazeIIIMixture

from tez_deepgaze.fast_dg3 import (
    apply_fast_forward,
    apply_fast_layernorm,
    assert_forward_matches,
    layer_norm_max_diff,
)

# Constructs pretrained DG3 five times (session fixture + four CPU-pinned copies).
pytestmark = pytest.mark.heavy


@pytest.fixture(scope="module")
def tiny_input(device, dg3_model):
    """A small image plus a 4-fixation history, on the model's device."""
    H, W = 192, 256          # small but divisible by downsample * readout_factor
    rng = np.random.RandomState(0)
    img = torch.from_numpy(rng.randint(0, 256, (1, 3, H, W)).astype(np.float32)).to(device)
    cb = torch.zeros((1, H, W), dtype=torch.float32, device=device)
    cb = cb - torch.logsumexp(cb.flatten(), 0)
    xh = torch.tensor([[120.0, 90.0, 60.0, 30.0]], device=device)
    yh = torch.tensor([[80.0, 70.0, 60.0, 50.0]], device=device)
    return img, cb, xh, yh


def _unpatch(model):
    """Undo apply_fast_forward on a shared model: the bound forward AND the
    capability flag. Popping only "forward" leaves accepts_shared_image=True on
    a model whose restored vendor forward cannot take a shared image, and
    _log_density_batch_core routes the shortcut on that flag."""
    model.__dict__.pop("forward", None)
    model.__dict__.pop("accepts_shared_image", None)


def test_model_is_a_mixture(dg3_model):
    """Guards the premise: the pretrained model really is the 10-member ensemble."""
    assert isinstance(dg3_model, DeepGazeIIIMixture)
    assert len(dg3_model.saliency_networks) == 10
    assert len(dg3_model.scanpath_networks) == 10


def test_hoisted_forward_is_bit_identical(dg3_model, tiny_input):
    img, cb, xh, yh = tiny_input
    diff = assert_forward_matches(dg3_model, img, cb, xh, yh, atol=0.0)
    assert diff == 0.0, f"expected exact equality, got {diff}"


def test_apply_and_restore(dg3_model, tiny_input):
    """Patching is explicit and reversible; nothing happens on import alone."""
    img, cb, xh, yh = tiny_input
    with torch.no_grad():
        before = dg3_model(img, cb, xh, yh).clone()

    n = apply_fast_forward(dg3_model)
    assert n == 1, f"expected to patch exactly one mixture, patched {n}"
    with torch.no_grad():
        after = dg3_model(img, cb, xh, yh)
    assert torch.equal(before, after)

    _unpatch(dg3_model)                              # restore the class forward
    with torch.no_grad():
        restored = dg3_model(img, cb, xh, yh)
    assert torch.equal(before, restored)


def test_hoist_holds_for_a_batch(dg3_model, device):
    """Several histories at once — the case training actually runs."""
    H, W = 192, 256
    rng = np.random.RandomState(1)
    img = torch.from_numpy(rng.randint(0, 256, (4, 3, H, W)).astype(np.float32)).to(device)
    cb = torch.zeros((4, H, W), dtype=torch.float32, device=device)
    cb = cb - torch.logsumexp(cb[0].flatten(), 0)
    xh = torch.tensor(rng.uniform(0, W, (4, 4)).astype(np.float32), device=device)
    yh = torch.tensor(rng.uniform(0, H, (4, 4)).astype(np.float32), device=device)
    assert assert_forward_matches(dg3_model, img, cb, xh, yh, atol=0.0) == 0.0


def test_shared_image_matches_replicated_batch():
    """One image + B histories must equal B copies of that image + B histories.

    This is the sharp arm's shortcut: the frozen backbone runs once instead of
    B times. It is only legitimate if the answer is unchanged.

    Pinned to CPU on purpose. Unlike the hoist, this path changes the batch size
    the convolutions see (1 instead of B), and which kernel a backend selects
    depends on that, so the two routes agree only to the last bits. The bound is
    therefore fp32 round-off, not zero.

    Exact equality holds on Apple silicon and does not generalise: on the
    cluster the two routes differ by 5.7e-6 on CPU. The size of the
    disagreement is platform-specific and on CUDA it
    is far larger — measured ~6e-3 in log-density, ~1e-3 bits at the fixated
    pixel, under cudnn.deterministic. That is the platform the reported numbers
    come from and the effect being measured is ~0.003 bits, so whether this
    shortcut belongs in a reported run is an open question this test does not
    settle. It only pins that the CPU routes agree to round-off.
    """
    import deepgaze_pytorch

    H, W, B = 192, 256, 5
    model = deepgaze_pytorch.DeepGazeIII(pretrained=True).eval()
    apply_fast_forward(model)

    rng = np.random.RandomState(7)
    one = torch.from_numpy(rng.randint(0, 256, (1, 3, H, W)).astype(np.float32))
    many = one.expand(B, -1, -1, -1).contiguous()
    cb1 = torch.zeros((1, H, W), dtype=torch.float32)
    cb1 = cb1 - torch.logsumexp(cb1.flatten(), 0)
    cbB = cb1.expand(B, -1, -1).contiguous()
    xh = torch.tensor(rng.uniform(0, W, (B, 4)).astype(np.float32))
    yh = torch.tensor(rng.uniform(0, H, (B, 4)).astype(np.float32))

    with torch.no_grad():
        replicated = model(many, cbB, xh, yh)
        shared = model(one, cb1, xh, yh)
    assert shared.shape == replicated.shape
    diff = float((shared - replicated).abs().max())
    assert diff < FP32_ROUNDING, f"shared-image path differs by {diff:.3e}"


def test_batch_core_routes_sharp_arm_through_the_shared_image():
    """`_log_density_batch_core` must take the shortcut, and must not change the answer.

    The previous test bounds the shortcut at the model boundary; this one proves
    the caller actually reaches it, since the whole saving comes from the caller
    handing over one image instead of B copies. The sharp arm qualifies here; so
    does the fixed-centre ablation, in the test below. Gaze-contingent foveation
    does not, because there every history sees a differently blurred picture.
    """
    import deepgaze_pytorch

    from tez_deepgaze.instrument import _log_density_batch_core

    H, W, B = 192, 256, 5
    dev = torch.device("cpu")           # same reason as the test above
    model = deepgaze_pytorch.DeepGazeIII(pretrained=True).eval()

    rng = np.random.RandomState(11)
    image = rng.randint(0, 256, (H, W, 3)).astype(np.uint8)
    cb = np.full((H, W), -np.log(H * W), dtype=np.float32)
    hx = [[float(v) for v in rng.uniform(0, W, 4)] for _ in range(B)]
    hy = [[float(v) for v in rng.uniform(0, H, 4)] for _ in range(B)]

    with torch.no_grad():                                  # unpatched: replicates
        assert not getattr(model, "accepts_shared_image", False)
        replicated = _log_density_batch_core(model, image, cb, hx, hy, dev).clone()

        apply_fast_forward(model)                          # patched: shares
        assert model.accepts_shared_image
        seen = {}
        handle = model.features.register_forward_pre_hook(
            lambda _mod, inputs: seen.update(batch=inputs[0].shape[0])
        )
        try:
            shared = _log_density_batch_core(model, image, cb, hx, hy, dev)
        finally:
            handle.remove()

    assert seen["batch"] == 1, f"backbone still saw {seen['batch']} images, expected 1"
    diff = float((shared - replicated).abs().max())
    assert diff < FP32_ROUNDING, f"sharp-arm shortcut changed the answer by {diff:.3e}"


def test_batch_core_shares_the_image_for_the_fixed_centre_arm():
    """The fixed-centre ablation qualifies for the shortcut; gaze-contingent does not.

    With the fovea pinned to the image centre every fixation on a stimulus sees
    the identical foveated picture, so the frozen backbone can run once per image
    instead of once per fixation — the same saving the sharp arm already takes.
    The shortcut is only legitimate if the log-density is unchanged, which is what
    the equality below asserts; the gaze-contingent case is checked too, because
    there the pictures really do differ and taking the shortcut would be wrong.

    Pinned to CPU, like the sharp-arm tests above. The bound is fp32 round-off
    rather than zero: this path changes the batch size the convolutions see, and
    which kernel a backend picks depends on that, so the two routes agree only up
    to the last bits. Measured 4.8e-6 on the cluster's x86 CPU. On CUDA the same
    comparison is three orders larger (~1e-3 bits at the fixated pixel, under
    cudnn.deterministic); that is a property of this shortcut generally, applying
    equally to the sharp arm's long-standing use of it, and is not settled here.
    """
    import deepgaze_pytorch

    from tez_deepgaze.foveate_input import Foveation
    from tez_deepgaze.instrument import _log_density_batch_core

    H, W, B = 192, 256, 5
    dev = torch.device("cpu")
    model = deepgaze_pytorch.DeepGazeIII(pretrained=True).eval()

    rng = np.random.RandomState(13)
    image = rng.randint(0, 256, (H, W, 3)).astype(np.uint8)
    cb = np.full((H, W), -np.log(H * W), dtype=np.float32)
    hx = [[float(v) for v in rng.uniform(0, W, 4)] for _ in range(B)]
    hy = [[float(v) for v in rng.uniform(0, H, 4)] for _ in range(B)]
    centred = Foveation(ppd=35.0, foveal_cpd=20.0, center_fovea=True)
    gazed = Foveation(ppd=35.0, foveal_cpd=20.0)

    def backbone_batch(fov):
        """(log-density, how many images the frozen backbone actually saw)."""
        seen = {}
        handle = model.features.register_forward_pre_hook(
            lambda _mod, inputs: seen.update(batch=inputs[0].shape[0])
        )
        try:
            with torch.no_grad():
                out = _log_density_batch_core(model, image, cb, hx, hy, dev,
                                              foveation=fov).clone()
        finally:
            handle.remove()
        return out, seen["batch"]

    replicated, n_seen = backbone_batch(centred)      # unpatched: no shortcut
    assert not getattr(model, "accepts_shared_image", False)
    assert n_seen == B

    apply_fast_forward(model)
    shared, n_seen = backbone_batch(centred)
    assert n_seen == 1, f"backbone still saw {n_seen} images, expected 1"
    diff = float((shared - replicated).abs().max())
    assert diff < FP32_ROUNDING, f"fixed-centre shortcut changed the answer by {diff:.3e}"

    _, n_seen = backbone_batch(gazed)
    assert n_seen == B, \
        "gaze-contingent foveation must NOT share the image — every history differs"


FP32_ROUNDING = 1e-5   # generous ceiling on fp32 round-off at these magnitudes


@pytest.mark.parametrize("features", [1, 8, 2048])
def test_fast_layernorm_agrees_with_the_vendor_one(features):
    """Rewriting the layer norm must only change the last bits, not the value.

    The vendor version is 58 % of a training step, so this is where the speed is;
    the whole case for taking it rests on the two forms being the same formula.
    They are not bit-identical — the reductions sum in a different order — so the
    bound is fp32 round-off rather than zero.
    """
    torch.manual_seed(0)
    diff = layer_norm_max_diff(features, (4, 96, 128), torch.device("cpu"))
    assert diff < FP32_ROUNDING, f"features={features} disagreed by {diff:.3e}"


def test_fast_layernorm_patches_every_norm_in_the_model(dg3_model):
    """All 70 of them: 7 per read-out stack across the 10 ensemble members."""
    from deepgaze_pytorch.layers import LayerNorm as DGLayerNorm

    expected = sum(isinstance(m, DGLayerNorm) for m in dg3_model.modules())
    assert expected == 70, f"model shape changed: {expected} layer norms, expected 70"
    try:
        assert apply_fast_layernorm(dg3_model) == expected
    finally:
        for m in dg3_model.modules():
            if isinstance(m, DGLayerNorm):
                m.__dict__.pop("forward", None)


def test_fast_layernorm_leaves_the_log_density_alone():
    """End to end: the quantity the thesis reports must not move.

    LL is reported in bits to three decimals with a standard error of ~0.017, so
    what matters is not that the log-density is identical but that it differs far
    below anything reportable. Checked as a total-variation distance too, since
    that is the honest statement about the predicted distribution.
    """
    import deepgaze_pytorch

    from tez_deepgaze.instrument import _log_density_batch_core

    H, W, B = 192, 256, 4
    dev = torch.device("cpu")
    model = deepgaze_pytorch.DeepGazeIII(pretrained=True).eval()
    apply_fast_forward(model)

    rng = np.random.RandomState(3)
    image = rng.randint(0, 256, (H, W, 3)).astype(np.uint8)
    cb = np.full((H, W), -np.log(H * W), dtype=np.float32)
    hx = [[float(v) for v in rng.uniform(0, W, 4)] for _ in range(B)]
    hy = [[float(v) for v in rng.uniform(0, H, 4)] for _ in range(B)]

    with torch.no_grad():
        ref = _log_density_batch_core(model, image, cb, hx, hy, dev).clone()
        apply_fast_layernorm(model)
        got = _log_density_batch_core(model, image, cb, hx, hy, dev)

    assert float((ref - got).abs().max()) < FP32_ROUNDING
    tv = float((ref.exp() - got.exp()).abs().sum(dim=(1, 2)).max()) / 2
    assert tv < 1e-5, f"predicted distribution moved by TV={tv:.3e}"


def test_fast_layernorm_still_trains():
    """Gradients must reach the norm's own weight and bias, not just flow past them.

    The rewrite replaces a fused autograd node with a chain of elementwise ops,
    which is exactly the kind of change that can silently detach a parameter.
    """
    from deepgaze_pytorch.layers import LayerNorm as DGLayerNorm

    ln = DGLayerNorm(8)
    torch.nn.init.normal_(ln.weight, mean=1.0, std=0.1)
    apply_fast_layernorm_on(ln)
    x = torch.randn(3, 8, 12, 16, requires_grad=True)
    ln(x).square().sum().backward()

    for name, p in (("weight", ln.weight), ("bias", ln.bias)):
        assert p.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(p.grad).all()
        assert float(p.grad.abs().max()) > 0, f"gradient to {name} is all zeros"
    assert x.grad is not None and torch.isfinite(x.grad).all()


def apply_fast_layernorm_on(module):
    """`apply_fast_layernorm` walks `.modules()`, which includes the module itself."""
    assert apply_fast_layernorm(module) == 1


def test_mismatched_batches_are_rejected(dg3_model, device):
    """A 3-image batch with 5 histories is a caller bug, not a broadcast."""
    H, W = 192, 256
    img = torch.zeros((3, 3, H, W), device=device)
    cb = torch.zeros((3, H, W), device=device)
    xh = torch.zeros((5, 4), device=device)
    yh = torch.zeros((5, 4), device=device)
    apply_fast_forward(dg3_model)
    try:
        with pytest.raises(ValueError, match="disagree"):
            with torch.no_grad():
                dg3_model(img, cb, xh, yh)
    finally:
        _unpatch(dg3_model)
