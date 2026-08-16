"""Space-variant foveation blur on the input image.

Foveation here means **resolution falloff**: the image is sharp at the point of
gaze and progressively lower-resolution toward the periphery, matching the way
human visual acuity drops with eccentricity. It is *geometry-preserving* — every
pixel stays where it is, only the local sharpness changes. It is **not** a
log-polar / cortical remap (that family magnifies the centre and compresses the
periphery, which looks like a fisheye); a fisheye/bulge appearance means a
geometric warp crept in, never a blur.

Model (Geisler & Perry 1998)
----------------------------
The highest resolvable spatial frequency falls with eccentricity ``e`` (degrees):

    f_c(e) = f_c0 * e2 / (e + e2)        [cycles / degree]

so acuity is ``f_c0`` at the fovea and halves every ``e2`` degrees. Human values:
``f_c0 ~ 40`` cyc/deg (foveal cutoff), ``e2 = 2.3`` deg — see ``FOVEAL_CPD`` below
for where the 40 comes from.

The falloff is realised as a continuous blend over a Gaussian pyramid:

  1. Eccentricity per pixel, in degrees, from the true pixels-per-degree ``ppd``.
  2. Target cutoff ``f_c(e)``, converted to cycles/pixel and clamped to the
     display Nyquist (0.5 cyc/px) — inside the Nyquist limit the image stays
     sharp, which gives a properly-sized sharp fovea instead of a tight dot.
  3. A fractional pyramid level ``L(e) = log2(0.5 / f_c_px(e))``.
  4. Linear interpolation between the two bracketing blur levels. The continuous
     blend is what avoids the ring / "peephole bubble" artifact of hard level
     selection.

Faithful foveation is *mild* on a screen-sized image
----------------------------------------------------
Acuity falloff is set by eccentricity in **degrees**, so the visible blur within
a frame depends on how many degrees the frame subtends. A lab image spans only
~25-35 deg, where human acuity is reduced but not collapsed — so faithful
foveation looks subtle. The dramatic "foveation" demos online assume the image
fills a much larger field (few px/deg) or steepen the falloff beyond human
acuity. To make the effect visibly stronger *honestly*, lower ``foveal_cpd``
below the ~40 human value (a coarser-than-human eye); this is exaggeration, not
a geometry change, and the blend stays smooth. Raising it above ~40 goes the other
way: the sharp disc grows until it covers most saccade targets and the transform
stops constraining the model at all.

Unit convention
---------------
Fixations are passed in **original-image pixels** (matching how DG3 stores
``x_hist`` / ``y_hist``).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torchvision.transforms.functional import gaussian_blur

# Geisler & Perry (1998) human constants.
#
# FOVEAL_CPD is derived from G&P's own fitted contrast-threshold model rather than
# taken from their introduction, which quotes ~50 cyc/deg as the general human
# resolution cutoff. Their eq. (1) is
#     CT(f, e) = CT0 * exp(alpha * f * (e + e2) / e2)
# and eq. (2) sets CT = 1 (max contrast) to get the critical eccentricity. Solving
# instead for f, and evaluating at e = 0, the e2 cancels:
#     f_c0 = ln(1 / CT0) / alpha
# With their least-squares alpha = 0.106 and the three CT0 they report:
#     CT0 = 1/64 (Robson & Graham 1981) -> 39.23 cyc/deg
#     CT0 = 1/75 (Arnow & Geisler)      -> 40.73
#     CT0 = 1/76 (Banks et al.)         -> 40.86
# i.e. 40.0 +/- 0.8 across all three, hence 40. This keeps both constants
# (f_c0 and e2) from the same fit; the introduction's prose pairs ~50 cyc/deg with
# a 2.5 deg half-acuity eccentricity, which is not the fitted e2 = 2.3 used here.
FOVEAL_CPD = 40.0   # foveal cutoff frequency (cycles/degree)
E2_DEG = 2.3        # eccentricity at which acuity halves (degrees)
# MIT1003 viewing geometry (Judd et al. 2009: "approximately two feet from a 19 inch
# computer screen of resolution 1280x1024"). A 19in 5:4 panel is 37.7 cm wide ->
# 33.97 px/cm; at 60.96 cm one degree spans 1.064 cm -> 36.1 px/deg (38.1 if the
# 19in is a CRT tube with ~18in viewable). The constant below stays at 35: in pixel
# space ppd enters the blur only as the offset e2*ppd in
#     L(e_px) = log2((e_px + e2*ppd) / (2*f_c0*e2))
# so ppd 35 vs 38 shifts the applied blur by <3% in sigma anywhere in the frame, and
# the lower value errs toward slightly MORE blur, i.e. the applied transform is if
# anything slightly stronger than the geometry strictly requires.
MIT1003_PPD = 35.0

# The value to convert pixels to degrees *for reporting*, which is not the value
# the blur is built from. The manipulation is defined in pixels: ppd cancels out
# of the sharp-radius scale (e2*ppd * (2*f_c0/ppd - 1) = e2*(2*f_c0 - ppd)), so
# MIT1003_PPD above fixes the transform and must never move. Only the degree
# *label* on those pixels is a convention, and 35 is the wrong one for it.
#
# 36.6 comes from Kummerer et al. 2022 stating the MIT1003 stimulus diagonal is
# about 35 dva; for a 1024x768 frame (1280 px diagonal) that back-solves to
# 1280/35 = 36.6 px/deg. Preferred over reconstructing it from Judd et al.'s rig
# ("approximately two feet from a 19 inch screen" -> 36.1 LCD / 38.1 CRT)
# because it is stated about this corpus rather than inferred from hardware.
#
# Reporting at 36.6 rather than 35 makes every degree-valued figure ~4.5%
# smaller (cpd 40's sharp disc is 2.83 deg rather than 2.96; the median human
# saccade 4.38 deg rather than 4.58). Pixel quantities -- including AMP_EDGES_PX,
# whose edges were chosen to land on the pixel radii -- are unaffected.
MIT1003_PPD_REPORT = 36.6

# A Gaussian blur of std sigma (px) has frequency response
# H(f) = exp(-2 * pi**2 * sigma**2 * f**2). Solving H(f) = 1/2 for the
# half-amplitude point (|H| = 1/2, i.e. -6 dB) gives the std whose response
# drops to half amplitude at frequency f (cyc/px): sigma = SIGMA_AT_UNIT_CUTOFF / f.
SIGMA_AT_UNIT_CUTOFF = math.sqrt(math.log(2.0) / (2.0 * math.pi**2))  # ~0.1874


def acuity_cutoff_cpd(
    e_deg: torch.Tensor, foveal_cpd: float = FOVEAL_CPD, e2_deg: float = E2_DEG
) -> torch.Tensor:
    """Resolvable cutoff frequency (cycles/degree) at eccentricity ``e_deg``."""
    return foveal_cpd * e2_deg / (e_deg + e2_deg)


def identity_radius_px(foveal_cpd: float, ppd: float, e2_deg: float = E2_DEG) -> float:
    """Radius (px) of the identity disc around gaze.

    Inside it the requested cutoff exceeds the display Nyquist (``ppd / 2``),
    the clamp in :meth:`Foveation.fractional_level` engages, and the transform
    is exactly the identity. From ``acuity_cutoff_cpd(e) = ppd / 2``:
    ``e = e2 * (2 * foveal_cpd / ppd - 1)`` degrees, converted to pixels below.
    """
    return max(0.0, e2_deg * (2.0 * foveal_cpd - ppd))


class Foveation(nn.Module):
    """Geometry-preserving space-variant foveation blur.

    Parameters
    ----------
    ppd:
        Pixels per degree of visual angle (viewing-geometry constant). MIT1003
        ~= 35.
    foveal_cpd:
        Foveal cutoff frequency (cycles/degree). Human ~= 40 (faithful, mild on
        screen-sized images; see ``FOVEAL_CPD``). Lower it to make foveation
        visibly stronger — a coarser-than-human eye, clearly an exaggeration.
        The sharp-fovea radius follows ``e2 * (2 * foveal_cpd - ppd)`` pixels.
    e2_deg:
        Eccentricity at which acuity halves (degrees). Human ~= 2.3.
    n_levels:
        Number of Gaussian pyramid levels. Level 0 is the sharp original; level
        ``i`` has std ``0.375 * 2**i`` px (octave-spaced cutoffs). 7 levels
        reach ~24 px blur, enough for deep periphery.
    """

    def __init__(
        self,
        ppd: float = MIT1003_PPD,
        foveal_cpd: float = FOVEAL_CPD,
        e2_deg: float = E2_DEG,
        n_levels: int = 7,
        center_fovea: bool = False,
    ) -> None:
        super().__init__()
        self.ppd = float(ppd)
        self.foveal_cpd = float(foveal_cpd)
        self.e2_deg = float(e2_deg)
        self.n_levels = int(n_levels)
        # Where the fovea sits, as opposed to how sharpness falls off with
        # distance from it. False (default) = gaze-contingent: the fovea tracks
        # the model's own current fixation and the input changes after every
        # saccade. True = pinned to the image centre for the whole scanpath,
        # which is what prior pixel-level foveated networks do. Holding the blur
        # profile identical and varying only this isolates the gaze-contingency
        # itself.
        self.center_fovea = bool(center_fovea)
        # Level i cutoff = 0.5 * 2**-i cyc/px  ->  sigma_i = 0.375 * 2**i px.
        sigmas = [0.0] + [
            SIGMA_AT_UNIT_CUTOFF / (0.5 * 2.0 ** (-i)) for i in range(1, n_levels)
        ]
        self.register_buffer(
            "level_sigmas", torch.tensor(sigmas, dtype=torch.float32), persistent=False
        )

    @property
    def sharp_radius_px(self) -> float:
        """This instance's identity-disc radius — :func:`identity_radius_px`."""
        return identity_radius_px(self.foveal_cpd, self.ppd, self.e2_deg)

    def blur_stack(self, img: torch.Tensor) -> torch.Tensor:
        """``(L, B, C, H, W)`` stack of uniformly-blurred copies (level 0 = sharp).

        The stack depends only on the image, not on the fixation, so callers
        that foveate the same image repeatedly can build it once and pass it
        back in via :meth:`foveate_shared_image`'s ``stack`` argument.
        """
        out = [img]
        for s in self.level_sigmas.tolist()[1:]:
            k = max(3, int(2 * round(3 * s) + 1))
            out.append(gaussian_blur(img, [k, k], [s, s]))
        return torch.stack(out, dim=0)

    def fractional_level(self, e_px: torch.Tensor) -> torch.Tensor:
        """Per-pixel fractional pyramid level for eccentricity ``e_px`` (pixels)."""
        e_deg = e_px / self.ppd
        f_c_cpd = acuity_cutoff_cpd(e_deg, self.foveal_cpd, self.e2_deg)
        f_c_cpp = (f_c_cpd / self.ppd).clamp(max=0.5)  # cyc/px, capped at Nyquist
        level = torch.log2(0.5 / f_c_cpp)              # 0 at fovea, grows outward
        return level.clamp(min=0.0, max=float(self.n_levels - 1))

    def _eccentricity_px(
        self, H: int, W: int, fix_x_px: torch.Tensor, fix_y_px: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Per-pixel eccentricity (px) from each fixation. Shape ``(B, H, W)``."""
        ys = torch.arange(H, dtype=torch.float32, device=device)
        xs = torch.arange(W, dtype=torch.float32, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        dx = gx.unsqueeze(0) - fix_x_px.view(-1, 1, 1).to(device)
        dy = gy.unsqueeze(0) - fix_y_px.view(-1, 1, 1).to(device)
        return torch.sqrt(dx * dx + dy * dy)

    def _blend(self, stack: torch.Tensor, e_px: torch.Tensor) -> torch.Tensor:
        """Space-variant blend of a blur ``stack`` ``(L, B, C, H, W)`` by
        per-pixel eccentricity ``e_px`` ``(B, H, W)``.

        Linear interpolation between bracketing levels: triangular hat weights
        on integer level nodes form a partition of unity, so the blend is a
        smooth, continuous space-variant blur (no rings, no bubble).
        """
        level = self.fractional_level(e_px)            # (B, H, W) in [0, L-1]
        out = torch.zeros_like(stack[0])
        for k in range(self.n_levels):
            w = (1.0 - (level - k).abs()).clamp(min=0.0)  # (B, H, W)
            out = out + w.unsqueeze(1) * stack[k]
        return out

    def forward(
        self, img: torch.Tensor, fix_x_px: torch.Tensor, fix_y_px: torch.Tensor
    ) -> torch.Tensor:
        """Foveate ``img`` (B, C, H, W) around per-sample fixations; same shape out."""
        _, _, H, W = img.shape
        e_px = self._eccentricity_px(H, W, fix_x_px, fix_y_px, img.device)
        stack = self.blur_stack(img)                   # (L, B, C, H, W)
        return self._blend(stack, e_px)

    def foveate_shared_image(
        self,
        img_1chw: torch.Tensor,
        fix_x_px: torch.Tensor,
        fix_y_px: torch.Tensor,
        stack: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Foveate ONE image (1, C, H, W) around ``B`` fixation centres at once.

        The blur pyramid is built a single time and shared across the ``B``
        centres, so cost is ``L`` blurs rather than ``B × L``. Used by the
        training loss, where every (history, target) pair on a stimulus shares
        the same source image but foveates around its own current fixation.
        ``stack`` optionally reuses a pyramid from :meth:`blur_stack` built
        from this same image, skipping the blur work entirely. Returns
        ``(B, C, H, W)``; equals stacking :meth:`forward` over the centres.
        """
        if img_1chw.shape[0] != 1:
            raise ValueError("foveate_shared_image expects a single image (batch dim 1)")
        _, _, H, W = img_1chw.shape
        B = int(fix_x_px.reshape(-1).shape[0])
        if stack is None:
            stack = self.blur_stack(img_1chw)
        stack = stack.expand(-1, B, -1, -1, -1)        # (L, B, C, H, W) view
        e_px = self._eccentricity_px(H, W, fix_x_px, fix_y_px, img_1chw.device)
        return self._blend(stack, e_px)
