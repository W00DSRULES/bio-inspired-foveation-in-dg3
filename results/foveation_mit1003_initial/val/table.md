# Normal vs gaze-contingent foveated DG3 on MIT1003 (foveation sweep)

- Heads: **trained per-(arm,fold) checkpoints**
- 10-fold CV, val split, folds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 104171 fixations (normal arm)
- Foveation: ppd=35; sweep foveal_cpd ∈ {40, 20, 10} (lower = stronger)
- Fixed-centre ablation at foveal_cpd ∈ {40, 20, 10}: same blur profile, fovea pinned to the image centre. foveated@C − center@C is the gaze-contingency term.

| arm | LL (bits/fix over uniform) | IG (bits/fix over centerbias) | NSS | AUC |
|---|---|---|---|---|
| normal | +2.492 ± 0.044 | +1.574 ± 0.025 | +3.308 ± 0.060 | +0.918 ± 0.002 |
| foveated@40 | +2.496 ± 0.046 | +1.578 ± 0.026 | +3.311 ± 0.063 | +0.918 ± 0.002 |
| foveated@20 | +2.486 ± 0.048 | +1.569 ± 0.027 | +3.299 ± 0.065 | +0.918 ± 0.002 |
| foveated@10 | +2.456 ± 0.050 | +1.538 ± 0.030 | +3.260 ± 0.069 | +0.916 ± 0.002 |
| center@40 | +2.493 ± 0.046 | +1.575 ± 0.026 | +3.304 ± 0.063 | +0.918 ± 0.002 |
| center@20 | +2.481 ± 0.048 | +1.563 ± 0.028 | +3.287 ± 0.066 | +0.918 ± 0.002 |
| center@10 | +2.443 ± 0.049 | +1.525 ± 0.029 | +3.228 ± 0.066 | +0.916 ± 0.002 |

**Gap vs normal (foveated − normal), fold-paired Δ ± 2 SE:**

| arm | Δ LL | Δ IG | Δ NSS | Δ AUC |
|---|---|---|---|---|
| foveated@40 | +0.0041 ± 0.0032 | +0.0041 ± 0.0032 | +0.0033 ± 0.0039 | +0.0003 ± 0.0002 |
| foveated@20 | -0.0052 ± 0.0053 | -0.0052 ± 0.0053 | -0.0087 ± 0.0097 | -0.0002 ± 0.0003 |
| foveated@10 | -0.0359 ± 0.0082 | -0.0359 ± 0.0082 | -0.0479 ± 0.0203 | -0.0017 ± 0.0004 |
| center@40 | +0.0012 ± 0.0035 | +0.0012 ± 0.0035 | -0.0045 ± 0.0065 | +0.0002 ± 0.0002 |
| center@20 | -0.0110 ± 0.0065 | -0.0110 ± 0.0065 | -0.0212 ± 0.0132 | -0.0003 ± 0.0003 |
| center@10 | -0.0491 ± 0.0092 | -0.0491 ± 0.0092 | -0.0798 ± 0.0218 | -0.0020 ± 0.0005 |

ΔIG foveated@40: fold-paired +0.0041 ± 0.0032 (2 SE), 2/10 folds negative; image-paired t = +3.13 over 1003 images.

ΔIG foveated@20: fold-paired -0.0052 ± 0.0053 (2 SE), 8/10 folds negative; image-paired t = -2.21 over 1003 images.

ΔIG foveated@10: fold-paired -0.0359 ± 0.0082 (2 SE), 10/10 folds negative; image-paired t = -9.92 over 1003 images.

ΔIG center@40: fold-paired +0.0012 ± 0.0035 (2 SE), 4/10 folds negative; image-paired t = +0.81 over 1003 images.

ΔIG center@20: fold-paired -0.0110 ± 0.0065 (2 SE), 9/10 folds negative; image-paired t = -4.27 over 1003 images.

ΔIG center@10: fold-paired -0.0491 ± 0.0092 (2 SE), 10/10 folds negative; image-paired t = -12.18 over 1003 images.

Arm rows: mean ± 2 SE across folds, dominated by image difficulty, which is common to all arms. Gap rows are fold-paired, so that component cancels — gap SEs are not comparable to arm SEs.
Sweep scanpaths: `scanpath_sweep.png`. Per-image IG behind the pairing: `per_image_ig.json` (recompute with `--paired-from`).
