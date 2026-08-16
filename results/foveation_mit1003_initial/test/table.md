# Normal vs gaze-contingent foveated DG3 on MIT1003 (foveation sweep)

- Heads: **trained per-(arm,fold) checkpoints**
- 10-fold CV, test split, folds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 104171 fixations (normal arm)
- Foveation: ppd=35; sweep foveal_cpd ∈ {40, 20, 10} (lower = stronger)
- Fixed-centre ablation at foveal_cpd ∈ {40, 20, 10}: same blur profile, fovea pinned to the image centre. foveated@C − center@C is the gaze-contingency term.

| arm | LL (bits/fix over uniform) | IG (bits/fix over centerbias) | NSS | AUC |
|---|---|---|---|---|
| normal | +2.492 ± 0.044 | +1.574 ± 0.025 | +3.309 ± 0.060 | +0.918 ± 0.002 |
| foveated@40 | +2.496 ± 0.046 | +1.578 ± 0.026 | +3.313 ± 0.063 | +0.918 ± 0.002 |
| foveated@20 | +2.486 ± 0.047 | +1.568 ± 0.027 | +3.302 ± 0.065 | +0.918 ± 0.002 |
| foveated@10 | +2.455 ± 0.050 | +1.538 ± 0.030 | +3.262 ± 0.069 | +0.916 ± 0.002 |
| center@40 | +2.493 ± 0.046 | +1.575 ± 0.026 | +3.306 ± 0.064 | +0.918 ± 0.002 |
| center@20 | +2.480 ± 0.047 | +1.563 ± 0.027 | +3.287 ± 0.066 | +0.918 ± 0.002 |
| center@10 | +2.442 ± 0.048 | +1.524 ± 0.028 | +3.228 ± 0.064 | +0.916 ± 0.002 |

**Gap vs normal (foveated − normal), fold-paired Δ ± 2 SE:**

| arm | Δ LL | Δ IG | Δ NSS | Δ AUC |
|---|---|---|---|---|
| foveated@40 | +0.0039 ± 0.0032 | +0.0039 ± 0.0032 | +0.0038 ± 0.0041 | +0.0003 ± 0.0002 |
| foveated@20 | -0.0059 ± 0.0053 | -0.0059 ± 0.0053 | -0.0074 ± 0.0090 | -0.0003 ± 0.0003 |
| foveated@10 | -0.0365 ± 0.0083 | -0.0365 ± 0.0083 | -0.0469 ± 0.0196 | -0.0018 ± 0.0004 |
| center@40 | +0.0009 ± 0.0038 | +0.0009 ± 0.0038 | -0.0033 ± 0.0070 | +0.0002 ± 0.0002 |
| center@20 | -0.0115 ± 0.0063 | -0.0115 ± 0.0063 | -0.0217 ± 0.0133 | -0.0003 ± 0.0003 |
| center@10 | -0.0497 ± 0.0087 | -0.0497 ± 0.0087 | -0.0808 ± 0.0209 | -0.0020 ± 0.0004 |

ΔIG foveated@40: fold-paired +0.0039 ± 0.0032 (2 SE), 2/10 folds negative; image-paired t = +2.96 over 1003 images.

ΔIG foveated@20: fold-paired -0.0059 ± 0.0053 (2 SE), 8/10 folds negative; image-paired t = -2.44 over 1003 images.

ΔIG foveated@10: fold-paired -0.0365 ± 0.0083 (2 SE), 10/10 folds negative; image-paired t = -10.09 over 1003 images.

ΔIG center@40: fold-paired +0.0009 ± 0.0038 (2 SE), 5/10 folds negative; image-paired t = +0.61 over 1003 images.

ΔIG center@20: fold-paired -0.0115 ± 0.0063 (2 SE), 9/10 folds negative; image-paired t = -4.48 over 1003 images.

ΔIG center@10: fold-paired -0.0497 ± 0.0087 (2 SE), 10/10 folds negative; image-paired t = -12.37 over 1003 images.

Arm rows: mean ± 2 SE across folds, dominated by image difficulty, which is common to all arms. Gap rows are fold-paired, so that component cancels — gap SEs are not comparable to arm SEs.
Sweep scanpaths: `scanpath_sweep.png`. Per-image IG behind the pairing: `per_image_ig.json` (recompute with `--paired-from`).
