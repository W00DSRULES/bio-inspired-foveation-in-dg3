# Normal vs gaze-contingent foveated DG3 on MIT1003 (foveation sweep)

- Heads: **pretrained heads (no fine-tuning)**
- 10-fold CV, val split, folds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 104171 fixations (normal arm)
- Foveation: ppd=35; sweep foveal_cpd ∈ {40, 20, 10} (lower = stronger)

| arm | LL (bits/fix over uniform) | IG (bits/fix over centerbias) | NSS | AUC |
|---|---|---|---|---|
| normal | +2.476 ± 0.044 | +1.558 ± 0.024 | +3.266 ± 0.062 | +0.917 ± 0.002 |
| foveated@40 | +2.476 ± 0.047 | +1.558 ± 0.026 | +3.273 ± 0.064 | +0.918 ± 0.002 |
| foveated@20 | +2.453 ± 0.049 | +1.536 ± 0.028 | +3.253 ± 0.065 | +0.917 ± 0.002 |
| foveated@10 | +2.402 ± 0.052 | +1.484 ± 0.032 | +3.190 ± 0.063 | +0.915 ± 0.002 |

**Gap vs normal (foveated − normal), fold-paired Δ ± 2 SE:**

| arm | Δ LL | Δ IG | Δ NSS | Δ AUC |
|---|---|---|---|---|
| foveated@40 | -0.0003 ± 0.0034 | -0.0003 ± 0.0034 | +0.0067 ± 0.0037 | +0.0002 ± 0.0002 |
| foveated@20 | -0.0224 ± 0.0071 | -0.0224 ± 0.0071 | -0.0128 ± 0.0098 | -0.0007 ± 0.0003 |
| foveated@10 | -0.0741 ± 0.0123 | -0.0741 ± 0.0123 | -0.0765 ± 0.0214 | -0.0028 ± 0.0005 |

ΔIG foveated@40: fold-paired -0.0003 ± 0.0034 (2 SE), 4/10 folds negative; image-paired t = -0.23 over 1003 images.

ΔIG foveated@20: fold-paired -0.0224 ± 0.0071 (2 SE), 10/10 folds negative; image-paired t = -8.01 over 1003 images.

ΔIG foveated@10: fold-paired -0.0741 ± 0.0123 (2 SE), 10/10 folds negative; image-paired t = -16.86 over 1003 images.

Arm rows: mean ± 2 SE across folds, dominated by image difficulty, which is common to all arms. Gap rows are fold-paired, so that component cancels — gap SEs are not comparable to arm SEs.
Sweep scanpaths: `scanpath_sweep.png`. Per-image IG behind the pairing: `per_image_ig.json` (recompute with `--paired-from`).
