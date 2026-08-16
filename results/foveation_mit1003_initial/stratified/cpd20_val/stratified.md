# Stratified analysis — normal vs foveated@20 on MIT1003

- Folds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], val split, 104171 fixations, paired per fixation.
- Sharp-fovea radius 11.5 px (0.33 deg); 0.6 % of saccades land inside it.

## Overall (paired)

| quantity | normal | foveated | Δ (foveated − normal) | t |
|---|---|---|---|---|
| LL (bits/fix) | — | — | -0.0050 ± 0.0013 | -3.89 |
| entropy (bits) | 16.987 | 16.998 | +0.0109 ± 0.0005 | +23.08 |
| mode distance (px) | 191.5 | 191.3 | -0.24 ± 0.23 | -1.05 |

Entropy rising alongside a likelihood drop means the model spread its mass (less certain). Entropy flat or falling means it stayed committed and was mislocated instead — a different failure.

## Calibration (spatial PIT vs uniform; 0 = calibrated)

- KS statistic: normal **0.0213**, foveated **0.0206**
- PIT mean: normal 0.5105, foveated 0.5100 (0.5 = uniform)

## Δ LL by saccade amplitude

| amplitude (px) | (deg) | inside sharp disc | share of fix. | Δ LL (bits/fix) | t |
|---|---|---|---|---|---|
| 0–11.5 | 0.0–0.3 | yes | 0.6 % | -0.0248 ± 0.0097 | -2.57 |
| 11.5–18 | 0.3–0.5 | no | 0.6 % | -0.0032 ± 0.0087 | -0.36 |
| 18–25 | 0.5–0.7 | no | 0.7 % | +0.0013 ± 0.0087 | +0.16 |
| 25–35 | 0.7–1.0 | no | 1.6 % | -0.0069 ± 0.0059 | -1.18 |
| 35–50 | 1.0–1.4 | no | 4.9 % | -0.0015 ± 0.0034 | -0.45 |
| 50–70 | 1.4–2.0 | no | 9.8 % | -0.0008 ± 0.0027 | -0.28 |
| 70–103.5 | 2.0–3.0 | no | 14.7 % | +0.0003 ± 0.0024 | +0.11 |
| 103.5–125 | 3.0–3.6 | no | 7.8 % | -0.0050 ± 0.0037 | -1.34 |
| 125–149.5 | 3.6–4.3 | no | 7.8 % | -0.0084 ± 0.0040 | -2.11 |
| 149.5–175 | 4.3–5.0 | no | 7.3 % | -0.0095 ± 0.0045 | -2.10 |
| 175–210 | 5.0–6.0 | no | 8.5 % | -0.0018 ± 0.0045 | -0.39 |
| 210–260 | 6.0–7.4 | no | 10.0 % | -0.0056 ± 0.0045 | -1.25 |
| 260–320 | 7.4–9.1 | no | 9.0 % | -0.0052 ± 0.0051 | -1.03 |
| 320–400 | 9.1–11.4 | no | 7.7 % | -0.0089 ± 0.0059 | -1.50 |
| 400–500 | 11.4–14.3 | no | 5.4 % | +0.0070 ± 0.0075 | +0.93 |
| 500–∞ | 14.3–∞ | no | 3.6 % | -0.0378 ± 0.0113 | -3.35 |

If the peripheral-preview account is right, Δ LL should be ~0 in the bins inside the sharp disc and grow with amplitude beyond it. A flat profile refutes it.

## Δ LL by fixation index

| fixation index | share of fix. | Δ LL (bits/fix) | t |
|---|---|---|---|
| 1 | 14.3 % | -0.0003 ± 0.0038 | -0.08 |
| 2 | 14.0 % | -0.0082 ± 0.0039 | -2.07 |
| 3 | 13.7 % | -0.0057 ± 0.0036 | -1.56 |
| 4–5 | 25.2 % | -0.0068 ± 0.0022 | -3.03 |
| 6–8 | 25.9 % | -0.0045 ± 0.0023 | -1.95 |
| 9–100 | 6.8 % | -0.0017 ± 0.0046 | -0.36 |
