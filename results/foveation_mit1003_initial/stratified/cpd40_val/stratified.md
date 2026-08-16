# Stratified analysis — normal vs foveated@40 on MIT1003

- Folds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], val split, 104171 fixations, paired per fixation.
- Sharp-fovea radius 103.5 px (2.96 deg); 32.9 % of saccades land inside it.

## Overall (paired)

| quantity | normal | foveated | Δ (foveated − normal) | t |
|---|---|---|---|---|
| LL (bits/fix) | — | — | +0.0039 ± 0.0008 | +5.20 |
| entropy (bits) | 16.987 | 16.984 | -0.0026 ± 0.0003 | -9.39 |
| mode distance (px) | 191.5 | 191.3 | -0.22 ± 0.18 | -1.22 |

Entropy rising alongside a likelihood drop means the model spread its mass (less certain). Entropy flat or falling means it stayed committed and was mislocated instead — a different failure.

## Calibration (spatial PIT vs uniform; 0 = calibrated)

- KS statistic: normal **0.0213**, foveated **0.0221**
- PIT mean: normal 0.5105, foveated 0.5106 (0.5 = uniform)

## Δ LL by saccade amplitude

| amplitude (px) | (deg) | inside sharp disc | share of fix. | Δ LL (bits/fix) | t |
|---|---|---|---|---|---|
| 0–11.5 | 0.0–0.3 | yes | 0.6 % | -0.0141 ± 0.0062 | -2.29 |
| 11.5–18 | 0.3–0.5 | yes | 0.6 % | -0.0038 ± 0.0059 | -0.65 |
| 18–25 | 0.5–0.7 | yes | 0.7 % | +0.0034 ± 0.0059 | +0.58 |
| 25–35 | 0.7–1.0 | yes | 1.6 % | -0.0041 ± 0.0037 | -1.11 |
| 35–50 | 1.0–1.4 | yes | 4.9 % | -0.0014 ± 0.0021 | -0.64 |
| 50–70 | 1.4–2.0 | yes | 9.8 % | +0.0015 ± 0.0015 | +0.99 |
| 70–103.5 | 2.0–3.0 | yes | 14.7 % | +0.0049 ± 0.0013 | +3.63 |
| 103.5–125 | 3.0–3.6 | no | 7.8 % | +0.0009 ± 0.0020 | +0.43 |
| 125–149.5 | 3.6–4.3 | no | 7.8 % | -0.0004 ± 0.0021 | -0.20 |
| 149.5–175 | 4.3–5.0 | no | 7.3 % | +0.0009 ± 0.0024 | +0.37 |
| 175–210 | 5.0–6.0 | no | 8.5 % | +0.0027 ± 0.0024 | +1.12 |
| 210–260 | 6.0–7.4 | no | 10.0 % | -0.0002 ± 0.0026 | -0.07 |
| 260–320 | 7.4–9.1 | no | 9.0 % | +0.0073 ± 0.0030 | +2.40 |
| 320–400 | 9.1–11.4 | no | 7.7 % | +0.0125 ± 0.0037 | +3.38 |
| 400–500 | 11.4–14.3 | no | 5.4 % | +0.0238 ± 0.0049 | +4.85 |
| 500–∞ | 14.3–∞ | no | 3.6 % | +0.0023 ± 0.0077 | +0.30 |

If the peripheral-preview account is right, Δ LL should be ~0 in the bins inside the sharp disc and grow with amplitude beyond it. A flat profile refutes it.

## Δ LL by fixation index

| fixation index | share of fix. | Δ LL (bits/fix) | t |
|---|---|---|---|
| 1 | 14.3 % | +0.0051 ± 0.0020 | +2.51 |
| 2 | 14.0 % | +0.0084 ± 0.0023 | +3.70 |
| 3 | 13.7 % | +0.0017 ± 0.0022 | +0.76 |
| 4–5 | 25.2 % | +0.0034 ± 0.0014 | +2.43 |
| 6–8 | 25.9 % | +0.0032 ± 0.0014 | +2.25 |
| 9–100 | 6.8 % | +0.0018 ± 0.0028 | +0.65 |
