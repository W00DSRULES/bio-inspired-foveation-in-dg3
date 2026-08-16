# Stratified analysis — normal vs foveated@10 on MIT1003

- Folds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], val split, 104171 fixations, paired per fixation.
- Sharp-fovea radius 0.0 px (0.00 deg); 0.0 % of saccades land inside it.

## Overall (paired)

| quantity | normal | foveated | Δ (foveated − normal) | t |
|---|---|---|---|---|
| LL (bits/fix) | — | — | -0.0354 ± 0.0018 | -19.29 |
| entropy (bits) | 16.987 | 17.030 | +0.0426 ± 0.0007 | +64.42 |
| mode distance (px) | 191.5 | 191.6 | +0.10 ± 0.27 | +0.36 |

Entropy rising alongside a likelihood drop means the model spread its mass (less certain). Entropy flat or falling means it stayed committed and was mislocated instead — a different failure.

## Calibration (spatial PIT vs uniform; 0 = calibrated)

- KS statistic: normal **0.0213**, foveated **0.0204**
- PIT mean: normal 0.5105, foveated 0.5098 (0.5 = uniform)

## Δ LL by saccade amplitude

| amplitude (px) | (deg) | inside sharp disc | share of fix. | Δ LL (bits/fix) | t |
|---|---|---|---|---|---|
| 0–11.5 | 0.0–0.3 | no | 0.6 % | -0.0164 ± 0.0130 | -1.26 |
| 11.5–18 | 0.3–0.5 | no | 0.6 % | -0.0187 ± 0.0122 | -1.54 |
| 18–25 | 0.5–0.7 | no | 0.7 % | -0.0076 ± 0.0110 | -0.70 |
| 25–35 | 0.7–1.0 | no | 1.6 % | -0.0078 ± 0.0077 | -1.01 |
| 35–50 | 1.0–1.4 | no | 4.9 % | -0.0061 ± 0.0047 | -1.30 |
| 50–70 | 1.4–2.0 | no | 9.8 % | -0.0156 ± 0.0038 | -4.16 |
| 70–103.5 | 2.0–3.0 | no | 14.7 % | -0.0138 ± 0.0035 | -3.97 |
| 103.5–125 | 3.0–3.6 | no | 7.8 % | -0.0235 ± 0.0054 | -4.36 |
| 125–149.5 | 3.6–4.3 | no | 7.8 % | -0.0273 ± 0.0058 | -4.69 |
| 149.5–175 | 4.3–5.0 | no | 7.3 % | -0.0356 ± 0.0066 | -5.40 |
| 175–210 | 5.0–6.0 | no | 8.5 % | -0.0324 ± 0.0064 | -5.07 |
| 210–260 | 6.0–7.4 | no | 10.0 % | -0.0462 ± 0.0064 | -7.17 |
| 260–320 | 7.4–9.1 | no | 9.0 % | -0.0570 ± 0.0073 | -7.77 |
| 320–400 | 9.1–11.4 | no | 7.7 % | -0.0429 ± 0.0084 | -5.09 |
| 400–500 | 11.4–14.3 | no | 5.4 % | -0.0614 ± 0.0109 | -5.64 |
| 500–∞ | 14.3–∞ | no | 3.6 % | -0.1502 ± 0.0161 | -9.30 |

If the peripheral-preview account is right, Δ LL should be ~0 in the bins inside the sharp disc and grow with amplitude beyond it. A flat profile refutes it.

## Δ LL by fixation index

| fixation index | share of fix. | Δ LL (bits/fix) | t |
|---|---|---|---|
| 1 | 14.3 % | -0.0515 ± 0.0056 | -9.20 |
| 2 | 14.0 % | -0.0574 ± 0.0058 | -9.83 |
| 3 | 13.7 % | -0.0317 ± 0.0051 | -6.17 |
| 4–5 | 25.2 % | -0.0321 ± 0.0032 | -10.04 |
| 6–8 | 25.9 % | -0.0261 ± 0.0032 | -8.11 |
| 9–100 | 6.8 % | -0.0107 ± 0.0066 | -1.63 |
