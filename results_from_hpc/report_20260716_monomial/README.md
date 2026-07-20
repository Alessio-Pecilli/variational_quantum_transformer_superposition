# HPC results — monomial Section-2 (Leonardo job finished 16 Jul 2026 ~18:05)
# Poly LCU abandoned; this is the reference Friday report + replot with log-y vs d.

## Full complexity table (ō = mean |a s^k|, T/d sweeps)

| label | T | d | k | ō | Haar d^(-(k+1)/2) | advantage | ō/Haar | loss | >Haar | >adv |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| T2_d16_k2 | 2 | 16 | 2 | 0.299136 | 0.015625 | 0.125 | 19.14 | 2.424 | yes | yes |
| T4_d16_k2 | 4 | 16 | 2 | 0.200828 | 0.015625 | 0.125 | 12.85 | 3.226 | yes | yes |
| T8_d16_k2 | 8 | 16 | 2 | 0.170267 | 0.015625 | 0.125 | 10.90 | 3.552 | yes | yes |
| T16_d16_k2 | 16 | 16 | 2 | 0.094199 | 0.015625 | 0.125 | 6.03 | 4.738 | yes | no |
| T32_d16_k2 | 32 | 16 | 2 | 0.092227 | 0.015625 | 0.125 | 5.90 | 4.778 | yes | no |
| T2_d16_k3 | 2 | 16 | 3 | 0.281290 | 0.003906 | 0.0663 | 72.01 | 2.549 | yes | yes |
| T4_d16_k3 | 4 | 16 | 3 | 0.163230 | 0.003906 | 0.0663 | 41.79 | 3.645 | yes | yes |
| T8_d16_k3 | 8 | 16 | 3 | 0.126816 | 0.003906 | 0.0663 | 32.47 | 4.147 | yes | yes |
| T16_d16_k3 | 16 | 16 | 3 | 0.069428 | 0.003906 | 0.0663 | 17.77 | 5.355 | yes | yes |
| T32_d16_k3 | 32 | 16 | 3 | 0.068032 | 0.003906 | 0.0663 | 17.42 | 5.387 | yes | yes |
| T32_d2_k2 | 32 | 2 | 2 | 0.398930 | 0.353553 | 1.000 | 1.13 | 2.935 | yes | no |
| T32_d4_k2 | 32 | 4 | 2 | 0.293566 | 0.125000 | 0.500 | 2.35 | 2.460 | yes | no |
| T32_d8_k2 | 32 | 8 | 2 | 0.266611 | 0.044194 | 0.250 | 6.03 | 2.648 | yes | yes |
| T32_d16_k2 | 32 | 16 | 2 | 0.092227 | 0.015625 | 0.125 | 5.90 | 4.778 | yes | no |
| T32_d32_k2 | 32 | 32 | 2 | 0.019911 | 0.005524 | 0.0625 | 3.60 | 7.845 | yes | no |
| T32_d2_k3 | 32 | 2 | 3 | 0.338295 | 0.250000 | 1.500 | 1.35 | 2.173 | yes | no |
| T32_d4_k3 | 32 | 4 | 3 | 0.238125 | 0.062500 | 0.530 | 3.81 | 2.878 | yes | no |
| T32_d8_k3 | 32 | 8 | 3 | 0.205606 | 0.015625 | 0.1875 | 13.16 | 3.169 | yes | yes |
| T32_d16_k3 | 32 | 16 | 3 | 0.068032 | 0.003906 | 0.0663 | 17.42 | 5.387 | yes | yes |
| T32_d32_k3 | 32 | 32 | 3 | 0.012418 | 0.000977 | 0.0234 | 12.71 | 8.790 | yes | no |

**Issue flagged by the professor:** at d=16/32 (T=32) loss is still 4.8–8.8, so ō sits near/below advantage even though above Haar. Next HPC pass retrains those points to loss ≤ 2.5 (cap 2000 epochs), still with T=32 ≥ d.

## Plots (regenerated)

- `mean_O_vs_T.png` — linear y (unchanged style)
- `mean_O_vs_d.png` — **log y** + Haar + advantage
- CSV: `full_complexity_table.csv`
