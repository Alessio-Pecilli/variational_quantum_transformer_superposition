# Quantum Sequences TFIM — run_full_v5_Lhalf_L1 (Leonardo)

**HPC job:** `51084137` COMPLETED in 14m12s on Leonardo (`boost_usr_prod`).  
**Remote:** `results/quantum_sequences/LB_Lhalf_L1_T32_d16_ks1-2-3-5-6_L16_n5`

**Config:** d=16, T=32, k={1,2,3,5,6}, n_seeds=5, train=256, test=128, conv=100% (~99 epochs).

## Protocol

| Model | Training | Eval plots |
|---|---|---|
| k-QSA / k-CSA (mono+poly) | **L_B** | L_half, L_1 at final params |
| nl-CSA iso/gen | **L_half** | L_half train/test |
| nl-CSA iso/gen CE | **L_1** Shannon CE | L_1 train/test |

## Plots (requested)

1. `plots/train_Lhalf_vs_k.png` — train L_half (~3 for all)
2. `plots/test_Lhalf_vs_k.png` — test L_half (~3)
3. `plots/train_L1_vs_k.png` — train L_1 Shannon (~3.3)
4. `plots/test_L1_vs_k.png` — test L_1 (~3.3)
5. `plots/train_LB_curves_k3.png` — L_B training curves k=3 (kQSA/kCSA mono+poly)

Also kept: L_B train/test vs k + mono/poly L_B vs L_half comparisons.

## Key numbers (mean, 5 seeds)

### L_half train (~3, flat in k)
| model | k=1 | k=3 | k=6 |
|---|---:|---:|---:|
| k-QSA | 3.02 | 2.99 | 3.00 |
| k-CSA | 3.02 | 3.02 | 3.01 |
| poly-k-QSA | 3.01 | 3.00 | 3.02 |
| poly-k-CSA | 3.03 | 3.01 | 3.00 |
| nl-CSA iso | 3.03 | — | — |
| nl-CSA gen | 3.01 | — | — |

### L_1 train Shannon (~3.3, flat in k)
| model | k=1 | k=3 | k=6 |
|---|---:|---:|---:|
| k-QSA | 3.33 | 3.30 | 3.31 |
| k-CSA | 3.33 | 3.33 | 3.32 |
| poly-k-QSA | 3.31 | 3.30 | 3.32 |
| poly-k-CSA | 3.34 | 3.32 | 3.31 |
| nl-CSA iso CE | 3.31 | — | — |
| nl-CSA gen CE | 3.31 | — | — |

## Observations

- **L_half ≈ 3** and **L_1 ≈ 3.3** for all models (as expected); almost flat vs k.
- Jensen: L_half ≤ L_1 holds (3.0 ≤ 3.3).
- L_B still separates models (poly better than mono); L_half/L_1 do not.
- Test tracks train closely for both metrics.
