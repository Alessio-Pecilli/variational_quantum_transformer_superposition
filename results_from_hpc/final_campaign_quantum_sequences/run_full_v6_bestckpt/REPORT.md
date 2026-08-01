# Quantum Sequences TFIM — run_full_v6_bestckpt (Leonardo)

**HPC job:** `51367020` COMPLETED in ~41 min (`boost_usr_prod`).  
**Remote:** `results/quantum_sequences/LB_Lhalf_L1_T32_d16_ks1-2-3-5-6_L16_n5_bestckpt_v2`

**Fix vs v5:** best-checkpoint no longer frozen at epoch-0 (`train_L_B` now tracks `min(history)`).

**Config:** d=16, T=32, k={1,2,3,5,6}, n_seeds=5, train=256, test=128.

## Protocol

| Model | Training | Eval |
|---|---|---|
| k-QSA / k-CSA (mono+poly) | **L_B** | L_half, L_1 at **best** params |
| nl-CSA iso/gen | **L_half** | L_half |
| nl-CSA iso/gen CE | **L_1** | L_1 |

## Key numbers (train, mean ± across seeds)

### L_B (k=1 / 3 / 6)
| model | k=1 | k=3 | k=6 |
|---|---:|---:|---:|
| k-QSA | 3.89 | 4.93 | 4.93 |
| poly-k-QSA | **3.06** | **3.15** | **3.17** |
| k-CSA | 6.45 | 6.74 | 6.68 |
| poly-k-CSA | 5.89 | 5.87 | 5.78 |

### L_half @ k=3
| model | L_half |
|---|---:|
| poly-k-QSA | **2.39** |
| k-QSA | 2.70 |
| k-CSA / poly-k-CSA | ~3.01 |
| nl-CSA iso | 3.02 |
| nl-CSA gen | 2.83 |

## Curves note (poly)
- **poly-k-QSA:** clear learning (~5.7 → ~3.15); final L_B vs k now matches the curve end.
- **poly-k-CSA:** still almost flat (~5.9); advantage vs mono CSA is mostly **better init**, little further descent.
