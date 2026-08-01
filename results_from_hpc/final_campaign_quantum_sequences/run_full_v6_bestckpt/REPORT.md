# Quantum Sequences TFIM — run_full_v6_bestckpt (Leonardo)

**HPC job:** `51367020` COMPLETED in 40m50s.  
**Remote:** `results/quantum_sequences/LB_Lhalf_L1_T32_d16_ks1-2-3-5-6_L16_n5_bestckpt_v2`  
**Fix:** best-checkpoint no longer freezes at init (`tol*inf` bug).

**Config:** d=16, T=32, k={1,2,3,5,6}, n_seeds=5, train=256, test=128.

## Key change vs v5

| | v5 (bug) | v6 (fixed) |
|---|---|---|
| reported final L_B | = **init** loss | = **best trained** checkpoint |
| poly-k-QSA k=3 | ~5.74 (fake) | **~3.15** (real) |
| k-QSA k=3 | ~6.77 (fake) | **~4.93** (real) |

## Train L_B (true finals, mean 5 seeds)

| model | k=1 | k=3 | k=6 |
|---|---:|---:|---:|
| k-QSA | 3.89 | 4.93 | 4.93 |
| k-CSA | 6.45 | 6.74 | 6.68 |
| poly-k-QSA | **3.06** | **3.15** | **3.17** |
| poly-k-CSA | 5.89 | 5.87 | 5.78 |

## L_half / L_1 (eval at best params)

- poly-k-QSA: L_half ~2.37–2.41, L_1 ~2.58–2.64 (below ~3)
- mono k-QSA: L_half ~2.54–2.78, L_1 ~2.79–3.08
- CSA / nl: still ~3.0 (L_half) / ~3.3 (L_1)

## Training curves k=3

- **poly-k-QSA:** clear descent 5.7 → 3.15 over ~550–600 epochs
- **k-QSA:** descent 6.8 → 4.9 over 400 epochs (hits max_epochs)
- **CSA mono/poly:** still nearly flat (optimization issue, not checkpoint bug)

## Plots

1. `train_Lhalf_vs_k.png` / `test_Lhalf_vs_k.png`
2. `train_L1_vs_k.png` / `test_L1_vs_k.png`
3. `train_LB_curves_k3.png`
4. `train_loss_L_B_vs_k.png` / `test_loss_L_B_vs_k.png` (now true finals)
5. mono/poly L_B vs L_half comparisons
