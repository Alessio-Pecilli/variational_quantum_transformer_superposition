# Classical sequences + complex ansatz — COMPLETE

**Job:** `51363561` (~14 min, early stop ~99 epochs, 100% conv)  
**Config:** T=32, d=16, Markov ρ=0.8, complex RX-RY-RZ ansatz, continuous L_B / L_half / L_1 for all models (same as quantum TFIM campaign).

## Result (confirms the PTB check)

| metric | k-QSA/k-CSA | nl-CSA |
|---|---|---|
| L_half | **~3.00–3.01** | **~3.01** |
| L_1 | **~3.31–3.33** | **~3.31** |
| L_B (train) | ~6.4–6.5 | — |

Everyone sits near **~3** on L_half / L_1 — exactly like TFIM quantum-sequences.

**Conclusion:** the PTB classical campaign “gap” (mu ~0.5 vs nl ~7) was **metric mismatch** (continuous embedding CE vs discrete vocab CE), not complex vs real ansatz.

## Plots
- `plots/train_Lhalf_vs_k.png`, `plots/test_Lhalf_vs_k.png`
- `plots/train_L1_vs_k.png`, `plots/test_L1_vs_k.png`
- `plots/train_loss_L_B_vs_k.png`, `plots/test_loss_L_B_vs_k.png`
- `plots/train_mono_L_B_vs_L_half_uniform.png`, `plots/train_poly_L_B_vs_L_half_uniform.png`
- `plots/train_LB_curves_k3.png`
