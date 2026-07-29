# FINAL campaign v2 (29 Jul 2026)

Revised professor plots: **μ panels**, **loss with 10 seeds**, **nl-CSA ~128 only with error bars**, **test hold-out loss**, **param counts in legend**, **appendix training curves at k=3**.

**Branch:** [`PennyLaneG`](https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/tree/PennyLaneG)  
**Pack folder:** [`results_from_hpc/final_campaign_v2/`](https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/tree/PennyLaneG/results_from_hpc/final_campaign_v2)

## Setup

| Part | Setting |
|------|---------|
| **Loss** | T=16, d=8, k=1,2,3,5,6, **10 seeds**, QSA L=16 (96 angles), CSA 128 matrices |
| **Loss models** | k-QSA, k-CSA, poly-k-QSA, poly-k-CSA, **nl-CSA iso~128 + gen~128 only** |
| **Test split** | PTB hold-out, `test_data_seed=4242`, 200 sentences |
| **μ panels** | target_loss=3.8, T≤32, k=1,2,3,5,6, mono+poly, **5 seeds** (42–46), advantage line k/C(d+k−1,k) |
| **HPC** | Leonardo job `50570347` (loss, ~10h), mu chain `50570724` (~10 min) |

## Main results (aligned loss = −log μ + log T)

- **Monomial grows with k:** k-QSA 4.28→6.62 (Δ=+2.34); k-CSA 4.71→7.54 (Δ=+2.83).
- **Polynomial nearly flat:** poly-k-QSA 4.20→4.88 (Δ=+0.68); poly-k-CSA 4.71→5.36 (Δ=+0.65).
- **nl-CSA ~128:** iso 6.58±0.03, gen 6.93±0.78 — above best μ-models; gen has wider seed variance.
- **Train vs test:** aligned loss on hold-out matches train within numerical noise.

## Files in this pack

```
final_campaign_v2/
  README.md
  EMAIL_PROF.md          ← copy/paste email draft
  loss/
    LOSS_CHECK.txt
    summary.json         ← full aggregates + per-seed curves
    plots/
      final_aligned_loss_vs_k.png
      final_aligned_loss_vs_k_test.png
      final_training_curves_k3_appendix.png
      … (bars/curves per k)
  mu/
    mu_vs_T.png
    mu_vs_d.png
    manifest.json
```

## Leonardo paths (full checkpoints)

- Loss: `results/final_loss/v2_T16_d8_ks1-2-3-5-6_L16_n10_test/`
- μ: `results/study/final_obar_T32_ks1-2-3-5-6_tl3.8/`

## Code

- [`run_final_loss.py`](https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/run_final_loss.py)
- [`run_study.py`](https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/run_study.py)
- [`hpc_final_loss_v2.sh`](https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/hpc_final_loss_v2.sh)
- [`hpc_final_obar.sh`](https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/hpc_final_obar.sh)

## Previous pack (v1, 8 seeds, ōbar panels)

[`results_from_hpc/final_campaign_definitive/`](../final_campaign_definitive/)
