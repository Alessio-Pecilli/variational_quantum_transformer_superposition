# Quantum Sequences TFIM — run_full_v4_newloss

**Config:** d=16, T=32, k={1,2,3,5,6}, n_seeds=5, train=256, test=128, early stopping (~99 epochs).

## Loss protocol (new)

| Model family | Training objective | Test metric |
|---|---|---|
| k-QSA, k-CSA (mono+poly) | **L_B** = −log F, F = (T+1)/(2T)·μ/ζ | L_B (learned φ frozen) |
| nl-CSA iso/gen | **L_half_uniform** = −2 log((1/T)Σ√p_j) | L_half_uniform |

Reference implementation: `new_loss/qsa_classical_models_new_loss_quantum_seq.py`.

## Plots

1. `plots/train_loss_L_B_vs_k.png` — train loss vs k (+ nl horizontal)
2. `plots/test_loss_L_B_vs_k.png` — test loss vs k (+ nl horizontal)
3. `plots/train_mono_L_B_vs_L_half_uniform.png` — mono kQSA/kCSA: L_B vs L_half on train (final params)
4. `plots/train_poly_L_B_vs_L_half_uniform.png` — poly kQSA/kCSA: L_B vs L_half on train (final params)

## Key numbers (mean, 5 seeds)

### Train L_B (kQSA/kCSA) — k=1 vs k=6
| model | k=1 | k=6 |
|---|---:|---:|
| k-QSA | 6.63 | 6.87 |
| k-CSA | 6.53 | 6.80 |
| poly-k-QSA | 5.91 | 5.82 |
| poly-k-CSA | 5.97 | 5.87 |

### Train L_half_uniform (eval, same final params) — nearly flat ~3.0
| model | k=1 | k=6 |
|---|---:|---:|
| k-QSA | 3.02 | 3.00 |
| k-CSA | 3.02 | 3.01 |
| poly-k-QSA | 3.01 | 3.02 |
| poly-k-CSA | 3.03 | 3.00 |

### nl-CSA (k-independent)
| model | train L_half | test L_half |
|---|---:|---:|
| nl-CSA iso | 3.03 | 3.03 |
| nl-CSA gen | 3.01 | 3.01 |

## Observations

- **Convergence:** 100% on all jobs (~99 epochs).
- **L_B vs L_half:** on train, L_B ~5.8–6.9 while L_half ~3.0 (flat in k). The two objectives measure different quantities (O(1) circuit readout vs T-setting uniform Renyi); they track the same ranking qualitatively but not numerically 1:1.
- **poly advantage:** poly-k-QSA/k-CSA achieve lower L_B than mono at all k; mono L_B grows mildly with k, poly stays ~5.8–6.1.
- **Test tracks train:** test L_B within ~0.1–0.2 of train for mu-models; nl test ≈ train.
