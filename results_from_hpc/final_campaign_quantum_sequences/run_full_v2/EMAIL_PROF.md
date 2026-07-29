# Email draft — Quantum Sequences TFIM (d=16, T=32) — run_full_v2

**To:** [professore]  
**Subject:** Quantum Sequences TFIM (d=16, T=32) — curve ALIGNED LOSS VS K + test mu-loss e fidelity

---

Buongiorno Professore,

le scrivo con l’aggiornamento sulla campagna **Quantum Sequences** (TFIM, stati complessi, senza embedding classico e senza input Hamiltoniano).

Ho completato la run `run_full_v2` con:
- `d=16`, `T=32`, `k={1,2,3,5,6}`
- `n_seeds=5`, `train=256`, `test=128`
- modelli: `k-QSA` / `poly-k-QSA` / `k-CSA` / `poly-k-CSA` (mono + poly come richiesto)
- early stopping su train loss (`-log mu`) con best checkpoint
- test su parametri TFIM `(J,h)` mai visti in training

**Pack GitHub (branch PennyLaneG):**  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/tree/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v2

**Script:**  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/run_quantum_sequences_loss.py

**Summary JSON:**  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v2/summary.json

---

## 1. Plot train — ALIGNED LOSS VS K

Asse: `-log mu + log T` (train).

https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v2/plots/final_aligned_loss_vs_k.png

**Lettura:**
- tutte le curve mono crescono con `k`
- `poly-k-QSA` resta nettamente sotto le controparti mono
- a `k=6`: `poly-k-QSA` ~18.37 vs `k-QSA` ~24.27 vs `k-CSA` ~24.21

---

## 2. Plot test — mu-loss (NUOVO)

Asse: `-log mu + log T` su test (stessa metrica del train, TFIM hold-out).

https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v2/plots/final_aligned_loss_vs_k_test_mu.png

**Lettura importante:**
- train e test mu-loss sono molto allineati (gap tipico ~0.02–0.15)
- questo indica che il training converge e generalizza bene sulla metrica coerente `mu`
- il rumore osservato in precedenza sulla fidelity non era dovuto a mancata convergenza

---

## 3. Plot test — prediction fidelity (T+1)

Asse: `-log(prediction fidelity)` su step `T+1`.

https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v2/plots/final_aligned_loss_vs_k_test.png

**Lettura:**
- dopo convergenza, la fidelity test è quasi piatta su `k` (~2.70–2.83)
- miglior punto: `k-CSA` a `k=3` → `-logF = 2.698`
- `poly-k-QSA` migliore in poly-family a `k=3` → `-logF = 2.731`
- differenze tra modelli molto più piccole rispetto alla mu-loss

---

## 4. Diagnostica convergenza train vs test

https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v2/plots/training_convergence_train_vs_test.png

Convergenza:
- `conv=100%` su tutti i job
- epoche medie ~99 (early stopping attivo, best checkpoint usato)

---

## 5. Numeri chiave (mean ± std, 5 seed)

### Train aligned loss
| model | k=1 | k=3 | k=6 |
|---|---:|---:|---:|
| k-QSA | 15.48 | 19.80 | 24.27 |
| poly-k-QSA | 14.05 | 16.85 | 18.37 |
| k-CSA | 15.37 | 19.87 | 24.21 |
| poly-k-CSA | 14.08 | 16.93 | 18.45 |

### Test mu-loss (aligned)
| model | k=1 | k=3 | k=6 |
|---|---:|---:|---:|
| k-QSA | 15.40 | 19.76 | 24.18 |
| poly-k-QSA | 14.17 | 16.90 | 18.47 |
| k-CSA | 15.41 | 19.80 | 24.11 |
| poly-k-CSA | 14.16 | 17.09 | 18.62 |

### Test `-logF` (T+1)
| model | k=1 | k=3 | k=6 |
|---|---:|---:|---:|
| k-QSA | 2.805 | 2.779 | 2.784 |
| poly-k-QSA | 2.788 | 2.731 | 2.796 |
| k-CSA | 2.762 | 2.698 | 2.828 |
| poly-k-CSA | 2.761 | 2.813 | 2.754 |

---

## 6. Conclusione operativa

1. Setup Quantum Sequences funzionante e riproducibile.
2. QSA mono+poly implementato e confrontato.
3. Con early stopping la loss converge stabilmente.
4. La metrica test più informativa al momento è la **mu-loss** (coerente col training); la fidelity T+1 resta quasi piatta e più rumorosa.

Prossimo step suggerito (se concorda):
- aumentare seed/epoche solo sulla fidelity metric,
- oppure cambiare readout test (es. allineamento fasi ottimale su `phi_j`) per ridurre rumore fidelity.

Cordiali saluti,  
Alessio
