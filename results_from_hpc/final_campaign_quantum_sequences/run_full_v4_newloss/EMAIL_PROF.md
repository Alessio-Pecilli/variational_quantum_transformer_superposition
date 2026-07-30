# Email draft — Quantum Sequences TFIM — nuova loss L_B / L_half (run_full_v4_newloss)

**To:** [professore]  
**Subject:** Quantum Sequences TFIM — nuova loss L_B (kQSA/kCSA) + L_half (nl-CSA) + confronto

---

Buongiorno Professore,

aggiornamento sulla campagna **Quantum Sequences** con le **nuove loss** da `new_loss/`:

- **k-QSA / k-CSA** (mono + poly): training su **L_B** = −log F, con F = (T+1)/(2T)·μ/ζ e fasi φ_j imparate
- **nl-CSA iso/gen**: training su **L_half_uniform** (ex Renyi CE uniforme), kernel softmax k-indipendente

Setup: `d=16`, `T=32`, `k={1,2,3,5,6}`, `n_seeds=5`, train=256, test=128, early stopping.

**Pack GitHub (branch PennyLaneG):**  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/tree/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v4_newloss

**Script:**  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/run_quantum_sequences_loss.py

**Riferimento loss:**  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/tree/PennyLaneG/new_loss

**Summary JSON:**  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v4_newloss/summary.json

---

## 1. Train loss vs k (L_B per mu-models, L_half per nl)

https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v4_newloss/plots/train_loss_L_B_vs_k.png

- poly-k-QSA migliore: L_B ~5.82 a k=6 vs k-QSA ~6.87
- nl-CSA iso ~3.03, gen ~3.01 (orizzontali)

---

## 2. Test loss vs k (stessa metrica del training)

https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v4_newloss/plots/test_loss_L_B_vs_k.png

- test allineato al train (gap tipico <0.15)
- φ_j imparate riusate a test senza ricalcolo

---

## 3. Confronto L_B vs L_half — mono (train, parametri finali)

https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v4_newloss/plots/train_mono_L_B_vs_L_half_uniform.png

- L_B ~6.5–6.9 (cresce leggermente con k)
- L_half ~3.0 (quasi piatta): le due metriche non coincidono numericamente ma L_half è stabile in k

---

## 4. Confronto L_B vs L_half — poly (train, parametri finali)

https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v4_newloss/plots/train_poly_L_B_vs_L_half_uniform.png

- L_B poly ~5.7–6.1, L_half ~3.0
- poly batte mono su L_B a tutti i k

---

## Numeri chiave (mean, 5 seed)

| model | train L_B k=1 | train L_B k=6 | train L_half k=6 | test L_B k=6 |
|---|---:|---:|---:|---:|
| k-QSA | 6.63 | 6.87 | 3.00 | 6.81 |
| k-CSA | 6.53 | 6.80 | 3.01 | 6.81 |
| poly-k-QSA | 5.91 | 5.82 | 3.02 | 5.90 |
| poly-k-CSA | 5.97 | 5.87 | 3.00 | 6.02 |
| nl-CSA iso | — | 3.03 | — | 3.03 |
| nl-CSA gen | — | 3.01 | — | 3.01 |

---

Cordiali saluti,  
Alessio
