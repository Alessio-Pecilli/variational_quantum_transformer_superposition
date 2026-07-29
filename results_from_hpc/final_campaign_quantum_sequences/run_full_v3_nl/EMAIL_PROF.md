# Email draft — Quantum Sequences TFIM (d=16, T=32) — run_full_v3_nl

**To:** [professore]  
**Subject:** Quantum Sequences TFIM (d=16, T=32) — ALIGNED LOSS VS K + nl-CSA iso/gen

---

Buongiorno Professore,

le scrivo con l’aggiornamento sulla campagna **Quantum Sequences** (TFIM, stati complessi, senza embedding classico e senza input Hamiltoniano).

Ho completato la run `run_full_v3_nl` con:
- `d=16`, `T=32`, `k={1,2,3,5,6}`
- `n_seeds=5`, `train=256`, `test=128`
- modelli mu: `k-QSA` / `poly-k-QSA` / `k-CSA` / `poly-k-CSA`
- **nl-CSA iso** e **nl-CSA gen** (kernel softmax, k-indipendenti, linee orizzontali)
- early stopping su train loss (`-log mu`) con best checkpoint
- test su parametri TFIM `(J,h)` mai visti in training
- protocollo fasi: a test si riusano le **stesse `phi_j` imparate** in training (non ricalcolate)

**Pack GitHub (branch PennyLaneG):**  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/tree/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v3_nl

**Script:**  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/run_quantum_sequences_loss.py

**Summary JSON:**  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v3_nl/summary.json

---

## 1. Plot train — ALIGNED LOSS VS K (+ nl)

Asse: `-log mu + log T` (train).

https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v3_nl/plots/final_aligned_loss_vs_k.png

**Lettura:**
- curve mono crescono con `k`
- `poly-k-QSA` resta nettamente sotto le controparti mono
- **nl-CSA iso** ~16.52 e **nl-CSA gen** ~15.66 (orizzontali, k-indep.)
- a bassi `k` i poly-mu battono nl; a `k` alti i mono stanno sopra nl

---

## 2. Plot test — mu-loss (+ nl)

Asse: `-log mu + log T` su test (stessa metrica del train, TFIM hold-out).

https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v3_nl/plots/final_aligned_loss_vs_k_test_mu.png

**Lettura:**
- train e test mu-loss allineati (gap tipico piccolo)
- nl-CSA test: iso ~16.62, gen ~15.75
- conferma: training convergente; a test si usano le **phi imparate**

---

## 3. Plot test — prediction fidelity T+1 (+ nl)

Asse: `-log(prediction fidelity)`.

https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v3_nl/plots/final_aligned_loss_vs_k_test.png

**Lettura:**
- fidelity test quasi piatta (~2.70–2.83), anche per nl (~2.79)
- la fidelity per-step è phase-invariant: **non usa `phi_j`**

---

## 4. Protocollo fasi `phi_j` (chiarimento)

Le `phi_j` sono parametri condivisi di lunghezza `T`, ottimizzati sul train e **riusati identici a test** nella mu-loss.

Non si ricalcola `phi_j* = -arg(A_j)` a test (sarebbe un allineamento oracolo, non il modello imparato).  
La fidelity T+1 è un readout per-step: una fase globale su `z_j` si cancella, quindi `phi` non entra.

---

## 5. Numeri chiave (mean, 5 seed)

### nl-CSA (k-indep.)
| model | train | test mu | test -logF |
|---|---:|---:|---:|
| nl-CSA iso | 16.52 | 16.62 | 2.791 |
| nl-CSA gen | 15.66 | 15.75 | 2.793 |

### Train aligned loss (mu-models)
| model | k=1 | k=3 | k=6 |
|---|---:|---:|---:|
| k-QSA | 15.48 | 19.80 | 24.27 |
| poly-k-QSA | 14.05 | 16.85 | 18.37 |
| k-CSA | 15.37 | 19.87 | 24.21 |
| poly-k-CSA | 14.08 | 16.93 | 18.45 |

---

Cordiali saluti,  
Alessio
