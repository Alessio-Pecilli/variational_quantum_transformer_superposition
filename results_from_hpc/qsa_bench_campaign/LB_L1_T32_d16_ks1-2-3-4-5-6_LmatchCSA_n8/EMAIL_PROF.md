# Email draft — L1/LB vs k with matched parameters (QSA≈CSA)

**Subject:** Re: L1 / L_B vs k — QSA e CSA a parità di parametri (~1024)

---

Buongiorno Professore,

come segnalato, nei plot precedenti QSA e CSA non erano confrontabili sul numero di parametri (QSA 384 vs CSA 1024). Abbiamo rifatto l’intera campagna su Leonardo con conteggio allineato.

**Parametri:** CSA/nl = **1024**; QSA = **1032** (L=43).  
1024 esatto per QSA non è raggiungibile: l’ansatz ha `2·L·n_qubits·3` angoli (multiplo di 24 con 4 qubit); 1032 è il valore più vicino.

Job Leonardo `53961135` (COMPLETED, ~9h). Pack:

https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/tree/PennyLaneG/results_from_hpc/qsa_bench_campaign/LB_L1_T32_d16_ks1-2-3-4-5-6_LmatchCSA_n8

Setup invariato: T=32, d=16, k=1..6, **8 seed**, pipeline condivisa.  
Plot L1: tutti i modelli, **senza chance**. Plot L_B: solo k-models.

---

### QUANTUM (TFIM)

**L1 train / test vs k**  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/qsa_bench_campaign/LB_L1_T32_d16_ks1-2-3-4-5-6_LmatchCSA_n8/quantum/plots/train_L1_vs_k.png  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/qsa_bench_campaign/LB_L1_T32_d16_ks1-2-3-4-5-6_LmatchCSA_n8/quantum/plots/test_L1_vs_k.png  

**L_B train / test vs k** (solo k-models)  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/qsa_bench_campaign/LB_L1_T32_d16_ks1-2-3-4-5-6_LmatchCSA_n8/quantum/plots/train_LB_vs_k.png  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/qsa_bench_campaign/LB_L1_T32_d16_ks1-2-3-4-5-6_LmatchCSA_n8/quantum/plots/test_LB_vs_k.png  

**Lettura:** a parità di budget parametrico, QSA e CSA sono molto più vicini. A k=6, L1 train ≈ **0.35** (k-QSA, n=1032) vs **0.29** (k-CSA, n=1024). Prima, con QSA a 384 parametri, lo stesso confronto era ~1.46 vs ~0.29. nl-iso ≈ 0.29, nl-gen ≈ 0.46.

| modello (n) | L1 k=1 | L1 k=3 | L1 k=6 |
|---|---:|---:|---:|
| k-QSA (1032) | 0.93±0.25 | 0.60±0.33 | **0.35±0.23** |
| poly-k-QSA (1032) | 1.00±0.18 | 0.45±0.21 | 0.41±0.29 |
| k-CSA (1024) | 0.89±0.02 | 0.36±0.02 | **0.29±0.03** |
| poly-k-CSA (1024) | 0.98±0.02 | 0.40±0.01 | 0.29±0.01 |
| nl-iso (1024) | **0.29±0.03** | | |
| nl-gen (1024) | 0.46±0.20 | | |

---

### CLASSICAL (Markov, ansatz complesso)

**L1 train / test vs k**  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/qsa_bench_campaign/LB_L1_T32_d16_ks1-2-3-4-5-6_LmatchCSA_n8/classical/plots/train_L1_vs_k.png  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/qsa_bench_campaign/LB_L1_T32_d16_ks1-2-3-4-5-6_LmatchCSA_n8/classical/plots/test_L1_vs_k.png  

**L_B train / test vs k** (solo k-models)  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/qsa_bench_campaign/LB_L1_T32_d16_ks1-2-3-4-5-6_LmatchCSA_n8/classical/plots/train_LB_vs_k.png  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/qsa_bench_campaign/LB_L1_T32_d16_ks1-2-3-4-5-6_LmatchCSA_n8/classical/plots/test_LB_vs_k.png  

Anche qui i conteggi sono in legenda (1032 vs 1024). CSA resta leggermente sotto; poly-kQSA migliora con k (~2.60 a k=6), mentre mono-kQSA sul classical peggiora con k.

Cordiali saluti,  
Alessio
