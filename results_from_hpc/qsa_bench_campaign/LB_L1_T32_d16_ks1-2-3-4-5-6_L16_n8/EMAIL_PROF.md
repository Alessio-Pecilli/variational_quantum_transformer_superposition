# Email draft — campagna HPC shared-pipeline (classical + quantum, k≤6, 8 seed)

**Subject:** Re: L1 / L_B vs k (classical + quantum) — pipeline condivisa, 8 seed, k=1..6

---

Buongiorno Professore,

come richiesto, abbiamo rifatto su Leonardo il confronto **classical + quantum** con pipeline condivisa (`qsa_bench`: un solo forward / una sola metrica), mediato su **8 seed**, **k = 1..6**.

Job Leonardo `52145458` (COMPLETED, ~2h). Pack:

https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/tree/PennyLaneG/results_from_hpc/qsa_bench_campaign/LB_L1_T32_d16_ks1-2-3-4-5-6_L16_n8

Setup: T=32, d=16, layers=16, train 256 / test 128.  
k-models allenati su \(\mathcal L_B\); nl-CSA su \(L_1\); **asse comune riportato = \(L_1\)**.  
Nei plot **L1 non c’è la chance line**; nei plot **L_B ci sono solo i k-models** (nl non usa quella loss).

---

### QUANTUM (TFIM) — i 4 plot richiesti

**L1 train / test vs k** (tutti i modelli, no chance)  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/qsa_bench_campaign/LB_L1_T32_d16_ks1-2-3-4-5-6_L16_n8/quantum/plots/train_L1_vs_k.png  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/qsa_bench_campaign/LB_L1_T32_d16_ks1-2-3-4-5-6_L16_n8/quantum/plots/test_L1_vs_k.png  

**L_B train / test vs k** (solo k-models)  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/qsa_bench_campaign/LB_L1_T32_d16_ks1-2-3-4-5-6_L16_n8/quantum/plots/train_LB_vs_k.png  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/qsa_bench_campaign/LB_L1_T32_d16_ks1-2-3-4-5-6_L16_n8/quantum/plots/test_LB_vs_k.png  

**Lettura (risposta ai due dubbi precedenti):**
1. **kCSA non sta “molto sopra” kQSA**: su quantum sta **sotto** (più parametri: 1024 vs 384). A k=3, L1 train ≈ **0.36** (k-CSA) vs **1.35** (k-QSA); poly-kQSA ≈ **0.70**. Il gap anomalo post-cambio loss era un problema di pipeline, non strutturale.
2. **nlCSA non è bloccato a ~3**: nl-iso L1 ≈ **0.29**, nl-gen ≈ **0.46** (imparano chiaramente).

Numeri tipici L1 train (media ± std, 8 seed):

| modello | k=1 | k=3 | k=6 |
|---|---:|---:|---:|
| k-QSA | 1.85±0.38 | 1.35±0.46 | 1.46±0.42 |
| poly-k-QSA | 1.31±0.06 | 0.70±0.05 | **0.58±0.05** |
| k-CSA | 0.89±0.02 | **0.36±0.02** | **0.29±0.03** |
| poly-k-CSA | 0.98±0.02 | 0.40±0.01 | 0.29±0.01 |
| nl-iso (k-indep.) | **0.29±0.03** | | |
| nl-gen (k-indep.) | 0.46±0.20 | | |

---

### CLASSICAL (Markov, ansatz complesso) — gli altri 4 plot

**L1 train / test vs k** (tutti i modelli, no chance)  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/qsa_bench_campaign/LB_L1_T32_d16_ks1-2-3-4-5-6_L16_n8/classical/plots/train_L1_vs_k.png  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/qsa_bench_campaign/LB_L1_T32_d16_ks1-2-3-4-5-6_L16_n8/classical/plots/test_L1_vs_k.png  

**L_B train / test vs k** (solo k-models)  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/qsa_bench_campaign/LB_L1_T32_d16_ks1-2-3-4-5-6_L16_n8/classical/plots/train_LB_vs_k.png  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/qsa_bench_campaign/LB_L1_T32_d16_ks1-2-3-4-5-6_L16_n8/classical/plots/test_LB_vs_k.png  

Anche qui **k-CSA ≤ k-QSA** su L1/L_B (es. k=3: L1 ≈ 2.57 CSA vs ≈ 3.07 QSA).  
nl-iso ≈ 2.69; nl-gen resta più alto (~3.01) — sul classical il guadagno di nl-gen è più debole che sul quantum.

---

In sintesi: con forward/metriche allineati, su **quantum** i due segnali sospetti (CSA “rotto” / nl fermo a ~3) **non si riproducono**. I plot sopra sono quelli da tenere per la discussione.

Cordiali saluti,  
Alessio
