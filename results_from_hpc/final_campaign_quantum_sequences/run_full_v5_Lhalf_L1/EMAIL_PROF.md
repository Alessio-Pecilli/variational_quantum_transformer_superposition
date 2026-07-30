# Email draft — Quantum Sequences TFIM — L_half / L_1 + curves k=3 (Leonardo)

**To:** [professore]  
**Subject:** Quantum Sequences TFIM — L_half (~3) e L_1 Shannon (~3.3) train/test + curve L_B a k=3

---

Buongiorno Professore,

aggiornamento **Quantum Sequences** (TFIM, ansatz complesso) con i plot richiesti su **L_half** e **L_1**, più le training curves **L_B** a k=3.

Eseguito su **Leonardo CINECA** (job `51084137`, ~14 min, conv=100%).

Protocollo:
- **k-QSA / k-CSA:** training con **L_B**; eval di **L_half** e **L_1** ai parametri finali
- **nl-CSA soft Renyi:** training con **L_half**
- **nl-CSA soft CE:** training con **L_1** (Shannon CE uniforme)

Setup: `d=16`, `T=32`, `k={1,2,3,5,6}`, `n_seeds=5`.

**Pack GitHub:**  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/tree/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v5_Lhalf_L1

**Script / HPC:**  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/run_quantum_sequences_loss.py  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/hpc_quantum_sequences_loss.sh

---

## 1. Train L_half vs k (~3 per tutti)

https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v5_Lhalf_L1/plots/train_Lhalf_vs_k.png

## 2. Test L_half vs k

https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v5_Lhalf_L1/plots/test_Lhalf_vs_k.png

## 3. Train L_1 (Shannon CE) vs k (~3.3 per tutti)

https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v5_Lhalf_L1/plots/train_L1_vs_k.png

## 4. Test L_1 vs k

https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v5_Lhalf_L1/plots/test_L1_vs_k.png

## 5. Training curves L_B a k=3 (k-QSA / k-CSA)

https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v5_Lhalf_L1/plots/train_LB_curves_k3.png

---

**Lettura:** L_half ~3.0 e L_1 ~3.3, entrambe piatte in k (come atteso). L_B continua a separare mono vs poly. Test ≈ train.

Cordiali saluti,  
Alessio
