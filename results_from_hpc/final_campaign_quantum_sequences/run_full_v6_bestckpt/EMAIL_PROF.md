# Email draft — Quantum Sequences v6 (best-ckpt fix)

**To:** [professore]  
**Subject:** Quantum Sequences TFIM — rifatti plot L_B / L_half / L_1 (fix best-checkpoint)

---

Buongiorno Professore,

abbiamo rifatto la campagna quantum-sequences su Leonardo dopo aver corretto un bug sul **best-checkpoint**: nei plot “finali” di v5 la L_B riportata era quella di **epoca 0**, non il minimo raggiunto in training (le curve invece erano corrette).

**Job:** `51367020` (~41 min) — COMPLETED  
**Pack:** https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/tree/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v6_bestckpt

### L_half / L_1 (li teniamo)
Train L_half: https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v6_bestckpt/plots/train_Lhalf_vs_k.png  
Test L_half: https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v6_bestckpt/plots/test_Lhalf_vs_k.png  
Train L_1: https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v6_bestckpt/plots/train_L1_vs_k.png  
Test L_1: https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v6_bestckpt/plots/test_L1_vs_k.png  

Con il checkpoint corretto, **poly-k-QSA** scende sotto ~3 (L_half ≈ 2.39 a k=3); CSA resta ~3.0–3.01; nl-gen ≈ 2.83.

### L_B vs k + curves k=3
Train L_B vs k: https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v6_bestckpt/plots/train_loss_L_B_vs_k.png  
Curves k=3: https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v6_bestckpt/plots/train_LB_curves_k3.png  

**Sul punto “i poly scendono poco”:**
- **poly-k-QSA:** no — impara chiaro (~5.7 → ~3.15); i marker finali ora coincidono con la fine della curva (~3.15, non più ~5.7).
- **poly-k-CSA:** sì, quasi piatto (~5.9); il vantaggio sul mono (~6.7) è soprattutto **init migliore**, poca dinamica successiva (come il mono CSA).

Cordiali saluti,  
Alessio
