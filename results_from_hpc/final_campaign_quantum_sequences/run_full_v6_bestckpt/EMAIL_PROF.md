# Email draft — Quantum Sequences TFIM — fix best-checkpoint (v6)

**To:** [professore]  
**Subject:** Quantum Sequences TFIM — fix checkpoint + L_B/L_half/L_1 aggiornati (Leonardo)

---

Buongiorno Professore,

aggiornamento **Quantum Sequences** dopo un bug nel best-checkpoint: la loss “finale” riportata era in realtà quella **iniziale** (`tol·∞` bloccava il salvataggio dei parametri). Corretto e rilanciato su Leonardo (job `51367020`, ~41 min).

Con i veri checkpoint trainati:
- **poly-k-QSA** scende chiaramente (es. k=3: L_B ~5.7 → **~3.15**)
- **k-QSA** scende ~6.8 → **~4.9**
- **k-CSA / poly-k-CSA** restano quasi piatti (problema di ottimizzazione, non di reporting)
- **L_half / L_1** per poly-k-QSA scendono sotto ~3 (L_half ~2.4, L_1 ~2.6)

Setup invariato: `d=16`, `T=32`, `k={1,2,3,5,6}`, `n_seeds=5`.

**Pack:**  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/tree/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v6_bestckpt

**Plot principali:**  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v6_bestckpt/plots/train_loss_L_B_vs_k.png  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v6_bestckpt/plots/test_loss_L_B_vs_k.png  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v6_bestckpt/plots/train_Lhalf_vs_k.png  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v6_bestckpt/plots/test_Lhalf_vs_k.png  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v6_bestckpt/plots/train_L1_vs_k.png  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v6_bestckpt/plots/test_L1_vs_k.png  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v6_bestckpt/plots/train_LB_curves_k3.png

Cordiali saluti,  
Alessio
