# Email draft — L_B loss campaign (classical PTB) COMPLETE

**To:** [professore]  
**Subject:** Campagna L_B classical — plot finali (L_B, L_half, L_1, curves k=3)

---

Buongiorno Professore,

campagna completata su Leonardo (10 seed, T=16, d=8, k=1,2,3,5,6). Ansatz classical-sequence non complesso.

**Pack:** https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/tree/PennyLaneG/results_from_hpc/final_campaign_LB

### Setup loss
- **k-QSA / k-CSA:** train su **L_B** = −log F  
- **nl-CSA Renyi:** train su **L_half** (plot L_half)  
- **nl-CSA CE:** train su **L_1** Shannon (plot L_1)

---

## 1. Loss vs k — train (L_B)

https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_LB/plots/final_aligned_loss_vs_k.png

poly-k-QSA migliore (~0.59–0.61, quasi piatta in k). k-QSA ~0.68→0.78. k-CSA più alto e più variabile.

## 2. Loss vs k — test

https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_LB/plots/final_aligned_loss_vs_k_test.png

Train≈test per i mu-models (come atteso sulla loss geometrica).

## 3–4. Confronto L_B vs L_half (parametri finali, train)

Mono: https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_LB/plots/final_LB_vs_Lhalf_mono_train.png  

Poly: https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_LB/plots/final_LB_vs_Lhalf_poly_train.png

L_half (eval) tipicamente **sotto** L_B (es. poly-k-QSA k=6: L_B≈0.59, L_half≈0.44).

## 5. L_half vs k — train / test

Train: https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_LB/plots/final_Lhalf_vs_k_train.png  

Test: https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_LB/plots/final_Lhalf_vs_k_test.png

## 6. L_1 (Shannon CE) vs k — train / test

Train: https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_LB/plots/final_L1_vs_k_train.png  

Test: https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_LB/plots/final_L1_vs_k_test.png

### Check importante (L_half / L_1 “basse” per k-QSA vs nl)

**Non è un bug.** Su PTB le due famiglie usano metriche diverse:
- **k-QSA/k-CSA:** CE continua su embedding in R^d → L_1 ~0.5–0.9  
- **nl-CSA:** CE discreta next-token sul vocabolario (V≈3045, log V≈8.0) → L_1 ~7.3  

Sul TFIM quantum-sequences invece **tutti** usano lo stesso readout continuo → L_half/L_1 ~3 per tutti (fair).

Sanity check: `check_LB_Lhalf_L1.py` (Jensen OK; scala nl ≈ log V).

Follow-up lanciato: **classical Markov sequences + complex ansatz** (stessa metrica continua del quantum), per isolare data vs ansatz — smoke locale già dà L_half≈3 / L_1≈3.3 per kQSA/kCSA/nl.

## 7. Training curves L_B a k=3 (solo k-QSA/k-CSA, senza nl)

https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_LB/plots/final_LB_training_curves_k3.png

---

Branch: https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/tree/PennyLaneG

Cordiali saluti,  
Alessio
