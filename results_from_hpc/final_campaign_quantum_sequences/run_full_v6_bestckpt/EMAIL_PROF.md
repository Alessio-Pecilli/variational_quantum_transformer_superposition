# Email draft — risposta su L_half/L_1 + training curves poly

**To:** [professore]  
**Subject:** Re: L_half/L_1 ok + training curves poly (quantum sequences, pack aggiornato)

---

Buongiorno Professore,

perfetto: **teniamo i plot L_half e L_1** — sono quelli che volevamo.

Pack aggiornato su Leonardo (job `51367020`, best-checkpoint corretto):  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/tree/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v6_bestckpt

**L_half train / test**  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v6_bestckpt/plots/train_Lhalf_vs_k.png  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v6_bestckpt/plots/test_Lhalf_vs_k.png  

**L_1 (Shannon CE) train / test**  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v6_bestckpt/plots/train_L1_vs_k.png  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v6_bestckpt/plots/test_L1_vs_k.png  

**L_B train / test (finali veri, post-fix)**  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v6_bestckpt/plots/train_loss_L_B_vs_k.png  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v6_bestckpt/plots/test_loss_L_B_vs_k.png  

**Training curves L_B a k=3**  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v6_bestckpt/plots/train_LB_curves_k3.png  

**Confronto L_B vs L_half (parametri finali, train)**  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v6_bestckpt/plots/train_mono_L_B_vs_L_half_uniform.png  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v6_bestckpt/plots/train_poly_L_B_vs_L_half_uniform.png  

---

### Sul dubbio: «i poly scendono così poco / sembrano bloccati?»

Aveva ragione a segnalarlo: c’era un’incongruenza tra curve e plot finali.

**Causa:** bug nel best-checkpoint. La condizione di miglioramento usava `tol · |best_loss|` con `best_loss = +∞` al primo step (`∞ < ∞` sempre falso), quindi i parametri “finali” salvati restavano quelli **iniziali**. Le **training curves** erano già corrette (loss live); i **plot L_B vs k** no (mostravano l’init). Ora è corretto e i due sono coerenti.

**Numeri a k=3 (media 5 seed), dopo il fix:**

| modello | L_B start | L_B fine | Δ |
|---|---:|---:|---:|
| k-QSA | ~6.8 | ~4.93 | −1.9 |
| **poly-k-QSA** | ~5.7 | **~3.15** | **−2.5** |
| k-CSA | ~6.9 | ~6.74 | ~0 |
| poly-k-CSA | ~5.9 | ~5.87 | ~0 |

**Lettura:**
1. **poly-k-QSA non è bloccato**: scende più del mono (−2.5 vs −1.9) e arriva a ~3.15. Parte già più basso (vantaggio del kernel poly all’init) e **continua a imparare** lungo ~550–600 epoche.
2. **poly-k-CSA / k-CSA**: sì, curve quasi piatte — poca dinamica sotto L_B con questo LR/schedule. Il fatto che il poly resti “migliore” del mono nei finali è soprattutto **init migliore** (~5.9 vs ~6.9), non learning successivo. Non è un freeze del solo poly: è un problema di ottimizzazione della famiglia CSA.
3. Quindi il paradosso «curve piatte ma finali buoni» era in parte artefatto del bug di reporting; con il fix, per **QSA-poly** curve e finali raccontano la stessa storia.

Cordiali saluti,  
Alessio
