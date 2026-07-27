# Email draft — FINAL campaign results (copy/paste)

**To:** [professore]  
**Subject:** FINAL campaign Leonardo — loss multi-seed + ōbar (T≤32) — risultati e link GitHub

---

Buongiorno Professore,

ho completato su Leonardo la campagna **FINAL** come da brief (loss multi-seed con barre d’errore; ōbar con target≈3.8; mono+poly sullo stesso plot; k=1,2,3,5,6; T fino a 32).

## Repository e branch

- Repo: https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition  
- Branch: https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/tree/PennyLaneG  

## Pack risultati (plot + summary + report)

Cartella principale:

https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/tree/PennyLaneG/results_from_hpc/final_campaign_definitive

README del pack (setup + numeri + caveat):

https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_definitive/README.md

Report loss / convergenza / punti ōbar sopra target:

https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_definitive/campaign/LOSS_CHECK.txt

### Figure loss (allineata: −log μ + log T; nl = Renyi)

- **Figura principale vs k:**  
  https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_definitive/loss/plots/final_aligned_loss_vs_k.png  
- Tutti i plot loss (barre e curve per k):  
  https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/tree/PennyLaneG/results_from_hpc/final_campaign_definitive/loss/plots  
- Summary JSON:  
  https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_definitive/loss/summary.json  

### Figure ōbar (MAX_T=32, mono+poly)

- ō vs T: https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_definitive/obar/mean_O_vs_T.png  
- ō vs T by d: https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_definitive/obar/mean_O_vs_T_by_d.png  
- ō vs d: https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_definitive/obar/mean_O_vs_d.png  
- ō vs d by T: https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_definitive/obar/mean_O_vs_d_by_T.png  
- Manifest: https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_definitive/obar/manifest.json  

## Setup in sintesi

| Parte | Setting |
|-------|---------|
| Loss | T=16, d=8, k=1,2,3,5,6, **8 seeds**, QSA L=16 |
| Loss models | k-QSA, k-CSA, poly-k-QSA, poly-k-CSA, nl-iso~288, nl-iso~128, nl-gen~128 |
| Ōbar | target_loss=3.8, max 2000 ep, T≤32, k=1,2,3,5,6, mono+poly |
| T=64 | **non incluso** (PTB ~3 frasi uniche) |

## Risultati principali (aligned loss, mean±std)

- **Monomio:** loss cresce con k (QSA L=16: 4.25 → 6.60, Δ≈+2.35; CSA: 4.74 → 7.34, Δ≈+2.60).  
- **Poly:** crescita molto più piatta (poly-QSA: 4.24 → 4.87, Δ≈+0.63; poly-CSA: 4.83 → 5.38, Δ≈+0.55).  
- **nl-CSA:** ~6.5–6.7, ancora sopra il best μ (~4.24) con il budget usato — da trattare come riferimento provvisorio.  
- **Ōbar:** 90 punti, 0 errori; 20/90 ancora sopra target 3.8 (principalmente k alti / d bassi; elenco in LOSS_CHECK).

## Codice (entry point)

- Loss runner: https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/run_final_loss.py  
- Ōbar grid: https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/run_study.py  
- HPC loss: https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/hpc_final_loss.sh  
- HPC ōbar: https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/hpc_final_obar.sh  
- Catena submit: https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/hpc_submit_finals_chain.sh  

## Contesto precedente (stesso repo / branch)

Indice pack HPC: https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/README.md  

- Debug v3 (T=16>d=8, 1 seed): https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/tree/PennyLaneG/results_from_hpc/debug_campaign_v3_T16_d8_seed42  
- K-study (allineamento Renyi/logT + μ/μ0): https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/tree/PennyLaneG/results_from_hpc/k_study_T16_d8_seed42  
- Report monomio 16/17 Jul: https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/tree/PennyLaneG/results_from_hpc/report_20260716_monomial  

Resto a disposizione per commenti o figure aggiuntive.

Cordiali saluti,  
Alessio
