# Email draft — FINAL campaign v2 revision 2 (copy/paste)

**To:** [professore]  
**Subject:** FINAL v2 — plot aggiornati (log μ₀/μ, small-data test, μ vs d, curves k=5)

---

Buongiorno Professore,

ho applicato le ultime modifiche ai plot. Tutto su GitHub:

**Pack:** https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/tree/PennyLaneG/results_from_hpc/final_campaign_v2

---

## 1. ALIGNED LOSS VS k (invariato + nuovi plot)

**Train (10 seed, d=8):**  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_v2/loss/plots/final_aligned_loss_vs_k.png

**Test hold-out:**  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_v2/loss/plots/final_aligned_loss_vs_k_test.png

### NUOVO: log(μ₀/μ) vs k

Metrica di miglioramento rispetto all'inizializzazione random:  
**log(μ₀/μ) = L_final − L₀**, dove L = −log μ + log T (nl-CSA: Δ Renyi).

https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_v2/loss/plots/final_log_mu_ratio_vs_k.png

Qui si vede chiaramente il comportamento crescente in k per il monomio (miglioramento decrescente / loss residua crescente), mentre il polinomio resta quasi piatto.

### NUOVO: small-data test (80 frasi train vs 800)

Per verificare il gap train/test con meno dati, ho lanciato una campagna parallela con **80 frasi** di training (un ordine di grandezza in meno). Job Leonardo `50763579` — i plot comparativi saranno in:

`results/final_loss/v2_small_T16_d8_ks1-2-3-5-6_L16_n10_n80_test/`

(aggiornerò il pack appena completato)

**Nota train≈test (800 frasi):** il dataset test è effettivamente diverso (seed 4242 vs 42). I delta ~0.001 per k-QSA/CSA sono attesi: la loss −log μ è geometrica sui parametri W,V, non memorizza le frasi. nl-CSA mostra gap reale (~0.07–0.09) essendo una rete neurale.

---

## 2. μ VS T / μ VS d

**μ vs T** (d=16, k=2,5; invariato):  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_v2/mu/mu_vs_T.png

**μ vs d** (T=32, **solo k=2,5**, log scale, stessi colori di μ vs T):  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_v2/mu/mu_vs_d.png

---

## 3. TRAINING CURVES (appendice)

**k=3** (con nl-CSA + advantage, curve smooth):  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_v2/loss/plots/final_training_curves_k3_appendix.png

**k=5** (nuovo):  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_v2/loss/plots/final_training_curves_k5_appendix.png

Le curve mono si interrompono a ep 400 (budget fisso); poly a 600, nl-gen a 800.

---

## Codice

- `run_final_loss.py` — loss + log(μ₀/μ) plot  
- `run_study.py` — μ panels  
- `hpc_final_loss_v2.sh` — campagna full (800 frasi)  
- `hpc_final_loss_small.sh` — campagna small-data (80 frasi)  

Branch: https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/tree/PennyLaneG

Cordiali saluti,  
Alessio
