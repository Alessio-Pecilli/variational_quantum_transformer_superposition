# Email draft — FINAL campaign v2 (copy/paste)

**To:** [professore]  
**Subject:** FINAL v2 Leonardo — aligned loss (10 seed + test) + μ vs T/d — link GitHub

---

Buongiorno Professore,

ho completato su Leonardo la **revisione dei plot FINAL** come da sue indicazioni. Di seguito i link e una sintesi dei risultati.

## Repository

- Repo: https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition  
- Branch: https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/tree/PennyLaneG  

## Pack risultati v2 (plot + summary)

Cartella principale:

https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/tree/PennyLaneG/results_from_hpc/final_campaign_v2

README (setup, numeri, file):

https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_v2/README.md

Report loss / convergenza:

https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_v2/loss/LOSS_CHECK.txt

---

## 1. ALIGNED LOSS VS k (revisione)

- **Figura principale (train, 10 seed, barre d'errore, param count in legenda, d nel titolo):**  
  https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_v2/loss/plots/final_aligned_loss_vs_k.png

- **Stesso plot su test hold-out** (PTB separato, seed 4242, 200 frasi):  
  https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_v2/loss/plots/final_aligned_loss_vs_k_test.png

- Tutti i plot loss (barre/curve per k, raw + aligned):  
  https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/tree/PennyLaneG/results_from_hpc/final_campaign_v2/loss/plots

- Summary JSON:  
  https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_v2/loss/summary.json

**Modifiche rispetto alla run precedente:**
- nl-CSA: **solo** iso~128 e gen~128 (~144 parametri ciascuno), **con barre d'errore** (10 seed)
- CSA/QSA: **10 seed** (prima 8) per barre più strette sui modelli classici
- Legenda: **numero di parametri** per ogni modello; titolo con **d=8**

**Risultati (aligned loss = −log μ + log T, T=16, d=8, mean±std, 10 seed):**

| Modello | k=1 | k=6 | Δ k=1→6 |
|---------|-----|-----|---------|
| k-QSA L=16 (96 ang) | 4.28±0.10 | 6.62±0.22 | +2.34 |
| k-CSA (128 mat) | 4.71±0.35 | 7.54±1.10 | +2.83 |
| poly-k-QSA (96 ang) | 4.20±0.58 | 4.88±0.06 | **+0.68** |
| poly-k-CSA (128 mat) | 4.71±0.85 | 5.36±0.37 | **+0.65** |
| nl-CSA iso ~128 | 6.58±0.03 | — | k-indip. |
| nl-CSA gen ~128 | 6.93±0.78 | — | k-indip. |

Il monomio **cresce** con k, il polinomio resta **quasi piatto** — coerente con la teoria. nl-CSA gen ha varianza seed più alta di iso; entrambi sopra i μ-models. Il test hold-out riproduce lo stesso ordine dei modelli (differenze <0.01 sulle medie).

---

## 2. μ VS T / μ VS d (al posto di ōbar)

- **μ vs T** (d=16, k=2,3,5; mono/poly/advantage; **10 seed**):  
  https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_v2/mu/mu_vs_T.png

- **μ vs d** (T=32, k=2,3,5; mono/poly/advantage; **10 seed**):  
  https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_v2/mu/mu_vs_d.png

- Manifest (griglia completa):  
  https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_v2/mu/manifest.json

Setup: target_loss=3.8, convergenza fino a 2000 ep, T≤32, k=1,2,3,5,6, mono+poly sulla stessa figura. Linea **advantage** = k / C(d+k−1, k). Colori fissi per k, linestyle per mono/poly/advantage.

---

## 3. TRAINING CURVES (appendice, k=3)

- Curve di training a **k=3** con nl-CSA e linea costante advantage:  
  https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_v2/loss/plots/final_training_curves_k3_appendix.png

---

## Setup tecnico

| Campagna | Parametri chiave |
|----------|------------------|
| Loss v2 | T=16, d=8, k=1,2,3,5,6, 10 seed, QSA L=16, job Leonardo `50570347` (~10h) |
| μ panels | T≤32, **10 seed** (42–51), target 3.8, jobs `50683086`+`50683088` (~11 min) |
| Test | seed 4242, 200 frasi PTB hold-out |

## Codice

- Loss runner: https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/run_final_loss.py  
- μ / grid study: https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/run_study.py  
- HPC loss v2: https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/hpc_final_loss_v2.sh  
- HPC μ: https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/hpc_final_obar.sh  

## Run precedente (v1, riferimento)

https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/tree/PennyLaneG/results_from_hpc/final_campaign_definitive

Resto a disposizione per commenti o ulteriori figure.

Cordiali saluti,  
Alessio
