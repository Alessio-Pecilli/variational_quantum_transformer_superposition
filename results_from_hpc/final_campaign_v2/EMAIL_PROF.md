# Email draft — FINAL campaign v2 (copy/paste)

**To:** [professore]  
**Subject:** FINAL v2 Leonardo — plot rivisti + risposte ai dubbi

---

Buongiorno Professore,

ho applicato tutte le modifiche cosmetiche ai plot e rispondo ai dubbi emersi dalla prima versione. Tutto pushato su GitHub, link sotto.

## Repository

- Repo: https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition  
- Branch: https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/tree/PennyLaneG  

## Pack risultati v2

https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/tree/PennyLaneG/results_from_hpc/final_campaign_v2

---

## 1. ALIGNED LOSS VS k

- **Train (10 seed, param in legenda, d=8):**  
  https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_v2/loss/plots/final_aligned_loss_vs_k.png

- **Test hold-out (PTB seed 4242, 200 frasi):**  
  https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_v2/loss/plots/final_aligned_loss_vs_k_test.png

- Tutti i plot: https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/tree/PennyLaneG/results_from_hpc/final_campaign_v2/loss/plots

**Modifiche applicate:**
- nl-CSA iso: ora solo **envelope** (banda colorata), rimossa la barra d'errore puntuale che causava l'aspetto asimmetrico
- nl-CSA gen: envelope per barra d'errore (invariato, già era ok)
- k-CSA: barre d'errore più grandi di QSA — confermato, dovuto all'ansatz classico più grande (128 matrici vs 96 angoli); ci teniamo così come concordato

**Risultati (aligned loss = −log μ + log T, T=16, d=8, mean±std, 10 seed):**

| Modello | k=1 | k=6 | Δ k=1→6 |
|---------|-----|-----|---------|
| k-QSA L=16 (96 ang) | 4.28±0.10 | 6.62±0.22 | +2.34 |
| k-CSA (128 mat) | 4.71±0.35 | 7.54±1.10 | +2.83 |
| poly-k-QSA (96 ang) | 4.20±0.58 | 4.88±0.06 | **+0.68** |
| poly-k-CSA (128 mat) | 4.71±0.85 | 5.36±0.37 | **+0.65** |
| nl-CSA iso ~128 (~144 par) | 6.58±0.03 | — | k-indip. |
| nl-CSA gen ~128 (~144 par) | 6.93±0.78 | — | k-indip. |

### Dubbio: nl-CSA iso asimmetria dell'envelope

I 10 seed iso danno una distribuzione **simmetrica**: min=6.536, max=6.632, mean=6.579, std=0.032, range=0.096. L'apparente asimmetria nella versione precedente era un artefatto della barra d'errore puntuale posizionata al bordo del plot. Con l'envelope (banda colorata) non c'è più ambiguità.

### Dubbio: train e test danno gli stessi numeri?

Il dataset test è effettivamente **diverso** (data_seed=42 per train, test_data_seed=4242 per test, frasi PTB separate). I delta sono:

- **k-QSA / k-CSA / poly**: Δ ≈ 0.001 — questo è **atteso**: la loss −log μ è una misura geometrica sui parametri W, V applicata alle frasi; il modello non memorizza le frasi specifiche, quindi frasi PTB dello stesso dominio danno valori quasi identici. Non c'è overfitting sulle frasi.
- **nl-CSA**: Δ ≈ +0.07 (iso) / +0.09 (gen) — **coerente** col fatto che nl-CSA è una rete neurale che fitta sulle frasi train, e mostra un gap di generalizzazione reale.

In sintesi: i numeri sono corretti, il quasi-zero gap per i μ-models riflette la natura geometrica della loss (non dipende dalle frasi specifiche), mentre nl-CSA mostra il gap atteso.

---

## 2. μ VS T / μ VS d

- **μ vs T** (d=16, **solo k=2,5** per leggibilità; mono/poly/advantage; 10 seed):  
  https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_v2/mu/mu_vs_T.png

- **μ vs d** (T=32, k=2,3,5; mono/poly/advantage; 10 seed; **asse μ in scala log**):  
  https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_v2/mu/mu_vs_d.png

- Manifest: https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_v2/mu/manifest.json

**Modifiche applicate:**
- μ vs T: ridotto a **k=2 e k=5** (prima k=2,3,5) per maggiore leggibilità
- μ vs d: asse y in **scala logaritmica**
- Entrambi: 10 seed (prima 5)

Setup: target_loss=3.8, convergenza fino a 2000 ep, T≤32, k=1,2,3,5,6, mono+poly+advantage sullo stesso plot. Colori fissi per k, linestyle: solid=mono, dashed=poly, dotted=advantage.

---

## 3. TRAINING CURVES (appendice, k=3)

- **Curve k=3 + nl-CSA + advantage (smoothed):**  
  https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_v2/loss/plots/final_training_curves_k3_appendix.png

**Modifiche applicate:**
- Tutte le curve medie ora con **smooth** (moving average 15 epoche) — CSA molto più leggibile
- nl-CSA iso e gen incluse nel plot
- Linea costante advantage: −log(k/C(d+k−1,k)) + log T = 6.46 per k=3, d=8, T=16

### Nota: perché le curve mono si interrompono prima?

Budget epoch diverso per famiglia: mono=400, poly=600, nl-iso=500, nl-gen=800. Le curve mono finiscono a ep 400 per design e vengono estese con l'ultimo valore (coda piatta).

### Nota: nl-CSA convergono vicino al limite advantage

Osservazione corretta — nl-CSA iso converge a 6.58, appena sopra l'advantage threshold (6.46). Questo suggerisce che con ~144 parametri il modello non-lineare classico non riesce a superare significativamente la soglia advantage per k=3, d=8.

---

## Setup tecnico

| Campagna | Parametri chiave |
|----------|------------------|
| Loss v2 | T=16, d=8, k=1,2,3,5,6, 10 seed, QSA L=16 (96 ang), CSA (128 mat) |
| nl-CSA | solo iso~128 + gen~128 (~144 par ciascuno), 10 seed |
| μ panels | T≤32, d≤32, 10 seed (42–51), target 3.8, mono+poly |
| Test | PTB hold-out, seed 4242, 200 frasi |

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
