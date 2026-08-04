# Email draft — formule loss + classical+complex + nl trainata con L_B (locale)

**To:** [professore]  
**Subject:** Re: formule CE nlCSA; classical+complex non cambia il PTB; nl con stessa L_B (locale)

---

Buongiorno Professore,

ha ragione: **classical + complex ansatz non cambia il discorso sul PTB**. Chiarisco le formule e cosa implica; poi un controllo locale con nl trainata sulla **stessa** \(L_B\) di kQSA/kCSA.

---

## A) Formule plottate / usate (due mondi)

### Mondo 1 — Classical PTB (`final_campaign_LB`)

**k-QSA / k-CSA** (train \(L_B\); L1 = CE continua):
\[
p_j=f_j^2=\frac{|\langle y_j|z_j\rangle|^2}{\|z_j\|^2},\qquad
L_1=-\frac1T\sum_{j=1}^{T}\log p_j.
\]

**nl-CSA** (pipeline standard lessicale):
\[
p_{t+1}^{\mathrm{tok}}=\mathrm{softmax}(\ell_t)_{y_{t+1}},\qquad
L_1^{\mathrm{nl}}=-\frac1T\sum_t\log p_{t+1}^{\mathrm{tok}}.
\]
Sì: è sempre \(-\sum\log p\), ma **\(p\) non è \(f_j^2\)**: è la probabilità del **token** sul vocabolario (\(V\sim3045\)).  
Per questo L1 nl~7 vs L1 mu~0.5–0.9: **inconciliabili** senza cambiare architettura (i mu-models non hanno head lessicale).  
Classical+complex **non** risolve questo: è un altro setting.

### Mondo 2 — Quantum / Classical Markov + complex ansatz

**Tutti** (k-QSA, k-CSA, nl soft) condividono:
\[
p_j=f_j^2=\frac{|\langle y_j|z_j\rangle|^2}{\|z_j\|^2},\qquad
L_1=-\frac1T\sum_j\log p_j,\qquad
L_B=-\log F,\quad F=\tfrac{T+1}{2T}\tfrac{\mu}{\zeta}.
\]
Cambia solo il kernel \(K\) (monomio/poly vs softmax di fedeltà). Qui la metrica L1 **è** allineata.

Nei plot L1 v5/v6 tipicamente: mu train \(L_B\) poi eval \(L_1\); nl (varianti CE) train \(L_1\) poi plot \(L_1\).  
**v5 L1 piatta ~3.3** = bug best-checkpoint (quasi init). **v6** = L1 ai best params.

---

## B) Controllo locale: nlCSA con la **stessa** loss \(L_B\) di kQSA/kCSA

Classical Markov + complex, k=3, 2 seed, mono k-QSA/k-CSA + nl iso/gen, **tutti train \(L_B\)**, eval \(L_B\) e \(L_1=\mathrm{CE_{unif}}\) (stessa \(p_j=f_j^2\)).

| modello | n_params | \(L_B\) finale | \(L_1\) finale |
|---|---:|---:|---:|
| k-QSA | 208 | 4.70±0.13 | 3.27±0.07 |
| k-CSA | 528 | **2.31±0.00** | **2.64±0.01** |
| nl-CSA iso | 512 | **2.29±0.00** | **2.65±0.01** |
| nl-CSA gen | 1024 | 4.06±0.09 | 3.15±0.00 |

Plot:  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/local_nl_same_LB_control/local_LB_train_all_models_k3.png  

**Lettura:** con la stessa \(L_B\) e la stessa CE_unif, **nl-iso ≈ k-CSA** (nessun “mu misteriosamente meglio di nl”). Il salto v5/v6 e i confronti misti train-\(L_B\) vs train-\(L_1\) sul quantum restano da leggere alla luce del checkpoint e del protocollo di training, non di una CE lessicale nascosta.

---

## C) Leonardo / calendario

Leonardo ancora in manutenzione. Sono fuori dall’8 al 23: se serve HPC, **resub fine agosto**.  
Questi check (formule + run locale allineato) **non** richiedono Leonardo. I checkpoint PTB li riprendiamo solo se ancora utili; per “stessa CE di nl” sui mu-models PTB resta **non fattibile** (no head lessicale).

Cordiali saluti,  
Alessio
