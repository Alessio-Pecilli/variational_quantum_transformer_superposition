# Email draft — risposta completa (formule, PTB vs complex, v5/v6, run locali, Leonardo)

**To:** [professore]  
**Subject:** Re: L1 / CE nlCSA vs kQSA–kCSA — formule, check locali, Leonardo out

---

Buongiorno Professore,

raccolgo qui tutte le risposte (formule, n_params, v5 vs v6, classical+complex, run locali).  
**Leonardo** è in manutenzione CINECA (login/compute down circa 3–7 ago; servizio non garantito fino al ~14). Sono fuori dall’8 al 23: se serve HPC faremo **resub a fine agosto**. I check sotto **non** richiedono Leonardo.

---

## 1) Ricalcolare L1 di k-QSA/k-CSA con la stessa CE di nl sul plot PTB?

https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_LB/plots/final_L1_vs_k_train.png

**No — non è la stessa quantità e non si può “riapplicare” la CE di nl ai mu-models.**

### Formule plottate (Classical PTB, `final_campaign_LB`)

**k-QSA / k-CSA** (train \(L_B\); L1 = CE **continua**):
\[
z_j=\sum_{i\le j} K_{ji}\,V x_i,\qquad
p_j=f_j^2=\frac{|\langle y_j|z_j\rangle|^2}{\|z_j\|^2},\qquad
L_1=-\frac1T\sum_{j=1}^{T}\log p_j.
\]
(\(K\) monomio \(s^k\) o poly; \(y_j\) = embedding del passo successivo.)

**nl-CSA** (pipeline **lessicale** standard):
\[
\ell_t=\mathrm{logits}_t\in\mathbb{R}^{V},\qquad
p_{t+1}^{\mathrm{tok}}=\mathrm{softmax}(\ell_t)_{\,y_{t+1}},\qquad
L_1^{\mathrm{nl}}=-\frac1T\sum_t\log p_{t+1}^{\mathrm{tok}}.
\]
Sì: è sempre \(-\sum\log p\), ma **\(p\) non è \(f_j^2\)**: è la probabilità del **token** sul vocabolario (\(V\sim3045\), \(\log V\sim8\)).  
Per questo L1 nl ~7 vs L1 mu ~0.5–0.9: **inconciliabili** senza cambiare architettura (k-QSA/k-CSA **non hanno** head lessicale).

In locale nel pack LB ci sono solo summary+plot (niente params). I checkpoint sono su Leonardo; anche scaricandoli **non** si ottiene la CE discreta di nl sui mu-models.

**Classical + complex ansatz non cambia questo discorso PTB**: è un setting diverso (sotto).

---

## 2) Numero di parametri

**μ vs T**  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_v2/mu/mu_vs_T.png  

Studio circuit section2 (ōbar), non i loss models kQSA/nl. `d=16`; `n_qubits` cresce con T (es. T=2→10, T=4→12, …). Nel manifest non c’è un `n_params` aggregato.

**L1 quantum / classical+complex** (T=32, d=16, L=16):  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v5_Lhalf_L1/plots/train_L1_vs_k.png  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_classical_complex/plots/train_L1_vs_k.png  

| modello | n_params |
|---|---:|
| k-QSA / poly-k-QSA | **416** (`2·L·n·3 + T`) |
| k-CSA / poly-k-CSA | **544** (`2d² + T`) |
| nl-CSA iso | **512** (`2d²`) |
| nl-CSA gen | **1024** (`4d²`) |

(PTB/v2 loss T=16,d=8,L=16: k-QSA **96**, k-CSA **128**, nl~128 **144**.)

---

## 3) Formule nel mondo quantum / classical Markov + complex ansatz

Qui **tutti** (k-QSA, k-CSA, nl soft iso/gen) usano lo **stesso** readout continuo — **non** c’è CE lessicale nascosta per nl:

\[
p_j=f_j^2=\frac{|\langle y_j|z_j\rangle|^2}{\|z_j\|^2},
\qquad
L_1=\mathrm{CE_{unif}}=-\frac1T\sum_j\log p_j,
\qquad
L_{1/2}=-2\log\Big(\frac1T\sum_j\sqrt{p_j}\Big),
\qquad
L_B=-\log F,\quad F=\tfrac{T+1}{2T}\tfrac{\mu}{\zeta}.
\]

Differisce solo il kernel \(K\): monomio/poly (mu) vs softmax di fedeltà (nl).

**Cosa viene tipicamente trainato vs plottato (L1 v5/v6 / classical-complex):**

| modello | train | quantità nel plot L1 |
|---|---|---|
| k-QSA / k-CSA (mono/poly) | \(L_B\) | eval \(L_1=\mathrm{CE_{unif}}\) sopra |
| nl-CSA iso/gen **CE** (refs L1) | \(L_1=\mathrm{CE_{unif}}\) | stessa \(L_1\) |
| nl-CSA iso/gen Renyi (refs L_half) | \(L_{1/2}\) | (altre figure) |

---

## 4) “Strano che mu batta nl / salto vs classical+complex” — metrica diversa di nl?

**No: in questo mondo la metrica L1 di nl è la stessa CE_unif.**  
Il salto rispetto a classical+complex si spiega così:

| pack | L1 mu @k=3 (es.) | L1 nl-CE | Cosa è successo |
|---|---:|---:|---|
| **classical+complex** | ~3.30–3.32 | ~3.31 | come **v5**: L1 riportata ≈ **epoca 0 / init** (bug best-checkpoint) → tutti “allineati” ~3.3 |
| **v5** quantum | ~3.30 | ~3.31 | stesso bug → piatta ~3.3 (sembra “ragionevole” ma non è il trained) |
| **v6** bestckpt | poly-QSA ~**2.61**, QSA ~2.99; CSA ~3.32 | ~3.16–3.31 | checkpoint **corretto**: dopo train \(L_B\) la L1 dei QSA **scende**; nl-CE resta ~3.3 |

Quindi: classical+complex **non** contraddice v6 per una CE diversa di nl; semplicemente quella campagna (come v5) non riportava la L1 ai best params trainati.

Plot:  
v5 https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v5_Lhalf_L1/plots/train_L1_vs_k.png  
v6 https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v6_bestckpt/plots/train_L1_vs_k.png  
classical+complex https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_classical_complex/plots/train_L1_vs_k.png  

---

## 5) Run locali di controllo (fatti)

### 5a) Solo mono k-QSA + mono k-CSA (come chiesto: 1 k, pochi seed, no poly/nl)

T=16, L=8, k=3, 2 seed. L1 init → final:

| | L1 init → final |
|---|---|
| mono k-QSA | ~3.38 → ~3.22 |
| mono k-CSA | ~3.32 → ~2.84 |

Con checkpoint corretto la L1 **scende** (coerente con v6, non con v5).  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/tree/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/local_mono_L1_control  

### 5b) nlCSA trainata con la **stessa** \(L_B\) di kQSA/kCSA, stessa CE_unif in eval

Classical Markov + complex, k=3, 2 seed; mono k-QSA/k-CSA + nl iso/gen; **tutti train \(L_B\)**.

| modello | n | \(L_B\) | \(L_1\) |
|---|---:|---:|---:|
| k-QSA | 208 | 4.70±0.13 | 3.27±0.07 |
| k-CSA | 528 | **2.31** | **2.64** |
| nl-CSA iso | 512 | **2.29** | **2.65** |
| nl-CSA gen | 1024 | 4.06±0.09 | 3.15 |

Plot:  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/local_nl_same_LB_control/local_LB_train_all_models_k3.png  

**Lettura:** a metrica e loss di training allineate, **nl-iso ≈ k-CSA**. Niente vantaggio “misterioso” dei mu per CE diversa di nl.

---

## 6) Cosa resta / Leonardo

- Campagna full classical+complex (k≤5, 10 seed) ancora in coda/bloccata dalla maintenance → **resub fine agosto** se serve.  
- Checkpoint PTB: scarico quando Leonardo torna; non sblocca comunque la CE lessicale sui mu-models.  
- Per chiudere il dubbio metriche / v5 vs v6 / nl vs mu i check locali sopra bastano **senza** aspettare Leonardo.

Cordiali saluti,  
Alessio
