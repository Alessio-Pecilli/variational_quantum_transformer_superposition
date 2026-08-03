# Email draft — risposta su L1 CE, n_params, v5 vs v6 + Leonardo out

**To:** [professore]  
**Subject:** Re: L1 kQSA/kCSA vs nl, n_params, v5 vs v6 — Leonardo in manutenzione

---

Buongiorno Professore,

rispondo punto per punto. **Nota operativa:** Leonardo è in manutenzione CINECA (login/compute down circa 3–7 agosto; servizio non garantito fino al ~14). Al momento non posso scaricare i checkpoint dalla macchina; riprendo appena riaprono l’accesso.

---

### 1) Ricalcolare L1 di k-QSA/k-CSA con la stessa CE di nl-CSA sul plot PTB?

Plot:  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_LB/plots/final_L1_vs_k_train.png

**In sintesi: no, non è la stessa quantità e non si può “riapplicare” la CE di nl ai mu-models.**

Sul pack classical PTB (`final_campaign_LB`) le due famiglie usano readout diversi:

- **k-QSA / k-CSA:** Shannon CE **continua**  
  \(L_1=\mathrm{CE_{unif}}=-\frac1T\sum_j\log p_j\) con \(p_j=f_j^2=\lvert\langle y_j\mid z_j\rangle\rvert^2/\lVert z_j\rVert^2\)  
  → valori ~0.5–0.9
- **nl-CSA:** Shannon CE **discreta** next-token sul vocabolario (softmax sui logits)  
  → valori ~7.3 (ordine di \(\log V\), V≈3045)

Il debug aveva ragione a segnalare la differenza: **non è un bug di implementazione della CE continua**, è che nl usa la pipeline lessicale. k-QSA/k-CSA **non hanno un head sul vocabolario**, quindi non esiste un modo fedele di valutare su di essi “la stessa CE di nl” senza cambiare modello.

In locale nel pack LB ci sono solo `summary.json` + plot (niente params). I checkpoint dovrebbero essere ancora su Leonardo sotto `results/final_loss/LB_...`; appena torna l’accesso li scarico, ma anche con i parametri finali **non si ottiene la CE discreta di nl** — al massimo si rivaluta la CE_unif continua già usata.

Il confronto *fair* (stessa CE_unif su \(f_j^2\) per tutti, incluso nl soft) è quello delle campagne **quantum-sequences / classical+complex ansatz**, non del PTB classical.

---

### 2) Numero di parametri nei plot richiesti

**μ vs T**  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_v2/mu/mu_vs_T.png  

Studio **circuit section2** (ōbar), non i modelli loss kQSA/nl. Config tipica `d=16`; `n_qubits` **cresce con T** (es. T=2 → 10 qubit, T=4 → 12, …). Nel manifest non c’è un `n_params` aggregato per curva.

**L1 quantum v5** e **classical+complex** (stesso setup T=32, d=16, L=16):  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v5_Lhalf_L1/plots/train_L1_vs_k.png  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_classical_complex/plots/train_L1_vs_k.png  

| modello | n_params |
|---|---:|
| k-QSA / poly-k-QSA | **416** (`2·L·n·3 + T`, RX/RY/RZ + φ) |
| k-CSA / poly-k-CSA | **544** (`2d² + T`) |
| nl-CSA iso | **512** (`2d²`) |
| nl-CSA gen | **1024** (`4d²`) |

Per riferimento, sul classical PTB / v2 loss (T=16, d=8, L=16): k-QSA **96** angoli, k-CSA **128** matrici, nl~128 **144** parametri.

---

### 3) v5 L1 “più ragionevole” di v6? + run locali di controllo

v5:  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v5_Lhalf_L1/plots/train_L1_vs_k.png  

v6 (best-ckpt):  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/run_full_v6_bestckpt/plots/train_L1_vs_k.png  

**Interpretazione:** in v5 la L1 riportata coincideva con valori da **parametri iniziali / epoca ~0** (bug sul best-checkpoint: `tol·∞` bloccava il salvataggio del best). Per questo è piatta ~3.3 per tutti e “sembra” più uniforme.  
In **v6** il checkpoint è corretto: L1 è valutata ai **best params dopo training su L_B**, e k-QSA/poly scendono sotto ~3 — è il comportamento atteso se L_B migliora anche la CE_unif.

**Controllo locale** (come chiesto: un solo k, pochi seed, solo mono k-QSA e mono k-CSA, senza poly e senza nl):

Config ridotta: T=16, d=16, L=8, k=3, 2 seed.

| modello | L1 init → final | Δ |
|---|---|---|
| mono k-QSA | ~3.38 → ~3.22 | ≈ −0.16 |
| mono k-CSA | ~3.32 → ~2.84 | ≈ −0.48 |

Con checkpoint corretto la L1 **scende** col training (coerente con v6), non resta piatta a ~3.3 (v5 ≈ init).

Dettaglio locale:  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/tree/PennyLaneG/results_from_hpc/final_campaign_quantum_sequences/local_mono_L1_control  
(se il pack non è ancora pushato: cartella locale omonima nel repo)

---

Appena Leonardo torna online: scarico i checkpoint LB e confermo l’inventario params; la conclusione sul punto 1 (CE discreta non applicabile a kQSA/kCSA) resta.

Cordiali saluti,  
Alessio
