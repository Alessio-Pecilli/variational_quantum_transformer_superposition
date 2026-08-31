# Email — qsa_bench_25_08 (readout ibrido, embedding trainable, excess + μ)

**Subject:** Re: ξ/α nel circuito, μ vs T/d, L₁^excess / L_B^excess — campagna completa (param-matched)

---

Buongiorno Professore,

in risposta alle sue indicazioni abbiamo implementato la pipeline **`qsa_bench_25_08`** e completato su Leonardo le due campagne (excess vs k e μ vs T/d). Di seguito il riepilogo punto per punto.

**Pack GitHub (plot + summary + aggregati):**

https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/tree/PennyLaneG/results_from_hpc/qsa_bench_2508

Codice: branch `PennyLaneG`, file [`qsa_bench_25_08.py`](https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/qsa_bench_25_08.py), driver [`run_qsa_bench_2508_campaign.py`](https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/run_qsa_bench_2508_campaign.py), μ [`run_mu_sweep_2508.py`](https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/run_mu_sweep_2508.py).

---

## 1) Token normalizzati vs non normalizzati (ξ e α)

Come da sua nota, il circuito fa **amplitude encoding** sui token normalizzati

\[
\xi_\ell = x_\ell / \sqrt{\alpha_\ell}, \qquad \alpha_\ell = \|x_\ell\|^2,
\]

mentre in fase di misura (μ e readout ibrido) servono le **norme dei token non normalizzati** \(x_\ell\). Nel codice:

- attention e costruzione di \(z_j\) usano \(\xi\) (ciò che il registro può caricare);
- la predizione usa il fattore \(\sqrt{\alpha_{j+1}}\) sul target (controllo \(R_t\)): \(A_j \leftarrow \sqrt{\alpha_{j+1}}\,\langle\xi_{j+1}|G z_j\rangle\).

Le funzioni prendono quindi un **vocabolario/embedding non normalizzato** `Xvoc`; a ogni step si ricostruiscono \(\xi\) e \(\alpha_t\) dalle norme correnti. Non c’è doppia convenzione tra modelli: un solo `forward()` / `metrics()` per tutti.

---

## 2) EMBEDDING — addestrato insieme al resto

**Sì: nell’implementazione nuova l’embedding è addestrato congiuntamente a \(W,V\) sui dati classici PTB.**

- Parametri: `concat(Xvoc, circuit_v)` ottimizzati insieme con Adam su **\(L_B\)** (k-models) / **\(L_1\)** (nl-CSA).
- Le norme \(\alpha_\ell\) **entrano esplicitamente nella loss** tramite il readout ibrido (non sono un post-processing).
- Sul **quantum TFIM** non c’è embedding: stati già unitari (\(\alpha \equiv 1\)), come prima.

Nelle campagne **precedenti** (pipeline `qsa_bench` / section2 su token già normalizzati sulla sfera) l’embedding PTB poteva essere trainable in `qsa_training`, ma **le norme non entravano** nel readout/loss — diverso regime rispetto a questa versione.

---

## 3) Loss: training su \(L_B\); plot su \(L_1^{\mathrm{excess}}\), \(L_B^{\mathrm{excess}}\)

- **Training:** sempre **\(L_B\)** per kQSA/kCSA (nl su \(L_1\)).
- **Plot finali:** **\(L_1^{\mathrm{excess}}\)** e **\(L_B^{\mathrm{excess}}\)** (capacity-normalized: sottraggono il floor architetturale \(p_j \le \alpha_{j+1}\)).

Stesso **parameter setting** dell’ultima campagna param-matched che le avevo mandato:

| | |
|---|---|
| T, d | 32, 16 |
| k | 1 … 6 |
| seed | 8 |
| CSA / nl | **1024** |
| QSA | **1032** (L=43; 1024 esatto non raggiungibile con l’ansatz) |
| Classical | PTB, frasi lunghezza 33 → T=32 step |
| Quantum | TFIM, stati unitari |

Job excess: **`54131638`** (COMPLETED).

### CLASSICAL — excess vs k

**\(L_1^{\mathrm{excess}}\) train / test**  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/qsa_bench_2508/LB_excess_T32_d16_ks1-2-3-4-5-6_LmatchCSA_n8/classical/plots/train_L1_excess_vs_k.png  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/qsa_bench_2508/LB_excess_T32_d16_ks1-2-3-4-5-6_LmatchCSA_n8/classical/plots/test_L1_excess_vs_k.png  

**\(L_B^{\mathrm{excess}}\) train / test** (solo k-models)  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/qsa_bench_2508/LB_excess_T32_d16_ks1-2-3-4-5-6_LmatchCSA_n8/classical/plots/train_LB_excess_vs_k.png  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/qsa_bench_2508/LB_excess_T32_d16_ks1-2-3-4-5-6_LmatchCSA_n8/classical/plots/test_LB_excess_vs_k.png  

**Lettura (train \(L_1^{\mathrm{excess}}\), 8 seed):** i kernel **poly** restano stabili e migliori dei mono; i **mono** (QSA e CSA) **peggiorano con k** su PTB con embedding trainable — analogo al comportamento sospetto già visto sulla campagna Markov, qui nel regime ibrido+excess.

| modello (n) | L1x k=1 | L1x k=3 | L1x k=6 |
|---|---:|---:|---:|
| k-QSA (1032) | 0.58±0.19 | 1.23±0.12 | **1.73±0.10** |
| poly-k-QSA (1032) | 0.41±0.20 | 0.52±0.18 | 0.53±0.18 |
| k-CSA (1024) | 0.35±0.14 | 0.95±0.18 | 1.25±0.18 |
| poly-k-CSA (1024) | **0.23±0.09** | **0.32±0.14** | **0.36±0.16** |
| nl-iso (1024) | 0.44±0.14 | | |
| nl-gen (1024) | 0.44±0.12 | | |

### QUANTUM — excess vs k

**\(L_1^{\mathrm{excess}}\) train / test**  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/qsa_bench_2508/LB_excess_T32_d16_ks1-2-3-4-5-6_LmatchCSA_n8/quantum/plots/train_L1_excess_vs_k.png  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/qsa_bench_2508/LB_excess_T32_d16_ks1-2-3-4-5-6_LmatchCSA_n8/quantum/plots/test_L1_excess_vs_k.png  

**\(L_B^{\mathrm{excess}}\) train / test**  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/qsa_bench_2508/LB_excess_T32_d16_ks1-2-3-4-5-6_LmatchCSA_n8/quantum/plots/train_LB_excess_vs_k.png  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/qsa_bench_2508/LB_excess_T32_d16_ks1-2-3-4-5-6_LmatchCSA_n8/quantum/plots/test_LB_excess_vs_k.png  

**Lettura (train \(L_1^{\mathrm{excess}}\)):** a parità di parametri, QSA e CSA sono vicini; entrambi scendono con k. A k=6: k-QSA **0.35±0.23**, k-CSA **0.29±0.03**, poly-k-CSA **0.29±0.01**, nl-iso **0.29±0.02**.

| modello (n) | L1x k=1 | L1x k=3 | L1x k=6 |
|---|---:|---:|---:|
| k-QSA (1032) | 0.93±0.25 | 0.60±0.33 | **0.35±0.23** |
| poly-k-QSA (1032) | 1.00±0.18 | 0.45±0.21 | 0.41±0.29 |
| k-CSA (1024) | 0.89±0.02 | 0.36±0.02 | **0.29±0.03** |
| poly-k-CSA (1024) | 0.98±0.02 | 0.40±0.01 | **0.29±0.01** |
| nl-iso (1024) | **0.29±0.02** | | |

---

## 4) μ — formula di circuito e plot rifatti (stessa config dei vecchi)

**μ non è più \(\exp(-L_B)\)** ma l’osservabile di circuito del readout ibrido (con il fattore \(\sqrt{\alpha}\) sul target), da `observables()` in `qsa_bench_25_08.py`.

**Config allineata ai plot che mi aveva allegato** (final_campaign_v2):

| asse | fisso | sweep | k in figura | modelli |
|------|-------|-------|-------------|---------|
| μ vs T | d=16 | T ∈ {2,4,8,16,32} | 2, 5 | mono + poly (kQSA + kCSA) |
| μ vs d | T=32 | d ∈ {2,4,8,16,32} | 2, 5 | mono + poly |
| advantage | \(k / \binom{d+k-1}{k}\) | | | linea tratteggiata |
| dati | PTB classico | | 10 seed | embedding trainable, train \(L_B\) |

Job μ: **`54577147`** (COMPLETED, 72/72 celle, walltime 4 giorni).

**μ vs T (d=16)**  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/qsa_bench_2508/mu_T32_d16_ks2-5_n10/mu_vs_T.png  

**μ vs d (T=32)**  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/qsa_bench_2508/mu_T32_d16_ks2-5_n10/mu_vs_d.png  

**Lettura sintetica:** con μ di circuito e norme in training, i **mono** restano in genere **sotto** la soglia di advantage (specie per T,d grandi). I **poly** si avvicinano o la superano in alcuni punti a **T piccolo** (es. k=5, T=2: margin fino a ~6× per kCSA-poly). Su μ vs d a T=32 i valori mono restano molto sotto advantage; poly-k=5 a d=32 è l’unico regime dove il margin supera ~0.25×.

---

## 5) Parametri QSA ≈ CSA

Come richiesto, **stesso budget parametrico** per il confronto QSA/CSA: **1032 vs 1024** (L=43, n=4 qubit). L’embedding trainable è **condiviso** tra i modelli a fissato (T,d,D) e non entra nel conteggio “circuito” in legenda — il match riguarda \(W,V\) come nelle campagne precedenti.

---

## 6) Validazione locale (embedding + α)

Abbiamo verificato in locale che \(\alpha_t = \|y_{\mathrm{unnorm}}\|^2\), che \(\|\xi\|=1\), che con/senza α la loss grezza rispetta i floor e che \(L_1 - L_1^{\mathrm{floor}} = L_1^{\mathrm{excess}}\). Su TFIM con \(\alpha=1\): \(L_B \equiv L_B^{\mathrm{excess}}\).

Note: [`results_from_hpc/qsa_bench_2508/local_embedding/README.md`](https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/qsa_bench_2508/local_embedding/README.md)

---

Restiamo a disposizione per approfondire (es. mono-classical che peggiora con k, o confronto diretto μ vecchio vs nuovo su una cella).

Cordiali saluti,  
Alessio
