# Email draft — clean-slate check su kCSA vs kQSA e nlCSA

**Subject:** Re: kCSA alto / nlCSA ~3 — check su pipeline condivisa (locale)

---

Buongiorno Professore,

aveva ragione a segnalare i due punti: dopo il cambio di loss, **kCSA restava molto sopra kQSA** e **nlCSA sembrava bloccato a ~3**. Li abbiamo presi come possibili bug di pipeline (non come conclusione scientifica) e abbiamo rifatto un confronto “clean-slate”.

### Cosa abbiamo cambiato nel check

Un solo forward, un solo calcolo delle metriche, un solo chance level. I modelli differiscono **solo** in `params → (W,V)` e nel kernel. Ansatz complesso per tutti (fase ottima in forma chiusa). Codice: `qsa_bench.py`.

Pack locale (T=16, d=8, k=1,2,3; JAX):  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/tree/PennyLaneG/results_from_hpc/qsa_bench_local

### Risposte ai due dubbi

**1) kCSA dovrebbe stare vicino a kQSA, invece era molto sopra**  
Sì: sulla pipeline precedente era un’anomalia. Su pipeline condivisa **kCSA ≈ kQSA** (mono e poly, su L1 e L_B). Il gap “CSA molto peggiore” dopo il cambio di loss non si riproduce; era quasi certamente un problema di ottimizzazione/parametrizzazione/reporting nella campagna HPC, non una differenza strutturale dei due modelli.

Plot diretto kQSA vs kCSA:  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/qsa_bench_local/plots/kqsa_vs_kcsa_L1.png

L1 train vs k (tutti i modelli + chance):  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/qsa_bench_local/plots/train_L1_vs_k.png

L_B train vs k:  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/qsa_bench_local/plots/train_LB_vs_k.png

**2) nlCSA fisso a ~3 (come se non ottimizzasse)**  
Anche qui: sulla shared path **nlCSA impara** (gain = chance − L1 ≫ 0; train L1 scende intorno a ~0.06–0.07 sul setting quantum locale, non resta a 3). Il valore ~3 visto nelle campagne precedenti è coerente con “vicino a chance / non ottimizzato / reporting init”, non con il comportamento del modello quando forward e loss sono allineati.

Gain (se ≈0 non ha imparato):  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/qsa_bench_local/plots/train_gain_vs_k.png

Barre L1 train/test a k=2 (quantum):  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/qsa_bench_local/plots/quantum_k2_L1_bars.png

Stesso check su sequenze classiche (ansatz complesso):  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/qsa_bench_local/plots/classical_k2_L1_bars.png

L1 test vs k:  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/qsa_bench_local/plots/test_L1_vs_k.png

### Numeri tipici (quantum, k=2, train L1)

| modello | L1 | chance | gain |
|---|---:|---:|---:|
| kQSA-mono | 0.054 | 2.56 | +2.51 |
| kCSA-mono | 0.054 | 1.81 | +1.76 |
| kQSA-poly | 0.105 | 2.59 | +2.48 |
| kCSA-poly | 0.105 | 1.92 | +1.82 |
| nlCSA-iso | 0.063 | 1.89 | +1.83 |
| nlCSA-gen | 0.073 | 2.56 | +2.49 |

### Caveat

Queste sono run **locali, piccole** (T=16, d=8, poche sequenze): servono a isolare i bug e a leggere l’andamento relativo, non come numeri definitivi da paper. Prima di ripartire su Leonardo a scala piena (T=32, d=16, multi-seed) allineiamo la campagna HPC a questa pipeline condivisa.

Cordiali saluti,  
Alessio
