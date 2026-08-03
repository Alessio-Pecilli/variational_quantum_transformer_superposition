# Control: L1 CE, params, v5 vs v6

## 1) `final_L1_vs_k_train.png` (classical PTB LB) — ricalcolare L1 kQSA/kCSA con CE di nlCSA?

**No, non è fattibile** (anche scaricando i checkpoint da Leonardo).

| | k-QSA / k-CSA | nl-CSA |
|---|---|---|
| Readout | continuo \(p_j=f_j^2=\lvert\langle y_j\mid z_j\rangle\rvert^2/\lVert z_j\rVert^2\) | logits su **vocabolario** + softmax |
| L1 nel pack LB | `CE_uniform_jax` (~0.5–0.9) | discrete next-token CE (~7.3 ≈ log V) |

- In locale il pack ha solo `summary.json` + plots (**nessun** `params_final.npz`).
- Anche con i params HPC, **non esiste** un head lessicale per k-QSA/k-CSA: non si può applicare la CE di nl senza inventare un nuovo modello.
- La CE “giusta” confrontabile con nl **su sequenze continue / complex ansatz** è la CE_unif su \(f_j^2\) (come quantum / classical-complex), non la CE discreta PTB.

## 2) Numero di parametri nei plot richiesti

### Classical PTB / final_campaign_LB & v2 **loss** (T=16, d=8, L=16)
| modello | n_params |
|---|---:|
| k-QSA / poly-k-QSA | **96** angoli (`2·L·n`, n=⌈log₂ d⌉=3) |
| k-CSA / poly-k-CSA | **128** (`2d²`) |
| nl-CSA ~128 | **144** (`6·L_nl·d·r`, r=3) |

### `mu_vs_T.png` (final_campaign_v2/mu)
Studio **circuit section2** (ōbar), **non** i modelli loss kQSA/nl.  
Config tipica: `d=16`, `circuit-mode=section2`, T variabile; `n_qubits` **cresce con T** (es. T=2 → 10 qubit, T=4 → 12, …).  
Nel `manifest.json` **non** c’è un campo `n_params` aggregato; i run sono etichettati per (T,k,mono/poly,seed).

### Quantum sequences v5 / classical_complex (T=32, d=16, L=16) — formula attuale
| modello | n_params |
|---|---:|
| k-QSA / poly-k-QSA | **416** = `2·L·n·3 + T` (RX/RY/RZ + φ) |
| k-CSA / poly-k-CSA | **544** = `2d² + T` (Hermitian gens + φ) |
| nl-CSA iso | **512** = `2d²` |
| nl-CSA gen | **1024** = `4d²` |

## 3) Perché v5 L1 (~3.3 piatta) “sembra più ragionevole” di v6

**v5 aveva il bug best-checkpoint:** `train_L_1` ≈ valore a **epoca 0** (init), non ai parametri trainati.  
**v6** (fix) valuta L1 ai **best params** dopo training su L_B → k-QSA/poly scendono sotto ~3.

### Controllo locale (mono only, k=3, 2 seed, T=16, L=8) — PASS
Path: `results_from_hpc/final_campaign_quantum_sequences/local_mono_L1_control/`

| family | L_B init→final | L1 init→final | ΔL1 |
|---|---|---|---|
| mono k-QSA | ~6.2 → ~4.6 | ~3.38 → ~3.22 | **−0.16** |
| mono k-CSA | ~4.8 → ~3.7 | ~3.32 → ~2.84 | **−0.48** |

Conclusione: con checkpoint corretto, **L1 scende** con il training L_B (come v6), non resta piatta a ~3.3 (come v5 = init).
