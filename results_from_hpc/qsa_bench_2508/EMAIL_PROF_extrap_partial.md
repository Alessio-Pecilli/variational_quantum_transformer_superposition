# Email — μ extrapolation (k=7 QSA-poly): punti per fit

**Subject:** Re: μ — punti per fit/estrapolazione (k=7 QSA-poly; d=8 sweep in T + T=64)

---

Buongiorno Professore,

in vista dei fit per estrapolare a d,T grandi (senza aumentare troppo i qubit), raccogliamo qui i **nuovi punti** su **solo kQSA-poly, k=7**, nello stesso schema JSON del pack μ v2 già inviato.

## Cosa c’è di nuovo

1. **Sweep T a d=8** (T∈{2,4,8,16}), 10 seed  
2. **Punto T=64, d=8** (10 seed) — primo ancoraggio a T grande  

**JSON unico pronto per i fit** (14 punti: questi + tutti i k=7 kqsa-poly del pack μ v2):  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/qsa_bench_2508/mu_extrap_k7_qsa_poly_READY_for_fits.json

Cartelle dettaglio:
- d=8, T=2..16: https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/tree/PennyLaneG/results_from_hpc/qsa_bench_2508/mu_extrap_k7_qsa_poly_d8_T2-16_n10  
- T=64 d=8: https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/tree/PennyLaneG/results_from_hpc/qsa_bench_2508/mu_extrap_k7_qsa_poly_T64_d8-16_n10  

Config comune: embedding trainable, train \(L_B\), μ a **fine training**, advantage \(k^2\log d/m_k\), param-match QSA≈CSA (a d=8: L=14 → 252 ≈ 256).

## Tabella — k=7 kqsa-poly, asse T a d=8 (+ T=64)

| T | d | μ (mean) | margin vs adv |
|---|--:|---------:|--------------:|
| 2 | 8 | 4.03e-3 | 0.136× |
| 4 | 8 | 1.32e-3 | 0.044× |
| 8 | 8 | 7.19e-4 | 0.024× |
| 16 | 8 | 4.62e-4 | 0.016× |
| 32 | 8 | 3.22e-4 | 0.011× *(già nel pack v2)* |
| 64 | 8 | 4.16e-4 | 0.014× |

Fino a T=32 la **decrescita in T** è chiara; a T=64 μ risale leggermente (ancora ≪ advantage). Utile come punto per il fit, da prendere con cautela.

Per confronto, a **d=16** (pack v2) i margin a T piccoli sono più alti (T=2 → 0.44×) e scendono con T — coerente col messaggio che il gap si riduce a k grande / T piccoli, ma senza battere la soglia in modo stabile.

## Già inviato (invariato)

- Plot μ v2 (advantage nuova, QSA/CSA separate, k=2,5,7):  
  https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/tree/PennyLaneG/results_from_hpc/qsa_bench_2508/mu_T32_d16_ks2-5-7_n10_v2  
- Excess L₁/L_B (param-matched):  
  https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/tree/PennyLaneG/results_from_hpc/qsa_bench_2508/LB_excess_T32_d16_ks1-2-3-4-5-6_LmatchCSA_n8  

## In corso / prossimo

- **T=64, d=16** (stessa config) — in esecuzione  
- Eventuale sweep **k=11** solo QSA-poly (μ vs T / μ vs d) — dopo  

Restiamo a disposizione per allineare la forma del fit (es. scaling in T a d fissato, o in d a T fissato).

Cordiali saluti,  
Alessio
