# Email — μ extrapolation P1 (k=7 QSA-poly, d=8, T=2..16)

**Subject:** Re: μ fit / estrapolazione — primi punti d=8, k=7, solo QSA-poly

---

Buongiorno Professore,

come da piano (fit a d,T grandi senza aumentare troppo i qubit), ecco il **primo blocco** di punti aggiuntivi, separato dal resto.

**Config:** k=7, solo **kqsa-poly**, d=8 (L=14, 252 param ≈ CSA 256), T∈{2,4,8,16}, 10 seed, μ a fine training, advantage \(k^2\log d/m_k\).

**Pack (4 celle + summary + JSON per fit):**  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/tree/PennyLaneG/results_from_hpc/qsa_bench_2508/mu_extrap_k7_qsa_poly_d8_T2-16_n10

File utili:
- `summary.json` — solo i 4 punti nuovi  
- [`k7_qsa_poly_for_fits.json`](https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/qsa_bench_2508/mu_extrap_k7_qsa_poly_d8_T2-16_n10/k7_qsa_poly_for_fits.json) — merge con tutti i **k=7 kqsa-poly** già presenti nel pack μ v2 (stesso schema row)

| T | d | μ (mean±std) | margin vs adv |
|---|---|-------------:|--------------:|
| 2 | 8 | 4.03e-3 ± 1.7e-3 | 0.136× |
| 4 | 8 | 1.32e-3 ± 4.5e-4 | 0.044× |
| 8 | 8 | 7.19e-4 ± 2.2e-4 | 0.024× |
| 16 | 8 | 4.62e-4 ± 1.8e-4 | 0.016× |

Anche a d=8 si vede la **decrescita in T**; a T=2 il margin è il più alto del set (~0.14×), ancora sotto soglia.

Prossimi (se fattibili): T=64 a d=8 e 16; poi eventuale sweep μ vs T/d a **k=11** solo QSA-poly.

Cordiali saluti,  
Alessio
