# Email — μ T=64 d=16 (k=7 QSA-poly)

**Subject:** Re: μ — punto T=64, d=16 (k=7 QSA-poly)

---

Buongiorno Professore,

completato anche il punto **T=64, d=16** (stessa config: solo kQSA-poly, k=7, 10 seed, μ a fine training).

| T | d | μ (mean ± std) | margin vs adv |
|---|--:|---------------:|--------------:|
| 64 | 16 | **2.34×10⁻⁵** ± 3.0×10⁻⁶ | **0.029×** |

Per confronto, a d=16 T=32 (pack precedente) era μ ≈ 3.07×10⁻⁵ (margin ≈ 0.039×): a T=64 μ continua a scendere leggermente, ancora sotto soglia.

**Pack T=64 (d=8 + d=16):**  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/tree/PennyLaneG/results_from_hpc/qsa_bench_2508/mu_extrap_k7_qsa_poly_T64_d8-16_n10

**JSON aggregato k=7 QSA-poly** (aggiornato, 15 punti):  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/qsa_bench_2508/mu_extrap_k7_qsa_poly_READY_for_fits.json

Cordiali saluti,  
Alessio
