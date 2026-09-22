# Email — μ T=64 d=8 (k=7 QSA-poly)

**Subject:** Re: μ — punto aggiuntivo T=64, d=8 (k=7 QSA-poly)

---

Buongiorno Professore,

aggiungiamo un punto per i fit a **T grande**, stessa config dei punti d=8 già mandati (solo **kQSA-poly**, k=7, 10 seed, μ a fine training, advantage \(k^2\log d/m_k\)).

**T=64, d=8**  
μ = **4.16×10⁻⁴** ± 7.6×10⁻⁵ margin vs advantage ≈ **0.014×**

Rispetto a T=32 a d=8 (pack precedente: μ ≈ 3.22×10⁻⁴, margin ≈ 0.011×) μ **risale leggermente** — ancora ben sotto soglia; utile come ancoraggio per l’estrapolazione, da interpretare con cautela.

**Cell + summary:**  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/tree/PennyLaneG/results_from_hpc/qsa_bench_2508/mu_extrap_k7_qsa_poly_T64_d8-16_n10

**JSON aggregato k=7 QSA-poly** (include anche questo punto, stesso schema di prima):  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/qsa_bench_2508/mu_extrap_k7_qsa_poly_READY_for_fits.json

In corso: **T=64, d=16** (stessa config).

Cordiali saluti,  
Alessio
