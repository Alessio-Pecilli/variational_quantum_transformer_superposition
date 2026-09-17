# Email — μ vs T / μ vs d rifatti (advantage nuova, QSA/CSA separate, k=2,5,7)

**Subject:** Re: plot μ — advantage \(k^2\log d/m_k\), QSA≠CSA, μ a fine training, k=7

---

Buongiorno Professore,

come anticipato, i plot di **loss excess** li teniamo; abbiamo invece **rifatto interamente i plot di μ** correggendo i punti che aveva segnalato. Di seguito il riepilogo (solo campagna μ).

**Pack GitHub (108/108 celle + plot + summary):**  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/tree/PennyLaneG/results_from_hpc/qsa_bench_2508/mu_T32_d16_ks2-5-7_n10_v2

Codice: branch `PennyLaneG` — [`run_mu_sweep_2508.py`](https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/run_mu_sweep_2508.py), advantage in [`qsa_bench_25_08.py`](https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/qsa_bench_25_08.py) (`mu_advantage`).

---

## Correzioni rispetto ai plot precedenti

1. **Advantage aggiornata:**  
   \[
   \frac{k^2\,\log d}{m_k},\qquad m_k=\binom{d+k-1}{k}
   \]
   (prima era \(k/m_k\); la nuova soglia è più alta, es. d=16,k=2: **0.082** vs 0.015).

2. **Niente più “salti”:** QSA e CSA sono **curve separate** (stesso colore per k; marker/linestyle diversi). Prima mono/poly aggregavano QSA+CSA sulla stessa serie → due punti per ascissa.

3. **μ a fine training** (`train_mu_final`), non al best-checkpoint su \(L_B\).

4. Aggiunto **k=7** oltre a k=2,5.

Config invariata rispetto agli screen vecchi: PTB classico, embedding trainable, train \(L_B\), param-match QSA **1032** / CSA **1024**, 10 seed; μ = osservabile di circuito (con \(\sqrt{\alpha}\)), non \(\exp(-L_B)\).

| asse | fisso | sweep | k |
|------|-------|-------|---|
| μ vs T | d=16 | T ∈ {2,4,8,16,32} | 2, 5, 7 |
| μ vs d | T=32 | d ∈ {2,4,8,16,32} | 2, 5, 7 |

---

## Plot

**μ vs T (d=16)**  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/qsa_bench_2508/mu_T32_d16_ks2-5-7_n10_v2/mu_vs_T.png

**μ vs d (T=32)**  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/qsa_bench_2508/mu_T32_d16_ks2-5-7_n10_v2/mu_vs_d.png

Summary JSON:  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/qsa_bench_2508/mu_T32_d16_ks2-5-7_n10_v2/summary.json

---

## Lettura (onesta)

Anche confrontandosi con la soglia **vecchia** i risultati non erano buoni; con la soglia **nuova** (più alta) è ancora più chiaro:

- **Non superiamo** l’advantage threshold in generale. Unico punto sopra soglia: **kCSA-poly, T=2, d=16, k=7** (margin ≈ **1.5×**). A T=2, k=5, kCSA-poly arriva a ≈0.46×; kQSA-poly k=7 ≈0.44×.
- L’**andamento decrescente in T** resta (e si vede bene): a T=32, d=16 i margin sono ≪1 (tipicamente \(10^{-3}\)–\(10^{-1}\) per i poly, quasi 0 per i mono). Come aveva intuito, è coerente col cambio di definizione di μ (circuito + α), non un bug di plotting.
- **Poly ≫ mono**; alzare **k** aiuta i poly ad avvicinarsi alla soglia **solo a T piccoli**.
- Su **μ vs d** a T=32: tutte le curve restano sotto advantage; a d=32, k=7 i poly arrivano a margin ≈0.10–0.12× (miglior trend “relativo” al crescere di k, ma ancora lontani dal beat).

In sintesi: i plot ora sono coerenti (no salti, advantage corretta, μ finale, QSA/CSA confrontabili a parità di parametri), ma **il messaggio scientifico su μ resta cauto** — ci avviciniamo in pochi regimi (T piccolo, poly, k alto), senza un vantaggio stabile al crescere di T o d.

I plot **L₁^excess / L_B^excess** (già inviati, param-matched) restano validi:  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/tree/PennyLaneG/results_from_hpc/qsa_bench_2508/LB_excess_T32_d16_ks1-2-3-4-5-6_LmatchCSA_n8

Restiamo a disposizione per eventuali sweep mirati (es. solo poly a T piccoli, o altro k).

Cordiali saluti,  
Alessio
