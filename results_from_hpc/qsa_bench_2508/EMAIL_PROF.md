# Email draft — qsa_bench_25_08: hybrid readout + trainable embedding

**Subject:** Re: nuovo circuito (norme in readout) — L₁^excess / L_B^excess su PTB + TFIM, e μ (parziale)

---

Buongiorno Professore,

abbiamo implementato e lanciato su Leonardo la pipeline **`qsa_bench_25_08`**: readout ibrido in cui i token restano **non normalizzati** nell’embedding, si fa amplitude encoding su `ξ = x/√α` (`α = ||x||²`), e le norme entrano nella misura tramite il controllo `R_t` (`A ← √α · ⟨ξ|…⟩`). Si addestra su **L_B**; i plot riportano le loss **capacity-normalized** `L₁^excess` e `L_B^excess` (tolgono il floor dovuto al cap `p ≤ α`).

Parametri allineati come nella campagna precedente: **CSA/nl = 1024**, **QSA = 1032** (L=43).  
Job Leonardo: excess `54131638`, μ-sweep `54131647` (ancora in corso al momento della bozza; classical excess già completo).

Pack (aggiornato con classical + anteprima μ/quantum):

https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/tree/PennyLaneG/results_from_hpc/qsa_bench_2508

---

### 1) Validazione locale dell’embedding (norme nel circuito)

Su machine locale abbiamo verificato il percorso:

1. `Xvoc` non normalizzato → `α_t = ||y||²` (identità numerica, errore ~1e−16) e `||ξ|| = 1`.
2. Stessi `W,V` random: **con α** le loss grezze salgono dei floor (`L1` 4.11 vs 2.04 senza α); `L1 − L1_floor = L1^excess`.
3. Train corto PTB (T=8, d=8, k=2): QSA≈CSA su excess; `<ρ>` ~0.33–0.38.
4. Quantum TFIM con stati unitari (`α ≡ 1`): `L_B ≡ L_B^excess` (diff 0), come previsto.

Quindi le norme entrano automaticamente con la nuova versione del circuito; non serve un termine ad hoc oltre all’embedding addestrato congiuntamente a `W,V`.

---

### 2) CLASSICAL — PTB, embedding trainable (Leonardo, COMPLETE)

Setup: T=32 predizioni (frasi di 33 parole), d=16, k=1..6, **8 seed**, Adam su L_B, emb congiunto.

**L₁^excess train / test vs k**  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/qsa_bench_2508/LB_excess_T32_d16_ks1-2-3-4-5-6_LmatchCSA_n8/classical/plots/train_L1_excess_vs_k.png  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/qsa_bench_2508/LB_excess_T32_d16_ks1-2-3-4-5-6_LmatchCSA_n8/classical/plots/test_L1_excess_vs_k.png  

**L_B^excess train / test vs k** (solo k-models)  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/qsa_bench_2508/LB_excess_T32_d16_ks1-2-3-4-5-6_LmatchCSA_n8/classical/plots/train_LB_excess_vs_k.png  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/qsa_bench_2508/LB_excess_T32_d16_ks1-2-3-4-5-6_LmatchCSA_n8/classical/plots/test_LB_excess_vs_k.png  

**Lettura (train L₁^excess):** i kernel **poly** restano stabili e migliori dei mono; i **mono** (sia QSA sia CSA) **peggiorano con k** su PTB (fenomeno già sospettato nella campagna Markov precedente, qui con emb trainable + excess). Poly-k-CSA è il migliore (~0.23–0.36); poly-k-QSA ~0.41–0.53; nl-iso/gen ~0.44.

| modello (n) | L1x k=1 | L1x k=3 | L1x k=6 |
|---|---:|---:|---:|
| k-QSA (1032) | 0.58±0.19 | 1.23±0.12 | **1.73±0.10** |
| poly-k-QSA (1032) | 0.41±0.20 | 0.52±0.18 | 0.53±0.18 |
| k-CSA (1024) | 0.35±0.14 | 0.95±0.18 | 1.25±0.18 |
| poly-k-CSA (1024) | **0.23±0.09** | 0.32±0.14 | **0.36±0.16** |
| nl-iso (1024) | 0.44±0.14 | | |
| nl-gen (1024) | 0.44±0.12 | | |

L_B^excess train (stesso ordine): poly-CSA ~0.24–0.29; poly-QSA ~0.33–0.36; mono-QSA sale da ~0.40 (k=1) a ~0.72 (k=6).

---

### 3) QUANTUM — TFIM (Leonardo, PARZIALE)

Al momento sono completi solo gli aggregati **k-QSA mono** (8 seed). L₁^excess **scende bene con k** (comportamento “sano”, in contrasto col classical mono su PTB):

| k | L1x train | LBx train | L1x test |
|--:|---:|---:|---:|
| 1 | 0.93 | 0.94 | 0.94 |
| 3 | 0.60 | 0.56 | 0.62 |
| 6 | **0.35** | **0.34** | **0.34** |

CSA / poly / nl e i plot quantum arriveranno a fine job; aggiorno appena disponibili.

---

### 4) μ vs T / μ vs d (Leonardo, PARZIALE)

μ = osservabile di circuito (con √α), **non** `exp(−L_B)`. Advantage = `k / C(d+k−1,k)`.  
Celle finite finora: **d=16, T∈{2,4,8}**, k∈{2,5} (mono+poly); mancano T=16,32 e lo sweep vs d a T=32.

Anteprima μ vs T (d=16, parziale):  
https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/results_from_hpc/qsa_bench_2508/mu_T32_d16_ks2-5_n10/mu_vs_T.png  

Su queste celle piccole-T, poly supera spesso la soglia di advantage (specie k=5); mono resta sotto — coerente con il fatto che il regime T≪d è difficile per μ.

---

Aggiorno con quantum completo e pannelli μ vs T / μ vs d definitivi non appena i due job terminano.

Cordiali saluti,  
Alessio
