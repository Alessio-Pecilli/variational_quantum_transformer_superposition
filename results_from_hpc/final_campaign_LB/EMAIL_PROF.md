# Email draft — L_B loss campaign (classical ansatz)

**To:** [professore]  
**Subject:** Nuova loss L_B — campagna classical-sequence + L_half / L_1 + curves k=3

---

Buongiorno Professore,

come concordato ho aggiornato il calcolo della loss seguendo i file in `new_loss/` (ansatz **non complessi**, classical-sequence):

- **k-QSA / k-CSA:** training con **L_B** = −log F; eval di **L_half** e **L_1** (Shannon CE uniforme) ai parametri finali
- **nl-CSA (Renyi):** training con **L_half_uniform** (plot L_half)
- **nl-CSA (CE):** training con **L_1** Shannon CE (plot L_1)

### Plot

1. Loss vs k — train/test (**L_B** per k-QSA/k-CSA, L_half per nl Renyi)
2. Confronto mono/poly — L_B vs L_half ai parametri finali (train)
3. **L_half vs k** — train + test (tutti i modelli; attesi ~3)
4. **L_1 vs k** — train + test (tutti; nl con training CE; attesi ~3)
5. **Training curves L_B a k=3** (k-QSA / k-CSA)

**Pack:** `results_from_hpc/final_campaign_LB/`  
**HPC:** `hpc_final_loss_LB.sh` → `results/final_loss/LB_T16_d8_ks1-2-3-5-6_L16_n10_test/`

Branch: https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/tree/PennyLaneG

Cordiali saluti,  
Alessio
