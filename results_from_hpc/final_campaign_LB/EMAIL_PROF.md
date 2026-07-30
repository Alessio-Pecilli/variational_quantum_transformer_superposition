# Email draft — L_B loss campaign (classical ansatz)

**To:** [professore]  
**Subject:** Nuova loss L_B — campagna classical-sequence (k-QSA/k-CSA) + confronto L_B vs L_half

---

Buongiorno Professore,

come concordato ho aggiornato il calcolo della loss seguendo i file in `new_loss/` (ansatz **non complessi**, classical-sequence):

- **k-QSA / k-CSA:** training con **L_B** = −log F, F = (T+1)/(2T)·μ/ζ (objective B del circuito, senza fasi trainabili)
- **nl-CSA:** training con **L_half_uniform** (= vecchia Renyi cross-entropy sui logits)

### Plot richiesti (10 seed, barre d'errore)

1. **Loss vs k — train** (`final_aligned_loss_vs_k.png`): L_B per k-QSA/k-CSA, L_half per nl-CSA  
2. **Loss vs k — test hold-out** (`final_aligned_loss_vs_k_test.png`)  
3. **Confronto mono — train** (`final_LB_vs_Lhalf_mono_train.png`): k-QSA + k-CSA, L_B vs L_half_uniform ai parametri finali  
4. **Confronto poly — train** (`final_LB_vs_Lhalf_poly_train.png`): poly-k-QSA + poly-k-CSA, stesso confronto  

(I due plot di confronto servono a verificare che L_B e L_half_uniform siano simili a convergenza.)

**Pack (plot quando job completo):**  
`results_from_hpc/final_campaign_LB/`

**HPC:** `hpc_final_loss_LB.sh` → `results/final_loss/LB_T16_d8_ks1-2-3-5-6_L16_n10_test/`

**Codice:** `--loss-objective L_B` in `run_final_loss.py`; implementazione in `qsa_training.py` + `classical_baselines.py`.

Branch: https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/tree/PennyLaneG

Cordiali saluti,  
Alessio
