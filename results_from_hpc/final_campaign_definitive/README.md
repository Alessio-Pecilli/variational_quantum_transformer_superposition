# FINAL campaign (Leonardo, completed 27 Jul 2026)

Professor brief — multi-seed loss + ōbar grids on PTB (classical μ path).

**Branch:** [`PennyLaneG`](https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/tree/PennyLaneG)  
**Pack folder:** [`results_from_hpc/final_campaign_definitive/`](https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/tree/PennyLaneG/results_from_hpc/final_campaign_definitive)

## What was run

### Loss (aligned: −log μ + log T vs Renyi)
- **T=16, d=8**, **k ∈ {1,2,3,5,6}**, **8 seeds** (1042–1049)
- Models on the **same** vs-k plot:
  - k-QSA L=16 (blue solid) — 96 angles
  - k-CSA (orange solid) — 128 matrices
  - poly-k-QSA L=16 (blue dashed)
  - poly-k-CSA (orange dashed)
  - nl-CSA iso ~288 (black dashed), iso ~128 (black solid), gen ~128 (black dotted)
- Epochs: mono 400 / poly 600 / nl 500 / nl-gen 800; batch 16; max 800 sentences

### Ōbar (target_loss = 3.8, train until converged, max 2000 ep)
- **MAX_T=32** (PTB has only ~3 unique T=64 sentences — skipped)
- **k ∈ {1,2,3,5,6}**, mono + poly overlay
- Panels: ō vs T (d=16), ō vs T by d (k=3), ō vs d (T=32), ō vs d by T (k=3)
- Haar / adv references kept (same color, different dash); **no T_lim axvline**
- **90** grid points, **0** errors; **20/90** still above target 3.8 (hard corners, mostly high-k / low-d)

## Headline numbers (aligned loss, mean ± std)

| Model | k=1 | k=6 | Δ (k1→k6) |
|-------|-----|-----|-----------|
| k-QSA L=16 | 4.247 ± 0.09 | 6.596 ± 0.24 | **+2.35** |
| k-CSA | 4.743 ± 0.38 | 7.342 ± 0.98 | **+2.60** |
| poly-k-QSA L=16 | 4.242 ± 0.65 | 4.867 ± 0.06 | **+0.63** |
| poly-k-CSA | 4.833 ± 0.91 | 5.381 ± 0.39 | **+0.55** |

nl-CSA (k-indep. Renyi): iso~288 **6.51**, iso~128 **6.58**, gen~128 **6.73** — all above best μ (~4.24); likely under-trained / hard under this budget (see warnings in `campaign/LOSS_CHECK.txt`).

**Takeaway:** poly kernel flattens growth of aligned loss with k; monomial QSA/CSA still rise strongly with k.

## Files in this pack

### Loss
- [loss/plots/final_aligned_loss_vs_k.png](loss/plots/final_aligned_loss_vs_k.png) — **main figure**
- [loss/plots/](loss/plots/) — bars + raw/aligned curves per k
- [loss/summary.json](loss/summary.json) — full multi-seed aggregates
- [loss/aggregates/](loss/aggregates/) — per-(model,k) JSON

### Ōbar
- [obar/mean_O_vs_T.png](obar/mean_O_vs_T.png)
- [obar/mean_O_vs_T_by_d.png](obar/mean_O_vs_T_by_d.png)
- [obar/mean_O_vs_d.png](obar/mean_O_vs_d.png)
- [obar/mean_O_vs_d_by_T.png](obar/mean_O_vs_d_by_T.png)
- [obar/manifest.json](obar/manifest.json)

### Campaign meta
- [campaign/LOSS_CHECK.txt](campaign/LOSS_CHECK.txt) — comparability + points above target
- [campaign/STATUS.txt](campaign/STATUS.txt) — `CAMPAIGN_COMPLETE`
- [campaign/PLAN.json](campaign/PLAN.json)

## Code entry points (repo)

- [`run_final_loss.py`](https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/run_final_loss.py)
- [`run_study.py`](https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/run_study.py) (`--final-obar-grid`)
- [`hpc_final_loss.sh`](https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/hpc_final_loss.sh) / [`hpc_final_obar.sh`](https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/hpc_final_obar.sh)
- [`hpc_submit_finals_chain.sh`](https://github.com/Alessio-Pecilli/variational_quantum_transformer_superposition/blob/PennyLaneG/hpc_submit_finals_chain.sh)

## Caveats
1. nl-CSA remain high vs μ-models under the chosen epochs/budget — treat as provisional refs.
2. Ōbar: 20/90 points did not reach target_loss=3.8 (listed in LOSS_CHECK); ō values there are still reported.
3. T=64 omitted by design (tiny PTB support).
4. Full per-seed checkpoints stay on Leonardo (`results/final_loss/...`, `results/study/final_obar_...`); this pack is plots + summaries for review.
