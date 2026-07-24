# FINAL campaign — one command

From the **Leonardo login node**, in the repo root (after `git pull`):

```bash
bash hpc_submit_finals_chain.sh
```

That’s it. You can leave: SLURM runs jobs **one after another** and writes everything under fixed dirs (resume-safe).

## What runs (in order)

1. **LOSS** — multi-seed (8), ks=`1,2,3,5,6`, 7 models (mono+poly QSA/CSA + 3 nl), error bars, aligned loss check  
2. **OBAR mono** — FINAL grid, `MAX_T=32`, `target_loss=3.8`, ks=`1,2,3,5,6`  
3. **OBAR + poly** — same folder; mono skipped via resume; poly overlaid on same plots  
4. **WRAPUP** — replot + `LOSS_CHECK.txt` + `INDEX.md`  
5. **(optional)** OBAR `T=64` if you set `TRY_T64=1` (PTB has only ~3 sentences at T=64)

## Where data lands

| What | Path |
|------|------|
| Campaign status / plan / checks | `results/final_campaign/definitive/` |
| Loss metrics + plots | `results/final_loss/definitive_T16_d8_ks1-2-3-5-6_L16_n8/` |
| Obar metrics + plots | `results/study/final_obar_T32_ks1-2-3-5-6_tl3.8/` |

## Monitor while away

```bash
squeue -u $USER
tail -f results/final_campaign/definitive/STATUS.txt
cat results/final_campaign/definitive/JOBS.txt
```

When finished:

```bash
cat results/final_campaign/definitive/INDEX.md
cat results/final_campaign/definitive/LOSS_CHECK.txt
```

## Useful overrides

```bash
# also attempt T=64 (weak data — duplicates sentences)
TRY_T64=1 bash hpc_submit_finals_chain.sh

# skip poly overlay on obar
INCLUDE_POLY_OBAR=0 bash hpc_submit_finals_chain.sh

# print sbatch plan without submitting
DRY_RUN=1 bash hpc_submit_finals_chain.sh
```

If a job times out: re-run the **same** `sbatch` / chain with the **same** `OUTPUT_DIR` — training resumes.
