# FINAL campaign — one command

From the **Leonardo login node**, in the repo root (after `git pull`):

```bash
scancel 50173780   # se ancora in coda il vecchio tentativo fallito
git pull
sbatch hpc_submit_finals_chain.sh
```

Questo job (partizione serial, ~minuti) fa solo da **orchestratore**: mette in coda
loss → obar → wrapup con dipendenze SLURM, poi termina. Puoi uscire.

## What runs (in order)

1. **LOSS** — multi-seed (8), ks=`1,2,3,5,6`, 7 modelli, barre d’errore  
2. **OBAR mono** — griglia FINAL, `MAX_T=32`, `target_loss=3.8`  
3. **OBAR + poly** — stesso folder (resume skippa il mono)  
4. **WRAPUP** — replot + `LOSS_CHECK.txt` + `INDEX.md`  
5. **(opz.)** OBAR `T=64` con `TRY_T64=1`

## Where data lands

| What | Path |
|------|------|
| Campaign status / plan / checks | `results/final_campaign/definitive/` |
| Loss | `results/final_loss/definitive_T16_d8_ks1-2-3-5-6_L16_n8/` |
| Obar | `results/study/final_obar_T32_ks1-2-3-5-6_tl3.8/` |

## Monitor

```bash
squeue -u $USER
tail -f results/final_campaign/definitive/STATUS.txt
cat results/final_campaign/definitive/JOBS.txt
```

## Overrides

```bash
sbatch --export=ALL,TRY_T64=1 hpc_submit_finals_chain.sh
sbatch --export=ALL,INCLUDE_POLY_OBAR=0 hpc_submit_finals_chain.sh
sbatch --export=ALL,SKIP_PREFLIGHT=1 hpc_submit_finals_chain.sh
```

Se un job scade: ri-lancia la catena (stessi `OUTPUT_DIR` → resume).
