#!/bin/bash
#SBATCH --job-name=final_rechain
#SBATCH --output=logs/final_rechain_%j.out
#SBATCH --error=logs/final_rechain_%j.err
#SBATCH --partition=lrd_all_serial
#SBATCH --account=iscrc_qusala
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#
# Re-queue FINAL chain AFTER a loss OOM / failed dependency.
# Same OUTPUT_DIR → loss resumes; then obar → wrapup.
#
#   sbatch hpc_rechain_after_loss_oom.sh

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$ROOT"
mkdir -p logs results/final_campaign/definitive

LOSS_DIR="${LOSS_DIR:-results/final_loss/definitive_T16_d8_ks1-2-3-5-6_L16_n8}"
OBAR_DIR="${OBAR_DIR:-results/study/final_obar_T32_ks1-2-3-5-6_tl3.8}"
CAMPAIGN_DIR="${CAMPAIGN_DIR:-results/final_campaign/definitive}"
FINAL_KS="${FINAL_KS:-1,2,3,5,6}"
TARGET_LOSS="${TARGET_LOSS:-3.8}"
MAX_T="${MAX_T:-32}"
INCLUDE_POLY_OBAR="${INCLUDE_POLY_OBAR:-1}"
STATUS="$CAMPAIGN_DIR/STATUS_rechain.txt"
JOBS="$CAMPAIGN_DIR/JOBS_rechain.txt"

_ts() { date -Iseconds 2>/dev/null || date; }
{
  echo "RECHAIN $(_ts)"
  echo "LOSS_DIR=$LOSS_DIR OBAR_DIR=$OBAR_DIR"
} | tee "$STATUS"
: > "$JOBS"

J1=$(sbatch --parsable --export=ALL,OUTPUT_DIR="$LOSS_DIR" "$ROOT/hpc_final_loss.sh")
echo "loss $J1" | tee -a "$JOBS" "$STATUS"

J2=$(sbatch --parsable --dependency="afterok:${J1}" \
  --export=ALL,MAX_T="$MAX_T",FINAL_KS="$FINAL_KS",TARGET_LOSS="$TARGET_LOSS",ALSO_POLY=0,OUTPUT_DIR="$OBAR_DIR" \
  "$ROOT/hpc_final_obar.sh")
echo "obar_mono $J2" | tee -a "$JOBS" "$STATUS"

PREV=$J2
if [[ "$INCLUDE_POLY_OBAR" == "1" ]]; then
  J3=$(sbatch --parsable --dependency="afterok:${J2}" \
    --export=ALL,MAX_T="$MAX_T",FINAL_KS="$FINAL_KS",TARGET_LOSS="$TARGET_LOSS",ALSO_POLY=1,OUTPUT_DIR="$OBAR_DIR" \
    "$ROOT/hpc_final_obar.sh")
  echo "obar_poly $J3" | tee -a "$JOBS" "$STATUS"
  PREV=$J3
fi

# wrapup script must exist from previous chain; recreate minimal if missing
WRAP="$CAMPAIGN_DIR/wrapup_job.sh"
if [[ ! -f "$WRAP" ]]; then
  echo "[WARN] missing wrapup_job.sh — skip wrapup submit" | tee -a "$STATUS"
else
  J4=$(sbatch --parsable --dependency="afterok:${PREV}" \
    --export=ALL,CAMPAIGN_DIR="$CAMPAIGN_DIR",LOSS_DIR="$LOSS_DIR",OBAR_DIR="$OBAR_DIR",OBAR64_DIR="" \
    "$WRAP")
  echo "wrapup $J4" | tee -a "$JOBS" "$STATUS"
fi

echo "RECHAIN QUEUED $(_ts)" | tee -a "$STATUS"
cat "$JOBS" | tee -a "$STATUS"
