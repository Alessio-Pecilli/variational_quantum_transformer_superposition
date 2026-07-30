#!/bin/bash
# Monitor L_B campaign jobs and write status log.
set -euo pipefail
cd /leonardo_work/IscrC_QuSALa/variational_quantum_transformer_sovrapposition
OUT=results/final_loss/LB_T16_d8_ks1-2-3-5-6_L16_n10_test
LOG=logs/monitor_LB_finalize.log
JOB1=51004497
JOB2=51073063
mkdir -p logs
echo "$(date -Is) monitor start job1=$JOB1 job2=$JOB2" >> "$LOG"
while true; do
  st1=$(sacct -j "$JOB1" -n -X -o State | head -1 | tr -d ' ')
  st2=$(sacct -j "$JOB2" -n -X -o State | head -1 | tr -d ' ')
  n=$(find "$OUT" -name metrics.json 2>/dev/null | wc -l)
  echo "$(date -Is) job1=$st1 job2=$st2 metrics=$n" >> "$LOG"
  done1=0; done2=0
  case "$st1" in COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY) done1=1 ;; esac
  case "$st2" in COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY) done2=1 ;; esac
  if [[ "$done1" == 1 && "$done2" == 1 ]]; then
    echo "$(date -Is) BOTH_DONE st1=$st1 st2=$st2 metrics=$n" >> "$LOG"
    ls -lt "$OUT"/plots 2>/dev/null | head -20 >> "$LOG" || true
    test -f "$OUT/summary.json" && echo "summary=yes" >> "$LOG" || echo "summary=no" >> "$LOG"
    break
  fi
  sleep 600
done
