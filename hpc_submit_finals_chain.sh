#!/bin/bash
#SBATCH --job-name=final_chain
#SBATCH --output=logs/final_chain_%j.out
#SBATCH --error=logs/final_chain_%j.err
#
#SBATCH --partition=lrd_all_serial
#SBATCH --account=iscrc_qusala
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#
# =============================================================================
# Orchestratore FINAL (da lanciare con sbatch).
# Non fa training: mette in coda la catena loss → obar → wrapup con
# --dependency=afterok, poi esce. Puoi uscire subito dopo lo sbatch.
#
#   sbatch hpc_submit_finals_chain.sh
#
# Override (esempi):
#   sbatch --export=ALL,TRY_T64=1,N_SEEDS=8 hpc_submit_finals_chain.sh
#   sbatch --export=ALL,INCLUDE_POLY_OBAR=0 hpc_submit_finals_chain.sh
#   sbatch --export=ALL,SKIP_PREFLIGHT=1 hpc_submit_finals_chain.sh
# =============================================================================

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$ROOT"
mkdir -p logs

N_SEEDS="${N_SEEDS:-8}"
KS="${KS:-1,2,3,5,6}"
LOSS_T="${LOSS_T:-16}"
LOSS_D="${LOSS_D:-8}"
QSA_LAYERS="${QSA_LAYERS:-16}"
MAX_T="${MAX_T:-32}"
FINAL_KS="${FINAL_KS:-1,2,3,5,6}"
TARGET_LOSS="${TARGET_LOSS:-3.8}"
INCLUDE_POLY_OBAR="${INCLUDE_POLY_OBAR:-1}"
TRY_T64="${TRY_T64:-0}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
DRY_RUN="${DRY_RUN:-0}"
CAMPAIGN_NAME="${CAMPAIGN_NAME:-definitive}"

KS_TAG="${KS//,/-}"
FINAL_KS_TAG="${FINAL_KS//,/-}"

LOSS_DIR="${LOSS_DIR:-results/final_loss/definitive_T${LOSS_T}_d${LOSS_D}_ks${KS_TAG}_L${QSA_LAYERS}_n${N_SEEDS}}"
OBAR_DIR="${OBAR_DIR:-results/study/final_obar_T${MAX_T}_ks${FINAL_KS_TAG}_tl${TARGET_LOSS}}"
OBAR64_DIR="${OBAR64_DIR:-results/study/final_obar_T64_ks${FINAL_KS_TAG}_tl${TARGET_LOSS}}"
CAMPAIGN_DIR="${CAMPAIGN_DIR:-results/final_campaign/${CAMPAIGN_NAME}}"

mkdir -p "$CAMPAIGN_DIR" "$LOSS_DIR" "$OBAR_DIR"
[[ "$TRY_T64" == "1" ]] && mkdir -p "$OBAR64_DIR"

STATUS="$CAMPAIGN_DIR/STATUS.txt"
PLAN="$CAMPAIGN_DIR/PLAN.json"
JOBS="$CAMPAIGN_DIR/JOBS.txt"
WRAPUP_SCRIPT="$CAMPAIGN_DIR/wrapup_job.sh"

_ts() { date -Iseconds 2>/dev/null || date; }

{
  echo "============================================================"
  echo "FINAL CAMPAIGN CHAIN  ($(_ts))  orchestrator_job=${SLURM_JOB_ID:-local}"
  echo "repo=$ROOT"
  echo "LOSS_DIR=$LOSS_DIR"
  echo "OBAR_DIR=$OBAR_DIR"
  echo "MAX_T=$MAX_T ks=$FINAL_KS target_loss=$TARGET_LOSS n_seeds=$N_SEEDS"
  echo "INCLUDE_POLY_OBAR=$INCLUDE_POLY_OBAR TRY_T64=$TRY_T64 DRY_RUN=$DRY_RUN"
  echo "============================================================"
} | tee "$STATUS"

# ---- env for preflight (venv py3.11, not login default py) -----------------
module purge
module load python/3.11.7
# shellcheck source=hpc_env.sh
source "$ROOT/hpc_env.sh"

if [[ "$SKIP_PREFLIGHT" != "1" && -f preflight_finals.py ]]; then
  echo "[preflight] using $VENV_PY ..." | tee -a "$STATUS"
  if ! "$VENV_PY" preflight_finals.py | tee -a "$STATUS"; then
    echo "[FATAL] preflight failed" | tee -a "$STATUS"
    exit 2
  fi
fi

cat > "$PLAN" <<EOF
{
  "created": "$(_ts)",
  "orchestrator_job": "${SLURM_JOB_ID:-local}",
  "campaign": "$CAMPAIGN_NAME",
  "n_seeds": $N_SEEDS,
  "ks_loss": "$KS",
  "obar_max_T": $MAX_T,
  "obar_ks": "$FINAL_KS",
  "target_loss": $TARGET_LOSS,
  "include_poly_obar": $INCLUDE_POLY_OBAR,
  "try_T64": $TRY_T64,
  "paths": {
    "loss": "$LOSS_DIR",
    "obar": "$OBAR_DIR",
    "obar_T64": "$OBAR64_DIR",
    "campaign": "$CAMPAIGN_DIR",
    "status": "$STATUS",
    "jobs": "$JOBS"
  }
}
EOF
echo "[plan] $PLAN" | tee -a "$STATUS"

# ---- wrapup child job ------------------------------------------------------
cat > "$WRAPUP_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=final_wrap
#SBATCH --output=logs/final_wrap_%j.out
#SBATCH --error=logs/final_wrap_%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --account=iscrc_qusala
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail
ROOT="$ROOT"
cd "\$ROOT"
export CAMPAIGN_DIR="$CAMPAIGN_DIR"
export LOSS_DIR="$LOSS_DIR"
export OBAR_DIR="$OBAR_DIR"
export OBAR64_DIR="$OBAR64_DIR"
STATUS="$STATUS"
REPORT="\$CAMPAIGN_DIR/LOSS_CHECK.txt"
INDEX="\$CAMPAIGN_DIR/INDEX.md"
JOBS_FILE="$JOBS"

module purge
module load python/3.11.7
source "\$ROOT/hpc_env.sh"

_ts() { date -Iseconds 2>/dev/null || date; }
echo "[wrapup] start \$(_ts)" | tee -a "\$STATUS"
mkdir -p logs "\$CAMPAIGN_DIR"

if [[ -f "\$LOSS_DIR/summary.json" ]]; then
  "\$VENV_PY" run_final_loss.py --replot-only "\$LOSS_DIR" || true
  echo "[wrapup] replotted loss" | tee -a "\$STATUS"
else
  echo "[wrapup] WARN missing \$LOSS_DIR/summary.json" | tee -a "\$STATUS"
fi

if [[ -f "\$OBAR_DIR/manifest.json" ]]; then
  "\$VENV_PY" run_study.py --replot-only "\$OBAR_DIR" || true
  echo "[wrapup] replotted obar" | tee -a "\$STATUS"
else
  echo "[wrapup] WARN missing \$OBAR_DIR/manifest.json" | tee -a "\$STATUS"
fi

if [[ -n "\$OBAR64_DIR" && -f "\$OBAR64_DIR/manifest.json" ]]; then
  "\$VENV_PY" run_study.py --replot-only "\$OBAR64_DIR" || true
  echo "[wrapup] replotted obar T64" | tee -a "\$STATUS"
fi

"\$VENV_PY" - <<'PY' | tee "\$REPORT"
import json, os
from pathlib import Path

loss_dir = Path(os.environ["LOSS_DIR"])
obar_dir = Path(os.environ["OBAR_DIR"])
lines = ["LOSS COMPARABILITY / CONVERGENCE REPORT", "=" * 60]
sp = loss_dir / "summary.json"
if not sp.exists():
    lines.append("MISSING " + str(sp))
    print("\\n".join(lines))
    raise SystemExit(0)
s = json.loads(sp.read_text(encoding="utf-8"))
cfg = s.get("config", {})
lines.append("T=%s d=%s ks=%s n_seeds=%s" % (cfg.get("T"), cfg.get("d"), cfg.get("ks"), cfg.get("n_seeds")))
lines.append(str(s.get("alignment_note", "")))
lines.append("")
lines.append("Mu-models (aligned = -log mu + log T):")
for p in sorted(s.get("mu_points_vs_k") or [], key=lambda x: (x["model"], int(x["k"]))):
    lines.append("  %-22s k=%s: %.4f +/- %.4f" % (p["model"], p["k"], p["aligned_loss_mean"], p["aligned_loss_std"]))
lines.append("")
lines.append("nl-CSA (Renyi, k-independent):")
for a in s.get("nl_refs") or []:
    lines.append("  %-22s: %.4f +/- %.4f" % (a["model"], a["aligned_loss_mean"], a.get("aligned_loss_std", 0)))
lines.append("")
lines.append("Trend notes:")
for t in s.get("trend_notes") or []:
    lines.append("  " + str(t))
lines.append("")
lines.append("Convergence warnings:")
warns = s.get("convergence_warnings") or []
if not warns:
    lines.append("  (none)")
else:
    for w in warns:
        lines.append("  - " + str(w))
mp = obar_dir / "manifest.json"
lines.append("")
lines.append("OBAR loss check:")
if mp.exists():
    m = json.loads(mp.read_text(encoding="utf-8"))
    rows = m.get("all_rows") or []
    target = float((m.get("train_opts") or {}).get("target_loss") or 3.8)
    bad = [r for r in rows if r.get("final_loss") is not None and float(r["final_loss"]) > target and r.get("error") is None]
    lines.append("  points=%d  above_target(%.1f)=%d" % (len(rows), target, len(bad)))
    for r in sorted(bad, key=lambda x: float(x["final_loss"]), reverse=True)[:25]:
        lines.append("  %s: loss=%.4f obar=%.4f" % (r.get("label"), float(r["final_loss"]), float(r.get("obar", float("nan")))))
else:
    lines.append("  MISSING " + str(mp))
print("\\n".join(lines))
PY

{
  echo "# FINAL campaign index"
  echo
  echo "- Wrapup: \$(_ts)"
  echo "- Loss data: \`\$LOSS_DIR\`"
  echo "- Obar data: \`\$OBAR_DIR\`"
  echo "- Loss check: \`\$REPORT\`"
  echo "- Status: \`\$STATUS\`"
  echo "- Jobs: \`\$JOBS_FILE\`"
  echo
  echo "## Loss plots"
  echo "- \`\$LOSS_DIR/plots/final_aligned_loss_vs_k.png\`"
  echo "- \`\$LOSS_DIR/summary.json\`"
  echo
  echo "## Obar plots"
  echo "- \`\$OBAR_DIR/mean_O_vs_T.png\`"
  echo "- \`\$OBAR_DIR/mean_O_vs_T_by_d.png\`"
  echo "- \`\$OBAR_DIR/mean_O_vs_d.png\`"
  echo "- \`\$OBAR_DIR/mean_O_vs_d_by_T.png\`"
  echo "- \`\$OBAR_DIR/manifest.json\`"
} > "\$INDEX"

echo "[wrapup] DONE \$(_ts)" | tee -a "\$STATUS"
echo "CAMPAIGN_COMPLETE \$(_ts)" >> "\$STATUS"
EOF
chmod +x "$WRAPUP_SCRIPT"

_submit() {
  local label="$1"; shift
  echo "[submit] $label :: $*" | tee -a "$STATUS" >&2
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_$label" | tee -a "$JOBS" >&2
    echo "DRY_$label"
    return 0
  fi
  local jid
  jid="$(sbatch --parsable "$@")"
  echo "$label $jid" | tee -a "$JOBS" >&2
  echo "$jid"
}

: > "$JOBS"

J_LOSS="$(_submit loss \
  --export=ALL,T="${LOSS_T}",D="${LOSS_D}",KS="${KS}",N_SEEDS="${N_SEEDS}",QSA_LAYERS="${QSA_LAYERS}",OUTPUT_DIR="${LOSS_DIR}" \
  "$ROOT/hpc_final_loss.sh")"

if [[ "$DRY_RUN" == "1" ]]; then DEP_LOSS=(); else DEP_LOSS=(--dependency="afterok:${J_LOSS}"); fi

J_OBAR="$(_submit obar_mono \
  "${DEP_LOSS[@]}" \
  --export=ALL,MAX_T="${MAX_T}",FINAL_KS="${FINAL_KS}",TARGET_LOSS="${TARGET_LOSS}",ALSO_POLY=0,OUTPUT_DIR="${OBAR_DIR}" \
  "$ROOT/hpc_final_obar.sh")"

PREV="$J_OBAR"
J_POLY="(skipped)"
if [[ "$INCLUDE_POLY_OBAR" == "1" ]]; then
  if [[ "$DRY_RUN" == "1" ]]; then DEP_POLY=(); else DEP_POLY=(--dependency="afterok:${J_OBAR}"); fi
  J_POLY="$(_submit obar_poly \
    "${DEP_POLY[@]}" \
    --export=ALL,MAX_T="${MAX_T}",FINAL_KS="${FINAL_KS}",TARGET_LOSS="${TARGET_LOSS}",ALSO_POLY=1,OUTPUT_DIR="${OBAR_DIR}" \
    "$ROOT/hpc_final_obar.sh")"
  PREV="$J_POLY"
fi

J_T64="(skipped)"
if [[ "$TRY_T64" == "1" ]]; then
  if [[ "$DRY_RUN" == "1" ]]; then DEP_T64=(); else DEP_T64=(--dependency="afterok:${PREV}"); fi
  POLY64=0; [[ "$INCLUDE_POLY_OBAR" == "1" ]] && POLY64=1
  J_T64="$(_submit obar_T64 \
    "${DEP_T64[@]}" \
    --export=ALL,MAX_T=64,FINAL_KS="${FINAL_KS}",TARGET_LOSS="${TARGET_LOSS}",ALSO_POLY="${POLY64}",OUTPUT_DIR="${OBAR64_DIR}" \
    "$ROOT/hpc_final_obar.sh")"
  PREV="$J_T64"
fi

if [[ "$DRY_RUN" == "1" ]]; then DEP_WRAP=(); else DEP_WRAP=(--dependency="afterok:${PREV}"); fi
J_WRAP="$(_submit wrapup \
  "${DEP_WRAP[@]}" \
  --export=ALL,CAMPAIGN_DIR="${CAMPAIGN_DIR}",LOSS_DIR="${LOSS_DIR}",OBAR_DIR="${OBAR_DIR}",OBAR64_DIR="${OBAR64_DIR}" \
  "$WRAPUP_SCRIPT")"

{
  echo
  echo "CHAIN QUEUED $(_ts)"
  echo "  orchestrator = ${SLURM_JOB_ID:-local}"
  echo "  1 loss      = $J_LOSS"
  echo "  2 obar_mono = $J_OBAR"
  echo "  3 obar_poly = $J_POLY"
  echo "  4 obar_T64  = $J_T64"
  echo "  5 wrapup    = $J_WRAP"
  echo
  echo "Monitor: squeue -u \$USER"
  echo "Status:  tail -f $STATUS"
  echo "Jobs:    cat $JOBS"
} | tee -a "$STATUS"

echo "OK — catena in coda. Orchestratore termina qui."
