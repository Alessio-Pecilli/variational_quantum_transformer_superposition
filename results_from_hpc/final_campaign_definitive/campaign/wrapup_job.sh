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
ROOT="/leonardo_work/IscrC_QuSALa/variational_quantum_transformer_sovrapposition"
cd "$ROOT"
export CAMPAIGN_DIR="results/final_campaign/definitive"
export LOSS_DIR="results/final_loss/definitive_T16_d8_ks1-2-3-5-6_L16_n8"
export OBAR_DIR="results/study/final_obar_T32_ks1-2-3-5-6_tl3.8"
export OBAR64_DIR="results/study/final_obar_T64_ks1-2-3-5-6_tl3.8"
STATUS="results/final_campaign/definitive/STATUS.txt"
REPORT="$CAMPAIGN_DIR/LOSS_CHECK.txt"
INDEX="$CAMPAIGN_DIR/INDEX.md"
JOBS_FILE="results/final_campaign/definitive/JOBS.txt"

module purge
module load python/3.11.7
source "$ROOT/hpc_env.sh"

_ts() { date -Iseconds 2>/dev/null || date; }
echo "[wrapup] start $(_ts)" | tee -a "$STATUS"
mkdir -p logs "$CAMPAIGN_DIR"

if [[ -f "$LOSS_DIR/summary.json" ]]; then
  "$VENV_PY" run_final_loss.py --replot-only "$LOSS_DIR" || true
  echo "[wrapup] replotted loss" | tee -a "$STATUS"
else
  echo "[wrapup] WARN missing $LOSS_DIR/summary.json" | tee -a "$STATUS"
fi

if [[ -f "$OBAR_DIR/manifest.json" ]]; then
  "$VENV_PY" run_study.py --replot-only "$OBAR_DIR" || true
  echo "[wrapup] replotted obar" | tee -a "$STATUS"
else
  echo "[wrapup] WARN missing $OBAR_DIR/manifest.json" | tee -a "$STATUS"
fi

if [[ -n "$OBAR64_DIR" && -f "$OBAR64_DIR/manifest.json" ]]; then
  "$VENV_PY" run_study.py --replot-only "$OBAR64_DIR" || true
  echo "[wrapup] replotted obar T64" | tee -a "$STATUS"
fi

"$VENV_PY" - <<'PY' | tee "$REPORT"
import json, os
from pathlib import Path

loss_dir = Path(os.environ["LOSS_DIR"])
obar_dir = Path(os.environ["OBAR_DIR"])
lines = ["LOSS COMPARABILITY / CONVERGENCE REPORT", "=" * 60]
sp = loss_dir / "summary.json"
if not sp.exists():
    lines.append("MISSING " + str(sp))
    print("\n".join(lines))
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
print("\n".join(lines))
PY

{
  echo "# FINAL campaign index"
  echo
  echo "- Wrapup: $(_ts)"
  echo "- Loss data: `$LOSS_DIR`"
  echo "- Obar data: `$OBAR_DIR`"
  echo "- Loss check: `$REPORT`"
  echo "- Status: `$STATUS`"
  echo "- Jobs: `$JOBS_FILE`"
  echo
  echo "## Loss plots"
  echo "- `$LOSS_DIR/plots/final_aligned_loss_vs_k.png`"
  echo "- `$LOSS_DIR/summary.json`"
  echo
  echo "## Obar plots"
  echo "- `$OBAR_DIR/mean_O_vs_T.png`"
  echo "- `$OBAR_DIR/mean_O_vs_T_by_d.png`"
  echo "- `$OBAR_DIR/mean_O_vs_d.png`"
  echo "- `$OBAR_DIR/mean_O_vs_d_by_T.png`"
  echo "- `$OBAR_DIR/manifest.json`"
} > "$INDEX"

echo "[wrapup] DONE $(_ts)" | tee -a "$STATUS"
echo "CAMPAIGN_COMPLETE $(_ts)" >> "$STATUS"
