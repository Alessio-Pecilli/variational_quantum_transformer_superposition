#!/bin/bash
#SBATCH --job-name=final_obar
#SBATCH --output=logs/final_obar_%j.out
#SBATCH --error=logs/final_obar_%j.err
#
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --account=iscrc_qusala
#
# FINAL obar (transformer / run_study) campaign — classical mu path.
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=48:00:00

# Panels (via run_study.py --final-obar-grid):
#   obar vs T:      d=16, k in FINAL_KS (default 1,2,3,5,6), T up to MAX_T
#   obar vs T by d: k=3, d=4,8,16, T up to MAX_T
#   obar vs d:      T=MAX_T, k=2,3, d=2..32
#   obar vs d by T: k=3, T=8,16,32(,64), all d
#
# DEFAULT MAX_T=32 (safe): PTB has only ~3 unique T=64 sentences.
# To try T=64 anyway:  sbatch --export=ALL,MAX_T=64 hpc_final_obar.sh
# ALSO_POLY=1 → also train poly kernel and overlay mono+poly on the same plots.
# target_loss default 3.8. Fixed OUTPUT_DIR → resume.
#
# Examples:
#   sbatch hpc_final_obar.sh
#   sbatch --export=ALL,MAX_T=32,ALSO_POLY=1 hpc_final_obar.sh
#   sbatch --export=ALL,MAX_T=64 hpc_final_obar.sh   # WARN: tiny T=64 corpus
#   sbatch --export=ALL,FINAL_KS=1,2,3,5,6 hpc_final_obar.sh

set -euo pipefail

MAX_T="${MAX_T:-32}"
# Slurm --export splits on commas: parents pass FINAL_KS as 1:2:3:5:6
FINAL_KS="${FINAL_KS:-1,2,3,5,6}"
FINAL_KS="${FINAL_KS//:/,}"
ALSO_POLY="${ALSO_POLY:-0}"
LOCAL_MAX_QUBITS="${LOCAL_MAX_QUBITS:-48}"
TARGET_LOSS="${TARGET_LOSS:-3.8}"
MAX_EPOCHS="${MAX_EPOCHS:-2000}"
EPOCHS="${EPOCHS:-}"
MAX_SENTENCES="${MAX_SENTENCES:-}"
KERNEL_MODE="${KERNEL_MODE:-monomial}"
POLY_TAG=""
if [[ "$ALSO_POLY" == "1" ]]; then
  POLY_TAG="_pluspoly"
fi
KS_TAG="${FINAL_KS//,/-}"
OUTPUT_DIR="${OUTPUT_DIR:-results/study/final_obar_T${MAX_T}_ks${KS_TAG}_tl${TARGET_LOSS}${POLY_TAG}}"

echo "=== JOB ${SLURM_JOB_ID:-local} STARTED at $(date) on $(hostname) ==="
echo "FINAL obar: MAX_T=$MAX_T ks=$FINAL_KS ALSO_POLY=$ALSO_POLY TARGET_LOSS=$TARGET_LOSS"
if [[ "$MAX_T" -ge 64 ]]; then
  echo "[WARN] PTB has only ~3 unique T=64 sentences; prefer MAX_T=32 unless you accept heavy duplication."
fi
echo "LOCAL_MAX_QUBITS=$LOCAL_MAX_QUBITS KERNEL_MODE=$KERNEL_MODE MAX_EPOCHS=$MAX_EPOCHS"
echo "OUTPUT_DIR=$OUTPUT_DIR (re-sbatch same dir to resume)"
echo "MPI tasks=${SLURM_NTASKS:-1}"

module purge
module load openmpi/4.1.6--gcc--12.2.0
module load python/3.11.7

# shellcheck source=hpc_env.sh
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}/hpc_env.sh"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUTF8=1
export JAX_ENABLE_X64=True
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export UCX_TLS=self,shm,rc,ud
export UCX_MEMTYPE_CACHE=n

test -f run_study.py || { echo "ERROR: run_study.py not found in $(pwd)"; exit 1; }
mkdir -p logs "$OUTPUT_DIR"

ARGS=(
  --long
  --skip-self-check
  --mpi
  --final-obar-grid
  --max-T "$MAX_T"
  --final-obar-ks "$FINAL_KS"
  --local-max-qubits "$LOCAL_MAX_QUBITS"
  --output-dir "$OUTPUT_DIR"
  --kernel-mode "$KERNEL_MODE"
  --target-loss "$TARGET_LOSS"
  --max-epochs "$MAX_EPOCHS"
)
if [[ "$ALSO_POLY" == "1" ]]; then
  ARGS+=(--also-poly)
fi
if [[ -n "$EPOCHS" ]]; then
  ARGS+=(--epochs "$EPOCHS")
fi
if [[ -n "$MAX_SENTENCES" ]]; then
  ARGS+=(--max-sentences "$MAX_SENTENCES")
fi

srun --mpi=pmix_v3 --export=ALL "$VENV_PY" run_study.py "${ARGS[@]}"

echo "=== JOB FINISHED at $(date) ==="
echo "Replot aesthetics:  $VENV_PY run_study.py --replot-only $OUTPUT_DIR"
