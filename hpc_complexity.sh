#!/bin/bash
#SBATCH --job-name=qsa_complexity
#SBATCH --output=logs/complexity_%j.out
#SBATCH --error=logs/complexity_%j.err
#
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --account=iscrc_qusala
#
# Obar scaling study (classical mu path). Training does not need the quantum circuit.
# Prefer hpc_all_section2.sh for the full MPI campaign.
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=12:00:00

# Examples:
#   sbatch hpc_complexity.sh
#   sbatch --export=ALL,MODE=d,T_FIXED=8,EXTRA_K=3,LOCAL_MAX_QUBITS=40 hpc_complexity.sh
#   sbatch --export=ALL,MODE=T,EXTRA_K3=1,LOCAL_MAX_QUBITS=40 hpc_complexity.sh

set -euo pipefail

MODE="${MODE:-both}"          # T | d | both
T_FIXED="${T_FIXED:-16}"       # for sweep vs d
EXTRA_K="${EXTRA_K:-3}"       # second k on d-sweep
EXTRA_K3="${EXTRA_K3:-1}"     # 1 => also k=3 on T-sweep
LOCAL_MAX_QUBITS="${LOCAL_MAX_QUBITS:-40}"
EPOCHS="${EPOCHS:-}"          # empty => run_study --long defaults
MAX_SENTENCES="${MAX_SENTENCES:-}"

echo "=== JOB ${SLURM_JOB_ID:-local} STARTED at $(date) on $(hostname) ==="
echo "complexity: MODE=$MODE T_FIXED=$T_FIXED EXTRA_K=$EXTRA_K LOCAL_MAX_QUBITS=$LOCAL_MAX_QUBITS"
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

OUTPUT_DIR="${OUTPUT_DIR:-results/study/definitive_complexity_T${T_FIXED}_mode${MODE}}"

test -f run_study.py || { echo "ERROR: run_study.py not found in $(pwd)"; exit 1; }
mkdir -p logs "$OUTPUT_DIR"

ARGS=(--long --skip-self-check --mpi --local-max-qubits "$LOCAL_MAX_QUBITS" --output-dir "$OUTPUT_DIR")
if [[ -n "$EPOCHS" ]]; then
  ARGS+=(--epochs "$EPOCHS")
fi
if [[ -n "$MAX_SENTENCES" ]]; then
  ARGS+=(--max-sentences "$MAX_SENTENCES")
fi

case "$MODE" in
  T)
    ARGS+=(--only T)
    if [[ "$EXTRA_K3" == "1" ]]; then
      ARGS+=(--extra-k3)
    fi
    ;;
  d)
    ARGS+=(--only d --d-sweep-T "$T_FIXED" --extra-k-on-d "$EXTRA_K")
    ;;
  both)
    if [[ "$EXTRA_K3" == "1" ]]; then
      ARGS+=(--extra-k3)
    fi
    ARGS+=(--d-sweep-T "$T_FIXED" --extra-k-on-d "$EXTRA_K")
    ;;
  *)
    echo "Unknown MODE=$MODE (use T|d|both)" >&2
    exit 1
    ;;
esac

echo "OUTPUT_DIR=$OUTPUT_DIR (re-sbatch same dir skips completed labels)"
echo "srun $VENV_PY run_study.py ${ARGS[*]}"
srun --mpi=pmix_v3 --export=ALL "$VENV_PY" run_study.py "${ARGS[@]}"

echo "=== JOB FINISHED at $(date) ==="
