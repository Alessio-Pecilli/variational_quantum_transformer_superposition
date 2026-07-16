#!/bin/bash
#SBATCH --job-name=qsa_all_s2
#SBATCH --output=logs/all_s2_%j.out
#SBATCH --error=logs/all_s2_%j.err
#
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --account=iscrc_qusala
#
# Section-2 campaign (classical): complexity → baselines → ppl-vs-k
# Runs phases ONE AFTER ANOTHER; each phase uses MPI via srun.
#
# MPI model:
#   - complexity (run_study --mpi): shard sweep points across ranks
#   - baselines / ppl-vs-k (--mpi): shard seeds across ranks
# Classical path uses JAX CPU (no per-rank GPU fight).
#
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=24:00:00

# Override examples:
#   sbatch --export=ALL,T=16,D=16,EPOCHS=300,N_SEEDS=5 hpc_all_section2.sh
#   sbatch --nodes=2 --ntasks-per-node=20 --export=ALL,SKIP_COMPLEXITY=1 hpc_all_section2.sh

set -euo pipefail

T="${T:-16}"
D="${D:-16}"
K="${K:-2}"
LAYERS="${LAYERS:-2}"
EPOCHS="${EPOCHS:-300}"
MAX_SENTENCES="${MAX_SENTENCES:-1000}"
N_SEEDS="${N_SEEDS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SEED="${SEED:-42}"
LR="${LR:-1e-3}"
NL_LR="${NL_LR:-5e-3}"
KS="${KS:-1 2 3 4}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-20}"
LOCAL_MAX_QUBITS="${LOCAL_MAX_QUBITS:-40}"
T_FIXED="${T_FIXED:-32}"
EXTRA_K="${EXTRA_K:-3}"
EXTRA_K3="${EXTRA_K3:-1}"
MODE="${MODE:-both}"

SKIP_COMPLEXITY="${SKIP_COMPLEXITY:-0}"
SKIP_BASELINES="${SKIP_BASELINES:-0}"
SKIP_PPL_K="${SKIP_PPL_K:-0}"

COMPLEXITY_DIR="${COMPLEXITY_DIR:-results/study/definitive_complexity_T${T_FIXED}_mode${MODE}}"
BASELINES_DIR="${BASELINES_DIR:-results/baselines_smoke/definitive_T${T}_d${D}_k${K}_ep${EPOCHS}_n${N_SEEDS}}"
PPL_ROOT="${PPL_ROOT:-results/baselines_smoke/ppl_vs_k_T${T}_d${D}_ep${EPOCHS}_n${N_SEEDS}}"

echo "=== JOB ${SLURM_JOB_ID:-local} STARTED at $(date) on $(hostname) ==="
echo "nodes=${SLURM_JOB_NUM_NODES:-?} tasks=${SLURM_NTASKS:-?} (MPI via srun)"
echo "phases: complexity=$([ "$SKIP_COMPLEXITY" = 1 ] && echo SKIP || echo RUN) |" \
     "baselines=$([ "$SKIP_BASELINES" = 1 ] && echo SKIP || echo RUN) |" \
     "ppl_vs_k=$([ "$SKIP_PPL_K" = 1 ] && echo SKIP || echo RUN)"

module purge
module load openmpi/4.1.6--gcc--12.2.0
module load python/3.11.7

source /leonardo_work/IscrC_QuSALa/venv_py311/bin/activate

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUTF8=1
export JAX_ENABLE_X64=True
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export UCX_TLS=self,shm,rc,ud
export UCX_RECONNECT_WAIT=15s
export UCX_CONNECT_TIMEOUT=300s
export UCX_MEMTYPE_CACHE=n

cd /leonardo_work/IscrC_QuSALa/vqt2/variational_quantum_transformer_superposition || exit 1
mkdir -p logs "$COMPLEXITY_DIR" "$BASELINES_DIR" "$PPL_ROOT"

run_mpi() {
  echo ""
  echo ">>>>> srun: $* <<<<<"
  srun --mpi=pmix_v3 "$@"
}

# --------------------------------------------------------------------------- #
# 1) Complexity / obar sweeps
# --------------------------------------------------------------------------- #
if [[ "$SKIP_COMPLEXITY" != "1" ]]; then
  echo ""
  echo "========== PHASE 1/3: COMPLEXITY (MPI shard sweep points) =========="
  CARGS=(
    --long --skip-self-check --mpi
    --local-max-qubits "$LOCAL_MAX_QUBITS"
    --output-dir "$COMPLEXITY_DIR"
    --d-sweep-T "$T_FIXED"
    --extra-k-on-d "$EXTRA_K"
  )
  if [[ "$EXTRA_K3" == "1" ]]; then
    CARGS+=(--extra-k3)
  fi
  case "$MODE" in
    T) CARGS+=(--only T) ;;
    d) CARGS+=(--only d) ;;
    both) ;;
    *) echo "Unknown MODE=$MODE"; exit 1 ;;
  esac
  run_mpi python3 run_study.py "${CARGS[@]}"
  echo "=== PHASE 1 DONE at $(date) ==="
else
  echo "=== PHASE 1 SKIPPED ==="
fi

# --------------------------------------------------------------------------- #
# 2) Baselines training curves (k-QSA / k-CSA / nl-CSA)
# --------------------------------------------------------------------------- #
if [[ "$SKIP_BASELINES" != "1" ]]; then
  echo ""
  echo "========== PHASE 2/3: BASELINES (MPI shard seeds) =========="
  run_mpi python3 run_baselines_smoke.py \
    --T "$T" \
    --d "$D" \
    --k "$K" \
    --layers "$LAYERS" \
    --epochs "$EPOCHS" \
    --max-sentences "$MAX_SENTENCES" \
    --n-seeds "$N_SEEDS" \
    --batch-size "$BATCH_SIZE" \
    --seed "$SEED" \
    --learning-rate "$LR" \
    --nl-learning-rate "$NL_LR" \
    --checkpoint-every "$CHECKPOINT_EVERY" \
    --output-dir "$BASELINES_DIR" \
    --resume \
    --mpi
  echo "=== PHASE 2 DONE at $(date) ==="
else
  echo "=== PHASE 2 SKIPPED ==="
fi

# --------------------------------------------------------------------------- #
# 3) PPL / curves vs k=1..4
# --------------------------------------------------------------------------- #
if [[ "$SKIP_PPL_K" != "1" ]]; then
  echo ""
  echo "========== PHASE 3/3: PPL vs k (MPI shard seeds, k sequential) =========="
  for KK in $KS; do
    OUT="$PPL_ROOT/k${KK}"
    mkdir -p "$OUT"
    echo "----- k=$KK -> $OUT -----"
    run_mpi python3 run_baselines_smoke.py \
      --T "$T" \
      --d "$D" \
      --k "$KK" \
      --layers "$LAYERS" \
      --epochs "$EPOCHS" \
      --max-sentences "$MAX_SENTENCES" \
      --n-seeds "$N_SEEDS" \
      --batch-size "$BATCH_SIZE" \
      --seed "$SEED" \
      --learning-rate "$LR" \
      --nl-learning-rate "$NL_LR" \
      --checkpoint-every "$CHECKPOINT_EVERY" \
      --output-dir "$OUT" \
      --resume \
      --mpi
  done
  echo "=== PHASE 3 DONE at $(date) ==="
else
  echo "=== PHASE 3 SKIPPED ==="
fi

echo ""
echo "=== JOB FINISHED at $(date) ==="
echo "complexity: $COMPLEXITY_DIR"
echo "baselines:  $BASELINES_DIR"
echo "ppl_vs_k:   $PPL_ROOT"
