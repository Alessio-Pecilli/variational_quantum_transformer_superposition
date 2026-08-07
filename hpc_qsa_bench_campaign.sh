#!/bin/bash
#SBATCH --job-name=qsa_bench_camp
#SBATCH --output=logs/qsa_bench_camp_%j.out
#SBATCH --error=logs/qsa_bench_camp_%j.err
#
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --account=iscrc_qusala
#
# Clean-slate shared-pipeline campaign (qsa_bench):
#   CLASSICAL + QUANTUM, multi-seed, k up to 6
#   plots: L1 train/test vs k (all models, no chance)
#          LB train/test vs k (k-models only)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=48:00:00

set -euo pipefail

T="${T:-32}"
D="${D:-16}"
N_QUBITS="${N_QUBITS:-4}"
# Slurm --export splits on commas: pass KS=1:2:3:4:5:6
KS="${KS:-1:2:3:4:5:6}"
KS="${KS//:/,}"
LAYERS="${LAYERS:-16}"
EPOCHS="${EPOCHS:-400}"
POLY_EPOCHS="${POLY_EPOCHS:-600}"
NL_EPOCHS="${NL_EPOCHS:-400}"
TRAIN_SIZE="${TRAIN_SIZE:-256}"
TEST_SIZE="${TEST_SIZE:-128}"
N_SEEDS="${N_SEEDS:-8}"
MIN_EPOCHS="${MIN_EPOCHS:-60}"
PATIENCE="${PATIENCE:-40}"
LR="${LR:-2e-3}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEED="${SEED:-7}"
MODEL_SEED_BASE="${MODEL_SEED_BASE:-1042}"
EVAL_EVERY="${EVAL_EVERY:-20}"
DT="${DT:-0.35}"
RHO="${RHO:-0.8}"
DATA_MODE="${DATA_MODE:-both}"
KS_TAG="${KS//,/-}"
OUTPUT_DIR="${OUTPUT_DIR:-results/qsa_bench_campaign/LB_L1_T${T}_d${D}_ks${KS_TAG}_L${LAYERS}_n${N_SEEDS}}"

echo "=== JOB ${SLURM_JOB_ID:-local} STARTED at $(date) on $(hostname) ==="
echo "qsa_bench campaign: mode=$DATA_MODE T=$T d=$D n_qubits=$N_QUBITS ks=$KS"
echo "layers=$LAYERS n_seeds=$N_SEEDS train=$TRAIN_SIZE test=$TEST_SIZE"
echo "epochs mono=$EPOCHS poly=$POLY_EPOCHS nl=$NL_EPOCHS lr=$LR batch=$BATCH_SIZE"
echo "OUTPUT_DIR=$OUTPUT_DIR"

module purge
module load openmpi/4.1.6--gcc--12.2.0
module load python/3.11.7

# shellcheck source=hpc_env.sh
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}/hpc_env.sh"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUTF8=1
export JAX_ENABLE_X64=True
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export MALLOC_ARENA_MAX=2

test -f run_qsa_bench_campaign.py || { echo "ERROR: run_qsa_bench_campaign.py not found in $(pwd)"; exit 1; }
test -f qsa_bench.py || { echo "ERROR: qsa_bench.py not found in $(pwd)"; exit 1; }
mkdir -p logs "$OUTPUT_DIR"

srun --mpi=pmix_v3 --mem=0 --export=ALL --cpu-bind=cores "$VENV_PY" run_qsa_bench_campaign.py \
  --data-mode "$DATA_MODE" \
  --T "$T" \
  --d "$D" \
  --n-qubits "$N_QUBITS" \
  --dt "$DT" \
  --classical-rho "$RHO" \
  --ks "$KS" \
  --layers "$LAYERS" \
  --epochs "$EPOCHS" \
  --poly-epochs "$POLY_EPOCHS" \
  --nl-epochs "$NL_EPOCHS" \
  --train-size "$TRAIN_SIZE" \
  --test-size "$TEST_SIZE" \
  --n-seeds "$N_SEEDS" \
  --min-epochs "$MIN_EPOCHS" \
  --patience "$PATIENCE" \
  --lr "$LR" \
  --batch-size "$BATCH_SIZE" \
  --seed "$SEED" \
  --model-seed-base "$MODEL_SEED_BASE" \
  --eval-every "$EVAL_EVERY" \
  --output-dir "$OUTPUT_DIR"

echo "=== JOB FINISHED at $(date) ==="
echo "Plots under:"
echo "  $OUTPUT_DIR/classical/plots/"
echo "  $OUTPUT_DIR/quantum/plots/"
