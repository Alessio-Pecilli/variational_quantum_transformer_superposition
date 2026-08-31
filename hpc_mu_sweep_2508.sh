#!/bin/bash
#SBATCH --job-name=qsa2508_mu
#SBATCH --output=logs/qsa2508_mu_%j.out
#SBATCH --error=logs/qsa2508_mu_%j.err
#
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --account=iscrc_qusala
#
# μ vs T / μ vs d on qsa_bench_25_08 (classical PTB, trainable emb, circuit μ)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=4-00:00:00

set -euo pipefail

TS="${TS:-2:4:8:16:32}"
DS="${DS:-2:4:8:16:32}"
TS="${TS//:/,}"
DS="${DS//:/,}"
FIXED_D="${FIXED_D:-16}"
FIXED_T="${FIXED_T:-32}"
KS="${KS:-2:5:7}"
KS="${KS//:/,}"
LAYERS="${LAYERS:-0}"
EPOCHS="${EPOCHS:-300}"
TRAIN_SIZE="${TRAIN_SIZE:-64}"
TEST_SIZE="${TEST_SIZE:-32}"
N_SEEDS="${N_SEEDS:-10}"
MIN_EPOCHS="${MIN_EPOCHS:-40}"
PATIENCE="${PATIENCE:-30}"
LR="${LR:-2e-3}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MODEL_SEED_BASE="${MODEL_SEED_BASE:-42}"
DATA_SEED="${DATA_SEED:-7}"
EVAL_EVERY="${EVAL_EVERY:-20}"
OUTPUT_DIR="${OUTPUT_DIR:-results/qsa_bench_2508/mu_T${FIXED_T}_d${FIXED_D}_ks${KS//,/-}_n${N_SEEDS}_v2}"

echo "=== JOB ${SLURM_JOB_ID:-local} STARTED at $(date) on $(hostname) ==="
echo "mu sweep: Ts=$TS ds=$DS fixed_d=$FIXED_D fixed_T=$FIXED_T ks=$KS"
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

test -f run_mu_sweep_2508.py || { echo "ERROR: run_mu_sweep_2508.py missing"; exit 1; }
test -f qsa_bench_25_08.py || { echo "ERROR: qsa_bench_25_08.py missing"; exit 1; }
test -f ptb_sentences.txt || { echo "ERROR: ptb_sentences.txt missing"; exit 1; }
mkdir -p logs "$OUTPUT_DIR"

srun --mpi=pmix_v3 --mem=0 --export=ALL --cpu-bind=cores "$VENV_PY" run_mu_sweep_2508.py \
  --Ts "$TS" \
  --ds "$DS" \
  --fixed-d "$FIXED_D" \
  --fixed-T "$FIXED_T" \
  --ks "$KS" \
  --layers "$LAYERS" \
  --epochs "$EPOCHS" \
  --train-size "$TRAIN_SIZE" \
  --test-size "$TEST_SIZE" \
  --n-seeds "$N_SEEDS" \
  --min-epochs "$MIN_EPOCHS" \
  --patience "$PATIENCE" \
  --lr "$LR" \
  --batch-size "$BATCH_SIZE" \
  --model-seed-base "$MODEL_SEED_BASE" \
  --data-seed "$DATA_SEED" \
  --eval-every "$EVAL_EVERY" \
  --mu-at final \
  --output-dir "$OUTPUT_DIR"

echo "=== JOB FINISHED at $(date) ==="
echo "Plots: $OUTPUT_DIR/mu_vs_T.png  $OUTPUT_DIR/mu_vs_d.png"
