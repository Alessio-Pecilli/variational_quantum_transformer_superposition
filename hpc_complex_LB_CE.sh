#!/bin/bash
#SBATCH --job-name=cseq_LB_CE
#SBATCH --output=logs/cseq_LB_CE_%j.out
#SBATCH --error=logs/cseq_LB_CE_%j.err
#
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --account=iscrc_qusala
#
# Classical Markov + COMPLEX ansatz: L_B + CE_unif (L_1) with shared p_j=f_j^2.
# k=1,2,3,5 ; n_seeds=10 ; param counts on plots.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=24:00:00

set -euo pipefail

T="${T:-32}"
D="${D:-16}"
KS="${KS:-1,2,3,5}"
KS="${KS//:/,}"
LAYERS="${LAYERS:-16}"
EPOCHS="${EPOCHS:-400}"
POLY_EPOCHS="${POLY_EPOCHS:-600}"
NL_EPOCHS="${NL_EPOCHS:-400}"
TRAIN_SIZE="${TRAIN_SIZE:-256}"
TEST_SIZE="${TEST_SIZE:-128}"
N_SEEDS="${N_SEEDS:-10}"
MIN_EPOCHS="${MIN_EPOCHS:-60}"
PATIENCE="${PATIENCE:-40}"
LR="${LR:-2e-3}"
SEED="${SEED:-7}"
MODEL_SEED_BASE="${MODEL_SEED_BASE:-1042}"
EVAL_EVERY="${EVAL_EVERY:-5}"
RHO="${RHO:-0.8}"
KS_TAG="${KS//,/-}"
OUTPUT_DIR="${OUTPUT_DIR:-results/classical_sequences/complex_LB_CE_T${T}_d${D}_ks${KS_TAG}_L${LAYERS}_n${N_SEEDS}}"

echo "=== JOB ${SLURM_JOB_ID:-local} STARTED at $(date) on $(hostname) ==="
echo "Classical + complex: L_B + CE_unif | T=$T d=$D ks=$KS seeds=$N_SEEDS"
echo "OUTPUT_DIR=$OUTPUT_DIR"

module purge
module load openmpi/4.1.6--gcc--12.2.0
module load python/3.11.7
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

test -f run_complex_LB_CE_campaign.py || { echo "ERROR: run_complex_LB_CE_campaign.py missing"; exit 1; }
mkdir -p logs "$OUTPUT_DIR"

srun --mpi=pmix_v3 --mem=0 --export=ALL --cpu-bind=cores "$VENV_PY" run_complex_LB_CE_campaign.py \
  --T "$T" \
  --d "$D" \
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
  --seed "$SEED" \
  --model-seed-base "$MODEL_SEED_BASE" \
  --eval-every "$EVAL_EVERY" \
  --classical-rho "$RHO" \
  --output-dir "$OUTPUT_DIR"

echo "=== JOB FINISHED at $(date) ==="
