#!/bin/bash
#SBATCH --job-name=final_loss_small
#SBATCH --output=logs/final_loss_small_%j.out
#SBATCH --error=logs/final_loss_small_%j.err
#
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --account=iscrc_qusala
#
# FINAL loss — small training set (80 sentences vs 800) to probe train/test gap.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=72:00:00

set -euo pipefail

T="${T:-16}"
D="${D:-8}"
KS="${KS:-1,2,3,5,6}"
KS="${KS//:/,}"
QSA_LAYERS="${QSA_LAYERS:-16}"
EPOCHS="${EPOCHS:-400}"
POLY_EPOCHS="${POLY_EPOCHS:-600}"
NL_EPOCHS="${NL_EPOCHS:-500}"
NL_EPOCHS_GENERAL="${NL_EPOCHS_GENERAL:-800}"
MAX_SENTENCES="${MAX_SENTENCES:-80}"
N_SEEDS="${N_SEEDS:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
DATA_SEED="${DATA_SEED:-42}"
TEST_DATA_SEED="${TEST_DATA_SEED:-4242}"
TEST_MAX_SENTENCES="${TEST_MAX_SENTENCES:-200}"
MODEL_SEED_BASE="${MODEL_SEED_BASE:-1042}"
LR="${LR:-1e-3}"
NL_LR="${NL_LR:-5e-3}"
NL_LR_GENERAL="${NL_LR_GENERAL:-8e-3}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-20}"
NL_PARAM_BUDGET_SMALL="${NL_PARAM_BUDGET_SMALL:-128}"
APPENDIX_CURVES_KS="${APPENDIX_CURVES_KS:-3,5}"
NO_POLY="${NO_POLY:-0}"
ISOLATE_JOBS="${ISOLATE_JOBS:-1}"
KS_TAG="${KS//,/-}"
OUTPUT_DIR="${OUTPUT_DIR:-results/final_loss/v2_small_T${T}_d${D}_ks${KS_TAG}_L${QSA_LAYERS}_n${N_SEEDS}_n${MAX_SENTENCES}_test}"

echo "=== JOB ${SLURM_JOB_ID:-local} STARTED at $(date) on $(hostname) ==="
echo "FINAL loss SMALL: T=$T d=$D ks=$KS max_sentences=$MAX_SENTENCES n_seeds=$N_SEEDS"
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

mkdir -p logs "$OUTPUT_DIR"
EXTRA=(--isolate-jobs)
if [[ "$NO_POLY" == "1" ]]; then
  EXTRA+=(--no-poly)
fi

srun --mpi=pmix_v3 --mem=0 --export=ALL --cpu-bind=cores "$VENV_PY" run_final_loss.py \
  --T "$T" --d "$D" --ks "$KS" --qsa-layers "$QSA_LAYERS" \
  --epochs "$EPOCHS" --poly-epochs "$POLY_EPOCHS" \
  --nl-epochs "$NL_EPOCHS" --nl-epochs-general "$NL_EPOCHS_GENERAL" \
  --max-sentences "$MAX_SENTENCES" --n-seeds "$N_SEEDS" \
  --test-data-seed "$TEST_DATA_SEED" --test-max-sentences "$TEST_MAX_SENTENCES" \
  --appendix-curves-ks "$APPENDIX_CURVES_KS" \
  --batch-size "$BATCH_SIZE" --data-seed "$DATA_SEED" \
  --model-seed-base "$MODEL_SEED_BASE" \
  --learning-rate "$LR" --nl-learning-rate "$NL_LR" \
  --nl-learning-rate-general "$NL_LR_GENERAL" \
  --nl-param-budget-small "$NL_PARAM_BUDGET_SMALL" \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --output-dir "$OUTPUT_DIR" --resume --mpi "${EXTRA[@]}"

echo "=== JOB FINISHED at $(date) ==="
