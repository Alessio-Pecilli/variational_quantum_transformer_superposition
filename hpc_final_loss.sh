#!/bin/bash
#SBATCH --job-name=final_loss
#SBATCH --output=logs/final_loss_%j.out
#SBATCH --error=logs/final_loss_%j.err
#
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --account=iscrc_qusala
#
# FINAL loss campaign (multi-seed, multi-k, mono+poly on same plot).
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=48:00:00

# Models (same plot):
#   k-QSA L=16, k-CSA (128 mats), poly-k-QSA L=16, poly-k-CSA,
#   nl-CSA iso ~288, nl-CSA iso ~128, nl-CSA gen ~128
# Mu-models trained at each k in KS (default 1,2,3,5,6); nl once (k-indep.).
# Poly kernel included by default (hoped to flatten loss growth with k).
#
# Examples:
#   sbatch hpc_final_loss.sh
#   sbatch --export=ALL,KS=1,2,3,5,6,N_SEEDS=8 hpc_final_loss.sh
#   sbatch --export=ALL,NO_POLY=1 hpc_final_loss.sh   # mono only

set -euo pipefail

T="${T:-16}"
D="${D:-8}"
KS="${KS:-1,2,3,5,6}"
QSA_LAYERS="${QSA_LAYERS:-16}"
EPOCHS="${EPOCHS:-400}"
POLY_EPOCHS="${POLY_EPOCHS:-600}"
NL_EPOCHS="${NL_EPOCHS:-500}"
NL_EPOCHS_GENERAL="${NL_EPOCHS_GENERAL:-800}"
MAX_SENTENCES="${MAX_SENTENCES:-1000}"
N_SEEDS="${N_SEEDS:-8}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DATA_SEED="${DATA_SEED:-42}"
MODEL_SEED_BASE="${MODEL_SEED_BASE:-1042}"
LR="${LR:-1e-3}"
NL_LR="${NL_LR:-5e-3}"
NL_LR_GENERAL="${NL_LR_GENERAL:-8e-3}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-20}"
NL_PARAM_BUDGET_SMALL="${NL_PARAM_BUDGET_SMALL:-128}"
NO_POLY="${NO_POLY:-0}"
KS_TAG="${KS//,/-}"
OUTPUT_DIR="${OUTPUT_DIR:-results/final_loss/definitive_T${T}_d${D}_ks${KS_TAG}_L${QSA_LAYERS}_n${N_SEEDS}}"

echo "=== JOB ${SLURM_JOB_ID:-local} STARTED at $(date) on $(hostname) ==="
echo "FINAL loss: T=$T d=$D ks=$KS QSA_L=$QSA_LAYERS n_seeds=$N_SEEDS no_poly=$NO_POLY"
echo "epochs mono=$EPOCHS poly=$POLY_EPOCHS nl=$NL_EPOCHS nl_gen=$NL_EPOCHS_GENERAL"
echo "OUTPUT_DIR=$OUTPUT_DIR (re-sbatch same dir to resume)"
echo "MPI tasks=${SLURM_NTASKS:-1}"

module purge
module load openmpi/4.1.6--gcc--12.2.0
module load python/3.11.7
module load cuda/12.1 2>/dev/null || true

# shellcheck source=hpc_env.sh
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}/hpc_env.sh"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUTF8=1
export JAX_ENABLE_X64=True
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

test -f run_final_loss.py || { echo "ERROR: run_final_loss.py not found in $(pwd)"; exit 1; }
mkdir -p logs "$OUTPUT_DIR"

EXTRA=()
if [[ "$NO_POLY" == "1" ]]; then
  EXTRA+=(--no-poly)
fi

srun --mpi=pmix_v3 --export=ALL "$VENV_PY" run_final_loss.py \
  --T "$T" \
  --d "$D" \
  --ks "$KS" \
  --qsa-layers "$QSA_LAYERS" \
  --epochs "$EPOCHS" \
  --poly-epochs "$POLY_EPOCHS" \
  --nl-epochs "$NL_EPOCHS" \
  --nl-epochs-general "$NL_EPOCHS_GENERAL" \
  --max-sentences "$MAX_SENTENCES" \
  --n-seeds "$N_SEEDS" \
  --batch-size "$BATCH_SIZE" \
  --data-seed "$DATA_SEED" \
  --model-seed-base "$MODEL_SEED_BASE" \
  --learning-rate "$LR" \
  --nl-learning-rate "$NL_LR" \
  --nl-learning-rate-general "$NL_LR_GENERAL" \
  --nl-param-budget-small "$NL_PARAM_BUDGET_SMALL" \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --output-dir "$OUTPUT_DIR" \
  --resume \
  --mpi \
  "${EXTRA[@]}"

echo "=== JOB FINISHED at $(date) ==="
echo "Replot aesthetics:  $VENV_PY run_final_loss.py --replot-only $OUTPUT_DIR"
