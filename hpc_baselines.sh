#!/bin/bash
#SBATCH --job-name=qsa_baselines
#SBATCH --output=logs/baselines_%j.out
#SBATCH --error=logs/baselines_%j.err
#
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --account=iscrc_qusala
#
# Classical training curves (k-QSA / k-CSA / nl-CSA): no circuit needed.
# Prefer hpc_all_section2.sh for full MPI campaign; this is the standalone job.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=08:00:00

# Best-practice loss plots: mean ± std over seeds + minibatch updates.
# Fixed OUTPUT_DIR so walltime/crash + re-sbatch resumes (skip done seeds / mid-epoch ckpt).
# MPI: shards seeds across SLURM tasks (default 5 tasks ≈ 5 seeds).
# Override at submit time, e.g.:
#   sbatch --export=ALL,T=8,D=8,K=2,EPOCHS=200,MAX_SENTENCES=1000,N_SEEDS=5 hpc_baselines.sh

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
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-20}"
OUTPUT_DIR="${OUTPUT_DIR:-results/baselines_smoke/definitive_T${T}_d${D}_k${K}_ep${EPOCHS}_n${N_SEEDS}}"

echo "=== JOB ${SLURM_JOB_ID:-local} STARTED at $(date) on $(hostname) ==="
echo "baselines: T=$T d=$D k=$K epochs=$EPOCHS frasi=$MAX_SENTENCES n_seeds=$N_SEEDS batch=$BATCH_SIZE"
echo "OUTPUT_DIR=$OUTPUT_DIR (re-sbatch same dir to resume)"
echo "MPI tasks=${SLURM_NTASKS:-1}"

module purge
module load openmpi/4.1.6--gcc--12.2.0
module load python/3.11.7
# GPU modules if lightning/jax-cuda present in venv (harmless if unused)
module load cuda/12.1 2>/dev/null || true

VENV_PY="/leonardo_work/IscrC_QuSALa/venv_py311/bin/python3"
source /leonardo_work/IscrC_QuSALa/venv_py311/bin/activate
test -x "$VENV_PY" || { echo "ERROR: missing $VENV_PY"; exit 1; }

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUTF8=1
export JAX_ENABLE_X64=True
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
echo "workdir=$(pwd) python=$VENV_PY"
test -f run_baselines_smoke.py || { echo "ERROR: run_baselines_smoke.py not found in $(pwd)"; exit 1; }
mkdir -p logs "$OUTPUT_DIR"

srun --mpi=pmix_v3 --export=ALL "$VENV_PY" run_baselines_smoke.py \
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
  --output-dir "$OUTPUT_DIR" \
  --resume \
  --mpi

echo "=== JOB FINISHED at $(date) ==="
