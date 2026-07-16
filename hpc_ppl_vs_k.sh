#!/bin/bash
#SBATCH --job-name=qsa_ppl_k
#SBATCH --output=logs/ppl_k_%j.out
#SBATCH --error=logs/ppl_k_%j.err
#
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --account=iscrc_qusala
#
# PPL / training curves vs k=1..4 for k-QSA and k-CSA (ablation: vary k only).
# Fixed per-k OUTPUT_DIR: re-sbatch resumes completed k / mid-run checkpoints.
# MPI: shards seeds across tasks for each k.
# Prefer hpc_all_section2.sh for the full campaign.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=12:00:00

set -euo pipefail

T="${T:-16}"
D="${D:-16}"
EPOCHS="${EPOCHS:-300}"
MAX_SENTENCES="${MAX_SENTENCES:-1000}"
N_SEEDS="${N_SEEDS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SEED="${SEED:-42}"
KS="${KS:-1 2 3 4}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-20}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/baselines_smoke/ppl_vs_k_T${T}_d${D}_ep${EPOCHS}_n${N_SEEDS}}"

echo "=== JOB ${SLURM_JOB_ID:-local} STARTED at $(date) on $(hostname) ==="
echo "ppl vs k: T=$T d=$D ks=($KS) epochs=$EPOCHS frasi=$MAX_SENTENCES n_seeds=$N_SEEDS"
echo "OUTPUT_ROOT=$OUTPUT_ROOT (re-sbatch same root to resume)"
echo "MPI tasks=${SLURM_NTASKS:-1}"

module purge
module load openmpi/4.1.6--gcc--12.2.0
module load python/3.11.7
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
mkdir -p logs "$OUTPUT_ROOT"

for K in $KS; do
  OUT="$OUTPUT_ROOT/k${K}"
  echo "----- k=$K -> $OUT -----"
  srun --mpi=pmix_v3 --export=ALL "$VENV_PY" run_baselines_smoke.py \
    --T "$T" \
    --d "$D" \
    --k "$K" \
    --epochs "$EPOCHS" \
    --max-sentences "$MAX_SENTENCES" \
    --n-seeds "$N_SEEDS" \
    --batch-size "$BATCH_SIZE" \
    --seed "$SEED" \
    --checkpoint-every "$CHECKPOINT_EVERY" \
    --output-dir "$OUT" \
    --resume \
    --mpi
done

echo "=== JOB FINISHED at $(date) ==="
