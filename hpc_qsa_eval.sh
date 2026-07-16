#!/bin/bash
#SBATCH --job-name=qsa_eval
#SBATCH --output=logs/qsa_eval_%j.out
#SBATCH --error=logs/qsa_eval_%j.err
#
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --account=iscrc_qusala
#
# Quantum forward eval of Section-2 circuit (lightning.gpu).
# Prefer 1 GPU up to ~30 qubits; see: python qsa_section2_circuit.py --leonardo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=02:00:00

# Examples:
#   sbatch --export=ALL,T=64,D=8,K=2 hpc_qsa_eval.sh
#   sbatch --export=ALL,T=128,D=16,K=3,DTYPE=complex64 hpc_qsa_eval.sh

set -euo pipefail

T="${T:-128}"
D="${D:-16}"
K="${K:-3}"
LAYERS="${LAYERS:-2}"
DEVICE="${DEVICE:-lightning.gpu}"
DTYPE="${DTYPE:-complex128}"
SEED="${SEED:-0}"

echo "=== JOB ${SLURM_JOB_ID:-local} STARTED at $(date) on $(hostname) ==="
echo "qsa_eval: T=$T d=$D k=$K device=$DEVICE dtype=$DTYPE"

module purge
module load openmpi/4.1.6--gcc--12.2.0
module load python/3.11.7
module load cuda/12.1 2>/dev/null || true

# shellcheck source=hpc_env.sh
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}/hpc_env.sh"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUTF8=1
export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false

test -f qsa_run.py || { echo "ERROR: qsa_run.py not found in $(pwd)"; exit 1; }
mkdir -p logs

# Single-GPU overlap readout (no MPI gather needed).
srun --mpi=pmix_v3 --export=ALL "$VENV_PY" qsa_run.py \
  --T "$T" \
  --d "$D" \
  --k "$K" \
  --layers "$LAYERS" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --seed "$SEED"

echo "=== JOB FINISHED at $(date) ==="
