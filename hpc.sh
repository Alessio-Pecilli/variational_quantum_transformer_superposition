#!/bin/bash
#SBATCH --job-name=vqt_train
#SBATCH --output=logs/vqt_%j.out
#SBATCH --error=logs/vqt_%j.err

#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --account=iscrc_qusala

#SBATCH --nodes=5
#SBATCH --ntasks-per-node=20     # 5 × 20 = 100 rank (ben sotto il limite 256)
#SBATCH --cpus-per-task=1
#SBATCH --mem=0                  # usa TUTTA la RAM del nodo BOOST
#SBATCH --time=05:30:00

# Legacy / multi-rank VQT entrypoint (main_hpc.py).
# For Section-2 classical campaign on Leonardo prefer ONE job:
#   sbatch hpc_all_section2.sh   # complexity → baselines → ppl-vs-k (MPI)
# Or standalone:
#   sbatch hpc_complexity.sh / hpc_baselines.sh / hpc_ppl_vs_k.sh
#   sbatch hpc_qsa_eval.sh       # quantum forward on lightning.gpu

echo "=== JOB $SLURM_JOB_ID STARTED at $(date) on $(hostname) ==="

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
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Forza UCX a usare i trasporti corretti (InfiniBand e Shared Memory standard)
export UCX_TLS=self,shm,rc,ud
# Aumenta i timeout per dare tempo ai rank di scambiarsi gli indirizzi
export UCX_RECONNECT_WAIT=15s
export UCX_CONNECT_TIMEOUT=300s
# Risolve spesso il problema "Shared memory error" su architetture NVIDIA/Atos
export UCX_MEMTYPE_CACHE=n

test -f main_hpc.py || { echo "ERROR: main_hpc.py not found in $(pwd)"; exit 1; }
mkdir -p logs

srun --mpi=pmix_v3 --export=ALL "$VENV_PY" main_hpc.py
