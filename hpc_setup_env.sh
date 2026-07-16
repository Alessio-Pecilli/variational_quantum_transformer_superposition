#!/bin/bash
# Run ONCE on a Leonardo LOGIN node (not inside a batch job):
#   bash hpc_setup_env.sh
#
# Creates/updates the shared project venv from requirements.txt.

set -euo pipefail

VENV_DIR="${VENV_DIR:-/leonardo_work/IscrC_QuSALa/venv_py311}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ="${REQ:-$REPO_DIR/requirements.txt}"

module purge
module load openmpi/4.1.6--gcc--12.2.0
module load python/3.11.7

if [[ ! -x "$VENV_DIR/bin/python3" ]]; then
  echo "Creating venv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

PY="$VENV_DIR/bin/python3"
echo "Using $PY"
echo "Installing from $REQ"

"$PY" -m pip install -U pip setuptools wheel
"$PY" -m pip install -r "$REQ"

echo "=== verify ==="
"$PY" -c "import jax, jaxlib, pennylane, mpi4py, numpy, matplotlib; print('jax', jax.__version__); print('pennylane', pennylane.__version__); print('mpi4py', mpi4py.__version__); print('OK')"
echo "Done. Re-submit: sbatch hpc_all_section2.sh"
