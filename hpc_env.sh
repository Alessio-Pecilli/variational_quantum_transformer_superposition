# Shared Leonardo env bootstrap. Sourced by hpc*.sh (not executed directly).
# Expects: modules already loaded (openmpi + python).

VENV_DIR="${VENV_DIR:-/leonardo_work/IscrC_QuSALa/venv_py311}"
VENV_PY="${VENV_PY:-$VENV_DIR/bin/python3}"

# Prefer submit dir (clone you sbatch from); fall back to script location.
_REPO_DIR="${SLURM_SUBMIT_DIR:-}"
if [[ -z "$_REPO_DIR" || ! -f "$_REPO_DIR/requirements.txt" ]]; then
  _REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
REQ_FILE="${REQ_FILE:-$_REPO_DIR/requirements.txt}"

if [[ ! -x "$VENV_PY" ]]; then
  echo "ERROR: venv python missing: $VENV_PY"
  echo "Run on login node: bash hpc_setup_env.sh"
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

_need_install=0
if ! "$VENV_PY" -c "import jax, pennylane, mpi4py, numpy, matplotlib" >/dev/null 2>&1; then
  _need_install=1
fi

if [[ "${FORCE_PIP:-0}" == "1" || "$_need_install" == "1" ]]; then
  if [[ ! -f "$REQ_FILE" ]]; then
    echo "ERROR: missing deps and no requirements at $REQ_FILE"
    echo "Run on login node: bash hpc_setup_env.sh"
    exit 1
  fi
  echo "=== pip install -r $REQ_FILE (into $VENV_DIR) ==="
  "$VENV_PY" -m pip install -U pip setuptools wheel
  "$VENV_PY" -m pip install -r "$REQ_FILE"
fi

"$VENV_PY" -c "import jax, pennylane, mpi4py; print('env ok: jax', jax.__version__, 'pennylane', pennylane.__version__, 'python', __import__('sys').executable)" || {
  echo "ERROR: core imports still failing after pip."
  echo "On login node run: bash hpc_setup_env.sh"
  exit 1
}

cd "${SLURM_SUBMIT_DIR:-$_REPO_DIR}" || exit 1
echo "workdir=$(pwd) python=$VENV_PY"
