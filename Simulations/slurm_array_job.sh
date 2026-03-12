#!/usr/bin/env bash
#SBATCH --job-name=fl-sim
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err

set -euo pipefail

# In some SLURM setups the batch script is executed from a spool copy under
# /var/spool/slurmd, so BASH_SOURCE-based paths are not stable. Prefer the
# original submit directory when available.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  SUBMIT_DIR="$SLURM_SUBMIT_DIR"
else
  SUBMIT_DIR="$(pwd)"
fi

if [[ -f "$SUBMIT_DIR/run_all.py" ]]; then
  # Submitted from Simulations/
  SIM_DIR="$SUBMIT_DIR"
  ROOT_DIR="$(cd "$SUBMIT_DIR/.." && pwd)"
elif [[ -f "$SUBMIT_DIR/Simulations/run_all.py" ]]; then
  # Submitted from repository root
  SIM_DIR="$SUBMIT_DIR/Simulations"
  ROOT_DIR="$SUBMIT_DIR"
else
  echo "Could not locate Simulations/run_all.py from SUBMIT_DIR='$SUBMIT_DIR'" >&2
  exit 1
fi

mkdir -p "$SIM_DIR/logs"
cd "$SIM_DIR"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "SLURM_ARRAY_TASK_ID is not set. Submit as a job array." >&2
  exit 1
fi

# Optional venv activation if present.
if [[ -f "$ROOT_DIR/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.venv/bin/activate"
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY="$PYTHON_BIN"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PY="$ROOT_DIR/.venv/bin/python"
else
  PY="python3"
fi

# RUN_ARGS can be exported at submit time, e.g.
# RUN_ARGS="--scenario poisson --reps 30"
# shellcheck disable=SC2086
"$PY" run_all.py \
  --config-index "$SLURM_ARRAY_TASK_ID" \
  --no-progress \
  --outdir "${OUTPUT_DIR:-outputs}" \
  ${RUN_ARGS:-}
