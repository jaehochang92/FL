#!/usr/bin/env bash
#SBATCH --job-name=fl-sim
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

mkdir -p "$ROOT_DIR/logs"
cd "$SCRIPT_DIR"

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
