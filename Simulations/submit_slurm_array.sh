#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY="$PYTHON_BIN"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PY="$ROOT_DIR/.venv/bin/python"
else
  PY="python3"
fi

RUN_ARGS="${RUN_ARGS:-}"

# shellcheck disable=SC2086
COUNT=$("$PY" run_all.py --list-configs ${RUN_ARGS} | wc -l | tr -d ' ')
if [[ "$COUNT" -le 0 ]]; then
  echo "No configurations found." >&2
  exit 1
fi
MAX_INDEX=$((COUNT - 1))

echo "Submitting SLURM array with indices 0-${MAX_INDEX}"
echo "RUN_ARGS='${RUN_ARGS}'"

sbatch \
  --array="0-${MAX_INDEX}" \
  --export="ALL,RUN_ARGS=${RUN_ARGS}" \
  "$SCRIPT_DIR/slurm_array_job.sh"
