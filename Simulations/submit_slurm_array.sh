#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

# Optional per-site defaults (not committed by default).
# Example location: Simulations/slurm_site_defaults.sh
if [[ -f "$SCRIPT_DIR/slurm_site_defaults.sh" ]]; then
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/slurm_site_defaults.sh"
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY="$PYTHON_BIN"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PY="$ROOT_DIR/.venv/bin/python"
else
  PY="python3"
fi

RUN_ARGS="${RUN_ARGS:-}"

ACCOUNT="${SLURM_ACCOUNT:-${ACCOUNT:-}}"
PARTITION="${SLURM_PARTITION:-${PARTITION:-}}"
QOS="${SLURM_QOS:-${QOS:-}}"
TIME_LIMIT="${SLURM_TIME:-${TIME_LIMIT:-}}"
MEMORY="${SLURM_MEM:-${MEMORY:-}}"
CPUS="${SLURM_CPUS_PER_TASK:-${CPUS:-}}"

usage() {
  cat <<'EOF'
Usage: bash Simulations/submit_slurm_array.sh [options]

Options:
  -A, --account ACCOUNT      SLURM account/project to charge
  -p, --partition PARTITION  SLURM partition/queue
  -q, --qos QOS              SLURM QoS
  -t, --time TIME            Walltime (e.g., 24:00:00)
  -m, --mem MEM              Memory per task (e.g., 8G)
  -c, --cpus N               CPUs per task
  -h, --help                 Show this message

Environment alternatives:
  SLURM_ACCOUNT, SLURM_PARTITION, SLURM_QOS, SLURM_TIME, SLURM_MEM,
  SLURM_CPUS_PER_TASK, RUN_ARGS, PYTHON_BIN

Persistent defaults:
  Create Simulations/slurm_site_defaults.sh to set ACCOUNT/PARTITION/QOS/TIME_LIMIT/MEMORY/CPUS once.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -A|--account)
      ACCOUNT="$2"
      shift 2
      ;;
    -p|--partition)
      PARTITION="$2"
      shift 2
      ;;
    -q|--qos)
      QOS="$2"
      shift 2
      ;;
    -t|--time)
      TIME_LIMIT="$2"
      shift 2
      ;;
    -m|--mem)
      MEMORY="$2"
      shift 2
      ;;
    -c|--cpus)
      CPUS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

# shellcheck disable=SC2086
COUNT=$("$PY" run_all.py --list-configs ${RUN_ARGS} | wc -l | tr -d ' ')
if [[ "$COUNT" -le 0 ]]; then
  echo "No configurations found." >&2
  exit 1
fi
MAX_INDEX=$((COUNT - 1))

echo "Submitting SLURM array with indices 0-${MAX_INDEX}"
echo "RUN_ARGS='${RUN_ARGS}'"
[[ -n "$ACCOUNT" ]] && echo "ACCOUNT='${ACCOUNT}'"
[[ -n "$PARTITION" ]] && echo "PARTITION='${PARTITION}'"
[[ -n "$QOS" ]] && echo "QOS='${QOS}'"

# Create the logs directory now so SLURM can open output files before the
# job script runs (otherwise SLURM falls back to /var/spool/slurmd/logs and
# fails with "Permission denied").
LOGS_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOGS_DIR"

SBATCH_ARGS=(
  "--array=0-${MAX_INDEX}"
  "--export=ALL,RUN_ARGS=${RUN_ARGS}"
  "--output=${LOGS_DIR}/slurm_%A_%a.out"
  "--error=${LOGS_DIR}/slurm_%A_%a.err"
)

[[ -n "$ACCOUNT" ]] && SBATCH_ARGS+=("--account=${ACCOUNT}")
[[ -n "$PARTITION" ]] && SBATCH_ARGS+=("--partition=${PARTITION}")
[[ -n "$QOS" ]] && SBATCH_ARGS+=("--qos=${QOS}")
[[ -n "$TIME_LIMIT" ]] && SBATCH_ARGS+=("--time=${TIME_LIMIT}")
[[ -n "$MEMORY" ]] && SBATCH_ARGS+=("--mem=${MEMORY}")
[[ -n "$CPUS" ]] && SBATCH_ARGS+=("--cpus-per-task=${CPUS}")

set +e
SUBMIT_OUTPUT=$(sbatch "${SBATCH_ARGS[@]}" "$SCRIPT_DIR/slurm_array_job.sh" 2>&1)
SUBMIT_EXIT=$?
set -e

if [[ $SUBMIT_EXIT -ne 0 ]]; then
  echo "$SUBMIT_OUTPUT" >&2
  if [[ "$SUBMIT_OUTPUT" == *"Must specify account for job"* ]]; then
    echo "Hint: your cluster requires an account. Re-run with:" >&2
    echo "  bash Simulations/submit_slurm_array.sh --account <your_account>" >&2
    echo "or set env var: SLURM_ACCOUNT=<your_account>" >&2
  fi
  exit $SUBMIT_EXIT
fi

echo "$SUBMIT_OUTPUT"
