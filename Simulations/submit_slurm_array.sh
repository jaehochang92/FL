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
Usage: bash Simulations/submit_slurm_array.sh [options] [run_all.py args]

Options:
  -A, --account ACCOUNT      SLURM account/project to charge
  -p, --partition PARTITION  SLURM partition/queue
  -q, --qos QOS              SLURM QoS
  -t, --time TIME            Walltime (e.g., 24:00:00)
  -m, --mem MEM              Memory per task (e.g., 8G)
  -c, --cpus N               CPUs per task
  -j, --jobs N               Target number of array tasks (shards reps)
  -r, --reps-per-job N       Replicates per task (shards reps)
  -h, --help                 Show this message

run_all.py arguments (passed through):
  --diag                     Use diagonal-only precision in EM (fast mode)
  --scenario SCENARIO        Run only one scenario
  --reps N                   Number of replicates
  --outdir DIR               Output directory
  [any other run_all.py args]

Environment alternatives:
  SLURM_ACCOUNT, SLURM_PARTITION, SLURM_QOS, SLURM_TIME, SLURM_MEM,
  SLURM_CPUS_PER_TASK, RUN_ARGS, PYTHON_BIN, MAX_ARRAY_SIZE

Persistent defaults:
  Create Simulations/slurm_site_defaults.sh to set ACCOUNT/PARTITION/QOS/TIME_LIMIT/MEMORY/CPUS once.

Examples:
  bash submit_slurm_array.sh --account myaccount --diag
  bash submit_slurm_array.sh -A myaccount --scenario poisson --reps 50 --diag
  bash submit_slurm_array.sh -A myaccount --jobs 999 --diag
  bash submit_slurm_array.sh -A myaccount --reps-per-job 4 --diag
  export RUN_ARGS="--diag"; bash submit_slurm_array.sh --account myaccount
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
    -j|--jobs)
      TARGET_JOBS="$2"
      shift 2
      ;;
    -r|--reps-per-job)
      REPS_PER_JOB="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      # Forward unrecognized arguments to run_all.py
      # If arg starts with --, it's a flag; collect it and its value if present
      if [[ "$1" == --* ]]; then
        RUN_ARGS="${RUN_ARGS} $1"
        shift
        # If next arg exists and doesn't start with --, treat as value for the flag
        if [[ $# -gt 0 && ! "$1" =~ ^- ]]; then
          RUN_ARGS="${RUN_ARGS} $1"
          shift
        fi
      else
        # Single-dash short option; forward as-is
        RUN_ARGS="${RUN_ARGS} $1"
        shift
      fi
      ;;
  esac
done

if [[ -n "${TARGET_JOBS:-}" && -n "${REPS_PER_JOB:-}" ]]; then
  echo "Error: use only one of --jobs or --reps-per-job." >&2
  exit 1
fi

CONFIG_LINES=$(
  # shellcheck disable=SC2086
  "$PY" run_all.py --list-configs ${RUN_ARGS} | grep -E '^[[:space:]]*[0-9]+[[:space:]]'
)
COUNT=$(echo "$CONFIG_LINES" | wc -l | tr -d ' ')
if [[ "$COUNT" -le 0 ]]; then
  echo "No configurations found." >&2
  exit 1
fi
MAX_INDEX=$((COUNT - 1))
MAX_ARRAY_SIZE=${MAX_ARRAY_SIZE:-999}

TOTAL_REPS=0
while IFS= read -r line; do
  reps=$(echo "$line" | sed -n 's/.*reps=\([0-9][0-9]*\).*/\1/p')
  if [[ -z "$reps" ]]; then
    echo "Failed to parse reps from: $line" >&2
    exit 1
  fi
  TOTAL_REPS=$((TOTAL_REPS + reps))
done <<< "$CONFIG_LINES"

if [[ -n "${REPS_PER_JOB:-}" ]]; then
  if [[ "$REPS_PER_JOB" -le 0 ]]; then
    echo "--reps-per-job must be positive." >&2
    exit 1
  fi
  SHARD_REPS=$REPS_PER_JOB
elif [[ -n "${TARGET_JOBS:-}" ]]; then
  if [[ "$TARGET_JOBS" -le 0 ]]; then
    echo "--jobs must be positive." >&2
    exit 1
  fi
  SHARD_REPS=$(( (TOTAL_REPS + TARGET_JOBS - 1) / TARGET_JOBS ))
else
  SHARD_REPS=""
fi

echo "Total configurations: ${COUNT}"
echo "Total reps across configs: ${TOTAL_REPS}"
if [[ -n "${SHARD_REPS}" ]]; then
  echo "Replicates per task: ${SHARD_REPS}"
fi
echo "RUN_ARGS='${RUN_ARGS}'"
[[ -n "$ACCOUNT" ]] && echo "ACCOUNT='${ACCOUNT}'"
[[ -n "$PARTITION" ]] && echo "PARTITION='${PARTITION}'"
[[ -n "$QOS" ]] && echo "QOS='${QOS}'"

# Create the logs directory now so SLURM can open output files before the
# job script runs (otherwise SLURM falls back to /var/spool/slurmd/logs and
# fails with "Permission denied").
LOGS_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOGS_DIR"

TASK_DIR="$SCRIPT_DIR/task_manifests"
mkdir -p "$TASK_DIR"
TASK_FILE="$TASK_DIR/task_manifest_$(date +%Y%m%d_%H%M%S)_$$.txt"
if [[ -n "${SHARD_REPS}" ]]; then
  : > "$TASK_FILE"
  while IFS= read -r line; do
    idx=$(echo "$line" | awk '{print $1}')
    reps=$(echo "$line" | sed -n 's/.*reps=\([0-9][0-9]*\).*/\1/p')
    start=0
    while [[ $start -lt $reps ]]; do
      count=$SHARD_REPS
      if [[ $((start + count)) -gt $reps ]]; then
        count=$((reps - start))
      fi
      echo "$idx $start $count" >> "$TASK_FILE"
      start=$((start + count))
    done
  done <<< "$CONFIG_LINES"
else
  : > "$TASK_FILE"
  while IFS= read -r line; do
    idx=$(echo "$line" | awk '{print $1}')
    echo "$idx 0 0" >> "$TASK_FILE"
  done <<< "$CONFIG_LINES"
fi

TASK_COUNT=$(wc -l < "$TASK_FILE" | tr -d ' ')
MAX_INDEX=$((TASK_COUNT - 1))

echo "Total tasks to submit: ${TASK_COUNT}"

START_INDEX=0
while [[ $START_INDEX -le $MAX_INDEX ]]; do
  END_INDEX=$((START_INDEX + MAX_ARRAY_SIZE - 1))
  if [[ $END_INDEX -gt $MAX_INDEX ]]; then
    END_INDEX=$MAX_INDEX
  fi
  ARRAY_COUNT=$((END_INDEX - START_INDEX + 1))
  ARRAY_MAX=$((ARRAY_COUNT - 1))

  echo "Submitting SLURM array for indices ${START_INDEX}-${END_INDEX} (array 0-${ARRAY_MAX})"

  SBATCH_ARGS=(
    "--array=0-${ARRAY_MAX}"
    "--export=ALL,RUN_ARGS=${RUN_ARGS},ARRAY_OFFSET=${START_INDEX},TASK_FILE=${TASK_FILE}"
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

  START_INDEX=$((END_INDEX + 1))
done
