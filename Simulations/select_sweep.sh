#!/usr/bin/env bash
#
# select_sweep.sh — Interactive selector for nmin vs K SLURM sweeps
#
# Thin wrapper that prompts for sweep type & scenario, then delegates all
# argument handling to submit_slurm_array.sh. Supports all SLURM and run_all.py
# options.
#
# Usage:
#   bash select_sweep.sh                                   # interactive prompts
#   bash select_sweep.sh --account myaccount --diag        # preset args + prompts
#   bash select_sweep.sh -A myaccount --jobs 900 --reps 200
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Collect all CLI args to pass through to submit_slurm_array.sh
SUBMIT_ARGS=("$@")

# =============================================================================
# Menu: Select sweep type
# =============================================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "SELECT SWEEP TYPE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1) nmin-sweep with fixed K"
echo "2) K-sweep with fixed nmin"
echo ""

read -p "Enter choice (1 or 2): " sweep_choice

case "$sweep_choice" in
  1)
    echo "✓ Selected: nmin-sweep"
    SUBMIT_ARGS+=("--sweep-type" "nmin")
    ;;
  2)
    echo "✓ Selected: K-sweep"
    SUBMIT_ARGS+=("--sweep-type" "K")
    ;;
  *)
    echo "Error: Invalid choice '$sweep_choice'. Please enter 1 or 2." >&2
    exit 1
    ;;
esac

# =============================================================================
# Menu: Select scenario
# =============================================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "SELECT SCENARIO(S)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1) All scenarios (quadratic, logistic, poisson)"
echo "2) Quadratic only"
echo "3) Logistic only"
echo "4) Poisson only"
echo ""

read -p "Enter choice (1-4): " scenario_choice

case "$scenario_choice" in
  1)
    echo "✓ Selected: All scenarios"
    ;;
  2)
    echo "✓ Selected: Quadratic only"
    SUBMIT_ARGS+=("--scenario" "quadratic")
    ;;
  3)
    echo "✓ Selected: Logistic only"
    SUBMIT_ARGS+=("--scenario" "logistic")
    ;;
  4)
    echo "✓ Selected: Poisson only"
    SUBMIT_ARGS+=("--scenario" "poisson")
    ;;
  *)
    echo "Error: Invalid choice '$scenario_choice'. Please enter 1-4." >&2
    exit 1
    ;;
esac

# =============================================================================
# Delegate to submit_slurm_array.sh
# =============================================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "SUBMITTING TO SLURM"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

bash "$SCRIPT_DIR/submit_slurm_array.sh" "${SUBMIT_ARGS[@]}"
