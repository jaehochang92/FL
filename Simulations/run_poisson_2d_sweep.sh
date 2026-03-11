#!/bin/bash
# Run Poisson regression 2D simulations for the paper
# Generates actual count data and fits Poisson GLM via IRLS to get MLE + empirical Fisher
# Two-sweep design:
#   1. K-scaling: K ∈ {100, 400, 1600} at nmin=100
#   2. n-scaling: nmin ∈ {100, 400, 1600} at K=100
# Total: 5 unique configs (nmin=100,K=100 shared)

set -e
cd "$(dirname "$0")/.."

REPS=50
SEED=20260306

# K-scaling sweep (fixed nmin=100)
for K in 100 400 1600; do
    OUTDIR="Simulations/outputs/poisson_2d_nmin100_k${K}"
    echo "=== Running nmin=100, K=${K}, ${REPS} reps ==="
    python Simulations/poisson_regression_sim.py \
        --k-values ${K} \
        --reps ${REPS} \
        --dim 2 \
        --n-min 100 \
        --n-max 400 \
        --feature-scale 0.7 \
        --em-iters 25 \
        --seed ${SEED} \
        --outdir "${OUTDIR}" \
        --no-progress
    echo "  -> Done, see ${OUTDIR}/metrics.csv"
done

# n-scaling sweep (fixed K=100, nmin=100 already done above)
for NMIN in 400 1600; do
    NMAX=$((NMIN * 4))
    OUTDIR="Simulations/outputs/poisson_2d_nmin${NMIN}_k100"
    echo "=== Running nmin=${NMIN}, K=100, ${REPS} reps ==="
    python Simulations/poisson_regression_sim.py \
        --k-values 100 \
        --reps ${REPS} \
        --dim 2 \
        --n-min ${NMIN} \
        --n-max ${NMAX} \
        --feature-scale 0.7 \
        --em-iters 25 \
        --seed ${SEED} \
        --outdir "${OUTDIR}" \
        --no-progress
    echo "  -> Done, see ${OUTDIR}/metrics.csv"
done

echo ""
echo "Poisson regression 2D simulations complete!"
echo "To generate figures, run: python Simulations/analyze_poisson_grid.py"
