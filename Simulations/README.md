# Simulations

Simulation code for the revised federated-learning experiments in the paper.

## Overview

The simulation framework is organized around a shared `Scenario` base class with three concrete scenarios:

- `quadratic`: Gaussian sample means with diagonal variance $\Sigma(\theta) = {\rm diag}(\theta \odot \theta)$
- `logistic`: multiclass logistic regression with $C = 6$ classes
- `poisson`: Poisson regression fit by IRLS

All scenarios use a fixed prior in $\mathbb{R}^3$ supported on five curves and report client-level RMSE for four estimators:

- `vaneb`: the proposed variance-adaptive NPEB estimator
- `npeb`: the Soloff et al. homoscedastic NPMLE baseline
- `adamix`: Gaussian-mixture empirical Bayes baseline
- `oracle`: Bayes posterior mean under the true prior and variance model

## Sweep Design

For each scenario, `run_all.py` executes two sweeps (paper-aligned):

- $n_{\min} \in \{50, 100, 200, 400, 800\}$ with $K = 200$ fixed
- $K \in \{50, 100, 200, 400, 800\}$ with $n_{\min} = 50$ fixed

The default run uses 100 replicates per configuration and sets client sizes as
$n_k \sim \mathrm{Unif}(n_{\min}, 2 n_{\min})$.

**Covariance modes:**

- **Full (default):** VANEB recomputes atom covariances using full Fisher matrices each EM iteration; baselines use fixed client-provided covariances. Most accurate but slower for large K.
- **Diagonal (`--diag` flag):** VANEB initializes with full Fisher but uses diagonal-only precision in 25 EM iterations, skipping expensive matrix operations. ~10-15% faster with small accuracy trade-off. Recommended for K ≥ 500.

Choice is controlled by the `--diag` CLI flag (see examples below).

## Quick Start

```bash
pip install -r Simulations/requirements.txt

# Full sweep for all three scenarios (full covariance mode)
python Simulations/run_all.py

# Full sweep with diagonal-only precision (faster, ~10-15% speedup)
python Simulations/run_all.py --diag

# Quick validation run
python Simulations/run_all.py --smoke

# Run one scenario only
python Simulations/run_all.py --scenario logistic

# Fast mode for specific config
python Simulations/run_all.py --config-index 5 --diag --reps 10

# List indexed configurations (useful for SLURM arrays)
python Simulations/run_all.py --list-configs

# Run exactly one indexed configuration
python Simulations/run_all.py --config-index 5

# Generate the six manuscript figures from completed outputs
python Simulations/make_figures.py

# Generate a 3D illustration of prior atoms
python Simulations/plot_prior_atoms_3d.py
```

## Performance Optimization: `--diag` Flag

For faster iteration on large-scale experiments, use the `--diag` flag to enable diagonal-only precision in EM updates:

### When to Use `--diag`

- **Large K:** K ≥ 500 where the 25 EM iterations dominate runtime
- **Rapid prototyping:** Trading small accuracy loss (~0.01–0.1 RMSE) for 10–15% speed
- **HPC sweeps:** Reducing per-job wall time to fit cluster allocations

### How It Works

1. **Initialization (unchanged):** VANEB uses full Fisher covariance → GLMixture baseline (preserves accuracy)
2. **EM updates (optimized):** 25 iterations use diagonal-only precision matrices:
   - Mahalanobis: Element-wise multiplication instead of einsum
   - Log-determinant: Sum-of-logs instead of eigenvalue decomposition
   - Atom updates: Element-wise division instead of matrix solve
3. **Trade-off:** ~10% faster, negligible accuracy impact for well-specified models

### Performance Measured

| Scenario | K | nmin | Full | Diagonal | Speedup |
|----------|---|------|------|----------|---------|
| poisson  | 200 | 40   | 5.6s | 5.1s     | 9% |
| poisson  | 400 | 40   | 11.9s| 10.4s    | 12% |
| logistic | 200 | 40   | ~7s  | ~6.5s    | ~7% |

### Configuration

All flags default to `use_diag=False` (full mode). Set explicitly or inherit from CLI:

```bash
# Full mode (baseline)
python run_all.py --scenario poisson --reps 100

# Diagonal-only mode (fast)
python run_all.py --scenario poisson --reps 100 --diag

# Mix: full mode for K sweep, diagonal for n_min sweep
python run_all.py --scenario poisson --config-index 0-4        # full (n_min sweep)
python run_all.py --scenario poisson --config-index 5-10 --diag # diag (K sweep)
```

The chosen mode is logged in `config.json` as `"use_diag": true/false`.

## Conda Environment (HPC)

Create the environment from YAML (recommended on clusters):

```bash
conda env create -f Simulations/environment.yml
conda activate fl-sim
```

Update an existing environment after dependency changes:

```bash
conda env update -f Simulations/environment.yml --prune
```

Batch-safe activation inside SLURM scripts:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fl-sim
```

Outputs are written under `Simulations/outputs/` when launched from the repository root, and under `outputs/` relative to `run_all.py` if launched from inside the `Simulations` directory.

## File Structure

| File | Purpose |
|------|---------|
| `scenario_base.py` | Shared configuration, prior geometry, estimators (VANEB, NPEB, AdaMix, Oracle), and scenario base class. Includes `mahal_diag()`, `logdet_diag()` for fast diagonal operations, and `_parallel_fit_clients()` for parallel client fitting. |
| `scenario_quadratic.py` | Quadratic sample-means scenario |
| `scenario_logistic.py` | Multiclass logistic scenario. Implements `batch_observed_fisher_diag()` for fast diagonal-only Fisher computation. |
| `scenario_poisson.py` | Poisson regression scenario. Implements `batch_poisson_fisher_diag()` for fast diagonal-only Fisher computation. |
| `run_all.py` | Unified sweep runner for all scenarios. Supports `--diag` flag to enable diagonal-only EM updates. |
| `make_figures.py` | Generate the six manuscript figures from completed outputs |
| `plot_prior_atoms_3d.py` | Create a 3D illustration of the five-curve prior atoms |
| `slurm_array_job.sh` | One SLURM array task = one indexed simulation config |
| `submit_slurm_array.sh` | Helper to submit the full config list as a SLURM array |
| `environment.yml` | Conda environment definition for local/HPC reproducibility |
| `requirements.txt` | Python dependencies |

## SLURM HPC Usage

Run from repository root:

```bash
# Submit all configs with diagonal-only precision (direct flags)
bash Simulations/submit_slurm_array.sh --account <your_account> --diag

# Submit with scenario and replicate options
bash Simulations/submit_slurm_array.sh --account <your_account> --scenario poisson --reps 50 --diag

# Alternative: use RUN_ARGS environment variable (legacy approach still supported)
export RUN_ARGS="--scenario poisson --reps 50 --diag"
bash Simulations/submit_slurm_array.sh --account <your_account>

# Or set SLURM_ACCOUNT as a persistent default
export SLURM_ACCOUNT=<your_account>
bash Simulations/submit_slurm_array.sh --diag
```

**Notes:**

- **Direct flag arguments:** `submit_slurm_array.sh` now forwards unrecognized `--flags` to `run_all.py` automatically. Use `--diag`, `--scenario`, `--reps`, `--outdir`, etc. directly on the command line.
- **Legacy RUN_ARGS:** Still supported for backward compatibility; `submit_slurm_array.sh` merges both CLI args and `RUN_ARGS`.
- **SLURM-specific options:** `--account`, `--partition`, `--qos`, `--time`, `--mem`, `--cpus` are reserved for SLURM configuration.
- **Array indexing:** Zero-based (`SLURM_ARRAY_TASK_ID` maps to `--config-index`).
- **Config discovery:** The script uses `python Simulations/run_all.py --list-configs` to determine array size.
- **Manual task execution:** `python Simulations/run_all.py --config-index <idx> --no-progress --diag`.
- **Python interpreter:** Set `PYTHON_BIN` if your cluster does not use `python3`.
- **Persistent defaults:** `submit_slurm_array.sh` auto-loads `Simulations/slurm_site_defaults.sh` (e.g., to set `ACCOUNT`, `PARTITION`, `TIME_LIMIT`).
- **Diagonal speedup:** Use `--diag` for 10–15% faster runs when K ≥ 500 (negligible accuracy trade-off).
- **Job limits:** If the config count exceeds 999, `submit_slurm_array.sh` automatically splits into multiple arrays and uses `ARRAY_OFFSET` to map indices. Override with `MAX_ARRAY_SIZE` if your cluster allows a different limit.

## Requirements

- Python 3.9+
- `npeb` package with MOSEK support for the NPMLE-based estimators
- Packages listed in `requirements.txt`
