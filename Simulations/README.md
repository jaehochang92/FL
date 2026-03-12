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

For each scenario, `run_all.py` executes two sweeps:

- $n_{\min} \in \{50, 200, 800\}$ with $K = 800$ fixed
- $K \in \{50, 200, 800\}$ with $n_{\min} = 800$ fixed

The default run uses 50 replicates per configuration.

## Quick Start

```bash
pip install -r Simulations/requirements.txt

# Full sweep for all three scenarios
python Simulations/run_all.py

# Quick validation run
python Simulations/run_all.py --smoke

# Run one scenario only
python Simulations/run_all.py --scenario logistic

# List indexed configurations (useful for SLURM arrays)
python Simulations/run_all.py --list-configs

# Run exactly one indexed configuration
python Simulations/run_all.py --config-index 5

# Generate the six manuscript figures from completed outputs
python Simulations/make_figures.py

# Generate a 3D illustration of prior atoms
python Simulations/plot_prior_atoms_3d.py
```

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
| `scenario_base.py` | Shared configuration, prior geometry, estimators, and scenario base class |
| `scenario_quadratic.py` | Quadratic sample-means scenario |
| `scenario_logistic.py` | Multiclass logistic scenario |
| `scenario_poisson.py` | Poisson regression scenario |
| `run_all.py` | Unified sweep runner for all scenarios |
| `make_figures.py` | Generate the six manuscript figures from completed outputs |
| `plot_prior_atoms_3d.py` | Create a 3D illustration of the five-curve prior atoms |
| `slurm_array_job.sh` | One SLURM array task = one indexed simulation config |
| `submit_slurm_array.sh` | Helper to submit the full config list as a SLURM array |
| `environment.yml` | Conda environment definition for local/HPC reproducibility |
| `requirements.txt` | Python dependencies |

## SLURM HPC Usage

Run from repository root:

```bash
# Optional: limit to a scenario or change reps
export RUN_ARGS="--scenario poisson --reps 50"

# Submit one array task per config index (many clusters require --account)
bash Simulations/submit_slurm_array.sh --account <your_account>

# Equivalent via environment variable
export SLURM_ACCOUNT=<your_account>
bash Simulations/submit_slurm_array.sh
```

Notes:

- Array indexing is zero-based (`SLURM_ARRAY_TASK_ID` maps to `--config-index`).
- The helper script computes the number of tasks using `python Simulations/run_all.py --list-configs`.
- To run a single task manually on a node: `python Simulations/run_all.py --config-index <idx> --no-progress`.
- You can set `PYTHON_BIN` to a specific interpreter if your cluster does not use `python3`.
- `submit_slurm_array.sh` also accepts `--partition`, `--qos`, `--time`, `--mem`, and `--cpus`.
- `submit_slurm_array.sh` auto-loads `Simulations/slurm_site_defaults.sh` when present.

## Requirements

- Python 3.9+
- `npeb` package with MOSEK support for the NPMLE-based estimators
- Packages listed in `requirements.txt`
