# Simulations

Simulation code for heteroskedastic personalized federated learning (Algorithm 2 in the paper).

## Overview

**Data model:**
$$
x_k \mid \theta_k \sim \mathcal{N}\!\left(\theta_k,\; \Sigma^{(k)}(\theta_k) / n^{(k)}\right)
$$
where $\Sigma(\theta)$ is a parameter-dependent diagonal covariance function and $n^{(k)}$ is client $k$'s local sample size.

**$G_0$:** Three-circle mixture in 2D (centroids: $(-2,0)$, $(2,0)$, $(0,2.5)$; radii: $1$; equal weights).

**Estimators:**
- **NPEB (proposed)**: NPMLE via `npeb.GLMixture` with MOSEK + EM refinement with atom-dependent covariance (heteroskedastic)
- **Soloff**: NPMLE homoscedastic baseline (assumes $\Sigma(\theta)$ constant, covariance estimated from $\text{diag}(x^2)$)
- **AdaMix**: Gaussian mixture baseline (Ozkara et al., 2023)
- **Oracle Bayes**: Posterior mean using true $G_0$

**Metric:** $\text{RMSE} = \left(\frac{1}{K}\sum_{k=1}^K \|\hat\theta_k - \theta_k\|^2\right)^{1/2}$

## Variance Functions

Select via `--sigma2-fn` or `SIGMA2_FN` environment variable:

| Option | Formula | Description |
|--------|---------|-------------|
| `quadratic` (default) | $\Sigma_j(\theta) = \theta_j^2$ | Variance grows quadratically |
| `linear` | $\Sigma_j(\theta) = \|\theta_j\|$ | Linear variance |
| `constant` | $\Sigma_j(\theta) = 1$ | Homoscedastic |
| `sqrt` | $\Sigma_j(\theta) = \sqrt{\|\theta_j\|}$ | Sublinear variance |

The variance function determines the ridgeline manifold geometry in Algorithm 2's atom-covariance coupling.

**Variance Clipping:** To comply with theoretical assumptions ($\underline{s} \leq \Sigma_j(\theta) \leq \overline{s}$), all variance functions are clipped to $[s_{\min}, s_{\max}] = [0.01, 100]$. The constant variance function naturally satisfies these bounds.

## Quick Start

```bash
# Install
pip install -r Simulations/requirements.txt

# Run the 3×3 grid (n_min × K) of 2D quadratic simulations
python Simulations/run_quad_grid.py

# Aggregate results + generate figures
python Simulations/analyze_results.py

# Table only (no figures)
python Simulations/analyze_results.py --no-plot

# Figures only (no table)
python Simulations/analyze_results.py --no-table
```

**Outputs:** `outputs/quad2d_nmin{N}_k{K}/metrics.csv`, `figures/quad_*.pdf`

## Key Implementation Details

**NPEB (Algorithm 2, heteroskedastic)**
- Initializes atoms from observations: $\{a_j\} \leftarrow \{x^{(k)}\}$
- MOSEK conic solver fits weights: $\max_w \sum_k \log \sum_j w_j \varphi_d(x^{(k)}; a_j, \Sigma^{(k)}_j)$
- EM refinement updates atoms, then **recomputes covariances** $\Sigma^{(k)}_j = \Sigma^{(k)}(a_j)$ at new locations
- Returns posterior mean: $\hat\theta^{(k)} = \sum_j r_{kj} a_j$ where $r_{kj} \propto w_j \varphi_d(x^{(k)}; a_j, \Sigma^{(k)}_j)$
- This atom-covariance coupling is essential for heteroskedastic consistency with the modified Tweedie formula.

**Soloff (homoscedastic baseline)**
- Assumes variance is **parameter-independent**: $\Sigma(\theta) = \text{const}$
- Estimates fixed covariance from data: $\Sigma_{\text{fixed}} = \text{diag}(\bar{x}^2)$ where $\bar{x}$ is empirical average
- NPMLE fitted with `homoscedastic=True`: all clients share same precision structure
- Provides a baseline showing cost of ignoring heteroskedasticity in the true data model

## File Structure

| File | Purpose |
|------|---------|
| `simulate_fl.py` | Core simulation driver (all 4 estimators) |
| `analyze_results.py` | Aggregate metrics + generate figures/LaTeX table |
| `run_quad_grid.py` | Runner: launches 3×3 grid of quadratic simulations |
| `logistic_regression_sim.py` | Federated logistic regression simulation |
| `requirements.txt` | Python dependencies |

## Outputs

- **`outputs/quad2d_nmin{N}_k{K}/metrics.csv`**: Per-replicate RMSE for each grid cell
- **`outputs/quad2d_nmin{N}_k{K}/snapshots/`**: Single-replicate point clouds
- **`outputs/quad2d_nmin{N}_k{K}/config.json`**: Configuration (variance function, prior, etc.)
- **`figures/quad_k_scaling.pdf`**: RMSE vs $K$ (fixed $n_{\min}=100$)
- **`figures/quad_nmin_effect.pdf`**: RMSE vs $n_{\min}$ (fixed $K=400$)
- **`figures/quad_improvement_heatmap.pdf`**: NPEB improvement over Soloff (%)

## Plotting Only

To regenerate figures from existing simulation data:

```bash
python Simulations/analyze_results.py --no-table
```

## Requirements

- Python 3.8+
- `npeb` package with MOSEK solver (**required** for NPEB estimator)
- Standard packages: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `scipy`, `tqdm`
