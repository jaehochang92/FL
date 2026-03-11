#!/usr/bin/env python3
"""
Federated Poisson Regression Simulation.

Each client:
1. Samples true parameter theta from a three-circle mixture prior
2. Generates count data: y_i ~ Poisson(exp(X_i^T theta)), i=1,...,n_k
3. Fits Poisson regression via IRLS/Newton to get theta_hat (MLE)
4. Reports (theta_hat, Fisher Information diagonal)

Heteroskedasticity arises from Fisher Information:
    I(theta) = X^T diag(exp(X theta)) X
which grows *exponentially* with ||theta||. This creates much stronger
parameter-dependent uncertainty than logistic regression (bounded by 0.25).

Key property: V(theta) = I(theta)^{-1} varies by orders of magnitude across
the three-circle prior support, making this an ideal regime for heteroskedastic
NPEB to outperform parametric methods.
"""

import numpy as np
import statsmodels.api as sm
import sys
import os
import time
import warnings

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Simulations.simulate_fl import (
    glmixture_npeb_estimator,
    adamix_estimator, generate_uniform_prior_support,
    sample_three_circle_prior,
    oracle_posterior_mean,
    oracle_posterior_mean_local_mc,
    SimConfig, rmse
)


# ──────────────────────────────────────────────────────────────────
# Poisson regression helpers
# ──────────────────────────────────────────────────────────────────

def generate_poisson_data(
    theta_true: np.ndarray,
    sample_size: int,
    rng: np.random.Generator,
    feature_scale: float = None,
) -> tuple:
    """Generate count data from Poisson regression model.

    Model: y_i ~ Poisson(exp(X_i^T theta_true))
    Features: X ~ N(0, feature_scale^2 * I_d)

    feature_scale controls the signal strength and heteroskedasticity:
    - Larger scale -> stronger signal, more Fisher heterogeneity,
      but also more precise MLEs (less room for shrinkage)
    - Default: 1/sqrt(d), balancing noise and heteroskedasticity

    Args:
        theta_true: (d,) true coefficients
        sample_size: number of observations
        rng: random generator
        feature_scale: std of features (default: 1/sqrt(d))

    Returns:
        (y, X): (n,) counts and (n, d) features
    """
    d = len(theta_true)
    if feature_scale is None:
        feature_scale = 1.0 / np.sqrt(d)

    X = rng.standard_normal(size=(sample_size, d)) * feature_scale

    # Linear predictor and mean
    eta = X @ theta_true          # (n,)
    eta = np.clip(eta, -10, 10)   # Prevent extreme rates
    mu = np.exp(eta)              # E[y] = exp(X^T theta)

    y = rng.poisson(mu)
    return y, X


def fit_poisson_regression(
    y: np.ndarray,
    X: np.ndarray,
    max_iter: int = 200,
) -> tuple:
    """Fit Poisson regression via statsmodels GLM (IRLS) and return MLE + Fisher diagonal.

    Uses IRLS (Iteratively Reweighted Least Squares), which is the standard
    algorithm for GLMs -- faster and more numerically stable than L-BFGS-B.
    Fitted values from the model are reused to compute the Fisher diagonal
    without an extra matrix multiplication.

    Args:
        y: (n,) counts
        X: (n, d) features
        max_iter: maximum IRLS iterations

    Returns:
        (theta_hat, fisher_diag): ((d,), (d,)) MLE and Fisher Information diagonal
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        glm = sm.GLM(y, X, family=sm.families.Poisson())
        result = glm.fit(maxiter=max_iter, tol=1e-12, disp=0)

    theta_hat = np.asarray(result.params)
    mu = np.asarray(result.fittedvalues)           # exp(X @ theta_hat), reuse directly
    fisher_diag = np.sum(mu[:, None] * (X ** 2), axis=0)
    return theta_hat, np.maximum(fisher_diag, 1e-6)


# ──────────────────────────────────────────────────────────────────
# Main simulation routines
# ──────────────────────────────────────────────────────────────────

def generate_poisson_federated_data(
    k: int,
    cfg: SimConfig,
    rng: np.random.Generator,
    feature_scale: float = None,
) -> tuple:
    """Generate K clients' Poisson regression data.

    Returns:
        (theta_true, theta_hat, fisher_diag, n_k, X_list, y_list)
    """
    d = cfg.dim
    theta_true = sample_three_circle_prior(
        k, d,
        np.array(cfg.prior_centers, dtype=float),
        np.array(cfg.prior_radii, dtype=float),
        np.array(cfg.prior_weights, dtype=float),
        rng,
    )

    theta_hat = np.zeros((k, d))
    fisher_diag = np.zeros((k, d))
    n_k = np.zeros(k, dtype=int)
    X_list = []
    y_list = []

    for i in range(k):
        n_i = rng.integers(cfg.n_min, cfg.n_max + 1)
        n_k[i] = n_i

        y, X = generate_poisson_data(theta_true[i], n_i, rng,
                                     feature_scale=feature_scale)
        X_list.append(X)
        y_list.append(y)

        theta_hat[i], fisher_diag[i] = fit_poisson_regression(y, X)

    return theta_true, theta_hat, fisher_diag, n_k, X_list, y_list


def run_poisson_simulation(
    k: int,
    rep: int,
    cfg: SimConfig,
    rng: np.random.Generator,
    feature_scale: float = None,
) -> dict:
    """Run one federated Poisson regression simulation.

    Uses the *population* (closed-form) Fisher Information for the variance
    function, avoiding the noise of empirical Fisher with small samples.

    For Poisson regression with X ~ N(0, σ_x^2 I_d):
        F_d(θ) = (σ_x^2 + σ_x^4 θ_d^2) exp(σ_x^2 |θ|^2 / 2)

    This is the exact per-observation Fisher Information diagonal in the
    population limit. It captures the exponential heteroskedasticity that
    makes Poisson regression ideal for heteroskedastic NPEB.
    """
    d = cfg.dim
    sx = feature_scale if feature_scale is not None else 1.0 / np.sqrt(d)
    sx2 = sx ** 2

    theta_true, theta_hat, fisher_diag, n_k, X_list, y_list = \
        generate_poisson_federated_data(k, cfg, rng, feature_scale=feature_scale)

    x = theta_hat

    # ── Population Fisher variance function ──
    def population_fisher_variance_fn(theta: np.ndarray) -> np.ndarray:
        """Closed-form per-observation variance from population Fisher.

        sigma2_d(theta) = 1 / F_d(theta)
        where F_d(theta) = (σ_x^2 + σ_x^4 θ_d^2) exp(σ_x^2 |θ|^2 / 2)

        NPEB uses: prec_{k,j} = n_k / sigma2(atom_j)  =  n_k * F_d(atom_j)
        """
        single = (theta.ndim == 1)
        if single:
            theta = theta[np.newaxis, :]

        theta_sq = theta ** 2                       # (m, d)
        norm_sq = np.sum(theta_sq, axis=1, keepdims=True)  # (m, 1)

        # Per-observation Fisher diagonal: F_d(θ) = (σ² + σ⁴θ_d²) exp(σ²|θ|²/2)
        fisher_diag_pop = (sx2 + sx2**2 * theta_sq) * np.exp(sx2 * norm_sq / 2)

        # Clip to prevent extreme values
        fisher_diag_pop = np.clip(fisher_diag_pop, 1e-4, 1e6)
        sigma2 = 1.0 / fisher_diag_pop

        if single:
            return sigma2[0]
        return sigma2

    # Observation variance for AdaMix: population Fisher at theta_hat, divided by n_k
    pop_fisher_at_xhat = (sx2 + sx2**2 * x**2) * np.exp(sx2 * np.sum(x**2, axis=1, keepdims=True) / 2)
    obs_var = np.maximum(1.0 / pop_fisher_at_xhat, 1e-6)  # (K, d) per-obs variance
    # Note: total obs variance = obs_var / n_k, but AdaMix expects the
    # obs_var such that theta_hat ~ N(theta, diag(obs_var)).
    # With model: theta_hat ~ N(theta, sigma2(theta)/n_k),
    # obs_var for AdaMix = sigma2(theta_hat) / n_k
    obs_var = obs_var / n_k[:, None]

    # MLE baseline
    rmse_mle = rmse(x, theta_true)

    # ── Oracle (grid-based with atom-dependent population variance) ──
    centers = np.array(cfg.prior_centers, dtype=float)
    radii_arr = np.array(cfg.prior_radii, dtype=float)

    oracle_atoms = generate_uniform_prior_support(
        centers, radii_arr,
        atoms_per_circle=500,
    )

    t0 = time.time()
    theta_oracle = oracle_posterior_mean(
        oracle_atoms, x, n_k, population_fisher_variance_fn
    )
    oracle_time = time.time() - t0

    # ── NPEB with population Fisher variance function ──
    t0 = time.time()
    theta_npeb, npeb_time = glmixture_npeb_estimator(
        x, n_k, population_fisher_variance_fn, cfg.em_iters, quiet_solver=True
    )
    npeb_time = time.time() - t0

    # ── AdaMix baseline ──
    try:
        theta_adamix = adamix_estimator(x, obs_var, cfg, rng)
    except Exception as e:
        print(f"AdaMix failed: {e}, using mean instead")
        theta_adamix = np.mean(x, axis=0, keepdims=True).repeat(k, axis=0)

    return {
        "k": k,
        "rep": rep,
        "nmin": cfg.n_min,
        "rmse_mle": rmse_mle,
        "rmse_oracle": rmse(theta_oracle, theta_true),
        "rmse_npeb": rmse(theta_npeb, theta_true),
        "rmse_adamix": rmse(theta_adamix, theta_true),
        "oracle_runtime_sec": oracle_time,
        "npeb_runtime_sec": npeb_time,
    }


# ──────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import pandas as pd
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Federated Poisson Regression Simulation"
    )
    parser.add_argument("--k-values", type=str, default="100")
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--dim", type=int, default=3)
    parser.add_argument("--n-min", type=int, default=100)
    parser.add_argument("--n-max", type=int, default=300)
    parser.add_argument("--em-iters", type=int, default=25)
    parser.add_argument("--feature-scale", type=float, default=None,
                        help="Feature std (default: 1/sqrt(dim))")
    parser.add_argument("--seed", type=int, default=20260306)
    parser.add_argument("--outdir", type=str, default="Simulations/outputs/poisson_test")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    k_values = [int(x) for x in args.k_values.split(",")]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.dim == 3:
        prior_centers = [[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.5, 0.0]]
    elif args.dim == 2:
        prior_centers = [[-2.0, 0.0], [2.0, 0.0], [0.0, 2.5]]
    else:
        raise ValueError(f"Poisson sim supports dim=2 or 3, got {args.dim}")

    cfg = SimConfig(
        k_values=k_values,
        reps=args.reps,
        dim=args.dim,
        radius=1.0,
        n_min=args.n_min,
        n_max=args.n_max,
        em_iters=args.em_iters,
        adamix_components=3,
        adamix_iters=20,
        adamix_lr=0.05,
        prior_centers=prior_centers,
        prior_radii=[1.0, 1.0, 1.0],
        prior_weights=[1/3, 1/3, 1/3],
        random_seed=args.seed,
    )

    rng = np.random.default_rng(args.seed)
    results = []

    for k in k_values:
        for rep in range(args.reps):
            if not args.no_progress:
                print(f"  K={k}, Rep={rep+1}/{args.reps}", end="\r")
            try:
                row = run_poisson_simulation(k, rep, cfg, rng,
                                            feature_scale=args.feature_scale)
                results.append(row)
            except Exception as e:
                print(f"Error at K={k}, Rep={rep}: {e}")
                continue

    df_results = pd.DataFrame(results)
    df_results.to_csv(outdir / "metrics.csv", index=False)

    # Summary
    print(f"\n{'='*60}")
    print(f"Poisson Regression Simulation ({args.dim}D)")
    print(f"{'='*60}")
    for k in k_values:
        df_k = df_results[df_results['k'] == k]
        print(f"\nK={k} ({len(df_k)} replicates):")
        print(f"  MLE:     {df_k['rmse_mle'].mean():.4f} ± {df_k['rmse_mle'].std():.4f}")
        print(f"  Oracle:  {df_k['rmse_oracle'].mean():.4f} ± {df_k['rmse_oracle'].std():.4f}")
        print(f"  NPEB:    {df_k['rmse_npeb'].mean():.4f} ± {df_k['rmse_npeb'].std():.4f}")
        print(f"  AdaMix:  {df_k['rmse_adamix'].mean():.4f} ± {df_k['rmse_adamix'].std():.4f}")

    print(f"\n✓ Saved: {outdir / 'metrics.csv'}")
