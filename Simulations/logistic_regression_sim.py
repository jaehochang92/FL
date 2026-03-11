#!/usr/bin/env python3
"""
Federated Logistic Regression Simulation.

Each client:
1. Samples training data (y, X) from a generative model
2. Fits logistic regression: y = sigmoid(X @ theta)
3. Computes Fisher Information matrix at the MLE estimate
4. Reports (theta_hat, Fisher_Information)

The heteroskedasticity arises naturally from:
- Different client data sizes
- Different feature distributions
- Logistic regression's non-constant Fisher Information
"""

import numpy as np
from scipy.special import expit, logsumexp  # sigmoid
from scipy.optimize import minimize
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Simulations.simulate_fl import (
    glmixture_npeb_estimator,
    adamix_estimator, generate_uniform_prior_support,
    SimConfig, rmse
)

import warnings
from sklearn.exceptions import ConvergenceWarning
import time


def oracle_posterior_mean_with_obs_var(
    theta_atoms: np.ndarray,
    x: np.ndarray,
    obs_var: np.ndarray,
) -> np.ndarray:
    """Oracle Bayes posterior mean using client-specific observation variances.
    
    This version directly uses the provided observation variances instead of
    computing them from a variance function. This is necessary for logistic
    regression where obs_var = Fisher^{-1} varies by client in a way that
    cannot be expressed as sigma2(theta) / n_k.
    
    Args:
        theta_atoms: (J, d) array of prior support atoms
        x: (K, d) observations (in logistic case, these are theta_hat)
        obs_var: (K, d) client-specific observation variances
        
    Returns:
        (K, d) posterior means for each client
    """
    K, d = x.shape
    J = theta_atoms.shape[0]
    
    # For each client i and atom j, compute likelihood:
    # log p(x_i | atom_j) = -0.5 * sum_l [(x_i,l - atom_j,l)^2 / obs_var_i,l + log(2π obs_var_i,l)]
    
    # Expand dimensions for broadcasting: (K, J, d)
    diff = x[:, None, :] - theta_atoms[None, :, :]
    obs_var_3d = obs_var[:, None, :]  # (K, 1, d) for broadcasting
    obs_var_safe = np.maximum(obs_var_3d, 1e-12)
    
    # Compute log-likelihood for each (client, atom) pair
    quad = np.sum((diff ** 2) / obs_var_safe, axis=2)  # (K, J)
    log_det = np.sum(np.log(2.0 * np.pi * obs_var_safe), axis=2)  # (K, J)
    log_w = -0.5 * (quad + log_det)  # (K, J)
    
    # Normalize weights using logsumexp for numerical stability
    log_norm = logsumexp(log_w, axis=1, keepdims=True)
    w = np.exp(log_w - log_norm)
    
    # Handle numerical issues
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    row_sums = np.sum(w, axis=1, keepdims=True)
    bad = (~np.isfinite(row_sums)) | (row_sums <= 0)
    w = w / np.maximum(row_sums, 1e-300)
    if np.any(bad):
        w[bad[:, 0], :] = 1.0 / J  # Uniform weights if all weights collapsed
    
    # Compute posterior mean as weighted average of atoms
    theta_atoms_safe = np.nan_to_num(theta_atoms, nan=0.0, posinf=0.0, neginf=0.0)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        theta_post = w @ theta_atoms_safe
    
    return np.nan_to_num(theta_post, nan=0.0, posinf=0.0, neginf=0.0)



def generate_logistic_data(
    theta_true: np.ndarray,
    sample_size: int,
    rng: np.random.Generator,
) -> tuple:
    """Generate classification data from logistic model.
    
    y ~ Bernoulli(sigmoid(X @ theta_true))
    X ~ N(0, I) in d dimensions
    
    Args:
        theta_true: (d,) true coefficients
        sample_size: number of samples
        rng: random generator
        
    Returns:
        (y, X): (sample_size,) labels and (sample_size, d) features
    """
    d = len(theta_true)
    X = rng.standard_normal(size=(sample_size, d))
    logits = X @ theta_true
    probs = expit(logits)  # sigmoid
    y = rng.binomial(1, probs)
    return y, X


def fit_logistic_regression(
    y: np.ndarray,
    X: np.ndarray,
    max_iter: int = 100,
) -> np.ndarray:
    """Fit logistic regression via coordinate descent or L-BFGS.
    
    Args:
        y: (n,) binary labels
        X: (n, d) features
        max_iter: max iterations
        
    Returns:
        theta_hat: (d,) estimated coefficients
    """
    d = X.shape[1]
    
    def logistic_loss(theta):
        """Negative log-likelihood."""
        logits = X @ theta
        # Avoid numerical overflow
        logits = np.clip(logits, -500, 500)
        return -np.mean(y * logits - np.log(1 + np.exp(logits)))
    
    result = minimize(
        logistic_loss,
        x0=np.zeros(d),
        method='L-BFGS-B',
        options={'maxiter': max_iter}
    )
    
    return result.x


def compute_fisher_information(
    y: np.ndarray,
    X: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    """Compute Fisher Information matrix for logistic regression.
    
    Fisher Information: I(theta) = X^T * Sigma(theta) * X
    where Sigma(theta)_ii = p_i * (1 - p_i), p_i = sigmoid(X_i @ theta)
    
    Args:
        y: (n,) binary labels (unused, just for API consistency)
        X: (n, d) features
        theta: (d,) estimated coefficients
        
    Returns:
        I: (d, d) Fisher Information matrix (precision)
    """
    logits = X @ theta
    logits = np.clip(logits, -500, 500)
    probs = expit(logits)  # sigmoid
    
    # Diagonal weights: p_i * (1 - p_i)
    weights = probs * (1 - probs)
    
    # X^T * diag(weights) * X
    # Numerical stable version
    X_weighted = X * weights[:, np.newaxis]
    fisher = X_weighted.T @ X
    
    return fisher


def generate_logistic_federated_data(
    k: int,
    cfg: SimConfig,
    rng: np.random.Generator,
) -> tuple:
    """Generate K client datasets from logistic regression model.
    
    Each client:
    1. Samples true parameter from prior (circle/sphere mixture)
    2. Generates training data
    3. Fits logistic regression
    4. Returns (estimated_coeff, fisher_information, sample_size, features, labels)
    
    Args:
        k: number of clients
        cfg: simulation config
        rng: random generator
        
    Returns:
        (theta_true, theta_hat, fisher_info, n_k, X_list, y_list)
        - theta_true: (k, d) true parameters at each client
        - theta_hat: (k, d) estimated parameters
        - fisher_info: (k, d) diagonal of Fisher (used as precision in heteroskedastic model)
        - n_k: (k,) sample sizes at each client
        - X_list: list of (n_i, d) feature matrices for Fisher computation
        - y_list: list of (n_i,) label vectors for Fisher computation
    """
    from Simulations.simulate_fl import sample_three_circle_prior
    
    d = cfg.dim
    
    # Sample true parameters from prior (mixture of circles/spheres)
    theta_true = sample_three_circle_prior(
        k,
        cfg.dim,
        np.array(cfg.prior_centers, dtype=float),
        np.array(cfg.prior_radii, dtype=float),
        np.array(cfg.prior_weights, dtype=float),
        rng,
    )
    
    # Initialize arrays to collect results
    theta_hat = np.zeros((k, d))
    fisher_diag = np.zeros((k, d))  # We'll store diagonal of Fisher
    n_k = np.zeros(k, dtype=int)
    X_list = []  # Store feature matrices for later Fisher computation
    y_list = []  # Store labels for Fisher computation
    
    # For each client, generate data and fit logistic regression
    for i in range(k):
        # Sample size for client i
        n_i = rng.integers(cfg.n_min, cfg.n_max + 1)
        n_k[i] = n_i
        
        # Generate data from logistic model
        y, X = generate_logistic_data(theta_true[i], n_i, rng)
        
        # Store features and labels for building variance functions later
        X_list.append(X)
        y_list.append(y)
        
        # Fit logistic regression
        theta_hat[i] = fit_logistic_regression(y, X)
        
        # Compute Fisher Information (we use diagonal as precision proxy)
        fisher = compute_fisher_information(y, X, theta_hat[i])
        
        # Store diagonal (standard precision handling in our framework)
        # Note: full Fisher is more correct, but we can also use it as covariance estimate
        fisher_diag[i] = np.diag(fisher)
    
    return theta_true, theta_hat, fisher_diag, n_k, X_list, y_list


def run_logistic_simulation(
    k: int,
    rep: int,
    cfg: SimConfig,
    rng: np.random.Generator,
) -> dict:
    """Run one logistic regression federated learning simulation.
    
    Args:
        k: number of clients
        rep: replication number
        cfg: simulation config
        rng: random generator
        
    Returns:
        dict with RMSE metrics for all estimators
    """
    # Generate logistic regression data
    theta_true, theta_hat, fisher_diag, n_k, X_list, y_list = generate_logistic_federated_data(
        k, cfg, rng
    )
    
    # In our framework, we treat theta_hat as observations x, fisher_diag as precision
    # This is: x = theta_hat ~ N(theta, Fisher^{-1})
    x = theta_hat  # "observations" are the estimated coefficients
    
    # Create observation variance as inverse of Fisher
    obs_var = np.maximum(1.0 / fisher_diag, 1e-6)  # Avoid division issues
    
    # Build a Fisher-based variance function for NPEB
    # This function computes Fisher Information diagonal at arbitrary theta values
    # using the feature matrices and labels from logistic fits.
    def fisher_variance_fn(theta: np.ndarray) -> np.ndarray:
        """Compute Fisher Information diagonal at theta using all client data.
        
        For each dimension, compute Fisher[d] = pooled estimate across clients.
        Fisher Information at theta: I[d] = sum_i p_i(1-p_i) * X[i,d]^2
        where p_i = sigmoid(X[i] @ theta)
        
        Args:
            theta: (d,) or (m, d) - single or multiple parameter vectors
            
        Returns:
            (d,) or (m, d) - Fisher Information diagonal
        """
        single = (theta.ndim == 1)
        if single:
            theta = theta[np.newaxis, :]  # (1, d)
        
        m, d = theta.shape
        fisher_out = np.zeros((m, d))
        
        # Pool across all clients to estimate Fisher structure
        for client_idx in range(len(X_list)):
            X_i = X_list[client_idx]  # (n_i, d)
            y_i = y_list[client_idx]  # (n_i,)
            
            # For each theta value, compute Fisher at that location
            logits = X_i @ theta.T  # (n_i, m)
            p_i = expit(logits)  # (n_i, m)
            w_i = p_i * (1 - p_i)  # (n_i, m) - Bernoulli variance weights
            
            # Fisher diagonal: sum_j w_ij * X_i[j,d]^2 for each dimension d
            for d_idx in range(d):
                fisher_out[:, d_idx] += np.sum(w_i * (X_i[:, d_idx:d_idx+1]**2), axis=0)
        
        # Clip to safe bounds [min_var, max_var]
        fisher_out = np.clip(fisher_out, 0.01, 30.0)
        
        if single:
            return fisher_out[0]
        return fisher_out
    
    # Run Oracle with fixed uniform prior support (scaled by dimension)
    atoms_per_circle = 50 * max(1, int(np.ceil(np.sqrt(cfg.dim))))
    oracle_atoms = generate_uniform_prior_support(
        np.array(cfg.prior_centers, dtype=float),
        np.array(cfg.prior_radii, dtype=float),
        atoms_per_circle=atoms_per_circle,
    )
    
    # Oracle: use Fisher-based variance
    t0 = time.time()
    theta_oracle = oracle_posterior_mean_with_obs_var(oracle_atoms, x, obs_var)
    oracle_time = time.time() - t0
    
    # NPEB: use Fisher-based variance function to ensure fair comparison
    t0 = time.time()
    theta_npeb, npeb_time = glmixture_npeb_estimator(
        x, n_k, fisher_variance_fn, cfg.em_iters, quiet_solver=True
    )
    npeb_time = time.time() - t0
    
    # AdaMix baseline
    try:
        theta_adamix = adamix_estimator(x, obs_var, cfg, rng)
    except Exception as e:
        print(f"AdaMix failed: {e}, using mean instead")
        theta_adamix = np.mean(x, axis=0)
    
    # Compute RMSE against true parameters
    return {
        "k": k,
        "rep": rep,
        "rmse_oracle": rmse(theta_oracle, theta_true),
        "rmse_npeb": rmse(theta_npeb, theta_true),
        "rmse_adamix": rmse(theta_adamix, theta_true),
        "oracle_runtime_sec": oracle_time,
        "npeb_runtime_sec": npeb_time,
    }


if __name__ == "__main__":
    # Test with small simulation
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--k-values", type=str, default="20")
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--dim", type=int, default=3)
    parser.add_argument("--n-min", type=int, default=50)
    parser.add_argument("--n-max", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260305)
    parser.add_argument("--outdir", type=str, default="outputs/logistic_test")
    args = parser.parse_args()
    
    import pandas as pd
    from pathlib import Path
    
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    
    # Parse k values
    k_values = [int(x) for x in args.k_values.split(",")]
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Config - use proper 3D centers (not degenerate zero-padding)
    if args.dim == 3:
        # Proper 3D placement: three circles in 3D space using all dimensions
        prior_centers = [[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.5, 0.0]]
    elif args.dim == 2:
        # Original 2D setup
        prior_centers = [[-2.0, 0.0], [2.0, 0.0], [0.0, 2.5]]
    else:
        raise ValueError(f"Logistic regression only supports dim=2 or dim=3, got dim={args.dim}")
    
    cfg = SimConfig(
        k_values=k_values,
        reps=args.reps,
        dim=args.dim,
        radius=1.0,
        n_min=args.n_min,
        n_max=args.n_max,
        em_iters=25,
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
            print(f"K={k}, Rep={rep+1}/{args.reps}", end="\r")
            try:
                row = run_logistic_simulation(k, rep, cfg, rng)
                results.append(row)
            except Exception as e:
                print(f"Error at K={k}, Rep={rep}: {e}")
                continue
    
    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(outdir / "metrics.csv", index=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Logistic Regression Simulation ({args.dim}D)")
    print(f"{'='*60}")
    for k in k_values:
        df_k = df_results[df_results['k'] == k]
        print(f"\nK={k} ({len(df_k)} replicates):")
        print(f"  Oracle:  {df_k['rmse_oracle'].mean():.4f} ± {df_k['rmse_oracle'].std():.4f}")
        print(f"  NPEB:    {df_k['rmse_npeb'].mean():.4f} ± {df_k['rmse_npeb'].std():.4f}")
        print(f"  AdaMix:  {df_k['rmse_adamix'].mean():.4f} ± {df_k['rmse_adamix'].std():.4f}")
    
    print(f"\n✓ Saved: {outdir / 'metrics.csv'}")
