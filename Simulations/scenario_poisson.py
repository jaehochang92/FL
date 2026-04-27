#!/usr/bin/env python3
"""
Scenario (iii): Poisson regression.

Model:
    θ^(k) ∈ R^3 ~ G_0  (5-curve mixture prior)
    Each client has n_k observations:
        X_i ~ N(0, Σ_{x,k}),  Y_i ~ Poisson(exp(X_i^T θ))
    MLE θ^(k)_hat via IRLS (statsmodels GLM).
    Population Fisher (full covariance):
        F(θ) = exp(θ^T Σ_x θ / 2) [ Σ_x + (Σ_x θ)(Σ_x θ)^T ]
"""

import warnings

import numpy as np
import statsmodels.api as sm
from scenario_base import (
    Scenario, SimConfig, DIM, VARIANCE_BOUNDS,
    sample_prior, _clip_spd, _batch_inv,
)
from typing import Dict, Callable, Optional

FEATURE_SCALE = .1  # σ_x
FEATURE_EIGEN_MIN = 0.5
FEATURE_EIGEN_MAX = 2.0

# The Poisson exp link makes Fisher information grow as exp(σ²‖θ‖²/2).  With
# the shared prior curves the trefoil knot reaches ‖θ‖≈4.9, giving a per-obs
# Fisher ~1963 — three orders of magnitude larger than the other curves.  As K
# increases, NPMLE is dominated by those clients and RMSE grows instead of
# shrinking.  Scaling θ down by 0.5 brings max Fisher to ~8, keeping all five
# curves in the same order of magnitude and obs_var within VARIANCE_BOUNDS.
PRIOR_SCALE = .7  # applied to sample_prior output in generate_data


def _sample_client_covariances(
    K: int,
    rng: np.random.Generator,
    eig_min: float = FEATURE_EIGEN_MIN,
    eig_max: float = FEATURE_EIGEN_MAX,
) -> np.ndarray:
    """Sample K SPD feature covariances Σ_{x,k} with bounded eigenvalues."""
    covariances = np.zeros((K, DIM, DIM))
    base_var = FEATURE_SCALE ** 2
    for k in range(K):
        Q, _ = np.linalg.qr(rng.standard_normal((DIM, DIM)))
        eigvals = base_var * rng.uniform(eig_min, eig_max, size=DIM)
        covariances[k] = Q @ np.diag(eigvals) @ Q.T
    return _clip_spd(covariances, min_eig=1e-8, max_eig=1e6)


def _population_fisher_full(
    theta: np.ndarray,
    sigma_x: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Closed-form population Fisher (full covariance) for Poisson regression.

    For X ~ N(0, Σ_x):
        F(θ) = exp(θ^T Σ_x θ / 2) [ Σ_x + (Σ_x θ)(Σ_x θ)^T ].
    """
    single = (theta.ndim == 1)
    if single:
        theta = theta[None, :]
    m = theta.shape[0]

    if sigma_x is None:
        sigma_x = np.eye(DIM) * (FEATURE_SCALE ** 2)

    sigma_x = np.asarray(sigma_x)
    if sigma_x.ndim == 2:
        sigma_x = np.broadcast_to(sigma_x, (m, DIM, DIM)).copy()
    elif sigma_x.ndim == 3 and sigma_x.shape[0] == m:
        sigma_x = sigma_x.copy()
    else:
        raise ValueError("sigma_x must have shape (d,d) or (m,d,d)")

    sigma_x = _clip_spd(sigma_x, min_eig=1e-8, max_eig=1e6)
    sigma_theta = np.einsum("mde,me->md", sigma_x, theta)
    quad = np.einsum("md,md->m", theta, sigma_theta)
    scale = np.exp(np.clip(quad / 2.0, -40.0, 40.0))[:, None, None]
    fisher = scale * (sigma_x + np.einsum("md,me->mde", sigma_theta, sigma_theta))
    fisher = _clip_spd(fisher, min_eig=1e-4, max_eig=1e6)
    if single:
        return fisher[0]
    return fisher


def generate_poisson_data(
    theta_true: np.ndarray,
    n: int,
    rng: np.random.Generator,
    chol_x: np.ndarray,
) -> tuple:
    """
    Generate Poisson regression data.
    Returns (y, X): y (n,) counts, X (n, d) features.
    """
    d = DIM
    X = rng.standard_normal(size=(n, d)) @ chol_x.T
    eta = np.clip(X @ theta_true, -10, 10)
    mu = np.exp(eta)
    y = rng.poisson(mu)
    return y, X


def fit_poisson_regression(y: np.ndarray, X: np.ndarray) -> tuple:
    """Fit Poisson GLM via statsmodels with robust regularization.

    Uses fit_regularized() to handle convergence issues robustly,
    falling back to standard IRLS if needed.

    Returns (theta_hat, fisher_full): MLE and empirical Fisher (full).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        glm = sm.GLM(y, X, family=sm.families.Poisson())
        
        try:
            # Use regularized fit for robustness (elastic net penalties)
            # Alpha=0 recovers MLE; use very small alpha for numerical stability
            result = glm.fit_regularized(L1_wt=0.0, alpha=1e-10, maxiter=300)
        except Exception:
            # Fallback to standard IRLS
            try:
                result = glm.fit(maxiter=300, tol=1e-10, disp=0)
            except Exception:
                # Last resort: return zero parameters with identity Fisher
                return np.zeros(X.shape[1]), np.eye(X.shape[1])
    
    theta_hat = np.asarray(result.params)
    
    # Compute empirical Fisher with numerical safety
    eta = np.clip(X @ theta_hat, -10.0, 10.0)
    mu = np.exp(eta)
    fisher_full = X.T @ (mu[:, None] * X)
    fisher_full = _clip_spd(fisher_full, min_eig=1e-6, max_eig=1e6)
    return theta_hat, fisher_full

def batch_poisson_fisher(X: np.ndarray, atoms: np.ndarray) -> np.ndarray:
    # X: (n, d), atoms: (M, d)
    eta = np.clip(np.einsum("md,nd->mn", atoms, X), -10, 10)
    mu = np.exp(eta)  # W = diag(mu)
    F = np.einsum("nd,mn,ne->mde", X, mu, X)
    return F


def batch_poisson_fisher_diag(X: np.ndarray, atoms: np.ndarray) -> np.ndarray:
    """Batch Poisson Fisher information (diagonal-only structure).
    
    Returns (M, d, d) matrices with zeros off-diagonal for fast EM updates.
    Computed from full Fisher but only diagonal is retained.
    
    Args:
        X: (n, d) feature matrix
        atoms: (M, d) parameter atoms
    
    Returns:
        F_diag: (M, d, d) diagonal precision matrices (zeros off-diagonal)
    """
    # Compute full Fisher first
    F_full = batch_poisson_fisher(X, atoms)  # (M, d, d)
    
    # Extract diagonal and create diagonal-only matrices
    M, d, _ = F_full.shape
    F_diag = np.zeros_like(F_full)
    for i in range(M):
        np.fill_diagonal(F_diag[i], np.diag(F_full[i]))
    return F_diag


class PoissonScenario(Scenario):
    name = "poisson"
    prior_scale = PRIOR_SCALE

    def get_obs_prec_fn(self, data: Dict) -> Callable:
        X_list = data["X_list"]
        use_diag = data.get("use_diag", False)
        
        def prec_fn(atoms: np.ndarray) -> np.ndarray:
            K = len(X_list)
            M = atoms.shape[0]
            prec = np.zeros((K, M, DIM, DIM))
            
            if use_diag:
                # Fast diagonal-only Fisher
                for k in range(K):
                    F_total = batch_poisson_fisher_diag(X_list[k], atoms)
                    prec[k] = _clip_spd(F_total, min_eig=1e-8, max_eig=1e8)
            else:
                # Full Fisher matrix
                for k in range(K):
                    F_total = batch_poisson_fisher(X_list[k], atoms)
                    prec[k] = _clip_spd(F_total, min_eig=1e-8, max_eig=1e8)
            
            return prec
        return prec_fn

    def variance_fn(self, theta: np.ndarray) -> np.ndarray:
        fisher = _population_fisher_full(theta)
        cov = _batch_inv(fisher, min_eig=1e-6, max_eig=1e6)
        return _clip_spd(
            cov,
            min_eig=VARIANCE_BOUNDS["s_min"],
            max_eig=VARIANCE_BOUNDS["s_max"],
        )

    def generate_data(self, K: int, cfg: SimConfig, rng: np.random.Generator) -> Dict:
        from scenario_base import _parallel_fit_clients
        
        weights = np.asarray(cfg.prior_weights)
        theta_true = sample_prior(K, weights, rng) * PRIOR_SCALE
        n_k = rng.integers(cfg.n_min, cfg.n_max + 1, size=K)
        sigma_x = _sample_client_covariances(K, rng)
        chol_x = np.linalg.cholesky(sigma_x)

        theta_hat = np.zeros((K, DIM))
        obs_cov = np.zeros((K, DIM, DIM))
        fisher_oracle = _population_fisher_full(theta_true, sigma_x)
        oracle_cov = _batch_inv(fisher_oracle, min_eig=1e-6, max_eig=1e6)
        oracle_obs_var = np.zeros((K, DIM, DIM))
        for i in range(K):
            oracle_obs_var[i] = _clip_spd(
                oracle_cov[i] / n_k[i],
                min_eig=VARIANCE_BOUNDS["s_min"] / n_k[i],
                max_eig=VARIANCE_BOUNDS["s_max"] / n_k[i],
            )

        # Generate data and collect fit tuples
        X_list = []
        fit_tuples = []
        for i in range(K):
            y, X = generate_poisson_data(theta_true[i], n_k[i], rng, chol_x[i])
            X_list.append(X)
            fit_tuples.append((y, X))

        # Fit clients in parallel
        fit_results = _parallel_fit_clients(
            fit_tuples,
            scenario_type="poisson",
            n_jobs=-1,
            backend="multiprocessing"
        )

        # Collect results
        for i, (th, fisher_full) in enumerate(fit_results):
            theta_hat[i] = th
            cov_i = _batch_inv(fisher_full[None, :, :], min_eig=1e-6, max_eig=1e6)[0]
            obs_cov[i] = _clip_spd(
                cov_i,
                min_eig=VARIANCE_BOUNDS["s_min"] / n_k[i],
                max_eig=VARIANCE_BOUNDS["s_max"] / n_k[i],
            )

        return {
            "theta_true": theta_true,
            "x": theta_hat,
            "obs_var": obs_cov,
            "oracle_obs_var": oracle_obs_var,
            "n_k": n_k,
            "X_list": X_list,
            "Sigma_x": sigma_x,
            "use_diag": cfg.use_diag,
        }
