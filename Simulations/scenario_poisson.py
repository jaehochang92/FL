#!/usr/bin/env python3
"""
Scenario (iii): Poisson regression.

Model:
    θ^(k) ∈ R^3 ~ G_0  (5-curve mixture prior)
    Each client has n_k observations:
        X_i ~ N(0, σ_x^2 I_3),  Y_i ~ Poisson(exp(X_i^T θ))
    MLE θ^(k)_hat via IRLS (statsmodels GLM).
    Population Fisher diagonal:
        F_j(θ) = (σ_x^2 + σ_x^4 θ_j^2) exp(σ_x^2 ||θ||^2 / 2)
"""

import warnings

import numpy as np
import statsmodels.api as sm
from scenario_base import (
    Scenario, SimConfig, DIM, VARIANCE_BOUNDS,
    sample_prior,
)
from typing import Dict

FEATURE_SCALE = 0.7  # σ_x


def _population_fisher_diag(theta: np.ndarray) -> np.ndarray:
    """Closed-form population Fisher diagonal for Poisson regression.

    F_j(θ) = (σ² + σ⁴ θ_j²) exp(σ² ||θ||² / 2)

    Args:
        theta: (d,) or (m, d)
    Returns:
        same shape as theta
    """
    sx2 = FEATURE_SCALE ** 2
    single = (theta.ndim == 1)
    if single:
        theta = theta[None, :]
    norm_sq = np.sum(theta ** 2, axis=1, keepdims=True)
    fisher = (sx2 + sx2 ** 2 * theta ** 2) * np.exp(sx2 * norm_sq / 2)
    fisher = np.clip(fisher, 1e-4, 1e6)
    if single:
        return fisher[0]
    return fisher


def generate_poisson_data(
    theta_true: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> tuple:
    """Generate Poisson regression data.

    Returns (y, X): y (n,) counts, X (n, d) features.
    """
    d = DIM
    X = rng.standard_normal(size=(n, d)) * FEATURE_SCALE
    eta = np.clip(X @ theta_true, -10, 10)
    mu = np.exp(eta)
    y = rng.poisson(mu)
    return y, X


def fit_poisson_regression(y: np.ndarray, X: np.ndarray) -> tuple:
    """Fit Poisson GLM via IRLS.

    Returns (theta_hat, fisher_diag): MLE and empirical Fisher diagonal.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        glm = sm.GLM(y, X, family=sm.families.Poisson())
        result = glm.fit(maxiter=200, tol=1e-12, disp=0)
    theta_hat = np.asarray(result.params)
    mu = np.asarray(result.fittedvalues)
    fisher_diag = np.sum(mu[:, None] * X ** 2, axis=0)
    return theta_hat, np.maximum(fisher_diag, 1e-6)


class PoissonScenario(Scenario):
    name = "poisson"

    def variance_fn(self, theta: np.ndarray) -> np.ndarray:
        """Population variance = 1 / Fisher diagonal, clipped."""
        fisher = _population_fisher_diag(theta)
        return np.clip(1.0 / fisher, VARIANCE_BOUNDS["s_min"], VARIANCE_BOUNDS["s_max"])

    def generate_data(self, K: int, cfg: SimConfig, rng: np.random.Generator) -> Dict:
        weights = np.asarray(cfg.prior_weights)
        theta_true = sample_prior(K, weights, rng)
        n_k = rng.integers(cfg.n_min, cfg.n_max + 1, size=K)

        theta_hat = np.zeros((K, DIM))
        obs_var = np.zeros((K, DIM))
        oracle_obs_var = self.variance_fn(theta_true) / n_k[:, None]

        for i in range(K):
            y, X = generate_poisson_data(theta_true[i], n_k[i], rng)
            th, fisher_diag = fit_poisson_regression(y, X)
            theta_hat[i] = th
            # Local empirical variance estimate for NPEB
            obs_var[i] = 1.0 / np.maximum(fisher_diag, 1e-6)

        return {
            "theta_true": theta_true,
            "x": theta_hat,
            "obs_var": obs_var,
            "oracle_obs_var": oracle_obs_var,
            "n_k": n_k,
        }
