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
    sample_prior, _clip_spd, _batch_inv,
)
from typing import Dict, Callable

FEATURE_SCALE = 1  # σ_x

# The Poisson exp link makes Fisher information grow as exp(σ²‖θ‖²/2).  With
# the shared prior curves the trefoil knot reaches ‖θ‖≈4.9, giving a per-obs
# Fisher ~1963 — three orders of magnitude larger than the other curves.  As K
# increases, NPMLE is dominated by those clients and RMSE grows instead of
# shrinking.  Scaling θ down by 0.5 brings max Fisher to ~8, keeping all five
# curves in the same order of magnitude and obs_var within VARIANCE_BOUNDS.
PRIOR_SCALE = .9  # applied to sample_prior output in generate_data


def _population_fisher_full(theta: np.ndarray) -> np.ndarray:
    """Closed-form population Fisher (full covariance) for Poisson regression.

    For X ~ N(0, σ² I):
        F(θ) = exp(σ² ||θ||² / 2) [ σ² I + σ⁴ θ θ^T ].
    """
    sx2 = FEATURE_SCALE ** 2
    single = (theta.ndim == 1)
    if single:
        theta = theta[None, :]
    norm_sq = np.sum(theta ** 2, axis=1)
    scale = np.exp(sx2 * norm_sq / 2.0)[..., None, None]
    outer = np.einsum("...i,...j->...ij", theta, theta)
    eye = np.eye(DIM)
    fisher = scale * (sx2 * eye + (sx2 ** 2) * outer)
    fisher = _clip_spd(fisher, min_eig=1e-4, max_eig=1e6)
    if single:
        return fisher[0]
    return fisher


def generate_poisson_data(
    theta_true: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> tuple:
    """
    Generate Poisson regression data.
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

    Returns (theta_hat, fisher_full): MLE and empirical Fisher (full).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        glm = sm.GLM(y, X, family=sm.families.Poisson())
        result = glm.fit(maxiter=200, tol=1e-12, disp=0)
    theta_hat = np.asarray(result.params)
    mu = np.asarray(result.fittedvalues)
    fisher_full = X.T @ (mu[:, None] * X)
    fisher_full = _clip_spd(fisher_full, min_eig=1e-6, max_eig=1e6)
    return theta_hat, fisher_full

def batch_poisson_fisher(X: np.ndarray, atoms: np.ndarray) -> np.ndarray:
    # X: (n, d), atoms: (M, d)
    eta = np.clip(np.einsum("md,nd->mn", atoms, X), -10, 10)
    mu = np.exp(eta)  # W = diag(mu)
    F = np.einsum("nd,mn,ne->mde", X, mu, X)
    return F

class PoissonScenario(Scenario):
    name = "poisson"
    prior_scale = PRIOR_SCALE

    def get_obs_prec_fn(self, data: Dict) -> Callable:
        X_list = data["X_list"]
        def prec_fn(atoms: np.ndarray) -> np.ndarray:
            K = len(X_list)
            M = atoms.shape[0]
            prec = np.zeros((K, M, DIM, DIM))
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
        weights = np.asarray(cfg.prior_weights)
        theta_true = sample_prior(K, weights, rng) * PRIOR_SCALE
        n_k = rng.integers(cfg.n_min, cfg.n_max + 1, size=K)

        theta_hat = np.zeros((K, DIM))
        obs_cov = np.zeros((K, DIM, DIM))
        oracle_obs_var = self.variance_fn(theta_true) / n_k[:, None, None]

        X_list = []
        for i in range(K):
            y, X = generate_poisson_data(theta_true[i], n_k[i], rng)
            X_list.append(X)
            th, fisher_full = fit_poisson_regression(y, X)
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
            "X_list": X_list
        }
