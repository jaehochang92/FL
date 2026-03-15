#!/usr/bin/env python3
"""
Scenario (i): Clients transmit sample means with quadratic variance.

Model:
    θ^(k) ~ G_0  (5-curve mixture prior in R^3)
    x^(k) | θ^(k) ~ N(θ^(k), Σ(θ^(k)) / n_k)
    Σ(θ) = diag(θ ⊙ θ), clipped to [s_min, s_max]

VANEB uses the known variance function; NPEB uses local sample variances.
"""

import numpy as np
from scenario_base import (
    Scenario, SimConfig, DIM, VARIANCE_BOUNDS,
    sample_prior, _clip_spd,
)
from typing import Dict


class QuadraticMeanScenario(Scenario):
    name = "quadratic"

    def variance_fn(self, theta: np.ndarray) -> np.ndarray:
        diag = np.clip(
            theta ** 2,
            VARIANCE_BOUNDS["s_min"],
            VARIANCE_BOUNDS["s_max"],
        )
        eye = np.eye(DIM)
        cov = np.einsum("...i,ij->...ij", diag, eye)
        return _clip_spd(cov, min_eig=1e-8, max_eig=1e8)

    def generate_data(self, K: int, cfg: SimConfig, rng: np.random.Generator) -> Dict:
        weights = np.asarray(cfg.prior_weights)
        theta_true = sample_prior(K, weights, rng)

        n_k = rng.integers(cfg.n_min, cfg.n_max + 1, size=K)
        sigma2 = self.variance_fn(theta_true)              # (K, 3, 3)
        obs_cov = sigma2 / n_k[:, None, None]              # true obs covariance
        z = rng.standard_normal(size=(K, DIM))
        chol = np.linalg.cholesky(obs_cov)
        x = theta_true + np.einsum("kij,kj->ki", chol, z)
        
        # For NPEB (Soloff), clients report local sample-variance estimates.
        # Simulate: cov_hat_k = obs_cov_k * chi2(n_k-1) / (n_k - 1)
        df = np.maximum(n_k - 1, 1)
        chi2_scale = rng.chisquare(df, size=(K,))
        obs_cov_estimated = obs_cov * (chi2_scale / df)[:, None, None]
        obs_cov_estimated = _clip_spd(obs_cov_estimated, min_eig=1e-10, max_eig=1e10)

        return {
            "theta_true": theta_true,
            "x": x,
            "obs_var": obs_cov_estimated,  # noisy local covariance for NPEB
            "oracle_obs_var": obs_cov,
            "n_k": n_k,
        }
