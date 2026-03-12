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
    sample_prior,
)
from typing import Dict


class QuadraticMeanScenario(Scenario):
    name = "quadratic"

    def variance_fn(self, theta: np.ndarray) -> np.ndarray:
        return np.clip(
            theta ** 2,
            VARIANCE_BOUNDS["s_min"],
            VARIANCE_BOUNDS["s_max"],
        )

    def generate_data(self, K: int, cfg: SimConfig, rng: np.random.Generator) -> Dict:
        weights = np.asarray(cfg.prior_weights)
        theta_true = sample_prior(K, weights, rng)

        n_k = rng.integers(cfg.n_min, cfg.n_max + 1, size=K)
        sigma2 = self.variance_fn(theta_true)             # (K, 3)
        obs_var = sigma2 / n_k[:, None]                    # true obs variance
        x = theta_true + rng.normal(size=(K, DIM)) * np.sqrt(obs_var)

        # For NPEB (Soloff), clients report local sample-variance estimates.
        # Simulate: var_hat_k = obs_var_k * chi2(n_k-1) / (n_k - 1)
        # which is the unbiased sample-variance estimator divided by n_k.
        df = np.maximum(n_k - 1, 1)
        chi2_scale = rng.chisquare(df, size=(K,))
        obs_var_estimated = obs_var * (chi2_scale / df)[:, None]

        return {
            "theta_true": theta_true,
            "x": x,
            "obs_var": obs_var_estimated,  # noisy local variance for NPEB
            "oracle_obs_var": obs_var,
            "n_k": n_k,
        }
