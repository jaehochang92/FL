#!/usr/bin/env python3
"""
Scenario (ii): Multiclass logistic regression with softmax (C = 6 classes).

Model:
    θ^(k) ∈ R^3 — 3-d parameter (shared across classes via feature projection)
    Each client has n_k observations:
        X_i ~ N(0, I_d),  Y_i ∈ {1,...,C}
        P(Y_i = c | X_i) = softmax(X_i @ W)[c]
    where W = θ ⊗ β_c maps R^3 → R^C using fixed class embeddings β_c.

    θ^(k)_hat is obtained via MLE (L-BFGS on cross-entropy).
    Fisher Information has closed form:
        I(θ) = Σ_c E_X[ p_c(1 - p_c) (β_c ⊗ X)(β_c ⊗ X)^T ]
    We use the diagonal of the population Fisher.
"""

import numpy as np
from scipy.special import softmax as scipy_softmax
from scipy.optimize import minimize
from scenario_base import (
    Scenario, SimConfig, DIM, VARIANCE_BOUNDS,
    sample_prior,
)
from typing import Dict

C = 6  # number of classes

# Fixed class embeddings β_c ∈ R^C used to map θ ∈ R^3 into C logits.
# Logit_c(x) = (β_c^T @ θ) * (x^T @ e_c_proj)
# For simplicity: W(θ) = B @ diag(θ) @ A  where B: C x 3, A: 3 x d_feat
# Actually we use a simpler structure:
#   logit_c(x) = x^T B_c θ    where B_c is a (d, d) matrix per class.
# Even simpler: logit_c(x) = (B_c x)^T θ, B_c ∈ R^{d x d}.
# We fix B_c as random orthogonal-ish matrices seeded deterministically.

_rng_class = np.random.RandomState(42)
CLASS_PROJECTIONS = []
for _ in range(C):
    M = _rng_class.randn(DIM, DIM)
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    CLASS_PROJECTIONS.append(U @ Vt)  # orthogonal matrix
CLASS_PROJECTIONS = np.array(CLASS_PROJECTIONS)  # (C, d, d)


def _logits(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Compute (n, C) logits for features X (n, d) and parameter θ (d,).

    logit_c = (B_c X^T)^T θ = X B_c^T θ
    """
    # X: (n, d), CLASS_PROJECTIONS: (C, d, d), theta: (d,)
    # For each class c: projected = X @ B_c^T @ theta → (n,)
    n = X.shape[0]
    out = np.zeros((n, C))
    for c in range(C):
        out[:, c] = X @ CLASS_PROJECTIONS[c].T @ theta
    return out


def _probs(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Softmax probabilities (n, C)."""
    logit = _logits(X, theta)
    return scipy_softmax(logit, axis=1)


def generate_multiclass_data(
    theta_true: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> tuple:
    """Generate multiclass classification data.

    Returns:
        (y, X): y (n,) integer labels in {0,...,C-1}, X (n, d) features
    """
    d = DIM
    X = rng.standard_normal(size=(n, d))
    probs = _probs(X, theta_true)  # (n, C)
    y = np.array([rng.choice(C, p=p) for p in probs])
    return y, X


def fit_multiclass_logistic(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Fit multiclass logistic regression via L-BFGS on cross-entropy.

    Returns theta_hat (d,).
    """
    n = X.shape[0]

    def neg_log_lik(theta):
        logit = _logits(X, theta)
        log_p = logit - np.log(np.sum(np.exp(logit - logit.max(axis=1, keepdims=True)), axis=1, keepdims=True) + 1e-300) - logit.max(axis=1, keepdims=True) + logit.max(axis=1, keepdims=True)
        # Numerically stable:
        log_p = logit - np.log(np.exp(logit).sum(axis=1, keepdims=True) + 1e-300)
        # clip for safety
        log_p = np.clip(log_p, -50, 0)
        return -np.mean(log_p[np.arange(n), y])

    def neg_log_lik_stable(theta):
        logit = _logits(X, theta)
        max_l = logit.max(axis=1, keepdims=True)
        log_sum_exp = max_l.ravel() + np.log(np.sum(np.exp(logit - max_l), axis=1))
        chosen_logit = logit[np.arange(n), y]
        return -np.mean(chosen_logit - log_sum_exp)

    result = minimize(neg_log_lik_stable, x0=np.zeros(DIM), method='L-BFGS-B',
                      options={'maxiter': 200})
    return result.x


def population_fisher_diag(theta: np.ndarray, n_mc: int = 2000) -> np.ndarray:
    """Population Fisher Information diagonal for the multiclass softmax model.

    I(θ)_{jj} = E_X[ Σ_c p_c(X,θ)(1-p_c(X,θ)) * (B_c X)_j^2 ]

    Uses Monte Carlo integration over X ~ N(0, I).

    Args:
        theta: (d,) or (m, d)
    Returns:
        (d,) or (m, d) Fisher diagonal
    """
    single = (theta.ndim == 1)
    if single:
        theta = theta[None, :]
    m, d = theta.shape
    rng_mc = np.random.RandomState(123)
    X_mc = rng_mc.randn(n_mc, d)

    fisher = np.zeros((m, d))
    for i in range(m):
        # logits: (n_mc, C)
        logit = _logits(X_mc, theta[i])
        probs = scipy_softmax(logit, axis=1)  # (n_mc, C)
        for c in range(C):
            w = probs[:, c] * (1 - probs[:, c])  # (n_mc,)
            projected = X_mc @ CLASS_PROJECTIONS[c].T  # (n_mc, d)
            # (B_c X)_j^2 * w → average over MC samples
            fisher[i] += np.mean(w[:, None] * projected ** 2, axis=0)

    fisher = np.maximum(fisher, 1e-6)
    if single:
        return fisher[0]
    return fisher


def empirical_fisher_diag(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Empirical Fisher diagonal at theta using data X.

    Returns (d,) diagonal of X^T diag(w) X summed over classes.
    """
    n, d = X.shape
    fisher = np.zeros(d)
    logit = _logits(X, theta)
    probs = scipy_softmax(logit, axis=1)
    for c in range(C):
        w = probs[:, c] * (1 - probs[:, c])
        projected = X @ CLASS_PROJECTIONS[c].T
        fisher += np.sum(w[:, None] * projected ** 2, axis=0)
    return np.maximum(fisher, 1e-6)


class LogisticScenario(Scenario):
    name = "logistic"

    def variance_fn(self, theta: np.ndarray) -> np.ndarray:
        """Population variance = 1 / Fisher diagonal, clipped."""
        fisher = population_fisher_diag(theta)
        return np.clip(1.0 / fisher, VARIANCE_BOUNDS["s_min"], VARIANCE_BOUNDS["s_max"])

    def generate_data(self, K: int, cfg: SimConfig, rng: np.random.Generator) -> Dict:
        weights = np.asarray(cfg.prior_weights)
        theta_true = sample_prior(K, weights, rng)
        n_k = rng.integers(cfg.n_min, cfg.n_max + 1, size=K)

        theta_hat = np.zeros((K, DIM))
        obs_var = np.zeros((K, DIM))
        oracle_obs_var = self.variance_fn(theta_true) / n_k[:, None]

        for i in range(K):
            y, X = generate_multiclass_data(theta_true[i], n_k[i], rng)
            theta_hat[i] = fit_multiclass_logistic(y, X)
            # Local Fisher-based variance estimate
            fisher_diag = empirical_fisher_diag(X, theta_hat[i])
            obs_var[i] = 1.0 / np.maximum(fisher_diag, 1e-6)

        return {
            "theta_true": theta_true,
            "x": theta_hat,
            "obs_var": obs_var,
            "oracle_obs_var": oracle_obs_var,
            "n_k": n_k,
        }
