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
from typing import Dict, Optional

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
CLASS_PROJECTIONS_T = np.transpose(CLASS_PROJECTIONS, (0, 2, 1))

# Cache for Monte Carlo Fisher approximation keyed by n_mc.
_MC_CACHE = {}


def _project_features(X: np.ndarray) -> np.ndarray:
    """Project X through all class-specific matrices.

    Returns:
        projected: (n, C, d) where projected[:, c, :] = X @ B_c^T
    """
    return np.einsum("nd,cdk->nck", X, CLASS_PROJECTIONS_T)


def _get_mc_cache(n_mc: int) -> tuple:
    """Return cached Monte Carlo projections used in population Fisher."""
    cached = _MC_CACHE.get(n_mc)
    if cached is not None:
        return cached

    rng_mc = np.random.RandomState(123)
    X_mc = rng_mc.randn(n_mc, DIM)
    proj_mc = _project_features(X_mc)          # (n_mc, C, d)
    proj_mc_sq = proj_mc ** 2                  # (n_mc, C, d)
    _MC_CACHE[n_mc] = (proj_mc, proj_mc_sq)
    return proj_mc, proj_mc_sq


def _logits(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Compute (n, C) logits for features X (n, d) and parameter θ (d,).

    logit_c = (B_c X^T)^T θ = X B_c^T θ
    """
    projected = _project_features(X)  # (n, C, d)
    return np.einsum("ncd,d->nc", projected, theta)


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
    # Vectorized categorical sampling per row.
    u = rng.random(n)[:, None]
    y = np.argmax(u <= np.cumsum(probs, axis=1), axis=1)
    return y, X


def fit_multiclass_logistic(y: np.ndarray, X: np.ndarray) -> tuple:
    """Fit multiclass logistic regression via L-BFGS on cross-entropy.

    Returns:
        theta_hat: (d,)
        fisher_diag: (d,) empirical Fisher at theta_hat
    """
    n = X.shape[0]
    projected = _project_features(X)  # (n, C, d)
    y_onehot = np.eye(C)[y]

    def neg_log_lik_and_grad(theta):
        logit = np.einsum("ncd,d->nc", projected, theta)
        max_l = logit.max(axis=1, keepdims=True)
        log_sum_exp = max_l.ravel() + np.log(np.sum(np.exp(logit - max_l), axis=1))
        chosen_logit = logit[np.arange(n), y]
        nll = -np.mean(chosen_logit - log_sum_exp)

        probs = scipy_softmax(logit, axis=1)
        # grad = -E[(y_onehot - p) * feature]
        grad = -np.einsum("nc,ncd->d", (y_onehot - probs), projected) / n
        return nll, grad

    def obj(theta):
        nll, _ = neg_log_lik_and_grad(theta)
        return nll

    def jac(theta):
        _, grad = neg_log_lik_and_grad(theta)
        return grad

    result = minimize(
        obj,
        x0=np.zeros(DIM),
        jac=jac,
        method="L-BFGS-B",
        options={"maxiter": 120, "gtol": 1e-6},
    )
    theta_hat = result.x
    fisher_diag = empirical_fisher_diag(X, theta_hat, projected=projected)
    return theta_hat, fisher_diag


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
    proj_mc, proj_mc_sq = _get_mc_cache(n_mc)  # (n_mc, C, d)

    fisher = np.zeros((m, d))
    batch_size = 256
    for start in range(0, m, batch_size):
        end = min(start + batch_size, m)
        theta_b = theta[start:end]  # (b, d)

        # logits: (b, n_mc, C)
        logits_b = np.einsum("ncd,bd->bnc", proj_mc, theta_b)
        probs_b = scipy_softmax(logits_b, axis=2)
        w_b = probs_b * (1.0 - probs_b)

        # (b, d): average over n_mc and sum over C
        fisher_b = np.einsum("bnc,ncd->bd", w_b, proj_mc_sq) / n_mc
        fisher[start:end] = fisher_b

    fisher = np.maximum(fisher, 1e-6)
    if single:
        return fisher[0]
    return fisher


def empirical_fisher_diag(
    X: np.ndarray,
    theta: np.ndarray,
    projected: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Empirical Fisher diagonal at theta using data X.

    Returns (d,) diagonal of X^T diag(w) X summed over classes.
    """
    if projected is None:
        projected = _project_features(X)
    logit = np.einsum("ncd,d->nc", projected, theta)
    probs = scipy_softmax(logit, axis=1)
    w = probs * (1.0 - probs)
    fisher = np.einsum("nc,ncd->d", w, projected ** 2)
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
            theta_hat_i, fisher_diag = fit_multiclass_logistic(y, X)
            theta_hat[i] = theta_hat_i
            # Local Fisher-based variance estimate
            obs_var[i] = 1.0 / np.maximum(fisher_diag, 1e-6)

        return {
            "theta_true": theta_true,
            "x": theta_hat,
            "obs_var": obs_var,
            "oracle_obs_var": oracle_obs_var,
            "n_k": n_k,
        }
