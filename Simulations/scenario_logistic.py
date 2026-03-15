#!/usr/bin/env python3
"""
Scenario (ii): Multiclass logistic regression with softmax (C = 3 classes).

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
import warnings
from scipy.special import softmax as scipy_softmax
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from scenario_base import (
    Scenario, SimConfig, DIM, VARIANCE_BOUNDS,
    sample_prior, _clip_spd, _batch_inv,
)
from typing import Dict, Optional, Callable

C = 3  # number of classes

# Fixed class embeddings β_c ∈ R^C used to map θ ∈ R^3 into C logits.
# Logit_c(x) = (β_c^T @ θ) * (x^T @ e_c_proj)
# For simplicity: W(θ) = B @ diag(θ) @ A  where B: C x 3, A: 3 x d_feat
# Actually we use a simpler structure:
#   logit_c(x) = x^T B_c θ    where B_c is a (d, d) matrix per class.
# Even simpler: logit_c(x) = (B_c x)^T θ, B_c ∈ R^{d x d}.
# We fix B_c as random orthogonal-ish matrices seeded deterministically.

# B1 = diag(1, -1, 0), B2 = diag(0, 1, -1), B3 = diag(-1, 0, 1)
CLASS_PROJECTIONS = np.array([
    np.diag([1.0, -1.0, 0.0]),
    np.diag([0.0, 1.0, -1.0]),
    np.diag([-1.0, 0.0, 1.0])
])
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


def generate_multiclass_data(theta_true: np.ndarray, n: int, rng: np.random.Generator) -> tuple:
    d = DIM
    X = rng.standard_normal(size=(n, d))
    probs = _probs(X, theta_true)  # (n, C)
    y = np.array([rng.choice(C, p=p) for p in probs]) 
    return y, X


def fit_multiclass_logistic(y: np.ndarray, X: np.ndarray) -> tuple:
    """Fit multiclass logistic regression via L-BFGS on cross-entropy.

    Returns:
        theta_hat: (d,)
        fisher_full: (d, d) empirical Fisher at theta_hat
    """
    n = X.shape[0]
    projected = _project_features(X)  # (n, C, d)

    # Fast initialization using sklearn; final estimate is still from the exact
    # constrained objective below.
    theta0 = np.zeros(DIM)
    try:
        X_fast = np.mean(projected, axis=1)  # (n, d)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = LogisticRegression(
                solver="lbfgs",
                fit_intercept=False,
                max_iter=60,
                tol=1e-4,
            )
            clf.fit(X_fast, y)
        A = CLASS_PROJECTIONS_T.reshape(C * DIM, DIM)
        b = clf.coef_.reshape(C * DIM)
        theta0, *_ = np.linalg.lstsq(A, b, rcond=None)
        theta0 = np.clip(theta0, -5.0, 5.0)
    except Exception:
        theta0 = np.zeros(DIM)

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
        x0=theta0,
        jac=jac,
        method="L-BFGS-B",
        options={"maxiter": 80, "gtol": 1e-6},
    )
    theta_hat = result.x
    fisher_full = empirical_fisher_full(X, theta_hat, projected=projected)
    return theta_hat, fisher_full

def batch_observed_fisher(projected: np.ndarray, atoms: np.ndarray) -> np.ndarray:
    n = projected.shape[0]
    logit = np.einsum("ncd,md->mnc", projected, atoms)
    probs = scipy_softmax(logit, axis=2)
    term1 = np.einsum("mnc,ncd,nce->mde", probs, projected, projected) / n
    mu = np.einsum("mnc,ncd->mnd", probs, projected)
    term2 = np.einsum("mnd,mne->mde", mu, mu) / n
    return term1 - term2

def population_fisher_full(theta: np.ndarray, n_mc: int = 2000) -> np.ndarray:
    """Population Fisher Information (full covariance) for multiclass softmax.

    Fisher(θ) = E_X[ Σ_c p_c f_c f_c^T - (Σ_c p_c f_c)(Σ_c p_c f_c)^T ],
    where f_c = B_c X and X ~ N(0, I).
    """
    single = (theta.ndim == 1)
    if single:
        theta = theta[None, :]
    m, d = theta.shape
    proj_mc, _ = _get_mc_cache(n_mc)  # (n_mc, C, d)

    fisher = np.zeros((m, d, d))
    batch_size = 128
    for start in range(0, m, batch_size):
        end = min(start + batch_size, m)
        theta_b = theta[start:end]  # (b, d)

        logits_b = np.einsum("ncd,bd->bnc", proj_mc, theta_b)   # (b,n_mc,C)
        probs_b = scipy_softmax(logits_b, axis=2)
        term1 = np.einsum("bnc,ncd,nce->bnde", probs_b, proj_mc, proj_mc)
        mu = np.einsum("bnc,ncd->bnd", probs_b, proj_mc)
        term2 = np.einsum("bnd,bne->bnde", mu, mu)
        fisher[start:end] = np.mean(term1 - term2, axis=1)

    fisher = _clip_spd(fisher, min_eig=1e-6, max_eig=1e6)
    if single:
        return fisher[0]
    return fisher


def empirical_fisher_full(
    X: np.ndarray,
    theta: np.ndarray,
    projected: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Empirical Fisher (full) at theta using observed data X."""
    if projected is None:
        projected = _project_features(X)
    n = X.shape[0]
    logit = np.einsum("ncd,d->nc", projected, theta)
    probs = scipy_softmax(logit, axis=1)
    term1 = np.einsum("nc,ncd,nce->de", probs, projected, projected) / n
    mu = np.einsum("nc,ncd->nd", probs, projected)
    term2 = np.einsum("nd,ne->de", mu, mu) / n
    fisher = term1 - term2
    return _clip_spd(fisher, min_eig=1e-6, max_eig=1e6)


class LogisticScenario(Scenario):
    name = "logistic"

    def variance_fn(self, theta: np.ndarray) -> np.ndarray:
        fisher = population_fisher_full(theta)
        cov = _batch_inv(fisher, min_eig=1e-6, max_eig=1e6)
        return _clip_spd(
            cov,
            min_eig=VARIANCE_BOUNDS["s_min"],
            max_eig=VARIANCE_BOUNDS["s_max"],
        )
    
    def get_obs_prec_fn(self, data: Dict) -> Callable:
        X_list = data["X_list"]
        n_k = data["n_k"]
        K = len(X_list)
        # 생성 시점에 미리 projection 해둠 (연산 최적화)
        proj_list = [_project_features(X) for X in X_list]

        def prec_fn(atoms: np.ndarray) -> np.ndarray:
            M = atoms.shape[0]
            prec = np.zeros((K, M, DIM, DIM))
            for k in range(K):
                F_1 = batch_observed_fisher(proj_list[k], atoms)
                # 1개 관측치 피셔에 n_k를 곱해 총 정밀도를 완성
                prec[k] = _clip_spd(F_1 * n_k[k], min_eig=1e-8, max_eig=1e8)
            return prec
        return prec_fn

    def generate_data(self, K: int, cfg: SimConfig, rng: np.random.Generator) -> Dict:
        weights = np.asarray(cfg.prior_weights)
        theta_true = sample_prior(K, weights, rng)
        n_k = rng.integers(cfg.n_min, cfg.n_max + 1, size=K)

        theta_hat = np.zeros((K, DIM))
        obs_cov = np.zeros((K, DIM, DIM))
        oracle_obs_var = self.variance_fn(theta_true) / n_k[:, None, None]

        X_list = []
        for i in range(K):
            y, X = generate_multiclass_data(theta_true[i], n_k[i], rng)
            X_list.append(X)
            theta_hat_i, fisher_full = fit_multiclass_logistic(y, X)
            theta_hat[i] = theta_hat_i
            cov_i = _batch_inv(fisher_full[None, :, :], min_eig=1e-6, max_eig=1e6)[0]
            cov_i_scaled = cov_i / n_k[i]  # <--- [핵심 추가] MLE의 올바른 점근적 분산
            obs_cov[i] = _clip_spd(
                cov_i_scaled, 
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
