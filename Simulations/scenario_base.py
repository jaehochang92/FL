#!/usr/bin/env python3
"""
Base class and estimator implementations for federated learning simulations.

Scenario hierarchy:
    Scenario (ABC)
      ├─ QuadraticMeanScenario   — clients transmit sample means, quadratic variance
      ├─ LogisticScenario        — clients transmit multiclass logit models (c=6)
      └─ PoissonScenario         — clients transmit Poisson regression models

Estimators (all scenarios share the same four):
    VANEB   — Variance-Adaptive NPEB (proposed, atom-dependent covariance EM)
    NPEB    — Soloff et al. homoscedastic NPMLE (local sample-variance estimate)
    AdaMix  — Ozkara et al. parametric Gaussian mixture + MAP
    Oracle  — Oracle Bayes with known prior and variance function

All covariance handling is now full-matrix (no enforced diagonal mode). VANEB
recomputes atom-dependent covariances each EM iteration; baselines use the
client-provided fixed covariance estimates.
"""

from __future__ import annotations

import io
import time
import warnings
from abc import ABC, abstractmethod
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

from npeb import GLMixture
from npeb.GLMixture import mvn_pdf

# ============================================================================
# Configuration
# ============================================================================

DIM = 3  # Fixed parameter dimension for all scenarios

VARIANCE_BOUNDS = {"s_min": 0.01, "s_max": 100.0}


@dataclass
class SimConfig:
    """Shared configuration across all scenarios."""
    K: int = 200
    reps: int = 100
    n_min: int = 50
    n_max: int = 100          # typically set to 2 * n_min by the runner
    em_iters: int = 25
    adamix_components: int = 5
    adamix_iters: int = 20
    adamix_lr: float = 0.05
    seed: int = 20260311
    # Prior: mixture of 5 curves in R^3 (trefoil knot + helix + tilted ellipse +
    # figure-8 + Viviani curve) — more complex than three circles.
    prior_n_curves: int = 5
    prior_weights: List[float] = field(
        default_factory=lambda: [0.2, 0.2, 0.2, 0.2, 0.2]
    )


# ============================================================================
# Complex prior geometry in R^3
# ============================================================================

def _trefoil_knot(t: np.ndarray) -> np.ndarray:
    """Trefoil knot parametrised by t ∈ [0, 2π), centered at (-2, 0, 0)."""
    x = np.sin(t) + 2 * np.sin(2 * t)
    y = np.cos(t) - 2 * np.cos(2 * t)
    z = -np.sin(3 * t)
    return np.column_stack([x - 2, y, z])


def _helix(t: np.ndarray) -> np.ndarray:
    """Helix parametrised by t ∈ [0, 2π), centered at (2, 0, 0)."""
    x = 1.2 * np.cos(t)
    y = 1.2 * np.sin(t)
    z = t / np.pi - 1  # maps [0,2π) to [-1,1)
    return np.column_stack([x + 2, y, z])


def _tilted_ellipse(t: np.ndarray) -> np.ndarray:
    """Tilted ellipse in a plane not aligned with any axis, center (0, 2.5, 0)."""
    a, b = 1.5, 0.8
    x = a * np.cos(t)
    y = b * np.sin(t) * np.cos(np.pi / 5)
    z = b * np.sin(t) * np.sin(np.pi / 5)
    return np.column_stack([x, y + 2.5, z])


def _figure_eight(t: np.ndarray) -> np.ndarray:
    """Figure-eight (lemniscate of Gerono) in 3D, center (0, -2.5, 1)."""
    x = np.sin(t)
    y = np.sin(t) * np.cos(t)
    z = 0.5 * np.sin(2 * t)
    return np.column_stack([x, y - 2.5, z + 1])


def _viviani_curve(t: np.ndarray) -> np.ndarray:
    """Viviani's curve (intersection of sphere and cylinder), center (0, 0, -2.5)."""
    x = 1 + np.cos(t)
    y = np.sin(t)
    z = 2 * np.sin(t / 2)
    return np.column_stack([x, y, z - 2.5])


PRIOR_CURVES = [_trefoil_knot, _helix, _tilted_ellipse, _figure_eight, _viviani_curve]


def sample_prior(
    K: int,
    weights: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample K points from the 5-curve mixture prior in R^3."""
    n_curves = len(PRIOR_CURVES)
    comp = rng.choice(n_curves, size=K, p=weights)
    t = rng.uniform(0.0, 2 * np.pi, size=K)
    points = np.zeros((K, DIM))
    for c in range(n_curves):
        mask = comp == c
        if mask.any():
            points[mask] = PRIOR_CURVES[c](t[mask])
    return points


def generate_prior_atoms(atoms_per_curve: int = 100) -> np.ndarray:
    """Generate fixed atoms along all 5 curves for Oracle evaluation."""
    t = np.linspace(0, 2 * np.pi, atoms_per_curve, endpoint=False)
    atoms = [curve(t) for curve in PRIOR_CURVES]
    return np.vstack(atoms)


# ============================================================================
# Utility functions
# ============================================================================

def rmse(theta_hat: np.ndarray, theta: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((theta_hat - theta) ** 2, axis=1))))


def _sanitize(theta: np.ndarray, clip: float = 1e6) -> np.ndarray:
    out = np.nan_to_num(theta, nan=0.0, posinf=clip, neginf=-clip)
    return np.clip(out, -clip, clip)


def _symmetrize(mat: np.ndarray) -> np.ndarray:
    return 0.5 * (mat + np.swapaxes(mat, -1, -2))


def _clip_spd(mat: np.ndarray, min_eig: float = 1e-6, max_eig: float = 1e6) -> np.ndarray:
    """Project symmetric matrix/matrices to SPD with eigenvalue clipping."""
    mat = _symmetrize(mat)
    vals, vecs = np.linalg.eigh(mat)
    vals_clipped = np.clip(vals, min_eig, max_eig)
    clipped = (vecs * vals_clipped[..., None, :]) @ np.swapaxes(vecs, -1, -2)
    return _symmetrize(clipped)


def _batch_inv(mat: np.ndarray, min_eig: float = 1e-6, max_eig: float = 1e6) -> np.ndarray:
    """Inverse of SPD matrices with eigen clipping for stability."""
    mat_pd = _clip_spd(mat, min_eig=min_eig, max_eig=max_eig)
    vals, vecs = np.linalg.eigh(mat_pd)
    vals_inv = 1.0 / np.clip(vals, min_eig, max_eig)
    inv = (vecs * vals_inv[..., None, :]) @ np.swapaxes(vecs, -1, -2)
    return _symmetrize(inv)


def _batch_logdet(mat: np.ndarray, min_eig: float = 1e-12) -> np.ndarray:
    """Log-determinant of SPD matrices with eigen clipping to avoid -inf."""
    mat_pd = _clip_spd(mat, min_eig=min_eig)
    vals = np.linalg.eigvalsh(mat_pd)
    vals = np.clip(vals, min_eig, None)
    return np.sum(np.log(vals), axis=-1)


# ============================================================================
# Estimators
# ============================================================================


def vaneb_estimator(
    x: np.ndarray,
    n_k: np.ndarray,
    sigma2_fn: Callable[[np.ndarray], np.ndarray],
    em_iters: int,
) -> Tuple[np.ndarray, float]:
    """VANEB: Variance-Adaptive NPEB (proposed).

    Heteroskedastic NPMLE with atom-dependent covariance recomputation.
    """
    from sklearn.preprocessing import normalize as sk_normalize
    k_obs, d = x.shape

    try:
        t0 = time.time()
        sigma2_init = sigma2_fn(x)                          # (k, d, d)
        prec_init = n_k[:, None, None] * _batch_inv(sigma2_init)

        # Use GLMixture for initialization (full covariance); subsequent EM is manual
        model = GLMixture(prec_type="general", homoscedastic=False)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            model.fit(x, prec_init, max_iter_em=0,
                      weight_thresh=0.0, solver="mosek",
                      row_condition=True, score_every=None)

        atoms = model.atoms.copy()
        weights = model.weights.copy()

        for _ in range(em_iters):
            sigma2_atoms = sigma2_fn(atoms)                 # (m, d, d)
            sigma2_atoms = _clip_spd(sigma2_atoms, min_eig=1e-8, max_eig=1e8)
            inv_atoms = _batch_inv(sigma2_atoms, min_eig=1e-8, max_eig=1e8)

            m = atoms.shape[0]
            diffs = x[:, None, :] - atoms[None, :, :]      # (k, m, d)
            prec_tensor = n_k[:, None, None, None] * inv_atoms[None, :, :, :]  # (k,m,d,d)

            mahal = np.einsum('kmd,kmde,kme->km', diffs, prec_tensor, diffs)
            log_det = _batch_logdet(prec_tensor)
            log_probs = -0.5 * (mahal + d * np.log(2 * np.pi) - log_det)

            log_probs_w = log_probs + np.log(np.maximum(weights, 1e-300))[None, :]
            log_probs_w -= log_probs_w.max(axis=1, keepdims=True)
            resp = np.exp(log_probs_w)
            resp /= np.maximum(resp.sum(axis=1, keepdims=True), 1e-300)

            # M-step for atom locations with full precision
            prec_x = np.einsum('kmde,kd->kme', prec_tensor, x)
            num = np.einsum('km,kme->me', resp, prec_x)                 # (m, d)
            prec_sum = np.einsum('km,kmde->mde', resp, prec_tensor)     # (m, d, d)
            prec_sum = _clip_spd(prec_sum, min_eig=1e-8, max_eig=1e8)

            new_atoms = np.zeros_like(atoms)
            for j in range(m):
                new_atoms[j] = np.linalg.solve(prec_sum[j], num[j])
            atoms = new_atoms

            Nk = resp.sum(axis=0)
            weights = Nk / np.maximum(Nk.sum(), 1e-12)

        # Final posterior mean
        sigma2_final = _clip_spd(sigma2_fn(atoms), min_eig=1e-8, max_eig=1e8)
        inv_final = _batch_inv(sigma2_final, min_eig=1e-8, max_eig=1e8)
        diffs = x[:, None, :] - atoms[None, :, :]
        prec_final = n_k[:, None, None, None] * inv_final[None, :, :, :]
        mahal = np.einsum('kmd,kmde,kme->km', diffs, prec_final, diffs)
        log_det = _batch_logdet(prec_final)
        log_probs = -0.5 * (mahal + d * np.log(2 * np.pi) - log_det)
        log_probs_w = log_probs + np.log(np.maximum(weights, 1e-300))[None, :]
        log_probs_w -= log_probs_w.max(axis=1, keepdims=True)
        resp = np.exp(log_probs_w)
        resp /= np.maximum(resp.sum(axis=1, keepdims=True), 1e-300)
        theta_hat = resp @ atoms

        return theta_hat, time.time() - t0
    except Exception as exc:
        raise RuntimeError(f"VANEB fitting failed: {exc}") from exc


def npeb_estimator(
    x: np.ndarray,
    n_k: np.ndarray,
    obs_var: np.ndarray,
    em_iters: int,
) -> Tuple[np.ndarray, float]:
    """NPEB: Soloff et al. homoscedastic NPMLE with local sample-variance estimates.

    Uses client-reported covariance estimates (fixed, not recomputed at atoms).
    """
    prec_fixed = _batch_inv(obs_var, min_eig=1e-8, max_eig=1e8)

    try:
        t0 = time.time()
        model = GLMixture(prec_type="general", homoscedastic=False)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            model.fit(x, prec_fixed, max_iter_em=em_iters,
                      weight_thresh=1e-4, solver="mosek",
                      row_condition=True, score_every=None)
        theta_hat = model.posterior_mean(x, prec_fixed)
        return theta_hat, time.time() - t0
    except Exception as exc:
        raise RuntimeError(f"NPEB (Soloff) fitting failed: {exc}") from exc


def adamix_estimator(
    x: np.ndarray,
    obs_var: np.ndarray,
    cfg: SimConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """AdaMix: parametric Gaussian mixture + MAP shrinkage."""
    k, _ = x.shape
    n_components = max(1, min(cfg.adamix_components, k))
    theta = _sanitize(x.copy())

    gmm = _fit_gmm(theta, n_components, int(rng.integers(1, 2**31 - 1)), 2)

    for _ in range(cfg.adamix_iters):
        resp, var = _gmm_resp_and_scales(theta, gmm)
        # Use diagonal of the provided covariance for a lightweight MAP update.
        obs_var_diag = np.diagonal(obs_var, axis1=1, axis2=2)
        inv_obs = 1.0 / np.maximum(obs_var_diag, 1e-12)
        prior_prec = np.zeros((k, 1))
        prior_loc = np.zeros_like(theta)
        for ell in range(gmm.n_components):
            cp = resp[:, [ell]] / var[ell]
            prior_prec += cp
            prior_loc += cp * gmm.means_[ell]
        numer = inv_obs * x + prior_loc
        denom = inv_obs + prior_prec
        theta_map = numer / np.maximum(denom, 1e-12)
        theta = _sanitize((1 - cfg.adamix_lr) * theta + cfg.adamix_lr * theta_map)
        gmm = _fit_gmm(theta, n_components, int(rng.integers(1, 2**31 - 1)), 1)

    return theta


def oracle_estimator(
    x: np.ndarray,
    atoms: np.ndarray,
    obs_var: np.ndarray,
) -> np.ndarray:
    """Oracle Bayes posterior mean with known prior atoms and true client variance.

    The benchmark uses the true prior support together with the true client-specific
    observation variance for each replicated client summary statistic.
    """
    _, d = x.shape
    atoms = np.asarray(atoms)
    obs_var = _clip_spd(obs_var, min_eig=1e-10, max_eig=1e10)
    inv_cov = _batch_inv(obs_var, min_eig=1e-10, max_eig=1e10)
    diff = x[:, None, :] - atoms[None, :, :]
    mahal = np.einsum('kmd,kde,kme->km', diff, inv_cov, diff)
    log_det = _batch_logdet(obs_var[:, None, :, :], min_eig=1e-10)
    log_w = -0.5 * (mahal + log_det + d * np.log(2 * np.pi))
    log_norm = logsumexp(log_w, axis=1, keepdims=True)
    w = np.exp(log_w - log_norm)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    row_sums = w.sum(axis=1, keepdims=True)
    bad = (~np.isfinite(row_sums)) | (row_sums <= 0)
    w = w / np.maximum(row_sums, 1e-300)
    if np.any(bad):
        w[bad[:, 0], :] = 1.0 / atoms.shape[0]
    theta_hat = np.einsum("kj,jd->kd", w, atoms)
    return np.nan_to_num(theta_hat, nan=0.0, posinf=0.0, neginf=0.0)


# ============================================================================
# GMM helpers for AdaMix
# ============================================================================

def _gmm_resp_and_scales(theta, gmm):
    theta_c = _sanitize(theta)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        resp = gmm.predict_proba(theta_c)
    resp = np.nan_to_num(resp, nan=0.0, posinf=0.0, neginf=0.0)
    row = resp.sum(axis=1, keepdims=True)
    bad = (~np.isfinite(row)) | (row <= 0)
    resp = resp / np.maximum(row, 1e-300)
    if np.any(bad):
        resp[bad[:, 0], :] = 1.0 / gmm.n_components
    cov = gmm.covariances_
    if np.ndim(cov) == 0:
        var = np.full(gmm.n_components, float(cov))
    elif np.ndim(cov) == 1:
        var = np.maximum(cov.astype(float), 1e-8)
    elif np.ndim(cov) == 2:
        var = np.maximum(np.mean(cov, axis=1), 1e-8)
    else:
        var = np.maximum(np.trace(cov, axis1=1, axis2=2) / cov.shape[-1], 1e-8)
    return resp, var


def _fit_gmm(theta, n_components, random_state, n_init):
    theta_c = _sanitize(theta)
    uc = np.unique(theta_c, axis=0).shape[0]
    mc = max(1, min(n_components, uc))
    last_exc = None
    for comp in range(mc, 0, -1):
        gmm = GaussianMixture(
            n_components=comp, covariance_type="spherical",
            reg_covar=1e-5, random_state=random_state, max_iter=200, n_init=n_init,
        )
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=RuntimeWarning)
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                gmm.fit(theta_c)
            return gmm
        except Exception as exc:
            last_exc = exc
    fallback = GaussianMixture(
        n_components=1, covariance_type="spherical",
        reg_covar=1e-4, random_state=random_state, max_iter=100, n_init=1,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        fallback.fit(theta_c)
    return fallback


# ============================================================================
# Abstract Scenario
# ============================================================================

class Scenario(ABC):
    """Base class for a federated learning simulation scenario."""

    name: str = "base"
    prior_scale: float = 1.0  # subclasses override when prior atoms are scaled

    @abstractmethod
    def generate_data(
        self, K: int, cfg: SimConfig, rng: np.random.Generator
    ) -> Dict:
        """Generate federated data for K clients.

        Returns a dict with at least:
            theta_true: (K, 3) true parameters
            x:          (K, 3) summary statistics (the "observations")
                obs_var:    (K, 3, 3) client-specific observation covariance
            n_k:        (K,)   sample sizes
        Subclasses may store extra fields (e.g. X_list, y_list for GLMs).
        """

    @abstractmethod
    def variance_fn(self, theta: np.ndarray) -> np.ndarray:
        """Population covariance function Σ(θ) → (·, 3, 3) SPD matrices."""

    def run_one(
        self, K: int, rep: int, cfg: SimConfig, rng: np.random.Generator
    ) -> Dict:
        """Run a single replication and return RMSE dict."""
        data = self.generate_data(K, cfg, rng)
        theta_true = data["theta_true"]
        x = data["x"]
        obs_var = data["obs_var"]
        n_k = data["n_k"]
        oracle_obs_var = data.get("oracle_obs_var", obs_var)

        # Oracle — use the same prior_scale applied in generate_data so atoms
        # are aligned with the actual theta_true support.
        atoms = generate_prior_atoms(atoms_per_curve=200) * self.prior_scale
        theta_oracle = oracle_estimator(x, atoms, oracle_obs_var)

        # VANEB (proposed)
        theta_vaneb, vaneb_time = vaneb_estimator(
            x, n_k, self.variance_fn, cfg.em_iters
        )

        # NPEB (Soloff — uses local obs_var, fixed)
        theta_npeb, npeb_time = npeb_estimator(
            x, n_k, obs_var, cfg.em_iters
        )

        # AdaMix
        try:
            theta_adamix = adamix_estimator(x, obs_var, cfg, rng)
        except Exception:
            theta_adamix = np.full_like(x, np.nan)

        return {
            "scenario": self.name,
            "K": K,
            "rep": rep,
            "n_min": cfg.n_min,
            "rmse_oracle": rmse(theta_oracle, theta_true),
            "rmse_vaneb": rmse(theta_vaneb, theta_true),
            "rmse_npeb": rmse(theta_npeb, theta_true),
            "rmse_adamix": rmse(theta_adamix, theta_true),
            "vaneb_time": vaneb_time,
            "npeb_time": npeb_time,
        }
