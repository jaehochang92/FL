import argparse
import io
import json
import time
import warnings
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from tqdm.auto import tqdm
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

from npeb import GLMixture
from npeb.GLMixture import mvn_pdf, solve_weights_mosek


# ============================================================================
# Predefined variance functions: Sigma(theta) -> component-wise variances
# ============================================================================

# Variance bounds for theoretical compliance (Assumption: s_min <= Sigma <= s_max)
VARIANCE_BOUNDS = {"s_min": 0.01, "s_max": 30.0}


def variance_quadratic(theta: np.ndarray) -> np.ndarray:
    """Quadratic variance: Sigma(theta) = diag(theta^2), clipped to [s_min, s_max]."""
    return np.clip(theta ** 2, VARIANCE_BOUNDS["s_min"], VARIANCE_BOUNDS["s_max"])


def variance_linear(theta: np.ndarray) -> np.ndarray:
    """Linear variance: Sigma(theta) = diag(|theta|), clipped to [s_min, s_max]."""
    return np.clip(np.abs(theta), VARIANCE_BOUNDS["s_min"], VARIANCE_BOUNDS["s_max"])


def variance_constant(theta: np.ndarray) -> np.ndarray:
    """Constant variance: Sigma(theta) = diag(1)."""
    return np.ones_like(theta)


def variance_sqrt(theta: np.ndarray) -> np.ndarray:
    """Square-root variance: Sigma(theta) = diag(sqrt(|theta|)), clipped to [s_min, s_max]."""
    return np.clip(np.sqrt(np.abs(theta)), VARIANCE_BOUNDS["s_min"], VARIANCE_BOUNDS["s_max"])


def variance_poisson(theta: np.ndarray, feature_scale: float = 0.7) -> np.ndarray:
    """Poisson Fisher-inverse variance (population).

    For Poisson regression y ~ Poisson(exp(X^T theta)) with X ~ N(0, sx^2 I),
    the population per-observation Fisher Information diagonal is:

        F_d(theta) = (sx^2 + sx^4 * theta_d^2) * exp(sx^2 * ||theta||^2 / 2)

    This function returns sigma2(theta) = 1 / F(theta), the per-observation
    variance that enters the compound-decision model:

        theta_hat_k ~ N(theta_k, sigma2(theta_k) / n_k)

    The exponential growth of Fisher with ||theta|| creates strong
    heteroskedasticity: clients at the periphery of the prior (large ||theta||)
    have much lower variance (more precise MLEs) than clients near the origin.

    Args:
        theta: (m, d) or (d,) parameter values
        feature_scale: std of regression features (default: 0.7)

    Returns:
        (m, d) or (d,) per-observation variances, clipped to [s_min, s_max]
    """
    sx2 = feature_scale ** 2
    theta_sq = theta ** 2
    if theta.ndim == 1:
        norm_sq = np.sum(theta_sq)
    else:
        norm_sq = np.sum(theta_sq, axis=1, keepdims=True)

    fisher_diag = (sx2 + sx2**2 * theta_sq) * np.exp(sx2 * norm_sq / 2)
    sigma2 = 1.0 / np.maximum(fisher_diag, 1e-12)
    return np.clip(sigma2, VARIANCE_BOUNDS["s_min"], VARIANCE_BOUNDS["s_max"])


VARIANCE_FUNCTIONS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "quadratic": variance_quadratic,
    "linear": variance_linear,
    "constant": variance_constant,
    "sqrt": variance_sqrt,
    "poisson": variance_poisson,
}


# ============================================================================
# Data structures and utilities
# ============================================================================


# ============================================================================
# Data structures and utilities
# ============================================================================

@dataclass
class SimConfig:
    k_values: List[int]
    reps: int
    dim: int
    radius: float
    n_min: int
    n_max: int
    em_iters: int
    adamix_components: int
    adamix_iters: int
    adamix_lr: float
    prior_centers: List[List[float]]
    prior_radii: List[float]
    prior_weights: List[float]
    random_seed: int


def sample_three_circle_prior(
    k: int,
    dim: int,
    centers: np.ndarray,
    radii: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample from mixture of spheres in d dimensions.
    
    For dim=2: samples from circles (as before)
    For dim>2: samples from spheres (uniform on sphere surface)
    """
    m = centers.shape[0]
    comp = rng.choice(m, size=k, p=weights)
    
    if dim == 2:
        # Original 2D circle sampling
        ang = rng.uniform(0.0, 2.0 * np.pi, size=k)
        unit = np.column_stack([np.cos(ang), np.sin(ang)])
    else:
        # Sample uniformly on unit sphere in d dimensions
        # Use Gaussian sampling + normalization
        unit = rng.standard_normal(size=(k, dim))
        unit = unit / np.linalg.norm(unit, axis=1, keepdims=True)
    
    return centers[comp] + radii[comp][:, None] * unit


def generate_uniform_prior_support(
    centers: np.ndarray,
    radii: np.ndarray,
    dim: int = 2,
    atoms_per_circle: int = 50,
) -> np.ndarray:
    """Generate fixed discrete uniform prior support on the spheres.
    
    This creates a fixed set of atoms on each sphere, independent of K.
    Scales atoms appropriately for higher dimensions.
    
    For dim=2: creates atoms on circles (via angles)
    For dim>2: creates atoms uniformly distributed on sphere surfaces
    
    Atoms per sphere scales as: 50 * ceil(sqrt(dim))
    This ensures roughly constant support density across dimensions.
    
    Args:
        centers: (m, d) array of sphere centers
        radii: (m,) array of sphere radii
        dim: dimension of space (inferred from centers if not provided)
        atoms_per_circle: base atoms per sphere (will scale by sqrt(dim))
    
    Returns:
        (m * scaled_atoms, d) array of atoms on spheres
    """
    m = centers.shape[0]
    actual_dim = centers.shape[1]
    
    # Scale atoms by dimension to maintain consistent support density
    scaled_atoms_per_circle = atoms_per_circle * max(1, int(np.ceil(np.sqrt(actual_dim))))
    total_atoms = m * scaled_atoms_per_circle
    atoms = np.zeros((total_atoms, actual_dim))
    
    rng = np.random.RandomState(42)  # Fixed seed for reproducibility
    
    for i in range(m):
        if actual_dim == 2:
            # 2D: equally-spaced angles on circle
            angles = np.linspace(0, 2.0 * np.pi, scaled_atoms_per_circle, endpoint=False)
            unit = np.column_stack([np.cos(angles), np.sin(angles)])
        else:
            # d-D: sample uniformly on sphere surface via Gaussian normalization
            unit = rng.standard_normal(size=(scaled_atoms_per_circle, actual_dim))
            unit = unit / np.linalg.norm(unit, axis=1, keepdims=True)
        
        atoms[i*scaled_atoms_per_circle:(i+1)*scaled_atoms_per_circle] = (
            centers[i] + radii[i] * unit
        )
    
    return atoms


def generate_data(
    k: int,
    cfg: SimConfig,
    rng: np.random.Generator,
    sigma2_fn: Callable[[np.ndarray], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic federated data with atom-dependent variance.
    
    Args:
        k: number of clients
        cfg: simulation configuration
        rng: random number generator
        sigma2_fn: variance function Sigma(theta) -> component-wise variances
    
    Returns:
        (theta, x, obs_var, n_k): true parameters, observations, variances, sample sizes
    """
    theta = sample_three_circle_prior(
        k,
        cfg.dim,
        np.array(cfg.prior_centers, dtype=float),
        np.array(cfg.prior_radii, dtype=float),
        np.array(cfg.prior_weights, dtype=float),
        rng,
    )
    n_k = rng.integers(cfg.n_min, cfg.n_max + 1, size=k)
    sigma2 = sigma2_fn(theta)
    obs_var = sigma2 / n_k[:, None]
    x = theta + rng.normal(size=(k, cfg.dim)) * np.sqrt(obs_var)
    return theta, x, obs_var, n_k


def rmse(theta_hat: np.ndarray, theta: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((theta_hat - theta) ** 2, axis=1))))


def _gmm_resp_and_scales(theta: np.ndarray, gmm: GaussianMixture) -> Tuple[np.ndarray, np.ndarray]:
    theta_clean = _sanitize_theta(theta)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        resp = gmm.predict_proba(theta_clean)
    resp = np.nan_to_num(resp, nan=0.0, posinf=0.0, neginf=0.0)
    row_sum = np.sum(resp, axis=1, keepdims=True)
    bad = (~np.isfinite(row_sum)) | (row_sum <= 0)
    resp = resp / np.maximum(row_sum, 1e-300)
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


def _sanitize_theta(theta: np.ndarray, clip_abs: float = 1e6) -> np.ndarray:
    theta_clean = np.nan_to_num(theta, nan=0.0, posinf=clip_abs, neginf=-clip_abs)
    return np.clip(theta_clean, -clip_abs, clip_abs)


def _fit_gmm_safe(
    theta: np.ndarray,
    n_components: int,
    random_state: int,
    n_init: int,
) -> GaussianMixture:
    theta_clean = _sanitize_theta(theta)
    unique_count = np.unique(theta_clean, axis=0).shape[0]
    max_comp = max(1, min(n_components, unique_count))

    last_exc: Optional[Exception] = None
    for comp in range(max_comp, 0, -1):
        gmm = GaussianMixture(
            n_components=comp,
            covariance_type="spherical",
            reg_covar=1e-5,
            random_state=random_state,
            max_iter=200,
            n_init=n_init,
        )
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=RuntimeWarning)
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                gmm.fit(theta_clean)
            return gmm
        except Exception as exc:
            last_exc = exc
            continue

    fallback = GaussianMixture(
        n_components=1,
        covariance_type="spherical",
        reg_covar=1e-4,
        random_state=random_state,
        max_iter=100,
        n_init=1,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        fallback.fit(theta_clean)
    return fallback


def adamix_estimator(
    x: np.ndarray,
    obs_var: np.ndarray,
    cfg: SimConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    k, _ = x.shape
    n_components = max(1, min(cfg.adamix_components, k))
    theta = _sanitize_theta(x.copy())

    gmm = _fit_gmm_safe(
        theta,
        n_components=n_components,
        random_state=int(rng.integers(1, 2**31 - 1)),
        n_init=2,
    )

    for _ in range(cfg.adamix_iters):
        resp, var = _gmm_resp_and_scales(theta, gmm)

        inv_obs = 1.0 / np.maximum(obs_var, 1e-12)
        prior_prec = np.zeros((k, 1))
        prior_loc = np.zeros_like(theta)
        for ell in range(gmm.n_components):
            comp_prec = resp[:, [ell]] / var[ell]
            prior_prec += comp_prec
            prior_loc += comp_prec * gmm.means_[ell]

        numer = inv_obs * x + prior_loc
        denom = inv_obs + prior_prec
        theta_map = numer / np.maximum(denom, 1e-12)
        theta = _sanitize_theta((1.0 - cfg.adamix_lr) * theta + cfg.adamix_lr * theta_map)

        gmm = _fit_gmm_safe(
            theta,
            n_components=n_components,
            random_state=int(rng.integers(1, 2**31 - 1)),
            n_init=1,
        )

    return theta


def oracle_posterior_mean(
    theta_atoms: np.ndarray,
    x: np.ndarray,
    n_k: np.ndarray,
    sigma2_fn: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Oracle Bayes posterior mean with atom-dependent variance.

    For heteroskedastic models x_i | theta_i ~ N(theta_i, Sigma(theta_i)/n_i),
    the oracle posterior weight for atom j given observation x_i is:

        w_ij  ∝  N(x_i; atom_j,  Sigma(atom_j) / n_i)

    Previously the code used Sigma(theta_true)/n_i for ALL atoms, which is
    incorrect and causes catastrophic weight collapse in high dimensions
    when Sigma varies strongly with theta.
    """
    k, d = x.shape
    J = theta_atoms.shape[0]

    # Compute variance at each atom location: (J, d)
    sigma2_atoms = sigma2_fn(theta_atoms)

    # Per-client, per-atom observation variance: (K, J, d)
    # obs_var_{i,j,l} = sigma2(atom_j)_l / n_i
    obs_var_3d = sigma2_atoms[None, :, :] / n_k[:, None, None]
    obs_var_safe = np.maximum(obs_var_3d, 1e-12)

    diff = x[:, None, :] - theta_atoms[None, :, :]          # (K, J, d)
    quad = np.sum((diff ** 2) / obs_var_safe, axis=2)        # (K, J)
    log_det = np.sum(np.log(2.0 * np.pi * obs_var_safe), axis=2)  # (K, J)
    log_w = -0.5 * (quad + log_det)                          # (K, J)

    log_norm = logsumexp(log_w, axis=1, keepdims=True)
    w = np.exp(log_w - log_norm)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    row_sums = np.sum(w, axis=1, keepdims=True)
    bad = (~np.isfinite(row_sums)) | (row_sums <= 0)
    w = w / np.maximum(row_sums, 1e-300)
    if np.any(bad):
        w[bad[:, 0], :] = 1.0 / J
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    theta_atoms_safe = np.nan_to_num(theta_atoms, nan=0.0, posinf=0.0, neginf=0.0)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        theta_post = w @ theta_atoms_safe
    return np.nan_to_num(theta_post, nan=0.0, posinf=0.0, neginf=0.0)


def oracle_posterior_mean_local_mc(
    x: np.ndarray,
    n_k: np.ndarray,
    sigma2_fn: Callable[[np.ndarray], np.ndarray],
    centers: np.ndarray,
    radii: np.ndarray,
    weights: np.ndarray,
    n_mc: int = 5000,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Oracle posterior mean via local Monte Carlo on sphere surfaces.

    For high-dimensional sphere priors, a fixed global grid cannot cover the
    sphere surface densely enough.  Instead, for each client *i* and each
    sphere *c*, we:
      1. Find the MAP projection of x_i onto sphere c.
      2. Sample n_mc atoms on a tangent-space cap around that projection,
         which concentrates MC mass where the posterior is.
      3. Evaluate the heteroskedastic likelihood at every sample.
      4. Combine across spheres with the prior mixture weights.

    Falls back to the global-grid oracle in 2-D where grids are dense enough.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    K, d = x.shape
    n_spheres = centers.shape[0]
    theta_post = np.zeros((K, d))

    for i in range(K):
        xi = x[i]    # (d,)
        ni = n_k[i]  # scalar

        all_atoms = []
        all_log_prior = []   # log prior weight per atom

        for c in range(n_spheres):
            mu_c = centers[c]          # (d,)
            r_c = radii[c]
            w_c = weights[c]

            # --- MAP projection onto sphere c ---
            direction = xi - mu_c
            direction_norm = np.linalg.norm(direction)
            if direction_norm < 1e-12:
                u_star = np.zeros(d); u_star[0] = 1.0
            else:
                u_star = direction / direction_norm   # unit vector toward x

            # --- tangent-space perturbation scale ---
            # Posterior spread on the sphere ≈ sqrt(sigma2 / n) / r
            theta_star = mu_c + r_c * u_star
            sigma2_star = sigma2_fn(theta_star[None, :])[0]          # (d,)
            perturb_scale = np.sqrt(np.mean(sigma2_star / ni)) / r_c
            perturb_scale = np.clip(perturb_scale, 0.01, 2.0)

            # --- sample atoms on cap around MAP ---
            z = rng.normal(size=(n_mc, d)) * perturb_scale
            z -= np.outer(z @ u_star, u_star)          # project out radial
            u_samples = u_star[None, :] + z
            u_samples /= np.linalg.norm(u_samples, axis=1, keepdims=True)
            atoms_c = mu_c[None, :] + r_c * u_samples  # (n_mc, d)

            all_atoms.append(atoms_c)
            # Uniform prior on each sphere → log_prior = log(w_c) (constant within sphere)
            all_log_prior.append(np.full(n_mc, np.log(max(w_c, 1e-300))))

        atoms = np.vstack(all_atoms)              # (n_spheres * n_mc, d)
        log_prior = np.concatenate(all_log_prior) # (n_spheres * n_mc,)

        # --- heteroskedastic likelihood at each atom ---
        sigma2_atoms = sigma2_fn(atoms)                            # (J, d)
        obs_var = sigma2_atoms / ni                                # (J, d)
        obs_var_safe = np.maximum(obs_var, 1e-12)
        diff = xi[None, :] - atoms                                 # (J, d)
        quad = np.sum(diff**2 / obs_var_safe, axis=1)             # (J,)
        log_det = np.sum(np.log(2.0 * np.pi * obs_var_safe), axis=1)  # (J,)
        log_lik = -0.5 * (quad + log_det)                          # (J,)

        log_w = log_lik + log_prior
        log_norm = logsumexp(log_w)
        w = np.exp(log_w - log_norm)
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        wsum = w.sum()
        if wsum > 0:
            w /= wsum
        else:
            w[:] = 1.0 / len(w)

        theta_post[i] = w @ atoms

    return theta_post


def glmixture_npeb_estimator(
    x: np.ndarray,
    n_k: np.ndarray,
    sigma2_fn: Callable[[np.ndarray], np.ndarray],
    em_iters: int,
    quiet_solver: bool,
) -> Tuple[np.ndarray, float]:
    """Fit NPEB estimator with atom-dependent heteroskedastic covariance.
    
    Implements Algorithm 2 with a custom EM loop that recomputes
    covariances at new atom locations after each M-step:
    
        1. Initial NPMLE solve (conic program) with prec at observation locations
        2. For each EM iteration:
           a. E-step: compute responsibilities r_kj
           b. M-step: update atom locations a_j
           c. Recompute precisions: prec^(k)_j = n^(k) / sigma2_fn(a_j)
           d. Update mixture weights w_j
    
    This atom-covariance coupling is essential for consistency with the
    modified Tweedie formula in the heteroskedastic setting.
    
    Args:
        x: (k, d) observations
        n_k: (k,) local sample sizes for each client
        sigma2_fn: variance function Sigma(theta) -> component-wise variances
        em_iters: number of EM iterations for atom refinement
        quiet_solver: whether to suppress MOSEK output
    
    Returns:
        (theta_hat, runtime): posterior means and elapsed time
    """
    from sklearn.preprocessing import normalize as sk_normalize

    k_obs, d = x.shape

    # --- helper: build (k, d) precision from atoms (m, d) -----------------
    def _prec_at_atoms(atoms: np.ndarray) -> np.ndarray:
        """For each client k and atom j, precision = n^(k) / sigma2(a_j).
        
        Returns (k_obs, d) precision where each row is computed at the
        observation location (used for kernel evaluation with x).
        For the M-step atom update we need per-atom precision, handled inline.
        """
        # We store the per-atom variance (m, d) and broadcast with n_k
        return None  # not used directly; see inline code

    try:
        t0 = time.time()

        # ============================================================
        # Step 1: Initial NPMLE solve (discretized) at observation atoms
        # ============================================================
        # Initial atoms = observations; prec^(k) = n^(k) / sigma2(x^(k))
        sigma2_init = sigma2_fn(x)  # (k, d)
        prec_init = n_k[:, None] / np.maximum(sigma2_init, 1e-12)  # (k, d)

        # Use GLMixture to solve the initial conic program (no EM iterations)
        model = GLMixture(prec_type="diagonal", homoscedastic=False)
        
        if quiet_solver:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                _ = model.fit(x, prec_init, max_iter_em=0,
                              weight_thresh=0.0, solver="mosek",
                              row_condition=True, score_every=None)
        else:
            _ = model.fit(x, prec_init, max_iter_em=0,
                          weight_thresh=0.0, solver="mosek",
                          row_condition=True, score_every=None)

        # Extract all atoms and weights from initial solve
        atoms = model.atoms.copy()  # (k_obs, d) - all observations kept as atoms
        weights = model.weights.copy()

        # ============================================================
        # Step 2: Custom EM with dynamic precision recomputation
        # ============================================================
        for _t in range(em_iters):
            # --- Compute precision at current atoms for each client ---
            # sigma2_atoms[j, :] = sigma2_fn(a_j)  shape (m, d)
            sigma2_atoms = sigma2_fn(atoms)  # (m, d)
            sigma2_atoms = np.maximum(sigma2_atoms, 1e-12)

            # For the kernel K[k, j] = phi(x^(k); a_j, Sigma^(k)_j)
            # where Sigma^(k)_j = sigma2(a_j) / n^(k)
            # precision = n^(k) / sigma2(a_j)
            # This is a (k, m, d) tensor conceptually, but we compute
            # log-probabilities directly.
            m = atoms.shape[0]
            # diffs: (k, m, d)
            diffs = x[:, np.newaxis, :] - atoms[np.newaxis, :, :]
            # prec_tensor[k, j, d_] = n_k[k] / sigma2_atoms[j, d_]
            prec_tensor = n_k[:, np.newaxis, np.newaxis] / sigma2_atoms[np.newaxis, :, :]  # (k, m, d)
            # Mahalanobis: sum over d of prec * diff^2 -> (k, m)
            mahal = np.sum(prec_tensor * diffs**2, axis=2)
            # log-det of precision: sum_d log(prec) -> (k, m)
            log_det = np.sum(np.log(prec_tensor), axis=2)
            log_probs = -0.5 * (mahal + d * np.log(2 * np.pi) - log_det)  # (k, m)

            # --- E-step: responsibilities ---
            log_probs_w = log_probs + np.log(np.maximum(weights, 1e-300))[np.newaxis, :]
            log_probs_w -= log_probs_w.max(axis=1, keepdims=True)  # row conditioning
            resp = np.exp(log_probs_w)
            resp /= np.maximum(resp.sum(axis=1, keepdims=True), 1e-300)  # (k, m)

            # --- M-step: update atom locations ---
            # a_j = (sum_k r_kj * prec^(k)_j * x^(k)) / (sum_k r_kj * prec^(k)_j)
            # prec^(k)_j = n_k[k] / sigma2_atoms[j, :]  (scalar per dimension)
            # numerator: sum_k r_kj * prec_kj_d * x_kd  for each j, d
            # resp: (k, m), prec_tensor: (k, m, d), x: (k, d)
            precX = prec_tensor * x[:, np.newaxis, :]  # (k, m, d)
            num = np.einsum('km,kmd->md', resp, precX)   # (m, d)
            dnm = np.einsum('km,kmd->md', resp, prec_tensor)  # (m, d)
            atoms = num / np.maximum(dnm, 1e-12)  # (m, d)

            # --- M-step: recompute covariances at new atom locations ---
            # (This happens at the top of the next iteration when
            #  sigma2_atoms = sigma2_fn(atoms) is called)

            # --- M-step: update mixture weights ---
            Nk = resp.sum(axis=0)  # (m,)
            weights = Nk / np.maximum(Nk.sum(), 1e-12)

        # ============================================================
        # Step 3: Posterior mean with final atom-dependent covariances
        # ============================================================
        # Recompute precision at final atoms for posterior mean
        sigma2_final = np.maximum(sigma2_fn(atoms), 1e-12)  # (m, d)
        diffs = x[:, np.newaxis, :] - atoms[np.newaxis, :, :]
        prec_final = n_k[:, np.newaxis, np.newaxis] / sigma2_final[np.newaxis, :, :]
        mahal = np.sum(prec_final * diffs**2, axis=2)
        log_det = np.sum(np.log(prec_final), axis=2)
        log_probs = -0.5 * (mahal + d * np.log(2 * np.pi) - log_det)

        log_probs_w = log_probs + np.log(np.maximum(weights, 1e-300))[np.newaxis, :]
        log_probs_w -= log_probs_w.max(axis=1, keepdims=True)
        resp = np.exp(log_probs_w)
        resp /= np.maximum(resp.sum(axis=1, keepdims=True), 1e-300)

        theta_hat = resp @ atoms  # (k, d)

        return theta_hat, time.time() - t0
    except Exception as exc:
        raise RuntimeError(f"GLMixture NPEB fitting failed: {exc}") from exc

def soloff_estimator(
    x: np.ndarray,
    n_k: np.ndarray,
    em_iters: int,
    quiet_solver: bool,
) -> Tuple[np.ndarray, float]:
    """Fit standard NPMLE with fixed parameter-independent covariance (Soloff et al.).
    
    Model: x^(k) ~ N(theta, Sigma/n^(k))
    where Sigma is fixed and does NOT depend on theta (unlike NPEB).
    
    Key differences from NPEB:
      - Sigma is estimated from data and stays fixed throughout EM (no atom-dependent recomputation)
      - Precision = n^(k) / Sigma still scales with sample sizes n^(k)
      - GLMixture.fit() runs standard EM with this constant precision
    
    Args:
        x: (k, d) observations
        n_k: (k,) local sample sizes for each client
        em_iters: number of EM iterations
        quiet_solver: whether to suppress solver output
    
    Returns:
        (theta_hat, runtime): posterior means and elapsed time
    """
    # Estimate fixed parameter-independent Sigma from data
    # Under the model x ~ N(theta, Sigma/n), we estimate Sigma
    # using a simple moment-based approach: mean(x^2) is a rough estimate
    sigma2_fixed = np.mean(x ** 2, axis=0)  # (d,) fixed variance
    sigma2_fixed = np.maximum(sigma2_fixed, 1e-12)
    
    # Precision scales with n^(k): prec^(k) = n^(k) / Sigma_fixed
    # This is (k, d) and stays constant throughout EM (no recomputation at atoms)
    prec_fixed = n_k[:, None] / sigma2_fixed[np.newaxis, :]  # (k, d)
    
    try:
        t0 = time.time()
        # Fit standard NPMLE with fixed (non-atom-dependent) precision
        # homoscedastic=False allows per-observation precision (k, d)
        # but we never recompute it at atom locations (unlike NPEB)
        model = GLMixture(prec_type="diagonal", homoscedastic=False)
        
        if quiet_solver:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                model.fit(x, prec_fixed, max_iter_em=em_iters,
                          weight_thresh=1e-4, solver="mosek",
                          row_condition=True, score_every=None)
        else:
            model.fit(x, prec_fixed, max_iter_em=em_iters,
                      weight_thresh=1e-4, solver="mosek",
                      row_condition=True, score_every=None)
        
        # Posterior mean with same fixed precision
        theta_hat = model.posterior_mean(x, prec_fixed)
        return theta_hat, time.time() - t0
    except Exception as exc:
        raise RuntimeError(f"GLMixture Soloff fitting failed: {exc}") from exc

def run_one(
    k: int,
    rep: int,
    cfg: SimConfig,
    rng: np.random.Generator,
    sigma2_fn: Callable[[np.ndarray], np.ndarray],
    quiet_solver: bool,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    theta, x, obs_var, n_k = generate_data(k, cfg, rng, sigma2_fn)

    theta_adamix = adamix_estimator(x, obs_var, cfg, rng)
    
    # Oracle: use local MC for d > 2 (global grid cannot cover high-D spheres);
    # fall back to the fast grid oracle in 2-D.
    prior_centers = np.array(cfg.prior_centers, dtype=float)
    prior_radii   = np.array(cfg.prior_radii, dtype=float)
    prior_weights = np.array(cfg.prior_weights, dtype=float)
    if cfg.dim > 2:
        theta_oracle = oracle_posterior_mean_local_mc(
            x, n_k, sigma2_fn,
            centers=prior_centers,
            radii=prior_radii,
            weights=prior_weights,
            n_mc=5000,
            rng=rng,
        )
    else:
        # Scale oracle grid density with observation precision: as n grows,
        # obs variance shrinks and the grid must be finer to avoid discretisation bias.
        oracle_apc = max(50, cfg.n_max // 4)
        oracle_atoms = generate_uniform_prior_support(
            prior_centers, prior_radii,
            dim=cfg.dim, atoms_per_circle=oracle_apc,
        )
        theta_oracle = oracle_posterior_mean(oracle_atoms, x, n_k, sigma2_fn)

    # NPEB estimator with atom-dependent covariance (Algorithm 2)
    theta_npeb, npeb_time = glmixture_npeb_estimator(x, n_k, sigma2_fn, cfg.em_iters, quiet_solver)
    
    # Soloff baseline: homoscedastic NPMLE with fixed covariance
    theta_soloff, soloff_time = soloff_estimator(x, n_k, cfg.em_iters, quiet_solver)

    row = {
        "k": k,
        "rep": rep,
        "rmse_adamix": rmse(theta_adamix, theta),
        "rmse_oracle": rmse(theta_oracle, theta),
        "rmse_npeb": rmse(theta_npeb, theta),
        "rmse_soloff": rmse(theta_soloff, theta),
        "npeb_runtime_sec": npeb_time,
        "soloff_runtime_sec": soloff_time,
        "n_mean": float(np.mean(n_k)),
        "n_min": int(np.min(n_k)),
        "n_max": int(np.max(n_k)),
    }

    snapshot = {
        "theta": theta,
        "x": x,
        "theta_oracle": theta_oracle,
        "theta_adamix": theta_adamix,
        "theta_npeb": theta_npeb,
        "theta_soloff": theta_soloff,
    }

    return row, snapshot


def parse_k_values(text: str) -> List[int]:
    return [int(s.strip()) for s in text.split(",") if s.strip()]


def parse_prior_centers(text: str) -> List[List[float]]:
    """Parse prior centers from string format.
    
    Format: '[x1,y1,z1,...]::[x2,y2,z2,...]::[x3,y3,z3,...]'
    Brackets are optional and stripped if present.
    Works for any dimension (not just 2D).
    Must have exactly 3 centers.
    """
    centers: List[List[float]] = []
    
    # Use double colon as separator
    for block in text.split("::"):
        block = block.strip()
        # Strip brackets if present
        if block.startswith('['):
            block = block[1:]
        if block.endswith(']'):
            block = block[:-1]
        
        if not block:
            continue
        vals = [float(v.strip()) for v in block.split(",") if v.strip()]
        if len(vals) == 0:
            raise ValueError(f"Empty center specification: {block}")
        centers.append(vals)
    if len(centers) != 3:
        raise ValueError("Three-circle prior requires exactly 3 centers.")
    
    # Check that all centers have the same dimension
    dim = len(centers[0])
    for i, c in enumerate(centers[1:], 1):
        if len(c) != dim:
            raise ValueError(f"All centers must have same dimension. Center 0 has {dim}, center {i} has {len(c)}.")
    
    return centers


def parse_float_list(text: str, expected_len: int, name: str) -> List[float]:
    vals = [float(v.strip()) for v in text.split(",") if v.strip()]
    if len(vals) != expected_len:
        raise ValueError(f"{name} must have exactly {expected_len} values.")
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(description="Heteroskedastic FL simulations with proposed NPEB, AdaMix baseline, and Oracle Bayes.")
    parser.add_argument("--k-values", type=str, default="200,500,1000", help="Comma-separated client counts.")
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--n-min", type=int, default=30)
    parser.add_argument("--n-max", type=int, default=400)
    parser.add_argument("--em-iters", type=int, default=25)
    parser.add_argument("--adamix-components", type=int, default=3)
    parser.add_argument("--adamix-iters", type=int, default=20)
    parser.add_argument("--adamix-lr", type=float, default=0.05)
    parser.add_argument("--prior-centers", type=str, default="-2,0;2,0;0,2.5", help="Three centers as 'x1,y1;x2,y2;x3,y3'.")
    parser.add_argument("--prior-radii", type=str, default="1.0,1.0,1.0", help="Three circle radii as 'r1,r2,r3'.")
    parser.add_argument("--prior-weights", type=str, default="0.333333,0.333333,0.333334", help="Three mixture weights as 'w1,w2,w3'.")
    parser.add_argument("--sigma2-fn", type=str, default="quadratic", 
                        choices=list(VARIANCE_FUNCTIONS.keys()),
                        help=f"Variance function: {', '.join(VARIANCE_FUNCTIONS.keys())}.")
    parser.add_argument("--seed", type=int, default=20260224)
    parser.add_argument("--outdir", type=str, default="Simulations/outputs")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar output.")
    parser.add_argument("--show-solver-logs", action="store_true", help="Show verbose GLMixture/MOSEK logs.")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\\.utils\\.extmath")
    warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\\.cluster\\._kmeans")

    # Select variance function
    sigma2_fn = VARIANCE_FUNCTIONS[args.sigma2_fn]

    prior_centers = parse_prior_centers(args.prior_centers)
    # Validate that prior centers have the correct dimension
    center_dim = len(prior_centers[0])
    if center_dim != args.dim:
        raise ValueError(f"Prior centers have dimension {center_dim}, but --dim is {args.dim}. They must match.")
    
    prior_radii = parse_float_list(args.prior_radii, expected_len=3, name="prior-radii")
    prior_weights = parse_float_list(args.prior_weights, expected_len=3, name="prior-weights")
    if any(r <= 0 for r in prior_radii):
        raise ValueError("All prior-radii must be positive.")
    sw = sum(prior_weights)
    if sw <= 0:
        raise ValueError("Sum of prior-weights must be positive.")
    prior_weights = [w / sw for w in prior_weights]

    cfg = SimConfig(
        k_values=parse_k_values(args.k_values),
        reps=args.reps,
        dim=args.dim,
        radius=args.radius,
        n_min=args.n_min,
        n_max=args.n_max,
        em_iters=args.em_iters,
        adamix_components=args.adamix_components,
        adamix_iters=args.adamix_iters,
        adamix_lr=args.adamix_lr,
        prior_centers=prior_centers,
        prior_radii=prior_radii,
        prior_weights=prior_weights,
        random_seed=args.seed,
    )

    outdir = Path(args.outdir)
    snapshots_dir = outdir / "snapshots"
    outdir.mkdir(parents=True, exist_ok=True)
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.random_seed)
    rows: List[Dict[str, float]] = []

    jobs = [(k, rep) for k in cfg.k_values for rep in range(cfg.reps)]
    iterator = jobs
    if not args.no_progress:
        iterator = tqdm(jobs, total=len(jobs), desc=f"n_min={cfg.n_min}", unit="run")

    for k, rep in iterator:
        row, snapshot = run_one(k, rep, cfg, rng, sigma2_fn, quiet_solver=not args.show_solver_logs)
        rows.append(row)
        if rep == 0:
            np.savez(
                snapshots_dir / f"snapshot_k{k}.npz",
                **snapshot,
            )

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "metrics.csv", index=False)
    
    # Save configuration including variance function name
    config_dict = asdict(cfg)
    config_dict["sigma2_fn"] = args.sigma2_fn
    with open(outdir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    summary = (
        df[["k", "rmse_adamix", "rmse_oracle", "rmse_npeb", "rmse_soloff"]]
        .groupby("k")
        .mean()
        .reset_index()
    )
    summary.to_csv(outdir / "summary_mean_rmse.csv", index=False)

    print(f"Saved metrics to {outdir / 'metrics.csv'}")
    print(f"Saved summary to {outdir / 'summary_mean_rmse.csv'}")
    print("NPEB backend: npeb.GLMixture with MOSEK")


if __name__ == "__main__":
    main()
