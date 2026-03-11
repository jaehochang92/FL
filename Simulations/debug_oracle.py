#!/usr/bin/env python3
"""Quick diagnostic to understand Oracle performance in Poisson setting."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Simulations.simulate_fl import (
    sample_three_circle_prior, generate_uniform_prior_support,
    oracle_posterior_mean, SimConfig
)
from Simulations.poisson_regression_sim import generate_poisson_federated_data
from scipy.special import logsumexp

rng = np.random.default_rng(42)
centers = np.array([[-2,0,0],[2,0,0],[0,2.5,0]], dtype=float)
radii = np.array([1.0,1.0,1.0])

cfg = SimConfig(
    k_values=[20], reps=1, dim=3, radius=1.0, n_min=200, n_max=200,
    em_iters=25, adamix_components=3, adamix_iters=20, adamix_lr=0.05,
    prior_centers=centers.tolist(), prior_radii=[1,1,1],
    prior_weights=[1/3,1/3,1/3], random_seed=42
)

theta_true, theta_hat, fisher_diag, n_k, X_list, y_list = \
    generate_poisson_federated_data(20, cfg, rng)

print("=== Client-level diagnostics ===")
for i in range(5):
    err = np.linalg.norm(theta_hat[i] - theta_true[i])
    obs_var_i = 1.0/fisher_diag[i]
    print(f"Client {i}: true={theta_true[i].round(3)}, hat={theta_hat[i].round(3)}, "
          f"err={err:.4f}, obs_var={obs_var_i.round(6)}")

# Build Fisher variance function (same as in sim)
total_obs = sum(X.shape[0] for X in X_list)

def fisher_variance_fn(theta):
    single = (theta.ndim == 1)
    if single:
        theta = theta[np.newaxis, :]
    m, d = theta.shape
    fisher_per_obs = np.zeros((m, d))
    tot = 0
    for X_i in X_list:
        eta = X_i @ theta.T
        eta = np.clip(eta, -10, 10)
        mu = np.exp(eta)
        for d_idx in range(d):
            fisher_per_obs[:, d_idx] += np.sum(mu * (X_i[:, d_idx:d_idx+1]**2), axis=0)
        tot += X_i.shape[0]
    fisher_per_obs /= max(tot, 1)
    sigma2 = 1.0 / np.clip(fisher_per_obs, 1e-4, 1e4)
    if single:
        return sigma2[0]
    return sigma2

# Check sigma2 at each center
print("\n=== sigma2 at prior centers ===")
sigma2_centers = fisher_variance_fn(centers)
for i, c in enumerate(centers):
    print(f"Center {c}: sigma2={sigma2_centers[i].round(6)}")

# Oracle atoms
atoms = generate_uniform_prior_support(centers, radii, atoms_per_circle=100)
print(f"\nOracle atoms: {atoms.shape}")

# Oracle posterior mean
theta_oracle = oracle_posterior_mean(atoms, theta_hat, n_k, fisher_variance_fn)

print("\n=== Oracle vs MLE for first 10 clients ===")
for i in range(10):
    oracle_err = np.linalg.norm(theta_oracle[i] - theta_true[i])
    mle_err = np.linalg.norm(theta_hat[i] - theta_true[i])
    print(f"Client {i}: true={theta_true[i].round(3)}, "
          f"oracle={theta_oracle[i].round(3)}, hat={theta_hat[i].round(3)}, "
          f"oracle_err={oracle_err:.4f}, mle_err={mle_err:.4f}")

oracle_rmse = np.sqrt(np.mean(np.sum((theta_oracle - theta_true)**2, axis=1)))
mle_rmse = np.sqrt(np.mean(np.sum((theta_hat - theta_true)**2, axis=1)))
print(f"\nOverall: Oracle RMSE={oracle_rmse:.4f}, MLE RMSE={mle_rmse:.4f}")

# Check: is the Oracle shrinking toward wrong centers?
print("\n=== Oracle shrinkage direction ===")
for i in range(5):
    # Which center is closest to theta_true?
    true_dists = [np.linalg.norm(theta_true[i] - c) for c in centers]
    oracle_dists = [np.linalg.norm(theta_oracle[i] - c) for c in centers]
    print(f"Client {i}: true_center={np.argmin(true_dists)} "
          f"(dist={min(true_dists):.3f}), "
          f"oracle_nearest_center={np.argmin(oracle_dists)} "
          f"(dist={min(oracle_dists):.3f})")
