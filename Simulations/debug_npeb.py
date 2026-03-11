#!/usr/bin/env python3
"""Diagnose why NPEB isn't shrinking for Poisson regression."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Simulations.simulate_fl import (
    glmixture_npeb_estimator, SimConfig, rmse
)
from Simulations.poisson_regression_sim import generate_poisson_federated_data

rng = np.random.default_rng(42)
centers = np.array([[-2,0,0],[2,0,0],[0,2.5,0]], dtype=float)

cfg = SimConfig(
    k_values=[50], reps=1, dim=3, radius=1.0, n_min=15, n_max=50,
    em_iters=25, adamix_components=3, adamix_iters=20, adamix_lr=0.05,
    prior_centers=centers.tolist(), prior_radii=[1,1,1],
    prior_weights=[1/3,1/3,1/3], random_seed=42
)

theta_true, theta_hat, fisher_diag, n_k, X_list, y_list = \
    generate_poisson_federated_data(50, cfg, rng, feature_scale=0.7)

x = theta_hat
obs_var = 1.0 / fisher_diag  # actual observation variance
total_obs = sum(X.shape[0] for X in X_list)

# Build the fisher_variance_fn
def fisher_variance_fn(theta):
    single = (theta.ndim == 1)
    if single:
        theta = theta[np.newaxis, :]
    m, d = theta.shape
    fp = np.zeros((m, d))
    tot = 0
    for X_i in X_list:
        eta = X_i @ theta.T
        eta = np.clip(eta, -10, 10)
        mu = np.exp(eta)
        for d_idx in range(d):
            fp[:, d_idx] += np.sum(mu * (X_i[:, d_idx:d_idx+1]**2), axis=0)
        tot += X_i.shape[0]
    fp /= max(tot, 1)
    sigma2 = 1.0 / np.clip(fp, 1e-4, 1e4)
    if single:
        return sigma2[0]
    return sigma2

# Compare sigma2_fn at theta_hat vs actual obs_var
print("=== Variance comparison: sigma2_fn/n vs actual obs_var ===")
sigma2_at_xhat = fisher_variance_fn(x)  # (50, 3)
model_obs_var = sigma2_at_xhat / n_k[:, None]

for i in range(10):
    print(f"Client {i} (n={n_k[i]}): "
          f"sigma2/n={model_obs_var[i].round(6)}, "
          f"actual_obs_var={obs_var[i].round(6)}, "
          f"ratio={np.mean(model_obs_var[i]/obs_var[i]):.2f}")

# What precision does the NPEB see?
print("\n=== NPEB precision: n_k / sigma2(x_k) ===")
prec_init = n_k[:, None] / sigma2_at_xhat
for i in range(10):
    print(f"Client {i} (n={n_k[i]}): prec={prec_init[i].round(1)}, "
          f"|theta_true|={np.linalg.norm(theta_true[i]):.2f}, "
          f"MLE_err={np.linalg.norm(x[i]-theta_true[i]):.3f}")

# Check: with these precisions, what is the kernel overlap?
print("\n=== Kernel widths (1/sqrt(prec)) ===")
widths = 1.0 / np.sqrt(prec_init)
print(f"Min width: {widths.min():.4f}")
print(f"Max width: {widths.max():.4f}")
print(f"Mean width: {widths.mean():.4f}")

# What is the average within-cluster distance?
# Group clients by which center they're closest to
cluster_assignments = np.argmin(
    np.linalg.norm(theta_true[:, None, :] - centers[None, :, :], axis=2), axis=1
)
for c in range(3):
    mask = cluster_assignments == c
    dists = np.linalg.norm(x[mask, None, :] - x[mask][None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    print(f"\nCluster {c} ({mask.sum()} clients): "
          f"avg within-cluster obs distance = {dists[dists<np.inf].mean():.3f}, "
          f"min = {dists[dists<np.inf].min():.3f}")

# Now run NPEB and check the output
print("\n=== Running NPEB (25 EM iters) ===")
theta_npeb, npeb_time = glmixture_npeb_estimator(
    x, n_k, fisher_variance_fn, 25, quiet_solver=True
)

# Compare NPEB to MLE
diff_npeb_mle = np.linalg.norm(theta_npeb - x, axis=1)
print(f"Mean |NPEB - MLE|: {diff_npeb_mle.mean():.6f}")
print(f"Max |NPEB - MLE|: {diff_npeb_mle.max():.6f}")
print(f"NPEB RMSE: {rmse(theta_npeb, theta_true):.4f}")
print(f"MLE  RMSE: {rmse(x, theta_true):.4f}")

# Check individual clients
print("\n=== First 5 clients: NPEB vs MLE ===")
for i in range(5):
    print(f"Client {i}: MLE={x[i].round(4)}, NPEB={theta_npeb[i].round(4)}, "
          f"diff={np.linalg.norm(theta_npeb[i]-x[i]):.6f}")
