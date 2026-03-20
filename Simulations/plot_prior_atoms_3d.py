#!/usr/bin/env python3
"""Generate a 2D heatmap of the Student's t prior density landscape.

The prior is a 2D multivariate Student's t distribution with ν=2 degrees of freedom
(heavy tails) and identity covariance. This demonstrates geometry where parametric
Gaussian mixtures (AdaMix) fail to capture tail behavior, but discrete empirical
Bayes with atoms in tail regions succeeds.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_t


ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    # Student's t prior parameters
    nu = 5  # degrees of freedom (Cauchy-like heavy tails)
    dim = 2
    loc = np.zeros(dim)  # mean
    shape = np.eye(dim)  # identity covariance (scale matrix)
    
    # Create Student's t distribution
    prior = multivariate_t(loc=loc, shape=shape, df=nu)
    
    # Create evaluation grid spanning [-5, 5] x [-5, 5]
    grid_size = 1000
    x = np.linspace(-5, 5, grid_size)
    y = np.linspace(-5, 5, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate density on grid
    pts = np.column_stack([X.ravel(), Y.ravel()])
    Z_flat = prior.pdf(pts)
    Z = Z_flat.reshape(X.shape)
    
    # Setup matplotlib
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
        }
    )
    
    fig = plt.figure(figsize=(8.0, 7.0), constrained_layout=True)
    ax = fig.add_subplot(111)
    
    # Create heatmap with logarithmic scale for better visualization of tails
    im = ax.contourf(X, Y, Z, levels=20, cmap="viridis")
    ax.contour(X, Y, Z, levels=10, colors="white", linewidths=0.5, alpha=0.3)
    
    ax.set_xlabel(r"$\theta_1$", fontsize=12)
    ax.set_ylabel(r"$\theta_2$", fontsize=12)
    ax.set_title(r"2D Student's $t$ Prior Density ($\nu={nu}$, Heavy Tails)", fontsize=13, pad=12)
    
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.3, alpha=0.1)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Density", fontsize=11)
    
    fig.savefig(FIGURE_DIR / "prior_t_2d.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "prior_t_2d.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
