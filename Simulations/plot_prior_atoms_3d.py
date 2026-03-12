#!/usr/bin/env python3
"""Generate an illustrative 3D plot of prior atoms from the five-curve prior."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scenario_base import PRIOR_CURVES


ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    atoms_per_curve = 400
    t = np.linspace(0.0, 2.0 * np.pi, atoms_per_curve, endpoint=False)

    labels = ["Trefoil", "Helix", "Tilted Ellipse", "Figure-8", "Viviani"]
    colors = ["#1f77b4", "#e76f51", "#2a9d8f", "#f4a261", "#6d597a"]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
        }
    )

    fig = plt.figure(figsize=(7.0, 5.6), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    for curve_fn, label, color in zip(PRIOR_CURVES, labels, colors):
        pts = curve_fn(t)
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            s=2,
            alpha=1,
            color=color,
            edgecolors="none",
            label=label,
        )

    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.set_zlabel(r"$\theta_3$")
    ax.set_title("Five-Curve Prior Atoms in $\\mathbb{R}^3$", pad=12)

    ax.view_init(elev=32, azim=-17)
    ax.grid(True, linewidth=0.3, alpha=0.1)
    ax.legend(loc="upper left", fontsize=8, frameon=False)

    # Keep visual scale balanced across axes.
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    spans = np.array(
        [x_limits[1] - x_limits[0], y_limits[1] - y_limits[0], z_limits[1] - z_limits[0]]
    )
    centers = np.array(
        [np.mean(x_limits), np.mean(y_limits), np.mean(z_limits)]
    )
    radius = .35 * max(spans)
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])

    fig.savefig(FIGURE_DIR / "prior_atoms_3d.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "prior_atoms_3d.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
