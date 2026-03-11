#!/usr/bin/env python3
"""Plot federated logistic regression simulation results."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({
    "font.size": 11,
    "figure.figsize": (7, 4.5),
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# Paths
OUTPUT_DIR = "Simulations/outputs/logistic_5d_k20_50"
FIG_DIR = "Simulations/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Load data
if not os.path.exists(os.path.join(OUTPUT_DIR, "metrics.csv")):
    print(f"No logistic results found at {OUTPUT_DIR}")
    exit(1)

df = pd.read_csv(os.path.join(OUTPUT_DIR, "metrics.csv"))

# Configuration
ESTIMATORS = ["oracle", "npeb", "soloff", "adamix"]
LABELS = {
    "oracle": "Oracle Bayes",
    "npeb": "NPEB",
    "soloff": "Soloff",
    "adamix": "AdaMix",
}
COLORS = {
    "oracle": "#2ecc71",
    "npeb": "#e74c3c",
    "soloff": "#3498db",
    "adamix": "#9b59b6",
}
MARKERS = {"oracle": "s", "npeb": "o", "soloff": "^", "adamix": "D"}

# ============================================================
# Figure: K-scaling for logistic regression (5D)
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(6, 4))

k_vals = sorted(df["k"].unique())

for est in ESTIMATORS:
    means = []
    stds = []
    ks_used = []
    for k in k_vals:
        if k in df["k"].values:
            col = f"rmse_{est}"
            subset = df[df["k"] == k]
            means.append(subset[col].mean())
            stds.append(subset[col].std() / np.sqrt(len(subset)))
            ks_used.append(k)
    
    if ks_used:
        ax.errorbar(
            ks_used, means, yerr=stds,
            marker=MARKERS[est], color=COLORS[est],
            label=LABELS[est], linewidth=2, markersize=7, capsize=3,
        )

ax.set_xlabel("Number of Clients $K$")
ax.set_ylabel("RMSE")
ax.set_title("Federated Logistic Regression ($d=5$)")
ax.set_xticks(k_vals)
ax.set_xticklabels(k_vals)
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "logistic_k_scaling.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(FIG_DIR, "logistic_k_scaling.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: logistic_k_scaling.pdf")

# ============================================================
# Figure: Bar chart comparing estimators at each K
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))

x_pos = np.arange(len(k_vals))
width = 0.2

for i, est in enumerate(ESTIMATORS):
    means = []
    stds = []
    for k in k_vals:
        col = f"rmse_{est}"
        subset = df[df["k"] == k]
        means.append(subset[col].mean())
        stds.append(subset[col].std())
    
    ax.bar(
        x_pos + i * width, means, width,
        label=LABELS[est], color=COLORS[est],
        yerr=stds, capsize=4, alpha=0.8,
    )

ax.set_xlabel("Number of Clients ($K$)")
ax.set_ylabel("RMSE")
ax.set_title("Logistic Regression: Estimator Comparison ($d=5$)")
ax.set_xticks(x_pos + width * 1.5)
ax.set_xticklabels([f"$K={k}$" for k in k_vals])
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "logistic_comparison.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(FIG_DIR, "logistic_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: logistic_comparison.pdf")

print("\nAll logistic figures generated successfully!")
