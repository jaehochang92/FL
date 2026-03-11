#!/usr/bin/env python3
"""Aggregate and plot 2D quadratic simulation results.

Usage:
    python Simulations/analyze_results.py            # from repo root
    python Simulations/analyze_results.py --no-plot   # table only
    python Simulations/analyze_results.py --no-table  # figures only
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NMIN_VALUES = [100, 400, 1600]
K_VALUES = [100, 400, 1600]
ESTIMATORS = ["oracle", "npeb", "soloff", "adamix"]
LABELS = {
    "oracle": "Oracle Bayes",
    "npeb": "NPEB (Proposed)",
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

# Paths are relative to the repo root
OUTPUT_DIR = "Simulations/outputs"
FIG_DIR = "Simulations/figures"


# ===================================================================
# 1. Load data
# ===================================================================
def load_grid_data(output_dir: str = OUTPUT_DIR):
    """Load metrics.csv for every (n_min, K) pair in the grid."""
    data = {}
    for nmin in NMIN_VALUES:
        for k in K_VALUES:
            path = os.path.join(output_dir, f"quad2d_nmin{nmin}_k{k}", "metrics.csv")
            if os.path.exists(path):
                data[(nmin, k)] = pd.read_csv(path)
    return data


# ===================================================================
# 2. Aggregation & LaTeX table
# ===================================================================
def print_summary_and_latex(data: dict):
    """Print summary stats and a LaTeX table to stdout."""
    rows = []
    for nmin in NMIN_VALUES:
        for k in K_VALUES:
            if (nmin, k) not in data:
                continue
            df = data[(nmin, k)]
            for est in ESTIMATORS:
                col = f"rmse_{est}"
                rows.append({
                    "n_min": nmin,
                    "K": k,
                    "estimator": est,
                    "mean": df[col].mean(),
                    "std": df[col].std(),
                })

    if not rows:
        print("No results found.")
        return

    rdf = pd.DataFrame(rows)
    pivot = rdf.pivot_table(index=["n_min", "K"], columns="estimator", values="mean")
    pivot_std = rdf.pivot_table(index=["n_min", "K"], columns="estimator", values="std")

    # ---- Plain text summary ----
    print("=== Summary (mean ± std) ===")
    for (nmin, k), row in pivot.iterrows():
        stds = pivot_std.loc[(nmin, k)]
        print(
            f"n_min={nmin:>4d}, K={k:>4d}: "
            f"Oracle {row['oracle']:.4f}±{stds['oracle']:.4f}  "
            f"NPEB {row['npeb']:.4f}±{stds['npeb']:.4f}  "
            f"Soloff {row['soloff']:.4f}±{stds['soloff']:.4f}  "
            f"AdaMix {row['adamix']:.4f}±{stds['adamix']:.4f}"
        )

    print("\n=== NPEB improvement over Soloff ===")
    for (nmin, k), row in pivot.iterrows():
        pct = (1 - row["npeb"] / row["soloff"]) * 100
        print(f"n_min={nmin:>4d}, K={k:>4d}: {pct:+.1f}%")

    # ---- LaTeX table ----
    print("\n=== LaTeX Table ===")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(
        r"\caption{RMSE performance (mean $\pm$ std over 50 replicates) under "
        r"quadratic variance $\Sigma(\b\theta) = \operatorname{diag}(\b\theta^2)$ "
        r"with three-circle mixture prior in $\mathbb{R}^2$.}"
    )
    print(r"\label{tab:results}")
    print(r"\begin{tabular}{@{}rr cccc@{}}")
    print(r"\toprule")
    print(r"$\un$ & $K$ & Oracle & NPEB & Soloff & AdaMix \\")
    print(r"\midrule")
    for (nmin, k), row in pivot.iterrows():
        stds = pivot_std.loc[(nmin, k)]
        print(
            f"${nmin}$ & ${k}$ & "
            f"${row['oracle']:.3f} \\pm {stds['oracle']:.3f}$ & "
            f"${row['npeb']:.3f} \\pm {stds['npeb']:.3f}$ & "
            f"${row['soloff']:.3f} \\pm {stds['soloff']:.3f}$ & "
            f"${row['adamix']:.3f} \\pm {stds['adamix']:.3f}$ \\\\"
        )
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


# ===================================================================
# 3. Figures
# ===================================================================
def generate_figures(data: dict, fig_dir: str = FIG_DIR):
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({
        "font.size": 11,
        "figure.figsize": (7, 4.5),
        "axes.grid": True,
        "grid.alpha": 0.3,
    })
    os.makedirs(fig_dir, exist_ok=True)

    # --- Figure 1: K-scaling (fixed n_min=100) ---
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    nmin_fixed = 100
    for est in ESTIMATORS:
        means, stds, ks_used = [], [], []
        for k in K_VALUES:
            if (nmin_fixed, k) in data:
                col = f"rmse_{est}"
                means.append(data[(nmin_fixed, k)][col].mean())
                stds.append(data[(nmin_fixed, k)][col].std() / np.sqrt(50))
                ks_used.append(k)
        if ks_used:
            ax.errorbar(
                ks_used, means, yerr=stds,
                marker=MARKERS[est], color=COLORS[est],
                label=LABELS[est], linewidth=2, markersize=7, capsize=3,
            )
    ax.set_xlabel("Number of Clients $K$")
    ax.set_ylabel("RMSE")
    ax.set_title(f"Scaling with $K$ (fixed $n_{{\\min}}={nmin_fixed}$)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(K_VALUES)
    ax.set_xticklabels(K_VALUES)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "quad_k_scaling.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(fig_dir, "quad_k_scaling.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: quad_k_scaling.pdf")

    # --- Figure 2: n_min effect (fixed K=400) ---
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    k_fixed = 400
    for est in ESTIMATORS:
        means, stds, nmins_used = [], [], []
        for nmin in NMIN_VALUES:
            if (nmin, k_fixed) in data:
                col = f"rmse_{est}"
                means.append(data[(nmin, k_fixed)][col].mean())
                stds.append(data[(nmin, k_fixed)][col].std() / np.sqrt(50))
                nmins_used.append(nmin)
        if nmins_used:
            ax.errorbar(
                nmins_used, means, yerr=stds,
                marker=MARKERS[est], color=COLORS[est],
                label=LABELS[est], linewidth=2, markersize=7, capsize=3,
            )
    ax.set_xlabel("Minimum Local Sample Size $n_{\\min}$")
    ax.set_ylabel("RMSE")
    ax.set_title(f"Effect of Local Sample Size (fixed $K={k_fixed}$)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(NMIN_VALUES)
    ax.set_xticklabels(NMIN_VALUES)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "quad_nmin_effect.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(fig_dir, "quad_nmin_effect.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: quad_nmin_effect.pdf")

    # --- Figure 3: Improvement heatmap ---
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    improve = np.full((3, 3), np.nan)
    for i, nmin in enumerate(NMIN_VALUES):
        for j, k in enumerate(K_VALUES):
            if (nmin, k) in data:
                npeb_mean = data[(nmin, k)]["rmse_npeb"].mean()
                soloff_mean = data[(nmin, k)]["rmse_soloff"].mean()
                improve[i, j] = (1 - npeb_mean / soloff_mean) * 100

    im = ax.imshow(improve, cmap="RdYlGn", vmin=0, vmax=20, aspect="auto")
    ax.set_xticks(range(3))
    ax.set_xticklabels(K_VALUES)
    ax.set_yticks(range(3))
    ax.set_yticklabels(NMIN_VALUES)
    ax.set_xlabel("$K$ (Number of Clients)")
    ax.set_ylabel("$n_{\\min}$ (Local Sample Size)")
    ax.set_title("NPEB Improvement over Soloff (%)")

    for i in range(3):
        for j in range(3):
            if not np.isnan(improve[i, j]):
                ax.text(
                    j, i, f"{improve[i, j]:.1f}%",
                    ha="center", va="center", fontsize=12, fontweight="bold",
                    color="white" if improve[i, j] > 10 else "black",
                )

    plt.colorbar(im, ax=ax, label="% Improvement")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "quad_improvement_heatmap.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(fig_dir, "quad_improvement_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: quad_improvement_heatmap.pdf")


# ===================================================================
# CLI
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="Aggregate & plot 2D quadratic results.")
    parser.add_argument("--no-table", action="store_true", help="Skip summary/LaTeX output")
    parser.add_argument("--no-plot", action="store_true", help="Skip figure generation")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Path to outputs/")
    parser.add_argument("--fig-dir", default=FIG_DIR, help="Path to figures/")
    args = parser.parse_args()

    data = load_grid_data(args.output_dir)
    if not data:
        print("No results found. Run simulations first.")
        sys.exit(1)

    if not args.no_table:
        print_summary_and_latex(data)

    if not args.no_plot:
        generate_figures(data, args.fig_dir)


if __name__ == "__main__":
    main()
