#!/usr/bin/env python3
"""Generate manuscript figures from completed simulation outputs."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = ROOT / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

ESTIMATORS = [
    ("rmse_vaneb", "VANEB", "#0B5FFF", "o"),
    ("rmse_npeb", "NPEB", "#E76F51", "s"),
    ("rmse_adamix", "AdaMix", "#2A9D8F", "^"),
    ("rmse_oracle", "Oracle", "#6C757D", "D"),
]

SCENARIO_TITLES = {
    "quadratic": "Quadratic variance",
    "logistic": "Multiclass logistic",
    "poisson": "Poisson regression",
}


def load_summary() -> pd.DataFrame:
    rows = []
    for run_dir in sorted(OUTPUT_DIR.iterdir()):
        if not run_dir.is_dir():
            continue

        scenario, k_token, nmin_token = run_dir.name.split("_")
        k_value = int(k_token[1:])
        nmin_value = int(nmin_token.replace("nmin", ""))

        metrics = pd.read_csv(run_dir / "metrics.csv")
        row = {
            "scenario": scenario,
            "K": k_value,
            "nmin": nmin_value,
            "reps": len(metrics),
        }
        for metric, _, _, _ in ESTIMATORS:
            row[f"{metric}_mean"] = metrics[metric].mean()
            row[f"{metric}_sem"] = metrics[metric].std(ddof=1) / (len(metrics) ** 0.5)
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["scenario", "K", "nmin"])


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D9DEE8", linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=10)


def plot_sweep(summary: pd.DataFrame, scenario: str, x_key: str, fixed_key: str, fixed_value: int, output_name: str) -> None:
    subset = summary[
        (summary["scenario"] == scenario) &
        (summary[fixed_key] == fixed_value)
    ].sort_values(x_key)

    fig, ax = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
    style_axes(ax)

    x_values = subset[x_key].to_list()
    x_positions = list(range(len(x_values)))
    for metric, label, color, marker in ESTIMATORS:
        means = subset[f"{metric}_mean"].to_numpy()
        sems = subset[f"{metric}_sem"].to_numpy()
        ax.plot(
            x_positions,
            means,
            color=color,
            marker=marker,
            linewidth=2.1,
            markersize=5.5,
            label=label,
        )
        ax.fill_between(x_positions, means - sems, means + sems, color=color, alpha=0.12)

    ax.set_xticks(x_positions, [str(value) for value in x_values])
    ax.set_xlabel(r"$K$" if x_key == "K" else r"$n_{\min}$", fontsize=11)
    ax.set_ylabel("RMSE", fontsize=11)
    ax.set_title(SCENARIO_TITLES[scenario], fontsize=12, pad=8)
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper right")

    pdf_path = FIGURE_DIR / f"{output_name}.pdf"
    png_path = FIGURE_DIR / f"{output_name}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.labelcolor": "#1F2933",
        "axes.titlecolor": "#1F2933",
        "xtick.color": "#1F2933",
        "ytick.color": "#1F2933",
    })

    summary = load_summary()

    plot_sweep(summary, "quadratic", "K", "nmin", 800, "quadratic_k_sweep")
    plot_sweep(summary, "quadratic", "nmin", "K", 800, "quadratic_nmin_sweep")
    plot_sweep(summary, "logistic", "K", "nmin", 800, "logistic_k_sweep")
    plot_sweep(summary, "logistic", "nmin", "K", 800, "logistic_nmin_sweep")
    plot_sweep(summary, "poisson", "K", "nmin", 800, "poisson_k_sweep")
    plot_sweep(summary, "poisson", "nmin", "K", 800, "poisson_nmin_sweep")


if __name__ == "__main__":
    main()