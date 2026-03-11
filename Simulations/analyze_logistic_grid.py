#!/usr/bin/env python
"""
Aggregate logistic regression sweep results and generate summary figures.

Requested sweeps:
1) Local sample sizes: n_min in {25, 100, 400} with fixed K=100
2) Number of clients: K in {25, 100, 400} with fixed n_min=100
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_logistic_grid_data(base_outdir='Simulations/outputs'):
    """Load and combine metrics from new 2-sweep design.
    
    Sweep 1: nmin ∈ {100, 200, 400} at fixed K=100
    Sweep 2: K ∈ {100, 200, 400} at fixed nmin=100
    """
    run_dirs = [
        'logistic_3d_nmin100_k100',
        'logistic_3d_nmin200_k100',
        'logistic_3d_nmin400_k100',
        'logistic_3d_nmin100_k100',  # duplicate, will be ignored
        'logistic_3d_nmin100_k200',
        'logistic_3d_nmin100_k400',
    ]
    all_data = []
    
    for run_dir in run_dirs:
        grid_dir = os.path.join(base_outdir, run_dir)
        metrics_file = os.path.join(grid_dir, 'metrics.csv')
        
        if os.path.exists(metrics_file):
            df = pd.read_csv(metrics_file)
            if 'nmin' not in df.columns or 'k' not in df.columns:
                parts = run_dir.split('_')
                nmin = int([p for p in parts if p.startswith('nmin')][0].replace('nmin', ''))
                kval = int([p for p in parts if p.startswith('k')][0].replace('k', ''))
                df['nmin'] = nmin
                df['k'] = kval
            all_data.append(df)
            print(f"Loaded {metrics_file}: {len(df)} rows")
        else:
            print(f"INFO: {metrics_file} not found (expected if not yet run)")
    
    if not all_data:
        print("ERROR: No logistic grid data found")
        return None
    
    combined = pd.concat(all_data, ignore_index=True)
    # Remove duplicate rows from overlapping nmin100_k100
    combined = combined.drop_duplicates(subset=['k', 'rep', 'nmin'], keep='first')
    return combined

def compute_statistics(df):
    """Compute mean and std by (nmin, k) for each method."""
    estimators = ['Oracle', 'NPEB', 'AdaMix']
    stats = []
    
    # Use lowercase 'k' to match CSV column name
    for n_min in sorted(df['nmin'].unique()):
        for k_val in sorted(df[df['nmin'] == n_min]['k'].unique()):
            subset = df[(df['nmin'] == n_min) & (df['k'] == k_val)]
            row = {'nmin': n_min, 'k': k_val}
            
            # Map to CSV column names: rmse_oracle, rmse_npeb, rmse_adamix
            for est in estimators:
                col_name = f'rmse_{est.lower()}'
                if col_name in subset.columns:
                    values = subset[col_name].dropna()
                    if len(values) > 0:
                        row[f'{est}_mean'] = values.mean()
                        row[f'{est}_std'] = values.std()
            
            stats.append(row)
    
    stats_df = pd.DataFrame(stats)
    return stats_df

def print_latex_table(stats_df):
    """Print results as LaTeX table matching quadratic format."""
    print("\n" + "="*80)
    print("LOGISTIC REGRESSION 3D RESULTS TABLE (LaTeX format)")
    print("="*80)
    
    print(r"\begin{tabular}{@{}rr ccc@{}}")
    print(r"\toprule")
    print(r"$\un$ & $K$ & Oracle & NPEB & AdaMix \\")
    print(r"\midrule")
    
    for n_min in sorted(stats_df['nmin'].unique()):
        subset = stats_df[stats_df['nmin'] == n_min].sort_values('k')
        first = True
        
        for _, row in subset.iterrows():
            if not first:
                print(r"\midrule")
            first = False
            
            print(f"${int(row['nmin'])}$ & ${int(row['k'])}$ ", end="")
            
            for est in ['Oracle', 'NPEB', 'AdaMix']:
                mean_key = f'{est}_mean'
                std_key = f'{est}_std'
                
                if mean_key in row and pd.notna(row[mean_key]):
                    mean_val = row[mean_key]
                    std_val = row[std_key]
                    print(f"& ${mean_val:.3f} \\pm {std_val:.3f}$ ", end="")
                else:
                    print("& -- ", end="")
            
            print(r"\\")
    
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print("="*80 + "\n")

def generate_figures(df, fig_dir='Simulations/figures', dpi=300):
    """Generate K-scaling and n_min effect plots."""
    os.makedirs(fig_dir, exist_ok=True)
    
    # Map CSV column names to display names
    estimators = ['Oracle', 'NPEB', 'AdaMix']
    csv_cols = {'Oracle': 'rmse_oracle', 'NPEB': 'rmse_npeb', 'AdaMix': 'rmse_adamix'}
    colors = {'Oracle': '#1f77b4', 'NPEB': '#ff7f0e', 'AdaMix': '#d62728'}
    markers = {'Oracle': 'o', 'NPEB': 's', 'AdaMix': 'D'}
    
    # Figure 1: K-scaling at fixed n_min=100
    fig, ax = plt.subplots(figsize=(7, 5))
    subset = df[df['nmin'] == 100]
        
    for est in estimators:
        col = csv_cols[est]
        k_vals = sorted(subset['k'].unique())
        means = []
        stds = []
        
        for k in k_vals:
            k_subset = subset[subset['k'] == k]
            if col in k_subset.columns:
                vals = k_subset[col].dropna()
                if len(vals) > 0:
                    means.append(vals.mean())
                    stds.append(vals.std())
        
        if means:
            ax.errorbar(k_vals[:len(means)], means, yerr=stds, 
                       label=est, marker=markers[est], color=colors[est],
                       linewidth=2, markersize=8, capsize=4)
        
    ax.set_xlabel('Number of Clients ($K$)', fontsize=11)
    ax.set_ylabel('RMSE', fontsize=11)
    ax.set_title('K-scaling at fixed $n_\\min = 100$', fontsize=11)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'logistic_k_scaling.pdf'), dpi=dpi, bbox_inches='tight')
    plt.savefig(os.path.join(fig_dir, 'logistic_k_scaling.png'), dpi=dpi, bbox_inches='tight')
    print(f"Saved: logistic_k_scaling.pdf")
    plt.close()
    
    # Figure 2: n_min effect at fixed K=100
    fig, ax = plt.subplots(figsize=(7, 5))
    subset = df[df['k'] == 100]
        
    for est in estimators:
        col = csv_cols[est]
        nmin_vals = sorted(subset['nmin'].unique())
        means = []
        stds = []
        
        for nmin in nmin_vals:
            nmin_subset = subset[subset['nmin'] == nmin]
            if col in nmin_subset.columns:
                vals = nmin_subset[col].dropna()
                if len(vals) > 0:
                    means.append(vals.mean())
                    stds.append(vals.std())
        
        if means:
            ax.errorbar(nmin_vals[:len(means)], means, yerr=stds,
                       label=est, marker=markers[est], color=colors[est],
                       linewidth=2, markersize=8, capsize=4)
        
    ax.set_xlabel('Min Local Sample Size ($n_\\min$)', fontsize=11)
    ax.set_ylabel('RMSE', fontsize=11)
    ax.set_title('Local-sample scaling at fixed $K = 100$', fontsize=11)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'logistic_nmin_effect.pdf'), dpi=dpi, bbox_inches='tight')
    plt.savefig(os.path.join(fig_dir, 'logistic_nmin_effect.png'), dpi=dpi, bbox_inches='tight')
    print(f"Saved: logistic_nmin_effect.pdf")
    plt.close()
    
    # Figure 3: Relative improvement heatmap (NPEB over AdaMix)
    improvement_data = []
    for n_min in sorted(df['nmin'].unique()):
        for k in sorted(df[df['nmin'] == n_min]['k'].unique()):
            subset = df[(df['nmin'] == n_min) & (df['k'] == k)]
            npeb_vals = subset['rmse_npeb'].dropna()
            adamix_vals = subset['rmse_adamix'].dropna()
            if len(npeb_vals) > 0 and len(adamix_vals) > 0:
                improvement = (adamix_vals.mean() - npeb_vals.mean()) / adamix_vals.mean() * 100
                improvement_data.append({'nmin': n_min, 'k': k, 'improvement': improvement})

    if improvement_data:
        imp_df = pd.DataFrame(improvement_data)
        pivot_data = imp_df.pivot(index='nmin', columns='k', values='improvement')
        fig, ax = plt.subplots(figsize=(8, 6))
        mat = pivot_data.to_numpy()
        im = ax.imshow(mat, cmap='RdBu_r', aspect='auto')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('NPEB Improvement over AdaMix (%)')
        ax.set_xticks(np.arange(pivot_data.shape[1]))
        ax.set_xticklabels([str(c) for c in pivot_data.columns])
        ax.set_yticks(np.arange(pivot_data.shape[0]))
        ax.set_yticklabels([str(r) for r in pivot_data.index])
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.1f}", ha='center', va='center', color='black')
        ax.set_xlabel('Number of Clients ($K$)', fontsize=12)
        ax.set_ylabel('Min Local Sample Size ($n_\\min$)', fontsize=12)
        ax.set_title('NPEB vs AdaMix: 3D Logistic Regression', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'logistic_improvement_heatmap.pdf'), dpi=dpi, bbox_inches='tight')
        plt.savefig(os.path.join(fig_dir, 'logistic_improvement_heatmap.png'), dpi=dpi, bbox_inches='tight')
        print(f"Saved: logistic_improvement_heatmap.pdf")
        plt.close()

if __name__ == '__main__':
    print("Loading logistic regression grid results...")
    df = load_logistic_grid_data()
    
    if df is not None and len(df) > 0:
        print(f"\nLoaded {len(df)} total records")
        print(f"n_min values: {sorted(df['nmin'].unique())}")
        print(f"K values: {sorted(df['k'].unique())}")
        print(f"Methods: {[col for col in df.columns if col not in ['nmin', 'k']]}")
        
        stats = compute_statistics(df)
        print_latex_table(stats)
        
        print("Generating figures...")
        generate_figures(df)
        print("Done!")
    else:
        print("No data loaded. Simulations may still be running.")
