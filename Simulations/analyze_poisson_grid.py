#!/usr/bin/env python
"""
Aggregate Poisson variance 2D sweep results and generate summary figures.

Sweeps (matching quadratic simulation design):
1) K-scaling: K ∈ {100, 400, 1600} with fixed nmin=100
2) n-scaling: nmin ∈ {100, 400, 1600} with fixed K=100
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_poisson_grid_data(base_outdir='Simulations/outputs'):
    """Load and combine metrics from 2-sweep design."""
    run_dirs = [
        'poisson_2d_nmin100_k100',
        'poisson_2d_nmin100_k400',
        'poisson_2d_nmin100_k1600',
        'poisson_2d_nmin400_k100',
        'poisson_2d_nmin1600_k100',
    ]
    all_data = []
    
    for run_dir in run_dirs:
        grid_dir = os.path.join(base_outdir, run_dir)
        metrics_file = os.path.join(grid_dir, 'metrics.csv')
        
        if os.path.exists(metrics_file):
            df = pd.read_csv(metrics_file)
            # Extract nmin and k from directory name
            parts = run_dir.split('_')
            nmin = int([p for p in parts if p.startswith('nmin')][0].replace('nmin', ''))
            kval = int([p for p in parts if p.startswith('k')][0].replace('k', ''))
            df['nmin'] = nmin
            if 'k' not in df.columns:
                df['k'] = kval
            all_data.append(df)
            print(f"Loaded {metrics_file}: {len(df)} rows")
        else:
            print(f"INFO: {metrics_file} not found (expected if not yet run)")
    
    if not all_data:
        print("ERROR: No Poisson grid data found")
        return None
    
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.drop_duplicates(subset=['k', 'rep', 'nmin'], keep='first')
    return combined

def compute_statistics(df):
    """Compute mean and std by (nmin, k) for each method."""
    estimators = ['Oracle', 'NPEB', 'AdaMix']
    stats = []
    
    for n_min in sorted(df['nmin'].unique()):
        for k_val in sorted(df[df['nmin'] == n_min]['k'].unique()):
            subset = df[(df['nmin'] == n_min) & (df['k'] == k_val)]
            row = {'nmin': n_min, 'k': k_val, 'count': len(subset)}
            
            for est in estimators:
                col_name = f'rmse_{est.lower()}'
                if col_name in subset.columns:
                    values = subset[col_name].dropna()
                    if len(values) > 0:
                        row[f'{est}_mean'] = values.mean()
                        row[f'{est}_std'] = values.std()
            
            stats.append(row)
    
    return pd.DataFrame(stats)

def generate_figures(df, fig_dir='Simulations/figures', dpi=300):
    """Generate K-scaling and n_min effect plots for Poisson variance."""
    os.makedirs(fig_dir, exist_ok=True)
    
    estimators = ['Oracle', 'NPEB', 'AdaMix']
    csv_cols = {e: f'rmse_{e.lower()}' for e in estimators}
    colors = {'Oracle': '#1f77b4', 'NPEB': '#ff7f0e', 'AdaMix': '#d62728'}
    markers = {'Oracle': 'o', 'NPEB': 's', 'AdaMix': 'D'}
    
    # Figure 1: K-scaling at fixed n_min=100
    fig, ax = plt.subplots(figsize=(7, 5))
    subset = df[df['nmin'] == 100]
    
    for est in estimators:
        col = csv_cols[est]
        if col not in subset.columns:
            continue
        k_vals = sorted(subset['k'].unique())
        means, stds = [], []
        for k in k_vals:
            vals = subset[subset['k'] == k][col].dropna()
            if len(vals) > 0:
                means.append(vals.mean())
                stds.append(vals.std() / np.sqrt(len(vals)))  # SEM
        if means:
            ax.errorbar(k_vals[:len(means)], means, yerr=stds,
                       label=est, marker=markers[est], color=colors[est],
                       linewidth=2, markersize=8, capsize=4)
    
    ax.set_xlabel('Number of Clients ($K$)', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('Poisson Variance: K-scaling ($n_{\\min} = 100$)', fontsize=12)
    ax.set_xscale('log')
    ax.set_xticks(k_vals)
    ax.set_xticklabels([str(int(v)) for v in k_vals])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'poisson_k_scaling.pdf'), dpi=dpi, bbox_inches='tight')
    plt.savefig(os.path.join(fig_dir, 'poisson_k_scaling.png'), dpi=dpi, bbox_inches='tight')
    print(f"Saved: poisson_k_scaling.pdf")
    plt.close()
    
    # Figure 2: n_min effect at fixed K=100
    fig, ax = plt.subplots(figsize=(7, 5))
    subset = df[df['k'] == 100]
    
    for est in estimators:
        col = csv_cols[est]
        if col not in subset.columns:
            continue
        nmin_vals = sorted(subset['nmin'].unique())
        means, stds = [], []
        for nmin in nmin_vals:
            vals = subset[subset['nmin'] == nmin][col].dropna()
            if len(vals) > 0:
                means.append(vals.mean())
                stds.append(vals.std() / np.sqrt(len(vals)))
        if means:
            ax.errorbar(nmin_vals[:len(means)], means, yerr=stds,
                       label=est, marker=markers[est], color=colors[est],
                       linewidth=2, markersize=8, capsize=4)
    
    ax.set_xlabel('Min Local Sample Size ($n_{\\min}$)', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('Poisson Variance: Local-sample scaling ($K = 100$)', fontsize=12)
    ax.set_xscale('log')
    ax.set_xticks(nmin_vals)
    ax.set_xticklabels([str(int(v)) for v in nmin_vals])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'poisson_nmin_effect.pdf'), dpi=dpi, bbox_inches='tight')
    plt.savefig(os.path.join(fig_dir, 'poisson_nmin_effect.png'), dpi=dpi, bbox_inches='tight')
    print(f"Saved: poisson_nmin_effect.pdf")
    plt.close()
    
    # Figure 3: NPEB improvement over AdaMix
    improvement_data = []
    for n_min in sorted(df['nmin'].unique()):
        for k in sorted(df[df['nmin'] == n_min]['k'].unique()):
            subset = df[(df['nmin'] == n_min) & (df['k'] == k)]
            npeb_vals = subset['rmse_npeb'].dropna()
            adamix_vals = subset['rmse_adamix'].dropna()
            if len(npeb_vals) > 0 and len(adamix_vals) > 0:
                improvement = (adamix_vals.mean() - npeb_vals.mean()) / adamix_vals.mean() * 100
                improvement_data.append({'nmin': n_min, 'k': k, 'improvement': improvement})

    if len(improvement_data) >= 2:
        imp_df = pd.DataFrame(improvement_data)
        
        # Bar chart version (more readable than heatmap for sparse grid)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # K-scaling improvement
        k_imp = imp_df[imp_df['nmin'] == 100].sort_values('k')
        if len(k_imp) > 0:
            axes[0].bar(range(len(k_imp)), k_imp['improvement'],
                       tick_label=[str(int(k)) for k in k_imp['k']],
                       color='#ff7f0e', alpha=0.8)
            axes[0].set_xlabel('Number of Clients ($K$)', fontsize=11)
            axes[0].set_ylabel('NPEB Improvement over AdaMix (%)', fontsize=11)
            axes[0].set_title('K-scaling ($n_{\\min} = 100$)', fontsize=11)
            axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            axes[0].grid(True, alpha=0.3, axis='y')
        
        # nmin-scaling improvement
        n_imp = imp_df[imp_df['k'] == 100].sort_values('nmin')
        if len(n_imp) > 0:
            axes[1].bar(range(len(n_imp)), n_imp['improvement'],
                       tick_label=[str(int(n)) for n in n_imp['nmin']],
                       color='#ff7f0e', alpha=0.8)
            axes[1].set_xlabel('Min Local Sample Size ($n_{\\min}$)', fontsize=11)
            axes[1].set_ylabel('NPEB Improvement over AdaMix (%)', fontsize=11)
            axes[1].set_title('$n_{\\min}$-scaling ($K = 100$)', fontsize=11)
            axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'poisson_improvement.pdf'), dpi=dpi, bbox_inches='tight')
        plt.savefig(os.path.join(fig_dir, 'poisson_improvement.png'), dpi=dpi, bbox_inches='tight')
        print(f"Saved: poisson_improvement.pdf")
        plt.close()


if __name__ == '__main__':
    print("Loading Poisson 2D grid results...")
    df = load_poisson_grid_data()
    
    if df is not None and len(df) > 0:
        print(f"\nLoaded {len(df)} total records")
        print(f"n_min values: {sorted(df['nmin'].unique())}")
        print(f"K values: {sorted(df['k'].unique())}")
        
        stats = compute_statistics(df)
        print("\n" + "="*80)
        print("POISSON 2D RESULTS")
        print("="*80)
        for _, row in stats.iterrows():
            print(f"\nnmin={int(row['nmin'])}, K={int(row['k'])} ({int(row['count'])} reps):")
            for est in ['Oracle', 'NPEB', 'AdaMix']:
                m, s = row.get(f'{est}_mean', None), row.get(f'{est}_std', None)
                if m is not None:
                    print(f"  {est:8s}: {m:.4f} ± {s:.4f}")
        
        # NPEB vs AdaMix improvement
        print("\n--- NPEB improvement over AdaMix ---")
        for _, row in stats.iterrows():
            npeb_m = row.get('NPEB_mean', None)
            ada_m = row.get('AdaMix_mean', None)
            if npeb_m and ada_m:
                imp = (ada_m - npeb_m) / ada_m * 100
                print(f"  nmin={int(row['nmin'])}, K={int(row['k'])}: {imp:+.1f}%")
        
        print("\nGenerating figures...")
        generate_figures(df)
        print("Done!")
    else:
        print("No data loaded. Simulations may still be running.")
