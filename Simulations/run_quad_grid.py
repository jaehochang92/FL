#!/usr/bin/env python3
"""Run 2D quadratic simulations: 3x3 grid of n_min × K."""

import subprocess
import sys
import os

os.chdir("/Users/jaechang/git/FL")

n_min_values = [100, 400, 1600]
k_values     = [100, 400, 1600]

# Standard 2D prior centers
prior_centers = "[-2.0,0.0]::[2.0,0.0]::[0.0,2.5]"

configs = []
for nmin in n_min_values:
    nmax = 3 * nmin
    for k in k_values:
        configs.append({
            "n_min": nmin,
            "n_max": nmax,
            "k": k,
            "outdir": f"Simulations/outputs/quad2d_nmin{nmin}_k{k}",
        })

total = len(configs)
for idx, cfg in enumerate(configs, 1):
    cmd = [
        "python", "Simulations/simulate_fl.py",
        "--k-values", str(cfg["k"]),
        "--reps", "50",
        "--n-min", str(cfg["n_min"]),
        "--n-max", str(cfg["n_max"]),
        "--em-iters", "25",
        "--sigma2-fn", "quadratic",
        "--dim", "2",
        "--prior-centers", prior_centers,
        "--outdir", cfg["outdir"],
        "--no-progress",
    ]
    
    print(f"\n[{idx}/{total}] n_min={cfg['n_min']}, K={cfg['k']}, n_max={cfg['n_max']}")
    print(f"  Output: {cfg['outdir']}")
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  ERROR: failed with return code {result.returncode}")
        sys.exit(1)
    print(f"  Done.")

print(f"\nAll {total} configurations completed!")
