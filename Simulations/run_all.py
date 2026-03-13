#!/usr/bin/env python3
"""
Unified simulation runner for all three scenarios.

Sweep design (for each scenario):
    n_min sweep:  n_min ∈ {50, 200, 800},  K = 800 fixed
    K sweep:      K ∈ {50, 200, 800},       n_min = 800 fixed

Usage:
    python Simulations/run_all.py                     # all scenarios, full sweep
    python Simulations/run_all.py --scenario quadratic --smoke  # quick test
    python Simulations/run_all.py --dry-run           # print configs only
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from tqdm.auto import tqdm

from scenario_base import SimConfig
from scenario_quadratic import QuadraticMeanScenario
from scenario_logistic import LogisticScenario
from scenario_poisson import PoissonScenario

SCENARIOS = {
    "quadratic": QuadraticMeanScenario,
    "logistic": LogisticScenario,
    "poisson": PoissonScenario,
}

SCRIPT_DIR = Path(__file__).resolve().parent

# Sweep configurations
NMIN_SWEEP = [100, 200, 400, 800, 1600, 3200]       # vary n_min, fix K = 800
K_SWEEP = NMIN_SWEEP.copy()      # vary K, fix n_min = 800
K_FIXED = 3200
NMIN_FIXED = 100
REPS = 20


def build_configs(smoke: bool = False):
    """Build list of (scenario_name, SimConfig) tuples for the full sweep."""
    configs = []
    reps = 3 if smoke else REPS

    for sc_name in SCENARIOS:
        # n_min sweep (K fixed)
        for nmin in (NMIN_SWEEP[:1] if smoke else NMIN_SWEEP):
            K = 20 if smoke else K_FIXED
            configs.append((sc_name, SimConfig(
                K=K, reps=reps, n_min=nmin, n_max=3 * nmin,
            )))
        # K sweep (n_min fixed)
        for K in (K_SWEEP[:1] if smoke else K_SWEEP):
            nmin = 50 if smoke else NMIN_FIXED
            configs.append((sc_name, SimConfig(
                K=K, reps=reps, n_min=nmin, n_max=3 * nmin,
            )))

    return configs


def run_sweep(sc_name: str, cfg: SimConfig, outdir: Path, no_progress: bool = False):
    """Run one (scenario, config) combination and save results."""
    scenario = SCENARIOS[sc_name]()
    rng = np.random.default_rng(cfg.seed)

    rows = []
    jobs = list(range(cfg.reps))
    it = jobs if no_progress else tqdm(jobs, desc=f"{sc_name} K={cfg.K} nmin={cfg.n_min}")

    for rep in it:
        try:
            row = scenario.run_one(cfg.K, rep, cfg, rng)
            rows.append(row)
        except Exception as e:
            print(f"  Error {sc_name} K={cfg.K} nmin={cfg.n_min} rep={rep}: {e}")

    df = pd.DataFrame(rows)
    tag = f"{sc_name}_K{cfg.K}_nmin{cfg.n_min}"
    run_dir = outdir / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(run_dir / "metrics.csv", index=False)
    with open(run_dir / "config.json", "w") as f:
        json.dump({
            "scenario": sc_name, "K": cfg.K, "n_min": cfg.n_min,
            "n_max": cfg.n_max, "reps": cfg.reps, "em_iters": cfg.em_iters,
            "seed": cfg.seed,
        }, f, indent=2)

    # Print summary
    print(f"\n  [{tag}] {len(df)} reps")
    for col in ["rmse_oracle", "rmse_vaneb", "rmse_npeb", "rmse_adamix"]:
        if col in df.columns:
            print(f"    {col:20s}: {df[col].mean():.4f} ± {df[col].std():.4f}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Unified FL simulation runner")
    parser.add_argument("--scenario", type=str, default=None,
                        choices=list(SCENARIOS.keys()),
                        help="Run only one scenario (default: all)")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke test (tiny K, few reps)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print configs without running")
    parser.add_argument("--list-configs", action="store_true",
                        help="List configs with zero-based indices and exit")
    parser.add_argument("--config-index", type=int, default=None,
                        help="Run only one zero-based config index")
    parser.add_argument("--reps", type=int, default=None,
                        help="Override number of Monte Carlo replicates")
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    configs = build_configs(smoke=args.smoke)
    if args.scenario:
        configs = [(n, c) for n, c in configs if n == args.scenario]

    if args.reps is not None:
        if args.reps <= 0:
            raise ValueError("--reps must be a positive integer")
        for _, cfg in configs:
            cfg.reps = args.reps

    if args.list_configs:
        for idx, (sc_name, cfg) in enumerate(configs):
            print(
                f"{idx:3d}  {sc_name:12s}  K={cfg.K:4d}  "
                f"nmin={cfg.n_min:4d}  nmax={cfg.n_max:4d}  reps={cfg.reps}"
            )
        return

    if args.config_index is not None:
        if args.config_index < 0 or args.config_index >= len(configs):
            raise IndexError(
                f"--config-index {args.config_index} out of range "
                f"[0, {len(configs) - 1}]"
            )
        configs = [configs[args.config_index]]

    if args.dry_run:
        for idx, (sc_name, cfg) in enumerate(configs):
            print(f"{idx:3d}  {sc_name:12s}  K={cfg.K:4d}  nmin={cfg.n_min:4d}  "
                  f"nmax={cfg.n_max:4d}  reps={cfg.reps}")
        return

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = SCRIPT_DIR / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Running {len(configs)} configurations...")
    t0 = time.time()
    for sc_name, cfg in configs:
        run_sweep(sc_name, cfg, outdir, no_progress=args.no_progress)
    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
