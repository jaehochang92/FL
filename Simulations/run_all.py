#!/usr/bin/env python3
"""
Unified simulation runner for all three scenarios.

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
from typing import Optional

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

# Sweep configurations (paper-aligned)
NMIN_SWEEP = [5, 10, 20, 40]
K_SWEEP = [50, 200, 800, 3200]
NMIN_FIXED = NMIN_SWEEP[3]
K_FIXED = K_SWEEP[3]
REPS = 200


def build_configs(smoke: bool = False, sweep_type: Optional[str] = None):
    """Build list of (scenario_name, SimConfig) tuples for the full sweep.
    
    Args:
        smoke: Quick smoke test with reduced sizes.
        sweep_type: Filter to 'nmin', 'K', or None (both). Defaults to None.
    """
    configs = []
    reps = 10 if smoke else REPS

    for sc_name in SCENARIOS:
        # n_min sweep (K fixed)
        if sweep_type is None or sweep_type == "nmin":
            for nmin in (NMIN_SWEEP[:1] if smoke else NMIN_SWEEP):
                K = 50 if smoke else K_FIXED
                configs.append((sc_name, SimConfig(
                    K=K, reps=reps, n_min=nmin, n_max=2 * nmin,
                )))
        # K sweep (n_min fixed)
        if sweep_type is None or sweep_type == "K":
            for K in (K_SWEEP[:1] if smoke else K_SWEEP):
                nmin = NMIN_FIXED
                configs.append((sc_name, SimConfig(
                    K=K, reps=reps, n_min=nmin, n_max=2 * nmin,
                )))

    return configs


def parse_config_index_spec(spec: str, max_index: int) -> list[int]:
    """Parse a config index spec like '5', '5-10', or '1,3,7-9'."""
    indices = []
    for token in (s.strip() for s in spec.split(",")):
        if not token:
            continue
        if "-" in token:
            parts = token.split("-", 1)
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise ValueError(f"Invalid range token '{token}' in --config-index")
            start = int(parts[0])
            end = int(parts[1])
            if start > end:
                raise ValueError(f"Invalid range '{token}': start must be <= end")
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(token))

    if not indices:
        raise ValueError("--config-index is empty")

    unique = list(dict.fromkeys(indices))
    for idx in unique:
        if idx < 0 or idx > max_index:
            raise IndexError(
                f"--config-index {idx} out of range [0, {max_index}]"
            )
    return unique


def run_sweep(
    sc_name: str,
    cfg: SimConfig,
    outdir: Path,
    no_progress: bool = False,
    rep_start: int = 0,
    rep_count: Optional[int] = None,
):
    """Run one (scenario, config) combination and save results."""
    scenario = SCENARIOS[sc_name]()

    rows = []
    if rep_count is None:
        rep_count = cfg.reps - rep_start
    rep_end = min(rep_start + rep_count, cfg.reps)
    if rep_start < 0 or rep_start >= cfg.reps:
        raise ValueError("rep_start must be in [0, reps)")
    if rep_count <= 0:
        raise ValueError("rep_count must be positive")

    jobs = list(range(rep_start, rep_end))
    it = jobs if no_progress else tqdm(jobs, desc=f"{sc_name} K={cfg.K} nmin={cfg.n_min}")

    for rep in it:
        try:
            # Per-rep RNG: each rep gets a unique seed regardless of which
            # partial job (rep_start) it runs in.  Partial runs are now
            # reproducible and independent across different rep_start values.
            rng = np.random.default_rng([cfg.seed, rep])
            row = scenario.run_one(cfg.K, rep, cfg, rng)
            rows.append(row)
        except Exception as e:
            print(f"  Error {sc_name} K={cfg.K} nmin={cfg.n_min} rep={rep}: {e}")

    df = pd.DataFrame(rows)
    tag = f"{sc_name}_K{cfg.K}_nmin{cfg.n_min}"
    run_dir = outdir / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    if rep_start == 0 and rep_end == cfg.reps:
        metrics_path = run_dir / "metrics.csv"
        config_path = run_dir / "config.json"
    else:
        metrics_path = run_dir / f"metrics_part_{rep_start}_{rep_end - 1}.csv"
        config_path = run_dir / f"config_part_{rep_start}_{rep_end - 1}.json"

    df.to_csv(metrics_path, index=False)
    with open(config_path, "w") as f:
        json.dump({
            "scenario": sc_name, "K": cfg.K, "n_min": cfg.n_min,
            "n_max": cfg.n_max, "reps": cfg.reps, "em_iters": cfg.em_iters,
            "seed": cfg.seed, "use_diag": cfg.use_diag,
            "rep_start": rep_start, "rep_end": rep_end,
        }, f, indent=2)

    # Print summary
    print(f"\n  [{tag}] {len(df)} reps (range {rep_start}-{rep_end - 1})")
    for col in ["rmse_oracle", "rmse_vaneb", "rmse_npeb", "rmse_adamix"]:
        if col in df.columns:
            print(f"    {col:20s}: {df[col].mean():.4f} ± {df[col].std():.4f}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Unified FL simulation runner")
    parser.add_argument("--scenario", type=str, default=None,
                        choices=list(SCENARIOS.keys()),
                        help="Run only one scenario (default: all)")
    parser.add_argument("--sweep-type", type=str, default=None,
                        choices=["nmin", "K"],
                        help="Run only nmin-sweep or K-sweep (default: both)")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick smoke test (tiny K, few reps)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print configs without running")
    parser.add_argument("--list-configs", action="store_true",
                        help="List configs with zero-based indices and exit")
    parser.add_argument("--config-index", type=str, default=None,
                        help="Run one/many zero-based config indices: '5', '5-10', or '1,3,7-9'")
    parser.add_argument("--reps", type=int, default=None,
                        help="Override number of Monte Carlo replicates")
    parser.add_argument("--rep-start", type=int, default=None,
                        help="Start index for replicate range (for sharded runs)")
    parser.add_argument("--rep-count", type=int, default=None,
                        help="Number of replicates to run from rep-start")
    parser.add_argument("--diag", action="store_true", default=False,
                        help="Use diagonal-only precision in EM updates (faster)")
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    configs = build_configs(smoke=args.smoke, sweep_type=args.sweep_type)
    if args.scenario:
        configs = [(n, c) for n, c in configs if n == args.scenario]

    # Apply --diag flag to all configs
    for _, cfg in configs:
        cfg.use_diag = args.diag

    if args.reps is not None:
        if args.reps <= 0:
            raise ValueError("--reps must be a positive integer")
        for _, cfg in configs:
            cfg.reps = args.reps

    if args.rep_start is not None and args.rep_start < 0:
        raise ValueError("--rep-start must be >= 0")
    if args.rep_count is not None and args.rep_count <= 0:
        raise ValueError("--rep-count must be a positive integer")
    if args.rep_start is None and args.rep_count is not None:
        raise ValueError("--rep-count requires --rep-start")

    if args.list_configs:
        for idx, (sc_name, cfg) in enumerate(configs):
            print(
                f"{idx:3d}  {sc_name:12s}  K={cfg.K:4d}  "
                f"nmin={cfg.n_min:4d}  nmax={cfg.n_max:4d}  reps={cfg.reps}  "
                f"use_diag={cfg.use_diag}"
            )
        return

    if args.config_index is not None:
        selected = parse_config_index_spec(args.config_index, len(configs) - 1)
        configs = [configs[i] for i in selected]

    if args.dry_run:
        for idx, (sc_name, cfg) in enumerate(configs):
            print(f"{idx:3d}  {sc_name:12s}  K={cfg.K:4d}  nmin={cfg.n_min:4d}  "
                  f"nmax={cfg.n_max:4d}  reps={cfg.reps}  use_diag={cfg.use_diag}")
        return

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = SCRIPT_DIR / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Running {len(configs)} configurations...")
    if args.diag:
        print("  Mode: DIAGONAL-ONLY precision (fast) in EM updates")
    else:
        print("  Mode: FULL matrix precision (baseline) in EM updates")
    t0 = time.time()
    for sc_name, cfg in configs:
        run_sweep(
            sc_name,
            cfg,
            outdir,
            no_progress=args.no_progress,
            rep_start=args.rep_start or 0,
            rep_count=args.rep_count,
        )
    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
