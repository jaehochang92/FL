# Logistic Regression 3D Grid Simulation Status

## Current Status: SIMULATIONS RUNNING ✅

Three parallel logistic regression simulations launched with **proper 3D geometry** (no zero-padding).

### Key Changes from Previous Version

**Dimension:**
- Centers at (-2,0,0), (2,0,0), (0,2.5,0) — proper 3D spheres

**Grid:** n_min, K ∈ {25, 100, 400}
- Scaled down to better match problem difficulty

### Simulation Configuration

**Grid Structure:**
- Local sample sizes: $n_{\min} \in \{25, 100, 400\}$
- Number of clients: $K \in \{25, 100, 400\}$
- Replicates per configuration: 50
- Total configurations: 9 (3×3 grid)
- Total logistic regression fits: 9 × 50 = 450

**Problem Setup (3D):**
- Dimension: **3** (fully utilizing ambient space)
- Prior: Three-circle mixture in **3D**
  - Circle 1: center (-2, 0, 0), radius 1
  - Circle 2: center (2, 0, 0), radius 1  
  - Circle 3: center (0, 2.5, 0), radius 1
  - Equal weights: 1/3 each
- Local sample sizes: $n^{(k)} \sim \text{Uniform}(n_{\min}, 3n_{\min})$
- Fisher Information heteroskedasticity: Per-client precision matrix from logistic regression fits

**Key Fix Applied:**
- Oracle now uses `oracle_posterior_mean_with_obs_var()` that correctly handles Fisher Information
- Previous bug: Oracle used uniform variance (1.0) → performed worse than all methods
- Fixed: Oracle uses actual client-specific Fisher^{-1} variances → will be best performer

**Estimators Evaluated:**
1. **Oracle Bayes**: True 3-circle mixture with Fisher-based atoms, posterior means (FIXED)
2. **NPEB**: Heteroskedastic nonparametric EB (our method, NPMLE with atom-covariance coupling)
4. **AdaMix**: Parametric Gaussian mixture (3 components)