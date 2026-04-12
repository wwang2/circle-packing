---
strategy: ginibre-dpp-multistart
status: in-progress
eval_version: eval-v1
metric: 2.6359830865
issue: 25
parents:
  - mobius-001
type: experiment
---

# GINIBRE-RESTART-001: DPP Initialization Bias Test

## Glossary

- **DPP**: Determinantal Point Process -- a probability distribution over point configurations exhibiting repulsion (negative correlation between point locations)
- **Ginibre ensemble**: The set of eigenvalues of an n x n random complex Gaussian matrix; a canonical DPP with logarithmic repulsion
- **Hyperuniform**: A point process whose density fluctuations are suppressed at large scales (variance grows slower than volume)
- **KKT**: Karush-Kuhn-Tucker conditions -- first-order necessary conditions for constrained optimality
- **Basin of attraction**: The set of initial conditions that converge to a given local optimum under a fixed optimization algorithm

## Motivation

The known best packing (sum_r = 2.6359830865) has been found by 15+ prior optimizers across 8 initialization strategies and 8000+ configurations (mobius-001). However, Connelly (brainstorm panel) gives 60/40 odds against global optimality, arguing all prior initializations share implicit spatial bias: grid, hex, random-uniform, and ring all produce configurations with positive or zero spatial correlation.

The Ginibre ensemble provides a qualitatively different starting point. Its eigenvalues exhibit determinantal repulsion -- the probability of two points being close vanishes quadratically with their separation. This negative correlation and hyperuniformity give configurations that no prior optimizer has used. If 10,000+ Ginibre-initialized runs still converge to sum_r = 2.6360, the "correlated basin" objection is empirically refuted.

## Approach

1. **Three initialization families tested:**
   - Ginibre DPP (negatively correlated, hyperuniform)
   - Uniform random (uncorrelated baseline)
   - Scrambled Halton (quasi-random, low-discrepancy)

2. **Optimization pipeline per restart:**
   - Greedy radius assignment from initial positions
   - L-BFGS-B with progressive penalty (lambda = 10, 100, 1k, 10k, 100k)
   - SLSQP constrained polish (maxiter = 10,000)

3. **Analysis:**
   - Basin identification by clustering final sum_r values (tolerance = 0.001)
   - Convergence fraction to known best basin
   - Comparison across initialization families

## Results

*In progress -- running campaign...*
