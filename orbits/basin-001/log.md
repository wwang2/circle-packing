---
strategy: basin-hopping-slsqp
status: in-progress
eval_version: v1
metric: 2.6154
issue: 5
parent: null
---

# Basin-Hopping with SLSQP Local Refinement

## Approach
Multi-start optimization with progressive penalty + L-BFGS-B, followed by SLSQP polish for exact constraint satisfaction, then perturbation-based local search.

## Results

### Run 1 (seed=42, n=26, 25 starts, 480s timeout)
- **Metric: 2.6154** (valid, evaluated)
- Baseline: 1.994 -> improvement of +0.621
- SOTA: ~2.636 -> gap of ~0.021
- Best penalty-phase metric: 2.6155 (start 16, mixed_sizes init)
- Best after SLSQP polish: 2.6154
- Perturbation phase: no significant improvement
- Seeds: numpy rng seed=42
- Time: 462s

### Key observations
- Progressive penalty (50 -> 200 -> 1000 -> 5000 -> 20000) with L-BFGS-B works well
- SLSQP polish converts ~1e-5 violations to valid solutions
- Mixed-size initialization (some large + many small) works better than uniform
- n=10 sanity check: 1.591013 (matches SOTA)
