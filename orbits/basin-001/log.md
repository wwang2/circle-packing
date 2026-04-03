---
strategy: basin-hopping-slsqp
status: in-progress
eval_version: v1
metric: 2.6310
issue: 5
parent: null
---

# Basin-Hopping with SLSQP Local Refinement

## Approach
Multi-start optimization with progressive penalty + L-BFGS-B, followed by SLSQP polish for exact constraint satisfaction, then perturbation-based local search.

## Results

### Best: seed=123, metric=2.6310
- **Metric: 2.6310** (valid, evaluated)
- Baseline: 1.994 -> improvement of +0.637 (+31.9%)
- SOTA: ~2.636 -> gap of ~0.005
- Penalty-phase peak: 2.6310 (start 4, greedy init)
- After SLSQP polish: 2.6310
- Perturbation phase: marginal improvement (2.630957 -> 2.630972)

### Seed scan results (n=26)
| Seed | Metric | Notes |
|------|--------|-------|
| 123  | 2.6310 | Best |
| 314  | 2.6212 | |
| 7    | 2.6186 | |
| 42   | 2.6154 | First run |
| 999  | 2.6114 | |
| 271  | 2.6060 | |
| 1337 | 0.0000 | Timeout before polish |

### Sanity check: n=10
- Metric: 1.5910 (matches SOTA 1.591+)

### Key observations
- Progressive penalty (50 -> 200 -> 1000 -> 5000 -> 20000) with L-BFGS-B works well
- SLSQP polish converts ~1e-5 violations to valid solutions
- Mixed-size and greedy initializations perform best
- Solution quality varies significantly with seed (2.606 - 2.631)
- Perturbation search provides only marginal improvement (~0.001)
- Main bottleneck: SLSQP polish is slow (~18s per call for n=26)
