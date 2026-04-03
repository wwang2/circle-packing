---
strategy: sweep-unexplored-n-values
status: complete
eval_version: eval-v1
metric: see-below
issue: 16
parent: null
---
# SWEEP-001: Under-explored n Values

## Results

| n  | Metric     | SOTA   | Ratio  |
|----|-----------|--------|--------|
| 24 | 2.4799216 | 2.530  | 98.0%  |
| 25 | 2.5268842 | 2.587  | 97.7%  |
| 27 | 2.5988993 | 2.685  | 96.8%  |
| 29 | 2.7100503 | 2.790  | 97.1%  |
| 31 | 2.7857263 | 2.889  | 96.4%  |

## Approach

Multi-start penalty L-BFGS-B with analytical gradients + lightweight basin-hopping + greedy local search refinement. 120s time budget per n value.

### Solver: solver_v4.py
- Phase 1 (20% budget): Multi-start with hex/grid/random initializations, L-BFGS-B with progressive penalty (mu = 10..100000)
- Phase 2 (65% budget): Lightweight basin-hopping with diverse perturbation strategies (shift, swap, shake, relocate worst, rotate, mirror). Only repair+local_search per hop (no L-BFGS-B).
- Phase 3 (15% budget): Fine local search refinement at decreasing step sizes.
