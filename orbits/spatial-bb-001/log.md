---
strategy: spatial-branch-and-bound-mccormick
status: in-progress
eval_version: eval-v1
metric: 2.7396
issue: 26
parents:
  - upperbound-001
type: experiment
---

# SPATIAL-BB-001: Spatial Branch-and-Bound with McCormick LP Relaxations

## Glossary

- **B&B**: Branch-and-bound — systematic partitioning of variable space with LP-based pruning
- **McCormick envelope**: Linear over/under-estimators for bilinear terms on bounded intervals
- **QCQP**: Quadratically Constrained Quadratic Program
- **UB**: Upper bound on maximum sum of radii

## Result (in-progress, interrupted by rate limit)

**No improvement over Fejes-Toth UB (2.7396) yet.** The McCormick LP relaxation on the full [0,1]^78 box is extremely loose — returning UB=0, indicating a formulation bug in the constraint direction or sign convention. Five solver versions (v1-v5) were attempted with different formulations.

## Approach

Implemented spatial B&B in Python with scipy.optimize.linprog:
1. **v1** (spatial_bb.py): Direct McCormick on expanded non-overlap constraints. UB=0 (bug).
2. **v2** (spatial_bb_v2.py): Reformulated with auxiliary variables for products. UB=0.
3. **v3** (spatial_bb_v3.py): Simplified to n=2 test case. UB=0.
4. **v4** (spatial_bb_v4.py): Different constraint orientation + sub-box branching.
5. **v5** (spatial_bb_v5.py): Latest attempt with corrected sign conventions.

## What's Left (interrupted)

1. Debug the McCormick formulation
2. Consider using SCIP (pyscipopt) instead of custom implementation
3. Try RLT cuts as an alternative to McCormick
4. Start with small n (n=2,3,4) to validate the formulation
5. The formulation may need SOCP relaxation instead of LP

## Files

- `spatial_bb.py` through `spatial_bb_v5.py` — Five iterations of the B&B solver
- `results.pkl`, `results_v3.pkl`, `results_v4.pkl` — Intermediate results
