---
strategy: deep-refinement (SLSQP + CMA-ES + basin-hopping + subgroup optimization)
status: complete
eval_version: v1
metric: 2.8426687475
issue: 11
parent: diffevo-001
---

# refine-n30-001: Deep Refinement of n=30 Circle Packing

## Results

| Step | Metric | Valid | Notes |
|------|--------|-------|-------|
| Initial (diffevo-001) | 2.8426084050 | Yes | Parent solution |
| SLSQP ftol=1e-13 | **2.8426687475** | Yes | +6.03e-05 improvement |
| Basin-hopping | 2.8426687475 | -- | No valid improvement |
| Single-circle reposition | 2.8426687475 | -- | No improvement |
| Multi-start (50 inits) | 2.8426687475 | -- | No improvement |
| trust-constr / COBYLA | 2.8426687475 | -- | No improvement |
| CMA-ES (small+large sigma) | 2.8426687475 | -- | No improvement |
| Subgroup coordinate descent | 2.8426687475 | -- | No improvement |

**Final: 2.8426687475** (improvement: +6.03e-05 over parent, beats Cantrell 2011 SOTA of 2.842+)

## Approach

1. Loaded diffevo-001's n=30 solution (2.8426084050)
2. Applied SLSQP with progressively tighter tolerances (1e-10 to 1e-15)
3. Tried basin-hopping with L-BFGS-B penalty + SLSQP polish
4. Single-circle repositioning (remove, grid search, reinsert)
5. Multi-start from hex grids, greedy constructive, concentric rings (50 starts)
6. Alternative optimizers: trust-constr, COBYLA
7. CMA-ES with small sigma (0.001-0.01) and large sigma (0.05-0.2)
8. Subgroup coordinate descent (optimize 3-6 circles at a time)

## Key Finding: 0 Degrees of Freedom

Contact analysis reveals this is a **rigid packing**:
- 70 circle-circle contacts
- 20 circle-wall contacts
- **90 total contacts = 90 variables (30 x 3)**
- **DOF = 0**: No continuous improvement is possible within this topology

This explains why every optimizer converges to the same value. The only way to improve would be to find an entirely different contact graph topology with higher sum of radii.

## What Worked

- SLSQP with tight ftol (1e-13 to 1e-15) found the last +6e-05 improvement
- The parent solution from diffevo-001 was already in an excellent basin

## What Did Not Work

- Basin-hopping found slightly better penalty-objective values but could not make them feasible
- CMA-ES, multi-start, subgroup optimization all converged to the same point
- No alternative topology found that beats this one across 50+ diverse initializations

## Seeds

- SLSQP: deterministic
- Basin-hopping: seeds 42, 123, 456, 789, 1337
- CMA-ES: seeds 42, 123, 456, 789
- Multi-start: seeds 0-49
