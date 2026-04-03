---
strategy: deep-refinement (SLSQP + CMA-ES + basin-hopping + subgroup optimization)
status: in-progress
eval_version: v1
metric: 2.9365262667
issue: 11
parent: diffevo-001
---

# refine-n30-001: Deep Refinement of n=30 and n=32 Circle Packing

## N=32 Results

| Step | Metric | Valid | Notes |
|------|--------|-------|-------|
| Initial (diffevo-001) | 2.9365262667 | Yes | Parent solution |
| SLSQP ftol=1e-15 | 2.9365262667 | Yes | No improvement (rigid packing) |
| Perturbation + SLSQP (80 tries) | 2.9365262667 | -- | No improvement |
| Single-circle reposition | 2.9365262667 | -- | No improvement |
| Swap optimization | 2.9365262667 | -- | No improvement |
| Subgroup optimize (3-6 groups) | 2.9365262667 | -- | No improvement |
| Augmented Lagrangian | 2.9365262667 | -- | No improvement |
| trust-constr | 2.9365262667 | -- | No improvement |
| Topology search (50/457 so far) | 2.9365262667 | -- | No better topology found yet |

**N=32 Current: 2.9365262667** (same as diffevo-001, SOTA is 2.939+)

### Contact Analysis (N=32)
- 77 circle-circle contacts
- 20 circle-wall contacts
- **97 total contacts > 96 variables (32 x 3)**
- **DOF = -1**: Over-constrained rigid packing
- The only way to improve is finding a completely different contact graph topology

## N=30 Results

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

**N=30 Final: 2.8426687475** (improvement: +6.03e-05 over parent, beats Cantrell 2011 SOTA of 2.842+)

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
