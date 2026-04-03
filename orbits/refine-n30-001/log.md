---
strategy: deep-refinement (SLSQP + CMA-ES + basin-hopping + subgroup optimization)
status: stuck
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

## N=32 Approach

1. Loaded diffevo-001's n=32 solution (2.9365262667)
2. SLSQP with ftol=1e-15, maxiter=20000 -- no improvement
3. Perturbation + SLSQP (80 trials, sigma 0.001-0.05) -- no improvement
4. Single-circle repositioning (80x80 grid search) -- no improvement
5. Swap optimization (100 random pair swaps) -- no improvement
6. Subgroup coordinate descent (groups of 3-6, 30 iters each) -- no improvement
7. Augmented Lagrangian method -- converged to same value
8. trust-constr with 10000 maxiter -- no improvement
9. Topology search: 457 diverse starts (greedy, hex, ring, perturb, symmetric) -- in progress
10. Basin-hopping with 5 seeds, 150 iterations each -- in progress
11. Differential evolution -- in progress

## N=30 Approach

1. Loaded diffevo-001's n=30 solution (2.8426084050)
2. Applied SLSQP with progressively tighter tolerances (1e-10 to 1e-15)
3. Tried basin-hopping with L-BFGS-B penalty + SLSQP polish
4. Single-circle repositioning (remove, grid search, reinsert)
5. Multi-start from hex grids, greedy constructive, concentric rings (50 starts)
6. Alternative optimizers: trust-constr, COBYLA
7. CMA-ES with small sigma (0.001-0.01) and large sigma (0.05-0.2)
8. Subgroup coordinate descent (optimize 3-6 circles at a time)

## Key Findings: Rigid Packings

### N=30: DOF = 0 (exactly constrained)
- 70 circle-circle contacts + 20 circle-wall contacts = 90 total
- 90 contacts = 90 variables (30 x 3) => DOF = 0

### N=32: DOF = -1 (over-constrained)
- 77 circle-circle contacts + 20 circle-wall contacts = 97 total
- 97 contacts > 96 variables (32 x 3) => DOF = -1

Both are rigid packings where no local continuous improvement is possible.
The only way to improve is to find a fundamentally different contact graph topology.

## What Worked

- N=30: SLSQP with tight ftol found +6e-05 improvement
- Contact analysis explains why all methods converge to same value

## What Did Not Work

- For both n=30 and n=32: all local methods converge to exact same point
- Basin-hopping, CMA-ES, multi-start, subgroup optimization all fail to escape
- 50+ diverse topology starts (n=32, in progress with 457 total) have not found a better basin

## Seeds

- SLSQP: deterministic
- Basin-hopping: seeds 42, 123, 456, 789, 1337
- CMA-ES: seeds 42, 123, 456, 789
- Multi-start: seeds 0-49
- Topology search: seed 42 (n=32)
- Perturbation: seed 42 (n=32)
