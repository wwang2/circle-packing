---
strategy: differential-evolution + basin-hopping
status: complete
eval_version: v1
metric: 2.9365 (n=32)
issue: 8
parent: null
---

# diffevo-001: L-BFGS-B Penalty + Basin-Hopping for Circle Packing

## Results

| n | Metric | SOTA | % of SOTA | Seeds |
|---|--------|------|-----------|-------|
| 10 | 1.5879 | 1.591 | 99.8% | 42 |
| 28 | 2.7377 | 2.737 | 100.0% | 42+refine:123 |
| 30 | 2.8369 | 2.842 | 99.8% | 42+refine:123 |
| 32 | 2.9365 | 2.939 | 99.9% | 42+refine:123 |

## Approach

1. **Multi-start initialization**: hex grid, random, concentric rings, with varying noise levels
2. **L-BFGS-B with progressive penalty**: penalty lambda from 10 to 1e10, with analytical gradient
3. **SLSQP constraint polish**: for n<=30, enforce exact constraints
4. **Basin-hopping refinement**: perturb best solution (position, radius, swap, shake, rotate) and re-optimize

## What Worked

- L-BFGS-B with analytical gradient is extremely fast (0.2-0.4s per start for n=32)
- Progressive penalty schedule (10 levels) drives solutions to near-feasibility
- Basin-hopping with diverse perturbation types (especially rotate and mixed) found improvements
- For n=28, the "rotate" perturbation at strength 0.30 found the best solution
- For n=32, "position" perturbation at strength 0.05 improved from 2.929 to 2.937

## What Did Not Work

- Pure DE on raw variables: too slow for n>15 (O(n^2) constraint evaluations per individual)
- SLSQP as primary optimizer for large n: too slow (624 constraints for n=32)
- Simple multi-start with 500 random seeds: saturates quickly, same local optima

## Key Insight

The bottleneck is not the optimizer but the **initialization topology**. Different circle arrangements lead to fundamentally different local optima. Basin-hopping with perturbations that change the topology (swap, shake, rotate) is more effective than random restarts.
