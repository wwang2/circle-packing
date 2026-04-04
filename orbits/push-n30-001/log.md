---
strategy: basin-hopping-n30
status: complete
eval_version: eval-v1
metric: 2.8426687475
issue: 17
parent: refine-n30-001
---
# PUSH-N30-001: Basin-Hopping for n=30

## Result
- **Metric: 2.8426687475** (valid, evaluated)
- Parent: 2.8426687475 (refine-n30-001)
- Improvement: ~6e-11 (marginal, within same topology)

## Strategies Attempted

1. **SLSQP refinement** of starting solution: no improvement (already locally optimal)
2. **Basin-hopping** (15 seeds x 500 hops, step=0.15): tiny gains at numerical precision level
3. **Basin-hopping large steps** (5 seeds x 400 hops, step=0.25): no improvement
4. **CMA-ES** (10 configs, sigma=0.05-0.3, pop=60-200): no improvement; all converged back
5. **Remove-and-reinsert** (all 30 circles): no improvement
6. **Multi-start from scratch** (200 starts): best ~2.798 (far below current)
7. **Radical search** (hex grids, K-big+rest-small, greedy, random): best ~2.825
8. **Differential evolution** (10 seeds): best ~2.80 from scratch
9. **Final aggressive BH** (15 seeds, high temp): no improvement

## Conclusion
The n=30 solution at 2.8426687475 is at a very rigid local optimum (likely 0 DOF).
All perturbation-based and restart-based methods converge back to the same topology.
No genuinely different topology was found that improves on this result.
The SOTA claim of "2.842+" is matched by our solution.
