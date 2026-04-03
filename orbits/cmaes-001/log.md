---
strategy: cma-es-global-optimization
status: complete
eval_version: eval-v1
metric: 2.6359830850
issue: 6
parent: sa-001
---

# CMAES-001: CMA-ES Global Optimization for Circle Packing

## Result
**metric = 2.6359830850** (valid, n=26) -- matches SOTA (~2.6360), no meaningful improvement over parent (2.6359830849).

## Approach

Five optimizer variants tried, all warm-started from sa-001's solution:

### V2: Full CMA-ES with penalty (joint positions+radii)
- Small sigma (0.005-0.02): CMA-ES shrinks radii to satisfy penalty, SLSQP polish recovers parent basin
- Medium sigma (0.05-0.1): Finds same or worse basins after polish
- Large sigma (0.2-0.3): Degrades significantly, polish recovers to 2.58-2.61
- Perturbation+polish found +8e-11 improvement (numerical noise)

### V3: Position-only CMA-ES + LP radii
- Separated topology search from radius optimization
- LP computes optimal radii given fixed positions
- All small sigma runs polish back to 2.6359830849
- Medium sigma reaches inferior basins (2.6359773948)
- Random restarts from scratch reach at most 2.615

### V4: Topology perturbation
- Swap circles (permute positions): always converges back to same basin
- Remove-reinsert smallest circles: no improvement found
- Multi-start from hex/concentric/random/symmetric: best was 2.6125
- Aggressive perturbation (30 trials): no improvement

### V5: Multi-optimizer polish
- SLSQP at ftol=1e-16: converged, no change
- trust-constr (interior point): degraded to 2.593
- Augmented Lagrangian: converged to inferior basin (2.6359773)
- 50 micro-perturbations + SLSQP: all return to same point

## Key Insights
- The n=26 SOTA solution at ~2.6360 is an extremely deep local optimum
- CMA-ES cannot escape this basin even with sigma=0.3 and pop=300
- Position-only CMA-ES + LP radii is a clean formulation but still converges to same basin
- The topology (which circles are neighbors) is fixed at this optimum; swaps and remove-reinsert do not help
- SLSQP is already converged to machine precision at this point
- To beat SOTA would likely require discovering a fundamentally different packing topology (different number of circle-circle contacts or different contact graph)
- Multiple local optima exist nearby (2.6359773948) but are strictly inferior

## Seeds
- CMA-ES seeds: 42, 137, 2024, 99, 314, 7777
- Perturbation seeds: 88888, 77777, 66666, 99999, 42424, 54321, 11111
- Parent: sa-001/solution_n26.json

## Files
- `optimizer_v2.py` -- CMA-ES with penalty (joint optimization)
- `optimizer_v3.py` -- position-only CMA-ES + LP radii
- `optimizer_v4.py` -- topology perturbation (swap, remove-reinsert, multi-start)
- `optimizer_v5.py` -- multi-optimizer polish (SLSQP, trust-constr, augmented Lagrangian)
- `final_polish.py` -- high-precision SLSQP polish
- `plot_solution.py` -- visualization
- `solution_n26.json` -- best solution
- `figures/solution_n26.png` -- visualization
- `run.sh` -- reproduction script
