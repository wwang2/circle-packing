---
strategy: jax-soft-body-annealing
status: in-progress
eval_version: v1
metric: 2.6359830849
issue: 12
parent: topo-001
---

## JAX Soft-Body Circle Packing with Constraint Annealing

### Result
**metric=2.6359830849** for n=26 (matches current best, no improvement yet).

### Approach
Differentiable soft-body physics with JAX:
1. Energy function: -alpha*sum(r) + beta*(overlap^2 + wall^2 + radius^2)
2. Constraint annealing: beta increases from 1 to 1e8 over optimization
3. Adam optimizer with cosine LR decay
4. Four annealing schedules: standard, slow, cyclic, fast
5. Multiple init strategies: ring, random, hex, warm-start from best known
6. SLSQP polish for exact constraint satisfaction

### Attempt 1: V1 optimizer (150 trajectories)
- 40 warm starts (4 schedules x 10 noise levels)
- 50 cold starts (ring/random/hex x 4 schedules)
- 30 large perturbations from best
- 20 cyclic re-annealing runs

**Result**: Best raw metric 2.6359830883 (cyclic_5), but violations ~4.6e-10.
After strict polishing, converges to 2.6359830849 -- same basin as all prior orbits.

### What I Learned
- Soft constraints with annealing WORK for this problem -- cyclic schedule recovers the known optimum reliably
- The "improvement" to 2.6359830883 was just constraint slop, not a real topology change
- All 150 trajectories land in the same topological basin or worse basins (~2.61-2.63)
- The attractor basin for this optimum is extraordinarily deep
- Need much more aggressive topology disruption to escape

### Seeds
Primary: 42, 1000-1049, 5000-5029, 8000-8019 (documented in optimizer.py)

### Attempts
| # | Strategy | Metric | Notes |
|---|----------|--------|-------|
| 1 | V1 optimizer (150 runs) | 2.6359830849 | Same basin, no improvement |
