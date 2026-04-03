---
strategy: random-topology-sampling
status: complete
eval_version: eval-v1
metric: 2.6359830849
issue: 10
parent: topo-001
---

# RANDOM-TOPO-001: Massive Random Topology Sampling

## Result
**metric=2.6359830849** -- no improvement over parent (topo-001). Confirms the n=26 optimal is extremely robust.

## Approach
Three rounds of search with 2500+ total configurations:

1. **Search v1** (1600+ configs, 389s): 8 initialization types (greedy constructive, grid, Poisson disk, cluster, ring variants, asymmetric, two-big, edge-heavy) + 600 perturbations (explode, rotate, mirror, redistribute, swap, remove-readd). Found 44 distinct topologies.

2. **Search v2** (400 configs + SA + DE, killed at ~22min): Corners-first, hex-defect, quasicrystal, fixed-topology-varied-sizes inits. Plus 20 simulated annealing runs with radical moves (scramble half, merge/split, full rotation). DE on full 78D parameter space was impractical.

3. **Search v3** (760 configs, 1793s): Apollonius gasket construction, stripe arrangements, golden spiral, Cantrell-style structural variants (10 different ring decompositions), maximal-hole greedy filling.

All configurations optimized via: L-BFGS-B penalty (progressive 10->10000) then SLSQP with analytical Jacobians.

## What Happened
Every approach converges to the same basin or a strictly worse one:
- **Best alternative topology**: 2.6359773948 (delta = -5.7e-6 from optimal)
- **Next best alternatives**: 2.6319 (stripe), 2.6311 (cantrell), 2.6310 (stripe)
- 44+ distinct contact graph topologies explored, all worse than the known optimum
- SA with radical moves (scramble half, merge/split) could not escape the basin
- Apollonius-style gasket filling reaches ~2.628 at best
- Stripe/grid patterns max at ~2.631

## What I Learned
- The n=26 optimum at 2.6360 is likely globally optimal. Its basin of attraction is enormous.
- Even starting from completely different structural patterns (stripes, spirals, Apollonius gaskets, quasicrystals), gradient descent converges to the same solution.
- The 1+8+12+4+1 ring decomposition (center + inner 8 + middle 12 + edge 4 + corner complement) is the unique optimal topology.
- Alternative topologies (different ring counts, no center, two centers) all yield lower metrics.
- This confirms the parent orbit's KKT analysis: zero DOF, all 78 dual variables strictly positive.
- To beat this result would require either: (a) a novel global optimization algorithm that can navigate extremely rugged landscapes, or (b) evidence that the problem has a better solution (which is unlikely given independent convergence from multiple groups).

## Seeds
All seeds documented in search scripts. Primary ranges: 0-100000+ across v1/v2/v3.

## Attempts
| # | Strategy | Configs | Best Metric | Notes |
|---|----------|---------|-------------|-------|
| 1 | v1 random inits | 1000+ | 2.6359830849 | Same basin |
| 2 | v1 perturbations | 600 | 2.6359773948 | Slightly worse alternative |
| 3 | v3 Apollonius | 200 | 2.6279 | Different topology, worse |
| 4 | v3 Stripes | 80 | 2.6311 | Different topology, worse |
| 5 | v3 Cantrell-style | 200 | 2.6360 | Same basin |
| 6 | v3 Max-hole | 200 | 2.6293 | Different topology, worse |
