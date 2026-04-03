---
strategy: genetic-topology-search
status: complete
eval_version: v1
metric: 2.6359830849
issue: 9
parent: topo-001
---

## Genetic Topology Search for n=26

### Result
**metric=2.6359830849** -- matches known best. No alternative topology found.

After 5000+ optimization attempts across 9 strategies (continuous and discrete), no contact graph topology was found with a higher sum of radii than the known optimum. This strongly suggests the known solution is the global optimum for n=26.

### Approach
Exhaustive search for alternative topological basins using both continuous optimization and discrete topology enumeration.

### What Happened

**Continuous methods (v4-v8):**
- v4: 2000 perturbed SLSQP starts with analytical Jacobian -- 397 unique fingerprints, all in same basin
- v5: Differential evolution (100 pop, 500 gen) -- converges to same basin
- v6: Constructive (Apollonian, biscuit, symmetry, reshuffling) -- best alternative: 2.6286
- v7: Basin hopping + dual annealing -- all step sizes return to same basin or much worse
- v8: KKT refinement iterations -- confirms machine-precision optimality

**Discrete topology search (v9):**
- Enumerated 1591 topological variants: swap 1, 2, or 3 contacts (circle-circle and wall)
- Only 8/1591 variants produced solvable KKT systems
- All 8 solutions had metric <= known best
- The 8 solvable variants were all single-contact swaps

**Key evidence for optimality:**
- 78 active constraints = 78 variables (zero DOF, fully rigid)
- All dual variables strictly positive (range [0.021, 0.950])
- KKT residual = 8.8e-16 (machine precision)
- Basin of attraction covers perturbations up to 50% of the unit square
- Closest non-contact gap is 0.014 (circles 8,18); all others > 0.12

### What I Learned
- The n=26 packing at metric=2.6360 is almost certainly the global optimum
- The basin of attraction is enormous -- no continuous optimizer can escape it
- Discrete topology changes mostly produce unsolvable systems (99.5% failure rate)
- The few solvable alternative topologies all have lower metric
- Beating this requires either (a) a fundamentally new mathematical insight or (b) a combinatorial search over much more distant topologies

### Seeds
Primary: 42. Phase offsets: +100, +200, +300.

### Attempts
| # | Strategy | Metric | Notes |
|---|----------|--------|-------|
| 1 | v4: Fast SLSQP (2000) | 2.6359830849 | 397 unique fingerprints, all same basin |
| 2 | v5: Diff Evolution | 2.6359830849 | 137 candidates polished, same basin |
| 3 | v6: Constructive | 2.6286 best alt | Apollonian, biscuit, symmetry all worse |
| 4 | topo_jump: Break contacts | 2.6359830849 | 200 attempts, 0 new topologies |
| 5 | v7: Basin hopping | 2.6359830849 | All step sizes return to same basin |
| 6 | v8: KKT refinement | 2.6359830849 | Confirms machine-precision optimality |
| 7 | v9: Discrete topo swap-1 | 2.6359830849 | 8/291 solvable, 0 better |
| 8 | v9: Discrete topo swap-2 | 2.6359830849 | 0/500 solvable from swap-2 |
| 9 | v9: Discrete topo swap-3 | 2.6359830849 | 0/500 solvable from swap-3 |
| 10 | v9: Wall+circle swap | 2.6359830849 | 0/300 solvable |
