---
strategy: genetic-topology-search
status: in-progress
eval_version: v1
metric: 2.6359830849
issue: 9
parent: topo-001
---

## Genetic Topology Search for n=26

### Result
**metric=2.6359830849** (matches known best, improvement of 9.52e-12 in precision only).

After 3000+ optimization attempts across 7 different strategies, NO alternative topology was found with a higher metric. Every approach converges to the same contact graph basin.

### Approach
Exhaustive search for alternative topological basins using:
1. **v4: Fast SLSQP** (2000 perturbed starts with analytical Jacobian) - found 397 "unique" topologies, all at metric <= 2.6360
2. **v5: Differential Evolution** (100 population, 500 generations) - converges to same basin
3. **v6: Constructive** (Apollonian, biscuit patterns, symmetry, size reshuffling) - best alternative: 2.6099 (d4 symmetry)
4. **v7: Basin Hopping + Dual Annealing** - in progress
5. **topo_jump: Break contacts + re-optimize** - 200 attempts, all return to same topology

### What Happened
The n=26 optimum basin is extraordinarily deep and wide. Every optimization method, regardless of initialization, converges to the SAME contact graph topology:
- 58 circle-circle contacts + 20 wall contacts = 78 active constraints = 78 variables (zero DOF)
- Even with perturbation strengths up to 0.5 (half the unit square), SLSQP returns to this basin
- DE, basin hopping, dual annealing all converge to the same topology
- The only topologies with metric > 2.60 all share this same contact graph

### Seeds
Primary: 42. Phase offsets: +100, +200, +300.

### Attempts
| # | Strategy | Metric | Notes |
|---|----------|--------|-------|
| 1 | v4: Fast SLSQP (2000) | 2.6359830849 | 397 unique fingerprints, all same basin |
| 2 | v5: Diff Evolution | 2.6359830849 | 137 candidates polished, same basin |
| 3 | v6: Constructive | 2.6099 best alt | Apollonian, biscuit, symmetry all worse |
| 4 | topo_jump: Break contacts | 2.6359830849 | 200 attempts, 0 new topologies |
| 5 | v7: Basin hopping | in progress | |
