---
strategy: contact-graph-topology-search
status: complete
eval_version: v1
metric: 2.6359830849
issue: 7
parent: nlp-001
---

## Contact Graph Topology Search for n=26

### Result
**metric=2.6359830849** -- matches current best (sa-001). KKT analysis confirms this is a verified local optimum with zero degrees of freedom.

### Approach
Systematic exploration of alternative topological basins via:
1. Contact graph analysis (58 circle-circle + 20 wall contacts, all gaps < 3e-10)
2. 500+ diverse initializations (rings, hex, random, Poisson disk, mixed-size, greedy constructive)
3. Basin hopping with topology-changing moves (swaps, reflections, rotations, cycles)
4. Remove-and-reinsert strategy (remove 1-3 smallest circles, re-optimize, re-add)
5. Multiple optimizers: SLSQP, COBYLA, trust-constr, L-BFGS-B (penalty), dual_annealing
6. Analytical Jacobian for faster SLSQP convergence
7. Wall topology enumeration (200+ wall-contact configurations)
8. Structural patterns (zigzag, cross, asymmetric, 3-cluster, edge-packing, corner-focused)
9. KKT system refinement via Newton's method (fsolve)

### What Happened
Every approach converges to the same topological basin. The KKT analysis is definitive:
- 78 active constraints (58 circle-circle + 20 wall) with 78 primal variables (26*3)
- All dual variables strictly positive (range [0.021, 0.950])
- KKT residual = 1.03e-15 (machine precision)
- System is fully determined: zero degrees of freedom at this optimum

No alternative topology found with metric > 2.636.

### What I Learned
- The n=26 optimal topology is a VERY deep, unique basin. The basin of attraction is enormous.
- Ring-based inits (1+8+12+4+1) reach ~2.636; all others plateau at 2.55-2.63.
- The solution is fully rigid: 78 active constraints = 78 variables, zero DOF.
- Improvement 2.6359830823 -> 2.6359830849 is purely numerical (2.59e-09).
- Beating this requires a fundamentally different contact graph topology.
- The "Tactical Maniac" result (2.63593) may have a similar topology with slightly worse precision.
- Global optimizers (dual_annealing) are too slow for 78D to be practical.

### Seeds
All seeds documented in individual scripts. Primary: 42, 123, 456, 789, 999, 7777, 9999.

### Attempts
| # | Strategy | Metric | Notes |
|---|----------|--------|-------|
| 1 | Fast penalty (300 inits) | 2.6359830823 | All converge to same basin |
| 2 | Search v1 (basin hopping) | 2.6359830849 | Numerical polish only |
| 3 | Precision (analytical Jac, 200 hops) | 2.6359830849 | Same basin |
| 4 | Multi-optimizer (COBYLA, trust-constr) | 2.6359830849 | All optimizers agree |
| 5 | Radical (greedy, maxhole, patterns) | 2.6265 max | Different topologies all worse |
| 6 | Wall topology (200 configs) | 2.6359830823 | No new basin found |
| 7 | Remove-and-reinsert (all 26 circles) | 2.6359830823 | Converges back to same |
| 8 | KKT refinement (Newton on KKT system) | 2.6359830849 | Confirms optimality |
