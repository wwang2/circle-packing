# Circle Packing in a Unit Square: Maximize Sum of Radii

## Research Question

Given a positive integer $n$, pack $n$ disjoint circles inside a unit square $[0,1]^2$ so as to **maximize the sum of their radii**.

Formally: find centers $(x_i, y_i)$ and radii $r_i > 0$ for $i = 1, \ldots, n$ such that:

1. **Containment:** Each circle lies entirely within the unit square:
   - $r_i \leq x_i \leq 1 - r_i$
   - $r_i \leq y_i \leq 1 - r_i$

2. **Non-overlap:** All circles are disjoint (non-intersecting):
   - $(x_i - x_j)^2 + (y_i - y_j)^2 \geq (r_i + r_j)^2$ for all $i \neq j$

3. **Objective:** Maximize $\sum_{i=1}^{n} r_i$

## Background

This is a classic nonconvex quadratically constrained optimization problem (QCQP) with many local optima. The non-overlap constraints make it NP-hard. The problem has been studied extensively by David W. Cantrell and Eckard Specht, with results catalogued on [Erich Friedman's page](https://erich-friedman.github.io/packing/cirRsqu/).

## Known Results (selected)

| n | Best known sum of radii | Source |
|---|------------------------|--------|
| 1 | 0.500 | Trivial |
| 10 | 1.591+ | Cantrell 2011 |
| 20 | 2.301+ | Cantrell 2011 |
| 26 | ~2.6359 | Multiple improvements over Cantrell's 2.634 |
| 30 | 2.842+ | Cantrell 2011 |
| 32 | 2.939+ | Berthold et al 2026 |

## Recent Advances (2025-2026)

The n=26 case has become a benchmark for AI-driven optimization:

- **AlphaEvolve** (DeepMind, 2025): 2.63586 — LLM-guided program evolution
- **FICO Xpress** (2025): 2.63592 — global MINLP solver
- **"Tactical Maniac"** (student, 2025): 2.63593 — multi-agent discovery, novel arrangement
- **OpenEvolve** reproduction: 2.63598 — multi-initialization + scipy optimization

For n=32, AlphaEvolve improved from 2.936 to 2.937, and further improvements to 2.939 have been recorded.

## Approach Space

Known effective approaches include:

1. **Nonlinear optimization** — formulate as QCQP, use solvers (SLSQP, L-BFGS-B, IPOPT)
2. **Global optimization** — branch-and-bound, interval arithmetic (FICO Xpress)
3. **Evolutionary strategies** — evolve initialization patterns or full packing programs
4. **Multi-start local search** — many random/structured initializations + local refinement
5. **Topology search** — enumerate contact graph topologies, then optimize within each
6. **Simulated annealing / basin-hopping** — escape local optima via perturbation

## Success Criteria

- **Primary metric:** Sum of radii (maximize)
- **Target:** Beat the current best known values for one or more n values
- **Focus values:** n ∈ {26, 32} (most actively contested), but improvements at any n ≤ 32 are valuable
- **Validation:** All constraints (containment + non-overlap) must be satisfied to machine precision (tolerance < 1e-10)
