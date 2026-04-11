---
strategy: krivine-handelman-lp-ub
type: experiment
status: in-progress
eval_version: eval-v1
metric: 2.6359830865
issue: 23
parents:
  - upperbound-001
---

# HANDELMAN-LP-001: Krivine–Handelman LP Hierarchy for UB Tightening

Parrilo's cheap rigorous fallback after `sparse-sos-001` (#20) demonstrated the Python SDP stack is inadequate. Krivine–Handelman Positivstellensatz gives a polytope-constrained LP hierarchy that can match low-level Lasserre at a fraction of the cost.

Parent `upperbound-001` (#13) gives the current sharpest known UB: **2.7396** (Fejes-Tóth area relaxation, Cauchy–Schwarz-ceilinged at ~2.73).

Target: any UB < 2.7396 is progress; < 2.68 would close most of the gap to the empirical 2.636.

## Deliverables

1. Krivine–Handelman lifted LP at levels k ∈ {2, 3, 4, 6}
2. Tightened UB per level, with wall clock and LP size stats
3. Primal-dual certificate at best level (rational rounding via `fractions.Fraction` if tight)
4. Report: `{level, lp_size, tightened_ub, wall_clock, solver, gap_reduction}`

## Metric semantics

This orbit is on the DUAL side. `metric:` inherits parent primal as lineage marker. The real deliverable is `tightened_ub` — should be strictly less than 2.7396.

## Autorun override

Minimum 3 random orderings of constraint lifting per level to guard against pathological solver behavior.
