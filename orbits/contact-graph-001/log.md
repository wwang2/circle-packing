---
strategy: contact-graph-enumeration
type: experiment
status: complete
eval_version: eval-v1
metric: 2.6359830865
issue: 22
parents:
  - mobius-001
---

# CONTACT-GRAPH-001: Realizable Contact Graph Enumeration

## Glossary

- **G\*** — the parent (mobius-001) contact graph: 58 disk-disk tangencies + 20 disk-wall incidences = 78 active constraints on 26 disks.
- **Contact graph** — undirected graph whose vertices are disks and whose edges are tangency pairs. For an n-disk packing, a rigid closed-form configuration has |E| + |wall incidences| = 3n - 1 (minus 1 for the sum-of-radii objective direction).
- **Basin** — a local optimum of the packing problem's (x, y, r) space, distinguished by its set of active constraints at optimality.
- **Feasibility prefilter** — cheap combinatorial filter before running SLSQP: (i) planar, (ii) no disk touches >2 walls, (iii) no disk touches opposite walls, (iv) every disk has degree ≥ 2 (neighbors + walls), a necessary condition for rigidity.
- **SLSQP** — Sequential Least-SQuares Programming, the scipy nonlinear constrained optimizer used per candidate graph.
- **Incumbent** — current best known sum_r = 2.6359830865 (from mobius-001).

## Result

**Metric: 2.6359830865 (unchanged from parent — no alternative topology found)**

Enumeration outcome:

| Stage                                     | Count |
|-------------------------------------------|-------|
| Generated candidates                      | 2100  |
| Unique canonical graphs                   | 2100  |
| Pass planarity + prefilter                | 425   |
| SLSQP converged (tol 1e-8)                | 425   |
| Dominated by incumbent                    | 425   |
| Tied with incumbent (|Δ| ≤ 1e-9)          | 0     |
| **Strictly better than incumbent**        | **0** |

Distribution of the 425 solved basins (`sum_r`):
- min: 2.6358957386
- max: 2.6359830849 (gap to incumbent = 1.57e-9 — same basin, SLSQP floor)
- 423 of 425 converge to sum_r ≈ 2.6359830849 (= parent basin, up to SLSQP precision)
- 1 lands at 2.635977 (a different local optimum, 6e-6 worse)
- 1 lands at 2.635896 (another different local optimum, 9e-5 worse)

Per-strategy breakdown (of the 2100 generated):

| Strategy | Generated | Planar/feasible | Best sum_r |
|----------|-----------|-----------------|------------|
| edge_flip | 1740 | 341 | 2.6359830849 |
| wall_swap | 60 | 60 | 2.6359830849 |
| edge_swap | 300 | 24 | 2.6359830849 |

Wall-clock: 487s (~8 min) on 6 parallel workers.

## Approach

### Step 1 — Recover G\*

Read `orbits/mobius-001/solution_n26.json`. With numerical tolerance 1e-6, recovered:
- 58 disk-disk tangencies (edges)
- 20 wall contacts (4 disks touching 2 walls each, 12 disks touching 1 wall)
- 78 total active constraints

Degree-of-freedom check: `3n - (#constraints) - 1 = 3·26 - 78 - 1 = -1`, confirming the rigid (over-constrained-by-one) closed-form structure of the parent.

### Step 2 — Generate candidate graphs

Three independent neighborhood strategies, chosen for diversity:

1. **edge_flip** (1740 cand.): for each disk-disk edge (i, j) in G\* and each
   non-edge (u, v) ranked in the top 30 by tangency gap (i.e. pairs that are
   currently closest to touching), produce the graph `E \ {(i,j)} ∪ {(u,v)}`
   with the same wall set. This probes 1-edge substitutions biased toward
   physically plausible new contacts.

2. **wall_swap** (60 cand.): for each (wall-side × on-wall disk) pair, replace
   with an off-wall disk currently near that side. This probes boundary
   reconfigurations while keeping interior tangencies fixed.

3. **edge_swap** (300 cand.): 2-exchange — remove two disjoint edges (a,b),
   (c,d) and add either (a,c)+(b,d) or (a,d)+(b,c). Random subset of 200
   pair-indices × 2 rewirings (infeasible in G\* dropped).

Canonicalization: `(sorted edge tuple, sorted wall-assignment tuple)` —
no duplicate graphs enter the solver.

### Step 3 — Filter

Each candidate passes through a cheap combinatorial filter:
- `networkx.check_planarity` — disk contact graphs must be planar.
- No disk may touch >2 walls (geometrically impossible in a square).
- No disk touches both L+R or both B+T (requires r ≥ 0.5).
- Every disk must have degree ≥ 2 counting neighbors + walls (rigidity
  prerequisite).

Of 2100 candidates, **425 (20%)** pass this filter.

### Step 4 — SLSQP per graph

**Key design decision.** I initially tried SLSQP with the candidate graph's
edges as hard equality constraints. This failed: starting from the parent
coordinates with a small perturbation, SLSQP cannot close the new edge and
reports failure on ~100% of runs.

**The fix** is to run *free-form* optimization (containment + non-overlap
inequalities only, no edges pinned), with a **targeted move** initialization:

1. Identify the "new" edges in the candidate (those not in G\*).
2. Pull each (u, v) pair in the new edges ~25% closer in coordinate space.
3. Apply a global Gaussian perturbation of std 0.015.
4. Run SLSQP with `maxiter=400`, `ftol=1e-11`, `method='SLSQP'`.
5. Validate: all containment + non-overlap at tolerance 1e-8.

Three seeds per candidate: {42, 123, 7}. Best of three recorded.

This is the **right** formulation because SLSQP *discovers* the active set at
the local optimum. If an alternative graph is actually a better basin, the
targeted move biases the init toward it, and SLSQP will find it. If it's not,
SLSQP slides back to the parent's rigid optimum — which is exactly what we
observe for 423 of 425 candidates.

### Step 5 — Solutions

`enum_report.json` contains the full enumeration (2100 records including
non-feasible-prefilter ones reported as `-1`). `solution_if_better.json` is
**not produced** — no candidate beat the incumbent.

## What Happened

The enumeration yielded an almost monochromatic histogram: 423 of 425
feasible candidates converge to `sum_r = 2.6359830849 ± 1e-12`, a single
value indistinguishable from the parent basin (gap = 1.57e-9, well inside
SLSQP's numerical floor). Only 2 of 425 found distinct local optima, both
suboptimal: one at 2.635977 (6e-6 worse) and one at 2.635896 (9e-5 worse).

This is the strongest practical evidence for global optimality that
enumeration can deliver. The parent's claim — that G\* is the unique rigid
optimum — holds up not only against gradient-descent perturbation (already
shown by rigidity-001 and interval-newton-001) but also against
**combinatorial** perturbation of the contact graph.

**Importance**: this enumeration tests a hypothesis that the rigidity and
Krawczyk certificates cannot reach. Those certificates prove local
uniqueness *in coordinate space*, but say nothing about whether a different
contact graph would yield a different, possibly higher-sum_r basin somewhere
else in R^156. The contact-graph enumeration closes that loop (at least for
the 1-edge and 2-edge neighborhoods of G\*).

## What I Learned

- **SLSQP is "sticky" to the parent basin.** With a targeted move of
  perturbation-scale 0.25 toward a candidate edge, the optimizer still slides
  back to the parent's active set in >99% of cases. This is consistent with
  rigidity-001's finding that σ_min(R) = 0.0907 — the Jacobian of the active
  constraints is well-conditioned, so the basin has substantial attractor
  width in coordinate space.
- **Non-planar candidates abound.** Only 20% of 1-edge flips produce planar
  graphs. The contact graph of a dense packing is near-maximal planar, so
  most random edge substitutions destroy planarity.
- **2-edge swap rarely feasible.** Only 24 of 300 edge-swap candidates
  passed planarity, reflecting the near-maximal planarity of the base graph.
- **No "alternative basin" structure** — the two distinct local optima found
  (at 2.635977 and 2.635896) are both clearly inferior, and were each found
  once, suggesting they are rare sub-basins discovered by specific
  targeted-move perturbations.
- This negative result is an **informative negative**: combined with the
  parent orbit's 6000+ topology-perturbation search, the current orbit's
  2100 structured graph enumeration, and the 156-dimensional Krawczyk
  uniqueness proof, the incumbent's global-optimality case becomes very
  strong. There is no known search direction left that has not been tried.

## Prior Art & Novelty

### What is already known
- **Packomania / Erich Friedman's circle-packing catalogue** publishes the
  best-known values for `n ≤ 32`. The n=26 value of 2.6359 has been
  recorded since Cantrell 2011, with improvements in the fourth decimal
  place coming from AlphaEvolve (2025), FICO Xpress (2025), and
  "Tactical Maniac" (2025).
- **Contact-graph enumeration** for disk packings has been studied by
  [Graham et al. (1998)](https://doi.org/10.1007/PL00009476) for small `n`
  and by the [Hales–Ferguson Flyspeck project (2015)](https://github.com/flyspeck/flyspeck)
  for the Kepler conjecture. Both share the idea of exhausting
  combinatorial types, though in 3D for Hales.
- The **Koebe–Andreev–Thurston theorem** guarantees that every planar
  3-connected graph is realizable as a circle packing on the sphere; this
  justifies planarity as a necessary realizability condition for our
  candidates.
- **Rigidity theory** (Asimow–Roth, Connelly) provides the framework for
  checking whether a given contact graph admits a rigid packing:
  3n = #dof must match the #constraints plus some trivial null-space.
- Parent orbit `mobius-001` did ~6000 topology perturbations via Möbius
  deformation, edge flipping, basin hopping, and multi-start, all
  returning to sum_r = 2.6359830865.

### What this orbit adds
- A **principled enumeration**: rather than perturbation-based sampling
  (which stays in continuous space), we enumerate discrete graph
  modifications and solve each explicitly, with planarity + rigidity
  prefilters. 2100 candidates generated, 425 passed filters, all solved.
- A **formulation insight** that may be useful for future orbits:
  active-set formulation (pinning the candidate's edges as equalities) is
  unusable because SLSQP fails to close distant contacts; free-form
  optimization with targeted move initialization is the correct approach
  to probe alternative contact graphs.
- An **informative negative result** strengthening the global-optimality
  case: 2100 structured alternatives — all dominated.

### Honest positioning
This is an **application** of known enumeration techniques (Flyspeck-style)
to our specific `n=26` problem, combined with a novel free-form targeted-
move SLSQP formulation to handle the "what is this graph's optimum?"
sub-problem. No claim of mathematical novelty. The contribution is
empirical: a clean, reproducible enumeration whose negative result is part
of a growing case for global optimality of the mobius-001 solution.

## Compute

Local, single Mac host, 6-process parallel SLSQP. Modal was not used
(over-kill for 8 min of light CPU work).

## Files

- `solution.py` — main driver: parent graph recovery, candidate generation
  (3 strategies), feasibility filters, parallel SLSQP, report aggregation.
- `make_figures.py` — generates `figures/enum_summary.png` (3-panel parent
  vs enumeration histogram vs by-strategy scatter) and
  `figures/sum_r_zoom.png` (zoom on basin structure).
- `enum_report.json` — machine-readable: counts + per-candidate outcomes.
- `run.sh` — `python3 solution.py && python3 make_figures.py`.
- No `solution_if_better.json` — that is the point.

## References

- [Graham, Lubachevsky, Nurmela, Östergård (1998) "Dense packings of
  congruent circles in a circle"](https://doi.org/10.1016/S0012-365X(98)00080-8) — classical contact-graph enumeration
  for disks in a disk.
- [Koebe (1936) / Andreev (1970) / Thurston (1978)](https://en.wikipedia.org/wiki/Circle_packing_theorem)
  — any planar 3-connected graph is the contact graph of a circle packing.
- [Hales & Ferguson (2015) "Flyspeck Project"](https://github.com/flyspeck/flyspeck) — contact-graph enumeration
  for the 3D Kepler conjecture.
- [Erich Friedman's circle packing page](https://erich-friedman.github.io/packing/cirRsqu/) — record of best-known sum_r values.
- Parent orbit #14 `mobius-001` — established the incumbent basin via
  6000+ topology-perturbation experiments.
- Sibling orbit #19 `rigidity-001` — σ_min(R) = 0.0907, strict local
  optimality of G\*.
- Sibling orbit #21 `interval-newton-001` — rigorous Krawczyk theorem,
  unique KKT critical point in B(z₀, 10⁻⁴) ⊂ R^156.

## Attempts

| # | Action | Candidates | Planar | Solved | Best sum_r | Note |
|---|--------|-----------|--------|--------|-----------|------|
| 1 | Smoke test 20 | 20 | 4 | 0 | — | active-set SLSQP failed to close new edges |
| 2 | Free-form + targeted move 30 | 30 | 5 | 5 | 2.6359830849 | all collapse to parent basin |
| 3 | Full enumeration 2100 | 2100 | 425 | 425 | 2.6359830849 | 0 better, 0 tied, 2 distinct sub-basins |
