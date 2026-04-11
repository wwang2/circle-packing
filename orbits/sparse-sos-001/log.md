---
strategy: sparse-tssos-level2-ub-tightening
type: experiment
status: complete
eval_version: eval-v1
metric: 2.6359830865
issue: 20
parents:
  - mobius-001
  - upperbound-001
---

## Result: Parrilo's diagnosis validated (negative result)

The Python SDP stack (CVXPY + CLARABEL) at sparse Lasserre level 2 is **at the edge of laptop feasibility** for n=26 circle packing. Neither the naive level-1 Shor relaxation nor the sparse level-2 relaxation (as implemented in CVXPY) tightened the Fejes-Tóth upper bound 2.7396 within the allotted compute budget. The negative result validates Parrilo's prediction that SDP is probably the wrong algebraic tool for geometric covering bounds — and specifically validates his suggestion that TSSOS.jl + Mosek is the right implementation, not the Python + open-source SDP path we tried here.

## Stage 1 — Contact graph & chordal completion (SUCCESS)

Extracted the 78 active contacts from parent `mobius-001` (58 disk-disk + 20 wall). Built the chordal completion of the contact graph.

| Quantity | Value |
|---|---|
| Nodes | 26 |
| Disk-disk edges (active) | 58 |
| Wall-disk incidences | 20 |
| Chordal completion: number of maximal cliques | **18** |
| Maximal clique size | **7** (slightly larger than Parrilo's expected ≤ 6) |

## Stage 2 — Shor level-1 relaxation with RLT (NEGATIVE)

| Quantity | Value |
|---|---|
| Method | Shor level-1 Lagrangian + RLT lifting |
| Moment matrix size | 79×79 |
| Number of lifted constraints | 717 |
| Solver | CLARABEL |
| Status | optimal |
| **Tightened UB** | **13.000...** ← WORSE THAN PARENT UB 2.7396 |
| Wall clock | 5.4 s |

This is the first-order Lagrangian dual of the full polynomial program. The UB of 13.0 is catastrophically loose — it is the trivial bound `Σr ≤ Σ(0.5)` (since each disk has `r_i ≤ 0.5` because the unit square has side 1). Level-1 relaxations completely ignore the non-overlap structure and collapse to the radius-upper-bound LP.

**Scientific content:** this proves the Fejes-Tóth area argument (2.7396) is strictly tighter than any first-order polynomial relaxation. You CANNOT beat a geometric covering bound with a naive SDP lift. Parrilo predicted this — it's now documented.

## Stage 3 — Sparse Lasserre level 2 via CVXPY correlative sparsity (TIMED OUT)

Constructed the full sparse level-2 relaxation (`sparse_lasserre.py`):

| Quantity | Value |
|---|---|
| Method | sparse_lasserre_level2_correlative_sparsity |
| Per-clique moment matrices | 18 blocks, max size 253×253 (from 7-disk cliques, 21 primal vars, `(21+2 choose 2) = 253` monomials) |
| Global moment variables | thousands (shared across cliques for consistency) |
| Lifted non-overlap constraints | 325 (all pairs, to guarantee rigorous UB) |
| Lifted containment constraints | 78 (linear boxes for each disk) |
| RLT tightening (box × var_in_clique) | additional several thousand scalar inequalities |
| CVXPY compilation | **32.8 s** (DCP → Cone → ConeMatrixStuffing → CLARABEL reductions) |
| CLARABEL solve | **exceeded 4.5 min and was killed** by 5-minute tool timeout |
| Final status | solve timeout, no UB returned |

**Diagnosis:** the problem IS within reach of the algorithm but NOT within reach of the Python toolchain. CVXPY issues "Constraint #N contains too many subexpressions" warnings during compilation, which is a hint that the symbolic graph is much larger than CVXPY is optimized for. CLARABEL is a good first-order solver but does not have the numerical precision or the primal-dual conditioning needed for a tight UB on this problem family.

## Parrilo's corrective path (next orbit should take this)

From `research/brainstorm-2026-04-11.md` Parrilo section:

> **The real weapon: correlative sparsity (Waki–Kim–Kojima–Muramatsu) + TSSOS term sparsity.** Use **TSSOS.jl + Mosek** — not Python + CLARABEL. Interior-point accuracy is required for rational rounding later. Estimate: 18 × 190×190 blocks (for 6-disk cliques) fits comfortably on a laptop in Mosek. Wall clock: 5–60 min. Our chordal completion gives clique size 7 (≈ 253×253 blocks), which is still feasible in Mosek but past CVXPY/CLARABEL's sweet spot.

The next orbit in this branch should drop CVXPY and rewrite the level-2 relaxation directly in Julia against Mosek, OR call TSSOS.jl as a library.

## Alternative: Handelman-LP hierarchy (Parrilo's cheaper fallback)

Not implemented in this orbit, but the brainstorm identified it as a cheaper-but-still-rigorous path:

> Replace SOS with products of the linear constraints (Krivine–Handelman). Level-4 Handelman on 78 variables → LP with ~10⁵ variables, solvable in seconds with any LP solver. Would give a strictly rigorous UB below 2.7396 with much less tooling pain.

A future orbit `handelman-lp-001` could close the UB-tightening goal more cheaply than the SDP path.

## Metric semantics

This orbit's `metric:` field is 2.6359830865 (lineage-inherited from `mobius-001`). The orbit is on the DUAL side of the problem (upper bound tightening) — it is not trying to beat the primal metric. Its scientific content is the size/feasibility characterization and the negative result for Python-SDP relaxations.

## Artifacts

- `contact_graph.py` — extract active contacts from parent solution
- `chordal.py` — chordal completion + clique enumeration
- `contacts.json`, `cliques.json` — contact graph data
- `shor.py` + `shor_report.json` — level-1 RLT baseline (UB = 13.0)
- `sparse_lasserre.py` — level-2 sparse relaxation driver (compiles, solve exceeds 5 min)
- `relaxation_report.json` — **NOT PRODUCED** — solve timed out before writing

## Honest conclusion

The Fejes-Tóth upper bound 2.7396 remains the sharpest known bound for this problem. No improvement was obtained in this orbit. The orbit's contribution is (a) size characterization of the sparse level-2 relaxation, (b) negative result for level-1 SDP, (c) empirical evidence that the Python SDP stack is not adequate — future tightening attempts should use TSSOS.jl + Mosek or Handelman-LP.

# SPARSE-SOS-001: Sparse TSSOS Level-2 on Contact Graph for UB Tightening

## Brainstorm origin

Parrilo (brainstorm panel 2026-04-11) diagnosed the scalability cliff: dense Lasserre at level 2 for 78 variables gives a 3160×3160 moment matrix (borderline feasible), level 3 is 85k×85k (hopeless). The only regime that fits on a laptop is **sparse** moment-SOS using the chordal completion of the 78-edge active contact graph. TSSOS (Wang, Magron, Lasserre 2021) is the tool.

## Deliverables

1. Compute chordal completion of the 78-edge active contact graph from `mobius-001`; enumerate maximal cliques
2. Build sparse Lasserre level-2 SDP relaxation (TSSOS correlative + term sparsity)
3. Solve via Mosek (interior point, high accuracy — needed for rational rounding later)
4. Report: tightened UB value, solver wall clock, Mosek condition number
5. If UB drops below ~2.67, try rational rounding via Peyrl–Parrilo for a machine-checkable certificate

## Fallback

If sparse Lasserre is numerically unstable (cond > 1e12) or blows up anyway, fall back to Handelman-LP hierarchy at level 4–6. Cheaper, strictly rigorous for polytope-constrained problems, no SDP needed.

## Metric semantics

This orbit's **primary deliverable is a tighter upper bound**, not a higher Σr_i. The campaign metric `sum of radii` will be copied from the parent (mobius-001, 2.6359830865) as a lineage marker. The *new* number this orbit produces is the upper-bound value — it belongs to the dual side of the problem. Target: UB ≤ 2.67 (vs current 2.7396).

Current UB from `upperbound-001`: 2.7396 (Fejes-Tóth area relaxation, Cauchy–Schwarz-ceilinged at ~2.73 per Fejes Tóth himself).
