---
strategy: krawczyk-interval-newton-certificate
type: study
status: partial
eval_version: eval-v1
metric: 2.6359830865
issue: 21
parents:
  - mobius-001
---

## Status: PARTIAL (needs cleanup — Krawczyk step missing)

This orbit completed Stage 1 of the Hales plan (KKT point recovery + interval arithmetic scaffold) but did NOT complete the Krawczyk contraction check. The deliverable — a rigorous local theorem via interval-Newton — is not yet produced. A cleanup iteration is needed to finish the Krawczyk step.

## What was completed

### Stage 1 — KKT point recovery (`recover_kkt.py` → `kkt_point.json`)

Recovered the full primal+dual state at the n=26 empirical optimum:

| Quantity | Value |
|---|---|
| `n_primal` | 78 (`x_i, y_i, r_i` for i=1..26) |
| `n_dual` | 78 (58 disk-disk + 20 wall Lagrange multipliers) |
| `n_eqns` | 156 (square KKT system) |
| `sum_r` | 2.6359830865 (matches parent `mobius-001`) |
| `stationarity_residual_inf` | **1.11 × 10⁻¹⁵** — machine precision |
| `active_tol` | 5 × 10⁻⁷ |
| `min |λ|` (disk-disk) | 0.0208 (at contact [5, 0] — matches rigidity-001's (0, 7) after reordering) |

The recovered state cross-validates `rigidity-001`'s findings: identical weakest-contact disks, identical multiplier magnitudes, identical stationarity residual. Two independent recoveries of the KKT point agree to machine precision, which is independent cross-validation of the parent optimum's correctness.

### Stage 2 — Minimal interval arithmetic library (`interval.py`)

Small but correct `Iv` class with `+`, `-`, `*`, point/interval construction, vector helpers, subset/contains-zero predicates. Uses a 4-ulp safety widening per operation (conservative outer bound on roundoff). Good enough for a first Krawczyk attempt at ε = 10⁻⁸; not publication-quality — for that you'd switch to `IntervalArithmetic.jl` or `boost::numeric::interval` with hardware directed rounding.

## What is MISSING (needs cleanup)

- **F(x, λ) encoding using interval arithmetic.** The 156-dimensional KKT residual function needs to be written so it accepts `Iv` inputs and returns `Iv` outputs.
- **Jacobian enclosure J(B).** Interval enclosure of the 156×156 Jacobian of F over a box B — can be done by automatic differentiation of F via the `Iv` class, or by hand-deriving the analytical Jacobian entries and evaluating each over B.
- **Krawczyk operator.** `K(B) = x₀ − C · F(x₀) + (I − C · J(B)) · (B − x₀)` where `C ≈ J(x₀)⁻¹` is a floating-point preconditioner.
- **Contraction check.** Verify `K(B) ⊂ interior(B)` → by Krawczyk's theorem, there exists a unique solution to `F = 0` in B.
- **ε sweep.** Start at `ε = 10⁻⁸`, expand until contraction fails. Report the maximum verified basin radius.

## Why this stopped at Stage 1

Wall-clock budget of the orbit-agent session was exhausted after recovering the KKT point and writing the interval library but before the full Krawczyk operator was coded. Estimated remaining work: 30–60 minutes of focused coding in a fresh session. All scaffolding is in place — the cleanup agent should go straight to Stage 2 without redoing the recovery.

## Cross-validation finding (unrelated to the Krawczyk goal but worth recording)

The KKT residual of 1.11 × 10⁻¹⁵ (3 × machine epsilon) is stronger empirical evidence than previously reported that `mobius-001`'s 2.6359830865 is a *true* KKT point of the circle-packing problem. Combined with `rigidity-001`'s `σ_min(R) = 0.0907`, the local story is now:

1. **Exact KKT**: stationarity residual is 3× float64 epsilon (two independent recoveries agree)
2. **Full rank**: σ_min(R) = 0.0907, well above 1e-6
3. **Strict complementarity**: all Lagrange multipliers positive, min magnitude 0.0208

These three facts, together, **morally certify strict local optimality** — but without the interval-Newton contraction, it is not yet a *theorem*. The missing Krawczyk step would upgrade "morally certain" to "mathematically proven local" with explicit error bars.

## Artifacts

- `recover_kkt.py` — Stage 1 driver (runs: reads parent, reconstructs KKT system, solves for λ)
- `kkt_point.json` — 78 primal + 78 dual values + contact graph structure
- `interval.py` — minimal interval arithmetic library (Iv class + vector/matmul helpers)
- `run.sh` — (not yet written) — reproducer once Stage 2+ is complete

## Next steps for cleanup

1. Write `krawczyk.py` that imports `interval.py`, codes F(x, λ) and J(B) over `Iv`
2. Implement the Krawczyk operator with `C = J(x₀)⁻¹` (float precondition) and `J(B)` enclosed
3. Contraction test at ε = 10⁻¹⁰, 10⁻⁸, 10⁻⁶, 10⁻⁴
4. Emit `certificate.json` and `certificate.md`
5. Update this frontmatter: `status: partial → complete`

# INTERVAL-NEWTON-001: Krawczyk Rigorous Local Certificate

## Brainstorm origin

Hales (brainstorm panel 2026-04-11) identified this as the fastest honest path from "numerically tight" to "theorem". Lineage: Tucker's Lorenz attractor proof, Flyspeck's local KKT checks, all rely on Krawczyk / interval-Newton contractions.

## Approach

1. Formalize the KKT system `F(x, λ) = 0`:
   - 78 stationarity equations `∂L/∂x_i = 0` (for each of x, y, r coordinates)
   - 78 active constraint residuals (58 disk-disk + 20 wall contacts)
2. Write `F` using Julia's `IntervalArithmetic.jl` + `ForwardDiff.jl` (or `IntervalLinearAlgebra.jl`)
3. Compute Krawczyk operator `K(B) = x₀ - C·F(x₀) + (I - C·F'(B))·(B - x₀)` where C ≈ F'(x₀)⁻¹ and B is a box of radius ε = 1e-6 around the numerical optimum
4. Verify `K(B) ⊆ interior(B)` — contraction mapping ⟹ unique true critical point in B
5. Extract: verified interval for Σr_i at the true critical point

## Deliverable

Rigorous theorem of the form:
> There exists a unique vector `(x̂, ŷ, r̂, λ̂) ∈ B(x*, 1e-6)` satisfying the KKT system for the circle packing problem, with `Σr̂_i ∈ [2.6359830864, 2.6359830866]`.

## Why this is a STUDY not an EXPERIMENT

Type is `study` because the deliverable is a formal/mathematical certificate, not a metric value. The `metric` field will stay null; the output is a verified interval + a Julia notebook that an external prover could re-run.

## Blind spot Hales called out

The coordinates are a **consequence** of the contact graph, not an input. Don't chase better coordinates — certify the graph-structured solution we already have, then the coordinates follow automatically from the KKT system solution.

## Extension (if time permits)

Extend the radius ε until contraction fails → certified basin-of-attraction radius. This gives a quantitative strict-local-optimality result: "no configuration within distance ρ of the optimum has Σr higher."
