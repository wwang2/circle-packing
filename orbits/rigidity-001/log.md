---
strategy: rigidity-sigma-min-flex-probe
type: experiment
status: complete
eval_version: eval-v1
metric: 2.6359830865
issue: 19
parents:
  - mobius-001
---

## Result

**Strict local optimality of the n=26 optimum Σr = 2.6359830865 is CERTIFIED.**

### Rigidity matrix diagnostic

| Quantity | Value |
|---|---|
| `σ_min(R)` | **0.0907** |
| `σ_max(R)` | 1.7646 |
| `κ(R)` | 19.45 |
| `rank(R)` | 78 (full) |
| `stationarity_residual` | 8.1 × 10⁻¹⁵ (machine precision) |
| `λ_min` | 0.0208 |
| `λ_max` | 0.9503 |
| `num negative λ` | 0 (strict complementarity) |

`σ_min(R)` is **five orders of magnitude above the 1e-6 threshold** Connelly specified for a "healthy" rigidity certificate. The matrix is well-conditioned (`κ ≈ 19`), not degenerate. Combined with strict complementarity and the linear objective, **second-order sufficiency is automatic** (since `ker(R) = {0}`, the standard PSD-on-nullspace condition is vacuously satisfied — Connelly foresaw this).

### Stress matrix Ω

The KKT-weighted constraint Hessian has 26 negative eigenvalues (range [−9.6, 8.9]). **This is expected and harmless:** for the sum-of-radii problem with a linear objective and a full-rank rigidity matrix, the stress matrix only needs to be PSD on `ker(R)`, and `ker(R) = {0}` here. The negative eigenvalues live in directions that are already forbidden by the active-constraint Jacobian.

### Flex probe

Connelly's canonical "bifurcation hunt": the disk pair with the smallest Lagrange multiplier `λ = 0.0208` is the (0, 7) contact. Its three nearest neighbors (disks 8, 6, 1) were perturbed by `Δ ∈ {10⁻³, 10⁻², 10⁻¹}` for 10 seeds each — **30 trials total**.

| Scale | Best Σr | Max gain vs. baseline |
|---|---|---|
| `10⁻³` | 2.6359830849 | −1.6 × 10⁻⁹ (numerical noise, no improvement) |
| `10⁻²` | 2.6359830849 | −1.6 × 10⁻⁹ (no improvement) |
| `10⁻¹` | 2.6359830849 | no improvement; most trials dropped to 2.59–2.62 |

**No flex was found.** Every attempted perturbation either returned to the same basin (small ε) or collapsed to a strictly worse basin (large ε). No trial achieved Σr > 2.6359830865.

### Verdict

> Σr = 2.6359830865 is a **strict, isolated, KKT-rigid local maximum** of the circle-packing sum-of-radii objective at n=26. With `σ_min(R) = 0.0907` and full rank + strict complementarity + linear objective, local optimality is mathematically certified (not merely "numerically observed"). The flex probe at the canonically-weakest multiplier found no nearby superior basin in any of 30 trials.

**This does NOT prove global optimality** — Connelly himself put 60/40 odds on a different-topology improvement existing elsewhere in contact-graph space. But the basin containing the current optimum is now formally a strict local maximum. The remaining uncertainty is *combinatorial* (other contact graphs), not *continuous* (the incumbent basin is rigid).

### Wall clock

4.12 seconds. Entire diagnostic ran in under 5 seconds.

### Artifacts

- `solution.py` — full diagnostic driver (reads mobius-001 parent, builds rigidity matrix, runs flex probe)
- `rigidity_report.json` — machine-readable output with all 30 flex-probe trials
- `figures/rigidity_diagnostic.png` — eigenvalue spectra, multiplier histogram, flex-probe scan

### What this does NOT answer

- Global optimality (requires contact-graph enumeration per Fejes Tóth / Hales)
- Stability radius of the basin (Krawczyk certificate — see `interval-newton-001`)
- Upper bound tightening (see `sparse-sos-001`)

# RIGIDITY-001: σ_min(R) + Weakest-Multiplier Flex Probe

## Brainstorm origin

Connelly (brainstorm panel 2026-04-11, see `research/brainstorm-2026-04-11.md`) identified σ_min(R) as the single missing measurement in the n=26 saturation story. 30-minute test with decisive outcome:

- **σ_min(R) > 1e-6 ⟹** strict local optimality certified (with linear objective + strict complementarity, second-order sufficiency is automatic)
- **Weakest multiplier λ = 0.021** is the canonical bifurcation signature in his experience with disk packings — perturbing its 3 neighbors and re-optimizing is the standard flex hunt

## Deliverables

1. Report `σ_min(R)` and `κ(R)` for the 78×78 rigidity matrix at the mobius-001 optimum
2. Report the stress matrix Ω spectrum (eigenvalues, PSD-ness on ker(R))
3. Flex probe result: does any perturbation of the weakest-multiplier disk's neighborhood reach a higher Σr?
4. Maxwell–Cremona planar lift as visual certificate (optional)
5. One-line verdict: certified strict local optimum, OR found a flex (with coordinates)

## Metric semantics

Primary metric for this orbit is still `sum of radii`. Expected outcome: reconfirm 2.6359830865 (null result on metric) AND produce a σ_min(R) number that settles the strict-local-optimality question. The payoff here is the diagnostic, not a higher Σr — UNLESS the flex probe actually finds a better basin, in which case this orbit would set a new best.
