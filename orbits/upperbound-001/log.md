---
strategy: convex-relaxation-upper-bound
status: complete
eval_version: eval-v1
metric: 2.7396
issue: 13
parent: null
---

# UPPERBOUND-001: Convex Relaxation Upper Bounds

## Result

Best upper bound for n=26: **2.7396** (Fejes Toth). Known solution: 2.636. Gap: 3.9%.

Exact bounds proven for n=1 (0.500), n=2 (0.586), n=4 (1.000).

| n | Best UB | Known LB | Gap | Method |
|---|---------|----------|-----|--------|
| 1 | 0.500 | 0.500 | 0.0% | SOCP (exact) |
| 2 | 0.586 | 0.586 | 0.0% | Pair bound (exact) |
| 3 | 0.879 | 0.765 | 14.9% | SOCP + pair |
| 4 | 1.000 | 1.000 | 0.0% | Top-4 sum (exact) |
| 5 | 1.201 | 1.085 | 10.7% | FT |
| 10 | 1.699 | 1.591 | 6.8% | FT |
| 15 | 2.081 | 2.037 | 2.2% | FT |
| 20 | 2.403 | 2.301 | 4.4% | FT |
| 26 | 2.740 | 2.636 | 3.9% | FT |
| 32 | 3.039 | 2.939 | 3.4% | FT |

## Approach

Systematic exploration of convex relaxation upper bounds:

1. **SDP (Shor relaxation)** with RLT cuts for small n. Exact for n=1 via containment.
2. **SOCP with geometric cuts**: pair bound (r1+r2 <= 2-sqrt(2)), top-4 sum <= 1.0, individual radius bounds from FT.
3. **Fejes Toth** density bound: sum(2*sqrt(3)*r_i^2) <= 1.
4. **Split/counting bounds**: parametric optimization splitting top-k and tail.
5. **Grid-based LP/SOCP**: discretized center positions with pairwise constraints.
6. **Boundary-aware analysis**: investigating whether boundary waste tightens FT.

## What Happened

- **SDP with RLT cuts**: gives area bound for n >= 2. RLT cuts from containment products are too weak; the relaxation allows the equal-radius solution that achieves FT.
- **Pair bound r1+r2 <= 2-sqrt(2)**: exact for n=2. Derived from the constraint that two circles in a unit square have max pairwise distance sqrt(2)*(1-r1-r2) >= r1+r2.
- **Top-4 sum <= 1.0**: exact for n=4 (four circles at corners, r=0.25 each). Valid because the n=4 optimal is provably 1.0.
- **For n >= 5**: all geometric constraints are slack at the FT-optimal equal-radius allocation. The individual radii (~0.1 for n=26) are far below the pair/containment thresholds.
- **Grid-based approaches**: suffer from LP relaxation gap; spread tiny radii across many grid points.
- **Oler-Groemer bounds**: discovered that previous Oler SOCP implementation was INVALID (dropped the linear 4r boundary term). Correctly applied, Oler gives a WEAKER bound than FT for sum-of-radii (the boundary correction helps packing, not bounding).

## What I Learned

1. **FT is the wall for n >= 5.** The hex packing density bound cannot be beaten by any area-based argument, and all LP/SOCP relaxations reduce to it when radii are small.

2. **Geometric constraints only help for small n** where individual radii are large enough (r > 0.25) to trigger pair/containment bounds.

3. **The Oler-Groemer boundary correction goes the wrong way** for upper bounding sum-of-radii. It says MORE circles fit than FT predicts, not fewer.

4. **To beat FT for n=26**, one would need: (a) higher-order Lasserre SDP hierarchy (order >= 2, with O(n^4) moment matrix), (b) branch-and-bound with LP/SDP relaxations, or (c) problem-specific valid inequalities that constrain packings of many small circles.

5. **The gap narrows as n grows** (from ~15% at n=3 to ~3.4% at n=32), suggesting FT becomes asymptotically tight. This is expected: larger n means the packing approaches the infinite hexagonal packing limit.

Seed: deterministic (no randomness in bound computation).
