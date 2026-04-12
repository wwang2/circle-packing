---
strategy: voronoi-cell-boundary-corrected-ub
status: in-progress
eval_version: eval-v1
metric: 2.7396
issue: 24
parents:
  - upperbound-001
type: experiment
---

# VORONOI-BOUNDARY-001: Voronoi-Cell Boundary-Corrected Upper Bound

## Glossary

- **UB**: Upper bound on maximum sum of radii
- **FT**: Fejes-Toth area relaxation (UB = 2.7396 for n=26)
- **n_b**: Number of circles touching the boundary (walls)
- **Voronoi cell**: Region of the square closer to a given circle center than any other

## Result (in-progress, interrupted by rate limit)

**Tightest universal UB: 2.7396 (no improvement over FT yet)**

However, a conditional bound was discovered: if the number of boundary-touching circles n_b >= 20 (which the actual optimal solution satisfies with n_b=20), the moment bound tightens to **2.7294** (gap reduced from 3.93% to 3.54%). This conditional bound needs more work to become a rigorous universal bound.

| Method | UB | Gap | Notes |
|---|---|---|---|
| Fejes-Toth (baseline) | 2.7396 | 3.93% | Hexagonal packing density |
| Moment (FT + wall) | 2.7396 | 3.93% | Wall constraint doesn't bind at n_b=0 |
| Cauchy-Schwarz | 2.8768 | 9.14% | Looser |
| Conditional n_b=20 | 2.7294 | 3.54% | Requires proving n_b >= 20 |
| Conditional n_b=21 | 2.7007 | 2.45% | Requires proving n_b >= 21 |

## Approach

Three bounding strategies implemented:
1. **voronoi_bound.py** — Full Voronoi cell analysis with shapely (needs shapely install)
2. **tight_bound.py** — Moment bound with wall perimeter constraint
3. **minkowski_bound.py** — Minkowski sum area analysis
4. **verify_lfunction.py** — L-function verification for Voronoi cells

## What's Left (interrupted)

1. Install shapely and run voronoi_bound.py for the full Voronoi cell analysis
2. Prove that any optimal solution must have n_b >= 18 (would give UB < 2.74)
3. Combine per-cell area analysis with Euler's formula constraint on edge counts
4. Try the Cauchy-Crofton integral-geometric approach (Connelly's brainstorm suggestion)

## Files

- `voronoi_bound.py` — Voronoi cell analysis (needs shapely)
- `tight_bound.py` — Moment bound with wall constraint (**working**)
- `minkowski_bound.py` — Minkowski area bounds
- `verify_lfunction.py` — L-function verification
- `figures/voronoi_analysis.png`, `bounds_comparison.png`
