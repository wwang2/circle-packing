"""
Analytic upper bounds for circle packing, tighter than Fejes Toth.

The Fejes Toth bound sqrt(n/(2*sqrt(3))) accounts for hexagonal packing density
but ignores boundary waste. Circles near the edges of the unit square have
larger Voronoi cells.

APPROACH 1: Groemer's inequality
For n non-overlapping circles of radii r_1,...,r_n in a convex body K:
  sum(r_i) <= (A(K) + P(K)*r_max + pi*r_max^2) / (2*sqrt(3)*r_max)

where A(K) = area, P(K) = perimeter, r_max = max radius.
For K = [0,1]^2: A = 1, P = 4.

This gives: sum(r_i) <= (1 + 4*r_max + pi*r_max^2) / (2*sqrt(3)*r_max)
         = 1/(2*sqrt(3)*r_max) + 4/(2*sqrt(3)) + pi*r_max/(2*sqrt(3))
         = 1/(2*sqrt(3)*r_max) + 2/sqrt(3) + pi*r_max/(2*sqrt(3))

This is minimized over r_max. Taking derivative:
d/dr = -1/(2*sqrt(3)*r^2) + pi/(2*sqrt(3)) = 0
=> r = 1/sqrt(pi)

At r = 1/sqrt(pi): bound = sqrt(pi)/(2*sqrt(3)) + 2/sqrt(3) + 1/(2*sqrt(3))
                         = (sqrt(pi) + 4 + 1)/(2*sqrt(3))
                         ... wait, let me recompute.

Actually, Groemer's result applies to packings where we know n but not r_max.
If we also know n, we can combine with the area bound.

APPROACH 2: Oler's inequality
For n non-overlapping unit disks in a convex body K:
  n <= (A + P*r + pi*r^2) / (2*sqrt(3)*r^2)

This bounds n given the area. Rearranging for our problem (variable radii):
not directly applicable.

APPROACH 3: Combined bound using the "sausage" integral.
For a convex body K, the area of the "inner parallel body" K_(-r) = {x: B(x,r) subset K}
is A(K_(-r)) = (1-2r)^2 for K = [0,1]^2 (when r <= 0.5).

The perimeter of K_(-r) is P(K_(-r)) = 4*(1-2r).

For circles of a fixed radius r:
- Each center must be in K_(-r), area = (1-2r)^2
- By the Fejes Toth bound applied to K_(-r):
  n_r * 2*sqrt(3)*r^2 <= (1-2r)^2 + P(K_(-r))*r + pi*r^2
  where the RHS is the area of the "outer parallel body" of K_(-r) by radius r.
  Wait, that's the Oler bound applied to K_(-r).

Actually, let me use a SIMPLER but effective approach.

APPROACH 4: LP over radius classes with improved packing density.
"""

import numpy as np
from scipy.optimize import minimize, linprog
import json
import sys
from pathlib import Path


def groemer_bound(n):
    """
    Groemer-type bound for circle packing in unit square.

    For n circles of varying radii in a convex body K with area A and perimeter P:

    The key inequality (following Betke, Henk, Wills 1994):
    sum(r_i) <= [A + P*R/(2) ] / (sqrt(3)*R)

    where R is a free parameter (related to maximum radius).

    Actually, let me use a more direct approach.

    For the unit square, with n circles of radii r_1,...,r_n:
    Each circle i has center in the "inner body" K_{-r_i} = [r_i, 1-r_i]^2.
    Area of K_{-r_i} = (1-2r_i)^2.

    The Voronoi cell of circle i (clipped to [0,1]^2) has area >= 2*sqrt(3)*r_i^2.
    But for boundary circles, the Voronoi cell extends to the boundary,
    and the "wasted" boundary area makes the effective cell larger.

    Specifically: the Voronoi cell of a circle of radius r touching one wall
    includes the strip between the wall and the circle, adding area ~ 2r * r = 2r^2.
    The Voronoi cell becomes at least 2*sqrt(3)*r^2 + boundary_waste.

    This is hard to quantify without knowing the configuration.
    """
    # Fall back to Fejes Toth for now
    return np.sqrt(n / (2 * np.sqrt(3)))


def optimal_fixed_radius_count(r, use_oler=True):
    """
    Maximum number of non-overlapping circles of radius r in [0,1]^2.

    Each center must be in [r, 1-r]^2, side length s = 1-2r.
    Centers must be pairwise at distance >= 2r.

    By the Oler-type bound:
    n(r) <= (s + 2r)^2 / (2*sqrt(3)*r^2) + (s + 2r) / (2r)
         = 1/(2*sqrt(3)*r^2) + 1/r

    Better: use the exact formula from hexagonal packing.
    In a rectangle s x s with circle radius r:
    - Rows: height per row = 2r (first row) + sqrt(3)*r (subsequent)
      n_rows = 1 + floor((s - 2r) / (sqrt(3)*r))
    - Circles per row: n_per_row = floor(s / (2r))
    - Alternate rows shifted, may fit one fewer.

    But we want an UPPER BOUND on n(r), not a lower bound.

    Upper bound: A(K_{-r}) / (min Voronoi area) = (1-2r)^2 / (2*sqrt(3)*r^2).
    But this uses the Fejes Toth density for the interior.

    More careful: including boundary effects from Oler's inequality.
    n(r) <= [(1-2r)^2 + 4*(1-2r)*r + pi*r^2] / (2*sqrt(3)*r^2)
          = [1 - 4r + 4r^2 + 4r - 8r^2 + pi*r^2] / (2*sqrt(3)*r^2)
          = [1 + (pi-4)*r^2] / (2*sqrt(3)*r^2)
          = 1/(2*sqrt(3)*r^2) + (pi-4)/(2*sqrt(3))
    """
    if use_oler:
        return 1.0 / (2*np.sqrt(3)*r**2) + (np.pi - 4) / (2*np.sqrt(3))
    else:
        s = max(1 - 2*r, 0)
        return s**2 / (2*np.sqrt(3)*r**2)


def mixed_radius_bound(n, verbose=False):
    """
    LP bound allowing mixed radii.

    Discretize possible radii: r_1 > r_2 > ... > r_K.
    Let n_k = number of circles of radius r_k.

    Constraints:
    - sum(n_k) = n
    - n_k >= 0
    - Packing feasibility: the circles must fit in [0,1]^2.

    The key constraint comes from "area accounting":
    sum(n_k * A_k) <= 1
    where A_k is the minimum Voronoi cell area for radius r_k.

    By Fejes Toth: A_k = 2*sqrt(3)*r_k^2.

    But we can do better: for each radius class, use the Oler bound.

    Actually, the constraint is simply sum(n_k * 2*sqrt(3)*r_k^2) <= 1
    (Fejes Toth), plus:
    - For the LARGEST radius class: n_1 <= optimal_fixed_radius_count(r_1)
    - Total area: sum(n_k * pi * r_k^2) <= 1

    Objective: maximize sum(n_k * r_k).

    This is a simple LP in n_k.
    """
    K = 200  # number of radius bins
    r_vals = np.linspace(0.001, 0.5, K)

    # Objective: maximize sum(n_k * r_k)
    c = -r_vals  # negative because linprog minimizes

    # Constraint 1: sum(n_k) = n
    A_eq = np.ones((1, K))
    b_eq = np.array([n])

    # Inequality constraints (A_ub @ x <= b_ub)
    A_ub_list = []
    b_ub_list = []

    # Constraint 2: Fejes Toth area: sum(n_k * 2*sqrt(3)*r_k^2) <= 1
    A_ub_list.append(2 * np.sqrt(3) * r_vals**2)
    b_ub_list.append(1.0)

    # Constraint 3: Standard area: sum(n_k * pi * r_k^2) <= 1
    A_ub_list.append(np.pi * r_vals**2)
    b_ub_list.append(1.0)

    # Constraint 4: For each radius r_k, the number of circles of this
    # radius or larger must be feasible.
    # sum_{j: r_j >= r_k} n_j <= optimal_count(r_k)
    # This is because if all those circles had radius r_k (smaller),
    # they'd STILL need to fit.
    # Wait, this isn't quite right. Circles of different sizes interact differently.
    # But it's a valid relaxation: replacing larger circles with smaller ones
    # makes it easier to pack (more room), so the count bound still applies.
    # Actually NO: we're bounding from above, so we need: if we have n_large
    # circles of radius >= r_k, they need at least n_large * 2sqrt(3)*r_k^2 area.
    # Already captured by constraint 2.

    # Constraint 5: Oler-type bound on total count at each radius
    # For each radius r_k: max number of circles of radius r_k is N_max(r_k).
    for k in range(K):
        row = np.zeros(K)
        row[k] = 1.0
        A_ub_list.append(row)
        b_ub_list.append(optimal_fixed_radius_count(r_vals[k]))

    A_ub = np.array(A_ub_list)
    b_ub = np.array(b_ub_list)

    # Bounds: n_k >= 0 (linprog default)
    bounds = [(0, None) for _ in range(K)]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method='highs')

    if result.success:
        bound = -result.fun
        if verbose:
            nk = result.x
            nonzero = nk > 0.01
            print(f"  Mixed-radius LP bound for n={n}: {bound:.6f}")
            print(f"  Active radius bins: {np.sum(nonzero)}")
            for k in np.where(nonzero)[0]:
                print(f"    r={r_vals[k]:.4f}: n={nk[k]:.2f}")
        return bound
    else:
        if verbose:
            print(f"  LP failed: {result.message}")
        return None


def combined_bound(n, verbose=False):
    """
    Combine multiple approaches for the tightest bound.
    """
    bounds = {}

    # Area bound
    bounds['area'] = np.sqrt(n / np.pi)

    # Fejes Toth
    bounds['fejes_toth'] = np.sqrt(n / (2 * np.sqrt(3)))

    # Mixed radius LP
    mr = mixed_radius_bound(n, verbose=verbose)
    if mr is not None:
        bounds['mixed_radius_lp'] = mr

    best = min(bounds.values())
    return best, bounds


if __name__ == "__main__":
    known_best = {
        1: 0.5000, 2: 0.5858, 3: 0.7645, 4: 1.0000, 5: 1.0854,
        10: 1.5911, 15: 2.0365, 20: 2.3010, 26: 2.6360, 30: 2.8425, 32: 2.9390,
    }

    if len(sys.argv) > 1:
        n_values = [int(x) for x in sys.argv[1:]]
    else:
        n_values = [1, 2, 3, 4, 5, 10, 15, 20, 26, 30, 32]

    print(f"{'n':>3} | {'Area':>8} | {'FT':>8} | {'Mixed LP':>8} | {'Best':>8} | {'Known':>8} | {'Gap':>8} | {'Gap%':>6}")
    print("-" * 80)

    all_results = {}
    for n in n_values:
        best, bounds = combined_bound(n, verbose=(n in [4, 10, 26]))

        known = known_best.get(n, None)
        gap = best - known if known else None
        gap_pct = 100 * gap / known if gap is not None else None

        print(f"{n:3d} | {bounds['area']:8.4f} | {bounds['fejes_toth']:8.4f} | "
              f"{bounds.get('mixed_radius_lp', 0):8.4f} | {best:8.4f} | "
              f"{known if known else 0:8.4f} | "
              f"{gap if gap else 0:8.4f} | {gap_pct if gap_pct else 0:5.1f}%")

        all_results[n] = bounds

    # Save
    output_path = Path(__file__).parent / "analytic_bounds.json"
    serializable = {str(n): {k: float(v) for k, v in b.items()} for n, b in all_results.items()}
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
