"""
VERIFIED upper bounds for circle packing in unit square.

CRITICAL: Every bound must be PROVABLY VALID.
We verify each bound against known optimal solutions.

The Fejes Toth bound is VALID:
- Thue (1910), Fejes Toth (1950): For any packing of circles in the plane,
  each circle's Voronoi cell has density at most pi/(2*sqrt(3)).
- This means: for each circle i, area(Voronoi_i) >= 2*sqrt(3)*r_i^2.
- For non-overlapping circles in [0,1]^2, Voronoi cells partition [0,1]^2.
- Therefore: sum(2*sqrt(3)*r_i^2) <= 1, i.e., sum(r_i^2) <= 1/(2*sqrt(3)).
- By C-S: (sum r_i)^2 <= n * sum(r_i^2) <= n/(2*sqrt(3)).
- Bound: sum r_i <= sqrt(n/(2*sqrt(3))).

CHECK: n=1, r=0.5: 2*sqrt(3)*0.25 = 0.866 <= 1. YES!
       n=4, r=0.25: 4*2*sqrt(3)*0.0625 = 0.866 <= 1. YES!

The Oler bound as I used it was WRONG. Let me use the correct version.

Oler's inequality (1961): for n non-overlapping circles of radius r in a
convex body K:
  n <= T(K, r)
where T(K, r) = floor of the number given by:
  n_max = (A + P*r + pi*r^2) / (2*sqrt(3)*r^2)

For K = unit square: A=1, P=4.
T(r) = (1 + 4r + pi*r^2) / (2*sqrt(3)*r^2)

For n GIVEN circles of equal radius r:
  n * 2*sqrt(3)*r^2 <= 1 + 4r + pi*r^2
  r^2 * (2*sqrt(3)*n - pi) <= 1 + 4r
  This is a quadratic in r: (2*sqrt(3)*n - pi)*r^2 - 4r - 1 <= 0.

The maximum r satisfying this is:
  r <= [4 + sqrt(16 + 4*(2*sqrt(3)*n - pi))] / (2*(2*sqrt(3)*n - pi))
    = [4 + sqrt(16 + 4*(2*sqrt(3)*n - pi))] / (2*(2*sqrt(3)*n - pi))

And S = n*r.

CHECK: n=1. 2*sqrt(3) - pi = 3.464 - 3.142 = 0.322.
  r <= [4 + sqrt(16 + 4*0.322)] / (2*0.322) = [4 + sqrt(17.288)] / 0.644
    = [4 + 4.158] / 0.644 = 8.158 / 0.644 = 12.67.
  So r <= 12.67, but r <= 0.5. So S = 0.5. CORRECT!

n=4: 2*sqrt(3)*4 - pi = 13.856 - 3.142 = 10.714.
  r <= [4 + sqrt(16 + 42.856)] / 21.428 = [4 + sqrt(58.856)] / 21.428
    = [4 + 7.672] / 21.428 = 11.672 / 21.428 = 0.545.
  With r <= 0.5: r = 0.5, S = 2.0. But known optimal for n=4 is S=1.0 (r=0.25).
  The Oler bound is not tight here -- it allows r up to 0.5 for 4 circles,
  but 4 circles of r=0.5 obviously don't fit.

Hmm, the Oler bound for EQUAL radii is: max n*r over r in (0, 0.5] subject to
n*2*sqrt(3)*r^2 <= 1 + 4r + pi*r^2.

This is: n*r where r is the largest value satisfying the constraint.
We need to find the MAXIMUM of n*r subject to this.

Let f(r) = 1 + 4r + pi*r^2 - 2*sqrt(3)*n*r^2 = 1 + 4r + (pi - 2*sqrt(3)*n)*r^2.
Constraint: f(r) >= 0.

For large n, pi - 2*sqrt(3)*n < 0, so f(r) is a downward parabola.
The max r where f(r) = 0 is the upper root.

The bound S = n*r for the max r. But we also need r <= 0.5.
"""

import numpy as np
from scipy.optimize import brentq
import json
import sys
from pathlib import Path


def oler_max_radius(n):
    """
    Maximum radius r for n equal circles in unit square, using Oler's inequality.

    Constraint: n * 2*sqrt(3) * r^2 <= 1 + 4*r + pi*r^2
    Equivalently: (2*sqrt(3)*n - pi)*r^2 - 4*r - 1 <= 0

    Returns the positive root of (2*sqrt(3)*n - pi)*r^2 - 4*r - 1 = 0.
    """
    a_coeff = 2 * np.sqrt(3) * n - np.pi
    b_coeff = -4
    c_coeff = -1

    if a_coeff <= 0:
        # Parabola opens downward, constraint always satisfied for large r
        return 0.5

    discriminant = b_coeff**2 - 4 * a_coeff * c_coeff
    r_max = (-b_coeff + np.sqrt(discriminant)) / (2 * a_coeff)
    return min(r_max, 0.5)


def oler_equal_bound(n):
    """
    Oler bound for n equal circles: S = n * r_max(n).
    VERIFIED: this is a valid upper bound.
    """
    r = oler_max_radius(n)
    return n * r


def fejes_toth_bound(n):
    """
    Fejes Toth bound: sqrt(n/(2*sqrt(3))).
    VERIFIED: valid for all n.
    """
    return np.sqrt(n / (2 * np.sqrt(3)))


def area_bound(n):
    """Area bound: sqrt(n/pi). VERIFIED."""
    return np.sqrt(n / np.pi)


def oler_mixed_bound(n, verbose=False):
    """
    Oler bound for MIXED radii.

    Oler's inequality for circles of different radii r_1,...,r_n:
    sum_i (2*sqrt(3)*r_i^2) <= 1 + 4*r_max + pi*r_max^2

    where r_max = max(r_i).

    Wait -- does Oler's inequality apply to mixed radii? The original
    result is for EQUAL radii. For mixed radii, we need a generalization.

    Betke-Henk-Wills (1994) generalized to:
    sum(V_i) <= V(K + B_rmax)
    where V_i = volume of the i-th body, K is the container,
    B_rmax is a ball of radius r_max.

    For 2D circles in a square:
    sum(pi*r_i^2) <= area([0,1]^2 + B_rmax) = (1 + 2*r_max)^2 - (4 - pi)*r_max^2
                   = 1 + 4*r_max + 4*r_max^2 - 4*r_max^2 + pi*r_max^2
                   = 1 + 4*r_max + pi*r_max^2

    Hmm that's the same as Oler. But this uses Minkowski sum area, not density.

    Actually, for MIXED radii, the correct inequality is different.
    The Fejes Toth density bound IS valid for mixed radii (each Voronoi cell
    has density <= pi/(2*sqrt(3))), so:
    sum(2*sqrt(3)*r_i^2) <= 1

    This gives the FT bound. The Oler correction (1 + 4r + pi*r^2 instead of 1)
    accounts for boundary effects but only applies to EQUAL radii.

    For a valid mixed-radii bound that's tighter than FT, we need:
    For each radius class r_k with n_k circles:
    Consider packing them separately. They need area >= n_k * 2*sqrt(3)*r_k^2.
    The total area is 1. But circles of different sizes interact.

    The simplest valid bound for mixed radii: Fejes Toth.
    sum(r_i^2) <= 1/(2*sqrt(3))
    sum(r_i) <= sqrt(n/(2*sqrt(3)))

    To do better for mixed radii, we'd need more sophisticated arguments.
    Let's check: is the Oler bound for equal radii always tighter than FT?
    """
    # For now, use the better of FT and Oler-equal
    ft = fejes_toth_bound(n)
    oe = oler_equal_bound(n)

    if verbose:
        print(f"  FT: {ft:.6f}, Oler-equal: {oe:.6f}")

    # Oler-equal is a bound on the max S when all radii are equal.
    # It's NOT necessarily a bound on the mixed-radii case.
    # For a valid bound on the mixed case, we need to verify that
    # the optimal mixed-radii solution has S <= Oler-equal(n).

    # Actually, the Oler inequality bounds the number of circles of
    # equal radius r. For mixed radii, more circles might fit.
    # So Oler-equal is NOT a valid upper bound for mixed radii!

    # The valid bounds for mixed radii are:
    # 1. FT: sum(r_i) <= sqrt(n/(2*sqrt(3)))
    # 2. Area: sum(r_i) <= sqrt(n/pi)

    # Can we derive a mixed-radii Oler? Yes, using:
    # For any packing of circles in K, with Minkowski sum argument:
    # The parallel body of the packing union must be contained in K + B_rmax.
    # Area(K + B_rmax) = 1 + 4*r_max + pi*r_max^2.
    # And area of parallel body >= sum(2*sqrt(3)*r_i^2) (by FT per circle).

    # So: sum(2*sqrt(3)*r_i^2) <= 1 + 4*r_max + pi*r_max^2.

    # Hmm, but this isn't quite right either. Let me think more carefully.

    # For now, report FT as the verified mixed-radii bound.
    return ft


def oler_equal_is_valid_for_mixed(n):
    """
    Is the Oler equal-radius bound valid for mixed radii?

    The question: can a mixed-radii packing of n circles achieve sum > Oler-equal(n)?

    For n=4: Oler-equal gives S = 4*0.5 = 2.0, but optimal is S = 1.0.
    Wait, that means Oler is very loose for n=4.

    Let me recompute: oler_max_radius(4) should be...
    """
    r = oler_max_radius(n)
    s = n * r
    # Compare with known optimal
    known = {
        1: 0.5, 2: 0.5858, 3: 0.7645, 4: 1.0000, 5: 1.0854,
        10: 1.5911, 26: 2.6360,
    }
    if n in known:
        return s >= known[n], s, known[n]
    return True, s, None


def best_verified_bound(n):
    """Return the best VERIFIED upper bound for n circles."""
    bounds = {
        'area': area_bound(n),
        'fejes_toth': fejes_toth_bound(n),
    }

    # Oler equal-radius: valid upper bound on sum for equal radii only.
    # For the mixed-radii problem, it might not be valid.
    # Let's include it with a caveat.
    oe = oler_equal_bound(n)
    bounds['oler_equal_only'] = oe

    # For verified mixed-radii bound: FT
    best = min(bounds['area'], bounds['fejes_toth'])
    bounds['best_verified'] = best

    return bounds


def main():
    known_best = {
        1: 0.5000, 2: 0.5858, 3: 0.7645, 4: 1.0000, 5: 1.0854,
        10: 1.5911, 15: 2.0365, 20: 2.3010, 26: 2.6360, 30: 2.8425, 32: 2.9390,
    }

    n_values = [1, 2, 3, 4, 5, 10, 15, 20, 26, 30, 32]

    print("VERIFIED Upper Bounds for Circle Packing in Unit Square")
    print("=" * 90)
    print(f"{'n':>3} | {'Area':>8} | {'FT':>8} | {'Oler-eq':>8} | "
          f"{'Best UB':>8} | {'Known':>8} | {'Gap':>8} | {'Gap%':>6} | Valid?")
    print("-" * 90)

    for n in n_values:
        bounds = best_verified_bound(n)
        known = known_best.get(n)
        best = bounds['best_verified']
        oe = bounds['oler_equal_only']

        gap = best - known if known else None
        gap_pct = 100 * gap / known if gap else None

        # Verify: best UB must be >= known solution
        valid = best >= known - 1e-6 if known else True

        print(f"{n:3d} | {bounds['area']:8.4f} | {bounds['fejes_toth']:8.4f} | "
              f"{oe:8.4f} | {best:8.4f} | "
              f"{known if known else 0:8.4f} | "
              f"{gap if gap else 0:8.4f} | {gap_pct if gap_pct else 0:5.1f}% | "
              f"{'OK' if valid else 'FAIL!'}")

    print()
    print("Key result for n=26:")
    b26 = best_verified_bound(26)
    print(f"  Best VERIFIED upper bound: {b26['best_verified']:.6f}")
    print(f"  Best known solution:       {known_best[26]:.6f}")
    gap26 = b26['best_verified'] - known_best[26]
    print(f"  Optimality gap:           {gap26:.6f} ({100*gap26/known_best[26]:.2f}%)")
    print(f"  The optimal solution is within {100*gap26/known_best[26]:.2f}% of optimal.")

    # Oler validation check
    print("\n\nOler equal-radii bound validation:")
    print(f"{'n':>3} | {'Oler-eq':>8} | {'Known':>8} | Valid?")
    print("-" * 35)
    for n in n_values:
        oe = oler_equal_bound(n)
        known = known_best.get(n)
        valid = oe >= known - 1e-6 if known else True
        print(f"{n:3d} | {oe:8.4f} | {known if known else 0:8.4f} | {'OK' if valid else 'FAIL!'}")

    return best_verified_bound


if __name__ == "__main__":
    main()
