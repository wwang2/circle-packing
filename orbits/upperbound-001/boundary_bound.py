"""
Tighter upper bound using boundary waste analysis.

Key insight: the Fejes Toth bound uses density pi/(2*sqrt(3)) ~ 0.9069.
But this is the MAXIMUM density achievable in hexagonal packing.
In a finite square, the boundary causes unavoidable waste.

APPROACH: Steiner formula / inner parallel body.

For a convex body K and packing of n circles of radii r_1,...,r_n:
Each circle center i is in the inner parallel body K_{-r_i}.

Define: for radius r, the "effective packing area" using Oler's inequality is:
  A_eff(r) = area(K_{-r}) + perim(K_{-r}) * r + pi * r^2
            = (1-2r)^2 + 4(1-2r)*r + pi*r^2
            = 1 - 4r + 4r^2 + 4r - 8r^2 + pi*r^2
            = 1 + (pi-4)*r^2

The maximum number of circles of radius r in K is:
  N(r) <= A_eff(r) / (2*sqrt(3)*r^2) = [1 + (pi-4)*r^2] / (2*sqrt(3)*r^2)

The sum of radii with n identical circles of radius r is:
  S(r) = n*r where n <= N(r)

Maximum S: over r, maximize n*r subject to n*r^2 * 2*sqrt(3) <= 1 + (pi-4)*r^2 and n is given.

From the constraint: r <= sqrt([1 + (pi-4)*r^2] / (2*sqrt(3)*n))

For large n: (pi-4)*r^2 is small, so r ~ 1/sqrt(2*sqrt(3)*n)
and S ~ n * 1/sqrt(2*sqrt(3)*n) = sqrt(n/(2*sqrt(3))) (Fejes Toth).

For finite n, the (pi-4)*r^2 term helps slightly but (pi-4) < 0, so it HURTS!
pi - 4 ≈ -0.858. So A_eff(r) = 1 - 0.858*r^2 < 1.

Wait, that means the Oler bound is TIGHTER than Fejes Toth!

Let me redo: with A_eff(r) = 1 + (pi-4)*r^2:
n <= [1 + (pi-4)*r^2] / (2*sqrt(3)*r^2)
   = 1/(2*sqrt(3)*r^2) + (pi-4)/(2*sqrt(3))

So: r^2 <= 1 / (2*sqrt(3) * [n - (pi-4)/(2*sqrt(3))])
       = 1 / (2*sqrt(3)*n - (pi-4))
       = 1 / (2*sqrt(3)*n - pi + 4)

And: S = n*r <= n / sqrt(2*sqrt(3)*n - pi + 4)
            = sqrt(n^2 / (2*sqrt(3)*n - pi + 4))

For n=26: S <= sqrt(676 / (2*sqrt(3)*26 - pi + 4))
           = sqrt(676 / (90.07 - 3.14 + 4))
           = sqrt(676 / 90.93)
           = sqrt(7.434)
           = 2.727

Compare Fejes Toth: sqrt(26/(2*sqrt(3))) = sqrt(7.506) = 2.740.

So the Oler bound gives 2.727, slightly tighter!

But this is for EQUAL radii. For mixed radii, the bound might be different.
Let me formulate the LP properly.
"""

import numpy as np
from scipy.optimize import minimize_scalar, linprog
import json
import sys
from pathlib import Path


def oler_bound_equal_radii(n):
    """
    Oler's inequality for n equal circles in unit square.

    N(r) <= [1 + (pi-4)*r^2] / (2*sqrt(3)*r^2)

    For n given: r^2 <= 1 / (2*sqrt(3)*n - pi + 4)
    Sum = n*r <= n / sqrt(2*sqrt(3)*n - pi + 4)
              = sqrt(n^2 / (2*sqrt(3)*n - pi + 4))
    """
    denom = 2 * np.sqrt(3) * n - np.pi + 4
    if denom <= 0:
        return np.sqrt(n / (2 * np.sqrt(3)))  # fallback
    return np.sqrt(n**2 / denom)


def oler_bound_mixed_lp(n, K=300, verbose=False):
    """
    LP for mixed radii using Oler's area accounting.

    For a mix of circles with n_k circles of radius r_k:
    - sum(n_k) = n
    - Each circle of radius r_k needs Voronoi cell area >= 2*sqrt(3)*r_k^2
    - But the total available "effective area" depends on the radii.

    The Oler inequality for mixed radii:
    sum_i [2*sqrt(3)*r_i^2] <= 1 + (pi-4)*r_max^2

    where r_max = max(r_i).

    Since pi-4 < 0, larger r_max makes the bound TIGHTER (less area available).

    For the LP, we can introduce a variable R = r_max and constrain:
    - r_k <= R for all k
    - sum(n_k * 2*sqrt(3)*r_k^2) <= 1 + (pi-4)*R^2

    This is nonlinear in R. Discretize R too.
    """
    r_vals = np.linspace(0.002, 0.5, K)
    dr = r_vals[1] - r_vals[0]

    # For each possible r_max value, solve the LP
    best_bound = float('inf')
    best_rmax = None

    rmax_values = np.linspace(0.02, 0.5, 50)

    for R in rmax_values:
        # LP: maximize sum(n_k * r_k) subject to:
        # sum(n_k) = n
        # n_k >= 0
        # r_k <= R for bins with r_k > R: n_k = 0
        # sum(n_k * 2*sqrt(3)*r_k^2) <= 1 + (pi-4)*R^2

        # Only use bins with r_k <= R
        valid = r_vals <= R + 1e-10
        r_valid = r_vals[valid]
        Kv = len(r_valid)

        if Kv == 0:
            continue

        c = -r_valid  # minimize negative sum

        # Equality: sum n_k = n
        A_eq = np.ones((1, Kv))
        b_eq = np.array([float(n)])

        # Inequality: sum(n_k * 2*sqrt(3)*r_k^2) <= 1 + (pi-4)*R^2
        A_ub = np.array([2 * np.sqrt(3) * r_valid**2])
        b_ub = np.array([1.0 + (np.pi - 4) * R**2])

        if b_ub[0] <= 0:
            continue  # infeasible

        bounds = [(0, None) for _ in range(Kv)]

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                        bounds=bounds, method='highs')

        if result.success:
            bound = -result.fun
            if bound < best_bound:
                best_bound = bound
                best_rmax = R

    if verbose and best_rmax is not None:
        print(f"  Oler LP bound for n={n}: {best_bound:.6f} (r_max={best_rmax:.4f})")

    return best_bound


def oler_bound_analytical(n):
    """
    Analytical Oler bound for mixed radii.

    With Oler: sum(2*sqrt(3)*r_i^2) <= 1 + (pi-4)*r_max^2.

    For equal radii r: n * 2*sqrt(3)*r^2 <= 1 + (pi-4)*r^2
    => r^2 * (2*sqrt(3)*n - (pi-4)) <= 1
    => r^2 <= 1 / (2*sqrt(3)*n + 4 - pi)

    Sum = n*r = n / sqrt(2*sqrt(3)*n + 4 - pi)
              = sqrt(n^2 / (2*sqrt(3)*n + 4 - pi))

    For unequal radii, by C-S:
    (sum r_i)^2 <= n * sum(r_i^2)
                <= n * [1 + (pi-4)*r_max^2] / (2*sqrt(3))

    This is maximized when r_max is as large as possible (since pi-4 < 0,
    the term (pi-4)*r_max^2 is negative, making the bound smaller).

    Wait, r_max is the maximum radius, so increasing r_max makes the right side SMALLER.
    The bound is minimized (tightest) when r_max is maximized.
    But r_max <= 0.5.

    With r_max = 0.5: sum(r_i^2) <= [1 + (pi-4)*0.25]/(2*sqrt(3))
                                   = [1 - 0.2146] / 3.464
                                   = 0.7854 / 3.464
                                   = 0.2268

    sum r_i <= sqrt(n * 0.2268) = sqrt(26 * 0.2268) = sqrt(5.897) = 2.428!

    But wait: if r_max = 0.5, we need at least one circle of radius 0.5,
    which nearly fills the square. Then the remaining 25 circles have
    very little room. So this bound is NOT achievable.

    The bound holds but is loose because we're not constraining the number
    of large circles. Let me fix this.

    BETTER: introduce a parameter R and optimize:
    sum r_i <= sqrt(n * [1 + (pi-4)*R^2] / (2*sqrt(3)))
    subject to: r_max <= R, and R <= 0.5.

    Since (pi-4) < 0, the bound DECREASES as R increases.
    But we need R >= r_max for the bound to be valid.
    The tightest bound has R as LARGE as possible.

    But we don't know r_max! Any R >= 0 gives a valid bound (since if r_max < R,
    the Oler bound with parameter R is weaker than with r_max, so it still holds).

    Wait, no. Oler's inequality is:
    sum(2*sqrt(3)*r_i^2) <= A_eff(r_max)
    where A_eff(r) = 1 + (pi-4)*r^2.

    If we SET R = r_max, this is exact. If we use R > r_max, we get a WEAKER bound
    (larger RHS since pi-4 < 0 and R > r_max means A_eff(R) < A_eff(r_max)).

    WAIT: (pi-4) < 0, so A_eff(r) = 1 + (pi-4)*r^2 is DECREASING in r.
    A_eff(r_max) < A_eff(r) for r < r_max.
    Using R >= r_max gives A_eff(R) <= A_eff(r_max), which is TIGHTER.

    Hmm, but r_max is the ACTUAL max radius, not a free parameter.
    The bound says sum(...) <= A_eff(r_max). We can't make the RHS smaller
    by using a larger R because A_eff(R) < A_eff(r_max) when R > r_max,
    which means the bound becomes INVALID (we'd be claiming the LHS is
    bounded by something smaller than it actually is).

    I think I was confused. Let me re-derive Oler carefully.

    Oler's inequality (for equal-radius circles of radius r):
    n <= A_eff(r) / (2*sqrt(3)*r^2)
    where A_eff(r) = A(K) + P(K)*r + pi*r^2 for convex K containing all circles.

    For K = [0,1]^2: A_eff(r) = 1 + 4*r + pi*r^2.

    WAIT! I had the wrong formula! Let me recalculate.
    A_eff(r) = area(K) + perimeter(K)*r + pi*r^2 = 1 + 4r + pi*r^2.

    NOT 1 + (pi-4)*r^2!

    Let me redo: the inner parallel body K_{-r} has:
    area = (1-2r)^2 = 1 - 4r + 4r^2
    perimeter = 4(1-2r) = 4 - 8r

    The OUTER parallel body of K_{-r} by radius r has area:
    A((K_{-r})_r) = area(K_{-r}) + perimeter(K_{-r})*r + pi*r^2
                  = (1-4r+4r^2) + (4-8r)*r + pi*r^2
                  = 1 - 4r + 4r^2 + 4r - 8r^2 + pi*r^2
                  = 1 + (pi-4)*r^2

    But Oler's bound is for circles of radius r with centers in K_{-r}:
    n * (2*sqrt(3)*r^2) <= A((K_{-r})_r) = 1 + (pi-4)*r^2

    Hmm wait, but A((K_{-r})_r) should be LARGER than A(K_{-r}) since
    we're expanding. Since pi-4 ≈ -0.858, we get 1 - 0.858*r^2 < 1.

    That means A((K_{-r})_r) < area(K) = 1. But expanding a set should give
    a larger area... unless the expansion "fills in the corners" by less than
    the original area. For a square, the inner body is a smaller square, and
    expanding it gives a rounded rectangle, which has area < 1 (the corners
    are rounded). So yes, A((K_{-r})_r) < 1 makes sense!

    OK, but there's ANOTHER version of Oler's inequality that uses K directly:
    n <= [A(K) + P(K)*r/2] / (2*sqrt(3)*r^2)
       = [1 + 2r] / (2*sqrt(3)*r^2)

    This is the "sausage bound" version. Let me compute both.
    """
    # Version 1: Using inner parallel body
    def bound_v1(r):
        A_eff = 1 + (np.pi - 4) * r**2
        if A_eff <= 0:
            return float('inf')
        max_n = A_eff / (2 * np.sqrt(3) * r**2)
        if max_n < n:
            return float('inf')  # can't pack n circles of this radius
        return n * r

    # Version 2: Using K directly (Oler)
    def bound_v2(r):
        A_eff = 1 + 2 * r  # [1 + P*r/2] = 1 + 4*r/2 = 1 + 2r
        max_n = A_eff / (2 * np.sqrt(3) * r**2)
        if max_n < n:
            return float('inf')
        return n * r

    # Version 3: Using full Steiner formula
    def bound_v3(r):
        A_eff = 1 + 4*r + np.pi*r**2  # Steiner formula for K expanded by r
        max_n = A_eff / (2 * np.sqrt(3) * r**2)
        if max_n < n:
            return float('inf')
        return n * r

    # Find optimal r for each version
    results = {}
    for name, func in [('inner_parallel', bound_v1),
                        ('oler_sausage', bound_v2),
                        ('steiner', bound_v3)]:
        best = 0
        for r in np.linspace(0.001, 0.5, 10000):
            val = func(r)
            if val < float('inf') and val > best:
                best = val
        results[name] = best

    return results


def compute_all(n_values=None, verbose=True):
    """Compute all bounds for comparison."""
    if n_values is None:
        n_values = [1, 2, 3, 4, 5, 10, 15, 20, 26, 30, 32]

    known_best = {
        1: 0.5000, 2: 0.5858, 3: 0.7645, 4: 1.0000, 5: 1.0854,
        10: 1.5911, 15: 2.0365, 20: 2.3010, 26: 2.6360, 30: 2.8425, 32: 2.9390,
    }

    print(f"{'n':>3} | {'FT':>7} | {'Oler-eq':>7} | {'InnerP':>7} | {'Sausage':>7} | "
          f"{'Steiner':>7} | {'OlerLP':>7} | {'Best':>7} | {'Known':>7} | {'Gap%':>6}")
    print("-" * 95)

    all_results = {}
    for n_val in n_values:
        ft = np.sqrt(n_val / (2 * np.sqrt(3)))
        oe = oler_bound_equal_radii(n_val)
        analytical = oler_bound_analytical(n_val)
        oler_lp = oler_bound_mixed_lp(n_val, verbose=False)

        bounds = {
            'fejes_toth': ft,
            'oler_equal': oe,
            **{f'oler_{k}': v for k, v in analytical.items()},
        }
        if oler_lp is not None:
            bounds['oler_lp'] = oler_lp

        best = min(bounds.values())
        known = known_best.get(n_val)
        gap_pct = 100 * (best - known) / known if known else None

        print(f"{n_val:3d} | {ft:7.4f} | {oe:7.4f} | "
              f"{analytical['inner_parallel']:7.4f} | {analytical['oler_sausage']:7.4f} | "
              f"{analytical['steiner']:7.4f} | "
              f"{oler_lp if oler_lp else 0:7.4f} | "
              f"{best:7.4f} | {known if known else 0:7.4f} | "
              f"{gap_pct if gap_pct else 0:5.1f}%")

        all_results[n_val] = bounds

    return all_results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        n_values = [int(x) for x in sys.argv[1:]]
    else:
        n_values = [1, 2, 3, 4, 5, 10, 15, 20, 26, 30, 32]

    results = compute_all(n_values)

    # Save
    output_path = Path(__file__).parent / "boundary_bounds.json"
    serializable = {str(n): {k: float(v) for k, v in b.items()} for n, b in results.items()}
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved to {output_path}")
