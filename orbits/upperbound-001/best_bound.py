"""
Best upper bound computation.

Consolidating all approaches and focusing on what works:

1. Area bound: sqrt(n/pi) -- basic, from sum(pi*r^2) <= 1
2. Fejes Toth: sqrt(n/(2*sqrt(3))) -- from hexagonal packing density
3. Oler (equal radii): tighter using boundary correction
4. New: optimized LP combining all valid constraints

Key formulas:
- sum(r_i^2) <= 1/pi (area)
- sum(r_i^2) <= 1/(2*sqrt(3)) (Fejes Toth density per Voronoi cell)
- For equal r: n*2*sqrt(3)*r^2 <= 1 + (pi-4)*r^2 (Oler with inner parallel body)

Oler for equal radii:
  r^2 <= 1 / [n*2*sqrt(3) - (pi-4)]
       = 1 / [2*sqrt(3)*n + 4 - pi]

  S = n*r = sqrt(n^2 / (2*sqrt(3)*n + 4 - pi))

For n=26: S = sqrt(676 / (90.07 + 0.858)) = sqrt(676/90.93) = sqrt(7.434) = 2.727

The question is: can we do better for MIXED radii?

For mixed radii, Oler's inequality says:
  sum(2*sqrt(3)*r_i^2) <= 1 + (pi-4)*r_max^2

But r_max is unknown. If we treat it as a free parameter R >= r_i for all i:
  For any R >= max(r_i): sum(2*sqrt(3)*r_i^2) <= 1 + (pi-4)*R^2

Since pi-4 < 0, the bound gets TIGHTER (more restrictive) for larger R.
The weakest (loosest) bound is at R = 0: sum(...) <= 1, which is Fejes Toth.
The tightest is at R = 0.5: sum(...) <= 1 + (pi-4)*0.25 ≈ 0.785.

But if R = 0.5, all radii must be <= 0.5, which is already true.

Actually, we need R >= max(r_i). If we CHOOSE R = max(r_i), the bound is
  sum(2*sqrt(3)*r_i^2) <= 1 + (pi-4)*max(r_i)^2

The LP should:
- Choose radii r_1 >= r_2 >= ... >= r_n >= 0
- Constraint: sum(2*sqrt(3)*r_i^2) <= 1 + (pi-4)*r_1^2  (since r_1 is max)
- i.e., 2*sqrt(3)*r_1^2 + sum_{i>1}(2*sqrt(3)*r_i^2) <= 1 + (pi-4)*r_1^2
- i.e., r_1^2*(2*sqrt(3) - pi + 4) + sum_{i>1}(2*sqrt(3)*r_i^2) <= 1
- Maximize sum(r_i)

This is a QCQP (quadratic constraint, linear objective).
We can solve it with SOCP.
"""

import numpy as np
from scipy.optimize import minimize, linprog
import json
import sys
from pathlib import Path


def fejes_toth_bound(n):
    """FT bound: sqrt(n/(2*sqrt(3)))."""
    return np.sqrt(n / (2 * np.sqrt(3)))


def oler_equal_bound(n):
    """Oler bound assuming equal radii: sqrt(n^2 / (2*sqrt(3)*n + 4 - pi))."""
    denom = 2 * np.sqrt(3) * n + 4 - np.pi
    return np.sqrt(n**2 / denom)


def oler_mixed_socp(n, verbose=False):
    """
    Oler bound for mixed radii via SOCP (Second Order Cone Program).

    Maximize sum(r_i)
    subject to:
      r_1 >= r_2 >= ... >= r_n >= 0
      r_i <= 0.5
      (2*sqrt(3) + 4 - pi) * r_1^2 + sum_{i>1} 2*sqrt(3)*r_i^2 <= 1

    This is a convex program (quadratic constraint with positive coefficients,
    linear objective).

    Using CVXPY.
    """
    try:
        import cvxpy as cp
    except ImportError:
        return None

    r = cp.Variable(n, name='r')
    s3 = 2 * np.sqrt(3)

    constraints = []
    constraints += [r >= 1e-10]
    constraints += [r <= 0.5]

    # Ordering
    for i in range(n-1):
        constraints += [r[i] >= r[i+1]]

    # Oler constraint: (2*sqrt(3) + 4 - pi)*r[0]^2 + sum_{i>1} 2*sqrt(3)*r[i]^2 <= 1
    alpha = s3 + 4 - np.pi  # coefficient for r_max^2
    # = 2*sqrt(3) + 4 - pi ≈ 3.464 + 4 - 3.142 = 4.322

    if n == 1:
        constraints += [alpha * cp.square(r[0]) <= 1]
    else:
        constraints += [alpha * cp.square(r[0]) + s3 * cp.sum_squares(r[1:]) <= 1]

    objective = cp.Maximize(cp.sum(r))
    prob = cp.Problem(objective, constraints)

    result = prob.solve(solver=cp.SCS, verbose=False, max_iters=20000)

    if verbose and r.value is not None:
        rv = r.value
        print(f"  Oler SOCP for n={n}: {result:.6f}")
        print(f"  r_max = {rv[0]:.6f}, r_min = {rv[-1]:.6f}")
        print(f"  Top 5 radii: {rv[:5]}")

    return result


def oler_mixed_analytical(n):
    """
    Analytical solution to the Oler mixed-radii problem.

    Maximize sum(r_i) subject to:
      alpha * r_1^2 + beta * sum_{i>1} r_i^2 <= 1
      r_1 >= r_2 >= ... >= r_n >= 0

    where alpha = 2*sqrt(3) + 4 - pi, beta = 2*sqrt(3).

    By Lagrange multipliers (ignoring ordering):
    d/dr_1 [sum r_i - lambda*(alpha*r_1^2 + beta*sum_{i>1} r_i^2)] = 0
    1 = 2*lambda*alpha*r_1 => r_1 = 1/(2*lambda*alpha)
    1 = 2*lambda*beta*r_j for j>1 => r_j = 1/(2*lambda*beta)

    So r_1/r_j = beta/alpha = 2*sqrt(3) / (2*sqrt(3) + 4 - pi)
    For n=26: beta/alpha = 3.464 / 4.322 = 0.8015
    So r_1 < r_j? That contradicts r_1 >= r_j.

    Wait: alpha > beta since alpha = beta + 4 - pi and 4 - pi > 0.
    So the Lagrange optimal has r_1 < r_j, meaning the ordering constraint
    is BINDING. The optimal is actually all radii equal!

    When all radii equal: n * alpha * r^2 <= 1 ... no wait.
    When r_1 = r_2 = ... = r_n = r:
    alpha * r^2 + beta * (n-1) * r^2 = [alpha + (n-1)*beta] * r^2 <= 1

    Sum = n*r, r <= 1/sqrt(alpha + (n-1)*beta)
    Sum = n / sqrt(alpha + (n-1)*beta)
        = n / sqrt(2*sqrt(3) + 4 - pi + (n-1)*2*sqrt(3))
        = n / sqrt(2*sqrt(3)*n + 4 - pi)

    This is EXACTLY the equal-radii Oler bound! So allowing mixed radii
    doesn't help -- the Oler bound is the same.

    Hmm, but the Lagrange analysis said r_1 should be SMALLER. That means
    with the ordering constraint, the optimum is at r_1 = r_2 = ... = r_n.
    The Lagrange solution without ordering gives a LOWER r_1 and higher r_j,
    which would give a HIGHER sum. But the ordering constraint forces r_1 >= r_j,
    and at the optimum r_1 = r_j.

    Actually wait: the unconstrained Lagrange gives r_1 < r_j.
    If we DROP the ordering constraint, the maximum sum is achieved with
    one circle smaller and (n-1) circles larger. Let's compute this!

    Without ordering:
    r_1 = 1/(2*lambda*alpha), r_j = 1/(2*lambda*beta) for j=2..n
    Constraint: alpha*r_1^2 + beta*sum(r_j^2) = alpha/(4*lambda^2*alpha^2) + beta*(n-1)/(4*lambda^2*beta^2)
    = 1/(4*lambda^2*alpha) + (n-1)/(4*lambda^2*beta) = 1

    4*lambda^2 = 1/alpha + (n-1)/beta
    => lambda^2 = [1/alpha + (n-1)/beta] / 4

    Sum = 1/(2*lambda*alpha) + (n-1)/(2*lambda*beta)
        = (1/(2*lambda)) * [1/alpha + (n-1)/beta]

    lambda = sqrt([1/alpha + (n-1)/beta] / 4) = sqrt([1/alpha + (n-1)/beta]) / 2

    Sum = [1/alpha + (n-1)/beta] / sqrt([1/alpha + (n-1)/beta])
        = sqrt(1/alpha + (n-1)/beta)

    Compare with equal radii:
    Sum = n / sqrt(alpha + (n-1)*beta)

    Ratio: sqrt(1/alpha + (n-1)/beta) / (n / sqrt(alpha + (n-1)*beta))
         = sqrt(1/alpha + (n-1)/beta) * sqrt(alpha + (n-1)*beta) / n
         = sqrt((1/alpha + (n-1)/beta) * (alpha + (n-1)*beta)) / n

    Let a = alpha, b = beta, m = n-1.
    Product = (1/a + m/b) * (a + m*b) = 1 + m*b/a + m*a/b + m^2
            = (1 + m)^2 + m*(b/a + a/b - 2)
            = n^2 + m*(b/a + a/b - 2)

    Since a != b, b/a + a/b > 2 (AM-GM), so product > n^2.
    Therefore: sqrt(product)/n > 1.

    The UNCONSTRAINED optimum is HIGHER than equal radii!
    The Oler bound with mixed radii is WEAKER (gives higher upper bound).

    So the ordering constraint actually helps tighten the bound.
    The tightest Oler bound is with equal radii: sqrt(n^2/(2*sqrt(3)*n + 4 - pi)).
    """
    s3 = 2 * np.sqrt(3)
    alpha = s3 + 4 - np.pi
    beta = s3

    # Equal radii bound (tightest with ordering)
    equal_bound = np.sqrt(n**2 / (alpha + (n-1)*beta))

    # Unconstrained bound (one smaller, rest larger)
    unconstrained = np.sqrt(1.0/alpha + (n-1)/beta)

    return equal_bound, unconstrained


def area_plus_oler_bound(n, verbose=False):
    """
    Combine area bound AND Oler bound.

    The area bound gives: sum(pi*r_i^2) <= 1, i.e., sum(r_i^2) <= 1/pi
    The Oler bound gives: sum(2*sqrt(3)*r_i^2) <= 1+(pi-4)*r_max^2

    For equal radii, Oler is always tighter:
    Oler: r^2 <= 1/(2*sqrt(3)*n + 4-pi) ≈ 1/(3.464n + 0.858)
    Area: r^2 <= 1/(pi*n) ≈ 1/(3.142n)

    Since 3.464 > 3.142, Oler is tighter for all n. Good.

    Can we combine them? Yes, but Oler already dominates area.

    Additional constraints:
    - r_i <= 0.5 for all i
    - x_i + r_i <= 1 and x_i >= r_i => center placement
    - For equal radii: fit in grid pattern

    Let me try adding the constraint that at most K circles can have
    r > r_threshold, based on packing arguments.

    For circles of radius r > 0.25: center in [r, 1-r]^2, side < 0.5.
    Max circles: floor(0.5/(2r)) in each direction... complicated.

    Let me try the NUMERIC SOCP to see if unequal radii beat equal.
    """
    return oler_mixed_socp(n, verbose=verbose)


def compute_best_bounds(n_values=None, verbose=True):
    """Compute the best bounds we have."""
    if n_values is None:
        n_values = [1, 2, 3, 4, 5, 10, 15, 20, 26, 30, 32]

    known_best = {
        1: 0.5000, 2: 0.5858, 3: 0.7645, 4: 1.0000, 5: 1.0854,
        10: 1.5911, 15: 2.0365, 20: 2.3010, 26: 2.6360, 30: 2.8425, 32: 2.9390,
    }

    print(f"{'n':>3} | {'Area':>8} | {'FT':>8} | {'Oler-eq':>8} | {'SOCP':>8} | "
          f"{'Best UB':>8} | {'Known':>8} | {'Gap':>8} | {'Gap%':>6}")
    print("-" * 85)

    all_results = {}
    for n_val in n_values:
        area = np.sqrt(n_val / np.pi)
        ft = fejes_toth_bound(n_val)
        oe = oler_equal_bound(n_val)
        socp = oler_mixed_socp(n_val, verbose=(verbose and n_val in [26]))

        bounds = {'area': area, 'fejes_toth': ft, 'oler_equal': oe}
        if socp is not None:
            bounds['oler_socp'] = socp

        best = min(bounds.values())
        known = known_best.get(n_val)
        gap = best - known if known else None
        gap_pct = 100 * gap / known if gap is not None else None

        print(f"{n_val:3d} | {area:8.4f} | {ft:8.4f} | {oe:8.4f} | "
              f"{socp if socp else 0:8.4f} | {best:8.4f} | "
              f"{known if known else 0:8.4f} | "
              f"{gap if gap else 0:8.4f} | {gap_pct if gap_pct else 0:5.1f}%")

        all_results[n_val] = bounds

    return all_results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        n_values = [int(x) for x in sys.argv[1:]]
    else:
        n_values = [1, 2, 3, 4, 5, 10, 15, 20, 26, 30, 32]

    results = compute_best_bounds(n_values, verbose=True)

    # Also show the analytical mixed vs equal comparison
    print("\n\nAnalytical comparison: equal vs mixed radii (Oler)")
    print(f"{'n':>3} | {'Equal':>8} | {'Mixed':>8} | {'Ratio':>6}")
    print("-" * 35)
    for n_val in n_values:
        eq, mx = oler_mixed_analytical(n_val)
        print(f"{n_val:3d} | {eq:8.4f} | {mx:8.4f} | {mx/eq:6.4f}")

    # Save results
    output_path = Path(__file__).parent / "best_bounds.json"
    serializable = {str(n): {k: float(v) for k, v in b.items()} for n, b in results.items()}
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved to {output_path}")
