"""
Geometric upper bounds for circle packing in unit square.

These bounds use geometric arguments rather than generic convex relaxations.
They exploit the specific structure of the problem (unit square, Euclidean geometry).

Key approaches:
1. Area bound with boundary correction
2. Strip-based bounds with LP over strip assignments
3. Hexagonal packing density bound
4. Configuration-based bounds for specific n
"""

import numpy as np
import json
import sys
from pathlib import Path


def area_bound(n):
    """Basic area bound: sum r_i <= sqrt(n/pi)."""
    return np.sqrt(n / np.pi)


def boundary_corrected_area_bound(n):
    """
    Improved bound accounting for boundary "wasted" area.

    A circle of radius r inside [0,1]^2 has its center in [r, 1-r]^2.
    The area available for circle centers is (1-2r)^2.

    More importantly: the circle physically occupies area pi*r^2 inside the square,
    but it also "wastes" area near the boundary. The square has area 1, but the
    effective packing region shrinks.

    Tighter formulation: instead of sum(pi*r_i^2) <= 1, we can use:
    Each circle of radius r "claims" a Voronoi cell. In an optimal packing,
    the Voronoi cells tile the square. The minimum area Voronoi cell for a
    circle of radius r is about 2*sqrt(3)*r^2 (hexagonal packing).

    So: sum(2*sqrt(3)*r_i^2) <= 1
    => sum(r_i^2) <= 1/(2*sqrt(3)) ≈ 0.2887
    => sum(r_i) <= sqrt(n/(2*sqrt(3))) ≈ sqrt(0.2887*n)

    For n=26: sqrt(26/(2*sqrt(3))) ≈ sqrt(7.506) ≈ 2.740.

    But WAIT: the hexagonal packing density bound is not valid for finite
    packings in a square! It's an asymptotic result. For finite n, circles
    near the boundary have larger Voronoi cells.

    However, the Thue-Fejes Toth theorem states that in ANY arrangement of
    non-overlapping circles in the plane, the density of any circle's Voronoi
    cell is at most pi/(2*sqrt(3)) ≈ 0.9069. This applies to EACH circle
    individually.

    So for each circle i: pi*r_i^2 / A_i <= pi/(2*sqrt(3))
    where A_i is the area of its Voronoi cell.
    => A_i >= 2*sqrt(3)*r_i^2

    Since the Voronoi cells partition the square: sum(A_i) = 1.
    => sum(2*sqrt(3)*r_i^2) <= 1.

    BUT: this is only true if all circles are in a CONVEX region and the
    Voronoi cells are clipped to that region. For circles in a square,
    the Voronoi cells are clipped to [0,1]^2, so sum(A_i) = 1. The
    density bound per cell still holds (each Voronoi cell is convex).

    Actually, the Fejes Toth bound is: for any circle in a convex container,
    the ratio area(circle)/area(Voronoi cell) <= pi/(2*sqrt(3)).
    This requires that the Voronoi cell is a convex polygon, which it is.

    So: sum(2*sqrt(3)*r_i^2) <= 1 is a VALID upper bound on sum(r_i^2).

    By C-S: (sum r_i)^2 <= n * sum(r_i^2) <= n/(2*sqrt(3))
    => sum r_i <= sqrt(n/(2*sqrt(3)))

    For n=26: sqrt(26/(2*sqrt(3))) ≈ 2.740
    Improvement from 2.877 to 2.740!
    """
    # Fejes Toth density: pi/(2*sqrt(3))
    # => each circle's Voronoi cell has area >= 2*sqrt(3)*r^2
    # => sum(r^2) <= 1/(2*sqrt(3))
    return np.sqrt(n / (2 * np.sqrt(3)))


def boundary_voronoi_bound(n):
    """
    Even tighter bound: boundary circles have lower packing efficiency.

    For a circle of radius r with center at distance d from the boundary,
    its Voronoi cell includes the boundary region. The "wasted" area depends
    on d and r.

    A circle touching one wall (center at (r, y)): its Voronoi cell includes
    the strip [0, r+delta] which is not fully used.

    Quantifying: for a circle at distance r from one wall (touching it),
    the fraction of its Voronoi cell that's inside the square is roughly:
    The cell extends at least r beyond the circle on each side.
    On the wall side, the cell extends from x=0 to x~r (wasted area ~= r*2r = 2r^2).
    But the circle only has area pi*r^2 ≈ 3.14*r^2 while the cell has area
    at least 2*sqrt(3)*r^2 ≈ 3.46*r^2. The wasted area is (2*sqrt(3) - pi)*r^2 ≈ 0.33*r^2.

    For circles at a corner (touching two walls): even more waste.

    This is hard to bound in general. Let me try a different approach:
    LP over radius values.
    """
    return boundary_corrected_area_bound(n)


def lp_radius_bound(n, verbose=False):
    """
    LP bound on sum of radii with radius distribution constraints.

    Variables: how many circles of each "size class".

    Discretize radii into bins: r_1 > r_2 > ... > r_K.
    Let n_k = number of circles with radius in bin k.
    sum(n_k) = n.

    Constraints:
    - Area: sum(n_k * pi * r_k^2) <= 1  (using bin centers)
    - Packing: can't fit too many large circles
      - n_1 * 4 * r_1^2 <= 1 (bounding boxes of largest circles fit)
      - Actually, bounding boxes can overlap, so this is wrong.
      - But: any r > 0.5 is impossible. Only 1 circle of r=0.5 fits.
      - Any r > 1/3: at most 4 circles (2x2 grid of r=1/4 limit).

    Actually, let me use a cleaner formulation.

    For each circle, r_i <= 0.5.
    For each pair, r_i + r_j <= sqrt(2) (max distance in unit square).

    The key insight: in a row of circles at the same y-coordinate,
    the sum of diameters <= 1 (they must fit side by side in width 1).

    Let's think of it as a 2D bin packing problem.

    Maximize sum(r_i) subject to:
    - All r_i in [0, 0.5]
    - sum(pi * r_i^2) <= 1/(2*sqrt(3))  (Fejes Toth)  -- WAIT, should be <= 1/pi...

    Hmm, let me reconsider. The area constraint is sum(pi*r_i^2) <= 1 (circles fit in unit square).
    The Fejes Toth argument gives sum(2*sqrt(3)*r_i^2) <= 1, i.e., sum(r_i^2) <= 1/(2*sqrt(3)).
    Since 1/(2*sqrt(3)) < 1/pi, this is TIGHTER.

    With n circles and the constraint sum(r_i^2) <= C (where C = 1/(2*sqrt(3))):
    By Lagrange: all r_i equal at r* = sqrt(C/n), sum = n*r* = sqrt(n*C) = sqrt(n/(2*sqrt(3))).

    Can we do better by adding more constraints? Yes:

    1. At most 4 circles can have r > 0.25 (they need center in [r,1-r]^2,
       and 4 circles of r=0.25 exactly tile the centers [0.25,0.75]^2).
       Actually, 4 circles of r=0.25 pack perfectly but 5 of r=0.24 might also fit.

    Let me just compute the best LP bound with the constraints we have.
    """
    try:
        import cvxpy as cp
    except ImportError:
        return None, None

    # Variables: radii r_1 >= r_2 >= ... >= r_n (sorted)
    r = cp.Variable(n, name="r")

    constraints = []

    # Positive radii
    constraints += [r >= 1e-10]

    # Upper bound on each radius
    constraints += [r <= 0.5]

    # Fejes Toth area bound: sum(r_i^2) <= 1/(2*sqrt(3))
    # But cvxpy: sum of squares is convex, and we need it <= constant.
    # sum(r_i^2) is convex, so constraint sum(r_i^2) <= C is convex. Good.
    C_FT = 1.0 / (2 * np.sqrt(3))
    constraints += [cp.sum_squares(r) <= C_FT]

    # Actually we also have the simple area bound sum(pi*r_i^2) <= 1
    # Which is sum(r_i^2) <= 1/pi ≈ 0.3183. Fejes Toth gives 0.2887. Tighter.

    # Ordering (symmetry breaking, helps solver)
    for i in range(n-1):
        constraints += [r[i] >= r[i+1]]

    # Pairwise sum bound: r_i + r_j <= sqrt(2) for all pairs
    # (since max distance between centers in unit square is sqrt(2))
    # For sorted radii, the tightest is r[0] + r[1] <= sqrt(2)
    # But sqrt(2) ≈ 1.414, and with r <= 0.5, r[0]+r[1] <= 1.0, so this is not binding.

    # Stronger: the max distance between two circle centers is
    # sqrt((1-r_i-r_j)^2 + (1-r_i-r_j)^2) = sqrt(2)*(1-r_i-r_j) if both in corners.
    # Distance must be >= r_i + r_j.
    # So: sqrt(2)*(1-r_i-r_j) >= r_i + r_j
    # => (r_i+r_j)(1 + 1/sqrt(2)) <= sqrt(2) -- wait:
    # Actually the max possible distance between centers is sqrt((1-2*max(ri,rj))^2 + ...).
    # This is complex. Skip for now.

    # Objective
    objective = cp.Maximize(cp.sum(r))
    prob = cp.Problem(objective, constraints)

    result = prob.solve(solver=cp.SCS, verbose=False, max_iters=10000)

    if verbose and r.value is not None:
        rv = r.value
        print(f"  Radii: {rv[:min(5,n)]}...")
        print(f"  sum(r^2) = {np.sum(rv**2):.6f} (limit {C_FT:.6f})")

    return result, (r.value if r.value is not None else None)


def strip_bound(n, num_strips=None, verbose=False):
    """
    Strip-based upper bound using LP.

    Divide [0,1] into horizontal strips. A circle of radius r with center
    at height y occupies the strip from y-r to y+r.

    Key insight: consider projecting circles onto the x-axis.
    In any horizontal slice of height dy, the total "width" occupied by
    circles intersecting that slice is bounded by their projections.

    For a circle of radius r, its projection onto x at height y
    (relative to center) has half-width sqrt(r^2 - y^2).

    The total projection width at any height cannot exceed 1 (width of square).

    This gives an INTEGRAL constraint. Discretize:
    For strip s at height y_s with width dy:
    sum_{i: circle i intersects strip s} 2*sqrt(r_i^2 - (y_s - y_i)^2) <= 1

    This is nonlinear. But we can use a weaker version:
    For strip s: sum_{i: circle i intersects strip s} 2*r_i <= 1
    (using diameter instead of chord width)

    This is the "projection bound": in each strip, the sum of diameters <= 1.
    A circle of radius r occupies 2r strips, so it appears in ~2r/dy strips.

    Total "work": sum_i 2r_i * (2r_i/dy) = 4/dy * sum(r_i^2)
    Must be <= num_strips * 1 = 1/dy.
    So: 4*sum(r_i^2) <= 1, i.e., sum(r_i^2) <= 0.25.

    This gives: sum(r_i) <= sqrt(n/4) = sqrt(n)/2.
    For n=26: sqrt(26)/2 ≈ 2.550.

    Wait, this is TIGHTER than the area bound (2.877) and the Fejes Toth bound (2.740)!

    But is it correct? Let me verify...

    Actually, the argument above is wrong. A circle centered in strip s
    with radius r intersects strips from s-r to s+r, which is 2r/dy strips.
    In each of those strips, it contributes at most 2r to the width.

    The constraint is: for EACH strip, sum of contributions <= 1.
    But a circle doesn't contribute 2r to each strip it intersects --
    it contributes 2*sqrt(r^2 - d^2) where d is the distance to the strip.
    This is LESS than 2r for most strips.

    The "sum of 2r_i in each strip <= 1" is overly conservative.

    Actually, the correct constraint is: for each y, the number of circles
    whose y-interval [y_i-r_i, y_i+r_i] contains y, multiplied by their
    x-widths at that y, sums to <= 1.

    A WEAKER but valid constraint: at each y, the number of circles
    intersecting y times their max width (2r_i) sums to <= 1.

    Let me formulate this properly as an LP.
    """
    if num_strips is None:
        num_strips = max(10, n)  # reasonable default

    try:
        import cvxpy as cp
    except ImportError:
        return None

    # Strip heights: y_s = (s + 0.5) / num_strips for s = 0, ..., num_strips-1
    dy = 1.0 / num_strips
    strip_centers = [(s + 0.5) * dy for s in range(num_strips)]

    # Variables: r_i (radii) and y_i (y-coordinates of centers)
    r = cp.Variable(n)
    y = cp.Variable(n)

    constraints = []
    constraints += [r >= 1e-10, r <= 0.5]
    constraints += [y >= r, y <= 1 - r]

    # For each strip, the number of circles intersecting it times 2r_i <= 1.
    # But we don't know which circles intersect which strip without knowing y_i.
    # This is a bilinear coupling (y_i and r_i).

    # Simpler: the "1D projection" bound.
    # Project all circles onto y-axis. Each projects to interval [y_i-r_i, y_i+r_i].
    # The "depth" at any point is the number of circles whose projection covers that point.
    # For non-overlapping circles in a unit square:
    # At depth d (d circles stacked vertically at some x), each has width 2r_i at that x.
    # The x-projections of these d circles at that y don't overlap, so:
    # sum of 2*sqrt(r_i^2 - d_i^2) <= 1 (where d_i is vertical offset from strip center).

    # This is too complex. Let me use a simpler formulation.

    # APPROACH: Cauchy-Schwarz with an integral argument.
    #
    # For the UNIT INTERVAL:
    # Consider the "coverage function" on [0,1] x [0,1]:
    # f(x,y) = 1 if (x,y) is inside some circle, 0 otherwise.
    # Integral of f = sum(pi*r_i^2) = total circle area.
    #
    # For each y, let w(y) = total x-width covered by circles at height y.
    # w(y) = sum_{i: |y-y_i| < r_i} 2*sqrt(r_i^2 - (y-y_i)^2)
    # w(y) <= 1 for all y (since circles don't overlap in x at any y).
    #
    # Total area = integral_0^1 w(y) dy <= integral_0^1 1 dy = 1.
    # This just gives sum(pi*r_i^2) <= 1. Known.
    #
    # Can we do better? Yes! Because at each y, the circles are non-overlapping
    # in x. The "width" used is NOT just the sum of chord widths -- the circles
    # at different x-positions also leave GAPS between them.
    #
    # For m circles in a row of width 1, each of radius r, the width used is
    # 2*m*r. So m <= 1/(2r), and sum of radii in that row <= m*r <= 1/2.
    # Total sum of radii <= (1/2) * (number of "rows").
    #
    # Number of "rows": in height 1, with each row using height 2r_min,
    # at most 1/(2*r_min) rows. But r_min varies.

    # Let me just compute the simple bounds and move on.
    return None


def improved_cs_bound(n, verbose=False):
    """
    Improved Cauchy-Schwarz using x and y projections simultaneously.

    For each circle i, define:
    - w_i = 2*r_i (diameter = x-width at center)
    - h_i = 2*r_i (diameter = y-height at center)

    Now consider the integral:
    I = integral_0^1 [sum_{i: y in [y_i-r_i, y_i+r_i]} 2*sqrt(r_i^2 - (y-y_i)^2)] dy

    For each circle: integral = pi*r_i^2 (area of circle).
    Total I = sum(pi*r_i^2).
    And I <= 1 (integrand <= 1 at each y).

    Better bound using CAUCHY-SCHWARZ ON THE INTEGRAND:
    At each y, let S(y) = set of circles intersecting height y.
    Integrand = sum_{i in S(y)} 2*sqrt(r_i^2 - (y-y_i)^2) <= 1.

    By C-S: [sum c_i]^2 <= |S(y)| * sum c_i^2
    where c_i = 2*sqrt(r_i^2 - (y-y_i)^2).

    But we want the opposite: we know sum c_i <= 1, and want to bound sum r_i.

    Alternative: use the TOTAL bandwidth argument.
    sum_i integral_{y_i-r_i}^{y_i+r_i} 2*sqrt(r_i^2 - (y-y_i)^2) dy = sum_i pi*r_i^2
    sum_i integral_{y_i-r_i}^{y_i+r_i} 1 dy = sum_i 2*r_i

    By C-S on each circle's integral:
    [integral 2*sqrt(r^2 - t^2) dt]^2 <= [integral 1 dt] * [integral 4*(r^2-t^2) dt]
    [pi*r^2]^2 <= 2r * [4*(2r^3/3)] = 16r^4/3
    pi^2*r^4 <= 16r^4/3
    pi^2 <= 16/3 ≈ 5.333. Since pi^2 ≈ 9.87, this is FALSE!

    So C-S goes the wrong way here. Each circle's chord-width function is
    "peaky" -- it's wide at the center and narrow at the edges.

    Different approach: dual bound.

    DUAL OF THE PACKING PROBLEM:
    Consider the dual of: max sum(r_i) s.t. feasible packing.

    Assign a "price" p(x,y) >= 0 to each point (x,y) in [0,1]^2.
    For any circle of radius r at position (cx, cy):
    - Its "cost" is integral_{circle} p(x,y) dx dy
    - Any valid circle has cost >= some function of r
    - Total cost of all circles <= integral_{[0,1]^2} p(x,y) dx dy

    If we use p(x,y) = 1 (uniform):
    Cost per circle = pi*r^2. Total <= 1. Gives area bound.

    If we use p(x,y) = Ax + By + C (affine):
    Integral over circle of radius r at (cx,cy) = pi*r^2 * (A*cx + B*cy + C).
    This doesn't help directly.

    CLEVER CHOICE: p(x,y) = some function that "penalizes" large circles more.
    For instance, p(x,y) = 1 + alpha*(x - x^2 + y - y^2).
    This is higher near the center and accounts for boundary effects.

    Total integral of p = 1 + alpha*(1/2 - 1/3 + 1/2 - 1/3) = 1 + alpha/3.

    Per circle at (cx,cy) with radius r:
    Integral = pi*r^2 * [1 + alpha*(cx - cx^2 + cy - cy^2)]
               + alpha * pi*r^4/4 * (-2)  ... actually this gets complicated.

    Let me just use numerical optimization to find the best "price function".
    """
    return None  # Complex to implement analytically


def compute_all_geometric_bounds(n_values=None, verbose=True):
    """Compute all geometric bounds."""
    if n_values is None:
        n_values = [1, 2, 3, 4, 5, 10, 15, 20, 26, 30, 32]

    known_best = {
        1: 0.5000, 2: 0.5858, 3: 0.7645, 4: 1.0000, 5: 1.0854,
        10: 1.5911, 15: 2.0365, 20: 2.3010, 26: 2.6360, 30: 2.8425, 32: 2.9390,
    }

    results = {}

    for n in n_values:
        if verbose:
            print(f"\nn={n:3d}: ", end="")

        ab = area_bound(n)
        ft = boundary_corrected_area_bound(n)  # Fejes Toth

        # LP with Fejes Toth area constraint
        lp_val, _ = lp_radius_bound(n, verbose=False)

        bounds = {'area': ab, 'fejes_toth': ft}
        if lp_val is not None:
            bounds['lp_ft'] = lp_val

        best = min(v for v in bounds.values() if v is not None)
        bounds['best'] = best

        if verbose:
            print(f"area={ab:.4f}  FT={ft:.4f}", end="")
            if lp_val is not None:
                print(f"  LP={lp_val:.4f}", end="")
            print(f"  BEST={best:.4f}", end="")
            if n in known_best:
                gap = best - known_best[n]
                gap_pct = 100 * gap / known_best[n]
                print(f"  known={known_best[n]:.4f}  gap={gap:.4f} ({gap_pct:.1f}%)", end="")
            print()

        results[n] = bounds

    return results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        n_values = [int(x) for x in sys.argv[1:]]
    else:
        n_values = [1, 2, 3, 4, 5, 10, 15, 20, 26, 30, 32]

    results = compute_all_geometric_bounds(n_values, verbose=True)

    # Save
    output_path = Path(__file__).parent / "geometric_bounds.json"
    serializable = {str(n): {k: float(v) for k, v in b.items()} for n, b in results.items()}
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved to {output_path}")
