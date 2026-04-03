"""
Boundary-aware upper bound for circle packing in a unit square.

Key insight: the Fejes Toth bound sum(2*sqrt(3)*r_i^2) <= 1 is tight for
infinite plane packing. In a finite square, circles near the boundary
are less efficiently packed.

APPROACH 1: Oler-Groemer boundary correction.
For n circles of equal radius r in a convex body K:
  n * pi * r^2 <= Area(K) + perimeter(K) * r + pi * r^2
  (Groemer's inequality, 1960)

For our unit square (area=1, perimeter=4):
  n * pi * r^2 <= 1 + 4*r + pi*r^2
  So: (n-1) * pi * r^2 <= 1 + 4*r
  r^2 <= (1 + 4*r) / ((n-1)*pi)

For MIXED radii (our generalization):
  sum_i pi * r_i^2 <= 1 + 4*r_max + pi*r_max^2  (generous version)
  But this only helps for the LARGEST circle.

Actually, Oler's theorem (1961) for convex body K states:
  n * A_hex(r) <= Area(K_r)
where K_r = K dilated by r (Minkowski sum with disk of radius r),
  = {x : dist(x, K) <= r}
and A_hex(r) = 2*sqrt(3)*r^2 is the hexagonal Voronoi cell area.

For unit square: Area(K_r) = (1+2r)^2 - (4-pi)*r^2 = 1 + 4r + pi*r^2.
So: n * 2*sqrt(3)*r^2 <= 1 + 4r + pi*r^2.
For equal radii.

For MIXED radii: use sum_i 2*sqrt(3)*r_i^2 <= 1 + 4*r_max + pi*r_max^2.
Not obvious this is valid. But it's commonly used.

APPROACH 2: Wasted area at boundary.
Consider a circle of radius r with center at (x,y) inside [r, 1-r]^2.
Its Voronoi cell in the plane packing is at least 2*sqrt(3)*r^2.
But within the unit square, part of this Voronoi cell may be OUTSIDE the square.

For a circle at distance d from the nearest boundary (d >= r, by containment):
The fraction of its Voronoi cell INSIDE the square depends on d.
For d >> r (center far from boundary): Voronoi cell is fully inside.
For d ≈ r (center near boundary): up to ~half the Voronoi cell is outside.

The "wasted" area per circle depends on its proximity to the boundary.
Total wasted area = sum_i waste(d_i, r_i).

APPROACH 3: Column/row LP with boundary effects.
Place circles in rows. In each row of height h, circles must pack in a strip.
The strip packing problem is 1D-ish and can be bounded more tightly.

APPROACH 4: Numerical optimization of Oler-style bound.
For a vector of radii r_1,...,r_n, find the TIGHTEST bound by optimizing
the "expanded area" function.

For mixed radii, define:
  S = sum(r_i)
  Q = sum(r_i^2)
  R = max(r_i) = r_1

The Oler bound for equal radii r gives: n*2*sqrt(3)*r^2 <= 1 + 4r + pi*r^2
=> 2*sqrt(3)*Q <= 1 + 4*R + pi*R^2  (replacing n*r^2 with Q, r with R)

But this is WRONG for mixed radii. The correct version:
sum_i 2*sqrt(3)*r_i^2 <= Area(K_{r_max})
This holds because each circle fits in the expanded square, and the
hexagonal density bound applies to the expanded region.

Actually, the Oler bound says:
For n non-overlapping disks of radius r in convex body K:
  n <= Area(K_r) / A_hex(r)

For mixed radii, this doesn't directly apply. We need:
For each circle i of radius r_i, it needs at least A_hex(r_i) = 2*sqrt(3)*r_i^2
of Voronoi area. Total Voronoi area = Area(K). But the Voronoi cells extend
to the boundary, so total Voronoi area = Area(K) = 1, giving sum(2*sqrt(3)*r_i^2) <= 1 (FT).

With the Oler correction: the Voronoi cells can extend OUTSIDE K by distance r_max.
So total Voronoi area <= Area(K_{r_max}) = 1 + 4*r_max + pi*r_max^2.
This gives: sum(2*sqrt(3)*r_i^2) <= 1 + 4*r_max + pi*r_max^2.
But this is WEAKER than FT (since RHS > 1). So Oler doesn't help for upper bounds!

Wait: Oler helps for LOWER bounding the area needed, which UPPER bounds
the number of circles of a given radius. For our sum-of-radii problem,
we're MAXIMIZING sum(r_i), and the FT bound gives sum(2*sqrt(3)*r_i^2) <= 1.
The Oler correction makes RHS LARGER, so it's a WEAKER constraint.

So FT (without Oler) is actually the TIGHTER bound for sum-of-radii optimization!

APPROACH 5: Density improvement for BOUNDARY circles.
The hexagonal packing density pi/(2*sqrt(3)) ≈ 0.9069 is for interior circles.
Circles touching the boundary have LOWER effective density (more wasted space).

For a circle of radius r touching the left boundary (center at (r, y)):
Its Voronoi cell extends to x < 0, but the square boundary truncates it.
The ACTUAL Voronoi area inside the square = (area of Voronoi cell inside [0,1]^2).

For hexagonal packing: each Voronoi cell is a regular hexagon of area 2*sqrt(3)*r^2.
For a boundary circle: the truncated Voronoi cell has area < 2*sqrt(3)*r^2 inside
the square, UNLESS the cell extends outside and we count the boundary region.

Hmm, this is the OPPOSITE of what helps. A boundary circle's Voronoi cell
inside the square is SMALLER, so sum of inside-Voronoi-areas < Area(square) = 1.
This means: sum(inside_voronoi_i) <= 1, and inside_voronoi_i < 2*sqrt(3)*r_i^2
for boundary circles. So the constraint is WEAKER, not stronger!

WAIT. Let me reconsider. The correct statement is:
- Each circle needs at least 2*sqrt(3)*r_i^2 of Voronoi area (from hex packing bound)
- The total available Voronoi area in the square is 1
- So sum(2*sqrt(3)*r_i^2) <= 1

For boundary circles, they actually need MORE Voronoi area inside the square,
because their hexagonal cell is cut by the boundary, and the remaining area
inside the square must still accommodate the circle. No wait...

The CORRECT framing: in any packing of disks in a convex body K, the
Delaunay triangulation gives triangles, and each triangle has area >= sqrt(3)/2 * (r_i+r_j)^2
... this is getting complicated.

Let me take a completely different approach.

APPROACH 6: NUMERICAL VERIFICATION-BASED BOUND.
Use a GRID-BASED APPROACH:
1. Discretize the possible circle centers on a fine grid.
2. For each pair of grid points at distance d, circles centered there can't both
   have radii summing to > d.
3. Formulate as an LP: maximize sum(r_i) over all grid points, subject to constraints.
4. The LP value is an UPPER BOUND on the packing (since we relaxed the discrete placement).

This is essentially a "clique cover" bound on the conflict graph.

For a K x K grid: K^2 possible centers, K^2 radius variables.
Constraints: for each pair (p,q) of grid points at distance d(p,q):
  r_p + r_q <= d(p,q) (if both centers are used)
  r_p <= dist(p, boundary)

But we don't know which n grid points are used. We need:
- Variables: r_p for each grid point p (r_p >= 0 means circle at p, r_p = 0 means no circle)
- "Usage" variables: u_p in {0,1} (relaxed to [0,1])
- Constraint: sum(u_p) <= n
- Constraint: r_p <= 0.5 * u_p
- Constraint: r_p + r_q <= d(p,q) for all pairs... NO, this is only valid when BOTH are used.

Actually for the LP relaxation:
- Just use r_p >= 0, r_p <= min(0.5, dist(p, boundary))
- For each pair: r_p + r_q <= d(p,q)
- Area: sum(2*sqrt(3)*r_p^2) <= 1 (not LP)
- "At most n circles": sum(indicator(r_p > 0)) <= n. In LP: use u_p >= r_p / 0.5, sum(u_p) <= n.

The cardinality constraint makes it hard. But note: in the LP without cardinality,
the optimal solution will use as many grid points as possible with tiny radii.
We NEED the cardinality constraint or the FT area constraint.

With the FT area constraint (which is convex/SOCP), we can solve:
max sum(r_p) s.t. r_p >= 0, r_p + r_q <= d(p,q) for all pairs, 2*sqrt(3)*sum(r_p^2) <= 1.

This is an SOCP. For a K x K grid with K=10, we have 100 variables and ~5000 pair constraints.
For K=20: 400 variables, ~80000 constraints. Feasible!

The key question: does the grid discretization weaken the bound?
YES: the optimal continuous solution may place centers between grid points.
But as K increases, the grid bound should converge to the true bound.

Let me implement this.
"""

import numpy as np
import cvxpy as cp
import json
import sys
from pathlib import Path
from itertools import combinations


def grid_socp_bound(n, K=10, verbose=False):
    """
    Grid-based SOCP upper bound.

    Place a K x K grid of potential center positions.
    For each grid point p, variable r_p = radius of circle at p.
    Constraints:
    - r_p >= 0
    - r_p <= dist(p, boundary of [0,1]^2)
    - For each pair (p,q): r_p + r_q <= dist(p,q)
    - sum(2*sqrt(3)*r_p^2) <= 1  (FT area bound)
    - At most n circles: need to encode this

    For the cardinality constraint, we use:
    - Each r_p <= M * u_p where u_p in [0,1]
    - sum(u_p) <= n
    - M = 0.5

    This is an LP with SOCP area constraint.
    """
    # Grid points
    # Use (i+0.5)/K for i in 0..K-1 to center points in cells
    coords = [(i+0.5)/K for i in range(K)]
    points = [(x, y) for x in coords for y in coords]
    N = len(points)  # K^2

    if verbose:
        print(f"  Grid SOCP: K={K}, {N} grid points, n={n}")

    # Distance to boundary for each point
    def dist_to_boundary(x, y):
        return min(x, 1-x, y, 1-y)

    boundary_dist = [dist_to_boundary(x, y) for (x, y) in points]

    # Pairwise distances
    # Only compute for pairs where dist < 1 (all pairs in unit square satisfy this)
    # but we only need the constraint when dist < r_p + r_q which is at most 1.

    r = cp.Variable(N, nonneg=True)
    u = cp.Variable(N, nonneg=True)  # usage indicators

    constraints = []

    # Radius bounds
    for p in range(N):
        constraints += [r[p] <= boundary_dist[p]]
        constraints += [r[p] <= 0.5 * u[p]]
        constraints += [u[p] <= 1]

    # Cardinality: at most n circles
    constraints += [cp.sum(u) <= n]

    # Pairwise non-overlap
    for p in range(N):
        for q in range(p+1, N):
            d = np.sqrt((points[p][0]-points[q][0])**2 + (points[p][1]-points[q][1])**2)
            if d < 1.0:  # Only add constraint if distance < max possible sum of radii
                constraints += [r[p] + r[q] <= d]

    # FT area bound
    constraints += [2*np.sqrt(3) * cp.sum_squares(r) <= 1]

    objective = cp.Maximize(cp.sum(r))
    prob = cp.Problem(objective, constraints)

    try:
        result = prob.solve(solver=cp.SCS, verbose=False, max_iters=50000, eps=1e-7)
        if verbose:
            print(f"  Status: {prob.status}, bound: {result:.6f}")
            if r.value is not None:
                rv = sorted(r.value, reverse=True)
                n_active = sum(1 for x in rv if x > 1e-4)
                print(f"  Active circles: {n_active}")
                print(f"  Top radii: {[f'{x:.4f}' for x in rv[:5]]}")
        return result
    except Exception as e:
        if verbose:
            print(f"  Error: {e}")
        return None


def grid_lp_bound(n, K=15, verbose=False):
    """
    Grid-based LP upper bound (without SOCP area constraint, using LP relaxation).

    Replace the SOCP area constraint with the linear Cauchy-Schwarz relaxation:
    sum(r_p) <= sqrt(n / (2*sqrt(3)))  [from CS: sum(r) <= sqrt(n * sum(r^2)) <= sqrt(n/(2*sqrt(3)))]

    Actually, let's keep FT as SOCP and let CVXPY handle it.
    Or use a different solver.

    Alternative: use scipy.optimize.linprog for pure LP.
    Drop the area constraint entirely and rely only on pairwise + cardinality.
    """
    coords = [(i+0.5)/K for i in range(K)]
    points = [(x, y) for x in coords for y in coords]
    N = len(points)

    def dist_to_boundary(x, y):
        return min(x, 1-x, y, 1-y)

    boundary_dist = [dist_to_boundary(x, y) for (x, y) in points]

    if verbose:
        print(f"  Grid LP: K={K}, {N} grid points, n={n}")

    # LP: maximize sum(r_p)
    # Variables: r_0, ..., r_{N-1}, u_0, ..., u_{N-1}
    # Total: 2*N variables

    c = np.zeros(2*N)
    c[:N] = -1  # maximize sum(r_p) = minimize -sum(r_p)

    A_ub_rows = []
    b_ub_rows = []

    # r_p <= boundary_dist[p]
    for p in range(N):
        row = np.zeros(2*N)
        row[p] = 1
        A_ub_rows.append(row)
        b_ub_rows.append(boundary_dist[p])

    # r_p <= 0.5 * u_p => r_p - 0.5*u_p <= 0
    for p in range(N):
        row = np.zeros(2*N)
        row[p] = 1
        row[N+p] = -0.5
        A_ub_rows.append(row)
        b_ub_rows.append(0)

    # u_p <= 1
    for p in range(N):
        row = np.zeros(2*N)
        row[N+p] = 1
        A_ub_rows.append(row)
        b_ub_rows.append(1)

    # sum(u_p) <= n
    row = np.zeros(2*N)
    row[N:] = 1
    A_ub_rows.append(row)
    b_ub_rows.append(n)

    # Pairwise: r_p + r_q <= d(p,q)
    n_pairs = 0
    for p in range(N):
        for q in range(p+1, N):
            d = np.sqrt((points[p][0]-points[q][0])**2 + (points[p][1]-points[q][1])**2)
            if d < 1.0:
                row = np.zeros(2*N)
                row[p] = 1
                row[q] = 1
                A_ub_rows.append(row)
                b_ub_rows.append(d)
                n_pairs += 1

    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_rows)

    if verbose:
        print(f"  Pair constraints: {n_pairs}")
        print(f"  Total constraints: {len(b_ub)}")

    bounds = [(0, None)] * N + [(0, 1)] * N  # r >= 0, u in [0,1]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if result.success:
        val = -result.fun
        if verbose:
            rv = sorted(result.x[:N], reverse=True)
            n_active = sum(1 for x in rv if x > 1e-4)
            print(f"  LP bound: {val:.6f}, active: {n_active}")
            print(f"  Top radii: {[f'{x:.4f}' for x in rv[:5]]}")
        return val
    else:
        if verbose:
            print(f"  LP failed: {result.message}")
        return None


def grid_socp_improved(n, K=12, verbose=False):
    """
    Improved grid SOCP with additional constraints.

    Beyond pairwise non-overlap and FT area:
    - Add "clique" constraints: for 3 mutually close points, triangle inequality on radii
    - Add boundary-aware area constraints per region
    """
    coords = [(i+0.5)/K for i in range(K)]
    points = [(x, y) for x in coords for y in coords]
    N = len(points)

    def dist_to_boundary(x, y):
        return min(x, 1-x, y, 1-y)

    boundary_dist = [dist_to_boundary(x, y) for (x, y) in points]

    # Precompute distances
    dist_matrix = np.zeros((N, N))
    for p in range(N):
        for q in range(p+1, N):
            d = np.sqrt((points[p][0]-points[q][0])**2 + (points[p][1]-points[q][1])**2)
            dist_matrix[p,q] = d
            dist_matrix[q,p] = d

    if verbose:
        print(f"  Grid SOCP improved: K={K}, {N} points, n={n}")

    r = cp.Variable(N, nonneg=True)
    u = cp.Variable(N, nonneg=True)

    constraints = []

    for p in range(N):
        constraints += [r[p] <= boundary_dist[p]]
        constraints += [r[p] <= 0.5 * u[p]]
        constraints += [u[p] <= 1]

    constraints += [cp.sum(u) <= n]

    # Pairwise
    pair_count = 0
    for p in range(N):
        for q in range(p+1, N):
            d = dist_matrix[p,q]
            if d < 1.0:
                constraints += [r[p] + r[q] <= d]
                pair_count += 1

    # Triple constraints: for any 3 points forming a "small" triangle,
    # the sum of their 3 radii is bounded.
    # For points p,q,s: r_p + r_q <= d(p,q), r_p + r_s <= d(p,s), r_q + r_s <= d(q,s)
    # These are already individual pair constraints.
    # But we can add: r_p + r_q + r_s <= (d(p,q) + d(p,s) + d(q,s)) / 2
    # This is VALID because:
    # 2*(r_p+r_q+r_s) = (r_p+r_q) + (r_p+r_s) + (r_q+r_s) <= d(p,q)+d(p,s)+d(q,s)
    # So r_p+r_q+r_s <= (d(p,q)+d(p,s)+d(q,s))/2
    # This is IMPLIED by the pairwise constraints! (Sum of 3 pairwise <= sum of 3 distances,
    # divide by 2: sum of radii <= half-perimeter.)

    # So triple constraints don't add anything new over pairwise.

    # FT area
    constraints += [2*np.sqrt(3) * cp.sum_squares(r) <= 1]

    # Regional area constraints: divide into 4 quadrants and apply FT to each
    for qx in range(2):
        for qy in range(2):
            x_lo, x_hi = qx*0.5, (qx+1)*0.5
            y_lo, y_hi = qy*0.5, (qy+1)*0.5
            # Points in this quadrant
            in_quad = [p for p in range(N) if x_lo <= points[p][0] < x_hi and y_lo <= points[p][1] < y_hi]
            if in_quad:
                # Circles centered in this quadrant have sum(2*sqrt(3)*r_i^2) <= quad_area + boundary
                # Quad area = 0.25, perimeter of quad within square boundary = ...
                # For a corner quadrant: 2 boundary sides of length 0.5, 2 internal sides.
                # The circles in this quadrant use area <= 0.25 (conservative).
                # FT: sum(2*sqrt(3)*r_i^2) for circles in quadrant <= 0.25 + boundary correction
                # Conservative (no boundary correction):
                # Actually this is WRONG: circles in the quadrant can extend into other quadrants.
                # Their AREA in the quadrant is less than their total area.
                # We can't simply apply FT per quadrant.

                # Instead: at most floor(n/4)+1 circles fit in each quadrant (by pigeonhole)
                # sum(u[p] for p in in_quad) <= n  (trivially true)
                # Not useful.
                pass

    if verbose:
        print(f"  Pair constraints: {pair_count}")

    objective = cp.Maximize(cp.sum(r))
    prob = cp.Problem(objective, constraints)

    try:
        result = prob.solve(solver=cp.SCS, verbose=False, max_iters=100000, eps=1e-7)
        if verbose:
            print(f"  Status: {prob.status}, bound: {result:.6f}")
            if r.value is not None:
                rv = sorted(r.value, reverse=True)
                n_active = sum(1 for x in rv if x > 1e-4)
                print(f"  Active circles: {n_active}")
                print(f"  Top radii: {[f'{x:.4f}' for x in rv[:5]]}")
        return result
    except Exception as e:
        if verbose:
            print(f"  Error: {e}")
        return None


from scipy.optimize import linprog


def main():
    known_best = {
        1: 0.5000, 2: 0.5858, 3: 0.7645, 4: 1.0000, 5: 1.0854,
        10: 1.5911, 15: 2.0365, 20: 2.3010, 26: 2.6360, 30: 2.8425, 32: 2.9390,
    }

    if len(sys.argv) > 1:
        n_values = [int(x) for x in sys.argv[1:]]
    else:
        n_values = [1, 2, 4, 10, 26]

    print("Boundary-Aware Upper Bounds")
    print("=" * 80)

    all_results = {}
    for n_val in n_values:
        print(f"\nn = {n_val}")
        print("-" * 60)

        ft = np.sqrt(n_val / (2*np.sqrt(3)))
        print(f"  FT bound: {ft:.6f}")

        # Grid LP (no area constraint, just pairwise + cardinality)
        K_lp = min(20, max(8, int(np.sqrt(n_val) * 3)))
        lp_val = grid_lp_bound(n_val, K=K_lp, verbose=True)

        # Grid SOCP (pairwise + cardinality + FT area)
        K_socp = min(15, max(6, int(np.sqrt(n_val) * 2)))
        socp_val = grid_socp_improved(n_val, K=K_socp, verbose=True)

        bounds = [ft]
        if lp_val is not None:
            bounds.append(lp_val)
        if socp_val is not None:
            bounds.append(socp_val)

        best = min(bounds)
        known = known_best.get(n_val, 0)
        gap = best - known if known else None

        print(f"\n  FT:       {ft:.6f}")
        if lp_val is not None:
            print(f"  Grid LP:  {lp_val:.6f}")
        if socp_val is not None:
            print(f"  Grid SOCP:{socp_val:.6f}")
        print(f"  BEST:     {best:.6f}")
        if known:
            print(f"  Known:    {known:.6f}")
            print(f"  Gap:      {gap:.6f} ({100*gap/known:.2f}%)")

        all_results[str(n_val)] = {
            'ft': ft,
            'grid_lp': lp_val,
            'grid_socp': socp_val,
            'best': best,
            'known': known,
            'gap': gap,
        }

    output_path = Path(__file__).parent / "boundary_aware_bounds.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if x is not None else None)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
