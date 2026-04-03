"""
Upper bounds for circle packing in a unit square (maximize sum of radii).

Multiple approaches:
1. Area-based (Cauchy-Schwarz) bound
2. Improved area bound with containment constraints
3. LP relaxation with McCormick envelopes
4. SDP relaxation (if solver available)

All bounds are PROVABLY valid upper bounds on the optimal objective.
"""

import numpy as np
import json
import sys
from pathlib import Path


# =============================================================================
# APPROACH 1: Area-based bounds
# =============================================================================

def area_bound_basic(n):
    """
    Basic area bound using Cauchy-Schwarz.

    Each circle has area pi*r_i^2. Total area <= 1 (unit square).
    So sum(r_i^2) <= 1/pi.
    By Cauchy-Schwarz: (sum r_i)^2 <= n * sum(r_i^2) <= n/pi.
    Therefore: sum r_i <= sqrt(n/pi).
    """
    return np.sqrt(n / np.pi)


def area_bound_containment(n):
    """
    Improved area bound accounting for containment.

    Each circle of radius r_i must be contained in [0,1]^2,
    so its center is in [r_i, 1-r_i]^2. The "effective area"
    each circle occupies is pi*r_i^2, but the available area
    for center placement is (1-2r_i)^2.

    A tighter bound: the circle of radius r occupies a "footprint"
    in the square. The total footprint area cannot exceed 1.

    Actually, we can use: each circle occupies a square of side 2r_i
    (its bounding box). These bounding boxes may overlap in complex ways,
    but the circles themselves don't overlap.

    Better: use the fact that circles fit in [0,1]^2, so r_i <= 0.5 for all i.
    Combined with Cauchy-Schwarz: sum r_i <= sqrt(n * sum r_i^2).

    We optimize: max sum(r_i) subject to:
      - sum(pi * r_i^2) <= 1  (area constraint)
      - 0 < r_i <= 0.5        (containment implies r <= 0.5)

    By Lagrange multipliers with just the area constraint:
    All r_i equal => r_i = 1/sqrt(n*pi), sum = sqrt(n/pi) (same as basic).

    But with r_i <= 0.5: if 1/sqrt(n*pi) <= 0.5 (i.e., n >= 4/pi ~ 1.27),
    the constraint is not active for n >= 2. So same bound for n >= 2.
    """
    r_equal = 1.0 / np.sqrt(n * np.pi)
    if r_equal <= 0.5:
        return np.sqrt(n / np.pi)
    else:
        # n=1: r <= 0.5, so bound is 0.5
        return 0.5


def area_bound_strip(n):
    """
    Strip-based area bound.

    Divide [0,1]^2 into horizontal strips of height h.
    In a strip of height h, circles must have r <= h/2.
    The number of strips is 1/h.

    In each strip, circles of radius r have centers in a strip of effective
    height h-2r. The total "width" of non-overlapping circles in a strip
    of width 1 is at most 1 (their projections onto x-axis, each of width 2r).

    So in a strip: sum of 2*r_i <= 1 for circles in that strip,
    meaning sum r_i <= 0.5 per strip.

    With k strips: total sum r_i <= 0.5 * k.

    But this is loose. A circle can span multiple strips.

    Simpler version: the sum of diameters projected onto x-axis <= 1
    for each "row" of circles. With at most ceil(sqrt(n)) rows,
    sum r_i <= 0.5 * ceil(sqrt(n)).

    This is typically looser than the area bound for larger n.
    """
    # Simple strip bound: pack circles in rows
    # Each row has circles with sum of diameters <= 1
    # Optimal: k rows of equal circles
    # k circles per row, each of radius r, need 2kr <= 1 and 2kr <= 1 (height)
    # Actually this gives an arrangement, not a bound.

    # For a valid upper bound using strips:
    # Any circle of radius r intersects at most ceil(2r/h)+1 strips of height h
    # This gets complicated. Return the basic area bound as fallback.
    return area_bound_basic(n)


def area_bound_quarter(n):
    """
    Quarter-square bound.

    Divide the unit square into 4 quadrants of size 0.5 x 0.5.
    A circle of radius r centered in one quadrant can extend into others.

    Key insight: for each circle, its center lies in exactly one quadrant.
    The circles centered in each quadrant must fit within that quadrant
    (possibly extending outside, but their mutual non-overlap still holds).

    Let n_k = number of circles centered in quadrant k (sum n_k = n).
    For circles in quadrant k: they fit in an effective 0.5 x 0.5 square
    (approximately). Apply the area bound to each quadrant:
    sum_{i in k} r_i <= sqrt(n_k / (4*pi)) * 2 = sqrt(n_k/pi)

    Wait, that's not right. In a 0.5x0.5 square:
    sum r_i^2 <= (0.5)^2 / pi = 0.25/pi
    sum r_i <= sqrt(n_k * 0.25/pi) = 0.5 * sqrt(n_k/pi)

    Total: sum over quadrants of 0.5*sqrt(n_k/pi).
    Maximized when all n_k = n/4: total = 4 * 0.5 * sqrt(n/(4*pi)) = sqrt(n/pi).
    Same bound! The subdivision doesn't help with just area + C-S.
    """
    return area_bound_basic(n)


# =============================================================================
# APPROACH 2: LP bound via projection / 1D argument
# =============================================================================

def projection_bound(n):
    """
    Projection bound using 1D argument.

    Project all circles onto the x-axis. Circle i projects to interval
    [x_i - r_i, x_i + r_i] of length 2*r_i.

    These intervals are contained in [0, 1].
    Non-overlapping circles project to potentially overlapping intervals
    (circles at different y can overlap in x-projection).

    But: in any vertical line x = c, at most floor(1/(2*r_min)) circles
    can be intersected. This is hard to use directly.

    Alternative 1D bound:
    Consider the "depth" of the arrangement. At any point x in [0,1],
    the number of circles whose x-projection covers x is bounded by
    the number of circles that can be stacked vertically: at most 1/(2*r_min).

    For a cleaner bound: by Helly's theorem in 1D, we can partition
    circles into "layers" where each layer has non-overlapping x-projections.
    Each layer contributes sum(2*r_i) <= 1, so sum(r_i) <= 0.5 per layer.

    The number of layers needed is the maximum "depth" at any x.
    If we could show depth <= d, then sum(r_i) <= 0.5 * d.

    For n=26, if depth <= 6: sum r_i <= 3.0 (loose).
    If depth <= 5: sum r_i <= 2.5 (too tight! contradicts known 2.636).

    So depth must be >= 6 for n=26. Bound: 3.0.
    This is looser than area bound (2.877).
    """
    # The depth is at most how many circles fit vertically:
    # In a column of height 1, circles of radius r need height >= 2r each.
    # With varying radii, max depth at any x is bounded by n.
    # This gives sum r_i <= 0.5 * n which is useless.

    # Better: depth <= floor(1/(2*r_min)) but we don't know r_min.
    # Just return area bound.
    return area_bound_basic(n)


# =============================================================================
# APPROACH 3: LP Relaxation with McCormick Envelopes
# =============================================================================

def lp_upper_bound(n, verbose=False):
    """
    LP relaxation for circle packing upper bound.

    Variables: x_i, y_i, r_i for i=1..n

    Objective: maximize sum(r_i)

    Constraints:
    - Containment: r_i <= x_i <= 1 - r_i, r_i <= y_i <= 1 - r_i
      Equivalently: x_i + r_i <= 1, x_i - r_i >= 0, same for y.
    - Non-overlap: (x_i-x_j)^2 + (y_i-y_j)^2 >= (r_i+r_j)^2

    The non-overlap constraint is nonconvex. We relax it.

    Expanding: x_i^2 - 2x_ix_j + x_j^2 + y_i^2 - 2y_iy_j + y_j^2
               >= r_i^2 + 2r_ir_j + r_j^2

    Introduce: X_ii = x_i^2, Y_ii = y_i^2, R_ii = r_i^2
               X_ij = x_ix_j, Y_ij = y_iy_j, R_ij = r_ir_j

    Non-overlap becomes LINEAR in these lifted variables:
    X_ii - 2X_ij + X_jj + Y_ii - 2Y_ij + Y_jj >= R_ii + 2R_ij + R_jj

    McCormick envelopes for product w = u*v where u in [uL, uU], v in [vL, vU]:
    w >= uL*v + u*vL - uL*vL
    w >= uU*v + u*vU - uU*vU
    w <= uL*v + u*vU - uL*vU
    w <= uU*v + u*vL - uU*vL

    Variable bounds:
    x_i in [0, 1], y_i in [0, 1], r_i in [0, 0.5]
    But tighter: x_i in [r_i, 1-r_i], so x_i depends on r_i.
    For McCormick, we need fixed bounds. Use:
    x_i in [0, 1], y_i in [0, 1], r_i in [0, 0.5]
    """
    try:
        import cvxpy as cp
    except ImportError:
        print("cvxpy not available, skipping LP bound")
        return None

    # Variables
    x = cp.Variable(n, name="x")
    y = cp.Variable(n, name="y")
    r = cp.Variable(n, name="r")

    # Lifted variables for diagonal terms
    Xii = cp.Variable(n, name="Xii")  # x_i^2
    Yii = cp.Variable(n, name="Yii")  # y_i^2
    Rii = cp.Variable(n, name="Rii")  # r_i^2

    constraints = []

    # Bounds on original variables
    constraints += [r >= 1e-6]  # positive radii (use small epsilon)
    constraints += [r <= 0.5]
    constraints += [x >= r]
    constraints += [x <= 1 - r]
    constraints += [y >= r]
    constraints += [y <= 1 - r]

    # Lifted variables for cross terms (only for pairs i<j)
    num_pairs = n * (n - 1) // 2
    if num_pairs == 0:
        # n=1: no pairs, just containment
        objective = cp.Maximize(cp.sum(r))
        prob = cp.Problem(objective, constraints)
        result = prob.solve(solver=cp.SCS, verbose=False)
        return result

    Xij = cp.Variable(num_pairs, name="Xij")  # x_i * x_j
    Yij = cp.Variable(num_pairs, name="Yij")  # y_i * y_j
    Rij = cp.Variable(num_pairs, name="Rij")  # r_i * r_j

    # McCormick envelopes for x_i^2 (u=v=x_i, bounds [0,1])
    # w = x_i^2: w >= 2*x_i*0 + 0 - 0 = 0 (trivial)
    #            w >= 2*x_i*1 - 1 = 2x_i - 1
    #            w <= x_i (from 0*x_i + x_i*1 - 0*1 = x_i)
    #            w <= x_i (from 1*x_i + x_i*0 - 1*0 = x_i)
    # So: Xii >= 0, Xii >= 2x_i - 1, Xii <= x_i
    for i in range(n):
        constraints += [Xii[i] >= 0]
        constraints += [Xii[i] >= 2*x[i] - 1]
        constraints += [Xii[i] <= x[i]]

    # McCormick for y_i^2 (same bounds [0,1])
    for i in range(n):
        constraints += [Yii[i] >= 0]
        constraints += [Yii[i] >= 2*y[i] - 1]
        constraints += [Yii[i] <= y[i]]

    # McCormick for r_i^2 (bounds [0, 0.5])
    # w = r_i^2, u=v=r_i in [0, 0.5]
    # w >= 0, w >= 2*r_i*0.5 - 0.25 = r_i - 0.25
    # w <= 0.5*r_i (from both upper McCormick)
    for i in range(n):
        constraints += [Rii[i] >= 0]
        constraints += [Rii[i] >= r[i] - 0.25]
        constraints += [Rii[i] <= 0.5 * r[i]]

    # McCormick for cross terms x_i*x_j, y_i*y_j, r_i*r_j
    pair_idx = 0
    pair_map = {}
    for i in range(n):
        for j in range(i+1, n):
            pair_map[(i,j)] = pair_idx

            # x_i * x_j: both in [0, 1]
            # w >= 0*x_j + x_i*0 - 0 = 0
            # w >= 1*x_j + x_i*1 - 1 = x_i + x_j - 1
            # w <= 0*x_j + x_i*1 - 0 = x_i
            # w <= 1*x_j + x_i*0 - 0 = x_j
            constraints += [Xij[pair_idx] >= 0]
            constraints += [Xij[pair_idx] >= x[i] + x[j] - 1]
            constraints += [Xij[pair_idx] <= x[i]]
            constraints += [Xij[pair_idx] <= x[j]]

            # y_i * y_j: both in [0, 1]
            constraints += [Yij[pair_idx] >= 0]
            constraints += [Yij[pair_idx] >= y[i] + y[j] - 1]
            constraints += [Yij[pair_idx] <= y[i]]
            constraints += [Yij[pair_idx] <= y[j]]

            # r_i * r_j: both in [0, 0.5]
            # w >= 0 + 0 - 0 = 0
            # w >= 0.5*r_j + r_i*0.5 - 0.25
            # w <= 0.5*r_j (from upper)
            # w <= 0.5*r_i
            constraints += [Rij[pair_idx] >= 0]
            constraints += [Rij[pair_idx] >= 0.5*r[i] + 0.5*r[j] - 0.25]
            constraints += [Rij[pair_idx] <= 0.5 * r[j]]
            constraints += [Rij[pair_idx] <= 0.5 * r[i]]

            pair_idx += 1

    # Non-overlap constraints (linearized):
    # X_ii - 2X_ij + X_jj + Y_ii - 2Y_ij + Y_jj >= R_ii + 2R_ij + R_jj
    for i in range(n):
        for j in range(i+1, n):
            k = pair_map[(i,j)]
            constraints += [
                Xii[i] - 2*Xij[k] + Xii[j] +
                Yii[i] - 2*Yij[k] + Yii[j] >=
                Rii[i] + 2*Rij[k] + Rii[j]
            ]

    # Objective: maximize sum of radii
    objective = cp.Maximize(cp.sum(r))

    prob = cp.Problem(objective, constraints)

    if verbose:
        print(f"LP relaxation for n={n}:")
        print(f"  Variables: {3*n + 3*n + 3*num_pairs}")
        print(f"  Constraints: {len(constraints)}")

    try:
        result = prob.solve(solver=cp.SCS, max_iters=10000, verbose=False)
        if prob.status in ['infeasible', 'unbounded']:
            if verbose:
                print(f"  Status: {prob.status}")
            return None
        if verbose:
            print(f"  Status: {prob.status}")
            print(f"  Upper bound: {result:.6f}")
            print(f"  Radii: {np.sort(r.value)[::-1][:5]}...")
        return result
    except Exception as e:
        if verbose:
            print(f"  Solver error: {e}")
        # Try ECOS
        try:
            result = prob.solve(solver=cp.ECOS, verbose=False)
            if verbose:
                print(f"  Status (ECOS): {prob.status}")
                print(f"  Upper bound: {result:.6f}")
            return result
        except:
            return None


# =============================================================================
# APPROACH 4: Tightened LP with additional valid inequalities
# =============================================================================

def lp_upper_bound_tight(n, verbose=False):
    """
    Tighter LP relaxation with additional valid inequalities:

    1. Area constraint: sum(pi * r_i^2) <= 1
       In lifted form: sum(pi * Rii) <= 1

    2. Containment-area: each circle fits in [r_i, 1-r_i]^2,
       so its available area is (1-2r_i)^2. The fraction of the
       unit square "used" by circle i is pi*r_i^2 / 1 = pi*r_i^2.
       Total: sum pi*r_i^2 <= 1.

    3. Pairwise distance bounds from geometry.

    4. Row packing: project onto x-axis. Non-overlapping circles at
       similar y-coordinates have non-overlapping x-projections.
       This is hard to encode without knowing y, but we can add:
       For any pair: |x_i - x_j| + |y_i - y_j| >= r_i + r_j
       (L1 distance >= sum of radii is WEAKER than L2, but it's a
       necessary condition that's easy to linearize.)
       Actually NO: L1 >= L2 is false. L1 >= r_i+r_j does NOT follow
       from L2 >= r_i+r_j. Skip this.

    5. Triangle inequality on distances: for triples i,j,k:
       d_ij + d_jk >= d_ik. With d_ij >= r_i+r_j, etc.
       This gives: some constraints on radii triples.
    """
    try:
        import cvxpy as cp
    except ImportError:
        return None

    x = cp.Variable(n, name="x")
    y = cp.Variable(n, name="y")
    r = cp.Variable(n, name="r")
    Xii = cp.Variable(n, name="Xii")
    Yii = cp.Variable(n, name="Yii")
    Rii = cp.Variable(n, name="Rii")

    num_pairs = n * (n - 1) // 2

    constraints = []

    # Basic bounds
    constraints += [r >= 1e-6, r <= 0.5]
    constraints += [x >= r, x <= 1 - r]
    constraints += [y >= r, y <= 1 - r]

    if num_pairs == 0:
        # n=1 case
        constraints += [np.pi * cp.sum(Rii) <= 1]
        for i in range(n):
            constraints += [Xii[i] >= 0, Xii[i] >= 2*x[i] - 1, Xii[i] <= x[i]]
            constraints += [Yii[i] >= 0, Yii[i] >= 2*y[i] - 1, Yii[i] <= y[i]]
            constraints += [Rii[i] >= 0, Rii[i] >= r[i] - 0.25, Rii[i] <= 0.5*r[i]]
        objective = cp.Maximize(cp.sum(r))
        prob = cp.Problem(objective, constraints)
        result = prob.solve(solver=cp.SCS, verbose=False)
        return result

    Xij = cp.Variable(num_pairs, name="Xij")
    Yij = cp.Variable(num_pairs, name="Yij")
    Rij = cp.Variable(num_pairs, name="Rij")

    # McCormick for squared terms (same as before)
    for i in range(n):
        constraints += [Xii[i] >= 0, Xii[i] >= 2*x[i] - 1, Xii[i] <= x[i]]
        constraints += [Yii[i] >= 0, Yii[i] >= 2*y[i] - 1, Yii[i] <= y[i]]
        constraints += [Rii[i] >= 0, Rii[i] >= r[i] - 0.25, Rii[i] <= 0.5*r[i]]

    pair_idx = 0
    pair_map = {}
    for i in range(n):
        for j in range(i+1, n):
            pair_map[(i,j)] = pair_idx
            constraints += [Xij[pair_idx] >= 0, Xij[pair_idx] >= x[i]+x[j]-1,
                          Xij[pair_idx] <= x[i], Xij[pair_idx] <= x[j]]
            constraints += [Yij[pair_idx] >= 0, Yij[pair_idx] >= y[i]+y[j]-1,
                          Yij[pair_idx] <= y[i], Yij[pair_idx] <= y[j]]
            constraints += [Rij[pair_idx] >= 0, Rij[pair_idx] >= 0.5*r[i]+0.5*r[j]-0.25,
                          Rij[pair_idx] <= 0.5*r[j], Rij[pair_idx] <= 0.5*r[i]]
            pair_idx += 1

    # Non-overlap (linearized)
    for i in range(n):
        for j in range(i+1, n):
            k = pair_map[(i,j)]
            constraints += [
                Xii[i] - 2*Xij[k] + Xii[j] + Yii[i] - 2*Yij[k] + Yii[j] >=
                Rii[i] + 2*Rij[k] + Rii[j]
            ]

    # ADDITIONAL VALID INEQUALITIES:

    # 1. Area constraint: sum(pi * r_i^2) <= 1
    constraints += [np.pi * cp.sum(Rii) <= 1]

    # 2. Containment strengthening: x_i >= r_i means x_i^2 >= r_i^2
    #    (since both are non-negative). So Xii >= Rii, Yii >= Rii.
    #    Wait: x_i >= r_i doesn't imply x_i^2 >= r_i^2 in the relaxation
    #    because Xii and Rii are relaxed. But we can ADD this as a valid cut:
    #    Since x_i >= r_i >= 0, we have x_i^2 >= r_i^2.
    for i in range(n):
        constraints += [Xii[i] >= Rii[i]]
        constraints += [Yii[i] >= Rii[i]]

    # 3. Also: (1-x_i) >= r_i, so (1-x_i)^2 >= r_i^2
    #    1 - 2x_i + x_i^2 >= r_i^2
    #    Xii[i] - 2x[i] + 1 >= Rii[i]
    for i in range(n):
        constraints += [Xii[i] - 2*x[i] + 1 >= Rii[i]]
        constraints += [Yii[i] - 2*y[i] + 1 >= Rii[i]]

    # 4. sum(r_i) <= n * 0.5 (trivial but helps solver)
    constraints += [cp.sum(r) <= n * 0.5]

    # 5. Symmetry-breaking: order radii (optional, may help or hurt)
    # Don't add this for now - it changes the feasible set structure.

    objective = cp.Maximize(cp.sum(r))
    prob = cp.Problem(objective, constraints)

    if verbose:
        print(f"Tight LP relaxation for n={n}:")
        print(f"  Pairs: {num_pairs}")

    try:
        result = prob.solve(solver=cp.SCS, max_iters=20000, verbose=False)
        if prob.status in ['infeasible', 'unbounded']:
            if verbose:
                print(f"  Status: {prob.status}")
            return None
        if verbose:
            print(f"  Status: {prob.status}")
            print(f"  Upper bound: {result:.6f}")
        return result
    except Exception as e:
        if verbose:
            print(f"  Error: {e}")
        return None


# =============================================================================
# APPROACH 5: SDP Relaxation
# =============================================================================

def sdp_upper_bound(n, verbose=False):
    """
    SDP relaxation using Shor's relaxation (moment matrix).

    Variables: v = [x_1,...,x_n, y_1,...,y_n, r_1,...,r_n] (3n variables)

    Introduce moment matrix M = [1; v] [1; v]^T, size (3n+1) x (3n+1).
    M[0,0] = 1
    M[i,j] = v_i * v_j (for i,j >= 1)
    M[0,i] = v_i

    Constraint: M psd (positive semidefinite)

    All quadratic constraints become linear in M.

    This is a (3n+1) x (3n+1) SDP, which is large for n=26 (79x79).
    Start with small n.
    """
    try:
        import cvxpy as cp
    except ImportError:
        return None

    dim = 3 * n + 1  # [1, x1..xn, y1..yn, r1..rn]

    M = cp.Variable((dim, dim), symmetric=True, name="M")

    constraints = []

    # M is PSD
    constraints += [M >> 0]

    # M[0,0] = 1
    constraints += [M[0, 0] == 1]

    # Index helpers
    def xi(i): return 1 + i           # x_i at index 1..n
    def yi(i): return 1 + n + i       # y_i at index n+1..2n
    def ri(i): return 1 + 2*n + i     # r_i at index 2n+1..3n

    # Bounds on first-order moments (from M[0, k] = E[v_k])
    for i in range(n):
        # x_i in [0, 1]
        constraints += [M[0, xi(i)] >= 0, M[0, xi(i)] <= 1]
        # y_i in [0, 1]
        constraints += [M[0, yi(i)] >= 0, M[0, yi(i)] <= 1]
        # r_i in [0, 0.5]
        constraints += [M[0, ri(i)] >= 0, M[0, ri(i)] <= 0.5]

    # Bounds on second-order moments
    for i in range(n):
        # x_i^2 in [0, 1]
        constraints += [M[xi(i), xi(i)] >= 0, M[xi(i), xi(i)] <= 1]
        # y_i^2 in [0, 1]
        constraints += [M[yi(i), yi(i)] >= 0, M[yi(i), yi(i)] <= 1]
        # r_i^2 in [0, 0.25]
        constraints += [M[ri(i), ri(i)] >= 0, M[ri(i), ri(i)] <= 0.25]

    # Containment: x_i >= r_i => x_i - r_i >= 0
    # In moment form: E[(x_i - r_i)] >= 0 => M[0,xi(i)] - M[0,ri(i)] >= 0
    # And: E[(x_i - r_i)^2] >= 0 is automatic from PSD
    # But stronger: E[(x_i - r_i) * 1] >= 0 and E[(1 - x_i - r_i) * 1] >= 0
    for i in range(n):
        constraints += [M[0, xi(i)] >= M[0, ri(i)]]  # x_i >= r_i
        constraints += [M[0, xi(i)] + M[0, ri(i)] <= 1]  # x_i + r_i <= 1
        constraints += [M[0, yi(i)] >= M[0, ri(i)]]  # y_i >= r_i
        constraints += [M[0, yi(i)] + M[0, ri(i)] <= 1]  # y_i + r_i <= 1

    # Localizing constraints for containment:
    # (x_i - r_i) >= 0 => for all j: (x_i - r_i)*v_j products are valid
    # E[(x_i - r_i) * (x_i - r_i)] >= 0:
    # M[xi,xi] - 2*M[xi,ri] + M[ri,ri] >= 0
    for i in range(n):
        constraints += [M[xi(i),xi(i)] - 2*M[xi(i),ri(i)] + M[ri(i),ri(i)] >= 0]
        # (1-x_i-r_i)(1-x_i-r_i) >= 0:
        # 1 - 2x_i - 2r_i + x_i^2 + 2x_ir_i + r_i^2 >= 0
        constraints += [1 - 2*M[0,xi(i)] - 2*M[0,ri(i)] + M[xi(i),xi(i)] + 2*M[xi(i),ri(i)] + M[ri(i),ri(i)] >= 0]
        constraints += [M[yi(i),yi(i)] - 2*M[yi(i),ri(i)] + M[ri(i),ri(i)] >= 0]
        constraints += [1 - 2*M[0,yi(i)] - 2*M[0,ri(i)] + M[yi(i),yi(i)] + 2*M[yi(i),ri(i)] + M[ri(i),ri(i)] >= 0]

    # Non-overlap: (x_i-x_j)^2 + (y_i-y_j)^2 >= (r_i+r_j)^2
    # Expanded: x_i^2 - 2x_ix_j + x_j^2 + y_i^2 - 2y_iy_j + y_j^2 >= r_i^2 + 2r_ir_j + r_j^2
    for i in range(n):
        for j in range(i+1, n):
            constraints += [
                M[xi(i),xi(i)] - 2*M[xi(i),xi(j)] + M[xi(j),xi(j)] +
                M[yi(i),yi(i)] - 2*M[yi(i),yi(j)] + M[yi(j),yi(j)] >=
                M[ri(i),ri(i)] + 2*M[ri(i),ri(j)] + M[ri(j),ri(j)]
            ]

    # Area constraint: sum(pi * r_i^2) <= 1
    constraints += [np.pi * sum(M[ri(i),ri(i)] for i in range(n)) <= 1]

    # Objective: maximize sum(r_i) = sum(M[0, ri(i)])
    objective = cp.Maximize(sum(M[0, ri(i)] for i in range(n)))

    prob = cp.Problem(objective, constraints)

    if verbose:
        print(f"SDP relaxation for n={n}:")
        print(f"  Matrix size: {dim}x{dim}")
        print(f"  Constraints: {len(constraints)}")

    try:
        result = prob.solve(solver=cp.SCS, max_iters=20000, verbose=False,
                          eps=1e-6)
        if prob.status in ['infeasible', 'unbounded']:
            if verbose:
                print(f"  Status: {prob.status}")
            return None
        if verbose:
            print(f"  Status: {prob.status}")
            print(f"  Upper bound: {result:.6f}")
            radii = [M.value[0, ri(i)] for i in range(n)]
            print(f"  Radii (sorted): {sorted(radii, reverse=True)[:5]}...")
        return result
    except Exception as e:
        if verbose:
            print(f"  Error: {e}")
        return None


# =============================================================================
# MAIN: Run all bounds and compare
# =============================================================================

def run_all_bounds(n_values=None, verbose=True):
    """Run all upper bound methods for given n values."""
    if n_values is None:
        n_values = [1, 2, 3, 4, 5, 10, 15, 20, 26]

    # Known best solutions for reference
    known_best = {
        1: 0.5000,
        2: 0.5858,  # 1 - 1/sqrt(2) + 0.5 ≈ each r=(sqrt(2)-1)/2
        3: 0.7645,
        4: 1.0000,  # 4 circles of r=0.25
        5: 1.0854,
        10: 1.5911,
        15: 2.0365,
        20: 2.3010,
        26: 2.6360,
        30: 2.8425,
        32: 2.9390,
    }

    results = {}

    for n in n_values:
        if verbose:
            print(f"\n{'='*60}")
            print(f"n = {n}")
            print(f"{'='*60}")

        bounds = {}

        # Area bound
        ab = area_bound_basic(n)
        bounds['area_basic'] = ab
        if verbose:
            print(f"  Area (Cauchy-Schwarz): {ab:.6f}")

        # Area bound with containment
        ac = area_bound_containment(n)
        bounds['area_contain'] = ac
        if verbose and ac != ab:
            print(f"  Area (containment):    {ac:.6f}")

        # LP relaxation
        if n <= 30:
            lp = lp_upper_bound(n, verbose=verbose)
            if lp is not None:
                bounds['lp_basic'] = lp
                if verbose:
                    print(f"  LP (McCormick):        {lp:.6f}")

        # Tight LP
        if n <= 30:
            lpt = lp_upper_bound_tight(n, verbose=verbose)
            if lpt is not None:
                bounds['lp_tight'] = lpt
                if verbose:
                    print(f"  LP (tight):            {lpt:.6f}")

        # SDP (only for small n due to computational cost)
        if n <= 15:
            sdp = sdp_upper_bound(n, verbose=verbose)
            if sdp is not None:
                bounds['sdp'] = sdp
                if verbose:
                    print(f"  SDP (Shor):            {sdp:.6f}")

        # Best bound
        best = min(v for v in bounds.values() if v is not None)
        bounds['best'] = best

        if verbose:
            print(f"\n  BEST UPPER BOUND:      {best:.6f}")
            if n in known_best:
                gap = best - known_best[n]
                gap_pct = 100 * gap / known_best[n]
                print(f"  Known best solution:   {known_best[n]:.6f}")
                print(f"  Gap:                   {gap:.6f} ({gap_pct:.2f}%)")

        results[n] = bounds

    return results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        n_values = [int(x) for x in sys.argv[1:]]
    else:
        n_values = [1, 2, 3, 4, 5, 10, 26]

    results = run_all_bounds(n_values, verbose=True)

    # Save results
    output_path = Path(__file__).parent / "bounds_results.json"
    serializable = {}
    for n, bounds in results.items():
        serializable[str(n)] = {k: float(v) if v is not None else None
                                 for k, v in bounds.items()}
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_path}")
