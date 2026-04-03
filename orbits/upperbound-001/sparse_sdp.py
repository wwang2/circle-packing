"""
Sparse SDP upper bound via pair-wise order-2 Lasserre relaxation.

The order-1 Lasserre relaxation gives bound = sqrt(n/pi) (area bound).
To do better, we need order-2 moments.

Full order-2 is impractical (O(n^4) matrix entries).
Instead: use SPARSE Lasserre (Waki et al., 2006).

The circle packing constraints have SPARSE structure:
- Each non-overlap constraint involves only 6 variables: (xi, yi, ri, xj, yj, rj)
- Each containment constraint involves 3 variables: (xi, yi, ri)

For the sparse relaxation:
- For each "clique" of variables, build a local moment matrix of order 2
- Link cliques via shared variables (consistency constraints)

For each pair (i,j), the clique is {xi, yi, ri, xj, yj, rj} (6 vars).
Order-2 moment matrix for 6 vars: monomials up to degree 2.
Number of monomials: C(6+2, 2) = 28.
So each local moment matrix is 28x28.

With n*(n-1)/2 pairs, total: 325 * 28^2 ~ 255K entries for n=26.
Plus consistency constraints. This should be feasible.

But actually, for n=26 with 325 pairs each with a 28x28 PSD constraint,
that's still quite large. Let me try for small n first.
"""

import numpy as np
import sys
import json
from pathlib import Path
from itertools import combinations_with_replacement


def sparse_order2_bound(n, verbose=False):
    """
    Sparse order-2 Lasserre for circle packing.

    For efficiency, instead of building the full order-2 moment matrix,
    we build order-2 moment matrices only for PAIRS of circles.

    For pair (i,j), variables are v = (xi, yi, ri, xj, yj, rj).
    Monomials of degree <= 2: 1, xi, yi, ri, xj, yj, rj,
      xi^2, xiyi, xiri, xixj, xiyj, xirj,
      yi^2, yiri, yixj, yiyj, yirj,
      ri^2, rixj, riyj, rirj,
      xj^2, xjyj, xjrj,
      yj^2, yjrj,
      rj^2
    That's 1 + 6 + 21 = 28 monomials.

    The 28x28 moment matrix M_ij has entries E[m_a * m_b] where m_a, m_b
    are monomials of degree <= 2. The entries are moments up to degree 4.

    This is too many new variables per pair. For n=26, 325 pairs * 28*29/2 = 325*406 ~ 132K variables.

    Simpler approach: just use the degree-2 LOCALIZING MATRIX for each pair.

    Actually, let me try a MUCH simpler approach that might work better:
    use the order-1 SDP but add SPECIFIC valid cuts derived from
    the structure of circle packing.
    """
    return _structural_sdp_bound(n, verbose)


def _structural_sdp_bound(n, verbose=False):
    """
    SDP with structural cuts specific to circle packing.

    Key insight: in any feasible packing, circles that are close together
    have their radii constrained by their mutual distances. We can derive
    valid inequalities from specific geometric configurations.

    CUT 1: Triplet cuts.
    For any three circles i,j,k, the three pairwise distances satisfy
    the triangle inequality. Combined with non-overlap:
    d_ij >= r_i + r_j, d_jk >= r_j + r_k, d_ik >= r_i + r_k
    Triangle ineq: d_ij + d_jk >= d_ik
    => (r_i + r_j) + (r_j + r_k) >= d_ik >= r_i + r_k
    => r_i + 2r_j + r_k >= r_i + r_k, which is trivially true (r_j > 0).

    But: d_ij^2 >= (r_i+r_j)^2 and d_ij <= sqrt(2) (diagonal of unit square).
    So (r_i+r_j)^2 <= 2, meaning r_i + r_j <= sqrt(2).
    For all pairs: r_i + r_j <= sqrt(2) ~ 1.414.

    This is a VALID CUT that we haven't used!

    CUT 2: Packing density.
    In any disk of radius R, the maximum packing density of circles is
    pi/(2*sqrt(3)) ~ 0.9069 (hexagonal packing).
    So sum(pi*r_i^2) <= 0.9069 * Area.
    For unit square: sum(r_i^2) <= 0.9069/pi ~ 0.2887.
    Compare with sum(r_i^2) <= 1/pi ~ 0.3183.
    But this density bound is for INFINITE packings and may not hold for
    finite packings in a square. Skip for now.

    CUT 3: For any circle i, the sum of radii of all circles it touches
    is bounded. Specifically, a circle of radius r can be tangent to at most
    k(r) circles, where k depends on r. For equal circles, k <= 6 (kissing number).

    CUT 4: Edge effects. Circles near the boundary have less room.
    r_i <= x_i <= 1 - r_i, so r_i <= 0.5.
    Also: x_i * (1-x_i) >= r_i * (1-r_i) ... no, that's not right.
    x_i >= r_i and x_i <= 1-r_i, so 2r_i <= 1, r_i <= 0.5.
    Also: x_i*(1-x_i) >= r_i*(1-r_i)? No, x_i could be r_i, giving r_i*(1-r_i).
    Not useful.

    CUT 5: diameter constraint.
    For any pair (i,j): distance between centers <= sqrt(2) (diagonal).
    Also: distance >= r_i + r_j. So r_i + r_j <= sqrt(2).

    In moment form: M[0,ri(i)] + M[0,ri(j)] <= sqrt(2).
    """
    try:
        import cvxpy as cp
    except ImportError:
        return None

    dim = 3 * n + 1
    def xi(i): return 1 + i
    def yi(i): return 1 + n + i
    def ri(i): return 1 + 2*n + i

    M = cp.Variable((dim, dim), symmetric=True, name="M")
    constraints = []

    # PSD and normalization
    constraints += [M >> 0]
    constraints += [M[0, 0] == 1]

    # First-order bounds
    for i in range(n):
        constraints += [M[0, xi(i)] >= 0, M[0, xi(i)] <= 1]
        constraints += [M[0, yi(i)] >= 0, M[0, yi(i)] <= 1]
        constraints += [M[0, ri(i)] >= 0, M[0, ri(i)] <= 0.5]

    # Second-order bounds
    for i in range(n):
        constraints += [M[xi(i), xi(i)] >= 0, M[xi(i), xi(i)] <= 1]
        constraints += [M[yi(i), yi(i)] >= 0, M[yi(i), yi(i)] <= 1]
        constraints += [M[ri(i), ri(i)] >= 0, M[ri(i), ri(i)] <= 0.25]

    # Containment
    for i in range(n):
        constraints += [M[0, xi(i)] >= M[0, ri(i)]]
        constraints += [M[0, xi(i)] + M[0, ri(i)] <= 1]
        constraints += [M[0, yi(i)] >= M[0, ri(i)]]
        constraints += [M[0, yi(i)] + M[0, ri(i)] <= 1]
        # Second-order containment
        constraints += [M[xi(i),xi(i)] - 2*M[xi(i),ri(i)] + M[ri(i),ri(i)] >= 0]
        constraints += [1 - 2*M[0,xi(i)] - 2*M[0,ri(i)] + M[xi(i),xi(i)] + 2*M[xi(i),ri(i)] + M[ri(i),ri(i)] >= 0]
        constraints += [M[yi(i),yi(i)] - 2*M[yi(i),ri(i)] + M[ri(i),ri(i)] >= 0]
        constraints += [1 - 2*M[0,yi(i)] - 2*M[0,ri(i)] + M[yi(i),yi(i)] + 2*M[yi(i),ri(i)] + M[ri(i),ri(i)] >= 0]

    # Non-overlap (basic linear in M)
    for i in range(n):
        for j in range(i+1, n):
            constraints += [
                M[xi(i),xi(i)] - 2*M[xi(i),xi(j)] + M[xi(j),xi(j)] +
                M[yi(i),yi(i)] - 2*M[yi(i),yi(j)] + M[yi(j),yi(j)] >=
                M[ri(i),ri(i)] + 2*M[ri(i),ri(j)] + M[ri(j),ri(j)]
            ]

    # Area constraint
    constraints += [np.pi * sum(M[ri(i),ri(i)] for i in range(n)) <= 1]

    # =================================================================
    # NEW CUTS
    # =================================================================

    # CUT 1: Pairwise sum of radii bounded by sqrt(2) (diagonal distance)
    # r_i + r_j <= sqrt(2) for all pairs
    sqrt2 = np.sqrt(2)
    for i in range(n):
        for j in range(i+1, n):
            constraints += [M[0, ri(i)] + M[0, ri(j)] <= sqrt2]

    # CUT 2: Second-order pairwise: E[(r_i + r_j)^2] <= 2 (since dist <= sqrt(2))
    # Rii + 2Rij + Rjj <= 2
    for i in range(n):
        for j in range(i+1, n):
            constraints += [M[ri(i),ri(i)] + 2*M[ri(i),ri(j)] + M[ri(j),ri(j)] <= 2]

    # CUT 3: In the x-direction, for circles at similar y:
    # The total "width" used by any set of circles whose y-ranges overlap
    # is bounded. Hard to encode without knowing y.
    # But: for ANY pair, either their x-ranges or y-ranges must allow them
    # to fit. This is already captured by non-overlap.

    # CUT 4: RLT cuts from containment products
    for i in range(n):
        for j in range(i+1, n):
            # (x_i - r_i)(x_j - r_j) >= 0
            constraints += [M[xi(i),xi(j)] - M[xi(i),ri(j)] - M[ri(i),xi(j)] + M[ri(i),ri(j)] >= 0]
            # (1-x_i-r_i)(1-x_j-r_j) >= 0
            constraints += [
                1 - M[0,xi(i)] - M[0,ri(i)] - M[0,xi(j)] - M[0,ri(j)]
                + M[xi(i),xi(j)] + M[xi(i),ri(j)] + M[ri(i),xi(j)] + M[ri(i),ri(j)] >= 0
            ]
            # (x_i - r_i)(1-x_j-r_j) >= 0
            constraints += [
                M[0,xi(i)] - M[0,ri(i)] - M[xi(i),xi(j)] - M[xi(i),ri(j)]
                + M[ri(i),xi(j)] + M[ri(i),ri(j)] >= 0
            ]
            # Same for y
            constraints += [M[yi(i),yi(j)] - M[yi(i),ri(j)] - M[ri(i),yi(j)] + M[ri(i),ri(j)] >= 0]
            constraints += [
                1 - M[0,yi(i)] - M[0,ri(i)] - M[0,yi(j)] - M[0,ri(j)]
                + M[yi(i),yi(j)] + M[yi(i),ri(j)] + M[ri(i),yi(j)] + M[ri(i),ri(j)] >= 0
            ]
            constraints += [
                M[0,yi(i)] - M[0,ri(i)] - M[yi(i),yi(j)] - M[yi(i),ri(j)]
                + M[ri(i),yi(j)] + M[ri(i),ri(j)] >= 0
            ]

            # r_i * (0.5 - r_j) >= 0
            constraints += [0.5*M[0,ri(i)] - M[ri(i),ri(j)] >= 0]
            constraints += [0.5*M[0,ri(j)] - M[ri(i),ri(j)] >= 0]

    # CUT 5: Strengthened non-overlap using containment.
    # Since x_i in [r_i, 1-r_i], we have:
    # (x_i - x_j)^2 <= max((1-r_i - r_j)^2, ...) <= 1 (loose)
    # But: x_i^2 <= x_i (since x_i in [0,1]) -- already have this from McCormick.
    # Key: x_i * r_i >= r_i^2 (since x_i >= r_i, r_i >= 0)
    for i in range(n):
        constraints += [M[xi(i), ri(i)] >= M[ri(i), ri(i)]]
        constraints += [M[yi(i), ri(i)] >= M[ri(i), ri(i)]]
        # Also: (1-x_i)*r_i >= r_i^2, so r_i - x_i*r_i >= r_i^2
        # M[0,ri(i)] - M[xi(i),ri(i)] >= M[ri(i),ri(i)]
        constraints += [M[0,ri(i)] - M[xi(i),ri(i)] >= M[ri(i),ri(i)]]
        constraints += [M[0,ri(i)] - M[yi(i),ri(i)] >= M[ri(i),ri(i)]]

    # CUT 6: Stronger area-like bound.
    # Each circle needs "elbow room": the square [x_i-r_i, x_i+r_i] x [y_i-r_i, y_i+r_i]
    # has area 4*r_i^2 and is contained in [0,1]^2.
    # These bounding squares CAN overlap, so we can't just say sum(4*r_i^2) <= 1.
    # But: the inscribed circles don't overlap, so we use the pi*r_i^2 bound.
    # Already have this.

    # CUT 7: Triple radius sum.
    # For any 3 circles, their pairwise distances satisfy triangle inequality.
    # This doesn't directly bound radii but creates relationships.
    # d_ij + d_ik >= d_jk, and d_ij >= r_i+r_j, d_ik >= r_i+r_k.
    # So: (r_i+r_j) + (r_i+r_k) >= d_jk >= r_j+r_k
    # => 2*r_i + r_j + r_k >= r_j + r_k, trivially true.
    # These are too weak.

    # CUT 8: Sector packing. For large n, consider that the corners
    # can only fit small circles. A circle touching both walls at a corner
    # has r <= 1/(2+sqrt(2)) ~ 0.293.
    # Actually, a circle in the corner with center at (r,r) uses wall space.
    # The corner region is constrained. But this is hard to encode generally.

    # Objective
    objective = cp.Maximize(sum(M[0, ri(i)] for i in range(n)))
    prob = cp.Problem(objective, constraints)

    if verbose:
        print(f"Structural SDP for n={n}:")
        print(f"  Matrix size: {dim}x{dim}")
        print(f"  Constraints: {len(constraints)}")

    try:
        result = prob.solve(solver=cp.SCS, max_iters=100000, verbose=False,
                          eps=1e-8)
        if verbose:
            print(f"  Status: {prob.status}")
            if result is not None:
                print(f"  Upper bound: {result:.6f}")
                if M.value is not None:
                    radii = [M.value[0, ri(i)] for i in range(n)]
                    print(f"  Radii (top 5): {sorted(radii, reverse=True)[:5]}")
        return result
    except Exception as e:
        if verbose:
            print(f"  Error: {e}")
        return None


if __name__ == "__main__":
    if len(sys.argv) > 1:
        n_values = [int(x) for x in sys.argv[1:]]
    else:
        n_values = [1, 2, 3, 4, 5, 10]

    known_best = {
        1: 0.5000, 2: 0.5858, 3: 0.7645, 4: 1.0000, 5: 1.0854,
        10: 1.5911, 15: 2.0365, 20: 2.3010, 26: 2.6360,
    }

    for n_val in n_values:
        result = sparse_order2_bound(n_val, verbose=True)
        area = np.sqrt(n_val / np.pi)
        best = min(result, area) if result is not None else area
        print(f"\n  Area bound:     {area:.6f}")
        print(f"  Best bound:     {best:.6f}")
        if n_val in known_best:
            gap = best - known_best[n_val]
            print(f"  Known best:     {known_best[n_val]:.6f}")
            print(f"  Gap:            {gap:.6f} ({100*gap/known_best[n_val]:.2f}%)")
        print()
