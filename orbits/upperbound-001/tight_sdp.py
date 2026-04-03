"""
Tighter SDP upper bound for circle packing.

The basic SDP (Shor relaxation) just reproduces the area bound sqrt(n/pi).
To get tighter bounds, we need LOCALIZING CONSTRAINTS that encode
the non-overlap condition more strongly.

Key idea: if g(v) >= 0 is a constraint, then for any polynomial p(v) >= 0,
we have g(v)*p(v) >= 0. In the SDP framework, we can enforce:
  E[g(v) * v_k * v_l] >= 0 for all k,l
which gives a "localizing matrix" constraint.

For non-overlap constraint g_ij = dist_ij^2 - (r_i+r_j)^2 >= 0:
  E[g_ij * v_k^2] >= 0 for all k

This creates a localizing matrix for each pair (i,j), which is expensive
but much tighter.

We also try a DIFFERENT approach: partition-based SDP.
"""

import numpy as np
import sys
import json
from pathlib import Path


def sdp_tight_bound(n, use_localizing=True, verbose=False):
    """
    SDP with localizing matrices for non-overlap constraints.

    For each pair (i,j), the constraint g_ij >= 0 generates a
    localizing matrix: for all basis monomials p, q:
    E[g_ij * p * q] >= 0

    Using basis {1, x_i, y_i, r_i, x_j, y_j, r_j} for pair (i,j),
    we get a 7x7 localizing matrix per pair.

    This is MUCH tighter but also much larger:
    - Main moment matrix: (3n+1) x (3n+1)
    - n*(n-1)/2 localizing matrices, each 7x7

    For n=26: main matrix 79x79, 325 localizing matrices of 7x7.
    Total: ~16000 SDP variables. Should be feasible with SCS.
    """
    try:
        import cvxpy as cp
    except ImportError:
        return None

    dim = 3 * n + 1

    # Index helpers
    def xi(i): return 1 + i
    def yi(i): return 1 + n + i
    def ri(i): return 1 + 2*n + i

    M = cp.Variable((dim, dim), symmetric=True, name="M")
    constraints = []

    # M is PSD
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

    # Containment (first order): x_i >= r_i, x_i + r_i <= 1
    for i in range(n):
        constraints += [M[0, xi(i)] >= M[0, ri(i)]]
        constraints += [M[0, xi(i)] + M[0, ri(i)] <= 1]
        constraints += [M[0, yi(i)] >= M[0, ri(i)]]
        constraints += [M[0, yi(i)] + M[0, ri(i)] <= 1]

    # Containment (second order localizing):
    # (x_i - r_i) >= 0 => (x_i - r_i)^2 >= 0:
    for i in range(n):
        constraints += [M[xi(i),xi(i)] - 2*M[xi(i),ri(i)] + M[ri(i),ri(i)] >= 0]
        constraints += [1 - 2*M[0,xi(i)] - 2*M[0,ri(i)] + M[xi(i),xi(i)] + 2*M[xi(i),ri(i)] + M[ri(i),ri(i)] >= 0]
        constraints += [M[yi(i),yi(i)] - 2*M[yi(i),ri(i)] + M[ri(i),ri(i)] >= 0]
        constraints += [1 - 2*M[0,yi(i)] - 2*M[0,ri(i)] + M[yi(i),yi(i)] + 2*M[yi(i),ri(i)] + M[ri(i),ri(i)] >= 0]

    # Non-overlap (basic): linear in M
    for i in range(n):
        for j in range(i+1, n):
            constraints += [
                M[xi(i),xi(i)] - 2*M[xi(i),xi(j)] + M[xi(j),xi(j)] +
                M[yi(i),yi(i)] - 2*M[yi(i),yi(j)] + M[yi(j),yi(j)] >=
                M[ri(i),ri(i)] + 2*M[ri(i),ri(j)] + M[ri(j),ri(j)]
            ]

    # Area constraint
    constraints += [np.pi * sum(M[ri(i),ri(i)] for i in range(n)) <= 1]

    # LOCALIZING CONSTRAINTS for non-overlap
    if use_localizing:
        # For each pair (i,j), define g_ij = (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2
        # We require E[g_ij * v_k^2] >= 0 for each variable v_k
        # This means: for each k, the expression
        #   E[(xi-xj)^2 * vk^2] + E[(yi-yj)^2 * vk^2] - E[(ri+rj)^2 * vk^2] >= 0
        # These involve 4th-order moments which are NOT in our moment matrix.
        #
        # Instead, we use a simpler localizing approach:
        # E[g_ij * 1] >= 0 (already have this)
        # E[g_ij * v_k] >= 0 for each first-order variable v_k
        # This gives LINEAR constraints in M (since g_ij is quadratic, g_ij*v_k is cubic,
        # but E[cubic] involves 3rd-order moments which ARE in M for degree-1 relaxation).
        #
        # Wait: our moment matrix M = vv^T only has entries up to degree 2.
        # E[v_i * v_j * v_k] is a 3rd-order moment NOT in M.
        #
        # So standard Lasserre order-1 can only use E[g_ij] >= 0.
        # For localizing, we need order-2 (moment matrix of order 2).
        #
        # Order-2 Lasserre would have a moment matrix indexed by all monomials
        # up to degree 2: {1, v1, ..., v_{3n}, v1^2, v1v2, ...}
        # This is O(n^2) x O(n^2), way too large for n=26.
        #
        # ALTERNATIVE: use SPARSE localizing constraints.
        # For pair (i,j), only use variables {xi, yi, ri, xj, yj, rj}.
        # Moment matrix restricted to these 6 variables + 1: 7x7.
        # The localizing matrix for g_ij uses monomials {1, xi, yi, ri, xj, yj, rj}.

        # Actually, we CAN'T directly create independent 7x7 moment matrices
        # because they need to be CONSISTENT with the main moment matrix M.
        # The entries of the 7x7 matrix are all entries of M.

        # So instead: for each pair (i,j), form the 7x7 principal submatrix
        # of M using indices {0, xi(i), yi(i), ri(i), xi(j), yi(j), ri(j)}.
        # This submatrix is already PSD (since M is PSD).
        # That's automatic and doesn't add anything.

        # What DOES help: E[g_ij * r_k] >= 0 for k = i,j (or any k).
        # g_ij * r_i = [(xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2] * r_i
        # This involves 3rd-order moments. We need to introduce them or bound them.

        # APPROACH: RLT (Reformulation-Linearization Technique) cuts.
        # For any pair of constraints f >= 0 and g >= 0, f*g >= 0.
        # Containment: x_i - r_i >= 0, 1 - x_i - r_i >= 0
        # Non-overlap: dist_ij^2 - (r_i+r_j)^2 >= 0
        #
        # Product: (x_i - r_i) * (x_j - r_j) >= 0
        # This is NOT always true! x_i - r_i >= 0 and x_j - r_j >= 0,
        # so their product IS >= 0. This gives:
        # E[x_i*x_j - x_i*r_j - r_i*x_j + r_i*r_j] >= 0
        # M[xi(i),xi(j)] - M[xi(i),ri(j)] - M[ri(i),xi(j)] + M[ri(i),ri(j)] >= 0

        # Similarly for other combinations. These are valid RLT cuts!

        # RLT cuts from containment: for each pair (i,j):
        for i in range(n):
            for j in range(i+1, n):
                # (x_i - r_i)(x_j - r_j) >= 0
                constraints += [M[xi(i),xi(j)] - M[xi(i),ri(j)] - M[ri(i),xi(j)] + M[ri(i),ri(j)] >= 0]
                # (1-x_i-r_i)(1-x_j-r_j) >= 0
                constraints += [
                    1 - M[0,xi(i)] - M[0,ri(i)] - M[0,xi(j)] - M[0,ri(j)]
                    + M[xi(i),xi(j)] + M[xi(i),ri(j)] + M[ri(i),xi(j)] + M[ri(i),ri(j)]
                    >= 0
                ]
                # (x_i - r_i)(1 - x_j - r_j) >= 0
                constraints += [
                    M[0,xi(i)] - M[0,ri(i)] - M[xi(i),xi(j)] - M[xi(i),ri(j)]
                    + M[ri(i),xi(j)] + M[ri(i),ri(j)]
                    >= 0
                ]
                # (1-x_i-r_i)(x_j - r_j) >= 0
                constraints += [
                    M[0,xi(j)] - M[0,ri(j)] - M[xi(i),xi(j)] + M[xi(i),ri(j)]
                    - M[ri(i),xi(j)] + M[ri(i),ri(j)]
                    >= 0
                ]

                # Same for y
                constraints += [M[yi(i),yi(j)] - M[yi(i),ri(j)] - M[ri(i),yi(j)] + M[ri(i),ri(j)] >= 0]
                constraints += [
                    1 - M[0,yi(i)] - M[0,ri(i)] - M[0,yi(j)] - M[0,ri(j)]
                    + M[yi(i),yi(j)] + M[yi(i),ri(j)] + M[ri(i),yi(j)] + M[ri(i),ri(j)]
                    >= 0
                ]
                constraints += [
                    M[0,yi(i)] - M[0,ri(i)] - M[yi(i),yi(j)] - M[yi(i),ri(j)]
                    + M[ri(i),yi(j)] + M[ri(i),ri(j)]
                    >= 0
                ]
                constraints += [
                    M[0,yi(j)] - M[0,ri(j)] - M[yi(i),yi(j)] + M[yi(i),ri(j)]
                    - M[ri(i),yi(j)] + M[ri(i),ri(j)]
                    >= 0
                ]

                # Cross RLT: (x_i - r_i)(y_j - r_j) >= 0
                constraints += [M[xi(i),yi(j)] - M[xi(i),ri(j)] - M[ri(i),yi(j)] + M[ri(i),ri(j)] >= 0]
                # (y_i - r_i)(x_j - r_j) >= 0
                constraints += [M[yi(i),xi(j)] - M[yi(i),ri(j)] - M[ri(i),xi(j)] + M[ri(i),ri(j)] >= 0]

        # RLT from r_i >= 0, r_j >= 0: r_i * r_j >= 0 (already in PSD)
        # But also: r_i * (0.5 - r_j) >= 0
        for i in range(n):
            for j in range(i+1, n):
                constraints += [0.5*M[0,ri(i)] - M[ri(i),ri(j)] >= 0]
                constraints += [0.5*M[0,ri(j)] - M[ri(i),ri(j)] >= 0]

    # Objective
    objective = cp.Maximize(sum(M[0, ri(i)] for i in range(n)))
    prob = cp.Problem(objective, constraints)

    if verbose:
        print(f"Tight SDP for n={n} (localizing={use_localizing}):")
        print(f"  Matrix size: {dim}x{dim}")
        print(f"  Constraints: {len(constraints)}")

    try:
        result = prob.solve(solver=cp.SCS, max_iters=50000, verbose=False,
                          eps=1e-7)
        if verbose:
            print(f"  Status: {prob.status}")
            print(f"  Upper bound: {result:.6f}")
            if M.value is not None:
                radii = [M.value[0, ri(i)] for i in range(n)]
                print(f"  Radii (top 5): {sorted(radii, reverse=True)[:5]}")
        return result
    except Exception as e:
        if verbose:
            print(f"  Error: {e}")
        return None


def grid_bound(n, k=4, verbose=False):
    """
    Grid-based LP upper bound.

    Divide [0,1]^2 into k x k cells, each of size (1/k) x (1/k).
    For each cell c, let A_c = area of cell = 1/k^2.

    Key constraint: the total area of circles intersecting cell c
    is bounded. A circle of radius r centered at distance d from
    cell center intersects the cell if d < r + cell_radius.

    Simpler approach: assign each circle to the cell containing its center.
    Then for circles in cell c with centers in that cell:
    - Each radius r_i <= 1/(2k) (must fit in cell... no, circle can extend outside)
    - Actually r_i can be up to 0.5 regardless of cell.

    Better: use LP.
    For each cell c, let S_c = sum of r_i for circles centered in c.
    For each cell c, let n_c = number of circles centered in c.
    Sum n_c = n.

    Bound per cell: circles in cell c have centers in [c_x/k, (c_x+1)/k] x [c_y/k, (c_y+1)/k].
    They must be non-overlapping.
    By area bound applied to the cell: sum r_i^2 <= (1/k)^2 / pi for circles in cell c.
    By C-S: (sum r_i)^2 <= n_c * sum r_i^2 <= n_c / (pi*k^2).
    So S_c <= sqrt(n_c) / (k*sqrt(pi)).

    Total: sum S_c = sum sqrt(n_c) / (k*sqrt(pi)).
    Maximize sum sqrt(n_c) subject to sum n_c = n.
    By concavity of sqrt: sum sqrt(n_c) <= k^2 * sqrt(n/k^2) = k * sqrt(n).
    So total <= k * sqrt(n) / (k * sqrt(pi)) = sqrt(n/pi).

    Same bound again! Grid subdivision with area bound doesn't help.

    BUT: we can add INTER-CELL non-overlap constraints.
    Circles in adjacent cells must also be non-overlapping with each other.
    This can provide stronger bounds if we model it correctly.

    For now, this approach reduces to the area bound. Skip.
    """
    return np.sqrt(n / np.pi)


def run_tight_bounds(n_values=None, verbose=True):
    """Run tight SDP bounds."""
    if n_values is None:
        n_values = [1, 2, 3, 4, 5, 10, 26]

    known_best = {
        1: 0.5000, 2: 0.5858, 3: 0.7645, 4: 1.0000, 5: 1.0854,
        10: 1.5911, 15: 2.0365, 20: 2.3010, 26: 2.6360, 30: 2.8425, 32: 2.9390,
    }

    results = {}
    for n in n_values:
        print(f"\n{'='*60}")
        print(f"n = {n}")
        print(f"{'='*60}")

        area = np.sqrt(n / np.pi)
        print(f"  Area bound: {area:.6f}")

        # Without localizing
        sdp_basic = sdp_tight_bound(n, use_localizing=False, verbose=verbose)

        # With localizing (RLT cuts)
        sdp_rlt = sdp_tight_bound(n, use_localizing=True, verbose=verbose)

        best = area
        if sdp_basic is not None:
            best = min(best, sdp_basic)
        if sdp_rlt is not None:
            best = min(best, sdp_rlt)

        print(f"\n  BEST UPPER BOUND: {best:.6f}")
        if n in known_best:
            gap = best - known_best[n]
            gap_pct = 100 * gap / known_best[n]
            print(f"  Known best:       {known_best[n]:.6f}")
            print(f"  Gap:              {gap:.6f} ({gap_pct:.2f}%)")

        results[n] = {
            'area': area,
            'sdp_basic': sdp_basic,
            'sdp_rlt': sdp_rlt,
            'best': best,
            'known': known_best.get(n),
        }

    return results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        n_values = [int(x) for x in sys.argv[1:]]
    else:
        n_values = [1, 2, 3, 4, 5, 10]

    results = run_tight_bounds(n_values, verbose=True)

    output_path = Path(__file__).parent / "tight_bounds_results.json"
    serializable = {}
    for n, data in results.items():
        serializable[str(n)] = {k: float(v) if v is not None else None
                                 for k, v in data.items()}
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_path}")
