"""
Shor (Lasserre level-1) SDP relaxation of the circle packing QCQP.

Variables: (x_i, y_i, r_i) for i=1..26, 78 variables total.

Maximize sum_i r_i subject to:
  Containment (linear):
    r_i >= 0
    x_i - r_i >= 0      <=>  x_i >= r_i
    1 - x_i - r_i >= 0  <=>  x_i + r_i <= 1
    y_i - r_i >= 0
    1 - y_i - r_i >= 0

  Non-overlap (quadratic non-convex):
    (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0

Shor lifting:
  Let z = (1, v1, ..., v78)^T  (v = [x_1,y_1,r_1, ..., x_n,y_n,r_n])
  Moment matrix M = z z^T, size 79x79, M[0,0] = 1
  Non-overlap constraint is linear in M because it is a quadratic in v.

We maximize sum_i M[0, 3i+3] (i.e. the 'r_i' entries in the first row),
subject to:
  M >= 0 (PSD)
  M[0,0] = 1
  Linear constraints on M[0, k] for containment
  The non-overlap quadratic:
    (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i+r_j)^2
    = M[ix,ix] - 2 M[ix,jx] + M[jx,jx]
    + M[iy,iy] - 2 M[iy,jy] + M[jy,jy]
    - M[ir,ir] - 2 M[ir,jr] - M[jr,jr]  >= 0
  (all indices shifted by +1 due to the leading 1)

  Box constraints: 0 <= x_i, y_i <= 1, 0 <= r_i <= 0.5 (tightens diagonal)
    M[ix, ix] >= M[0, ix]^? -- we add M[ix, ix] <= 1, and x*(1-x) >= 0 type
    (We'll add: M[kk] - M[0,k] <= 0 for variables in [0,1].)
    Also McCormick/RLT: (1-x_i)*x_i >= 0, etc.

This is standard Shor relaxation. It is known to be loose for packing problems
but gives a rigorous UB and serves as the level-1 baseline before we add
sparse level-2 clique boosts.
"""
import json
import time
from pathlib import Path
import numpy as np
import cvxpy as cp

HERE = Path(__file__).parent
CONTACTS = HERE / "contacts.json"
OUT = HERE / "shor_report.json"

N = 26
NVAR = 3 * N  # 78

def var_index(i, coord):
    """coord in {'x','y','r'}"""
    return 3 * i + {"x": 0, "y": 1, "r": 2}[coord]


def main():
    data = json.loads(CONTACTS.read_text())
    n = data["n"]
    assert n == N

    # 79x79 moment matrix
    dim = NVAR + 1  # entry 0 is 1, entries 1..78 are variables
    M = cp.Variable((dim, dim), symmetric=True)
    constraints = [M >> 0, M[0, 0] == 1]

    def vi(i, c):
        return var_index(i, c) + 1  # +1 for leading constant

    # ------- containment (linear on first row) -------
    for i in range(n):
        xi = vi(i, "x"); yi = vi(i, "y"); ri = vi(i, "r")
        constraints += [
            M[0, ri] >= 0,          # r >= 0
            M[0, xi] - M[0, ri] >= 0,        # x >= r
            1 - M[0, xi] - M[0, ri] >= 0,    # x + r <= 1
            M[0, yi] - M[0, ri] >= 0,
            1 - M[0, yi] - M[0, ri] >= 0,
            M[0, ri] <= 0.5,
        ]

    # ------- non-overlap (quadratic) -------
    # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    # = M[xi,xi] - 2 M[xi,xj] + M[xj,xj] + (same for y) - M[ri,ri] - 2 M[ri,rj] - M[rj,rj]  >= 0
    for i in range(n):
        for j in range(i + 1, n):
            xi = vi(i, "x"); xj = vi(j, "x")
            yi = vi(i, "y"); yj = vi(j, "y")
            ri = vi(i, "r"); rj = vi(j, "r")
            lhs = (M[xi, xi] - 2 * M[xi, xj] + M[xj, xj]
                   + M[yi, yi] - 2 * M[yi, yj] + M[yj, yj]
                   - M[ri, ri] - 2 * M[ri, rj] - M[rj, rj])
            constraints.append(lhs >= 0)

    # ------- RLT / McCormick box tightenings -------
    # x_i in [0,1] => x_i^2 <= x_i, i.e. M[xi,xi] <= M[0,xi]
    # Also M[xi,xi] >= M[0,xi]^2 (implied by PSD)
    # r_i in [0, 0.5]   => r_i * (0.5 - r_i) >= 0 => M[ri,ri] <= 0.5 * M[0,ri]
    for i in range(n):
        xi = vi(i, "x"); yi = vi(i, "y"); ri = vi(i, "r")
        constraints += [
            M[xi, xi] <= M[0, xi],
            M[yi, yi] <= M[0, yi],
            M[ri, ri] <= 0.5 * M[0, ri],
        ]
        # RLT from x - r >= 0 and 1 - x - r >= 0:
        # (x - r)*(x + r) = x^2 - r^2 >= 0  (in [0,1]^2 with r >= 0)
        constraints.append(M[xi, xi] - M[ri, ri] >= 0)
        constraints.append(M[yi, yi] - M[ri, ri] >= 0)
        # r * (x - r) >= 0 => M[ri,xi] - M[ri,ri] >= 0
        constraints.append(M[ri, xi] - M[ri, ri] >= 0)
        constraints.append(M[ri, yi] - M[ri, ri] >= 0)
        # r * (1 - x - r) >= 0 => M[0,ri] - M[ri,xi] - M[ri,ri] >= 0
        constraints.append(M[0, ri] - M[ri, xi] - M[ri, ri] >= 0)
        constraints.append(M[0, ri] - M[ri, yi] - M[ri, ri] >= 0)
        # x * (1 - x) >= 0  (already implied by xi^2 <= xi)
        # r * r <= r * 0.5 already added

    # ------- objective -------
    obj = cp.Maximize(sum(M[0, vi(i, "r")] for i in range(n)))
    prob = cp.Problem(obj, constraints)

    t0 = time.time()
    print(f"Solving Shor SDP: {dim}x{dim} moment matrix, {len(constraints)} constraints...")
    prob.solve(solver=cp.CLARABEL, verbose=False)
    t1 = time.time()

    status = prob.status
    ub = float(prob.value) if prob.value is not None else None
    wall = t1 - t0

    print(f"Status: {status}")
    print(f"UB (Shor L1): {ub}")
    print(f"Wall: {wall:.2f}s")
    print(f"Parent UB: 2.7396  Parent primal: 2.6359830865")
    if ub is not None:
        gap_old = (2.7396 - 2.6359830865) / 2.6359830865 * 100
        gap_new = (ub - 2.6359830865) / 2.6359830865 * 100
        print(f"Gap old: {gap_old:.2f}%   Gap new: {gap_new:.2f}%")

    report = {
        "method": "shor_level1_with_RLT",
        "lasserre_level": 1,
        "moment_matrix_size": dim,
        "num_constraints": len(constraints),
        "solver": "CLARABEL",
        "status": status,
        "tightened_ub": ub,
        "parent_ub": 2.7396,
        "parent_primal": 2.6359830865,
        "wall_clock_seconds": wall,
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
