"""Stage 1: Recover (x*, lambda*) and the contact graph from the parent orbit's
n=26 numerical optimum.

Reads: orbits/mobius-001/solution_n26.json (26 circles at sum_r = 2.6359830865)
Writes: orbits/interval-newton-001/kkt_point.json
   {
     "n": 26,
     "x": [x1,..,xn,  y1,..,yn,  r1,..,rn],          # length 78
     "contacts_dd": [[i,j], ...],                     # disk-disk active contacts
     "contacts_wall": [[i, side], ...],               # side in {"L","R","B","T"}
     "lambda_dd": [...],                              # one per disk-disk contact
     "lambda_wall": [...],                            # one per wall contact
     "sum_r": 2.6359830865,
     "kkt_residual_inf": ...
   }

The KKT system for max sum_r subject to
    g_ij  =  (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2  >= 0
    w_i^L =  x_i - r_i                                       >= 0
    w_i^R =  1 - x_i - r_i                                    >= 0
    w_i^B =  y_i - r_i                                        >= 0
    w_i^T =  1 - y_i - r_i                                    >= 0

Lagrangian L = sum_i r_i - sum_k mu_k * g_k    (mu_k >= 0)

Stationarity:
  d/dx_i: sum_{j: (i,j) active} mu_ij * (-2)(x_i - x_j)
          + sum_{j: (j,i) active} mu_ji * ( 2)(x_i - x_j)    [same with sign]
          - mu_i^L * (1) - mu_i^R * (-1)
        = 0
  d/dy_i: analogous
  d/dr_i: 1 - sum_{j active with i} mu_k * (-2)(r_i + r_j)
          - mu_i^L * (-1) - mu_i^R * (-1) - mu_i^B * (-1) - mu_i^T * (-1)
        = 0

Simplified (moving terms): at an active contact (i,j),
    dg_ij/dx_i = 2 (x_i - x_j),   dg_ij/dx_j = -2 (x_i - x_j)
    dg_ij/dy_i = 2 (y_i - y_j),   dg_ij/dy_j = -2 (y_i - y_j)
    dg_ij/dr_i = -2 (r_i + r_j),  dg_ij/dr_j = -2 (r_i + r_j)

Walls:
    dw_i^L/dx_i = 1,   dw_i^L/dr_i = -1
    dw_i^R/dx_i = -1,  dw_i^R/dr_i = -1
    dw_i^B/dy_i = 1,   dw_i^B/dr_i = -1
    dw_i^T/dy_i = -1,  dw_i^T/dr_i = -1

KKT: nabla f - sum_k mu_k nabla g_k = 0
with f = sum r_i (so nabla f has 1 only in the r slots).
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path("/Users/wujiewang/code/circle-packing/.worktrees/interval-newton-001")
PARENT = ROOT / "orbits" / "mobius-001" / "solution_n26.json"
OUT = ROOT / "orbits" / "interval-newton-001" / "kkt_point.json"

# Tolerance for declaring a contact "active". We use a loose tolerance first
# (1e-7) to catch the active set, then verify residuals are actually small.
ACTIVE_TOL = 5e-7


def load_parent() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = json.loads(PARENT.read_text())
    arr = np.array(data["circles"], dtype=np.float64)
    x = arr[:, 0].copy()
    y = arr[:, 1].copy()
    r = arr[:, 2].copy()
    return x, y, r


def identify_contacts(x, y, r, tol=ACTIVE_TOL):
    n = len(x)
    dd = []  # list of (i, j, slack)  with i<j
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            slack = (dx * dx + dy * dy) - (r[i] + r[j]) ** 2  # >= 0
            if slack < tol:
                dd.append((i, j, slack))
    walls = []  # (i, side, slack)
    for i in range(n):
        for side, slack in (
            ("L", x[i] - r[i]),
            ("R", 1.0 - x[i] - r[i]),
            ("B", y[i] - r[i]),
            ("T", 1.0 - y[i] - r[i]),
        ):
            if slack < tol:
                walls.append((i, side, slack))
    return dd, walls


def build_kkt_jacobian(x, y, r, contacts_dd, contacts_wall):
    """Return (M, b) of size (3n, m) and (3n,) such that  M @ mu = b  ==> KKT.

    M[row, k] = dg_k / dz[row], where z = [x(n), y(n), r(n)]
    b[row]    = df / dz[row] (1 at r slots, 0 elsewhere)

    Then stationarity is:  b - M @ mu = 0,  i.e. M mu = b.
    """
    n = len(x)
    m_dd = len(contacts_dd)
    m_w = len(contacts_wall)
    m = m_dd + m_w
    M = np.zeros((3 * n, m), dtype=np.float64)
    b = np.zeros(3 * n, dtype=np.float64)
    b[2 * n : 3 * n] = 1.0  # derivative of sum r

    for k, (i, j, _) in enumerate(contacts_dd):
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        dr = r[i] + r[j]
        # partials of g_ij = dx^2 + dy^2 - dr^2
        M[0 * n + i, k] = 2.0 * dx
        M[0 * n + j, k] = -2.0 * dx
        M[1 * n + i, k] = 2.0 * dy
        M[1 * n + j, k] = -2.0 * dy
        M[2 * n + i, k] = -2.0 * dr
        M[2 * n + j, k] = -2.0 * dr

    for k, (i, side, _) in enumerate(contacts_wall):
        col = m_dd + k
        if side == "L":  # g = x_i - r_i
            M[0 * n + i, col] = 1.0
            M[2 * n + i, col] = -1.0
        elif side == "R":  # g = 1 - x_i - r_i
            M[0 * n + i, col] = -1.0
            M[2 * n + i, col] = -1.0
        elif side == "B":  # g = y_i - r_i
            M[1 * n + i, col] = 1.0
            M[2 * n + i, col] = -1.0
        elif side == "T":  # g = 1 - y_i - r_i
            M[1 * n + i, col] = -1.0
            M[2 * n + i, col] = -1.0
    return M, b


def main():
    x, y, r = load_parent()
    n = len(x)
    print(f"Loaded n={n} circles, sum_r = {r.sum():.12f}")

    dd, walls = identify_contacts(x, y, r, tol=ACTIVE_TOL)
    print(f"Active disk-disk contacts: {len(dd)}")
    print(f"Active wall contacts: {len(walls)}")
    print(f"Total contacts: {len(dd) + len(walls)}")

    slacks_dd = [s for _, _, s in dd]
    slacks_w = [s for _, _, s in walls]
    if slacks_dd:
        print(f"  disk-disk slack range: [{min(slacks_dd):.3e}, {max(slacks_dd):.3e}]")
    if slacks_w:
        print(f"  wall slack range    : [{min(slacks_w):.3e}, {max(slacks_w):.3e}]")

    M, b = build_kkt_jacobian(x, y, r, dd, walls)
    print(f"KKT Jacobian M shape = {M.shape}, b shape = {b.shape}")
    print(f"rank(M) = {np.linalg.matrix_rank(M)}")

    # Solve M mu = b in least squares to get the dual.
    mu, residuals, rank, sv = np.linalg.lstsq(M, b, rcond=None)
    kkt_residual = M @ mu - b
    print(f"lstsq rank = {rank}, min sing = {sv[-1]:.3e}, max sing = {sv[0]:.3e}")
    print(f"||M mu - b||_inf = {np.linalg.norm(kkt_residual, np.inf):.3e}")
    print(f"mu range: [{mu.min():.3e}, {mu.max():.3e}]  (nonneg required)")
    n_neg = int((mu < -1e-9).sum())
    if n_neg:
        print(f"  WARNING: {n_neg} multipliers are negative (> 1e-9 in magnitude)")
        # Show the worst offenders
        idx = np.argsort(mu)[:5]
        for k in idx:
            if k < len(dd):
                print(f"    dd  k={k}: i,j={dd[k][:2]} mu={mu[k]:.3e}")
            else:
                kk = k - len(dd)
                print(f"    wall k={k}: i,side={walls[kk][:2]} mu={mu[k]:.3e}")

    # If 78 constraints and rank is full (78), also solve the square system
    m = M.shape[1]
    if m == 3 * n and rank == 3 * n:
        mu_exact = np.linalg.solve(M, b)
        res_exact = np.linalg.norm(M @ mu_exact - b, np.inf)
        print(f"exact solve: ||res||_inf = {res_exact:.3e}")
        print(f"mu_exact range: [{mu_exact.min():.3e}, {mu_exact.max():.3e}]")
        mu = mu_exact

    # Save
    contacts_dd = [[int(i), int(j)] for i, j, _ in dd]
    contacts_wall = [[int(i), s] for i, s, _ in walls]
    m_dd = len(contacts_dd)
    lambda_dd = mu[:m_dd].tolist()
    lambda_wall = mu[m_dd:].tolist()

    OUT.write_text(
        json.dumps(
            {
                "n": int(n),
                "x": x.tolist(),
                "y": y.tolist(),
                "r": r.tolist(),
                "sum_r": float(r.sum()),
                "contacts_dd": contacts_dd,
                "contacts_wall": contacts_wall,
                "lambda_dd": lambda_dd,
                "lambda_wall": lambda_wall,
                "n_primal": 3 * int(n),
                "n_dual": int(m),
                "n_eqns": 3 * int(n) + int(m),
                "stationarity_residual_inf": float(
                    np.linalg.norm(M @ mu - b, np.inf)
                ),
                "active_tol": ACTIVE_TOL,
            },
            indent=2,
        )
    )
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
