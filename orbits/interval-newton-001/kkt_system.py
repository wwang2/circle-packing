"""The KKT system F(z) = 0 for n=26 circle packing at the contact graph
identified in kkt_point.json.

z = concatenate([x (n), y (n), r (n), lam (m)])    shape = 3n + m
with n=26, m=78. Total dim = 156.

Sign convention
---------------
We write the Lagrangian with g_k ≤ 0 (infeasible) form:

    f(r) = sum_i r_i                        (maximize)
    g_ij(x,y,r) = (r_i+r_j)^2 - (x_i-x_j)^2 - (y_i-y_j)^2   ≤ 0  (disk-disk)
    g^L_i = r_i - x_i                       ≤ 0               (left wall)
    g^R_i = r_i + x_i - 1                   ≤ 0               (right wall)
    g^B_i = r_i - y_i                       ≤ 0               (bottom)
    g^T_i = r_i + y_i - 1                   ≤ 0               (top)

Lagrangian for MAXIMIZING f subject to g_k ≤ 0:
    L = f - sum_k lam_k g_k,   lam_k ≥ 0

Stationarity:  nabla f - sum_k lam_k nabla g_k = 0
Primal feas :  g_k = 0  at active k  (we only track active constraints).

This is the dual of what `recover_kkt.py` used. Here, since g has flipped
sign, the multipliers should now be NON-NEGATIVE. Let's verify when we
evaluate at (x*, -lam_from_stage1).

=== F(z) dimensions ===
  F[0 : n]        stationarity in x:  -sum_k lam_k dg_k/dx_i = 0
  F[n : 2n]       stationarity in y:  same, for y_i
  F[2n : 3n]      stationarity in r:  1 - sum_k lam_k dg_k/dr_i = 0
  F[3n : 3n+m]    g_k(x,y,r) = 0    (active constraint residuals)

=== Gradient entries for g ===
  disk-disk contact (i,j):
    g_ij = (r_i+r_j)^2 - (x_i-x_j)^2 - (y_i-y_j)^2
    dg/dx_i = -2(x_i - x_j)     dg/dx_j = +2(x_i - x_j)
    dg/dy_i = -2(y_i - y_j)     dg/dy_j = +2(y_i - y_j)
    dg/dr_i = +2(r_i + r_j)     dg/dr_j = +2(r_i + r_j)

  left wall on i:  g = r_i - x_i
    dg/dx_i = -1,   dg/dr_i = +1
  right wall:      g = r_i + x_i - 1
    dg/dx_i = +1,   dg/dr_i = +1
  bottom:          g = r_i - y_i
    dg/dy_i = -1,   dg/dr_i = +1
  top:             g = r_i + y_i - 1
    dg/dy_i = +1,   dg/dr_i = +1

=== Hessian entries for g (for J_zz part) ===
  Disk-disk Hessian has nonzero second derivatives:
    d^2 g/d x_i^2  = -2,   d^2 g / d x_j^2 = -2,  d^2 g / dx_i dx_j = +2
    similarly for y
    d^2 g/d r_i^2  = +2,   d^2 g / d r_j^2 = +2,  d^2 g / dr_i dr_j = +2
  Walls have zero Hessian (linear g).
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class Contact:
    kind: str           # 'dd' | 'L' | 'R' | 'B' | 'T'
    i: int
    j: int = -1         # only used for 'dd'

    def grad_entries(self, n: int) -> list[tuple[int, int]]:
        """Return list of (row_index_in_z[0:3n], sign_coeff_factor_type).
        coeff_factor_type is just a marker. We compute the actual coefficients
        in evaluate-time functions below."""
        raise NotImplementedError  # not actually needed — we use direct eval


@dataclass
class KKTSystem:
    n: int
    contacts: list[Contact]           # length m
    # useful indices
    @property
    def m(self) -> int:
        return len(self.contacts)
    @property
    def n_primal(self) -> int:
        return 3 * self.n
    @property
    def n_dual(self) -> int:
        return self.m
    @property
    def dim(self) -> int:
        return self.n_primal + self.n_dual

    # --- parse the kkt_point.json contact lists ---
    @staticmethod
    def from_point_file(path: pathlib.Path) -> "KKTSystem":
        data = json.loads(path.read_text())
        n = int(data["n"])
        contacts: list[Contact] = []
        for i, j in data["contacts_dd"]:
            contacts.append(Contact(kind="dd", i=int(i), j=int(j)))
        for i, s in data["contacts_wall"]:
            contacts.append(Contact(kind=s, i=int(i)))
        return KKTSystem(n=n, contacts=contacts)

    # --- slicing ---
    def slice_x(self, z): return z[0: self.n]
    def slice_y(self, z): return z[self.n: 2*self.n]
    def slice_r(self, z): return z[2*self.n: 3*self.n]
    def slice_l(self, z): return z[3*self.n: 3*self.n + self.m]

    # --- F(z) --- works for any ring: float, mpmath.iv, etc. ---
    def F(self, z, *, zero, one) -> list:
        """Return F(z) as a list of length 3n+m.

        `zero` and `one` are additive/multiplicative identities in the ring —
        this lets us use the same code with floats (zero=0.0, one=1.0) or
        mpmath.iv (zero=iv.mpf(0), one=iv.mpf(1)).
        """
        n = self.n
        m = self.m
        x = [z[i] for i in range(n)]
        y = [z[n + i] for i in range(n)]
        r = [z[2*n + i] for i in range(n)]
        lam = [z[3*n + k] for k in range(m)]

        Fstat_x = [zero] * n        # rows 0..n-1
        Fstat_y = [zero] * n        # rows n..2n-1
        Fstat_r = [one] * n         # rows 2n..3n-1, start with +1 (from df/dr_i)
        Fcon    = [zero] * m        # rows 3n..3n+m-1

        for k, c in enumerate(self.contacts):
            lk = lam[k]
            if c.kind == "dd":
                i, j = c.i, c.j
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dr = r[i] + r[j]
                # g = dr^2 - dx^2 - dy^2
                g = dr*dr - dx*dx - dy*dy
                Fcon[k] = g
                # gradients
                #   dg/dx_i = -2 dx,  dg/dx_j = +2 dx
                #   dg/dy_i = -2 dy,  dg/dy_j = +2 dy
                #   dg/dr_i = +2 dr,  dg/dr_j = +2 dr
                tx = lk * dx         # coeff; we use 2 * lk * dx below
                ty = lk * dy
                trr = lk * dr
                # - sum lam_k dg_k/dz_p  =>
                #   stat_x[i] -= lk * (-2 dx) =  +2 lk dx
                #   stat_x[j] -= lk * (+2 dx) =  -2 lk dx
                Fstat_x[i] = Fstat_x[i] + (tx + tx)
                Fstat_x[j] = Fstat_x[j] - (tx + tx)
                Fstat_y[i] = Fstat_y[i] + (ty + ty)
                Fstat_y[j] = Fstat_y[j] - (ty + ty)
                # stat_r[i] -= lk * (+2 dr) = -2 lk dr
                Fstat_r[i] = Fstat_r[i] - (trr + trr)
                Fstat_r[j] = Fstat_r[j] - (trr + trr)
            elif c.kind == "L":   # g = r_i - x_i
                i = c.i
                Fcon[k] = r[i] - x[i]
                # dg/dx_i = -1,  dg/dr_i = +1
                # stat_x[i] -= lk * (-1) = +lk
                # stat_r[i] -= lk * (+1) = -lk
                Fstat_x[i] = Fstat_x[i] + lk
                Fstat_r[i] = Fstat_r[i] - lk
            elif c.kind == "R":   # g = r_i + x_i - 1
                i = c.i
                Fcon[k] = r[i] + x[i] - one
                # dg/dx_i = +1, dg/dr_i = +1
                Fstat_x[i] = Fstat_x[i] - lk
                Fstat_r[i] = Fstat_r[i] - lk
            elif c.kind == "B":   # g = r_i - y_i
                i = c.i
                Fcon[k] = r[i] - y[i]
                Fstat_y[i] = Fstat_y[i] + lk
                Fstat_r[i] = Fstat_r[i] - lk
            elif c.kind == "T":   # g = r_i + y_i - 1
                i = c.i
                Fcon[k] = r[i] + y[i] - one
                Fstat_y[i] = Fstat_y[i] - lk
                Fstat_r[i] = Fstat_r[i] - lk
            else:
                raise ValueError(c.kind)

        return Fstat_x + Fstat_y + Fstat_r + Fcon

    # --- Jacobian: returns dense (dim, dim) matrix, entries are ring elements ---
    def jacobian(self, z, *, zero, one):
        """J[p, q] = dF_p/dz_q.

        Block layout:
          J[0:3n, 0:3n]       = H(lam)           (Hessian of Lagrangian wrt primal)
          J[0:3n, 3n:3n+m]    = -A(x,y,r)^T      (A is Jacobian of active g's)
          J[3n:3n+m, 0:3n]    = +A(x,y,r)
          J[3n:3n+m, 3n:3n+m] = 0
        """
        n = self.n
        m = self.m
        dim = 3*n + m
        # Build row-major list of dicts for sparse fill, then densify.
        J = [[zero]*dim for _ in range(dim)]

        x = [z[i] for i in range(n)]
        y = [z[n + i] for i in range(n)]
        r = [z[2*n + i] for i in range(n)]
        lam = [z[3*n + k] for k in range(m)]

        # -- Hessian block (upper-left, 3n x 3n) --
        # Contribution from each disk-disk contact: -lam * Hess(g).
        # Hess(g_ij) has structure:
        #   dx^2 terms: -2 on (x_i,x_i), -2 on (x_j,x_j), +2 on (x_i,x_j) & (x_j,x_i)
        #   dy^2 same on y block
        #   (r_i+r_j)^2 terms: +2 on (r_i,r_i), +2 on (r_j,r_j), +2 on (r_i,r_j) & (r_j,r_i)
        # So -lam * Hess yields (with h = +2 lam):
        #   J[x_i,x_i] += +h, J[x_j,x_j] += +h, J[x_i,x_j] += -h, J[x_j,x_i] += -h
        #   J[y_i,y_i] += +h, J[y_j,y_j] += +h, J[y_i,y_j] += -h, J[y_j,y_i] += -h
        #   J[r_i,r_i] += -h, J[r_j,r_j] += -h, J[r_i,r_j] += -h, J[r_j,r_i] += -h
        for k, c in enumerate(self.contacts):
            if c.kind != "dd":
                continue
            lk = lam[k]
            i, j = c.i, c.j
            h = lk + lk   # h = 2 lk
            # x block
            J[0*n+i][0*n+i] = J[0*n+i][0*n+i] + h
            J[0*n+j][0*n+j] = J[0*n+j][0*n+j] + h
            J[0*n+i][0*n+j] = J[0*n+i][0*n+j] - h
            J[0*n+j][0*n+i] = J[0*n+j][0*n+i] - h
            # y block
            J[1*n+i][1*n+i] = J[1*n+i][1*n+i] + h
            J[1*n+j][1*n+j] = J[1*n+j][1*n+j] + h
            J[1*n+i][1*n+j] = J[1*n+i][1*n+j] - h
            J[1*n+j][1*n+i] = J[1*n+j][1*n+i] - h
            # r block
            J[2*n+i][2*n+i] = J[2*n+i][2*n+i] - h
            J[2*n+j][2*n+j] = J[2*n+j][2*n+j] - h
            J[2*n+i][2*n+j] = J[2*n+i][2*n+j] - h
            J[2*n+j][2*n+i] = J[2*n+j][2*n+i] - h
        # Walls have zero Hessian.

        # -- A block: for each constraint k, fill row/col with dg_k/dz_primal --
        # J_lower_left  [3n+k, q] =  +dg_k/dz_q
        # J_upper_right [q, 3n+k] =  -dg_k/dz_q
        for k, c in enumerate(self.contacts):
            row_c = 3*n + k
            def put(q: int, v):
                J[row_c][q] = J[row_c][q] + v
                J[q][row_c] = J[q][row_c] - v
            if c.kind == "dd":
                i, j = c.i, c.j
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dr = r[i] + r[j]
                two = one + one
                # dg/dx_i = -2 dx, dg/dx_j = +2 dx
                put(0*n+i, -(two*dx))
                put(0*n+j, +(two*dx))
                put(1*n+i, -(two*dy))
                put(1*n+j, +(two*dy))
                put(2*n+i, +(two*dr))
                put(2*n+j, +(two*dr))
            elif c.kind == "L":
                i = c.i
                put(0*n+i, -one)
                put(2*n+i, +one)
            elif c.kind == "R":
                i = c.i
                put(0*n+i, +one)
                put(2*n+i, +one)
            elif c.kind == "B":
                i = c.i
                put(1*n+i, -one)
                put(2*n+i, +one)
            elif c.kind == "T":
                i = c.i
                put(1*n+i, +one)
                put(2*n+i, +one)

        return J


# --- Smoke test / float-ring sanity check ---

def _main():
    HERE = pathlib.Path(__file__).resolve().parent
    sys = KKTSystem.from_point_file(HERE / "kkt_point.json")
    data = json.loads((HERE / "kkt_point.json").read_text())
    n = sys.n
    m = sys.m
    print(f"n={n}, m={m}, dim={sys.dim}")

    x = np.array(data["x"])
    y = np.array(data["y"])
    r = np.array(data["r"])
    # Flip sign: stage1 used L=f-sum_k mu_k g_k with g's that had the *opposite*
    # sign convention from our kkt_system.py. So the multipliers from stage1
    # need a sign flip to match this file's convention.
    lam_dd = -np.array(data["lambda_dd"])
    lam_w  = -np.array(data["lambda_wall"])
    lam = np.concatenate([lam_dd, lam_w])

    z = np.concatenate([x, y, r, lam]).tolist()

    F = sys.F(z, zero=0.0, one=1.0)
    F_arr = np.array(F, dtype=float)
    stat_x = F_arr[0:n]
    stat_y = F_arr[n:2*n]
    stat_r = F_arr[2*n:3*n]
    con    = F_arr[3*n:3*n+m]
    print(f"||F_stat_x||_inf = {np.linalg.norm(stat_x, np.inf):.3e}")
    print(f"||F_stat_y||_inf = {np.linalg.norm(stat_y, np.inf):.3e}")
    print(f"||F_stat_r||_inf = {np.linalg.norm(stat_r, np.inf):.3e}")
    print(f"||F_con||_inf    = {np.linalg.norm(con, np.inf):.3e}")
    print(f"||F||_inf total = {np.linalg.norm(F_arr, np.inf):.3e}")
    print(f"lambdas: min={lam.min():.3e} max={lam.max():.3e} (should be ≥ 0)")

    J = sys.jacobian(z, zero=0.0, one=1.0)
    J_arr = np.array([[float(c) for c in row] for row in J], dtype=float)
    print(f"J shape: {J_arr.shape}")
    print(f"rank(J) = {np.linalg.matrix_rank(J_arr)}")
    # condition number
    sv = np.linalg.svd(J_arr, compute_uv=False)
    print(f"||J|| = {sv[0]:.3e}, smin(J) = {sv[-1]:.3e}, cond = {sv[0]/sv[-1]:.3e}")


if __name__ == "__main__":
    _main()
