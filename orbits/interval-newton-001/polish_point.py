"""Stage 2: Newton-polish the KKT point so F(z*) is at machine precision.

The parent's numerical point has constraint residuals ~1e-10 (the optimizer's
ftol). This matters for Krawczyk: we need the box *center* z0 to be nearly
exact so that F(z0) is ~1e-15, not ~1e-10. Otherwise the box radius must be
larger than necessary.

Approach: run Newton on F(z) = 0 in float64:
    z_{k+1} = z_k - J(z_k)^{-1} F(z_k)
for 10 iterations. Since the system is full-rank, this converges quadratically.

Output: kkt_point_polished.json (same schema as kkt_point.json, with the
refined z and reduced residual).
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

from kkt_system import KKTSystem

HERE = pathlib.Path(__file__).resolve().parent


def main():
    sys = KKTSystem.from_point_file(HERE / "kkt_point.json")
    data = json.loads((HERE / "kkt_point.json").read_text())

    n = sys.n
    m = sys.m

    x = np.array(data["x"])
    y = np.array(data["y"])
    r = np.array(data["r"])
    lam_dd = -np.array(data["lambda_dd"])
    lam_w  = -np.array(data["lambda_wall"])
    lam = np.concatenate([lam_dd, lam_w])
    z = np.concatenate([x, y, r, lam]).astype(np.float64)

    def eval_F(z):
        return np.array(sys.F(z.tolist(), zero=0.0, one=1.0), dtype=np.float64)

    def eval_J(z):
        rows = sys.jacobian(z.tolist(), zero=0.0, one=1.0)
        return np.array([[float(c) for c in row] for row in rows], dtype=np.float64)

    print("Newton polish iterations:")
    for it in range(12):
        F = eval_F(z)
        rn = np.linalg.norm(F, np.inf)
        print(f"  it {it:2d}: ||F||_inf = {rn:.3e}")
        if rn < 1e-15:
            break
        J = eval_J(z)
        dz = np.linalg.solve(J, F)
        z = z - dz

    F = eval_F(z)
    print(f"final ||F||_inf = {np.linalg.norm(F, np.inf):.3e}")

    # Also report the sum of r at the polished point.
    r_new = z[2*n: 3*n]
    print(f"sum_r (polished) = {r_new.sum():.16f}")
    r_old = np.array(data["r"])
    print(f"sum_r (parent)   = {r_old.sum():.16f}")
    print(f"delta sum_r      = {r_new.sum() - r_old.sum():+.3e}")
    print(f"||r_new - r_old||_inf = {np.linalg.norm(r_new - r_old, np.inf):.3e}")

    # Also check multipliers stayed positive
    lam_new = z[3*n: 3*n + m]
    print(f"lambda range: [{lam_new.min():.6e}, {lam_new.max():.6e}]")

    # Save
    out = {
        "n": int(n),
        "m": int(m),
        "x": z[0:n].tolist(),
        "y": z[n:2*n].tolist(),
        "r": z[2*n:3*n].tolist(),
        "lambda": lam_new.tolist(),
        "contacts_dd": data["contacts_dd"],
        "contacts_wall": data["contacts_wall"],
        "sum_r_polished": float(r_new.sum()),
        "sum_r_parent": float(r_old.sum()),
        "F_inf_polished": float(np.linalg.norm(F, np.inf)),
        "convention": "L = f - sum lam g; g <= 0; lam >= 0",
    }
    (HERE / "kkt_point_polished.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {HERE / 'kkt_point_polished.json'}")


if __name__ == "__main__":
    main()
