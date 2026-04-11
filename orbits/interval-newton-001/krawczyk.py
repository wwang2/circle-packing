"""Stage 3: Krawczyk interval-Newton contraction certificate.

Let  F : R^156 -> R^156  be the polynomial KKT system of kkt_system.py, and
let  z0  be the Newton-polished numerical solution stored in
kkt_point_polished.json, satisfying ||F(z0)||_inf ~ 4e-16.

Fix an interval box
    B  =  z0 + [-eps, +eps]^156
and a float-precision preconditioner
    C  =  J(z0)^-1   in float64.

The Krawczyk operator is
    K(B)  =  z0  -  C F(z0)  +  (I  -  C J(B)) (B - z0)

where  J(B)  is a rigorous interval enclosure of the Jacobian over B, and all
arithmetic on the right-hand side is done with mpmath.iv rigorous interval
arithmetic at >= 30 decimal digits (much tighter than double precision, so
the "C J(B)" term is computed with essentially no rounding-error inflation).

Krawczyk's theorem [Krawczyk 1969; see Moore & Kearfott & Cloud 2009]:
    If  K(B)  is contained in the interior of  B,
    then  F  has a unique zero in  B  and it lies in  K(B).

Further, any zero of F in B is contained in B intersect K(B).

This gives us:
  - EXISTENCE and UNIQUENESS of a true KKT critical point in B
  - TIGHT ENCLOSURE: the true z* lies in K(B), so we can bound sum_r by
    sum of the intersection of B and K(B) over the r-coordinates.

Output: certificate.json + a sweep plot over eps.
"""
from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass

import numpy as np
from mpmath import iv, mpf, mp

from kkt_system import KKTSystem

HERE = pathlib.Path(__file__).resolve().parent

# Use enough digits that interval rounding does not dominate our eps.
# For eps ~ 1e-12 we want ~20+ digits of precision.
iv.dps = 50
mp.dps = 50

ZERO_IV = iv.mpf(0)
ONE_IV = iv.mpf(1)


def load_polished():
    data = json.loads((HERE / "kkt_point_polished.json").read_text())
    n = data["n"]
    m = data["m"]
    x = np.array(data["x"])
    y = np.array(data["y"])
    r = np.array(data["r"])
    lam = np.array(data["lambda"])
    z0 = np.concatenate([x, y, r, lam])
    sys = KKTSystem.from_point_file(HERE / "kkt_point.json")
    return sys, z0, data


def mpf_vec(z_float: np.ndarray):
    """Convert a float64 vector to a list of exact mpmath.iv point intervals.
    Each entry becomes [mpf(x), mpf(x)] -- an exact singleton interval
    representing the float itself."""
    return [iv.mpf(mpf(float(v))) for v in z_float]


def box_vec(z_center: np.ndarray, eps: float):
    """Build interval box [z - eps, z + eps] as mpmath.iv intervals."""
    e = mpf(eps)
    return [iv.mpf([mpf(float(v)) - e, mpf(float(v)) + e]) for v in z_center]


def iv_vec_sub(a, b):
    return [a[i] - b[i] for i in range(len(a))]


def iv_matmul_float_iv(C_float: np.ndarray, v_iv: list):
    """Compute C @ v where C is float matrix, v is interval vector. Lifts C
    to exact point intervals so the result is rigorous."""
    d, n = C_float.shape
    out = [ZERO_IV] * d
    for i in range(d):
        s = ZERO_IV
        row = C_float[i]
        for j in range(n):
            s = s + iv.mpf(mpf(float(row[j]))) * v_iv[j]
        out[i] = s
    return out


def iv_matmul_float_ivmat(C_float: np.ndarray, A_iv):
    """(d x n) float @ (n x n) interval matrix -> (d x n) interval matrix.
    A_iv is a list of lists of iv.mpf."""
    d, n = C_float.shape
    nn = len(A_iv)
    assert n == nn
    cols = len(A_iv[0])
    out = [[ZERO_IV] * cols for _ in range(d)]
    for i in range(d):
        for j in range(cols):
            s = ZERO_IV
            for k in range(n):
                s = s + iv.mpf(mpf(float(C_float[i, k]))) * A_iv[k][j]
            out[i][j] = s
    return out


def iv_ivmat_ivvec(M_iv, v_iv):
    d = len(M_iv)
    n = len(v_iv)
    out = [ZERO_IV] * d
    for i in range(d):
        s = ZERO_IV
        row = M_iv[i]
        for j in range(n):
            s = s + row[j] * v_iv[j]
        out[i] = s
    return out


def iv_contains_in_interior(K_iv, B_iv) -> bool:
    """Return True iff K[i] is a strict subset of B[i] for every i."""
    for i in range(len(K_iv)):
        kL, kU = K_iv[i].a, K_iv[i].b   # mpmath 'a','b' = lo,hi
        bL, bU = B_iv[i].a, B_iv[i].b
        if not (bL < kL and kU < bU):
            return False
    return True


def iv_widths(v_iv):
    return [float(v.b - v.a) for v in v_iv]


def krawczyk_attempt(sys: KKTSystem, z0: np.ndarray, eps: float, C_float: np.ndarray,
                     *, verbose: bool = True):
    """Run the Krawczyk contraction test at radius eps. Returns a dict with:
        contracted: bool
        K_widths_max, K_widths_avg
        F_z0_inf (float)
        wall_time
        K_intervals  (mpmath.iv list or None)
    """
    t0 = time.time()
    dim = len(z0)
    B_iv = box_vec(z0, eps)
    z0_iv = mpf_vec(z0)           # exact interval point at z0

    # Step 1: F(z0) in rigorous intervals (evaluating the polynomial at an
    # exact point gives a very tight interval already; imprecision comes only
    # from rounding).
    Fz0_iv = sys.F(z0_iv, zero=ZERO_IV, one=ONE_IV)
    Fz0_inf = max(float(max(abs(v.a), abs(v.b))) for v in Fz0_iv)

    # Step 2: Interval Jacobian over B
    #    J_B[p][q] is an interval enclosure of dF_p/dz_q over B.
    J_B = sys.jacobian(B_iv, zero=ZERO_IV, one=ONE_IV)
    # sys.jacobian returns a list of lists of iv.mpf — exactly what we need.

    # Step 3: middle term:  I  -  C @ J_B   (interval matrix, dim x dim)
    CJ = iv_matmul_float_ivmat(C_float, J_B)
    # Subtract from identity
    ImCJ = [[(ONE_IV - CJ[i][j]) if i == j else (-CJ[i][j]) for j in range(dim)]
            for i in range(dim)]
    # We want ||I - C J_B|| to be small. A quick check: compute row-sum norm.
    rowsum_norm_widths = max(
        sum(float(max(abs(ImCJ[i][j].a), abs(ImCJ[i][j].b))) for j in range(dim))
        for i in range(dim)
    )

    # Step 4: B - z0   (just [-eps, +eps]^dim)
    Bmz0 = [iv.mpf([mpf(-eps), mpf(+eps)]) for _ in range(dim)]

    # Step 5: (I - C J_B) @ (B - z0)
    term2 = iv_ivmat_ivvec(ImCJ, Bmz0)

    # Step 6: K(B) = z0 - C F(z0) + term2
    CFz0 = iv_matmul_float_iv(C_float, Fz0_iv)
    K_iv = [z0_iv[i] - CFz0[i] + term2[i] for i in range(dim)]

    # Step 7: contraction check
    contracted = iv_contains_in_interior(K_iv, B_iv)

    widths = iv_widths(K_iv)
    dt = time.time() - t0
    if verbose:
        print(f"  eps = {eps:.2e}   ||F(z0)|| = {Fz0_inf:.3e}   "
              f"||I - C J_B||_rowsum = {rowsum_norm_widths:.3e}   "
              f"max K width = {max(widths):.3e}   "
              f"contracted = {contracted}   [{dt:.1f}s]")

    return {
        "eps": eps,
        "contracted": contracted,
        "K_widths_max": float(max(widths)),
        "K_widths_avg": float(sum(widths) / len(widths)),
        "F_z0_inf": Fz0_inf,
        "rowsum_ImCJ": rowsum_norm_widths,
        "wall_time": dt,
        "K": K_iv,
        "B": B_iv,
    }


def main():
    sys, z0, data = load_polished()
    dim = len(z0)
    print(f"Loaded polished point, dim = {dim}")
    print(f"sum_r = {np.array(z0[2*sys.n:3*sys.n]).sum():.16f}")

    # Build C = J(z0)^-1 in float64.
    z0_list = z0.tolist()
    J0 = np.array(
        [[float(c) for c in row] for row in sys.jacobian(z0_list, zero=0.0, one=1.0)],
        dtype=np.float64,
    )
    C = np.linalg.inv(J0)
    print(f"J(z0) float cond = {np.linalg.cond(J0):.3e}")
    print(f"||C||_inf = {np.linalg.norm(C, np.inf):.3e}")

    # Epsilon sweep — start tight, expand until contraction fails.
    eps_list = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    results = []
    print("\n=== Krawczyk sweep ===")
    last_ok = None
    for eps in eps_list:
        res = krawczyk_attempt(sys, z0, eps, C, verbose=True)
        if res["contracted"]:
            last_ok = res
        else:
            results.append(res)
            break
        results.append(res)

    if last_ok is None:
        print("\n!!! Contraction failed at all tested eps. "
              "Either z0 is not a genuine KKT point, or the preconditioner is off. "
              "Investigate before declaring defeat.")
        (HERE / "krawczyk_sweep.json").write_text(
            json.dumps([{k: v for k, v in r.items() if k not in ("K", "B")} for r in results], indent=2)
        )
        return

    # Build the certificate from the smallest successful eps (tightest).
    # Actually: for the certificate statement we want a particular eps. Use
    # the SMALLEST one that still contracts — this gives the tightest
    # guaranteed box and the tightest interval on sum_r.
    certified = [r for r in results if r["contracted"]]
    cert_small = certified[0]   # smallest eps
    cert_big = certified[-1]    # biggest eps that still contracted

    # Compute verified interval for sum_r from K of the smallest-eps run.
    K = cert_small["K"]
    n = sys.n
    r_intervals = K[2*n : 3*n]
    sum_r_iv = ZERO_IV
    for ri in r_intervals:
        sum_r_iv = sum_r_iv + ri
    sum_r_lo = float(sum_r_iv.a)
    sum_r_hi = float(sum_r_iv.b)
    print(f"\n=== Certified sum_r interval (eps = {cert_small['eps']:.0e}) ===")
    print(f"sum_r ∈ [{sum_r_lo:.16f}, {sum_r_hi:.16f}]")
    print(f"width = {sum_r_hi - sum_r_lo:.3e}")

    # Save certificate.
    import hashlib
    contact_key = json.dumps(
        {"dd": data["contacts_dd"], "wall": data["contacts_wall"]},
        sort_keys=True
    ).encode()
    contact_hash = hashlib.sha256(contact_key).hexdigest()
    center_key = json.dumps(z0.tolist()).encode()
    center_hash = hashlib.sha256(center_key).hexdigest()

    certificate = {
        "method": "krawczyk_interval_newton",
        "tool": "mpmath.iv (rigorous arbitrary-precision intervals)",
        "dps": iv.dps,
        "dim": dim,
        "num_active_constraints": len(data["contacts_dd"]) + len(data["contacts_wall"]),
        "num_dd_contacts": len(data["contacts_dd"]),
        "num_wall_contacts": len(data["contacts_wall"]),
        "contact_graph_hash": f"sha256:{contact_hash}",
        "box_center_hash": f"sha256:{center_hash}",
        "box_center_F_inf": cert_small["F_z0_inf"],
        "smallest_contracted_eps": cert_small["eps"],
        "largest_contracted_eps": cert_big["eps"],
        "contraction_verified": True,
        "sum_r_lower_bound": sum_r_lo,
        "sum_r_upper_bound": sum_r_hi,
        "sum_r_width": sum_r_hi - sum_r_lo,
        "sum_r_midpoint": 0.5 * (sum_r_lo + sum_r_hi),
        "sweep": [
            {k: v for k, v in r.items() if k not in ("K", "B")} for r in results
        ],
    }
    (HERE / "certificate.json").write_text(json.dumps(certificate, indent=2))
    print(f"\nWrote {HERE / 'certificate.json'}")

    # Also save the box intervals for the tightest box as an auxiliary file.
    aux = {
        "eps": cert_small["eps"],
        "K_lo": [float(k.a) for k in K],
        "K_hi": [float(k.b) for k in K],
        "B_lo": [float(b.a) for b in cert_small["B"]],
        "B_hi": [float(b.b) for b in cert_small["B"]],
    }
    (HERE / "krawczyk_box.json").write_text(json.dumps(aux, indent=2))


if __name__ == "__main__":
    main()
