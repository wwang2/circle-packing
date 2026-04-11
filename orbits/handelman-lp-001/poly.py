"""
Sparse polynomial arithmetic for the Krivine-Handelman LP.

Variables: u[0..77] — x_1..x_26, y_1..y_26, r_1..r_26.

A polynomial is a dict {monomial_tuple: coefficient} where monomial_tuple is
a sorted tuple of (var_index, exponent) pairs with positive exponent.
The empty tuple () represents the constant monomial 1.

All coefficients are Python floats; we never go through dense expansion.
"""
from __future__ import annotations

from typing import Dict, Tuple

Monomial = Tuple[Tuple[int, int], ...]
Poly = Dict[Monomial, float]

N_DISKS = 26
N_VARS = 3 * N_DISKS  # 78

X = lambda i: i                 # 0..25
Y = lambda i: N_DISKS + i       # 26..51
R = lambda i: 2 * N_DISKS + i   # 52..77


def mono_mul(a: Monomial, b: Monomial) -> Monomial:
    """Multiply two monomials represented as sorted ((var, exp), ...) tuples."""
    if not a:
        return b
    if not b:
        return a
    # Merge two sorted lists
    out = []
    i = j = 0
    la, lb = len(a), len(b)
    while i < la and j < lb:
        va, ea = a[i]
        vb, eb = b[j]
        if va == vb:
            out.append((va, ea + eb))
            i += 1
            j += 1
        elif va < vb:
            out.append((va, ea))
            i += 1
        else:
            out.append((vb, eb))
            j += 1
    if i < la:
        out.extend(a[i:])
    if j < lb:
        out.extend(b[j:])
    return tuple(out)


def mono_degree(m: Monomial) -> int:
    return sum(e for _, e in m)


def poly_add(p: Poly, q: Poly, scale: float = 1.0) -> Poly:
    """Return p + scale * q."""
    out = dict(p)
    for m, c in q.items():
        nc = out.get(m, 0.0) + scale * c
        if nc == 0.0:
            if m in out:
                del out[m]
        else:
            out[m] = nc
    return out


def poly_mul(p: Poly, q: Poly) -> Poly:
    out: Poly = {}
    for mp, cp in p.items():
        if cp == 0.0:
            continue
        for mq, cq in q.items():
            if cq == 0.0:
                continue
            m = mono_mul(mp, mq)
            c = cp * cq
            nc = out.get(m, 0.0) + c
            if nc == 0.0:
                if m in out:
                    del out[m]
            else:
                out[m] = nc
    return out


def poly_scale(p: Poly, s: float) -> Poly:
    return {m: c * s for m, c in p.items() if c != 0.0}


def var_poly(v: int, coeff: float = 1.0) -> Poly:
    """Polynomial = coeff * u_v."""
    return {((v, 1),): coeff}


def const_poly(c: float) -> Poly:
    if c == 0.0:
        return {}
    return {(): c}


def poly_neg(p: Poly) -> Poly:
    return {m: -c for m, c in p.items()}


def build_constraints():
    """Build the containment linear constraints and non-overlap quadratics.

    Returns:
      lin_polys: list of degree-1 polynomials, all ≥ 0 on feasible set.
        ordering:
          for i in 0..25:
            r_i ≥ 0
            x_i − r_i ≥ 0
            (1 − x_i) − r_i ≥ 0
            y_i − r_i ≥ 0
            (1 − y_i) − r_i ≥ 0
            (0.5 − r_i) ≥ 0
        → 6 per disk × 26 = 156 linear constraints.
      quad_polys: list of 325 non-overlap quadratics g_{ij} ≥ 0.
      lin_labels, quad_labels: human-readable labels.
    """
    lin_polys = []
    lin_labels = []
    for i in range(N_DISKS):
        xi, yi, ri = X(i), Y(i), R(i)
        # r_i ≥ 0
        lin_polys.append({((ri, 1),): 1.0})
        lin_labels.append(f"r_{i}>=0")
        # x_i − r_i ≥ 0
        lin_polys.append({((xi, 1),): 1.0, ((ri, 1),): -1.0})
        lin_labels.append(f"x_{i}-r_{i}>=0")
        # 1 − x_i − r_i ≥ 0
        lin_polys.append({(): 1.0, ((xi, 1),): -1.0, ((ri, 1),): -1.0})
        lin_labels.append(f"1-x_{i}-r_{i}>=0")
        # y_i − r_i ≥ 0
        lin_polys.append({((yi, 1),): 1.0, ((ri, 1),): -1.0})
        lin_labels.append(f"y_{i}-r_{i}>=0")
        # 1 − y_i − r_i ≥ 0
        lin_polys.append({(): 1.0, ((yi, 1),): -1.0, ((ri, 1),): -1.0})
        lin_labels.append(f"1-y_{i}-r_{i}>=0")
        # 0.5 − r_i ≥ 0
        lin_polys.append({(): 0.5, ((ri, 1),): -1.0})
        lin_labels.append(f"0.5-r_{i}>=0")

    quad_polys = []
    quad_labels = []
    for i in range(N_DISKS):
        for j in range(i + 1, N_DISKS):
            # (x_i − x_j)² + (y_i − y_j)² − (r_i + r_j)²
            xi, xj = X(i), X(j)
            yi, yj = Y(i), Y(j)
            ri, rj = R(i), R(j)

            g: Poly = {}
            # (xi − xj)² = xi² − 2 xi xj + xj²
            g[((xi, 2),)] = g.get(((xi, 2),), 0.0) + 1.0
            g[((xj, 2),)] = g.get(((xj, 2),), 0.0) + 1.0
            g[((xi, 1), (xj, 1))] = g.get(((xi, 1), (xj, 1)), 0.0) - 2.0
            # (yi − yj)² = yi² − 2 yi yj + yj²
            g[((yi, 2),)] = g.get(((yi, 2),), 0.0) + 1.0
            g[((yj, 2),)] = g.get(((yj, 2),), 0.0) + 1.0
            g[((yi, 1), (yj, 1))] = g.get(((yi, 1), (yj, 1)), 0.0) - 2.0
            # − (ri + rj)² = − ri² − 2 ri rj − rj²
            g[((ri, 2),)] = g.get(((ri, 2),), 0.0) - 1.0
            g[((rj, 2),)] = g.get(((rj, 2),), 0.0) - 1.0
            g[((ri, 1), (rj, 1))] = g.get(((ri, 1), (rj, 1)), 0.0) - 2.0

            quad_polys.append(g)
            quad_labels.append(f"g_{i}_{j}")

    return lin_polys, lin_labels, quad_polys, quad_labels


def objective_poly() -> Poly:
    """The linear objective Σ r_i."""
    out: Poly = {}
    for i in range(N_DISKS):
        out[((R(i), 1),)] = 1.0
    return out


def evaluate_poly(p: Poly, u):
    """Numerically evaluate p at a point u (length-78 array)."""
    val = 0.0
    for m, c in p.items():
        prod = c
        for v, e in m:
            prod *= u[v] ** e
        val += prod
    return val
