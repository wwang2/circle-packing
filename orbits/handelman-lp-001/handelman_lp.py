"""
Krivine–Handelman LP for upper-bounding Σ r_i on the circle-packing feasible set.

Mathematical content
--------------------
Let P = {u ∈ ℝ^78 : h_k(u) ≥ 0 for k = 1..M_lin, g_{ij}(u) ≥ 0 for each pair}.
The linear constraints define a bounded polytope (the containment box). By
Krivine–Handelman, for any polynomial p strictly positive on P there is a
representation
    p = σ_0 + Σ_α c_α · h^α   (+ Σ_β c_β · g^β in mixed mode)
with σ_0 ≥ 0 constant, c_α ≥ 0, and finitely many nonzero terms.

We apply this to p := UB − Σ_i r_i. For each fixed level k, we enumerate all
allowed products and solve the LP

    min UB
    s.t.  UB − Σ_i r_i − σ_0 − Σ c_α h^α − Σ c_β g^β ≡ 0    (polynomial identity)
           σ_0, c_α, c_β ≥ 0 ,    UB free.

The polynomial identity is expanded coefficient-by-coefficient on the monomial
basis in u = (x, y, r), giving one equality per monomial.

Level semantics
---------------
Mode A (pure linear): only products h^α with |α| ≤ k.
Mode B (mixed):       products h^α · g^β with |α| + 2|β| ≤ k (g has degree 2).

Output
------
A dict with
    { 'level': k, 'mode': 'A' | 'B',
      'n_handelman_products': ...,
      'n_monomials': ...,
      'ub': ...,
      'status': ...,
      'wall_clock': ... }
"""
from __future__ import annotations

import itertools
import random
import time
from typing import List, Tuple

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from poly import (
    N_DISKS,
    N_VARS,
    R,
    Poly,
    build_constraints,
    objective_poly,
    poly_mul,
    poly_scale,
)


# ----------------------------------------------------------------------
# Enumerate Handelman products
# ----------------------------------------------------------------------


def enumerate_linear_products(lin_polys: List[Poly], level: int) -> List[Tuple[Tuple[int, ...], Poly]]:
    """Enumerate all products h_{i_1} * h_{i_2} * ... * h_{i_a} with a ≤ level,
    i_1 ≤ i_2 ≤ ... (multiset — each distinct product appears once).

    Returns list of (tuple_of_indices_with_level, product_polynomial).
    The empty tuple () represents σ_0 = constant 1.
    """
    M = len(lin_polys)
    out: List[Tuple[Tuple[int, ...], Poly]] = []
    # a = 0: constant 1
    out.append(((), {(): 1.0}))
    # a = 1..level
    # Cache products of small prefixes to avoid redoing work
    for a in range(1, level + 1):
        for combo in itertools.combinations_with_replacement(range(M), a):
            # Multiply in order (linear polys are small, so this is cheap)
            p = lin_polys[combo[0]]
            for idx in combo[1:]:
                p = poly_mul(p, lin_polys[idx])
            out.append((combo, p))
    return out


def enumerate_mixed_products(
    lin_polys: List[Poly],
    quad_polys: List[Poly],
    level: int,
    include_hhh: bool = False,
    include_gg: bool = False,
    lin_per_disk: int = None,
) -> List[Tuple[Tuple, Poly]]:
    """Mixed Handelman: products h^α · g^β with |α| + 2|β| ≤ level.

    Args:
      level: total polynomial degree cap.
      include_hhh: whether to include degree-3 h_i h_j h_k products.
        Off by default — 156 choose 3 = 648k products is too many.
      include_gg: whether to include degree-4 g_i g_j products.
      lin_per_disk: if set, restrict linear constraints to only these slots
        per disk. Useful for shrinking the search space.

    The index key is (tuple_of_h_indices, tuple_of_g_indices).
    """
    M_h = len(lin_polys)
    M_g = len(quad_polys)
    out: List[Tuple[Tuple, Poly]] = []

    # constant
    out.append((((), ()), {(): 1.0}))

    # single h
    for i in range(M_h):
        out.append((((i,), ()), lin_polys[i]))

    # single g  (degree 2 — allowed if level ≥ 2)
    if level >= 2:
        for j in range(M_g):
            out.append((((), (j,)), quad_polys[j]))

    # h_i * h_j  (degree 2)
    if level >= 2:
        for i, j in itertools.combinations_with_replacement(range(M_h), 2):
            p = poly_mul(lin_polys[i], lin_polys[j])
            out.append((((i, j), ()), p))

    # h_i * g_j  (degree 3) — always included at level ≥ 3
    if level >= 3:
        for i in range(M_h):
            for j in range(M_g):
                p = poly_mul(lin_polys[i], quad_polys[j])
                out.append((((i,), (j,)), p))
        if include_hhh:
            for combo in itertools.combinations_with_replacement(range(M_h), 3):
                p = lin_polys[combo[0]]
                for k in combo[1:]:
                    p = poly_mul(p, lin_polys[k])
                out.append(((combo, ()), p))

    # g_i * g_j  (degree 4)
    if level >= 4 and include_gg:
        for i, j in itertools.combinations_with_replacement(range(M_g), 2):
            p = poly_mul(quad_polys[i], quad_polys[j])
            out.append((((), (i, j)), p))

    return out


# ----------------------------------------------------------------------
# LP assembly
# ----------------------------------------------------------------------


def build_lp(products: List[Tuple[Tuple, Poly]], obj_poly: Poly, verbose: bool = False):
    """Given enumerated Handelman products, build the LP

        min UB
        s.t. UB · δ_{m=1} − obj[m] − Σ_α c_α · p_α[m] = 0  for every monomial m
             c_α ≥ 0, UB free.

    We gather the full monomial support first, then build a sparse (COO) matrix.

    Variables: [UB, c_0, c_1, ..., c_{K−1}], K = len(products).
    """
    # 1. Build monomial → row-index map
    mono_to_row: dict = {}

    def mono_row(m):
        if m in mono_to_row:
            return mono_to_row[m]
        r = len(mono_to_row)
        mono_to_row[m] = r
        return r

    # Register objective and constant first
    mono_row(())  # constant row
    for m in obj_poly:
        mono_row(m)

    # Register all product monomials
    for _, p in products:
        for m in p:
            mono_row(m)

    n_rows = len(mono_to_row)
    n_vars = 1 + len(products)  # UB + one c per product

    if verbose:
        print(f"[LP] {n_rows} monomials (rows) × {n_vars} vars")

    rows_ij = []
    cols_ij = []
    vals_ij = []

    # UB column: coefficient +1 on the constant row
    rows_ij.append(mono_to_row[()])
    cols_ij.append(0)
    vals_ij.append(1.0)

    # Each product α contributes −p_α[m] on row m, column α+1
    for k, (_, p) in enumerate(products):
        col = k + 1
        for m, c in p.items():
            rows_ij.append(mono_to_row[m])
            cols_ij.append(col)
            vals_ij.append(-c)

    # RHS: obj[m] on each row
    rhs = np.zeros(n_rows)
    for m, c in obj_poly.items():
        rhs[mono_to_row[m]] = c

    A = coo_matrix((vals_ij, (rows_ij, cols_ij)), shape=(n_rows, n_vars)).tocsr()

    return A, rhs, n_vars, n_rows, mono_to_row


def solve_lp_highs(A, rhs, n_vars: int, verbose: bool = False) -> dict:
    """Solve

        min  UB
        s.t. A x = rhs
             x_0 (UB) free, x_1..x_{n-1} ≥ 0.
    """
    import highspy

    h = highspy.Highs()
    if not verbose:
        h.silent()
    else:
        h.setOptionValue("output_flag", True)

    lp = highspy.HighsLp()
    lp.num_col_ = n_vars
    lp.num_row_ = A.shape[0]
    lp.sense_ = highspy.ObjSense.kMinimize

    # Objective: minimize UB = x_0
    col_cost = np.zeros(n_vars)
    col_cost[0] = 1.0
    lp.col_cost_ = col_cost.tolist()

    inf = highspy.kHighsInf
    col_lower = np.zeros(n_vars)
    col_upper = np.full(n_vars, inf)
    col_lower[0] = -inf  # UB free
    lp.col_lower_ = col_lower.tolist()
    lp.col_upper_ = col_upper.tolist()

    # Equality rows: A x = rhs
    lp.row_lower_ = rhs.tolist()
    lp.row_upper_ = rhs.tolist()

    # CSC format for HiGHS
    A_csc = A.tocsc()
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A_csc.indptr.tolist()
    lp.a_matrix_.index_ = A_csc.indices.tolist()
    lp.a_matrix_.value_ = A_csc.data.tolist()

    status = h.passModel(lp)
    run_status = h.run()

    info = h.getInfo()
    sol = h.getSolution()
    model_status = h.getModelStatus()

    ub = sol.col_value[0] if len(sol.col_value) > 0 else float("nan")
    return {
        "ub": float(ub),
        "status": str(model_status),
        "iters": int(info.simplex_iteration_count),
        "obj": float(info.objective_function_value),
    }


# ----------------------------------------------------------------------
# Top-level runner
# ----------------------------------------------------------------------


def run_level(
    mode: str,
    level: int,
    seed: int,
    verbose: bool = False,
    include_hhh: bool = False,
    include_gg: bool = False,
) -> dict:
    """Run one (mode, level, seed) combination.

    mode: 'A' (pure linear) or 'B' (mixed).
    seed: random ordering of products (for robustness check).
    """
    t0 = time.time()
    lin_polys, lin_labels, quad_polys, quad_labels = build_constraints()

    if mode == "A":
        products = enumerate_linear_products(lin_polys, level)
    elif mode == "B":
        products = enumerate_mixed_products(
            lin_polys,
            quad_polys,
            level,
            include_hhh=include_hhh,
            include_gg=include_gg,
        )
    else:
        raise ValueError(f"unknown mode {mode}")

    t_enum = time.time() - t0

    # Random ordering (does not affect LP optimum, only row/column ordering)
    rng = random.Random(seed)
    idx = list(range(len(products)))
    rng.shuffle(idx)
    products = [products[i] for i in idx]

    t1 = time.time()
    obj = objective_poly()
    A, rhs, n_vars, n_rows, _ = build_lp(products, obj, verbose=verbose)
    t_build = time.time() - t1

    t2 = time.time()
    result = solve_lp_highs(A, rhs, n_vars, verbose=verbose)
    t_solve = time.time() - t2

    out = {
        "mode": mode,
        "level": level,
        "seed": seed,
        "n_products": len(products),
        "n_vars": int(n_vars),
        "n_monomials": int(n_rows),
        "nnz": int(A.nnz),
        "ub": result["ub"],
        "status": result["status"],
        "iters": result["iters"],
        "t_enumerate": t_enum,
        "t_build": t_build,
        "t_solve": t_solve,
        "t_total": time.time() - t0,
    }
    if verbose:
        print(out)
    return out


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "B"
    level = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    res = run_level(mode, level, seed, verbose=True)
    print(res)
