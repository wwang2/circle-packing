"""
Sparse Lasserre level-2 relaxation using correlative sparsity.

For each maximal clique C of the chordal completion of the contact graph,
we build a *local* moment matrix over the variables (x_i, y_i, r_i) for i in C,
indexed by monomials of degree up to 2. Call this M_C.

Constraints:
  - Each M_C is PSD.
  - Linear containment inequalities on disk i are added as localizing
    vectors in every clique C containing i (degree <=2 localizing matrix
    since g is degree 1 and we want degree <= 2 polynomial certificates,
    the localizing matrix L_{C,g} := M_C_deg1 * g has degree <= 2*1 + 1 = 3,
    but we only need PSD of the degree-1 localizing matrix of size
    (1 + 3|C|) * 1 because g is linear).
    Simpler: for linear g, the localizing matrix equals g(u)*M_C_deg1 where
    M_C_deg1 is (3|C|+1) x (3|C|+1) and the entries are linear combinations
    of moments in M_C up to degree 3. Since M_C only goes to degree 2, we
    can only enforce g(u) >= 0 as a vector inequality:
    sum_a g_coeff[a] * M_C[0, monomial_a] >= 0, which is just the lifted
    linear constraint. To get stronger, we must localize against a degree-1
    basis: (g(u) * M_C[1:, 1:]) which is degree 1+1=2, covered by M_C.

  - Each non-overlap constraint g_{ij}(u) = (x_i-x_j)^2 + (y_i-y_j)^2 - (r_i+r_j)^2
    is degree 2. For Lasserre level 2 (s=2), we want the localizing matrix
    of order (s - ceil(deg(g)/2)) = (2 - 1) = 1: a scalar constraint
    g_{ij}(u) * 1 >= 0, which is the lifted quadratic. To get real power
    we need LEVEL 3, moment matrices of degree 3, so localizing matrix of
    degree 1 basis times degree-2 g giving degree 4 -- unreachable.

    So for level-2 Lasserre, the non-overlap constraint is only enforced
    as a SCALAR lifted inequality (same as Shor). The tightening comes
    from the fact that M_C has degree-2 moments (cross terms like
    x_i*x_j, r_i*r_j as PROPER matrix entries, linked to monomial entries)
    and the PSD constraint couples them.

  - Consistency: for any two cliques C, C' that share variables, the moments
    indexed by monomials in the shared variables must match between M_C and M_{C'}.

Goal:
  max sum_i r_i
  subject to: M_C PSD for each clique, containment as lifted inequalities,
  non-overlap g_{ij} >= 0 as lifted scalar inequalities, consistency linking.

This is correlative-sparsity Lasserre L2 (Waki-Kim-Kojima-Muramatsu 2006).
"""
import json
import time
from itertools import combinations_with_replacement
from pathlib import Path
import numpy as np
import cvxpy as cp

HERE = Path(__file__).parent
CONTACTS = HERE / "contacts.json"
CLIQUES = HERE / "cliques.json"
OUT = HERE / "relaxation_report.json"


def monomials_up_to_degree(var_names, max_deg):
    """Return list of monomials as sorted tuples of var names (with repetition).
    Includes the empty tuple for the constant 1.
    Order: degree 0, then degree 1 lex, then degree 2 lex, etc.
    """
    mons = [()]
    for d in range(1, max_deg + 1):
        mons.extend(combinations_with_replacement(var_names, d))
    return [tuple(sorted(m)) for m in mons]


def main():
    contacts = json.loads(CONTACTS.read_text())
    cliqs = json.loads(CLIQUES.read_text())
    n = contacts["n"]
    cliques = [tuple(c) for c in cliqs["cliques"]]

    # Variable naming: for disk i use 'x{i}', 'y{i}', 'r{i}'
    def vnames(disks):
        out = []
        for i in disks:
            out += [f"x{i}", f"y{i}", f"r{i}"]
        return out

    # Global monomial -> global index (for consistency across cliques)
    # Since each non-overlap constraint only involves disks within a clique,
    # consistency is needed on the ROW/COL of M_C corresponding to shared vars.
    # We'll use a GLOBAL moment dictionary:
    #   mu[mon] = scalar cvxpy variable
    # Each M_C[a, b] == mu[a ++ b] (sorted multiset).
    # Then M_C PSD enforces local positivity and the SHARED variables auto-link.

    mu = {}  # sorted-tuple -> cvxpy Variable

    def get_mu(mon):
        key = tuple(sorted(mon))
        if key not in mu:
            mu[key] = cp.Variable(name="mu_" + "_".join(key) if key else "mu_1")
        return mu[key]

    # constant 1
    one = get_mu(())

    constraints = [one == 1]

    # Build M_C for each clique
    clique_mats = []
    for idx, C in enumerate(cliques):
        vars_C = vnames(C)
        mons = monomials_up_to_degree(vars_C, 2)  # degree up to 2
        dim = len(mons)  # should be C(3|C|+2, 2)
        # Build the moment matrix as a symbolic cvxpy Expression
        # Each entry M[a,b] = mu[mons[a] + mons[b]]
        rows = []
        for a in range(dim):
            row = []
            for b in range(dim):
                key = tuple(sorted(mons[a] + mons[b]))
                row.append(get_mu(key))
            rows.append(row)
        M_C = cp.bmat([[cp.reshape(rows[a][b], (1, 1), order="C") for b in range(dim)] for a in range(dim)])
        constraints.append(M_C >> 0)
        clique_mats.append((C, mons, M_C))

    # Containment: for each disk, add lifted linear inequalities
    # r_i >= 0, x_i >= r_i, 1-x_i-r_i >= 0, y_i>=r_i, 1-y_i-r_i>=0
    # Lifted: just take the mu-variable of degree-1 monomials.
    for i in range(n):
        xi = get_mu((f"x{i}",))
        yi = get_mu((f"y{i}",))
        ri = get_mu((f"r{i}",))
        constraints += [
            ri >= 0,
            xi - ri >= 0,
            one - xi - ri >= 0,
            yi - ri >= 0,
            one - yi - ri >= 0,
            ri <= 0.5 * one,
        ]

    # ALSO containment as localizing matrices:
    # For each clique C and each disk i in C, multiply the containment
    # polynomial g(u) by every monomial of degree <= 1 in u_C, giving
    # degree-<=2 polynomials, which lift into mu.
    # I.e. enforce: sum over monomials m of deg<=1: g(u) * m * m' >= 0 as a PSD matrix.
    # Since g is linear, g*m*m' has degree <= 3; but we only need g*m >= 0 for
    # each m of degree 1, which gives scalar inequalities.
    # Stronger: localizing MATRIX of degree 1: L[a,b] = mu[g_mons * m_a * m_b]
    # where m_a, m_b are degree <= 1 monomials in u_C. This is a (3|C|+1) x (3|C|+1)
    # PSD matrix. The entries are degree <= 3 in u, which DO NOT LIVE in our
    # level-2 moment dictionary; so we cannot add this.
    # So we're limited to scalar localizations of the linear containments,
    # AND the PSD matrix of the level-2 moment matrix itself.
    # HOWEVER: we CAN enforce the localizing VECTOR g(u)*(1, u1, u2, ...)
    # which has degree 2, covered. This gives inequalities:
    #   g(u) * 1 >= 0  (already in)
    #   g(u) * u_k >= 0 for each var u_k in C
    # No wait -- g*u_k >= 0 is NOT implied by g >= 0 and u_k >= 0 for
    # quadratic terms, and this is exactly the RLT tightening we want.
    # Let me add these (Positivstellensatz order-1 lift).
    # g_box are: r, x-r, 1-x-r, y-r, 1-y-r
    for C in cliques:
        vars_C = vnames(C)
        for i in C:
            # g1 = r_i, g2 = x_i - r_i, g3 = 1 - x_i - r_i, g4 = y_i - r_i, g5 = 1 - y_i - r_i
            # Multiply each g by each u_k in C: get quadratic >= 0
            for uk in vars_C:
                mu_uk = get_mu((uk,))
                mu_rk = get_mu(tuple(sorted((f"r{i}", uk))))
                mu_xk = get_mu(tuple(sorted((f"x{i}", uk))))
                mu_yk = get_mu(tuple(sorted((f"y{i}", uk))))
                # r_i * u_k >= 0 only if u_k >= 0, which holds for x,y,r (all in [0,1])
                constraints.append(mu_rk >= 0)
                constraints.append(mu_xk - mu_rk >= 0)  # (x_i - r_i) * u_k >= 0 if u_k>=0
                constraints.append(mu_uk - mu_xk - mu_rk >= 0)  # (1 - x_i - r_i)*u_k
                constraints.append(mu_yk - mu_rk >= 0)
                constraints.append(mu_uk - mu_yk - mu_rk >= 0)
                # Also (0.5 - r_i) * u_k >= 0
                constraints.append(0.5 * mu_uk - mu_rk >= 0)

    # Non-overlap (quadratic): for each pair (i,j) THAT IS WITHIN SOME CLIQUE:
    # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    # Lifts to: mu[x_i x_i] - 2 mu[x_i x_j] + mu[x_j x_j] + ... >= 0
    clique_set_pairs = set()
    for C in cliques:
        for i, j in combinations_with_replacement(sorted(C), 2):
            if i < j:
                clique_set_pairs.add((i, j))

    # Also add all ORIGINAL contact-graph edges, since they must be in some clique
    for a, b in contacts["disk_disk_edges"]:
        i, j = sorted((a, b))
        clique_set_pairs.add((i, j))

    # And additionally ALL pairs (non-overlap) — we want them all but many
    # won't be in any clique. We can still enforce them as scalar global lifted
    # inequalities IF the relevant quadratic moments exist. Add all pairs where
    # the quadratic moments exist in mu, otherwise we'd need to add them.
    # For correctness (rigorous UB) we should include ALL 325 pairs.
    # We'll materialize the quadratic moments if not yet present.
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    non_overlap_lift_count = 0
    for i, j in all_pairs:
        def mom(a, b):
            return get_mu(tuple(sorted((a, b))))
        xi_xi = mom(f"x{i}", f"x{i}")
        xj_xj = mom(f"x{j}", f"x{j}")
        xi_xj = mom(f"x{i}", f"x{j}")
        yi_yi = mom(f"y{i}", f"y{i}")
        yj_yj = mom(f"y{j}", f"y{j}")
        yi_yj = mom(f"y{i}", f"y{j}")
        ri_ri = mom(f"r{i}", f"r{i}")
        rj_rj = mom(f"r{j}", f"r{j}")
        ri_rj = mom(f"r{i}", f"r{j}")
        lhs = (xi_xi - 2 * xi_xj + xj_xj
               + yi_yi - 2 * yi_yj + yj_yj
               - ri_ri - 2 * ri_rj - rj_rj)
        constraints.append(lhs >= 0)
        non_overlap_lift_count += 1

    # Objective: maximize sum of r_i (first-order moments)
    obj_expr = sum(get_mu((f"r{i}",)) for i in range(n))
    prob = cp.Problem(cp.Maximize(obj_expr), constraints)

    total_cliq_dim = sum(len(list(combinations_with_replacement(vnames(C), 2))) + 1 + 3*len(C)
                         for C in cliques)
    num_mu = len(mu)
    print(f"Sparse Lasserre level-2:")
    print(f"  cliques: {len(cliques)}, max size: {max(len(c) for c in cliques)}")
    print(f"  moment variables: {num_mu}")
    print(f"  total constraints: {len(constraints)}")
    print(f"  non-overlap lifted: {non_overlap_lift_count}")
    clique_dims = [len(monomials_up_to_degree(vnames(C), 2)) for C in cliques]
    print(f"  per-clique M_C sizes: min={min(clique_dims)}, max={max(clique_dims)}")
    print(f"  total SDP block size: {sum(clique_dims)}")

    t0 = time.time()
    try:
        prob.solve(solver=cp.CLARABEL, verbose=True, max_iter=200)
    except Exception as e:
        print(f"CLARABEL failed: {e}")
        print("Falling back to SCS...")
        prob.solve(solver=cp.SCS, verbose=True, max_iters=20000)
    t1 = time.time()
    wall = t1 - t0

    ub = float(prob.value) if prob.value is not None else None
    print(f"\nStatus: {prob.status}")
    print(f"UB (sparse L2): {ub}")
    print(f"Wall: {wall:.2f}s")
    print(f"Parent UB: 2.7396  Parent primal: 2.6359830865")

    report = {
        "method": "sparse_lasserre_level2_correlative_sparsity",
        "lasserre_level": 2,
        "num_cliques": len(cliques),
        "max_clique_size": max(len(c) for c in cliques),
        "clique_dims": clique_dims,
        "total_sdp_block_size": sum(clique_dims),
        "num_moment_variables": num_mu,
        "num_constraints": len(constraints),
        "solver": "CLARABEL",
        "status": prob.status,
        "tightened_ub": ub,
        "parent_ub": 2.7396,
        "parent_primal": 2.6359830865,
        "wall_clock_seconds": wall,
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
