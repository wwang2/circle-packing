"""
Spatial Branch-and-Bound with McCormick LP relaxations for circle packing UB.

Problem: maximize sum(r_i) for n=26 circles in [0,1]^2, no overlap, all inside.

The non-overlap constraint for circles i,j is:
    (xi - xj)^2 + (yi - yj)^2 >= (ri + rj)^2

Expanding:
    xi^2 - 2*xi*xj + xj^2 + yi^2 - 2*yi*yj + yj^2 - ri^2 - 2*ri*rj - rj^2 >= 0

This contains quadratic terms: xi^2, xj^2, yi^2, yj^2, ri^2, rj^2 (univariate)
and bilinear terms: xi*xj, yi*yj, ri*rj.

On a box [a, b] for variable z:
  - z^2 is underestimated by the secant: z^2 >= a*z + b*z - a*b  (WRONG for under)
  - Actually for z^2 on [a,b]:
    - Convex envelope (underestimator = function itself, overestimator = secant)
    - z^2 >= z^2 (trivially, but we need linear)
    - For UNDER-estimating z^2: tangent line at any point c: z^2 >= 2c*z - c^2
      Best tangent depends on LP dual, but we can add several
    - For OVER-estimating z^2: secant: z^2 <= (a+b)*z - a*b

For bilinear w = u*v on [uL, uU] x [vL, vU], McCormick envelopes:
  w >= uL*v + vL*u - uL*vL   (underestimator 1)
  w >= uU*v + vU*u - uU*vU   (underestimator 2)
  w <= uU*v + vL*u - uU*vL   (overestimator 1)
  w <= uL*v + vU*u - uL*vU   (overestimator 2)

Strategy:
1. Introduce auxiliary variables for each quadratic/bilinear term
2. Add McCormick envelope constraints
3. Reformulate non-overlap as linear constraints on auxiliaries
4. Solve LP relaxation: maximize sum(r_i)
5. Branch on variable with widest McCormick gap

Key optimization: we don't need ALL 325 non-overlap constraints.
The incumbent solution tells us which pairs are actually close.
We start with all pairs but can prune pairs that are provably separated
on a given box.
"""

import numpy as np
import json
import time
import heapq
from scipy.optimize import linprog
from multiprocessing import Pool
import os
import sys

# ---------- Problem data ----------
N = 26
INCUMBENT = 2.6359830865  # Best known feasible sum_r

def load_incumbent():
    """Load the incumbent solution to get warm-start bounds."""
    sol_path = os.path.join(os.path.dirname(__file__),
                            "../../research/solutions/mobius-001/solution_n26.json")
    if not os.path.exists(sol_path):
        # Try from the main repo
        sol_path = "/Users/wujiewang/code/circle-packing/research/solutions/mobius-001/solution_n26.json"

    with open(sol_path) as f:
        data = json.load(f)

    circles = np.array(data["circles"])  # shape (26, 3): x, y, r
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    return x, y, r

# ---------- McCormick LP relaxation ----------

def build_mccormick_lp(box_lo, box_hi, use_tangent_cuts=True, incumbent_val=INCUMBENT):
    """
    Build the McCormick LP relaxation for circle packing on a given box.

    Variables layout:
      x_0..x_{n-1}, y_0..y_{n-1}, r_0..r_{n-1}  (3n primal variables)
      Then auxiliary variables for quadratic terms needed in non-overlap constraints.

    For each pair (i,j), the non-overlap constraint involves:
      xi^2, xj^2, xi*xj, yi^2, yj^2, yi*yj, ri^2, rj^2, ri*rj

    We introduce auxiliary variables:
      For each variable z in {x_0,...,x_{n-1},y_0,...,y_{n-1},r_0,...,r_{n-1}}:
        w_z = z^2  (one per variable, shared across pairs)
      For each pair (i,j):
        pxij = xi*xj
        pyij = yi*yj
        prij = ri*rj

    Total auxiliaries: 3n (squares) + 3*C(n,2) (cross terms) = 78 + 3*325 = 1053
    Total variables: 78 + 1053 = 1131

    Returns: (ub, status) where ub is the LP upper bound on sum(r_i).
    Returns None if infeasible.
    """
    n = N
    npairs = n * (n - 1) // 2  # 325

    # Variable indices
    # Primal: x[0..25], y[26..51], r[52..77]
    # Squares: sq_x[78..103], sq_y[104..129], sq_r[130..155]
    # Cross terms per pair (i,j) with i<j, ordered lexicographically:
    #   px[156..480], py[481..805], pr[806..1130]

    nvar = 3*n + 3*n + 3*npairs  # 78 + 78 + 975 = 1131

    ix = lambda i: i                # x_i
    iy = lambda i: n + i            # y_i
    ir = lambda i: 2*n + i          # r_i
    isq_x = lambda i: 3*n + i      # x_i^2
    isq_y = lambda i: 3*n + n + i   # y_i^2
    isq_r = lambda i: 3*n + 2*n + i # r_i^2

    pair_idx = {}
    pidx = 0
    for i in range(n):
        for j in range(i+1, n):
            pair_idx[(i,j)] = pidx
            pidx += 1

    ipx = lambda i,j: 3*n + 3*n + pair_idx[(i,j)]           # x_i*x_j
    ipy = lambda i,j: 3*n + 3*n + npairs + pair_idx[(i,j)]  # y_i*y_j
    ipr = lambda i,j: 3*n + 3*n + 2*npairs + pair_idx[(i,j)] # r_i*r_j

    # Objective: maximize sum(r_i) = minimize -sum(r_i)
    c = np.zeros(nvar)
    for i in range(n):
        c[ir(i)] = -1.0

    # Bounds
    lb = np.zeros(nvar)
    ub_var = np.zeros(nvar)

    for i in range(n):
        lb[ix(i)] = box_lo[i]
        ub_var[ix(i)] = box_hi[i]
        lb[iy(i)] = box_lo[n + i]
        ub_var[iy(i)] = box_hi[n + i]
        lb[ir(i)] = box_lo[2*n + i]
        ub_var[ir(i)] = box_hi[2*n + i]

    # Bounds for square auxiliary variables
    for i in range(n):
        for offset, isq, ivar in [(0, isq_x, ix), (1, isq_y, iy), (2, isq_r, ir)]:
            lo_v = lb[ivar(i)]
            hi_v = ub_var[ivar(i)]
            # z^2 on [lo, hi]: range is [min(lo^2, hi^2, ...), max(lo^2, hi^2)]
            if lo_v >= 0:
                sq_lo = lo_v**2
                sq_hi = hi_v**2
            elif hi_v <= 0:
                sq_lo = hi_v**2
                sq_hi = lo_v**2
            else:
                sq_lo = 0.0
                sq_hi = max(lo_v**2, hi_v**2)
            lb[isq(i)] = sq_lo
            ub_var[isq(i)] = sq_hi

    # Bounds for cross-term auxiliary variables (McCormick bounds)
    for i in range(n):
        for j in range(i+1, n):
            for icross, ivar1, ivar2 in [(ipx, ix, ix), (ipy, iy, iy), (ipr, ir, ir)]:
                u_lo = lb[ivar1(i)]
                u_hi = ub_var[ivar1(i)]
                v_lo = lb[ivar2(j)]
                v_hi = ub_var[ivar2(j)]

                products = [u_lo*v_lo, u_lo*v_hi, u_hi*v_lo, u_hi*v_hi]
                lb[icross(i,j)] = min(products)
                ub_var[icross(i,j)] = max(products)

    # Build inequality constraints: A_ub @ x <= b_ub
    rows = []
    rhs = []

    # --- Containment constraints ---
    # r_i <= x_i <= 1 - r_i  =>  r_i - x_i <= 0  and  x_i + r_i <= 1
    # r_i <= y_i <= 1 - r_i  =>  r_i - y_i <= 0  and  y_i + r_i <= 1
    for i in range(n):
        # r_i - x_i <= 0
        row = np.zeros(nvar)
        row[ir(i)] = 1.0
        row[ix(i)] = -1.0
        rows.append(row)
        rhs.append(0.0)

        # x_i + r_i <= 1
        row = np.zeros(nvar)
        row[ix(i)] = 1.0
        row[ir(i)] = 1.0
        rows.append(row)
        rhs.append(1.0)

        # r_i - y_i <= 0
        row = np.zeros(nvar)
        row[ir(i)] = 1.0
        row[iy(i)] = -1.0
        rows.append(row)
        rhs.append(0.0)

        # y_i + r_i <= 1
        row = np.zeros(nvar)
        row[iy(i)] = 1.0
        row[ir(i)] = 1.0
        rows.append(row)
        rhs.append(1.0)

    # --- McCormick envelopes for squares: w = z^2 on [a, b] ---
    # Overestimator (secant): w <= (a+b)*z - a*b
    # Underestimator (tangent at midpoint): w >= 2*m*z - m^2 where m = (a+b)/2
    # Better: add tangent at both endpoints:
    #   w >= 2*a*z - a^2
    #   w >= 2*b*z - b^2

    for i in range(n):
        for isq, ivar in [(isq_x, ix), (isq_y, iy), (isq_r, ir)]:
            a = lb[ivar(i)]
            b = ub_var[ivar(i)]

            # Overestimator: w <= (a+b)*z - a*b  =>  w - (a+b)*z <= -a*b
            row = np.zeros(nvar)
            row[isq(i)] = 1.0
            row[ivar(i)] = -(a + b)
            rows.append(row)
            rhs.append(-a * b)

            if use_tangent_cuts:
                # Underestimator tangent at a: w >= 2a*z - a^2  =>  -w + 2a*z <= a^2
                row = np.zeros(nvar)
                row[isq(i)] = -1.0
                row[ivar(i)] = 2.0 * a
                rows.append(row)
                rhs.append(a**2)

                # Underestimator tangent at b: w >= 2b*z - b^2  =>  -w + 2b*z <= b^2
                row = np.zeros(nvar)
                row[isq(i)] = -1.0
                row[ivar(i)] = 2.0 * b
                rows.append(row)
                rhs.append(b**2)

                # Tangent at midpoint for tighter bound
                m = (a + b) / 2.0
                row = np.zeros(nvar)
                row[isq(i)] = -1.0
                row[ivar(i)] = 2.0 * m
                rows.append(row)
                rhs.append(m**2)

    # --- McCormick envelopes for bilinear terms: w = u*v ---
    for i in range(n):
        for j in range(i+1, n):
            for icross, ivar1, ivar2 in [(ipx, ix, ix), (ipy, iy, iy), (ipr, ir, ir)]:
                u_lo = lb[ivar1(i)]
                u_hi = ub_var[ivar1(i)]
                v_lo = lb[ivar2(j)]
                v_hi = ub_var[ivar2(j)]

                # Underestimators:
                # w >= uL*v + vL*u - uL*vL  =>  -w + uL*v + vL*u <= uL*vL
                row = np.zeros(nvar)
                row[icross(i,j)] = -1.0
                row[ivar2(j)] = u_lo
                row[ivar1(i)] = v_lo
                rows.append(row)
                rhs.append(u_lo * v_lo)

                # w >= uU*v + vU*u - uU*vU  =>  -w + uU*v + vU*u <= uU*vU
                row = np.zeros(nvar)
                row[icross(i,j)] = -1.0
                row[ivar2(j)] = u_hi
                row[ivar1(i)] = v_hi
                rows.append(row)
                rhs.append(u_hi * v_hi)

                # Overestimators:
                # w <= uU*v + vL*u - uU*vL  =>  w - uU*v - vL*u <= -uU*vL
                row = np.zeros(nvar)
                row[icross(i,j)] = 1.0
                row[ivar2(j)] = -u_hi
                row[ivar1(i)] = -v_lo
                rows.append(row)
                rhs.append(-u_hi * v_lo)

                # w <= uL*v + vU*u - uL*vU  =>  w - uL*v - vU*u <= -uL*vU
                row = np.zeros(nvar)
                row[icross(i,j)] = 1.0
                row[ivar2(j)] = -u_lo
                row[ivar1(i)] = -v_hi
                rows.append(row)
                rhs.append(-u_lo * v_hi)

    # --- Non-overlap constraints ---
    # (xi - xj)^2 + (yi - yj)^2 >= (ri + rj)^2
    # Expanding:
    #   xi^2 - 2*xi*xj + xj^2 + yi^2 - 2*yi*yj + yj^2 >= ri^2 + 2*ri*rj + rj^2
    # Rearranging:
    #   xi^2 - 2*xi*xj + xj^2 + yi^2 - 2*yi*yj + yj^2 - ri^2 - 2*ri*rj - rj^2 >= 0
    #
    # Using auxiliary variables:
    #   sq_x[i] - 2*px[i,j] + sq_x[j] + sq_y[i] - 2*py[i,j] + sq_y[j]
    #   - sq_r[i] - 2*pr[i,j] - sq_r[j] >= 0
    #
    # For the LP relaxation to be a VALID upper bound on sum(r), we need
    # the non-overlap constraints to be RELAXED (made easier to satisfy).
    #
    # The non-overlap constraint says: LHS >= 0 where LHS = distance^2 - (ri+rj)^2
    # To relax: we want a WEAKER version that is implied by the original.
    #
    # We need to UNDER-estimate the LHS (make it easier to be >= 0).
    # LHS = (xi^2 - 2*xi*xj + xj^2) + (yi^2 - 2*yi*yj + yj^2) - (ri^2 + 2*ri*rj + rj^2)
    #
    # For valid relaxation (UNDER-estimating LHS):
    # - xi^2, xj^2, yi^2, yj^2: use UNDERESTIMATORS (tangent cuts)
    # - xi*xj, yi*yj: appear with -2 coefficient, so use OVERESTIMATORS to maximize -2*pxij
    #   Actually: -2*pxij, so to underestimate LHS we need to OVERESTIMATE pxij.
    # - ri^2, rj^2: appear with -1 coefficient, so use OVERESTIMATORS
    # - ri*rj: appears with -2 coefficient, so use OVERESTIMATOR
    #
    # But we already have McCormick envelopes constraining the auxiliary variables.
    # The LP automatically handles this: it will use the envelopes to find the
    # loosest valid combination. We just state the non-overlap in terms of auxiliaries.
    # The constraint is:
    #   sq_x[i] - 2*px[i,j] + sq_x[j] + sq_y[i] - 2*py[i,j] + sq_y[j]
    #   - sq_r[i] - 2*pr[i,j] - sq_r[j] >= 0
    #
    # As <= constraint: negate and <= 0

    for i in range(n):
        for j in range(i+1, n):
            row = np.zeros(nvar)
            # -(sq_x[i] - 2*px[i,j] + sq_x[j] + sq_y[i] - 2*py[i,j] + sq_y[j]
            #   - sq_r[i] - 2*pr[i,j] - sq_r[j]) <= 0
            row[isq_x(i)] = -1.0
            row[ipx(i,j)] = 2.0
            row[isq_x(j)] = -1.0
            row[isq_y(i)] = -1.0
            row[ipy(i,j)] = 2.0
            row[isq_y(j)] = -1.0
            row[isq_r(i)] = 1.0
            row[ipr(i,j)] = 2.0
            row[isq_r(j)] = 1.0
            rows.append(row)
            rhs.append(0.0)

    A_ub = np.array(rows)
    b_ub = np.array(rhs)

    bounds = list(zip(lb, ub_var))

    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                        options={'presolve': True, 'time_limit': 60.0})

        if result.success:
            return -result.fun, 'optimal', result.x
        elif result.status == 2:  # infeasible
            return None, 'infeasible', None
        else:
            return None, f'failed: {result.message}', None
    except Exception as e:
        return None, f'error: {str(e)}', None


def tighten_variable_bounds(box_lo, box_hi):
    """
    Use containment constraints to tighten bounds.
    r_i <= x_i <= 1 - r_i, r_i <= y_i <= 1 - r_i
    This means: x_i >= r_i, so if r_lo > x_lo, tighten x_lo.
    Also: x_i + r_i <= 1, so x_i <= 1 - r_lo and r_i <= 1 - x_lo.
    """
    n = N
    lo = box_lo.copy()
    hi = box_hi.copy()

    changed = True
    iterations = 0
    while changed and iterations < 10:
        changed = False
        iterations += 1
        for i in range(n):
            xi_lo, xi_hi = lo[i], hi[i]
            yi_lo, yi_hi = lo[n+i], hi[n+i]
            ri_lo, ri_hi = lo[2*n+i], hi[2*n+i]

            # x_i >= r_i
            new_xi_lo = max(xi_lo, ri_lo)
            # x_i <= 1 - r_i
            new_xi_hi = min(xi_hi, 1.0 - ri_lo)
            # r_i <= x_i (so r_hi <= x_hi)
            new_ri_hi_x = min(ri_hi, xi_hi)
            # x_i + r_i <= 1 => r_i <= 1 - x_lo
            new_ri_hi_x2 = min(ri_hi, 1.0 - xi_lo)

            # y_i >= r_i
            new_yi_lo = max(yi_lo, ri_lo)
            # y_i <= 1 - r_i
            new_yi_hi = min(yi_hi, 1.0 - ri_lo)
            # r_i <= y_i
            new_ri_hi_y = min(ri_hi, yi_hi)
            # y_i + r_i <= 1 => r_i <= 1 - y_lo
            new_ri_hi_y2 = min(ri_hi, 1.0 - yi_lo)

            new_ri_hi = min(new_ri_hi_x, new_ri_hi_x2, new_ri_hi_y, new_ri_hi_y2, ri_hi)

            if new_xi_lo > lo[i] + 1e-12:
                lo[i] = new_xi_lo
                changed = True
            if new_xi_hi < hi[i] - 1e-12:
                hi[i] = new_xi_hi
                changed = True
            if new_yi_lo > lo[n+i] + 1e-12:
                lo[n+i] = new_yi_lo
                changed = True
            if new_yi_hi < hi[n+i] - 1e-12:
                hi[n+i] = new_yi_hi
                changed = True
            if new_ri_hi < hi[2*n+i] - 1e-12:
                hi[2*n+i] = new_ri_hi
                changed = True

            # Also: r_i >= 0 always
            lo[2*n+i] = max(lo[2*n+i], 0.0)

            # Check feasibility
            if lo[i] > hi[i] + 1e-12 or lo[n+i] > hi[n+i] + 1e-12 or lo[2*n+i] > hi[2*n+i] + 1e-12:
                return None, None  # infeasible

    return lo, hi


def get_initial_box():
    """Get the initial box with tightened bounds."""
    n = N
    lo = np.zeros(3*n)
    hi = np.zeros(3*n)

    for i in range(n):
        lo[i] = 0.0       # x_i lower
        hi[i] = 1.0       # x_i upper
        lo[n+i] = 0.0     # y_i lower
        hi[n+i] = 1.0     # y_i upper
        lo[2*n+i] = 0.0   # r_i lower
        hi[2*n+i] = 0.5   # r_i upper (diameter <= 1)

    return tighten_variable_bounds(lo, hi)


def select_branching_variable(box_lo, box_hi, lp_sol):
    """
    Select variable to branch on. Strategy: pick the variable where the
    McCormick gap is largest, focusing on radius variables since they
    directly appear in the objective.

    McCormick gap for z^2: gap = (hi-lo)^2 / 4 at the midpoint.
    We weight radius variables higher since they affect the objective.
    """
    n = N
    best_var = -1
    best_score = -1.0

    # Check all variables
    for k in range(3*n):
        width = box_hi[k] - box_lo[k]
        if width < 1e-8:
            continue

        # Weight: radius variables get 3x weight
        weight = 3.0 if k >= 2*n else 1.0
        score = weight * width

        if score > best_score:
            best_score = score
            best_var = k

    return best_var


def spatial_branch_and_bound(time_limit=600, node_limit=50000, target_ub=2.70,
                              verbose=True):
    """
    Run spatial branch-and-bound to compute an upper bound on sum(r_i).

    Returns dict with:
      - ub: best upper bound found
      - nodes_explored: number of B&B nodes
      - nodes_pruned: number pruned
      - time: wall-clock seconds
      - history: list of (time, ub, nodes) tuples
    """
    start_time = time.time()

    # Initial box
    box_lo, box_hi = get_initial_box()
    if box_lo is None:
        return {'ub': 0.0, 'status': 'infeasible_root'}

    if verbose:
        print(f"Initial box: x in [{box_lo[0]:.3f}, {box_hi[0]:.3f}], "
              f"r in [{box_lo[2*N]:.3f}, {box_hi[2*N]:.3f}]")

    # Solve root LP
    root_ub, root_status, root_sol = build_mccormick_lp(box_lo, box_hi)

    if root_ub is None:
        return {'ub': float('inf'), 'status': f'root_lp_{root_status}'}

    if verbose:
        print(f"Root LP: UB = {root_ub:.6f}, status = {root_status}")
        print(f"Root LP time: {time.time() - start_time:.1f}s")

    if root_ub <= INCUMBENT:
        return {'ub': root_ub, 'nodes_explored': 1, 'nodes_pruned': 0,
                'time': time.time() - start_time, 'history': [(0, root_ub, 1)],
                'status': 'optimal_at_root'}

    # Priority queue: (-ub, node_id, box_lo, box_hi)
    # We use max-heap by negating UB (we want to process highest UB first
    # to improve the global UB quickly? Actually for UB, we want to process
    # the node with the highest local UB because that's where the global UB
    # comes from. The global UB = max over all leaf nodes of their local UBs.
    # So to tighten the global UB, we should subdivide the node with highest UB.)

    node_counter = 0
    pq = []
    heapq.heappush(pq, (-root_ub, node_counter, box_lo, box_hi))
    node_counter += 1

    global_ub = root_ub
    nodes_explored = 1
    nodes_pruned = 0
    history = [(time.time() - start_time, root_ub, 1)]

    best_node_ubs = {0: root_ub}  # Track UBs of all active nodes

    while pq and nodes_explored < node_limit:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            if verbose:
                print(f"Time limit reached: {elapsed:.1f}s")
            break

        neg_ub, nid, blo, bhi = heapq.heappop(pq)
        local_ub = -neg_ub

        # Remove from active set
        if nid in best_node_ubs:
            del best_node_ubs[nid]

        # Check if already pruned
        if local_ub <= INCUMBENT:
            nodes_pruned += 1
            continue

        # Select branching variable
        branch_var = select_branching_variable(blo, bhi, None)
        if branch_var < 0:
            # All variables are fixed -- this node can't be subdivided
            continue

        mid = (blo[branch_var] + bhi[branch_var]) / 2.0

        # Create two child boxes
        for child_idx in range(2):
            child_lo = blo.copy()
            child_hi = bhi.copy()

            if child_idx == 0:
                child_hi[branch_var] = mid
            else:
                child_lo[branch_var] = mid

            # Tighten bounds
            child_lo, child_hi = tighten_variable_bounds(child_lo, child_hi)
            if child_lo is None:
                nodes_pruned += 1
                continue

            # Solve child LP
            child_ub, child_status, child_sol = build_mccormick_lp(child_lo, child_hi)
            nodes_explored += 1

            if child_ub is None or child_ub <= INCUMBENT:
                nodes_pruned += 1
                continue

            # Add to priority queue
            heapq.heappush(pq, (-child_ub, node_counter, child_lo, child_hi))
            best_node_ubs[node_counter] = child_ub
            node_counter += 1

        # Recompute global UB = max over all active leaf UBs
        if best_node_ubs:
            global_ub = max(best_node_ubs.values())
        else:
            global_ub = INCUMBENT

        history.append((elapsed, global_ub, nodes_explored))

        if verbose and nodes_explored % 50 == 0:
            print(f"Nodes: {nodes_explored}, Pruned: {nodes_pruned}, "
                  f"Global UB: {global_ub:.6f}, Active: {len(pq)}, "
                  f"Time: {elapsed:.1f}s")

        if global_ub <= target_ub:
            if verbose:
                print(f"Target UB {target_ub} reached: {global_ub:.6f}")
            break

    elapsed = time.time() - start_time
    history.append((elapsed, global_ub, nodes_explored))

    return {
        'ub': global_ub,
        'nodes_explored': nodes_explored,
        'nodes_pruned': nodes_pruned,
        'active_nodes': len(pq),
        'time': elapsed,
        'history': history,
        'status': 'completed'
    }


def run_fejes_toth_bound():
    """Fejes-Toth area bound for reference."""
    # sum(2*sqrt(3)*r_i^2) <= 1
    # maximize sum(r_i) subject to sum(2*sqrt(3)*r_i^2) <= 1, 0 <= r_i <= 0.5
    # By Cauchy-Schwarz / Lagrange: optimal at equal radii
    # r* = sqrt(1/(n*2*sqrt(3)))
    # sum(r_i) = n * r* = sqrt(n / (2*sqrt(3)))
    n = N
    ft_ub = np.sqrt(n / (2.0 * np.sqrt(3.0)))
    return ft_ub


def quick_root_relaxation():
    """Just solve the root LP to see how loose it is."""
    box_lo, box_hi = get_initial_box()
    if box_lo is None:
        return None, 'infeasible'

    print(f"Box ranges after tightening:")
    print(f"  x: [{box_lo[0]:.4f}, {box_hi[0]:.4f}]")
    print(f"  y: [{box_lo[N]:.4f}, {box_hi[N]:.4f}]")
    print(f"  r: [{box_lo[2*N]:.4f}, {box_hi[2*N]:.4f}]")

    t0 = time.time()
    ub, status, sol = build_mccormick_lp(box_lo, box_hi)
    t1 = time.time()

    print(f"\nRoot McCormick LP relaxation:")
    print(f"  UB = {ub:.6f}" if ub is not None else f"  Status: {status}")
    print(f"  Time: {t1-t0:.2f}s")
    print(f"  Fejes-Toth UB = {run_fejes_toth_bound():.6f}")
    print(f"  Incumbent = {INCUMBENT:.10f}")

    if ub is not None and sol is not None:
        r_vals = sol[2*N:3*N]
        print(f"  LP radii: min={np.min(r_vals):.4f}, max={np.max(r_vals):.4f}, "
              f"sum={np.sum(r_vals):.4f}")

    return ub, status


def run_with_incumbent_neighborhood(margin=0.15, time_limit=300, node_limit=20000,
                                     verbose=True):
    """
    Run B&B starting from a box centered around the incumbent solution.

    This is much more practical than starting from the full [0,1]^78 box:
    the McCormick envelopes are tighter on smaller boxes, so the LP relaxation
    starts closer to the true optimum.

    The idea: if the global optimum is within 'margin' of the incumbent in
    every variable, then the UB from this restricted B&B is a valid bound
    conditional on that assumption. We can gradually increase margin to
    cover more of the space.
    """
    x_inc, y_inc, r_inc = load_incumbent()
    n = N

    box_lo = np.zeros(3*n)
    box_hi = np.zeros(3*n)

    for i in range(n):
        box_lo[i] = max(0.0, x_inc[i] - margin)
        box_hi[i] = min(1.0, x_inc[i] + margin)
        box_lo[n+i] = max(0.0, y_inc[i] - margin)
        box_hi[n+i] = min(1.0, y_inc[i] + margin)
        box_lo[2*n+i] = max(0.0, r_inc[i] - margin)
        box_hi[2*n+i] = min(0.5, r_inc[i] + margin)

    # Tighten
    box_lo, box_hi = tighten_variable_bounds(box_lo, box_hi)
    if box_lo is None:
        return {'ub': 0.0, 'status': 'infeasible'}

    if verbose:
        widths = box_hi - box_lo
        print(f"Neighborhood box (margin={margin}):")
        print(f"  x widths: min={np.min(widths[:n]):.4f}, max={np.max(widths[:n]):.4f}")
        print(f"  y widths: min={np.min(widths[n:2*n]):.4f}, max={np.max(widths[n:2*n]):.4f}")
        print(f"  r widths: min={np.min(widths[2*n:]):.4f}, max={np.max(widths[2*n:]):.4f}")

    # Solve root LP on this tighter box
    t0 = time.time()
    root_ub, root_status, root_sol = build_mccormick_lp(box_lo, box_hi)
    t1 = time.time()

    if verbose:
        print(f"\nRoot LP on neighborhood box:")
        if root_ub is not None:
            print(f"  UB = {root_ub:.6f}")
        else:
            print(f"  Status: {root_status}")
        print(f"  Time: {t1-t0:.2f}s")

    if root_ub is None:
        return {'ub': float('inf'), 'status': f'root_{root_status}'}

    if root_ub <= INCUMBENT:
        return {'ub': root_ub, 'nodes_explored': 1, 'nodes_pruned': 0,
                'time': t1-t0, 'history': [(0, root_ub, 1)], 'status': 'pruned_at_root'}

    # Now run B&B from this box
    node_counter = 0
    pq = []
    heapq.heappush(pq, (-root_ub, node_counter, box_lo, box_hi))
    node_counter += 1

    global_ub = root_ub
    nodes_explored = 1
    nodes_pruned = 0
    history = [(0, root_ub, 1)]
    best_node_ubs = {0: root_ub}

    while pq and nodes_explored < node_limit:
        elapsed = time.time() - t0
        if elapsed > time_limit:
            if verbose:
                print(f"Time limit reached: {elapsed:.1f}s")
            break

        neg_ub, nid, blo, bhi = heapq.heappop(pq)
        local_ub = -neg_ub

        if nid in best_node_ubs:
            del best_node_ubs[nid]

        if local_ub <= INCUMBENT:
            nodes_pruned += 1
            continue

        branch_var = select_branching_variable(blo, bhi, None)
        if branch_var < 0:
            continue

        mid = (blo[branch_var] + bhi[branch_var]) / 2.0

        for child_idx in range(2):
            child_lo = blo.copy()
            child_hi = bhi.copy()

            if child_idx == 0:
                child_hi[branch_var] = mid
            else:
                child_lo[branch_var] = mid

            child_lo, child_hi = tighten_variable_bounds(child_lo, child_hi)
            if child_lo is None:
                nodes_pruned += 1
                continue

            child_ub, child_status, child_sol = build_mccormick_lp(child_lo, child_hi)
            nodes_explored += 1

            if child_ub is None or child_ub <= INCUMBENT:
                nodes_pruned += 1
                continue

            heapq.heappush(pq, (-child_ub, node_counter, child_lo, child_hi))
            best_node_ubs[node_counter] = child_ub
            node_counter += 1

        if best_node_ubs:
            global_ub = max(best_node_ubs.values())
        else:
            global_ub = INCUMBENT

        history.append((elapsed, global_ub, nodes_explored))

        if verbose and nodes_explored % 100 == 0:
            print(f"Nodes: {nodes_explored}, Pruned: {nodes_pruned}, "
                  f"Global UB: {global_ub:.6f}, Active: {len(pq)}, "
                  f"Time: {elapsed:.1f}s")

        if global_ub <= INCUMBENT * 1.001:  # Within 0.1% of incumbent
            if verbose:
                print(f"Near-optimal: UB={global_ub:.6f} vs incumbent={INCUMBENT:.6f}")
            break

    elapsed = time.time() - t0
    history.append((elapsed, global_ub, nodes_explored))

    return {
        'ub': global_ub,
        'nodes_explored': nodes_explored,
        'nodes_pruned': nodes_pruned,
        'active_nodes': len(pq),
        'time': elapsed,
        'history': history,
        'status': 'completed',
        'margin': margin
    }


# ----- Reduced formulation: only model radius variables + constraints -----

def build_reduced_lp(r_lo, r_hi, x_inc, y_inc, r_inc, pair_min_dist=None):
    """
    A reduced LP that only has radius variables (26 vars) and uses the
    incumbent geometry to derive linear constraints.

    The idea: given a fixed geometry (x_i, y_i from incumbent), the non-overlap
    constraint becomes:
        d_ij^2 >= (r_i + r_j)^2
    where d_ij = ||(x_i, y_i) - (x_j, y_j)|| is FIXED.

    So: r_i + r_j <= d_ij  (since both are positive)

    This is a LINEAR constraint! The LP is:
        maximize sum(r_i)
        subject to r_i + r_j <= d_ij  for all i<j
                   r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
                   r_lo[i] <= r_i <= r_hi[i]

    This gives an UB on sum(r) given that geometry. It's not a true global UB
    since different geometries might pack better, but it tells us the maximum
    possible improvement from just changing radii.
    """
    n = N

    # Compute pairwise distances from incumbent
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt((x_inc[i] - x_inc[j])**2 + (y_inc[i] - y_inc[j])**2)
            dists[i, j] = d
            dists[j, i] = d

    # Objective: maximize sum(r_i)
    c = -np.ones(n)

    # Inequality constraints
    rows = []
    rhs = []

    # Non-overlap: r_i + r_j <= d_ij
    for i in range(n):
        for j in range(i+1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            rows.append(row)
            rhs.append(dists[i, j])

    # Containment: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    for i in range(n):
        for bound_val in [x_inc[i], 1.0 - x_inc[i], y_inc[i], 1.0 - y_inc[i]]:
            row = np.zeros(n)
            row[i] = 1.0
            rows.append(row)
            rhs.append(bound_val)

    A_ub = np.array(rows)
    b_ub = np.array(rhs)

    bounds = list(zip(r_lo, r_hi))

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if result.success:
        return -result.fun, result.x
    else:
        return None, None


# ---------- Fejes-Toth with packing-specific strengthening ----------

def fejes_toth_with_pair_cuts():
    """
    Strengthen Fejes-Toth by adding pairwise distance constraints.

    The FT bound assumes independent radii with only the area constraint.
    But circles must also not overlap, which imposes r_i + r_j <= d_ij.
    Using the MINIMUM possible distance between centers (from the containment
    constraints: both centers in [r_i, 1-r_i] x [r_i, 1-r_i]), we get
    a constraint that limits how large close pairs can be.

    This is an LP:
        maximize sum(r_i)
        subject to sum(2*sqrt(3)*r_i^2) <= 1   (FT area -- nonlinear, relax by tangent)
                   r_i + r_j <= sqrt(2) * (1 - r_i - r_j) + ... (various pair bounds)
                   0 <= r_i <= 0.5

    Actually, since sum(2*sqrt(3)*r_i^2) <= 1 is convex quadratic, we can linearize it
    at the incumbent solution to get a valid cut.
    """
    n = N
    x_inc, y_inc, r_inc = load_incumbent()

    # Outer approximation of sum(2*sqrt(3)*r_i^2) <= 1
    # Tangent at incumbent: 2*sqrt(3) * sum(2*r_inc[i] * (r_i - r_inc[i]) + r_inc[i]^2) <= 1
    # => 4*sqrt(3) * sum(r_inc[i] * r_i) <= 1 + 2*sqrt(3) * sum(r_inc[i]^2)

    coeff = 2.0 * np.sqrt(3.0)

    c = -np.ones(n)  # maximize sum(r_i)

    rows = []
    rhs_list = []

    # FT tangent cut at incumbent
    row = np.zeros(n)
    for i in range(n):
        row[i] = 2.0 * coeff * r_inc[i]
    rows.append(row)
    rhs_list.append(1.0 + coeff * np.sum(r_inc**2))

    # Add more tangent cuts at different points
    # At equal radii: r_eq = sqrt(1/(n*2*sqrt(3)))
    r_eq = np.sqrt(1.0 / (n * coeff))
    row = np.zeros(n)
    for i in range(n):
        row[i] = 2.0 * coeff * r_eq
    rows.append(row)
    rhs_list.append(1.0 + coeff * n * r_eq**2)

    # Pair bound: for any two circles, the maximum of r_i + r_j is bounded
    # by their distance. The maximum distance between two points each in
    # [r, 1-r]^2 is sqrt(2)*(1-2r) for equal radii. For the general case:
    # r_i + r_j <= 2 - sqrt(2) (the pair bound from upperbound-001)
    for i in range(n):
        for j in range(i+1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            rows.append(row)
            rhs_list.append(2.0 - np.sqrt(2.0))  # ~0.586

    A_ub = np.array(rows)
    b_ub = np.array(rhs_list)

    bounds = [(0.0, 0.5)] * n

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if result.success:
        return -result.fun, result.x
    return None, None


if __name__ == '__main__':
    print("=" * 70)
    print("SPATIAL BRANCH-AND-BOUND FOR CIRCLE PACKING UPPER BOUND")
    print("=" * 70)

    # Phase 1: Root relaxation on full box
    print("\n--- Phase 1: Root LP on full box ---")
    root_ub, root_status = quick_root_relaxation()

    ft_ub = run_fejes_toth_bound()
    print(f"\nFejes-Toth UB: {ft_ub:.6f}")
    print(f"Incumbent:     {INCUMBENT:.10f}")

    # Phase 2: FT with pair cuts
    print("\n--- Phase 2: FT + pair cuts ---")
    ft_pair_ub, ft_pair_sol = fejes_toth_with_pair_cuts()
    if ft_pair_ub is not None:
        print(f"FT + pair cuts UB: {ft_pair_ub:.6f}")

    # Phase 3: Reduced LP at incumbent geometry
    print("\n--- Phase 3: Reduced LP at incumbent geometry ---")
    x_inc, y_inc, r_inc = load_incumbent()
    r_lo = np.zeros(N)
    r_hi = np.full(N, 0.5)
    reduced_ub, reduced_sol = build_reduced_lp(r_lo, r_hi, x_inc, y_inc, r_inc)
    if reduced_ub is not None:
        print(f"Reduced LP UB (fixed geometry): {reduced_ub:.6f}")
        print(f"  This means: even at the incumbent's center positions,")
        print(f"  the LP-optimal radii give sum = {reduced_ub:.6f}")

    # Phase 4: B&B on neighborhood of incumbent
    print("\n--- Phase 4: B&B on neighborhood ---")
    for margin in [0.05, 0.10]:
        print(f"\n  Margin = {margin}:")
        result = run_with_incumbent_neighborhood(
            margin=margin, time_limit=120, node_limit=5000, verbose=True)
        print(f"  Result: UB={result['ub']:.6f}, nodes={result['nodes_explored']}, "
              f"time={result['time']:.1f}s")

    print("\n" + "=" * 70)
    print("DONE")
