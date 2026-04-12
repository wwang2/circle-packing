"""
Spatial Branch-and-Bound v3: Hybrid FT + distance formulation.

Lessons from v1 and v2:
- v1 (full McCormick on [0,1]^78): root LP = 13.0 (trivial), converges slowly
- v2 (reduced variables, distance bounds): root LP = 0.0 (all distances zero
  because center boxes overlap on the full domain)

The problem: neither approach alone works on the full box.
- FT gives a good bound (2.74) but doesn't improve with branching
- Distance bounds give perfect bounds at tiny boxes but are useless on large ones

Solution: COMBINE them.

For each B&B node with center boxes B_i:
1. Compute d_min(i,j) for all pairs
2. Build radius LP with:
   a. Pairwise: r_i + r_j <= d_min(i,j)  (only for non-overlapping box pairs)
   b. FT area: tangent linearization of sum(2*sqrt(3)*r_i^2) <= 1
   c. Containment: r_i <= min(x_lo, 1-x_hi, y_lo, 1-y_hi)
   d. Individual FT: r_i <= sqrt(1 / (2*sqrt(3)))  [single-circle area bound]
   e. Top-k sum bounds from FT applied to subsets

The FT tangent cuts are the secret weapon: they provide a globally valid bound
that the distance constraints can then tighten as boxes shrink.

Key insight: on the full box, FT gives 2.74. As we branch, the distance
constraints kick in and (eventually) push the bound below 2.74. The question
is: how much branching is needed?
"""

import numpy as np
import json
import time
import heapq
from scipy.optimize import linprog
import os
import sys
import pickle

N = 26
INCUMBENT = 2.6359830865


def load_incumbent():
    sol_path = "/Users/wujiewang/code/circle-packing/research/solutions/mobius-001/solution_n26.json"
    with open(sol_path) as f:
        data = json.load(f)
    circles = np.array(data["circles"])
    return circles[:, 0], circles[:, 1], circles[:, 2]


def min_distance_boxes(xi_lo, xi_hi, yi_lo, yi_hi, xj_lo, xj_hi, yj_lo, yj_hi):
    """Minimum Euclidean distance between two axis-aligned rectangles."""
    gap_x = max(xi_lo - xj_hi, xj_lo - xi_hi, 0.0)
    gap_y = max(yi_lo - yj_hi, yj_lo - yi_hi, 0.0)
    return np.sqrt(gap_x**2 + gap_y**2)


def max_radius_containment(x_lo, x_hi, y_lo, y_hi):
    """Max radius given center box, to stay inside [0,1]^2."""
    return min(x_lo, 1.0 - x_hi, y_lo, 1.0 - y_hi)


def solve_hybrid_lp(x_lo, x_hi, y_lo, y_hi, r_inc):
    """
    Solve the hybrid LP combining FT tangent cuts and distance bounds.

    Variables: r_0, ..., r_{n-1}

    Constraints:
    1. Pairwise distance: r_i + r_j <= d_min(i,j) for separated box pairs
    2. FT tangent linearizations at multiple points
    3. Containment: r_i <= r_max_contain(i)
    4. Non-negativity: r_i >= 0
    """
    n = N
    coeff_ft = 2.0 * np.sqrt(3.0)

    # Objective: maximize sum(r_i)
    c = -np.ones(n)

    rows = []
    rhs = []

    # ---- 1. Pairwise distance bounds ----
    dist_constraints_added = 0
    for i in range(n):
        for j in range(i+1, n):
            d = min_distance_boxes(x_lo[i], x_hi[i], y_lo[i], y_hi[i],
                                   x_lo[j], x_hi[j], y_lo[j], y_hi[j])
            if d > 1e-12:
                row = np.zeros(n)
                row[i] = 1.0
                row[j] = 1.0
                rows.append(row)
                rhs.append(d)
                dist_constraints_added += 1

    # ---- 2. FT tangent cuts ----
    # The constraint sum(coeff_ft * r_i^2) <= 1 is convex.
    # Tangent at point r_ref: coeff_ft * sum(2*r_ref[i]*r_i - r_ref[i]^2) <= 1
    # => sum(2*coeff_ft*r_ref[i]*r_i) <= 1 + coeff_ft*sum(r_ref[i]^2)

    # Tangent at incumbent
    row = np.zeros(n)
    for i in range(n):
        row[i] = 2.0 * coeff_ft * r_inc[i]
    rows.append(row.copy())
    rhs.append(1.0 + coeff_ft * np.sum(r_inc**2))

    # Tangent at equal radii
    r_eq = np.sqrt(1.0 / (n * coeff_ft))
    row = np.zeros(n)
    row[:] = 2.0 * coeff_ft * r_eq
    rows.append(row.copy())
    rhs.append(1.0 + coeff_ft * n * r_eq**2)

    # Multiple tangent points for tighter outer approximation
    for alpha in [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0]:
        r_ref = r_inc * alpha
        r_ref = np.minimum(r_ref, 0.5)
        row = np.zeros(n)
        for i in range(n):
            row[i] = 2.0 * coeff_ft * r_ref[i]
        rows.append(row.copy())
        rhs.append(1.0 + coeff_ft * np.sum(r_ref**2))

    # Tangent at various uniform values
    for r_val in [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]:
        row = np.zeros(n)
        row[:] = 2.0 * coeff_ft * r_val
        rows.append(row.copy())
        rhs.append(1.0 + coeff_ft * n * r_val**2)

    # ---- 3. Containment bounds ----
    r_max = np.array([max_radius_containment(x_lo[i], x_hi[i], y_lo[i], y_hi[i])
                      for i in range(n)])

    bounds = [(0.0, max(0.0, r_max[i])) for i in range(n)]

    # ---- 4. Pair bound: r_i + r_j <= 2 - sqrt(2) ~ 0.586 ----
    # This is valid for any two circles in [0,1]^2
    pair_bound = 2.0 - np.sqrt(2.0)
    for i in range(n):
        for j in range(i+1, n):
            # Only add if tighter than the distance bound
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            rows.append(row)
            rhs.append(pair_bound)

    A_ub = np.array(rows)
    b_ub = np.array(rhs)

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                    options={'presolve': True})

    if result.success:
        return -result.fun, result.x, dist_constraints_added
    else:
        return None, None, dist_constraints_added


def spatial_bb_v3(time_limit=600, node_limit=200000, verbose=True):
    """
    Run the hybrid spatial B&B.

    Start with full box. FT cuts keep the root LP at ~2.74.
    Branch on center positions to activate distance constraints.
    """
    start_time = time.time()
    n = N
    x_inc, y_inc, r_inc = load_incumbent()

    # Initial box
    x_lo = np.zeros(n)
    x_hi = np.ones(n)
    y_lo = np.zeros(n)
    y_hi = np.ones(n)

    root_ub, root_sol, root_dists = solve_hybrid_lp(x_lo, x_hi, y_lo, y_hi, r_inc)

    if verbose:
        ft_ub = np.sqrt(n / (2*np.sqrt(3)))
        print(f"Root LP: UB = {root_ub:.6f}" if root_ub else "Root infeasible")
        print(f"  Distance constraints active: {root_dists}")
        print(f"  FT UB: {ft_ub:.6f}")
        print(f"  Incumbent: {INCUMBENT:.10f}")

    if root_ub is None or root_ub <= INCUMBENT:
        return {'ub': root_ub or 0, 'nodes': 1, 'pruned': 0,
                'time': time.time()-start_time, 'history': [(0, root_ub, 1)]}

    # Smart branching: prioritize circles that appear in many binding constraints
    # For now, branch on the circle with widest center box that has the
    # largest radius in the LP solution (these contribute most to the objective)

    # B&B data structure: store (x_lo, x_hi, y_lo, y_hi) per node
    # Use a heap with (-ub, node_id, state)
    pq = [(-root_ub, 0, x_lo, x_hi, y_lo, y_hi, 0)]
    nc = 1
    explored = 1
    pruned = 0
    active = {0: root_ub}
    global_ub = root_ub
    history = [(0.0, root_ub, 1)]
    lp = time.time()
    best_ub_ever = root_ub

    while pq and explored < node_limit:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break

        neg_ub, nid, xl, xh, yl, yh, dep = heapq.heappop(pq)
        local_ub = -neg_ub

        if nid in active:
            del active[nid]

        if local_ub <= INCUMBENT:
            pruned += 1
            continue

        # Re-evaluate to get LP solution for branching decision
        ub_reeval, r_sol, _ = solve_hybrid_lp(xl, xh, yl, yh, r_inc)
        explored += 1

        if ub_reeval is None or ub_reeval <= INCUMBENT:
            pruned += 1
            continue

        # Select branching variable
        # Strategy: branch on circle with largest r_sol[i] * width_i
        # This targets circles contributing most to the UB excess
        best_score = -1
        best_i = -1
        best_c = 0

        for i in range(n):
            xw = xh[i] - xl[i]
            yw = yh[i] - yl[i]
            r_val = r_sol[i] if r_sol is not None else 0.1

            score_x = r_val * xw
            score_y = r_val * yw

            if score_x > best_score:
                best_score = score_x
                best_i = i
                best_c = 0
            if score_y > best_score:
                best_score = score_y
                best_i = i
                best_c = 1

        if best_score < 1e-10:
            continue

        # Bisect
        if best_c == 0:
            mid = (xl[best_i] + xh[best_i]) / 2.0
        else:
            mid = (yl[best_i] + yh[best_i]) / 2.0

        for half in range(2):
            cxl, cxh = xl.copy(), xh.copy()
            cyl, cyh = yl.copy(), yh.copy()

            if best_c == 0:
                if half == 0:
                    cxh[best_i] = mid
                else:
                    cxl[best_i] = mid
            else:
                if half == 0:
                    cyh[best_i] = mid
                else:
                    cyl[best_i] = mid

            # Quick feasibility: containment must allow positive radii
            # (at least for enough circles to be interesting)
            child_ub, child_sol, _ = solve_hybrid_lp(cxl, cxh, cyl, cyh, r_inc)
            explored += 1

            if child_ub is None or child_ub <= INCUMBENT:
                pruned += 1
                continue

            heapq.heappush(pq, (-child_ub, nc, cxl, cxh, cyl, cyh, dep+1))
            active[nc] = child_ub
            nc += 1

        # Update global UB
        if active:
            global_ub = max(active.values())
        else:
            global_ub = INCUMBENT

        if global_ub < best_ub_ever:
            best_ub_ever = global_ub

        now = time.time()
        if verbose and (now - lp > 5.0):
            print(f"t={elapsed:.1f}s  nodes={explored}  pruned={pruned}  "
                  f"active={len(pq)}  UB={global_ub:.6f}  "
                  f"depth={dep}  branch=c{best_i}{'x' if best_c==0 else 'y'}")
            lp = now

        history.append((elapsed, global_ub, explored))

    elapsed = time.time() - start_time
    history.append((elapsed, global_ub, explored))

    if verbose:
        print(f"\nFinal: UB={global_ub:.6f}, nodes={explored}, pruned={pruned}, "
              f"active={len(pq)}, time={elapsed:.1f}s")

    return {
        'ub': global_ub,
        'nodes': explored,
        'pruned': pruned,
        'active': len(pq),
        'time': elapsed,
        'history': history,
    }


def analyze_distance_structure():
    """
    Analyze the incumbent solution to understand which pairs are close
    and how the distance constraints interact with FT.

    This helps us understand whether B&B can practically tighten the bound.
    """
    n = N
    x_inc, y_inc, r_inc = load_incumbent()

    print(f"Incumbent solution analysis (n={n}):")
    print(f"  Sum of radii: {np.sum(r_inc):.10f}")
    print(f"  Radii range: [{np.min(r_inc):.6f}, {np.max(r_inc):.6f}]")
    print(f"  Mean radius: {np.mean(r_inc):.6f}")

    # Compute pairwise distances and gaps
    distances = []
    gaps = []  # dist_ij - (r_i + r_j)
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt((x_inc[i] - x_inc[j])**2 + (y_inc[i] - y_inc[j])**2)
            g = d - (r_inc[i] + r_inc[j])
            distances.append(d)
            gaps.append(g)

    distances = np.array(distances)
    gaps = np.array(gaps)

    print(f"\n  Pairwise distances:")
    print(f"    Min: {np.min(distances):.6f}")
    print(f"    Max: {np.max(distances):.6f}")
    print(f"    Mean: {np.mean(distances):.6f}")
    print(f"\n  Non-overlap gaps (dist - (ri+rj)):")
    print(f"    Min: {np.min(gaps):.10f}")
    print(f"    Max: {np.max(gaps):.6f}")
    print(f"    Contacts (gap < 1e-6): {np.sum(gaps < 1e-6)}")
    print(f"    Near-contacts (gap < 0.01): {np.sum(gaps < 0.01)}")

    # FT analysis
    coeff = 2.0 * np.sqrt(3.0)
    ft_area = coeff * np.sum(r_inc**2)
    print(f"\n  FT area usage: {ft_area:.6f} / 1.0 ({ft_area*100:.2f}%)")
    ft_ub = np.sqrt(n / coeff)
    print(f"  FT UB: {ft_ub:.6f}")
    print(f"  FT gap: {(ft_ub - np.sum(r_inc)) / np.sum(r_inc) * 100:.2f}%")

    # What if we use the incumbent geometry exactly?
    # LP: max sum(r) s.t. r_i + r_j <= d_ij, containment
    from scipy.optimize import linprog
    c_lp = -np.ones(n)
    rows = []
    rhs_list = []
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt((x_inc[i] - x_inc[j])**2 + (y_inc[i] - y_inc[j])**2)
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            rows.append(row)
            rhs_list.append(d)

    A_ub = np.array(rows)
    b_ub = np.array(rhs_list)

    bounds = []
    for i in range(n):
        rmax = min(x_inc[i], 1-x_inc[i], y_inc[i], 1-y_inc[i])
        bounds.append((0.0, rmax))

    res = linprog(c_lp, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if res.success:
        print(f"\n  LP at exact incumbent geometry: sum(r) = {-res.fun:.10f}")
        print(f"  (should match incumbent: {np.sum(r_inc):.10f})")
        r_lp = res.x
        # How much slack in each constraint?
        slack_count = 0
        for k, (i, j) in enumerate([(i,j) for i in range(n) for j in range(i+1,n)]):
            d = np.sqrt((x_inc[i] - x_inc[j])**2 + (y_inc[i] - y_inc[j])**2)
            s = d - (r_lp[i] + r_lp[j])
            if s < 1e-8:
                slack_count += 1
        print(f"  Binding distance constraints: {slack_count}")

    # Compute how much "room" there is in the FT bound
    # If we perturb centers, how much can sum(r) increase?
    print(f"\n  Theoretical analysis:")
    print(f"  For the FT bound to be tight, we need equal radii r* = {r_eq_val(n):.6f}")
    print(f"  The incumbent has varying radii: std = {np.std(r_inc):.6f}")
    print(f"  By Cauchy-Schwarz: sum(r) <= sqrt(n * sum(r^2))")
    cs_ub = np.sqrt(n * np.sum(r_inc**2))
    print(f"  Cauchy-Schwarz UB with incumbent areas: {cs_ub:.6f}")

    return distances, gaps


def r_eq_val(n):
    """Equal-radius value at FT optimum."""
    return np.sqrt(1.0 / (n * 2.0 * np.sqrt(3.0)))


def comprehensive_analysis():
    """
    Run the full analysis + B&B with proper time allocation.
    """
    print("=" * 70)
    print("SPATIAL B&B v3: HYBRID FT + DISTANCE BOUNDS")
    print("=" * 70)

    # Analysis
    print("\n--- Incumbent Analysis ---")
    analyze_distance_structure()

    # Root LP
    print("\n--- Root LP (full box) ---")
    n = N
    x_inc, y_inc, r_inc = load_incumbent()
    x_lo, x_hi = np.zeros(n), np.ones(n)
    y_lo, y_hi = np.zeros(n), np.ones(n)

    root_ub, root_sol, root_dc = solve_hybrid_lp(x_lo, x_hi, y_lo, y_hi, r_inc)
    ft_ub = np.sqrt(n / (2*np.sqrt(3)))
    print(f"Root LP UB: {root_ub:.6f}")
    print(f"FT UB:      {ft_ub:.6f}")
    print(f"Distance constraints: {root_dc}")

    # B&B
    print("\n--- Branch and Bound (10 min) ---")
    result = spatial_bb_v3(time_limit=600, node_limit=200000, verbose=True)

    return result


if __name__ == '__main__':
    result = comprehensive_analysis()

    # Save results
    results_path = os.path.join(os.path.dirname(__file__), 'results_v3.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(result, f)
    print(f"\nResults saved to {results_path}")
