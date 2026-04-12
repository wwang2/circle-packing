"""
Spatial Branch-and-Bound v4: Full-variable LP with FT + distance cuts.

Key insight from v3 failure: projecting out center variables kills the LP
because containment (r_i <= x_i) couples r_i to x_i, and when x_i can
be 0, we get r_i <= 0.

Fix: keep all 78 variables (x_i, y_i, r_i), use:
1. Containment as LINEAR constraints: r_i <= x_i, x_i + r_i <= 1, etc.
2. FT tangent cuts (linear in r_i only, globally valid)
3. Pair bound: r_i + r_j <= 2 - sqrt(2) (globally valid)
4. Distance-based cuts on boxes where center separation is known

This LP has 78 variables and is fast to solve. The bound starts at FT (2.74)
and improves as branching on center positions generates distance cuts.

The LP relaxation for the non-overlap constraint on a box:
  (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2

We linearize this using:
  - For separated box pairs: r_i + r_j <= d_min(i,j) (distance bound)
  - For overlapping box pairs: no useful linear constraint from geometry alone

The containment constraints alone provide: r_i <= 0.5, and with FT,
sum(r_i) <= 2.74. Branching on centers creates separation that activates
distance constraints.

Additionally, we add "interaction cuts" from the non-overlap constraint:
For any pair (i,j), we can derive a valid linear inequality by relaxing
the quadratic constraint on the box.

Specifically, on a box where xi in [a, b] and xj in [c, d]:
  (xi - xj)^2 >= min_{box}(xi - xj)^2

But this minimum is 0 if [a,b] and [c,d] overlap. More useful:
the RLT (Reformulation-Linearization Technique) product of
containment constraints gives additional valid cuts.
"""

import numpy as np
import json
import time
import heapq
from scipy.optimize import linprog
import os
import pickle

N = 26
INCUMBENT = 2.6359830865


def load_incumbent():
    sol_path = "/Users/wujiewang/code/circle-packing/research/solutions/mobius-001/solution_n26.json"
    with open(sol_path) as f:
        data = json.load(f)
    circles = np.array(data["circles"])
    return circles[:, 0], circles[:, 1], circles[:, 2]


def solve_lp_on_box(x_lo, x_hi, y_lo, y_hi, r_lo, r_hi, r_inc, add_rlt=True):
    """
    Solve the LP relaxation on a box.

    Variables: x_0,...,x_{n-1}, y_0,...,y_{n-1}, r_0,...,r_{n-1}
    Indices: x_i = i, y_i = n+i, r_i = 2n+i

    Returns: (ub, solution_vector) or (None, None) if infeasible
    """
    n = N
    nvar = 3 * n  # 78
    coeff_ft = 2.0 * np.sqrt(3.0)

    # Objective: maximize sum(r_i)
    c = np.zeros(nvar)
    for i in range(n):
        c[2*n + i] = -1.0

    rows = []
    rhs = []

    # ---- Containment constraints (LINEAR) ----
    for i in range(n):
        # r_i <= x_i  =>  r_i - x_i <= 0
        row = np.zeros(nvar)
        row[2*n+i] = 1.0
        row[i] = -1.0
        rows.append(row); rhs.append(0.0)

        # x_i + r_i <= 1
        row = np.zeros(nvar)
        row[i] = 1.0
        row[2*n+i] = 1.0
        rows.append(row); rhs.append(1.0)

        # r_i <= y_i  =>  r_i - y_i <= 0
        row = np.zeros(nvar)
        row[2*n+i] = 1.0
        row[n+i] = -1.0
        rows.append(row); rhs.append(0.0)

        # y_i + r_i <= 1
        row = np.zeros(nvar)
        row[n+i] = 1.0
        row[2*n+i] = 1.0
        rows.append(row); rhs.append(1.0)

    # ---- FT tangent cuts ----
    # sum(coeff_ft * r_i^2) <= 1
    # Tangent at r_ref: sum(2*coeff_ft*r_ref[i]*r_i) <= 1 + coeff_ft*sum(r_ref[i]^2)

    tangent_points = [r_inc]

    # Equal radii
    r_eq = np.sqrt(1.0 / (n * coeff_ft))
    tangent_points.append(np.full(n, r_eq))

    # Scaled versions of incumbent
    for alpha in [0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]:
        tangent_points.append(np.minimum(r_inc * alpha, 0.5))

    # Various uniform values
    for rv in [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]:
        tangent_points.append(np.full(n, rv))

    for r_ref in tangent_points:
        row = np.zeros(nvar)
        for i in range(n):
            row[2*n+i] = 2.0 * coeff_ft * r_ref[i]
        rows.append(row)
        rhs.append(1.0 + coeff_ft * np.sum(r_ref**2))

    # ---- Pair bound: r_i + r_j <= 2 - sqrt(2) ----
    pair_bound = 2.0 - np.sqrt(2.0)
    for i in range(n):
        for j in range(i+1, n):
            row = np.zeros(nvar)
            row[2*n+i] = 1.0
            row[2*n+j] = 1.0
            rows.append(row)
            rhs.append(pair_bound)

    # ---- Distance-based cuts for separated center boxes ----
    dist_cuts = 0
    for i in range(n):
        for j in range(i+1, n):
            # Min distance between center box i and center box j
            gap_x = max(x_lo[i] - x_hi[j], x_lo[j] - x_hi[i], 0.0)
            gap_y = max(y_lo[i] - y_hi[j], y_lo[j] - y_hi[i], 0.0)
            d_min = np.sqrt(gap_x**2 + gap_y**2)

            if d_min > 1e-10:
                # r_i + r_j <= d_min
                row = np.zeros(nvar)
                row[2*n+i] = 1.0
                row[2*n+j] = 1.0
                rows.append(row)
                rhs.append(d_min)
                dist_cuts += 1

    # ---- RLT cuts from products of containment constraints ----
    if add_rlt:
        # Product of (x_i - r_i >= 0) and (x_j - r_j >= 0) gives:
        # x_i*x_j - x_i*r_j - r_i*x_j + r_i*r_j >= 0
        # We can't use this directly (bilinear), but we can use
        # McCormick relaxation on each bilinear term for the current box.
        #
        # On box [a,b] x [c,d], McCormick lower bound for u*v:
        #   u*v >= max(a*v + c*u - a*c, b*v + d*u - b*d)
        #
        # For x_i*r_j on [x_lo[i],x_hi[i]] x [r_lo[j],r_hi[j]]:
        #   x_i*r_j >= x_lo[i]*r_j + r_lo[j]*x_i - x_lo[i]*r_lo[j]
        #   x_i*r_j >= x_hi[i]*r_j + r_hi[j]*x_i - x_hi[i]*r_hi[j]
        #
        # This gets complex. Let's add a simpler RLT cut:
        # From (1 - x_i - r_i >= 0) and (r_j >= 0):
        #   r_j*(1 - x_i - r_i) >= 0
        #   r_j - r_j*x_i - r_j*r_i >= 0
        # McCormick lower bound for r_j*x_i:
        #   r_j*x_i >= r_lo[j]*x_i + x_lo[i]*r_j - r_lo[j]*x_lo[i]
        # So: r_j - [r_lo[j]*x_i + x_lo[i]*r_j - r_lo[j]*x_lo[i]] - r_j*r_i >= 0
        #
        # This is still bilinear in r_j*r_i. Skip for now.
        pass

    # ---- Symmetry-breaking ----
    # Order circles by radius (approximately): the largest circle is circle 0
    # In the incumbent, radii are ordered. We can add:
    # r_0 >= r_1 >= ... >= r_{n-1}  (if we relabel)
    # But the problem doesn't specify labels, so we can't do this without
    # changing the problem structure. Skip.

    A_ub = np.array(rows)
    b_ub = np.array(rhs)

    bounds = []
    for i in range(n):
        bounds.append((x_lo[i], x_hi[i]))
    for i in range(n):
        bounds.append((y_lo[i], y_hi[i]))
    for i in range(n):
        bounds.append((r_lo[i], r_hi[i]))

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                    options={'presolve': True, 'time_limit': 30.0})

    if result.success:
        return -result.fun, result.x, dist_cuts
    elif result.status == 2:
        return None, None, dist_cuts  # infeasible
    else:
        return None, None, dist_cuts


def spatial_bb_v4(time_limit=600, node_limit=200000, verbose=True):
    """Run spatial B&B with full-variable LP."""
    start_time = time.time()
    n = N
    x_inc, y_inc, r_inc = load_incumbent()

    # Initial box
    x_lo, x_hi = np.zeros(n), np.ones(n)
    y_lo, y_hi = np.zeros(n), np.ones(n)
    r_lo = np.zeros(n)
    r_hi = np.full(n, 0.5)

    # Root LP
    root_ub, root_sol, root_dc = solve_lp_on_box(
        x_lo, x_hi, y_lo, y_hi, r_lo, r_hi, r_inc)

    ft_ub = np.sqrt(n / (2*np.sqrt(3)))

    if verbose:
        print(f"Root LP: UB = {root_ub:.6f}" if root_ub else "Root infeasible")
        print(f"  FT UB: {ft_ub:.6f}")
        print(f"  Incumbent: {INCUMBENT:.10f}")
        print(f"  Distance cuts at root: {root_dc}")
        if root_sol is not None:
            r_vals = root_sol[2*n:]
            print(f"  LP radii: min={np.min(r_vals):.4f}, max={np.max(r_vals):.4f}, "
                  f"sum={np.sum(r_vals):.4f}")

    if root_ub is None or root_ub <= INCUMBENT:
        return {'ub': root_ub or 0, 'nodes': 1, 'time': time.time()-start_time,
                'history': [(0, root_ub, 1)]}

    # B&B
    # State: (x_lo, x_hi, y_lo, y_hi, r_lo, r_hi)
    pq = [(-root_ub, 0, x_lo, x_hi, y_lo, y_hi, r_lo, r_hi, 0)]
    nc = 1
    explored = 1
    pruned_count = 0
    active = {0: root_ub}
    global_ub = root_ub
    history = [(0.0, root_ub, 1)]
    last_print = time.time()

    while pq and explored < node_limit:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break

        neg_ub, nid, xl, xh, yl, yh, rl, rh, dep = heapq.heappop(pq)
        local_ub = -neg_ub

        if nid in active:
            del active[nid]

        if local_ub <= INCUMBENT:
            pruned_count += 1
            continue

        # Select branching variable
        # Branch on the variable (among x, y, r) with the widest range,
        # weighted by the LP solution value
        best_score = -1
        best_idx = -1  # index in 0..3n-1

        for i in range(n):
            # x_i
            w = xh[i] - xl[i]
            if w > best_score:
                best_score = w
                best_idx = i
            # y_i
            w = yh[i] - yl[i]
            if w > best_score:
                best_score = w
                best_idx = n + i
            # r_i
            w = rh[i] - rl[i]
            if w > best_score:
                best_score = w
                best_idx = 2*n + i

        if best_score < 1e-8:
            continue

        # Bisect
        if best_idx < n:
            # x variable
            i = best_idx
            mid = (xl[i] + xh[i]) / 2.0
            children = []
            for half in range(2):
                cxl, cxh = xl.copy(), xh.copy()
                if half == 0:
                    cxh[i] = mid
                else:
                    cxl[i] = mid
                children.append((cxl, cxh, yl.copy(), yh.copy(), rl.copy(), rh.copy()))
        elif best_idx < 2*n:
            # y variable
            i = best_idx - n
            mid = (yl[i] + yh[i]) / 2.0
            children = []
            for half in range(2):
                cyl, cyh = yl.copy(), yh.copy()
                if half == 0:
                    cyh[i] = mid
                else:
                    cyl[i] = mid
                children.append((xl.copy(), xh.copy(), cyl, cyh, rl.copy(), rh.copy()))
        else:
            # r variable
            i = best_idx - 2*n
            mid = (rl[i] + rh[i]) / 2.0
            children = []
            for half in range(2):
                crl, crh = rl.copy(), rh.copy()
                if half == 0:
                    crh[i] = mid
                else:
                    crl[i] = mid
                children.append((xl.copy(), xh.copy(), yl.copy(), yh.copy(), crl, crh))

        for cxl, cxh, cyl, cyh, crl, crh in children:
            # Tighten bounds via containment propagation
            changed = True
            feasible = True
            for _ in range(5):
                if not changed:
                    break
                changed = False
                for ii in range(n):
                    # r <= x => r_hi <= x_hi, x_lo >= r_lo
                    new_rh = min(crh[ii], cxh[ii], cyh[ii], 1-cxl[ii], 1-cyl[ii])
                    new_xl = max(cxl[ii], crl[ii])
                    new_xh = min(cxh[ii], 1-crl[ii])
                    new_yl = max(cyl[ii], crl[ii])
                    new_yh = min(cyh[ii], 1-crl[ii])

                    if new_rh < crh[ii] - 1e-12:
                        crh[ii] = new_rh; changed = True
                    if new_xl > cxl[ii] + 1e-12:
                        cxl[ii] = new_xl; changed = True
                    if new_xh < cxh[ii] - 1e-12:
                        cxh[ii] = new_xh; changed = True
                    if new_yl > cyl[ii] + 1e-12:
                        cyl[ii] = new_yl; changed = True
                    if new_yh < cyh[ii] - 1e-12:
                        cyh[ii] = new_yh; changed = True

                    if cxl[ii] > cxh[ii] + 1e-10 or cyl[ii] > cyh[ii] + 1e-10 or \
                       crl[ii] > crh[ii] + 1e-10:
                        feasible = False
                        break

            if not feasible:
                pruned_count += 1
                continue

            child_ub, child_sol, _ = solve_lp_on_box(
                cxl, cxh, cyl, cyh, crl, crh, r_inc)
            explored += 1

            if child_ub is None or child_ub <= INCUMBENT:
                pruned_count += 1
                continue

            heapq.heappush(pq, (-child_ub, nc, cxl, cxh, cyl, cyh, crl, crh, dep+1))
            active[nc] = child_ub
            nc += 1

        # Update global UB
        if active:
            global_ub = max(active.values())
        else:
            global_ub = INCUMBENT

        now = time.time()
        if verbose and (now - last_print > 5.0):
            print(f"t={elapsed:.1f}s  nodes={explored}  pruned={pruned_count}  "
                  f"active={len(pq)}  UB={global_ub:.6f}  depth={dep}")
            last_print = now

        history.append((elapsed, global_ub, explored))

    elapsed = time.time() - start_time
    history.append((elapsed, global_ub, explored))

    if verbose:
        print(f"\nFinal: UB={global_ub:.6f}, nodes={explored}, pruned={pruned_count}, "
              f"active={len(pq)}, time={elapsed:.1f}s")

    return {
        'ub': global_ub,
        'nodes': explored,
        'pruned': pruned_count,
        'active': len(pq),
        'time': elapsed,
        'history': history,
    }


def targeted_bb(margin=0.15, time_limit=300, verbose=True):
    """
    Run B&B on a neighborhood around the incumbent.

    This doesn't prove a global UB, but demonstrates that the approach
    works and quantifies how tight the bound can get with bounded compute.
    """
    n = N
    x_inc, y_inc, r_inc = load_incumbent()

    x_lo = np.maximum(0, x_inc - margin)
    x_hi = np.minimum(1, x_inc + margin)
    y_lo = np.maximum(0, y_inc - margin)
    y_hi = np.minimum(1, y_inc + margin)
    r_lo = np.maximum(0, r_inc - margin)
    r_hi = np.minimum(0.5, r_inc + margin)

    # Tighten
    for _ in range(10):
        for i in range(n):
            r_hi[i] = min(r_hi[i], x_hi[i], y_hi[i], 1-x_lo[i], 1-y_lo[i])
            x_lo[i] = max(x_lo[i], r_lo[i])
            x_hi[i] = min(x_hi[i], 1-r_lo[i])
            y_lo[i] = max(y_lo[i], r_lo[i])
            y_hi[i] = min(y_hi[i], 1-r_lo[i])

    root_ub, root_sol, root_dc = solve_lp_on_box(
        x_lo, x_hi, y_lo, y_hi, r_lo, r_hi, r_inc)

    if verbose:
        print(f"Targeted B&B (margin={margin}):")
        print(f"  Root LP: UB = {root_ub:.6f}" if root_ub else "  Root infeasible")
        print(f"  Distance cuts: {root_dc}")
        widths = np.concatenate([x_hi-x_lo, y_hi-y_lo, r_hi-r_lo])
        print(f"  Variable widths: min={np.min(widths):.4f}, max={np.max(widths):.4f}")

    if root_ub is None or root_ub <= INCUMBENT:
        return {'ub': root_ub or 0, 'nodes': 1}

    # Run B&B from this box
    start_time = time.time()
    pq = [(-root_ub, 0, x_lo, x_hi, y_lo, y_hi, r_lo, r_hi, 0)]
    nc = 1; explored = 1; pruned_count = 0
    active = {0: root_ub}
    global_ub = root_ub
    history = [(0.0, root_ub, 1)]
    last_print = time.time()

    while pq and explored < 200000:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break

        neg_ub, nid, xl, xh, yl, yh, rl, rh, dep = heapq.heappop(pq)
        local_ub = -neg_ub

        if nid in active:
            del active[nid]
        if local_ub <= INCUMBENT:
            pruned_count += 1
            continue

        # Branch on widest variable
        best_w = -1; best_k = -1
        for i in range(n):
            for w, k in [(xh[i]-xl[i], i), (yh[i]-yl[i], n+i), (rh[i]-rl[i], 2*n+i)]:
                if w > best_w:
                    best_w = w; best_k = k

        if best_w < 1e-8:
            continue

        if best_k < n:
            idx = best_k
            mid = (xl[idx] + xh[idx]) / 2.0
            children_spec = []
            for half in range(2):
                cxl, cxh = xl.copy(), xh.copy()
                if half == 0: cxh[idx] = mid
                else: cxl[idx] = mid
                children_spec.append((cxl, cxh, yl.copy(), yh.copy(), rl.copy(), rh.copy()))
        elif best_k < 2*n:
            idx = best_k - n
            mid = (yl[idx] + yh[idx]) / 2.0
            children_spec = []
            for half in range(2):
                cyl, cyh = yl.copy(), yh.copy()
                if half == 0: cyh[idx] = mid
                else: cyl[idx] = mid
                children_spec.append((xl.copy(), xh.copy(), cyl, cyh, rl.copy(), rh.copy()))
        else:
            idx = best_k - 2*n
            mid = (rl[idx] + rh[idx]) / 2.0
            children_spec = []
            for half in range(2):
                crl, crh = rl.copy(), rh.copy()
                if half == 0: crh[idx] = mid
                else: crl[idx] = mid
                children_spec.append((xl.copy(), xh.copy(), yl.copy(), yh.copy(), crl, crh))

        for cxl, cxh, cyl, cyh, crl, crh in children_spec:
            # Propagate containment
            feasible = True
            for _ in range(5):
                for ii in range(n):
                    crh[ii] = min(crh[ii], cxh[ii], cyh[ii], 1-cxl[ii], 1-cyl[ii])
                    cxl[ii] = max(cxl[ii], crl[ii])
                    cxh[ii] = min(cxh[ii], 1-crl[ii])
                    cyl[ii] = max(cyl[ii], crl[ii])
                    cyh[ii] = min(cyh[ii], 1-crl[ii])
                    if cxl[ii] > cxh[ii]+1e-10 or cyl[ii]>cyh[ii]+1e-10 or crl[ii]>crh[ii]+1e-10:
                        feasible = False; break
                if not feasible:
                    break

            if not feasible:
                pruned_count += 1; continue

            child_ub, _, _ = solve_lp_on_box(cxl, cxh, cyl, cyh, crl, crh, r_inc)
            explored += 1

            if child_ub is None or child_ub <= INCUMBENT:
                pruned_count += 1; continue

            heapq.heappush(pq, (-child_ub, nc, cxl, cxh, cyl, cyh, crl, crh, dep+1))
            active[nc] = child_ub
            nc += 1

        if active:
            global_ub = max(active.values())
        else:
            global_ub = INCUMBENT

        now = time.time()
        if verbose and (now - last_print > 5.0):
            print(f"  t={elapsed:.1f}s  nodes={explored}  pruned={pruned_count}  "
                  f"active={len(pq)}  UB={global_ub:.6f}  depth={dep}")
            last_print = now

        history.append((elapsed, global_ub, explored))

    elapsed = time.time() - start_time
    history.append((elapsed, global_ub, explored))

    if verbose:
        print(f"  Final: UB={global_ub:.6f}, nodes={explored}, pruned={pruned_count}, "
              f"active={len(pq)}, time={elapsed:.1f}s")

    return {
        'ub': global_ub,
        'nodes': explored,
        'pruned': pruned_count,
        'active': len(pq),
        'time': elapsed,
        'history': history,
        'margin': margin,
    }


if __name__ == '__main__':
    print("=" * 70)
    print("SPATIAL B&B v4: FULL-VARIABLE LP WITH FT + DISTANCE CUTS")
    print("=" * 70)

    n = N
    ft_ub = np.sqrt(n / (2*np.sqrt(3)))
    x_inc, y_inc, r_inc = load_incumbent()

    print(f"\nReference: FT={ft_ub:.6f}, Incumbent={INCUMBENT:.10f}")

    # Phase 1: Root LP on full box
    print("\n--- Phase 1: Root LP (full box) ---")
    root_ub, root_sol, root_dc = solve_lp_on_box(
        np.zeros(n), np.ones(n), np.zeros(n), np.ones(n),
        np.zeros(n), np.full(n, 0.5), r_inc)
    print(f"Root LP UB: {root_ub:.6f}" if root_ub else "Infeasible")
    print(f"Distance cuts: {root_dc}")

    if root_sol is not None:
        r_vals = root_sol[2*n:]
        print(f"LP radii: min={np.min(r_vals):.4f}, max={np.max(r_vals):.4f}, "
              f"sum={np.sum(r_vals):.4f}")

    # Phase 2: Targeted B&B at different margins
    print("\n--- Phase 2: Targeted B&B ---")
    results = {}
    for margin in [0.02, 0.05, 0.10]:
        print()
        result = targeted_bb(margin=margin, time_limit=120, verbose=True)
        results[margin] = result

    # Phase 3: Full-box B&B
    print("\n--- Phase 3: Full-box B&B (5 min) ---")
    full_result = spatial_bb_v4(time_limit=300, node_limit=100000, verbose=True)
    results['full'] = full_result

    # Save
    results_path = os.path.join(os.path.dirname(__file__), 'results_v4.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {results_path}")
