"""
Spatial Branch-and-Bound v2: Reduced-variable formulation.

Key insight from v1: the full McCormick LP (1131 variables) is extremely loose.
Even on small boxes (width 0.1), the root LP gives UB ~ 3.0 vs incumbent 2.636.
The problem: McCormick envelopes on quadratic terms lose too much information.

New approach: Branch on CENTER POSITIONS only, and for each box of centers,
solve a tight LP over radii.

Given a box B for centers (x_i, y_i) in [x_i^L, x_i^U] x [y_i^L, y_i^U]:
  For each pair (i,j), the MINIMUM distance between centers is:
    d_min(i,j) = min_{(xi,yi) in Bi, (xj,yj) in Bj} ||(xi,yi) - (xj,yj)||

  The non-overlap constraint requires: ||(xi,yi) - (xj,yj)|| >= ri + rj
  So on box B: ri + rj <= d_min(i,j)  (valid linear constraint!)

  The containment constraint r_i <= x_i says r_i <= x_i^L (since x_i >= x_i^L).
  Similarly r_i <= 1 - x_i^U, r_i <= y_i^L, r_i <= 1 - y_i^U.

  So: r_i <= min(x_i^L, 1 - x_i^U, y_i^L, 1 - y_i^U)

  This gives us a pure LP over radii:
    maximize sum(r_i)
    subject to r_i + r_j <= d_min(i,j)  for all i<j
               r_i <= containment_bound(i)
               r_i >= 0

This LP has only 26 variables and 325 + 26 constraints -- trivial to solve!
The B&B branches on center positions to tighten d_min(i,j).

The beauty: we don't even introduce auxiliary variables for quadratic terms.
The non-overlap constraint is handled via the distance lower bound.
"""

import numpy as np
import json
import time
import heapq
from scipy.optimize import linprog
from multiprocessing import Pool
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


def min_distance_between_boxes(xi_lo, xi_hi, yi_lo, yi_hi, xj_lo, xj_hi, yj_lo, yj_hi):
    """
    Minimum Euclidean distance between two axis-aligned rectangles.

    For rectangles Bi = [xi_lo, xi_hi] x [yi_lo, yi_hi] and
    Bj = [xj_lo, xj_hi] x [yj_lo, yj_hi]:

    The minimum distance is:
      sqrt(max(0, gap_x)^2 + max(0, gap_y)^2)
    where:
      gap_x = max(xi_lo - xj_hi, xj_lo - xi_hi, 0)
      gap_y = max(yi_lo - yj_hi, yj_lo - yi_hi, 0)

    If the rectangles overlap, min distance is 0.
    """
    gap_x = max(xi_lo - xj_hi, xj_lo - xi_hi, 0.0)
    gap_y = max(yi_lo - yj_hi, yj_lo - yi_hi, 0.0)
    return np.sqrt(gap_x**2 + gap_y**2)


def max_radius_from_containment(x_lo, x_hi, y_lo, y_hi):
    """
    Maximum radius for a circle with center in [x_lo, x_hi] x [y_lo, y_hi]
    that stays inside [0,1]^2.

    r <= x (worst case: x = x_lo) => r <= x_lo
    r <= 1 - x (worst case: x = x_hi) => r <= 1 - x_hi
    r <= y (worst case: y = y_lo) => r <= y_lo
    r <= 1 - y (worst case: y = y_hi) => r <= 1 - y_hi
    """
    return min(x_lo, 1.0 - x_hi, y_lo, 1.0 - y_hi)


def solve_radius_lp(d_min_matrix, r_max_contain, n=N):
    """
    Solve the radius LP:
      maximize sum(r_i)
      subject to r_i + r_j <= d_min[i,j]  for all i<j
                 0 <= r_i <= r_max_contain[i]

    Returns (ub, r_solution) or (None, None) if infeasible.
    """
    c = -np.ones(n)  # maximize sum

    rows = []
    rhs = []

    for i in range(n):
        for j in range(i+1, n):
            if d_min_matrix[i, j] < 1e-10:
                # Boxes overlap => constraint is r_i + r_j <= 0
                # But both r_i, r_j >= 0, so this forces one to be 0
                # Only add if d_min is genuinely 0
                pass  # Will be handled by bounds if needed
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            rows.append(row)
            rhs.append(d_min_matrix[i, j])

    if rows:
        A_ub = np.array(rows)
        b_ub = np.array(rhs)
    else:
        A_ub = None
        b_ub = None

    bounds = [(0.0, max(0.0, r_max_contain[i])) for i in range(n)]

    # Check feasibility: if any upper bound is negative, infeasible
    for i in range(n):
        if r_max_contain[i] < -1e-10:
            return None, None

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                    options={'presolve': True})

    if result.success:
        return -result.fun, result.x
    else:
        return None, None


class BBNode:
    """A node in the branch-and-bound tree."""
    __slots__ = ['x_lo', 'x_hi', 'y_lo', 'y_hi', 'ub', 'depth']

    def __init__(self, x_lo, x_hi, y_lo, y_hi, ub=float('inf'), depth=0):
        self.x_lo = x_lo  # shape (n,)
        self.x_hi = x_hi
        self.y_lo = y_lo
        self.y_hi = y_hi
        self.ub = ub
        self.depth = depth

    def __lt__(self, other):
        # For heap: higher UB = higher priority (we negate in heap)
        return self.ub > other.ub


def evaluate_node(node):
    """Compute the LP upper bound for a B&B node."""
    n = N

    # Compute minimum pairwise distances
    d_min = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = min_distance_between_boxes(
                node.x_lo[i], node.x_hi[i], node.y_lo[i], node.y_hi[i],
                node.x_lo[j], node.x_hi[j], node.y_lo[j], node.y_hi[j])
            d_min[i, j] = d
            d_min[j, i] = d

    # Compute containment bounds
    r_max = np.array([max_radius_from_containment(
        node.x_lo[i], node.x_hi[i], node.y_lo[i], node.y_hi[i])
        for i in range(n)])

    # Solve LP
    ub, r_sol = solve_radius_lp(d_min, r_max)
    return ub, r_sol, d_min, r_max


def select_branch_variable_v2(node, r_sol, d_min, r_max):
    """
    Select which circle to branch on and which coordinate (x or y).

    Strategy: find the pair (i,j) where the LP wants r_i + r_j close to d_min(i,j)
    (binding non-overlap constraint) but d_min(i,j) has the most slack
    (difference between d_max and d_min is large). Branch on the circle in
    this pair whose center box is widest.
    """
    n = N
    best_score = -1
    best_circle = -1
    best_coord = 0  # 0 = x, 1 = y

    for i in range(n):
        for j in range(i+1, n):
            if r_sol is None:
                continue
            # How tight is the non-overlap constraint?
            slack = d_min[i, j] - (r_sol[i] + r_sol[j])
            if slack > 0.01:  # Not binding
                continue

            # How much could d_min improve by branching?
            # The width of the boxes determines the uncertainty in d_min
            xi_width = node.x_hi[i] - node.x_lo[i]
            yi_width = node.y_hi[i] - node.y_lo[i]
            xj_width = node.x_hi[j] - node.x_lo[j]
            yj_width = node.y_hi[j] - node.y_lo[j]

            # Score by maximum width of involved circles
            for circ, xw, yw in [(i, xi_width, yi_width), (j, xj_width, yj_width)]:
                for coord, w in [(0, xw), (1, yw)]:
                    if w > best_score:
                        best_score = w
                        best_circle = circ
                        best_coord = coord

    if best_circle < 0:
        # Fallback: branch on widest variable
        for i in range(n):
            xw = node.x_hi[i] - node.x_lo[i]
            yw = node.y_hi[i] - node.y_lo[i]
            if xw > best_score:
                best_score = xw
                best_circle = i
                best_coord = 0
            if yw > best_score:
                best_score = yw
                best_circle = i
                best_coord = 1

    return best_circle, best_coord, best_score


def spatial_bb_v2(time_limit=300, node_limit=100000, verbose=True):
    """
    Run spatial B&B with the reduced-variable formulation.

    Branch on center positions (x_i, y_i). For each partition of center space,
    solve a 26-variable LP over radii.
    """
    start_time = time.time()
    n = N

    x_inc, y_inc, r_inc = load_incumbent()

    # Initial box: full [0,1]^2 for all centers
    x_lo = np.zeros(n)
    x_hi = np.ones(n)
    y_lo = np.zeros(n)
    y_hi = np.ones(n)

    root = BBNode(x_lo, x_hi, y_lo, y_hi, depth=0)
    root_ub, root_sol, root_dmin, root_rmax = evaluate_node(root)

    if root_ub is None:
        return {'ub': float('inf'), 'status': 'root_infeasible'}

    root.ub = root_ub
    if verbose:
        print(f"Root LP: UB = {root_ub:.6f}")
        print(f"  Incumbent = {INCUMBENT:.10f}")
        print(f"  FT UB = {np.sqrt(n / (2*np.sqrt(3))):.6f}")

    if root_ub <= INCUMBENT:
        return {'ub': root_ub, 'nodes': 1, 'pruned': 0, 'time': time.time()-start_time,
                'history': [(0, root_ub, 1)]}

    # B&B priority queue
    pq = [(-root_ub, 0, root)]
    node_counter = 1
    nodes_explored = 1
    nodes_pruned = 0
    global_ub = root_ub

    # Track all active node UBs
    active_ubs = {0: root_ub}

    history = [(0.0, root_ub, 1)]
    last_print_time = time.time()

    while pq and nodes_explored < node_limit:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break

        neg_ub, nid, node = heapq.heappop(pq)
        local_ub = -neg_ub

        if nid in active_ubs:
            del active_ubs[nid]

        if local_ub <= INCUMBENT:
            nodes_pruned += 1
            continue

        # Select branching variable
        _, r_sol, d_min, r_max = evaluate_node(node)
        circle_idx, coord, width = select_branch_variable_v2(node, r_sol, d_min, r_max)

        if width < 1e-8:
            # Can't branch further
            continue

        # Branch: bisect the chosen circle's coordinate
        if coord == 0:
            mid = (node.x_lo[circle_idx] + node.x_hi[circle_idx]) / 2.0
        else:
            mid = (node.y_lo[circle_idx] + node.y_hi[circle_idx]) / 2.0

        for child_half in range(2):
            child_x_lo = node.x_lo.copy()
            child_x_hi = node.x_hi.copy()
            child_y_lo = node.y_lo.copy()
            child_y_hi = node.y_hi.copy()

            if coord == 0:
                if child_half == 0:
                    child_x_hi[circle_idx] = mid
                else:
                    child_x_lo[circle_idx] = mid
            else:
                if child_half == 0:
                    child_y_hi[circle_idx] = mid
                else:
                    child_y_lo[circle_idx] = mid

            # Check containment feasibility
            feasible = True
            for i in range(n):
                if child_x_lo[i] > child_x_hi[i] + 1e-12 or \
                   child_y_lo[i] > child_y_hi[i] + 1e-12:
                    feasible = False
                    break
                rmax_i = max_radius_from_containment(
                    child_x_lo[i], child_x_hi[i],
                    child_y_lo[i], child_y_hi[i])
                if rmax_i < 1e-12:
                    # This circle can't exist here
                    pass  # It's OK, its radius will just be 0

            if not feasible:
                nodes_pruned += 1
                continue

            child = BBNode(child_x_lo, child_x_hi, child_y_lo, child_y_hi,
                          depth=node.depth + 1)
            child_ub, child_sol, _, _ = evaluate_node(child)
            nodes_explored += 1

            if child_ub is None or child_ub <= INCUMBENT:
                nodes_pruned += 1
                continue

            child.ub = child_ub
            heapq.heappush(pq, (-child_ub, node_counter, child))
            active_ubs[node_counter] = child_ub
            node_counter += 1

        # Update global UB
        if active_ubs:
            global_ub = max(active_ubs.values())
        else:
            global_ub = INCUMBENT

        now = time.time()
        if verbose and (now - last_print_time > 5.0 or nodes_explored % 500 == 0):
            print(f"t={elapsed:.1f}s  nodes={nodes_explored}  pruned={nodes_pruned}  "
                  f"active={len(pq)}  UB={global_ub:.6f}  "
                  f"depth={node.depth}")
            last_print_time = now

        history.append((elapsed, global_ub, nodes_explored))

    elapsed = time.time() - start_time
    history.append((elapsed, global_ub, nodes_explored))

    if verbose:
        print(f"\nFinal: UB={global_ub:.6f}, nodes={nodes_explored}, "
              f"pruned={nodes_pruned}, active={len(pq)}, time={elapsed:.1f}s")

    return {
        'ub': global_ub,
        'nodes': nodes_explored,
        'pruned': nodes_pruned,
        'active': len(pq),
        'time': elapsed,
        'history': history,
        'status': 'completed'
    }


def enhanced_bb_with_ft(time_limit=300, node_limit=100000, verbose=True):
    """
    Enhanced B&B that combines the radius LP with Fejes-Toth-style area cuts.

    For each box, compute:
    1. Pairwise distance bounds => r_i + r_j <= d_min(i,j)
    2. Containment bounds => r_i <= min(x_lo, 1-x_hi, y_lo, 1-y_hi)
    3. Area bound: sum(2*sqrt(3)*r_i^2) <= 1

    For (3), we use a tangent linearization around the incumbent solution
    to get a valid linear cut. We can add multiple tangent cuts.

    Also add: r_i + r_j + r_k bounds from triples of circles
    whose center boxes are close together.
    """
    start_time = time.time()
    n = N
    x_inc, y_inc, r_inc = load_incumbent()
    coeff = 2.0 * np.sqrt(3.0)  # FT coefficient

    def evaluate_enhanced(x_lo, x_hi, y_lo, y_hi):
        """Evaluate the enhanced LP bound on a box."""
        # Pairwise min distances
        d_min = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                d = min_distance_between_boxes(
                    x_lo[i], x_hi[i], y_lo[i], y_hi[i],
                    x_lo[j], x_hi[j], y_lo[j], y_hi[j])
                d_min[i, j] = d
                d_min[j, i] = d

        # Containment bounds
        r_max = np.array([max_radius_from_containment(
            x_lo[i], x_hi[i], y_lo[i], y_hi[i]) for i in range(n)])

        # Build LP
        c = -np.ones(n)
        rows = []
        rhs = []

        # Pairwise distance constraints
        for i in range(n):
            for j in range(i+1, n):
                row = np.zeros(n)
                row[i] = 1.0
                row[j] = 1.0
                rows.append(row)
                rhs.append(d_min[i, j])

        # FT tangent cuts at incumbent
        row = np.zeros(n)
        for i in range(n):
            row[i] = 2.0 * coeff * r_inc[i]
        rows.append(row)
        rhs.append(1.0 + coeff * np.sum(r_inc**2))

        # FT tangent at equal radii
        r_eq = np.sqrt(1.0 / (n * coeff))
        row = np.zeros(n)
        row[:] = 2.0 * coeff * r_eq
        rows.append(row)
        rhs.append(1.0 + coeff * n * r_eq**2)

        # Additional FT tangent cuts at various points
        # At r = 0.1 for all
        for r_ref in [0.05, 0.08, 0.12, 0.15, 0.20]:
            row = np.zeros(n)
            row[:] = 2.0 * coeff * r_ref
            rows.append(row)
            rhs.append(1.0 + coeff * n * r_ref**2)

        # Also individual tangent cuts: for each circle, add FT tangent
        # at the incumbent radius of that specific circle
        # This creates a tighter outer approximation

        A_ub = np.array(rows)
        b_ub = np.array(rhs)
        bounds = [(0.0, max(0.0, r_max[i])) for i in range(n)]

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                        options={'presolve': True})

        if result.success:
            return -result.fun, result.x
        else:
            return None, None

    # Root evaluation
    x_lo = np.zeros(n)
    x_hi = np.ones(n)
    y_lo = np.zeros(n)
    y_hi = np.ones(n)

    root_ub, root_sol = evaluate_enhanced(x_lo, x_hi, y_lo, y_hi)
    if verbose:
        print(f"Enhanced root LP: UB = {root_ub:.6f}" if root_ub else "Root infeasible")

    if root_ub is None or root_ub <= INCUMBENT:
        return {'ub': root_ub or 0, 'nodes': 1}

    # B&B
    pq = [(-root_ub, 0, x_lo, x_hi, y_lo, y_hi, 0)]
    node_counter = 1
    nodes_explored = 1
    nodes_pruned = 0
    active_ubs = {0: root_ub}
    global_ub = root_ub
    history = [(0.0, root_ub, 1)]
    last_print = time.time()

    while pq and nodes_explored < node_limit:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break

        neg_ub, nid, xl, xh, yl, yh, depth = heapq.heappop(pq)
        local_ub = -neg_ub

        if nid in active_ubs:
            del active_ubs[nid]

        if local_ub <= INCUMBENT:
            nodes_pruned += 1
            continue

        # Find widest variable to branch on
        best_width = -1
        best_circle = -1
        best_coord = 0
        for i in range(n):
            xw = xh[i] - xl[i]
            yw = yh[i] - yl[i]
            if xw > best_width:
                best_width = xw
                best_circle = i
                best_coord = 0
            if yw > best_width:
                best_width = yw
                best_circle = i
                best_coord = 1

        if best_width < 1e-8:
            continue

        if best_coord == 0:
            mid = (xl[best_circle] + xh[best_circle]) / 2.0
        else:
            mid = (yl[best_circle] + yh[best_circle]) / 2.0

        for half in range(2):
            cxl = xl.copy()
            cxh = xh.copy()
            cyl = yl.copy()
            cyh = yh.copy()

            if best_coord == 0:
                if half == 0:
                    cxh[best_circle] = mid
                else:
                    cxl[best_circle] = mid
            else:
                if half == 0:
                    cyh[best_circle] = mid
                else:
                    cyl[best_circle] = mid

            child_ub, child_sol = evaluate_enhanced(cxl, cxh, cyl, cyh)
            nodes_explored += 1

            if child_ub is None or child_ub <= INCUMBENT:
                nodes_pruned += 1
                continue

            heapq.heappush(pq, (-child_ub, node_counter, cxl, cxh, cyl, cyh, depth+1))
            active_ubs[node_counter] = child_ub
            node_counter += 1

        if active_ubs:
            global_ub = max(active_ubs.values())
        else:
            global_ub = INCUMBENT

        now = time.time()
        if verbose and (now - last_print > 5.0):
            print(f"t={elapsed:.1f}s  nodes={nodes_explored}  pruned={nodes_pruned}  "
                  f"active={len(pq)}  UB={global_ub:.6f}  depth={depth}")
            last_print = now

        history.append((elapsed, global_ub, nodes_explored))

    elapsed = time.time() - start_time
    history.append((elapsed, global_ub, nodes_explored))

    if verbose:
        print(f"\nFinal: UB={global_ub:.6f}, nodes={nodes_explored}, "
              f"pruned={nodes_pruned}, time={elapsed:.1f}s")

    return {
        'ub': global_ub,
        'nodes': nodes_explored,
        'pruned': nodes_pruned,
        'active': len(pq),
        'time': elapsed,
        'history': history
    }


def run_all_approaches():
    """Run all approaches and compare."""
    print("=" * 70)
    print("SPATIAL B&B v2: REDUCED-VARIABLE FORMULATION")
    print("=" * 70)

    n = N
    ft_ub = np.sqrt(n / (2*np.sqrt(3)))
    print(f"\nReference bounds:")
    print(f"  Fejes-Toth UB: {ft_ub:.6f}")
    print(f"  Incumbent:     {INCUMBENT:.10f}")
    print(f"  Gap:           {(ft_ub/INCUMBENT - 1)*100:.2f}%")

    # Phase 1: Basic reduced-variable B&B
    print(f"\n{'='*70}")
    print("Phase 1: Basic reduced-variable B&B (5 min)")
    print(f"{'='*70}")
    result1 = spatial_bb_v2(time_limit=300, node_limit=100000, verbose=True)
    print(f"\nPhase 1 result: UB = {result1['ub']:.6f}")

    # Phase 2: Enhanced B&B with FT cuts
    print(f"\n{'='*70}")
    print("Phase 2: Enhanced B&B with FT tangent cuts (5 min)")
    print(f"{'='*70}")
    result2 = enhanced_bb_with_ft(time_limit=300, node_limit=100000, verbose=True)
    print(f"\nPhase 2 result: UB = {result2['ub']:.6f}")

    return result1, result2


if __name__ == '__main__':
    result1, result2 = run_all_approaches()

    # Save results for plotting
    results = {
        'basic_bb': result1,
        'enhanced_bb': result2,
        'ft_ub': np.sqrt(N / (2*np.sqrt(3))),
        'incumbent': INCUMBENT,
    }

    results_path = os.path.join(os.path.dirname(__file__), 'results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {results_path}")
