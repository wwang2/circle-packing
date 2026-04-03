"""
v6: Constructive approach - build packings from mathematical structures.

Instead of random perturbation, construct specific packing patterns:
1. Start from optimal n=25 packing, add 1 circle in largest gap
2. Start from optimal n=27 packing, remove worst circle
3. Apollonian-style: iteratively add circles in largest gaps
4. Symmetry-based: force specific symmetry groups
5. "Biscuit" patterns: rows of circles with offsets
"""

import json
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import os
import time

WORKDIR = os.path.dirname(os.path.abspath(__file__))
N = 26
SEED = 42


def load_solution(path):
    with open(path) as f:
        data = json.load(f)
    circles = np.array(data["circles"])
    return circles[:, 0], circles[:, 1], circles[:, 2]


def save_solution(x, y, r, path):
    circles = [[float(x[i]), float(y[i]), float(r[i])] for i in range(len(x))]
    with open(path, 'w') as f:
        json.dump({"circles": circles}, f, indent=2)


def is_feasible(x, y, r, tol=1e-10):
    n = len(x)
    if np.any(r <= 0): return False
    if np.any(x - r < -tol) or np.any(1 - x - r < -tol): return False
    if np.any(y - r < -tol) or np.any(1 - y - r < -tol): return False
    for i in range(n):
        for j in range(i+1, n):
            dist2 = (x[i]-x[j])**2 + (y[i]-y[j])**2
            sr = r[i] + r[j]
            if dist2 < sr*sr - 2*tol*sr:
                return False
    return True


def fast_slsqp(x0, y0, r0, maxiter=2000):
    """SLSQP with analytical Jacobian."""
    n = len(x0)
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    n_wall = 4 * n
    n_sep = len(pairs)
    n_cons = n_wall + n_sep

    def objective(v):
        return -np.sum(v[2*n:3*n])

    def obj_jac(v):
        g = np.zeros(3*n)
        g[2*n:3*n] = -1.0
        return g

    def all_constraints(v):
        x, y, r = v[:n], v[n:2*n], v[2*n:3*n]
        c = np.empty(n_cons)
        for i in range(n):
            c[4*i] = x[i] - r[i]
            c[4*i+1] = 1.0 - x[i] - r[i]
            c[4*i+2] = y[i] - r[i]
            c[4*i+3] = 1.0 - y[i] - r[i]
        idx = n_wall
        for i, j in pairs:
            c[idx] = (x[i]-x[j])**2 + (y[i]-y[j])**2 - (r[i]+r[j])**2
            idx += 1
        return c

    def all_constraints_jac(v):
        x, y, r = v[:n], v[n:2*n], v[2*n:3*n]
        J = np.zeros((n_cons, 3*n))
        for i in range(n):
            J[4*i, i] = 1.0; J[4*i, 2*n+i] = -1.0
            J[4*i+1, i] = -1.0; J[4*i+1, 2*n+i] = -1.0
            J[4*i+2, n+i] = 1.0; J[4*i+2, 2*n+i] = -1.0
            J[4*i+3, n+i] = -1.0; J[4*i+3, 2*n+i] = -1.0
        idx = n_wall
        for i, j in pairs:
            dx = x[i] - x[j]; dy = y[i] - y[j]; sr = r[i] + r[j]
            J[idx, i] = 2*dx; J[idx, j] = -2*dx
            J[idx, n+i] = 2*dy; J[idx, n+j] = -2*dy
            J[idx, 2*n+i] = -2*sr; J[idx, 2*n+j] = -2*sr
            idx += 1
        return J

    constraints = [{'type': 'ineq', 'fun': all_constraints, 'jac': all_constraints_jac}]
    bounds = [(0.001, 0.999)]*n + [(0.001, 0.999)]*n + [(0.001, 0.5)]*n
    v0 = np.concatenate([x0, y0, r0])

    result = minimize(objective, v0, method='SLSQP', jac=obj_jac,
                      constraints=constraints, bounds=bounds,
                      options={'maxiter': maxiter, 'ftol': 1e-15})

    x, y, r = result.x[:n], result.x[n:2*n], result.x[2*n:3*n]
    return x, y, r, np.sum(r), is_feasible(x, y, r)


def find_largest_gap(x, y, r):
    """Find the largest empty circle that fits in the unit square."""
    # Sample candidate centers on a grid
    grid = np.linspace(0.02, 0.98, 50)
    best_r = 0
    best_pos = (0.5, 0.5)

    for gx in grid:
        for gy in grid:
            # Max radius at this position
            max_r = min(gx, 1-gx, gy, 1-gy)
            for i in range(len(x)):
                dist = np.sqrt((gx-x[i])**2 + (gy-y[i])**2)
                max_r = min(max_r, dist - r[i])
            if max_r > best_r:
                best_r = max_r
                best_pos = (gx, gy)

    return best_pos[0], best_pos[1], best_r


def greedy_apollonian(n, seed=42):
    """Build packing greedily: add circles in largest gaps."""
    rng = np.random.RandomState(seed)
    x, y, r = [], [], []

    # Start with one big circle
    x.append(0.5); y.append(0.5); r.append(0.4)

    while len(x) < n:
        xa, ya, ra = np.array(x), np.array(y), np.array(r)
        gx, gy, gr = find_largest_gap(xa, ya, ra)
        if gr < 0.005:
            # Fall back to random placement
            gx = rng.uniform(0.05, 0.95)
            gy = rng.uniform(0.05, 0.95)
            gr = 0.01
        x.append(gx)
        y.append(gy)
        r.append(gr * 0.95)  # slightly smaller to avoid overlaps

    return np.array(x), np.array(y), np.array(r)


def biscuit_pattern(n_rows, circles_per_row, seed=42):
    """Create biscuit (brick wall) pattern."""
    rng = np.random.RandomState(seed)
    x, y, r = [], [], []

    total = sum(circles_per_row)
    r_est = 0.5 / max(circles_per_row)

    for row_idx, n_in_row in enumerate(circles_per_row):
        y_pos = (row_idx + 0.5) / n_rows
        x_offset = 0.5 / n_in_row if row_idx % 2 == 0 else 0
        for col in range(n_in_row):
            x_pos = (col + 0.5) / n_in_row
            x.append(x_pos)
            y.append(y_pos)
            r.append(r_est * 0.9)

    x, y, r = np.array(x[:N]), np.array(y[:N]), np.array(r[:N])
    r = np.clip(r, 0.01, 0.49)
    x = np.clip(x, r+0.001, 1-r-0.001)
    y = np.clip(y, r+0.001, 1-r-0.001)
    return x, y, r


def symmetric_pattern(symmetry_type, seed=42):
    """Create packing with specific symmetry."""
    rng = np.random.RandomState(seed)

    if symmetry_type == 'd4':
        # D4 symmetry: 4 quadrants mirror each other
        # Need 26 = 4*6 + 2 (6 in each quadrant + 2 on axes)
        x, y, r = [], [], []
        # Center circle
        x.append(0.5); y.append(0.5); r.append(0.12)
        # On axes (shared by quadrants)
        x.append(0.5); y.append(0.15); r.append(0.10)  # bottom
        # 6 circles in Q1, mirror to other quadrants
        q1_x = [0.75, 0.85, 0.65, 0.90, 0.78, 0.60]
        q1_y = [0.75, 0.55, 0.60, 0.85, 0.90, 0.85]
        q1_r = [0.10, 0.09, 0.08, 0.07, 0.06, 0.05]
        for qx, qy, qr in zip(q1_x, q1_y, q1_r):
            x.append(qx); y.append(qy); r.append(qr)
            x.append(1-qx); y.append(qy); r.append(qr)  # mirror x
            x.append(qx); y.append(1-qy); r.append(qr)  # mirror y
            x.append(1-qx); y.append(1-qy); r.append(qr)  # mirror both

        x, y, r = np.array(x[:N]), np.array(y[:N]), np.array(r[:N])

    elif symmetry_type == 'c2':
        # C2 (180 degree rotation): 26 = 2*13
        x, y, r = [], [], []
        for k in range(13):
            cx = rng.uniform(0.05, 0.95)
            cy = rng.uniform(0.05, 0.95)
            cr = rng.uniform(0.04, 0.12)
            x.extend([cx, 1-cx])
            y.extend([cy, 1-cy])
            r.extend([cr, cr])

        x, y, r = np.array(x[:N]), np.array(y[:N]), np.array(r[:N])

    elif symmetry_type == 'c4':
        # C4 (90 degree rotation): need 26 = 4*6 + 2
        x, y, r = [], [], []
        # Center
        x.append(0.5); y.append(0.5); r.append(0.13)
        # On diagonal
        x.append(0.5); y.append(0.08); r.append(0.08)
        # 6 circles, rotate 4 times
        for k in range(6):
            cx = rng.uniform(0.55, 0.95)
            cy = rng.uniform(0.55, 0.95)
            cr = rng.uniform(0.04, 0.10)
            # Rotate: (cx,cy), (1-cy,cx), (1-cx,1-cy), (cy,1-cx)
            for dx, dy in [(cx, cy), (1-cy, cx), (1-cx, 1-cy), (cy, 1-cx)]:
                x.append(dx); y.append(dy); r.append(cr)

        x, y, r = np.array(x[:N]), np.array(y[:N]), np.array(r[:N])

    else:
        x, y, r = greedy_apollonian(N, seed)

    r = np.clip(r, 0.01, 0.49)
    x = np.clip(x, r+0.001, 1-r-0.001)
    y = np.clip(y, r+0.001, 1-r-0.001)
    return x, y, r


def main():
    t0 = time.time()
    rng = np.random.RandomState(SEED)

    known_path = os.path.join(WORKDIR, '..', 'topo-001', 'solution_n26.json')
    xk, yk, rk = load_solution(known_path)
    known_metric = np.sum(rk)
    print(f"Known best: {known_metric:.10f}")

    best_metric = known_metric
    best_sol = (xk.copy(), yk.copy(), rk.copy())

    results = []

    # ===== 1. Greedy Apollonian =====
    print(f"\n{'='*60}")
    print("1. Greedy Apollonian construction")
    print(f"{'='*60}")

    for seed in range(50):
        x0, y0, r0 = greedy_apollonian(N, seed=SEED+seed)
        x2, y2, r2, metric, feasible = fast_slsqp(x0, y0, r0)
        results.append(('apollonian', seed, metric, feasible))
        if feasible and metric > best_metric:
            best_metric = metric
            best_sol = (x2, y2, r2)
            print(f"  [seed={seed}] *** NEW BEST: {metric:.10f} ***")
        if seed < 3 or (feasible and metric > 2.63):
            print(f"  [seed={seed}] metric={metric:.10f}, feasible={feasible}")

    elapsed = time.time() - t0
    top = max((m for _, _, m, f in results if f), default=0)
    print(f"  Best Apollonian: {top:.10f}, time: {elapsed:.0f}s")

    # ===== 2. Biscuit patterns =====
    print(f"\n{'='*60}")
    print("2. Biscuit (brick wall) patterns")
    print(f"{'='*60}")

    biscuit_configs = [
        [5, 4, 5, 4, 4, 4],       # 5+4+5+4+4+4 = 26
        [4, 5, 4, 5, 4, 4],       # 4+5+4+5+4+4 = 26
        [5, 5, 4, 4, 4, 4],       # 5+5+4+4+4+4 = 26
        [6, 5, 5, 5, 5],          # 6+5+5+5+5 = 26
        [5, 5, 6, 5, 5],          # 5+5+6+5+5 = 26
        [4, 4, 5, 4, 5, 4],       # 4+4+5+4+5+4 = 26
        [3, 4, 3, 4, 3, 4, 3, 2], # 3+4+3+4+3+4+3+2 = 26
        [5, 4, 5, 4, 4, 4],       # different offset
        [4, 3, 4, 3, 4, 3, 3, 2], # many rows
        [6, 5, 6, 5, 4],          # wide top
    ]

    for ci, config in enumerate(biscuit_configs):
        for seed_off in range(10):
            x0, y0, r0 = biscuit_pattern(len(config), config, seed=SEED+ci*100+seed_off)
            x0 += rng.uniform(-0.02, 0.02, N)
            y0 += rng.uniform(-0.02, 0.02, N)
            r0 = np.clip(r0, 0.01, 0.49)
            x0 = np.clip(x0, r0+0.001, 1-r0-0.001)
            y0 = np.clip(y0, r0+0.001, 1-r0-0.001)

            x2, y2, r2, metric, feasible = fast_slsqp(x0, y0, r0)
            results.append(('biscuit', ci, metric, feasible))
            if feasible and metric > best_metric:
                best_metric = metric
                best_sol = (x2, y2, r2)
                print(f"  [config {ci}, seed {seed_off}] *** NEW BEST: {metric:.10f} ***")

        if (ci+1) % 5 == 0:
            elapsed = time.time() - t0
            print(f"  Config {ci+1}/{len(biscuit_configs)}, time: {elapsed:.0f}s")

    # ===== 3. Symmetry patterns =====
    print(f"\n{'='*60}")
    print("3. Symmetry-based patterns")
    print(f"{'='*60}")

    for sym in ['d4', 'c2', 'c4']:
        for seed_off in range(30):
            x0, y0, r0 = symmetric_pattern(sym, seed=SEED+seed_off)
            x2, y2, r2, metric, feasible = fast_slsqp(x0, y0, r0)
            results.append(('symmetric', sym, metric, feasible))
            if feasible and metric > best_metric:
                best_metric = metric
                best_sol = (x2, y2, r2)
                print(f"  [{sym}, seed {seed_off}] *** NEW BEST: {metric:.10f} ***")
            if seed_off < 2:
                print(f"  [{sym}, seed {seed_off}] metric={metric:.10f}, feasible={feasible}")

        elapsed = time.time() - t0
        print(f"  {sym} done, time: {elapsed:.0f}s")

    # ===== 4. Known-best with size reshuffling =====
    print(f"\n{'='*60}")
    print("4. Size reshuffling of known best")
    print(f"{'='*60}")

    # Sort circles by radius, try different size orderings at same positions
    sorted_idx = np.argsort(rk)
    positions_x = xk.copy()
    positions_y = yk.copy()

    for att in range(200):
        # Random permutation of radii assigned to positions
        perm = rng.permutation(N)
        x0 = positions_x.copy()
        y0 = positions_y.copy()
        r0 = rk[perm].copy()

        # Ensure containment
        r0 = np.clip(r0, 0.01, 0.49)
        x0 = np.clip(x0, r0+0.001, 1-r0-0.001)
        y0 = np.clip(y0, r0+0.001, 1-r0-0.001)

        x2, y2, r2, metric, feasible = fast_slsqp(x0, y0, r0)
        if feasible and metric > best_metric:
            best_metric = metric
            best_sol = (x2, y2, r2)
            print(f"  [att {att}] *** NEW BEST: {metric:.10f} ***")

        if (att+1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  Progress: {att+1}/200, best={best_metric:.10f}, time={elapsed:.0f}s")

    # ===== 5. Near-symmetric perturbations of known best =====
    print(f"\n{'='*60}")
    print("5. Symmetry-breaking perturbations")
    print(f"{'='*60}")

    # The known solution is approximately symmetric (rotation ~C1).
    # Try making it more symmetric, then re-optimize.
    for att in range(100):
        x0, y0, r0 = xk.copy(), yk.copy(), rk.copy()

        # Apply a partial symmetry transform
        mode = att % 5
        if mode == 0:
            # Reflect top-bottom, average
            alpha = rng.uniform(0.1, 0.5)
            y0 = y0 * (1-alpha) + (1-y0) * alpha
        elif mode == 1:
            # Reflect left-right, average
            alpha = rng.uniform(0.1, 0.5)
            x0 = x0 * (1-alpha) + (1-x0) * alpha
        elif mode == 2:
            # Rotate 90, average
            alpha = rng.uniform(0.1, 0.5)
            nx = x0 * (1-alpha) + y0 * alpha
            ny = y0 * (1-alpha) + (1-x0) * alpha
            x0, y0 = nx, ny
        elif mode == 3:
            # Scale radii to be more uniform
            alpha = rng.uniform(0.1, 0.5)
            mean_r = np.mean(r0)
            r0 = r0 * (1-alpha) + mean_r * alpha
        elif mode == 4:
            # Shift center of mass to (0.5, 0.5)
            alpha = rng.uniform(0.1, 0.5)
            cx, cy = np.mean(x0), np.mean(y0)
            x0 = x0 + alpha * (0.5 - cx)
            y0 = y0 + alpha * (0.5 - cy)

        r0 = np.clip(r0, 0.01, 0.49)
        x0 = np.clip(x0, r0+0.001, 1-r0-0.001)
        y0 = np.clip(y0, r0+0.001, 1-r0-0.001)

        x2, y2, r2, metric, feasible = fast_slsqp(x0, y0, r0)
        if feasible and metric > best_metric:
            best_metric = metric
            best_sol = (x2, y2, r2)
            print(f"  [att {att}] *** NEW BEST: {metric:.10f} ***")

        if (att+1) % 25 == 0:
            elapsed = time.time() - t0
            print(f"  Progress: {att+1}/100, best={best_metric:.10f}, time={elapsed:.0f}s")

    # Save
    sol_path = os.path.join(WORKDIR, 'solution_n26.json')
    save_solution(*best_sol, sol_path)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"FINAL: {best_metric:.10f} (known={known_metric:.10f})")
    print(f"Improvement: {best_metric - known_metric:.2e}")
    print(f"Time: {elapsed:.0f}s")

    # Summary of all attempts
    print(f"\nAll results by category:")
    for cat in ['apollonian', 'biscuit', 'symmetric']:
        cat_results = [m for c, _, m, f in results if c == cat and f]
        if cat_results:
            print(f"  {cat}: best={max(cat_results):.10f}, mean={np.mean(cat_results):.6f}")

    return best_metric


if __name__ == '__main__':
    main()
