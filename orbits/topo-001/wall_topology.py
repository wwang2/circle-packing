"""
Wall Topology Search: The current solution has 20 wall contacts.
What if we try different wall-contact assignments?

Current wall contacts: circles touching L,R,B,T walls.
What if a different set of circles touched the walls?

Also: try solutions where we systematically change which circles
are in corners vs edges vs interior.
"""

import json
import numpy as np
from scipy.optimize import minimize
import os
import time

WORKDIR = os.path.dirname(os.path.abspath(__file__))
N = 26

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
    for i in range(n):
        if r[i] <= 0: return False
        if x[i] - r[i] < -tol or 1 - x[i] - r[i] < -tol: return False
        if y[i] - r[i] < -tol or 1 - y[i] - r[i] < -tol: return False
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if dist < r[i] + r[j] - tol: return False
    return True

def optimize_slsqp_jac(x0, y0, r0, maxiter=15000):
    n = len(x0)
    params0 = np.concatenate([x0, y0, r0])
    constraints = []
    for i in range(n):
        def cf_xl(p, i=i): return p[i] - p[2*n+i]
        def cj_xl(p, i=i):
            g = np.zeros(3*n); g[i] = 1.0; g[2*n+i] = -1.0; return g
        constraints.append({'type': 'ineq', 'fun': cf_xl, 'jac': cj_xl})
        def cf_xr(p, i=i): return 1 - p[i] - p[2*n+i]
        def cj_xr(p, i=i):
            g = np.zeros(3*n); g[i] = -1.0; g[2*n+i] = -1.0; return g
        constraints.append({'type': 'ineq', 'fun': cf_xr, 'jac': cj_xr})
        def cf_yl(p, i=i): return p[n+i] - p[2*n+i]
        def cj_yl(p, i=i):
            g = np.zeros(3*n); g[n+i] = 1.0; g[2*n+i] = -1.0; return g
        constraints.append({'type': 'ineq', 'fun': cf_yl, 'jac': cj_yl})
        def cf_yr(p, i=i): return 1 - p[n+i] - p[2*n+i]
        def cj_yr(p, i=i):
            g = np.zeros(3*n); g[n+i] = -1.0; g[2*n+i] = -1.0; return g
        constraints.append({'type': 'ineq', 'fun': cf_yr, 'jac': cj_yr})
        def cf_rp(p, i=i): return p[2*n+i] - 1e-6
        def cj_rp(p, i=i):
            g = np.zeros(3*n); g[2*n+i] = 1.0; return g
        constraints.append({'type': 'ineq', 'fun': cf_rp, 'jac': cj_rp})
    for i in range(n):
        for j in range(i+1, n):
            def cf_no(p, i=i, j=j):
                return (p[i]-p[j])**2 + (p[n+i]-p[n+j])**2 - (p[2*n+i]+p[2*n+j])**2
            def cj_no(p, i=i, j=j):
                g = np.zeros(3*n)
                dx = p[i]-p[j]; dy = p[n+i]-p[n+j]; sr = p[2*n+i]+p[2*n+j]
                g[i]=2*dx; g[j]=-2*dx; g[n+i]=2*dy; g[n+j]=-2*dy
                g[2*n+i]=-2*sr; g[2*n+j]=-2*sr
                return g
            constraints.append({'type': 'ineq', 'fun': cf_no, 'jac': cj_no})
    result = minimize(
        lambda p: (-np.sum(p[2*n:3*n]),
                   np.concatenate([np.zeros(2*n), -np.ones(n)])),
        params0, method='SLSQP', jac=True, constraints=constraints,
        options={'maxiter': maxiter, 'ftol': 1e-15, 'disp': False}
    )
    x, y, r = result.x[:n], result.x[n:2*n], result.x[2*n:3*n]
    return x, y, r, np.sum(r), result.success

def init_with_wall_config(n, n_corners, n_bottom, n_right, n_top, n_left, seed=0):
    """Create initialization with specific wall-touching configuration.
    n_corners: circles in corners (touching 2 walls)
    n_bottom/right/top/left: circles touching respective walls (1 wall)
    Remaining are interior circles.
    """
    rng = np.random.RandomState(seed)
    x, y, r = [], [], []

    # Corner circles
    corners = [(0, 0), (1, 0), (0, 1), (1, 1)]
    for ci in range(min(n_corners, 4)):
        cx, cy = corners[ci]
        cr = 0.08 + rng.uniform(-0.02, 0.02)
        x.append(cr if cx == 0 else 1 - cr)
        y.append(cr if cy == 0 else 1 - cr)
        r.append(cr)

    # Bottom wall
    for i in range(n_bottom):
        cr = 0.10 + rng.uniform(-0.03, 0.03)
        px = rng.uniform(0.15, 0.85)
        x.append(px); y.append(cr); r.append(cr)

    # Right wall
    for i in range(n_right):
        cr = 0.10 + rng.uniform(-0.03, 0.03)
        py = rng.uniform(0.15, 0.85)
        x.append(1-cr); y.append(py); r.append(cr)

    # Top wall
    for i in range(n_top):
        cr = 0.10 + rng.uniform(-0.03, 0.03)
        px = rng.uniform(0.15, 0.85)
        x.append(px); y.append(1-cr); r.append(cr)

    # Left wall
    for i in range(n_left):
        cr = 0.10 + rng.uniform(-0.03, 0.03)
        py = rng.uniform(0.15, 0.85)
        x.append(cr); y.append(py); r.append(cr)

    # Interior circles
    n_placed = len(x)
    n_interior = n - n_placed
    for i in range(n_interior):
        cr = 0.10 + rng.uniform(-0.03, 0.04)
        x.append(rng.uniform(0.15, 0.85))
        y.append(rng.uniform(0.15, 0.85))
        r.append(cr)

    x, y, r = np.array(x[:n]), np.array(y[:n]), np.array(r[:n])
    r = np.maximum(r, 0.02)
    x = np.clip(x, r+0.001, 1-r-0.001)
    y = np.clip(y, r+0.001, 1-r-0.001)
    return x, y, r

def main():
    t0 = time.time()
    parent_path = os.path.join(WORKDIR, '..', 'nlp-001', 'solution_n26.json')
    x0, y0, r0 = load_solution(parent_path)
    parent_metric = np.sum(r0)
    print(f"Parent metric: {parent_metric:.10f}")

    best_metric = parent_metric
    best_x, best_y, best_r = x0.copy(), y0.copy(), r0.copy()

    # Current solution wall config: 4 corners, 3 bottom, 3 right, 4 top, 3 left, 3 extra wall
    # Total wall: 20. Interior: 6 (counting center + inner ring not touching wall)
    # Let's try different distributions

    configs = []
    # (n_corners, n_bottom, n_right, n_top, n_left)
    # Must have: n_corners + n_bottom + n_right + n_top + n_left <= 26
    for nc in [2, 3, 4]:
        for nb in [1, 2, 3, 4]:
            for nr in [1, 2, 3, 4]:
                for nt in [1, 2, 3, 4]:
                    for nl in [1, 2, 3, 4]:
                        total_wall = nc + nb + nr + nt + nl
                        if total_wall < 10 or total_wall > 22:
                            continue
                        n_interior = N - total_wall
                        if n_interior < 3 or n_interior > 16:
                            continue
                        configs.append((nc, nb, nr, nt, nl))

    print(f"Total wall configurations to try: {len(configs)}")

    # Sample a subset
    rng = np.random.RandomState(42)
    if len(configs) > 200:
        indices = rng.choice(len(configs), 200, replace=False)
        configs = [configs[i] for i in indices]
        print(f"Sampled down to {len(configs)}")

    for idx, (nc, nb, nr, nt, nl) in enumerate(configs):
        for seed in range(3):
            x, y, r = init_with_wall_config(N, nc, nb, nr, nt, nl, seed=seed+idx*10)
            x, y, r, m, s = optimize_slsqp_jac(x, y, r, maxiter=8000)

            if s and is_feasible(x, y, r) and m > best_metric:
                print(f"  IMPROVED: wall({nc},{nb},{nr},{nt},{nl},s{seed}) -> {m:.10f}")
                best_metric, best_x, best_y, best_r = m, x, y, r
                save_solution(best_x, best_y, best_r,
                            os.path.join(WORKDIR, 'solution_n26.json'))

        if idx % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{idx}/{len(configs)}] {elapsed:.0f}s, best={best_metric:.10f}")

    # Now try n=25 -> add one circle
    print(f"\n=== N=25 + 1 approach ===")
    # Remove each circle, optimize 25, then add the 26th
    for remove_idx in range(N):
        keep = [i for i in range(N) if i != remove_idx]
        xk, yk, rk = best_x[keep], best_y[keep], best_r[keep]

        # Optimize 25 circles
        xk, yk, rk, mk, sk = optimize_slsqp_jac(xk, yk, rk, maxiter=10000)
        if not sk:
            continue

        # Find biggest gap for 26th circle
        rng2 = np.random.RandomState(remove_idx)
        best_gap = 0
        best_pos = None
        for _ in range(5000):
            px = rng2.uniform(0.01, 0.99)
            py = rng2.uniform(0.01, 0.99)
            gap = min(px, 1-px, py, 1-py)
            for j in range(25):
                d = np.sqrt((px-xk[j])**2 + (py-yk[j])**2) - rk[j]
                gap = min(gap, d)
                if gap < 0.005:
                    break
            if gap > best_gap:
                best_gap = gap
                best_pos = (px, py)

        if best_pos and best_gap > 0.005:
            x26 = np.append(xk, best_pos[0])
            y26 = np.append(yk, best_pos[1])
            r26 = np.append(rk, best_gap * 0.99)

            x26, y26, r26, m26, s26 = optimize_slsqp_jac(x26, y26, r26, maxiter=10000)
            if s26 and is_feasible(x26, y26, r26) and m26 > best_metric:
                print(f"  IMPROVED: remove+add [{remove_idx}] -> {m26:.10f}")
                best_metric, best_x, best_y, best_r = m26, x26, y26, r26
                save_solution(best_x, best_y, best_r,
                            os.path.join(WORKDIR, 'solution_n26.json'))

    elapsed = time.time() - t0
    print(f"\n=== FINAL ===")
    print(f"Parent:  {parent_metric:.10f}")
    print(f"Best:    {best_metric:.10f}")
    print(f"Delta:   {best_metric - parent_metric:.2e}")
    print(f"Time:    {elapsed:.0f}s")

if __name__ == '__main__':
    main()
