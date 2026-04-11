"""
High-precision optimization with analytical Jacobian.
The key idea: provide analytical gradients to SLSQP for much faster convergence,
and try many more initializations.
"""

import json
import numpy as np
from scipy.optimize import minimize
import os
import time
from itertools import combinations

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
            if dist < r[i] + r[j] - tol:
                return False
    return True

def optimize_jac(x0, y0, r0, maxiter=15000):
    """SLSQP with analytical objective Jacobian."""
    n = len(x0)
    params0 = np.concatenate([x0, y0, r0])

    def obj_and_jac(p):
        val = -np.sum(p[2*n:3*n])
        grad = np.zeros(3*n)
        grad[2*n:3*n] = -1.0
        return val, grad

    constraints = []
    for i in range(n):
        # x_i - r_i >= 0
        def cf_xl(p, i=i): return p[i] - p[2*n+i]
        def cj_xl(p, i=i):
            g = np.zeros(3*n)
            g[i] = 1.0; g[2*n+i] = -1.0
            return g
        constraints.append({'type': 'ineq', 'fun': cf_xl, 'jac': cj_xl})

        # 1 - x_i - r_i >= 0
        def cf_xr(p, i=i): return 1 - p[i] - p[2*n+i]
        def cj_xr(p, i=i):
            g = np.zeros(3*n)
            g[i] = -1.0; g[2*n+i] = -1.0
            return g
        constraints.append({'type': 'ineq', 'fun': cf_xr, 'jac': cj_xr})

        # y_i - r_i >= 0
        def cf_yl(p, i=i): return p[n+i] - p[2*n+i]
        def cj_yl(p, i=i):
            g = np.zeros(3*n)
            g[n+i] = 1.0; g[2*n+i] = -1.0
            return g
        constraints.append({'type': 'ineq', 'fun': cf_yl, 'jac': cj_yl})

        # 1 - y_i - r_i >= 0
        def cf_yr(p, i=i): return 1 - p[n+i] - p[2*n+i]
        def cj_yr(p, i=i):
            g = np.zeros(3*n)
            g[n+i] = -1.0; g[2*n+i] = -1.0
            return g
        constraints.append({'type': 'ineq', 'fun': cf_yr, 'jac': cj_yr})

        # r_i >= eps
        def cf_rp(p, i=i): return p[2*n+i] - 1e-6
        def cj_rp(p, i=i):
            g = np.zeros(3*n)
            g[2*n+i] = 1.0
            return g
        constraints.append({'type': 'ineq', 'fun': cf_rp, 'jac': cj_rp})

    for i in range(n):
        for j in range(i+1, n):
            # (x_i-x_j)^2 + (y_i-y_j)^2 - (r_i+r_j)^2 >= 0
            def cf_no(p, i=i, j=j):
                return (p[i]-p[j])**2 + (p[n+i]-p[n+j])**2 - (p[2*n+i]+p[2*n+j])**2
            def cj_no(p, i=i, j=j):
                g = np.zeros(3*n)
                dx = p[i] - p[j]
                dy = p[n+i] - p[n+j]
                sr = p[2*n+i] + p[2*n+j]
                g[i] = 2*dx
                g[j] = -2*dx
                g[n+i] = 2*dy
                g[n+j] = -2*dy
                g[2*n+i] = -2*sr
                g[2*n+j] = -2*sr
                return g
            constraints.append({'type': 'ineq', 'fun': cf_no, 'jac': cj_no})

    result = minimize(
        obj_and_jac,
        params0,
        method='SLSQP',
        jac=True,
        constraints=constraints,
        options={'maxiter': maxiter, 'ftol': 1e-15, 'disp': False}
    )

    x = result.x[:n]
    y = result.x[n:2*n]
    r = result.x[2*n:3*n]
    return x, y, r, np.sum(r), result.success

def init_ring_variant(n, n1, n2, seed=0):
    """Ring with configurable inner/outer sizes. n1=inner ring, n2=outer ring."""
    rng = np.random.RandomState(seed)
    n_corners = 4
    n_extra = n - 1 - n1 - n2 - n_corners

    x, y, r = [], [], []

    # Center
    x.append(0.5 + rng.normal(0, 0.005))
    y.append(0.5 + rng.normal(0, 0.005))
    r.append(0.137 + rng.uniform(-0.005, 0.005))

    # Inner ring
    r1_orbit = 0.25 + rng.uniform(-0.03, 0.03)
    r1_size = 0.5 / (n1 + 2) + rng.uniform(-0.01, 0.01)
    for i in range(n1):
        angle = 2*np.pi*i/n1 + rng.uniform(-0.1, 0.1)
        x.append(0.5 + r1_orbit*np.cos(angle))
        y.append(0.5 + r1_orbit*np.sin(angle))
        r.append(r1_size + rng.uniform(-0.01, 0.01))

    # Outer ring
    r2_orbit = 0.42 + rng.uniform(-0.04, 0.04)
    r2_size = 0.5 / (n2 + 3) + rng.uniform(-0.01, 0.01)
    for i in range(n2):
        angle = 2*np.pi*i/n2 + rng.uniform(-0.1, 0.1)
        px = 0.5 + r2_orbit*np.cos(angle)
        py = 0.5 + r2_orbit*np.sin(angle)
        x.append(px); y.append(py)
        r.append(r2_size + rng.uniform(-0.01, 0.01))

    # Corners
    for cx, cy in [(0.09, 0.09), (0.91, 0.09), (0.09, 0.91), (0.91, 0.91)]:
        x.append(cx + rng.normal(0, 0.01))
        y.append(cy + rng.normal(0, 0.01))
        r.append(0.085 + rng.uniform(-0.015, 0.015))

    # Extra circles
    for i in range(max(0, n_extra)):
        x.append(rng.uniform(0.1, 0.9))
        y.append(rng.uniform(0.1, 0.9))
        r.append(0.06 + rng.uniform(-0.02, 0.02))

    x, y, r = np.array(x[:n]), np.array(y[:n]), np.array(r[:n])
    r = np.maximum(r, 0.015)
    x = np.clip(x, r+0.001, 1-r-0.001)
    y = np.clip(y, r+0.001, 1-r-0.001)
    return x, y, r

def init_edge_packing(n, seed=0):
    """Pack circles along edges first, then fill interior."""
    rng = np.random.RandomState(seed)

    x, y, r = [], [], []

    # Edge circles: ~12-16 circles along the 4 edges
    n_per_edge = [3, 3, 3, 3]  # baseline
    extra_edge = rng.randint(0, 4)
    n_per_edge[extra_edge] += 1

    for edge in range(4):
        ne = n_per_edge[edge]
        for i in range(ne):
            t = (i + 0.5) / ne
            if edge == 0:    # bottom
                px, py = t, 0.0
            elif edge == 1:  # right
                px, py = 1.0, t
            elif edge == 2:  # top
                px, py = t, 1.0
            else:            # left
                px, py = 0.0, t

            r_val = 0.5 / (ne + 1) * rng.uniform(0.8, 1.0)
            px = np.clip(px, r_val + 0.001, 1 - r_val - 0.001)
            py = np.clip(py, r_val + 0.001, 1 - r_val - 0.001)
            x.append(px); y.append(py); r.append(r_val)

    # Interior circles
    n_interior = n - len(x)
    for i in range(n_interior):
        x.append(rng.uniform(0.15, 0.85))
        y.append(rng.uniform(0.15, 0.85))
        r.append(rng.uniform(0.06, 0.14))

    x, y, r = np.array(x[:n]), np.array(y[:n]), np.array(r[:n])
    r = np.maximum(r, 0.015)
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

    # ====== Phase 0: Polish parent with Jacobian ======
    print("\n=== Phase 0: Polish parent with analytical Jacobian ===")
    x1, y1, r1, m1, s1 = optimize_jac(x0, y0, r0, maxiter=20000)
    if s1 and is_feasible(x1, y1, r1) and m1 > best_metric:
        print(f"  Polish improved: {best_metric:.10f} -> {m1:.10f}")
        best_metric, best_x, best_y, best_r = m1, x1, y1, r1
    else:
        print(f"  No improvement from polish: {m1:.10f}")

    # ====== Phase 1: Systematic ring variants ======
    print(f"\n=== Phase 1: Ring Topology Variants ===")
    count = 0
    for n1 in range(5, 11):      # inner ring: 5-10 circles
        for n2 in range(7, 15):   # outer ring: 7-14 circles
            if n1 + n2 + 5 > N + 3:  # need room for center + corners + extra
                continue
            for seed in range(5):
                x, y, r = init_ring_variant(N, n1, n2, seed=seed*100 + n1*10 + n2)
                x, y, r, m, s = optimize_jac(x, y, r, maxiter=8000)

                if s and is_feasible(x, y, r) and m > best_metric:
                    print(f"  IMPROVED: ring({n1},{n2},s{seed}) -> {m:.10f} (+{m-parent_metric:.2e})")
                    best_metric, best_x, best_y, best_r = m, x, y, r
                    save_solution(best_x, best_y, best_r,
                                os.path.join(WORKDIR, 'solution_n26.json'))

                count += 1
                if count % 50 == 0:
                    elapsed = time.time() - t0
                    print(f"  [{count}] {elapsed:.0f}s, best={best_metric:.10f}")

    # ====== Phase 2: Edge packing variants ======
    print(f"\n=== Phase 2: Edge Packing ===")
    for seed in range(30):
        x, y, r = init_edge_packing(N, seed=seed)
        x, y, r, m, s = optimize_jac(x, y, r, maxiter=8000)
        if s and is_feasible(x, y, r) and m > best_metric:
            print(f"  IMPROVED: edge(s{seed}) -> {m:.10f}")
            best_metric, best_x, best_y, best_r = m, x, y, r
        if seed % 10 == 0:
            print(f"  Edge seed={seed}: {m:.8f}, best={best_metric:.10f}")

    # ====== Phase 3: Aggressive basin hopping with Jacobian ======
    print(f"\n=== Phase 3: Jacobian Basin Hopping ===")
    rng = np.random.RandomState(7777)

    for hop in range(200):
        x2, y2, r2 = best_x.copy(), best_y.copy(), best_r.copy()

        ptype = rng.randint(0, 8)
        if ptype == 0:
            # Large random displacement of 1-2 circles
            for _ in range(rng.randint(1, 3)):
                i = rng.randint(N)
                x2[i] = rng.uniform(0.05, 0.95)
                y2[i] = rng.uniform(0.05, 0.95)
                r2[i] = rng.uniform(0.03, 0.15)
        elif ptype == 1:
            # Swap positions of 2 non-adjacent circles
            i, j = rng.choice(N, 2, replace=False)
            x2[i], x2[j] = x2[j], x2[i]
            y2[i], y2[j] = y2[j], y2[i]
        elif ptype == 2:
            # Global noise
            scale = rng.choice([0.02, 0.05, 0.1, 0.15])
            x2 += rng.normal(0, scale, N)
            y2 += rng.normal(0, scale, N)
            r2 *= (1 + rng.normal(0, scale*0.3, N))
        elif ptype == 3:
            # Reflect half the circles
            k = rng.randint(N//4, 3*N//4)
            idxs = rng.choice(N, k, replace=False)
            axis = rng.randint(0, 2)
            if axis == 0:
                x2[idxs] = 1 - x2[idxs]
            else:
                y2[idxs] = 1 - y2[idxs]
        elif ptype == 4:
            # Cycle 3 circles
            i, j, k = rng.choice(N, 3, replace=False)
            x2[i], x2[j], x2[k] = x2[k], x2[i], x2[j]
            y2[i], y2[j], y2[k] = y2[k], y2[i], y2[j]
        elif ptype == 5:
            # Scale subset radii up/down
            k = rng.randint(2, N//2)
            idxs = rng.choice(N, k, replace=False)
            factor = rng.choice([0.7, 0.8, 0.9, 1.1, 1.2, 1.3])
            r2[idxs] *= factor
        elif ptype == 6:
            # Rotate entire packing
            angle = rng.uniform(-0.5, 0.5)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            cx, cy = 0.5, 0.5
            for i in range(N):
                dx, dy = x2[i]-cx, y2[i]-cy
                x2[i] = cx + cos_a*dx - sin_a*dy
                y2[i] = cy + sin_a*dx + cos_a*dy
        elif ptype == 7:
            # Move small circle to biggest gap
            i = rng.choice(np.argsort(r2)[:6])  # pick from smallest
            # Find gaps: sample random points
            best_gap = 0
            for _ in range(200):
                px = rng.uniform(0.05, 0.95)
                py = rng.uniform(0.05, 0.95)
                gap = min(px, 1-px, py, 1-py)
                for j in range(N):
                    if j == i: continue
                    d = np.sqrt((px-x2[j])**2 + (py-y2[j])**2) - r2[j]
                    gap = min(gap, d)
                if gap > best_gap:
                    best_gap = gap
                    x2[i] = px
                    y2[i] = py
                    r2[i] = gap * 0.95

        r2 = np.maximum(r2, 0.005)
        x2 = np.clip(x2, r2+0.001, 1-r2-0.001)
        y2 = np.clip(y2, r2+0.001, 1-r2-0.001)

        x2, y2, r2, m2, s2 = optimize_jac(x2, y2, r2, maxiter=8000)

        if s2 and is_feasible(x2, y2, r2) and m2 > best_metric:
            print(f"  Hop {hop}: IMPROVED {best_metric:.10f} -> {m2:.10f} (type={ptype})")
            best_metric, best_x, best_y, best_r = m2, x2, y2, r2
            save_solution(best_x, best_y, best_r,
                        os.path.join(WORKDIR, 'solution_n26.json'))

        if hop % 50 == 0:
            elapsed = time.time() - t0
            print(f"  Hop {hop}/200, {elapsed:.0f}s, best={best_metric:.10f}")

    # Final polish
    print(f"\n=== Final Polish ===")
    x_f, y_f, r_f, m_f, s_f = optimize_jac(best_x, best_y, best_r, maxiter=30000)
    if s_f and is_feasible(x_f, y_f, r_f) and m_f > best_metric:
        best_metric, best_x, best_y, best_r = m_f, x_f, y_f, r_f

    save_solution(best_x, best_y, best_r, os.path.join(WORKDIR, 'solution_n26.json'))

    elapsed = time.time() - t0
    print(f"\n=== FINAL ===")
    print(f"Parent:  {parent_metric:.10f}")
    print(f"Best:    {best_metric:.10f}")
    print(f"Delta:   {best_metric - parent_metric:.2e}")
    print(f"Time:    {elapsed:.0f}s")

if __name__ == '__main__':
    main()
