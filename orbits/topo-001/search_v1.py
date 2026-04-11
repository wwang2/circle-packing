"""
Topology Search v1: Multi-strategy search for better circle packing n=26.
Focuses on escaping the known basin via diverse initializations + SLSQP.
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

def optimize_slsqp(x0, y0, r0, maxiter=8000):
    """SLSQP optimization of circle packing."""
    n = len(x0)
    params0 = np.concatenate([x0, y0, r0])

    constraints = []
    # Containment
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[i] - p[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[i] - p[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[n+i] - p[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[n+i] - p[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[2*n+i] - 1e-6})

    # Non-overlap (squared distance form for better conditioning)
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: (
                    (p[i]-p[j])**2 + (p[n+i]-p[n+j])**2 - (p[2*n+i]+p[2*n+j])**2
                )
            })

    result = minimize(
        lambda p: -np.sum(p[2*n:3*n]),
        params0,
        method='SLSQP',
        constraints=constraints,
        options={'maxiter': maxiter, 'ftol': 1e-15, 'disp': False}
    )

    x = result.x[:n]
    y = result.x[n:2*n]
    r = result.x[2*n:3*n]
    metric = np.sum(r)
    return x, y, r, metric, result.success

def is_feasible(x, y, r, tol=1e-8):
    n = len(x)
    for i in range(n):
        if r[i] < -tol: return False
        if x[i] - r[i] < -tol or 1 - x[i] - r[i] < -tol: return False
        if y[i] - r[i] < -tol or 1 - y[i] - r[i] < -tol: return False
    for i in range(n):
        for j in range(i+1, n):
            dist2 = (x[i]-x[j])**2 + (y[i]-y[j])**2
            if dist2 < (r[i]+r[j])**2 - tol: return False
    return True

# ---- Initialization generators ----

def init_concentric_rings(n, seed=0):
    """Concentric ring pattern (known to work well)."""
    rng = np.random.RandomState(seed)
    center = (0.5, 0.5)

    # 1 center + ring1 + ring2 + corners
    x, y, r = [], [], []

    # Center
    x.append(0.5 + rng.normal(0, 0.02))
    y.append(0.5 + rng.normal(0, 0.02))
    r.append(0.13)

    # Inner ring: ~8 circles
    n_inner = 7 + rng.randint(0, 3)  # 7-9
    r_inner = 0.18 + rng.uniform(-0.02, 0.02)
    for i in range(n_inner):
        angle = 2 * np.pi * i / n_inner + rng.normal(0, 0.1)
        x.append(0.5 + r_inner * np.cos(angle) + rng.normal(0, 0.01))
        y.append(0.5 + r_inner * np.sin(angle) + rng.normal(0, 0.01))
        r.append(0.10 + rng.uniform(-0.02, 0.02))

    # Outer ring: fill remainder minus 4 corners
    n_corners = 4
    n_outer = n - 1 - n_inner - n_corners
    r_outer = 0.38 + rng.uniform(-0.03, 0.03)
    for i in range(max(0, n_outer)):
        angle = 2 * np.pi * i / max(n_outer, 1) + rng.normal(0, 0.1)
        x.append(0.5 + r_outer * np.cos(angle) + rng.normal(0, 0.01))
        y.append(0.5 + r_outer * np.sin(angle) + rng.normal(0, 0.01))
        r.append(0.09 + rng.uniform(-0.02, 0.02))

    # Corners
    corners = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
    for cx, cy in corners[:n_corners]:
        if len(x) >= n:
            break
        x.append(cx + rng.normal(0, 0.02))
        y.append(cy + rng.normal(0, 0.02))
        r.append(0.08 + rng.uniform(-0.02, 0.02))

    # Truncate or pad
    while len(x) > n:
        x.pop(); y.pop(); r.pop()
    while len(x) < n:
        x.append(rng.uniform(0.15, 0.85))
        y.append(rng.uniform(0.15, 0.85))
        r.append(0.05)

    x, y, r = np.array(x), np.array(y), np.array(r)
    r = np.maximum(r, 0.02)
    x = np.clip(x, r + 0.001, 1 - r - 0.001)
    y = np.clip(y, r + 0.001, 1 - r - 0.001)
    return x, y, r

def init_grid_based(n, rows, cols, seed=0):
    """Grid-based initialization with variable radii."""
    rng = np.random.RandomState(seed)
    positions = []
    for i in range(rows):
        for j in range(cols):
            cx = (j + 0.5) / cols
            cy = (i + 0.5) / rows
            positions.append((cx, cy))

    rng.shuffle(positions)
    positions = positions[:n]
    while len(positions) < n:
        positions.append((rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)))

    x = np.array([p[0] for p in positions]) + rng.normal(0, 0.01, n)
    y = np.array([p[1] for p in positions]) + rng.normal(0, 0.01, n)
    r_base = 0.4 / max(rows, cols)
    r = np.full(n, r_base) + rng.uniform(-0.01, 0.01, n)
    r = np.maximum(r, 0.02)
    x = np.clip(x, r + 0.001, 1 - r - 0.001)
    y = np.clip(y, r + 0.001, 1 - r - 0.001)
    return x, y, r

def init_poisson_disk(n, seed=0, min_dist=0.12):
    """Poisson disk sampling initialization."""
    rng = np.random.RandomState(seed)
    points = []
    for _ in range(n * 100):
        if len(points) >= n:
            break
        px, py = rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95)
        ok = True
        for qx, qy in points:
            if (px-qx)**2 + (py-qy)**2 < min_dist**2:
                ok = False
                break
        if ok:
            points.append((px, py))

    while len(points) < n:
        points.append((rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)))

    x = np.array([p[0] for p in points[:n]])
    y = np.array([p[1] for p in points[:n]])
    r = np.full(n, min_dist * 0.4) + rng.uniform(-0.005, 0.005, n)
    r = np.maximum(r, 0.02)
    x = np.clip(x, r + 0.001, 1 - r - 0.001)
    y = np.clip(y, r + 0.001, 1 - r - 0.001)
    return x, y, r

def init_two_layer(n, seed=0):
    """Two distinct size classes: large interior + small periphery."""
    rng = np.random.RandomState(seed)
    n_large = 5 + rng.randint(0, 8)  # 5-12 large
    n_small = n - n_large

    r_large = 0.12 + rng.uniform(-0.03, 0.03)
    r_small = 0.06 + rng.uniform(-0.02, 0.02)

    x, y, r = [], [], []

    # Large circles: spread across interior
    for i in range(n_large):
        angle = 2 * np.pi * i / n_large + rng.normal(0, 0.2)
        dist = 0.15 + rng.uniform(0, 0.25)
        x.append(0.5 + dist * np.cos(angle))
        y.append(0.5 + dist * np.sin(angle))
        r.append(r_large + rng.normal(0, 0.01))

    # Small circles: edges and gaps
    for i in range(n_small):
        x.append(rng.uniform(0.08, 0.92))
        y.append(rng.uniform(0.08, 0.92))
        r.append(r_small + rng.normal(0, 0.01))

    x, y, r = np.array(x), np.array(y), np.array(r)
    r = np.maximum(r, 0.02)
    x = np.clip(x, r + 0.001, 1 - r - 0.001)
    y = np.clip(y, r + 0.001, 1 - r - 0.001)
    return x, y, r

def init_diagonal_bands(n, seed=0):
    """Circles arranged along diagonal bands."""
    rng = np.random.RandomState(seed)
    n_bands = 3 + rng.randint(0, 4)
    x, y, r = [], [], []

    per_band = n // n_bands
    extra = n % n_bands

    for b in range(n_bands):
        nb = per_band + (1 if b < extra else 0)
        offset = (b + 0.5) / n_bands
        for i in range(nb):
            t = (i + 0.5) / nb
            x.append(t + rng.normal(0, 0.02))
            y.append(offset + rng.normal(0, 0.03))
            r.append(0.08 + rng.uniform(-0.03, 0.03))

    x, y, r = np.array(x[:n]), np.array(y[:n]), np.array(r[:n])
    r = np.maximum(r, 0.02)
    x = np.clip(x, r + 0.001, 1 - r - 0.001)
    y = np.clip(y, r + 0.001, 1 - r - 0.001)
    return x, y, r

def init_sunflower(n, seed=0):
    """Sunflower/golden angle spiral."""
    rng = np.random.RandomState(seed)
    golden = (1 + np.sqrt(5)) / 2

    x, y, r = [], [], []
    for i in range(n):
        theta = 2 * np.pi * i / golden**2
        rad = 0.45 * np.sqrt((i + 0.5) / n)
        px = 0.5 + rad * np.cos(theta) + rng.normal(0, 0.01)
        py = 0.5 + rad * np.sin(theta) + rng.normal(0, 0.01)
        x.append(px)
        y.append(py)
        r.append(0.08 + rng.uniform(-0.02, 0.02))

    x, y, r = np.array(x), np.array(y), np.array(r)
    r = np.maximum(r, 0.02)
    x = np.clip(x, r + 0.001, 1 - r - 0.001)
    y = np.clip(y, r + 0.001, 1 - r - 0.001)
    return x, y, r

def basin_hop_from(x0, y0, r0, n_hops=30, seed=42):
    """Basin hopping: perturb + re-optimize."""
    rng = np.random.RandomState(seed)
    n = len(x0)
    best_metric = np.sum(r0)
    best_x, best_y, best_r = x0.copy(), y0.copy(), r0.copy()

    for hop in range(n_hops):
        # Choose perturbation type
        ptype = rng.randint(0, 5)
        x2, y2, r2 = x0.copy(), y0.copy(), r0.copy()

        if ptype == 0:
            # Random displacement of 1-3 circles
            nc = rng.randint(1, 4)
            idxs = rng.choice(n, nc, replace=False)
            for idx in idxs:
                x2[idx] += rng.normal(0, 0.05)
                y2[idx] += rng.normal(0, 0.05)
                r2[idx] *= rng.uniform(0.7, 1.1)
        elif ptype == 1:
            # Swap two circles
            i, j = rng.choice(n, 2, replace=False)
            x2[i], x2[j] = x2[j], x2[i]
            y2[i], y2[j] = y2[j], y2[i]
        elif ptype == 2:
            # Global noise
            scale = rng.uniform(0.01, 0.08)
            x2 += rng.normal(0, scale, n)
            y2 += rng.normal(0, scale, n)
            r2 *= (1 + rng.normal(0, scale*0.5, n))
        elif ptype == 3:
            # Rotate a subset
            k = rng.randint(3, n//2)
            idxs = rng.choice(n, k, replace=False)
            cx = np.mean(x2[idxs])
            cy = np.mean(y2[idxs])
            angle = rng.uniform(-0.5, 0.5)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            for idx in idxs:
                dx, dy = x2[idx] - cx, y2[idx] - cy
                x2[idx] = cx + cos_a * dx - sin_a * dy
                y2[idx] = cy + sin_a * dx + cos_a * dy
        elif ptype == 4:
            # Reflect a subset
            k = rng.randint(2, n//3)
            idxs = rng.choice(n, k, replace=False)
            axis = rng.choice(['x', 'y', 'diag'])
            if axis == 'x':
                y2[idxs] = 1 - y2[idxs]
            elif axis == 'y':
                x2[idxs] = 1 - x2[idxs]
            else:
                for idx in idxs:
                    x2[idx], y2[idx] = y2[idx], x2[idx]

        r2 = np.maximum(r2, 0.01)
        x2 = np.clip(x2, r2 + 0.001, 1 - r2 - 0.001)
        y2 = np.clip(y2, r2 + 0.001, 1 - r2 - 0.001)

        x2, y2, r2, metric, success = optimize_slsqp(x2, y2, r2, maxiter=5000)

        if success and is_feasible(x2, y2, r2) and metric > best_metric:
            print(f"  Hop {hop}: IMPROVED {best_metric:.10f} -> {metric:.10f} (type={ptype})")
            best_metric = metric
            best_x, best_y, best_r = x2.copy(), y2.copy(), r2.copy()
            # Update base for next hops
            x0, y0, r0 = x2.copy(), y2.copy(), r2.copy()

        if hop % 10 == 0:
            print(f"  Hop {hop}/{n_hops}, best={best_metric:.10f}")

    return best_x, best_y, best_r, best_metric

def main():
    t0 = time.time()
    parent_path = os.path.join(WORKDIR, '..', 'nlp-001', 'solution_n26.json')
    x0, y0, r0 = load_solution(parent_path)
    parent_metric = np.sum(r0)
    print(f"Parent metric: {parent_metric:.10f}")

    best_metric = parent_metric
    best_x, best_y, best_r = x0.copy(), y0.copy(), r0.copy()
    all_results = []

    # ====== Phase 1: Diverse initializations ======
    print("\n=== Phase 1: Diverse Initializations ===")

    generators = []

    # Concentric rings (many variants)
    for seed in range(30):
        generators.append(('ring', seed, lambda s=seed: init_concentric_rings(N, s)))

    # Grid variants
    for rows, cols in [(4,7), (5,6), (5,5), (6,5), (7,4), (3,9), (4,6)]:
        for seed in range(5):
            generators.append(('grid', f"{rows}x{cols}_s{seed}",
                             lambda r=rows, c=cols, s=seed: init_grid_based(N, r, c, s)))

    # Poisson disk
    for seed in range(20):
        for md in [0.10, 0.12, 0.14, 0.16]:
            generators.append(('poisson', f"md{md}_s{seed}",
                             lambda s=seed, m=md: init_poisson_disk(N, s, m)))

    # Two-layer
    for seed in range(20):
        generators.append(('2layer', seed, lambda s=seed: init_two_layer(N, s)))

    # Diagonal bands
    for seed in range(15):
        generators.append(('diag', seed, lambda s=seed: init_diagonal_bands(N, s)))

    # Sunflower
    for seed in range(15):
        generators.append(('sunflower', seed, lambda s=seed: init_sunflower(N, s)))

    print(f"Total initializations: {len(generators)}")

    for idx, (name, param, gen_fn) in enumerate(generators):
        x, y, r = gen_fn()
        x, y, r, metric, success = optimize_slsqp(x, y, r, maxiter=5000)

        if success and is_feasible(x, y, r):
            all_results.append((name, param, metric))
            if metric > best_metric:
                print(f"  [{idx}] IMPROVED: {name}({param}) -> {metric:.10f} (+{metric-parent_metric:.2e})")
                best_metric = metric
                best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
                save_solution(best_x, best_y, best_r,
                            os.path.join(WORKDIR, 'solution_n26.json'))

        if idx % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{idx}/{len(generators)}] {elapsed:.0f}s, best={best_metric:.10f}")

    # Sort results
    valid = [(n, p, m) for n, p, m in all_results if m > 2.5]
    valid.sort(key=lambda t: t[2], reverse=True)
    print(f"\nTop 10 initializations:")
    for name, param, metric in valid[:10]:
        print(f"  {name}({param}): {metric:.10f}")

    # ====== Phase 2: Basin hopping from best ======
    print(f"\n=== Phase 2: Basin Hopping (from {best_metric:.10f}) ===")
    x2, y2, r2, m2 = basin_hop_from(best_x, best_y, best_r, n_hops=50, seed=42)
    if m2 > best_metric:
        best_metric = m2
        best_x, best_y, best_r = x2, y2, r2

    # Also basin hop from parent
    print(f"\n=== Phase 2b: Basin Hopping from parent ===")
    x2, y2, r2, m2 = basin_hop_from(x0, y0, r0, n_hops=50, seed=123)
    if m2 > best_metric:
        best_metric = m2
        best_x, best_y, best_r = x2, y2, r2

    # ====== Phase 3: Re-optimize best with more iterations ======
    print(f"\n=== Phase 3: Polish best ({best_metric:.10f}) ===")
    x3, y3, r3, m3, _ = optimize_slsqp(best_x, best_y, best_r, maxiter=20000)
    if is_feasible(x3, y3, r3) and m3 > best_metric:
        best_metric = m3
        best_x, best_y, best_r = x3, y3, r3

    # Save final
    save_solution(best_x, best_y, best_r, os.path.join(WORKDIR, 'solution_n26.json'))

    elapsed = time.time() - t0
    print(f"\n=== FINAL RESULT ===")
    print(f"Parent:  {parent_metric:.10f}")
    print(f"Best:    {best_metric:.10f}")
    print(f"Delta:   {best_metric - parent_metric:.2e}")
    print(f"Time:    {elapsed:.0f}s")

    return best_metric

if __name__ == '__main__':
    main()
