"""
Radical Topology Search: Try fundamentally different packing strategies.
The key insight is that ALL optimization approaches converge to the SAME
topology. We need to force exploration of DIFFERENT topologies.

Strategies:
1. Constraint-based topology forcing: force specific pairs to be in contact
2. Size-class variations: try distributions with different numbers of size classes
3. Completely different structural patterns (no concentric rings)
4. Greedy constructive approach: place circles one at a time
5. Known good patterns from literature for similar n values
"""

import json
import numpy as np
from scipy.optimize import minimize, differential_evolution
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

def optimize_slsqp(x0, y0, r0, maxiter=10000):
    n = len(x0)
    params0 = np.concatenate([x0, y0, r0])
    constraints = []
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[i] - p[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[i] - p[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[n+i] - p[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[n+i] - p[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[2*n+i] - 1e-6})
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
        params0, method='SLSQP', constraints=constraints,
        options={'maxiter': maxiter, 'ftol': 1e-15, 'disp': False}
    )
    x = result.x[:n]
    y = result.x[n:2*n]
    r = result.x[2*n:3*n]
    return x, y, r, np.sum(r), result.success

def is_feasible(x, y, r, tol=1e-8):
    n = len(x)
    for i in range(n):
        if r[i] < -tol: return False
        if x[i] - r[i] < -tol or 1 - x[i] - r[i] < -tol: return False
        if y[i] - r[i] < -tol or 1 - y[i] - r[i] < -tol: return False
    for i in range(n):
        for j in range(i+1, n):
            if (x[i]-x[j])**2 + (y[i]-y[j])**2 < (r[i]+r[j])**2 - tol:
                return False
    return True

def greedy_packing(n, seed=0, strategy='largest_first'):
    """Build packing greedily: place one circle at a time."""
    rng = np.random.RandomState(seed)
    x, y, r = np.zeros(n), np.zeros(n), np.zeros(n)

    # Start with the largest circle in center
    if strategy == 'largest_first':
        r_schedule = np.linspace(0.14, 0.06, n)
    elif strategy == 'uniform':
        r_schedule = np.full(n, 0.10)
    elif strategy == 'bimodal':
        r_schedule = np.array([0.13]*5 + [0.11]*8 + [0.09]*8 + [0.07]*5)[:n]
    elif strategy == 'random_size':
        r_schedule = rng.uniform(0.05, 0.14, n)
        r_schedule.sort()
        r_schedule = r_schedule[::-1]
    else:
        r_schedule = np.full(n, 0.10)

    placed = 0
    for i in range(n):
        target_r = r_schedule[i]
        best_r_placed = 0
        best_pos = None

        # Try many random positions
        for _ in range(500):
            px = rng.uniform(target_r, 1 - target_r)
            py = rng.uniform(target_r, 1 - target_r)

            # Check non-overlap with placed circles
            ok = True
            max_r = min(target_r, px, 1-px, py, 1-py)
            for j in range(placed):
                dist = np.sqrt((px - x[j])**2 + (py - y[j])**2)
                max_r = min(max_r, dist - r[j])
                if max_r < 0.01:
                    ok = False
                    break

            if ok and max_r > best_r_placed:
                best_r_placed = max_r
                best_pos = (px, py)

        if best_pos is not None:
            x[i] = best_pos[0]
            y[i] = best_pos[1]
            r[i] = best_r_placed
            placed += 1
        else:
            # Place randomly with tiny radius
            x[i] = rng.uniform(0.05, 0.95)
            y[i] = rng.uniform(0.05, 0.95)
            r[i] = 0.01
            placed += 1

    return x, y, r

def maxhole_greedy(n, seed=0):
    """Place circles at the point farthest from all existing circles/walls."""
    rng = np.random.RandomState(seed)
    x, y, r = [], [], []

    # Grid for evaluation
    grid_size = 100
    gx = np.linspace(0, 1, grid_size)
    gy = np.linspace(0, 1, grid_size)
    GX, GY = np.meshgrid(gx, gy)
    GX_flat = GX.ravel()
    GY_flat = GY.ravel()

    for i in range(n):
        # Distance to nearest wall
        dist_wall = np.minimum(np.minimum(GX_flat, 1-GX_flat),
                              np.minimum(GY_flat, 1-GY_flat))

        # Distance to nearest existing circle boundary
        min_dist = dist_wall.copy()
        for j in range(len(x)):
            d = np.sqrt((GX_flat - x[j])**2 + (GY_flat - y[j])**2) - r[j]
            min_dist = np.minimum(min_dist, d)

        # Place at maximum empty space
        # Add some randomness to avoid always picking the same topology
        noise = rng.uniform(0, 0.005, len(min_dist))
        best_idx = np.argmax(min_dist + noise)
        max_r = min_dist[best_idx]

        if max_r > 0.005:
            x.append(GX_flat[best_idx])
            y.append(GY_flat[best_idx])
            r.append(max_r)
        else:
            x.append(rng.uniform(0.05, 0.95))
            y.append(rng.uniform(0.05, 0.95))
            r.append(0.005)

    return np.array(x), np.array(y), np.array(r)

def zigzag_init(n, seed=0):
    """Zigzag pattern: alternating rows offset."""
    rng = np.random.RandomState(seed)
    n_rows = 4 + rng.randint(0, 3)  # 4-6 rows
    r_base = 0.5 / (n_rows + 1)

    x, y, r = [], [], []
    circles_placed = 0

    for row in range(n_rows):
        if circles_placed >= n:
            break
        n_in_row = n // n_rows + (1 if row < n % n_rows else 0)
        y_pos = (row + 0.5) / n_rows
        offset = 0.5 / n_in_row if row % 2 == 1 else 0

        for col in range(n_in_row):
            if circles_placed >= n:
                break
            x_pos = (col + 0.5) / n_in_row + offset * 0.5
            x.append(np.clip(x_pos + rng.normal(0, 0.01), 0.05, 0.95))
            y.append(np.clip(y_pos + rng.normal(0, 0.01), 0.05, 0.95))
            r.append(r_base + rng.uniform(-0.01, 0.01))
            circles_placed += 1

    while len(x) < n:
        x.append(rng.uniform(0.1, 0.9))
        y.append(rng.uniform(0.1, 0.9))
        r.append(0.04)

    x, y, r = np.array(x[:n]), np.array(y[:n]), np.array(r[:n])
    r = np.maximum(r, 0.02)
    x = np.clip(x, r+0.001, 1-r-0.001)
    y = np.clip(y, r+0.001, 1-r-0.001)
    return x, y, r

def corner_focused_init(n, seed=0):
    """Put medium circles in corners, large in center band."""
    rng = np.random.RandomState(seed)

    x, y, r = [], [], []

    # 4 corner circles (medium)
    for cx, cy in [(0.12, 0.12), (0.88, 0.12), (0.12, 0.88), (0.88, 0.88)]:
        x.append(cx); y.append(cy); r.append(0.11 + rng.uniform(-0.01, 0.01))

    # Edge midpoint circles
    for cx, cy in [(0.5, 0.12), (0.5, 0.88), (0.12, 0.5), (0.88, 0.5)]:
        x.append(cx + rng.normal(0, 0.02))
        y.append(cy + rng.normal(0, 0.02))
        r.append(0.10 + rng.uniform(-0.02, 0.02))

    # Center region
    n_center = n - 8
    for i in range(n_center):
        angle = 2*np.pi*i/n_center + rng.uniform(-0.3, 0.3)
        rad = 0.15 + rng.uniform(0, 0.15)
        x.append(0.5 + rad*np.cos(angle) + rng.normal(0, 0.02))
        y.append(0.5 + rad*np.sin(angle) + rng.normal(0, 0.02))
        r.append(0.08 + rng.uniform(-0.02, 0.03))

    x, y, r = np.array(x[:n]), np.array(y[:n]), np.array(r[:n])
    r = np.maximum(r, 0.02)
    x = np.clip(x, r+0.001, 1-r-0.001)
    y = np.clip(y, r+0.001, 1-r-0.001)
    return x, y, r

def cross_pattern_init(n, seed=0):
    """Cross/plus pattern: circles along horizontal and vertical axes."""
    rng = np.random.RandomState(seed)

    x, y, r = [], [], []

    # Horizontal bar: ~n/2 circles
    n_h = n // 2
    for i in range(n_h):
        px = (i + 0.5) / n_h
        py = 0.5 + rng.normal(0, 0.03)
        x.append(px); y.append(py)
        r.append(0.09 + rng.uniform(-0.02, 0.02))

    # Vertical bar: remaining
    n_v = n - n_h
    for i in range(n_v):
        px = 0.5 + rng.normal(0, 0.03)
        py = (i + 0.5) / n_v
        x.append(px); y.append(py)
        r.append(0.08 + rng.uniform(-0.02, 0.02))

    x, y, r = np.array(x[:n]), np.array(y[:n]), np.array(r[:n])
    r = np.maximum(r, 0.02)
    x = np.clip(x, r+0.001, 1-r-0.001)
    y = np.clip(y, r+0.001, 1-r-0.001)
    return x, y, r

def asymmetric_init(n, seed=0):
    """Deliberately asymmetric: break the symmetry of the known solution."""
    rng = np.random.RandomState(seed)

    x, y, r = [], [], []

    # One very large circle off-center
    x.append(0.3 + rng.uniform(-0.1, 0.1))
    y.append(0.3 + rng.uniform(-0.1, 0.1))
    r.append(0.18 + rng.uniform(-0.03, 0.03))

    # Pack rest around it
    for i in range(1, n):
        x.append(rng.uniform(0.06, 0.94))
        y.append(rng.uniform(0.06, 0.94))
        r.append(rng.uniform(0.04, 0.12))

    x, y, r = np.array(x[:n]), np.array(y[:n]), np.array(r[:n])
    r = np.maximum(r, 0.02)
    x = np.clip(x, r+0.001, 1-r-0.001)
    y = np.clip(y, r+0.001, 1-r-0.001)
    return x, y, r

def three_cluster_init(n, seed=0):
    """Three clusters of circles."""
    rng = np.random.RandomState(seed)

    centers = [(0.3, 0.3), (0.7, 0.3), (0.5, 0.75)]
    n_per = [n//3, n//3, n - 2*(n//3)]

    x, y, r = [], [], []
    for ci, (cx, cy) in enumerate(centers):
        for i in range(n_per[ci]):
            angle = 2*np.pi*i/n_per[ci] + rng.uniform(-0.5, 0.5)
            rad = rng.uniform(0.02, 0.18)
            x.append(cx + rad*np.cos(angle))
            y.append(cy + rad*np.sin(angle))
            r.append(rng.uniform(0.05, 0.11))

    x, y, r = np.array(x[:n]), np.array(y[:n]), np.array(r[:n])
    r = np.maximum(r, 0.02)
    x = np.clip(x, r+0.001, 1-r-0.001)
    y = np.clip(y, r+0.001, 1-r-0.001)
    return x, y, r

def de_optimize(x0, y0, r0, maxiter=500, seed=42):
    """Differential evolution - global optimizer."""
    n = len(x0)

    def obj(params):
        x = params[:n]
        y = params[n:2*n]
        r = params[2*n:3*n]

        penalty = 0
        for i in range(n):
            v = max(0, r[i] - x[i]); penalty += v**2 * 1e6
            v = max(0, x[i] + r[i] - 1); penalty += v**2 * 1e6
            v = max(0, r[i] - y[i]); penalty += v**2 * 1e6
            v = max(0, y[i] + r[i] - 1); penalty += v**2 * 1e6
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                v = max(0, r[i] + r[j] - dist)
                penalty += v**2 * 1e6

        return -np.sum(r) + penalty

    bounds = [(0.01, 0.99)]*n + [(0.01, 0.99)]*n + [(0.005, 0.25)]*n

    result = differential_evolution(
        obj, bounds, maxiter=maxiter, seed=seed, tol=1e-12,
        init='sobol', popsize=20, mutation=(0.5, 1.5), recombination=0.9
    )

    x = result.x[:n]
    y = result.x[n:2*n]
    r = result.x[2*n:3*n]
    return x, y, r

def main():
    t0 = time.time()
    parent_path = os.path.join(WORKDIR, '..', 'nlp-001', 'solution_n26.json')
    x0, y0, r0 = load_solution(parent_path)
    parent_metric = np.sum(r0)
    print(f"Parent metric: {parent_metric:.10f}")

    best_metric = parent_metric
    best_x, best_y, best_r = x0.copy(), y0.copy(), r0.copy()

    results = []

    def try_init(name, x, y, r):
        nonlocal best_metric, best_x, best_y, best_r
        x2, y2, r2, metric, success = optimize_slsqp(x, y, r, maxiter=10000)
        if success and is_feasible(x2, y2, r2):
            results.append((name, metric))
            if metric > best_metric:
                print(f"  IMPROVED: {name} -> {metric:.10f} (+{metric-parent_metric:.2e})")
                best_metric = metric
                best_x, best_y, best_r = x2.copy(), y2.copy(), r2.copy()
                save_solution(best_x, best_y, best_r,
                            os.path.join(WORKDIR, 'solution_n26.json'))
            return metric
        return 0

    # ====== Strategy 1: Greedy constructive ======
    print("\n=== Greedy Constructive ===")
    for strategy in ['largest_first', 'uniform', 'bimodal', 'random_size']:
        for seed in range(20):
            x, y, r = greedy_packing(N, seed=seed, strategy=strategy)
            m = try_init(f"greedy_{strategy}_s{seed}", x, y, r)
            if seed % 5 == 0:
                print(f"  {strategy} seed={seed}: {m:.6f}, best={best_metric:.10f}")

    # ====== Strategy 2: Max-hole greedy ======
    print("\n=== Max-Hole Greedy ===")
    for seed in range(20):
        x, y, r = maxhole_greedy(N, seed=seed)
        m = try_init(f"maxhole_s{seed}", x, y, r)
        if seed % 5 == 0:
            print(f"  seed={seed}: {m:.6f}, best={best_metric:.10f}")

    # ====== Strategy 3: Structural patterns ======
    print("\n=== Structural Patterns ===")
    for seed in range(15):
        x, y, r = zigzag_init(N, seed=seed)
        try_init(f"zigzag_s{seed}", x, y, r)

        x, y, r = corner_focused_init(N, seed=seed)
        try_init(f"corner_s{seed}", x, y, r)

        x, y, r = cross_pattern_init(N, seed=seed)
        try_init(f"cross_s{seed}", x, y, r)

        x, y, r = asymmetric_init(N, seed=seed)
        try_init(f"asym_s{seed}", x, y, r)

        x, y, r = three_cluster_init(N, seed=seed)
        try_init(f"3cluster_s{seed}", x, y, r)

        if seed % 5 == 0:
            elapsed = time.time() - t0
            print(f"  seed={seed}, {elapsed:.0f}s, best={best_metric:.10f}")

    # ====== Strategy 4: Differential Evolution ======
    print("\n=== Differential Evolution ===")
    for seed in range(5):
        print(f"  DE seed={seed}...")
        x, y, r = de_optimize(x0, y0, r0, maxiter=300, seed=seed)
        m = try_init(f"DE_s{seed}", x, y, r)
        print(f"  DE seed={seed}: {m:.6f}")

    # ====== Strategy 5: Symmetry breaking of parent ======
    print("\n=== Symmetry Breaking ===")
    rng = np.random.RandomState(42)
    # The parent has approximate mirror symmetry. Break it.
    for trial in range(30):
        x2, y2, r2 = x0.copy(), y0.copy(), r0.copy()

        # Pick a random subset and mirror/rotate them
        k = rng.randint(3, N//2)
        idxs = rng.choice(N, k, replace=False)

        op = rng.randint(0, 5)
        if op == 0:
            # Reflect x
            x2[idxs] = 1 - x2[idxs]
        elif op == 1:
            # Reflect y
            y2[idxs] = 1 - y2[idxs]
        elif op == 2:
            # Rotate 90 degrees around center
            for idx in idxs:
                dx, dy = x2[idx] - 0.5, y2[idx] - 0.5
                x2[idx] = 0.5 - dy
                y2[idx] = 0.5 + dx
        elif op == 3:
            # Shift subset
            shift = rng.uniform(-0.15, 0.15, 2)
            x2[idxs] += shift[0]
            y2[idxs] += shift[1]
        elif op == 4:
            # Scramble positions within subset
            perm = rng.permutation(len(idxs))
            x2[idxs] = x2[idxs[perm]]
            y2[idxs] = y2[idxs[perm]]

        r2 = np.maximum(r2, 0.01)
        x2 = np.clip(x2, r2+0.001, 1-r2-0.001)
        y2 = np.clip(y2, r2+0.001, 1-r2-0.001)

        m = try_init(f"symbreak_op{op}_t{trial}", x2, y2, r2)

    elapsed = time.time() - t0
    print(f"\n  Symmetry breaking done, {elapsed:.0f}s, best={best_metric:.10f}")

    # ====== Summary ======
    results.sort(key=lambda t: t[1], reverse=True)
    print(f"\n=== Top 20 Results ===")
    for name, metric in results[:20]:
        print(f"  {name}: {metric:.10f}")

    save_solution(best_x, best_y, best_r, os.path.join(WORKDIR, 'solution_n26.json'))

    elapsed = time.time() - t0
    print(f"\n=== FINAL ===")
    print(f"Parent:  {parent_metric:.10f}")
    print(f"Best:    {best_metric:.10f}")
    print(f"Delta:   {best_metric - parent_metric:.2e}")
    print(f"Time:    {elapsed:.0f}s")

if __name__ == '__main__':
    main()
