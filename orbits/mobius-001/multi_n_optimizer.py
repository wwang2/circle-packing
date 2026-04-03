#!/usr/bin/env python3
"""
Multi-n optimizer: try to find good solutions for n=10, 20, 30.
These are less contested than n=26 and n=32 but still valuable.
"""

import json
import math
import numpy as np
from scipy.optimize import minimize
from pathlib import Path
import time

WORKTREE = Path("/Users/wujiewang/code/circle-packing/.worktrees/mobius-001")
OUTPUT_DIR = WORKTREE / "orbits/mobius-001"

def fp(*args, **kwargs):
    print(*args, **kwargs, flush=True)

def save_solution(circles, path):
    data = {"circles": [[float(c[0]), float(c[1]), float(c[2])] for c in circles]}
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def max_violation(circles):
    n = len(circles)
    mv = 0.0
    for i in range(n):
        x, y, r = circles[i]
        mv = max(mv, r - x, x + r - 1, r - y, y + r - 1)
    for i in range(n):
        for j in range(i + 1, n):
            dx = circles[i, 0] - circles[j, 0]
            dy = circles[i, 1] - circles[j, 1]
            d = math.sqrt(dx * dx + dy * dy)
            mv = max(mv, circles[i, 2] + circles[j, 2] - d)
    return mv

def optimize_slsqp(init_circles, maxiter=8000):
    n = len(init_circles)
    x0 = init_circles.flatten()

    def objective(x):
        return -np.sum(x[2::3])

    def all_con(x):
        xs = x[0::3]; ys = x[1::3]; rs = x[2::3]
        vals = []
        vals.extend(xs - rs)
        vals.extend(1.0 - xs - rs)
        vals.extend(ys - rs)
        vals.extend(1.0 - ys - rs)
        vals.extend(rs - 1e-6)
        for i in range(n):
            dx = xs[i] - xs[i+1:]
            dy = ys[i] - ys[i+1:]
            rsum = rs[i] + rs[i+1:]
            dist_sq = dx*dx + dy*dy
            vals.extend(dist_sq - rsum**2)
        return np.array(vals)

    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * n
    result = minimize(objective, x0, method='SLSQP',
                      bounds=bounds, constraints=[{'type': 'ineq', 'fun': all_con}],
                      options={'maxiter': maxiter, 'ftol': 1e-15})
    out = result.x.reshape(n, 3)
    return out, -result.fun

def random_init(n, rng):
    r_base = 0.4 / math.sqrt(n * math.pi)
    circles = np.zeros((n, 3))
    placed = 0
    for attempt in range(n * 300):
        if placed >= n:
            break
        r = r_base * (0.3 + rng.random() * 1.4)
        x = r + rng.random() * (1 - 2*r)
        y = r + rng.random() * (1 - 2*r)
        ok = True
        for k in range(placed):
            dx = x - circles[k, 0]
            dy = y - circles[k, 1]
            if math.sqrt(dx*dx + dy*dy) < r + circles[k, 2] + 0.001:
                ok = False
                break
        if ok:
            circles[placed] = [x, y, r]
            placed += 1
    for i in range(placed, n):
        circles[i] = [rng.random() * 0.8 + 0.1, rng.random() * 0.8 + 0.1, 0.01]
    return circles

def ring_init(n, rng):
    circles = np.zeros((n, 3))
    r_center = 0.12 + rng.random() * 0.08
    circles[0] = [0.5, 0.5, r_center]
    idx = 1
    for ring in range(1, 5):
        num_in_ring = min(4 + ring * 2 + rng.randint(-1, 3), n - idx)
        if num_in_ring <= 0:
            break
        ring_r = 0.12 * ring + rng.random() * 0.05
        for k in range(num_in_ring):
            if idx >= n:
                break
            angle = 2 * math.pi * k / num_in_ring + rng.normal(0, 0.1)
            r = 0.03 + rng.random() * 0.06
            cx = 0.5 + ring_r * math.cos(angle)
            cy = 0.5 + ring_r * math.sin(angle)
            cx = np.clip(cx, r + 0.002, 1 - r - 0.002)
            cy = np.clip(cy, r + 0.002, 1 - r - 0.002)
            circles[idx] = [cx, cy, r]
            idx += 1
    for i in range(idx, n):
        circles[i] = [rng.random() * 0.8 + 0.1, rng.random() * 0.8 + 0.1, 0.02]
    return circles

def hex_init(n, rng):
    circles = np.zeros((n, 3))
    side = int(math.ceil(math.sqrt(n * 2 / math.sqrt(3))))
    r_base = 0.45 / (side + 1)
    idx = 0
    for row in range(side + 2):
        y = r_base + row * r_base * math.sqrt(3)
        if y > 1 - r_base:
            break
        offset = r_base if row % 2 else 0
        x = r_base + offset
        while x < 1 - r_base and idx < n:
            r = r_base * (0.7 + rng.random() * 0.6)
            circles[idx] = [x + rng.normal(0, r_base*0.1),
                           y + rng.normal(0, r_base*0.1), r]
            idx += 1
            x += 2 * r_base
    for i in range(idx, n):
        circles[i] = [rng.random() * 0.8 + 0.1, rng.random() * 0.8 + 0.1, 0.02]
    circles[:, 2] = np.clip(circles[:, 2], 0.005, 0.45)
    circles[:, 0] = np.clip(circles[:, 0], circles[:, 2] + 0.002, 1 - circles[:, 2] - 0.002)
    circles[:, 1] = np.clip(circles[:, 1], circles[:, 2] + 0.002, 1 - circles[:, 2] - 0.002)
    return circles

def optimize_n(n, n_inits=300, n_basin=500):
    """Optimize for a given n."""
    fp(f"\n{'='*60}")
    fp(f"Optimizing n={n}")
    fp(f"{'='*60}")

    best_metric = 0
    best_circles = None

    methods = [random_init, ring_init, hex_init]
    inits_per_method = n_inits // len(methods)

    # Phase 1: Multi-start
    for method in methods:
        method_name = method.__name__
        for seed in range(inits_per_method):
            rng = np.random.RandomState(seed * 31 + hash(method_name) % 10000)
            try:
                init = method(n, rng)
                opt, metric = optimize_slsqp(init, maxiter=5000)
                viol = max_violation(opt)
                if viol <= 1e-10 and metric > best_metric:
                    best_metric = metric
                    best_circles = opt.copy()
            except:
                pass
        fp(f"  {method_name}: best so far = {best_metric:.10f}")

    if best_circles is None:
        fp(f"  No valid solution found for n={n}")
        return None, 0

    # Phase 2: Basin hopping
    fp(f"  Basin hopping from {best_metric:.10f}...")
    rng = np.random.RandomState(42)
    for trial in range(n_basin):
        c = best_circles.copy()
        method = rng.randint(0, 6)
        if method == 0:
            i, j = rng.choice(n, 2, replace=False)
            c[i, 0], c[j, 0] = c[j, 0], c[i, 0]
            c[i, 1], c[j, 1] = c[j, 1], c[i, 1]
        elif method == 1:
            c[:, 0] += rng.normal(0, 0.02, n)
            c[:, 1] += rng.normal(0, 0.02, n)
        elif method == 2:
            c[:, 2] *= (1 + rng.normal(0, 0.05, n))
        elif method == 3:
            sz = rng.randint(2, max(3, n//3))
            idx = rng.choice(n, sz, replace=False)
            angle = rng.uniform(-0.4, 0.4)
            cx, cy = np.mean(c[idx, 0]), np.mean(c[idx, 1])
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            for ii in idx:
                dx, dy = c[ii, 0] - cx, c[ii, 1] - cy
                c[ii, 0] = cx + cos_a * dx - sin_a * dy
                c[ii, 1] = cy + sin_a * dx + cos_a * dy
        elif method == 4:
            i = rng.randint(n)
            c[i, 0] = rng.random() * 0.8 + 0.1
            c[i, 1] = rng.random() * 0.8 + 0.1
        elif method == 5:
            for _ in range(rng.randint(2, 4)):
                i, j = rng.choice(n, 2, replace=False)
                c[i, 0], c[j, 0] = c[j, 0], c[i, 0]
                c[i, 1], c[j, 1] = c[j, 1], c[i, 1]

        c[:, 2] = np.clip(c[:, 2], 0.005, 0.45)
        c[:, 0] = np.clip(c[:, 0], c[:, 2] + 0.001, 1 - c[:, 2] - 0.001)
        c[:, 1] = np.clip(c[:, 1], c[:, 2] + 0.001, 1 - c[:, 2] - 0.001)

        try:
            opt, metric = optimize_slsqp(c, maxiter=5000)
            viol = max_violation(opt)
            if viol <= 1e-10 and metric > best_metric + 1e-12:
                best_metric = metric
                best_circles = opt.copy()
                fp(f"    trial {trial}: {metric:.10f} (NEW BEST)")
        except:
            pass

        if trial % 100 == 99:
            fp(f"      ... {trial+1}/{n_basin}")

    fp(f"  FINAL n={n}: {best_metric:.10f}")
    return best_circles, best_metric

def main():
    start = time.time()

    known_bests = {10: 1.591, 20: 2.301, 30: 2.842}

    for n in [10, 20, 30]:
        circles, metric = optimize_n(n, n_inits=200, n_basin=500)
        if circles is not None:
            save_solution(circles, OUTPUT_DIR / f"solution_n{n}.json")
            fp(f"  Saved solution_n{n}.json: {metric:.10f}")
            fp(f"  Known best: ~{known_bests.get(n, '?')}")

    fp(f"\nTotal time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
