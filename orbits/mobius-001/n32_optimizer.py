#!/usr/bin/env python3
"""
Optimizer for n=32 circle packing.
Known best: 2.939+ (Berthold et al 2026).

Strategy: Multi-start with structured initializations + SLSQP polish.
Try ring patterns, hex patterns, grid patterns, and random inits.
"""

import json
import math
import numpy as np
from scipy.optimize import minimize
from pathlib import Path
import time

WORKTREE = Path("/Users/wujiewang/code/circle-packing/.worktrees/mobius-001")
OUTPUT_DIR = WORKTREE / "orbits/mobius-001"
N = 32

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

def optimize_slsqp(init_circles, maxiter=10000):
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

def ring_init(n, rng, n_rings=None):
    """Concentric ring pattern."""
    circles = np.zeros((n, 3))
    if n_rings is None:
        n_rings = rng.randint(3, 6)

    # Center circle
    r_center = 0.10 + rng.random() * 0.05
    circles[0] = [0.5, 0.5, r_center]
    idx = 1

    remaining = n - 1
    ring_sizes = []
    for ring in range(n_rings):
        if ring < n_rings - 1:
            sz = max(3, remaining // (n_rings - ring) + rng.randint(-2, 3))
            sz = min(sz, remaining)
        else:
            sz = remaining
        ring_sizes.append(sz)
        remaining -= sz
        if remaining <= 0:
            break

    for ring, sz in enumerate(ring_sizes):
        if sz <= 0:
            break
        ring_radius = 0.12 + 0.12 * (ring + 1) + rng.random() * 0.03
        r_circle = 0.03 + rng.random() * 0.05
        phase = rng.random() * 2 * math.pi

        for k in range(sz):
            if idx >= n:
                break
            angle = phase + 2 * math.pi * k / sz
            cx = 0.5 + ring_radius * math.cos(angle)
            cy = 0.5 + ring_radius * math.sin(angle)
            r = r_circle * (0.7 + rng.random() * 0.6)
            cx = np.clip(cx, r + 0.002, 1 - r - 0.002)
            cy = np.clip(cy, r + 0.002, 1 - r - 0.002)
            circles[idx] = [cx, cy, r]
            idx += 1

    for i in range(idx, n):
        r = 0.02
        circles[i] = [rng.random() * 0.8 + 0.1, rng.random() * 0.8 + 0.1, r]

    return circles

def hex_init(n, rng):
    """Hexagonal grid pattern with perturbation."""
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
            cx = x + rng.normal(0, r_base * 0.1)
            cy = y + rng.normal(0, r_base * 0.1)
            cx = np.clip(cx, r + 0.002, 1 - r - 0.002)
            cy = np.clip(cy, r + 0.002, 1 - r - 0.002)
            circles[idx] = [cx, cy, r]
            idx += 1
            x += 2 * r_base

    for i in range(idx, n):
        r = 0.02
        circles[i] = [rng.random() * 0.8 + 0.1, rng.random() * 0.8 + 0.1, r]

    return circles

def grid_init(n, rng):
    """Grid pattern with size variations."""
    side = int(math.ceil(math.sqrt(n)))
    r_base = 0.45 / (side + 0.5)
    circles = np.zeros((n, 3))
    idx = 0

    for i in range(side + 1):
        for j in range(side + 1):
            if idx >= n:
                break
            cx = (i + 0.5) / (side + 0.5) * 0.9 + 0.05
            cy = (j + 0.5) / (side + 0.5) * 0.9 + 0.05
            cx += rng.normal(0, r_base * 0.15)
            cy += rng.normal(0, r_base * 0.15)
            r = r_base * (0.6 + rng.random() * 0.8)
            cx = np.clip(cx, r + 0.002, 1 - r - 0.002)
            cy = np.clip(cy, r + 0.002, 1 - r - 0.002)
            circles[idx] = [cx, cy, r]
            idx += 1

    return circles

def random_init(n, rng):
    """Random placement with collision avoidance."""
    r_base = 0.4 / math.sqrt(n * math.pi)
    circles = np.zeros((n, 3))
    placed = 0
    for attempt in range(n * 200):
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

def corner_fill_init(n, rng):
    """Start with big circles in corners and fill progressively."""
    circles = np.zeros((n, 3))
    # 4 corner circles
    r_corner = 0.15 + rng.random() * 0.05
    circles[0] = [r_corner, r_corner, r_corner]
    circles[1] = [1-r_corner, r_corner, r_corner]
    circles[2] = [r_corner, 1-r_corner, r_corner]
    circles[3] = [1-r_corner, 1-r_corner, r_corner]

    # Center circle
    r_center = 0.10 + rng.random() * 0.05
    circles[4] = [0.5, 0.5, r_center]

    # Edge midpoint circles
    r_edge = 0.08 + rng.random() * 0.04
    if n > 5:
        circles[5] = [0.5, r_edge, r_edge]
    if n > 6:
        circles[6] = [0.5, 1-r_edge, r_edge]
    if n > 7:
        circles[7] = [r_edge, 0.5, r_edge]
    if n > 8:
        circles[8] = [1-r_edge, 0.5, r_edge]

    # Fill remaining with random
    for i in range(9, n):
        r = 0.02 + rng.random() * 0.06
        for _ in range(50):
            x = r + rng.random() * (1 - 2*r)
            y = r + rng.random() * (1 - 2*r)
            ok = True
            for j in range(i):
                dx = x - circles[j, 0]
                dy = y - circles[j, 1]
                if math.sqrt(dx*dx + dy*dy) < r + circles[j, 2] + 0.001:
                    ok = False
                    break
            if ok:
                circles[i] = [x, y, r]
                break
        else:
            circles[i] = [rng.random() * 0.8 + 0.1, rng.random() * 0.8 + 0.1, 0.01]

    return circles

def d4_symmetric_init(n, rng):
    """D4-symmetric initialization (4-fold + reflections)."""
    circles = np.zeros((n, 3))
    # Place circles with D4 symmetry: (x,y), (1-x,y), (x,1-y), (1-x,1-y), (y,x), etc.
    n_unique = n // 8 + 1
    idx = 0

    # Center
    r = 0.08 + rng.random() * 0.08
    circles[idx] = [0.5, 0.5, r]
    idx += 1

    for _ in range(n_unique):
        if idx >= n:
            break
        r = 0.03 + rng.random() * 0.07
        x = 0.05 + rng.random() * 0.4
        y = 0.05 + rng.random() * 0.4

        # Generate D4 orbit: (x,y), (1-x,y), (x,1-y), (1-x,1-y), (y,x), (1-y,x), (y,1-x), (1-y,1-x)
        points = set()
        for sx in [x, 1-x]:
            for sy in [y, 1-y]:
                points.add((round(sx, 8), round(sy, 8)))
                points.add((round(sy, 8), round(sx, 8)))

        for (px, py) in points:
            if idx >= n:
                break
            pr = r * (0.9 + rng.random() * 0.2)
            cx = np.clip(px, pr + 0.002, 1 - pr - 0.002)
            cy = np.clip(py, pr + 0.002, 1 - pr - 0.002)
            circles[idx] = [cx, cy, pr]
            idx += 1

    for i in range(idx, n):
        circles[i] = [rng.random() * 0.8 + 0.1, rng.random() * 0.8 + 0.1, 0.02]

    return circles

def main():
    start_time = time.time()
    fp(f"Optimizing n={N} circle packing")

    best_metric = 0
    best_circles = None
    total = 0

    init_methods = [
        ('ring', ring_init),
        ('hex', hex_init),
        ('grid', grid_init),
        ('random', random_init),
        ('corner', corner_fill_init),
        ('d4_sym', d4_symmetric_init),
    ]

    seeds_per_method = 150

    for method_name, method_fn in init_methods:
        method_best = 0
        for seed in range(seeds_per_method):
            rng = np.random.RandomState(seed * 17 + hash(method_name) % 10000)
            try:
                init = method_fn(N, rng)
                opt, metric = optimize_slsqp(init, maxiter=5000)
                viol = max_violation(opt)
                total += 1

                if viol <= 1e-10:
                    method_best = max(method_best, metric)
                    if metric > best_metric:
                        best_metric = metric
                        best_circles = opt.copy()
                        if metric > 2.90:
                            fp(f"  {method_name} seed={seed}: {metric:.10f} (NEW BEST)")
            except:
                pass

        fp(f"  {method_name}: best={method_best:.10f} ({total} total)")

    # Basin hopping from best found
    if best_circles is not None:
        fp(f"\n=== Basin hopping from best ({best_metric:.10f}) ===")
        rng = np.random.RandomState(42)
        for trial in range(500):
            c = best_circles.copy()
            # Random perturbation
            method = rng.randint(0, 5)
            if method == 0:
                # Swap positions
                i, j = rng.choice(N, 2, replace=False)
                c[i, 0], c[j, 0] = c[j, 0], c[i, 0]
                c[i, 1], c[j, 1] = c[j, 1], c[i, 1]
            elif method == 1:
                # Perturb all
                c[:, 0] += rng.normal(0, 0.02, N)
                c[:, 1] += rng.normal(0, 0.02, N)
            elif method == 2:
                # Scale radii
                c[:, 2] *= (1 + rng.normal(0, 0.05, N))
            elif method == 3:
                # Rotate subset
                n_rot = rng.randint(3, N//2)
                idx = rng.choice(N, n_rot, replace=False)
                angle = rng.uniform(-0.3, 0.3)
                cx, cy = np.mean(c[idx, 0]), np.mean(c[idx, 1])
                cos_a, sin_a = math.cos(angle), math.sin(angle)
                for ii in idx:
                    dx, dy = c[ii, 0] - cx, c[ii, 1] - cy
                    c[ii, 0] = cx + cos_a * dx - sin_a * dy
                    c[ii, 1] = cy + sin_a * dx + cos_a * dy
            elif method == 4:
                # Move one circle randomly
                i = rng.randint(N)
                c[i, 0] = rng.random() * 0.8 + 0.1
                c[i, 1] = rng.random() * 0.8 + 0.1

            c[:, 2] = np.clip(c[:, 2], 0.005, 0.45)
            c[:, 0] = np.clip(c[:, 0], c[:, 2] + 0.002, 1 - c[:, 2] - 0.002)
            c[:, 1] = np.clip(c[:, 1], c[:, 2] + 0.002, 1 - c[:, 2] - 0.002)

            try:
                opt, metric = optimize_slsqp(c, maxiter=5000)
                viol = max_violation(opt)
                if viol <= 1e-10 and metric > best_metric + 1e-12:
                    best_metric = metric
                    best_circles = opt.copy()
                    fp(f"  trial {trial}: {metric:.10f} (NEW BEST)")
            except:
                pass

            if trial % 100 == 99:
                fp(f"    ... {trial+1}/500, best={best_metric:.10f}")

    elapsed = time.time() - start_time
    fp(f"\n=== FINAL ===")
    fp(f"Total configs: {total}")
    fp(f"Time: {elapsed:.1f}s")
    fp(f"Best metric (n={N}): {best_metric:.10f}")

    if best_circles is not None:
        fp(f"Valid: {max_violation(best_circles) <= 1e-10}")
        save_solution(best_circles, OUTPUT_DIR / f"solution_n{N}.json")
        fp(f"Saved to solution_n{N}.json")


if __name__ == "__main__":
    main()
