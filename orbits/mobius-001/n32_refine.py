#!/usr/bin/env python3
"""
Intensive refinement for n=32 solution.
Start from the best found (2.9334) and do aggressive basin hopping + multi-strategy perturbation.
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

def load_solution(path):
    with open(path) as f:
        return np.array(json.load(f)["circles"])

def save_solution(circles, path):
    data = {"circles": [[float(c[0]), float(c[1]), float(c[2])] for c in circles]}
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def sum_radii(c):
    return float(np.sum(c[:, 2]))

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

def perturb(circles, rng, scale=0.1):
    n = len(circles)
    c = circles.copy()
    method = rng.randint(0, 10)

    if method == 0:
        # Swap 1-2 pairs
        for _ in range(rng.randint(1, 3)):
            i, j = rng.choice(n, 2, replace=False)
            c[i, 0], c[j, 0] = c[j, 0], c[i, 0]
            c[i, 1], c[j, 1] = c[j, 1], c[i, 1]
    elif method == 1:
        # Small position perturbation
        c[:, 0] += rng.normal(0, scale * 0.05, n)
        c[:, 1] += rng.normal(0, scale * 0.05, n)
    elif method == 2:
        # Radius perturbation
        c[:, 2] *= (1 + rng.normal(0, scale * 0.1, n))
    elif method == 3:
        # Rotate subset
        sz = rng.randint(3, n//2)
        idx = rng.choice(n, sz, replace=False)
        angle = rng.uniform(-0.4, 0.4)
        cx, cy = np.mean(c[idx, 0]), np.mean(c[idx, 1])
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        for ii in idx:
            dx, dy = c[ii, 0] - cx, c[ii, 1] - cy
            c[ii, 0] = cx + cos_a * dx - sin_a * dy
            c[ii, 1] = cy + sin_a * dx + cos_a * dy
    elif method == 4:
        # Mirror subset
        subset = rng.choice(n, rng.randint(2, n//3), replace=False)
        if rng.random() < 0.5:
            c[subset, 0] = 1.0 - c[subset, 0]
        else:
            c[subset, 1] = 1.0 - c[subset, 1]
    elif method == 5:
        # Move one circle randomly
        i = rng.randint(n)
        c[i, 0] = rng.random() * 0.8 + 0.1
        c[i, 1] = rng.random() * 0.8 + 0.1
        c[i, 2] = max(0.01, c[i, 2] * (0.5 + rng.random()))
    elif method == 6:
        # Scale all radii uniformly
        factor = 1 + rng.normal(0, 0.02)
        c[:, 2] *= factor
    elif method == 7:
        # Swap positions of nearby circles
        i = rng.randint(n)
        dists = np.sqrt((c[:, 0] - c[i, 0])**2 + (c[:, 1] - c[i, 1])**2)
        dists[i] = 999
        j = np.argmin(dists)
        c[i, 0], c[j, 0] = c[j, 0], c[i, 0]
        c[i, 1], c[j, 1] = c[j, 1], c[i, 1]
    elif method == 8:
        # Translate a cluster
        center = rng.randint(n)
        dists = np.sqrt((c[:, 0] - c[center, 0])**2 + (c[:, 1] - c[center, 1])**2)
        cluster = np.where(dists < 0.25)[0]
        dx = rng.normal(0, 0.03)
        dy = rng.normal(0, 0.03)
        c[cluster, 0] += dx
        c[cluster, 1] += dy
    elif method == 9:
        # Enlarge one, shrink neighbors
        i = rng.randint(n)
        delta = rng.uniform(0.005, 0.02)
        c[i, 2] += delta
        dists = np.sqrt((c[:, 0] - c[i, 0])**2 + (c[:, 1] - c[i, 1])**2)
        dists[i] = 999
        neighbors = np.argsort(dists)[:4]
        for j in neighbors:
            c[j, 2] -= delta / 4

    c[:, 2] = np.clip(c[:, 2], 0.005, 0.45)
    c[:, 0] = np.clip(c[:, 0], c[:, 2] + 0.001, 1 - c[:, 2] - 0.001)
    c[:, 1] = np.clip(c[:, 1], c[:, 2] + 0.001, 1 - c[:, 2] - 0.001)
    return c


def main():
    start_time = time.time()

    base = load_solution(OUTPUT_DIR / "solution_n32.json")
    best_metric = sum_radii(base)
    best_circles = base.copy()
    fp(f"Starting n=32 metric: {best_metric:.10f}")

    # Phase 1: Intensive basin hopping (2000 trials)
    fp("\n=== Phase 1: Basin hopping (2000 trials) ===")
    rng = np.random.RandomState(54321)
    for trial in range(2000):
        scale = 0.05 + 0.3 * (trial % 20) / 20
        c = perturb(best_circles, rng, scale=scale)
        try:
            opt, metric = optimize_slsqp(c, maxiter=5000)
            viol = max_violation(opt)
            if viol <= 1e-10 and metric > best_metric + 1e-12:
                best_metric = metric
                best_circles = opt.copy()
                fp(f"  trial {trial}: {metric:.10f} (NEW BEST)")
        except:
            pass

        if trial % 200 == 199:
            fp(f"    ... {trial+1}/2000, best={best_metric:.10f}")

    # Phase 2: Multi-start around best with fresh random seeds
    fp("\n=== Phase 2: Fresh multi-start near best (500 trials) ===")
    for seed in range(500):
        rng2 = np.random.RandomState(seed + 100000)
        c = best_circles.copy()
        # Strong perturbation
        c[:, 0] += rng2.normal(0, 0.03, N)
        c[:, 1] += rng2.normal(0, 0.03, N)
        c[:, 2] *= (1 + rng2.normal(0, 0.05, N))
        c[:, 2] = np.clip(c[:, 2], 0.005, 0.45)
        c[:, 0] = np.clip(c[:, 0], c[:, 2] + 0.001, 1 - c[:, 2] - 0.001)
        c[:, 1] = np.clip(c[:, 1], c[:, 2] + 0.001, 1 - c[:, 2] - 0.001)

        try:
            opt, metric = optimize_slsqp(c, maxiter=5000)
            viol = max_violation(opt)
            if viol <= 1e-10 and metric > best_metric + 1e-12:
                best_metric = metric
                best_circles = opt.copy()
                fp(f"  seed {seed}: {metric:.10f} (NEW BEST)")
        except:
            pass

        if seed % 100 == 99:
            fp(f"    ... {seed+1}/500, best={best_metric:.10f}")

    elapsed = time.time() - start_time
    fp(f"\n=== FINAL ===")
    fp(f"Time: {elapsed:.1f}s")
    fp(f"Best metric (n=32): {best_metric:.10f}")
    fp(f"Valid: {max_violation(best_circles) <= 1e-10}")

    save_solution(best_circles, OUTPUT_DIR / "solution_n32.json")
    fp("Saved!")


if __name__ == "__main__":
    main()
