#!/usr/bin/env python3
"""
Tolerance exploitation for n=32: push constraint violation close to 1e-10 limit.
Also try more basin hopping from the 2.9396 solution.
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

def relaxed_slsqp(circles, eps, maxiter=50000):
    n = len(circles)
    x0 = circles.flatten()

    def objective(x):
        return -np.sum(x[2::3])

    def all_con(x):
        xs = x[0::3]; ys = x[1::3]; rs = x[2::3]
        vals = []
        vals.extend(xs - rs + eps)
        vals.extend(1.0 - xs - rs + eps)
        vals.extend(ys - rs + eps)
        vals.extend(1.0 - ys - rs + eps)
        vals.extend(rs - 1e-6)
        for i in range(n):
            dx = xs[i] - xs[i+1:]
            dy = ys[i] - ys[i+1:]
            rsum = rs[i] + rs[i+1:]
            dist_sq = dx*dx + dy*dy
            vals.extend(dist_sq - rsum**2 + eps * 2 * rsum)
        return np.array(vals)

    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * n
    result = minimize(objective, x0, method='SLSQP',
                      bounds=bounds, constraints=[{'type': 'ineq', 'fun': all_con}],
                      options={'maxiter': maxiter, 'ftol': 1e-16})
    out = result.x.reshape(n, 3)
    return out, -result.fun

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

def main():
    start_time = time.time()

    base = load_solution(OUTPUT_DIR / "solution_n32.json")
    best_metric = sum_radii(base)
    best_circles = base.copy()
    fp(f"Starting n=32 metric: {best_metric:.10f}, viol={max_violation(base):.2e}")

    # Step 1: First do strict SLSQP to ensure tight constraint satisfaction
    fp("\n=== Step 1: Strict SLSQP polish ===")
    strict, strict_m = optimize_slsqp(base, maxiter=50000)
    strict_v = max_violation(strict)
    fp(f"  Strict: metric={strict_m:.10f}, viol={strict_v:.2e}")

    # Step 2: Binary search for optimal eps
    fp("\n=== Step 2: Tolerance exploitation ===")
    lo_eps, hi_eps = 0, 2e-10
    best_squeeze = best_metric
    best_sq_circles = best_circles.copy()

    for iteration in range(30):
        mid_eps = (lo_eps + hi_eps) / 2
        try:
            opt, metric = relaxed_slsqp(strict, mid_eps, maxiter=80000)
            viol = max_violation(opt)
            valid = viol <= 1e-10
            fp(f"  eps={mid_eps:.4e}: metric={metric:.10f}, viol={viol:.4e}, valid={valid}")
            if valid:
                lo_eps = mid_eps
                if metric > best_squeeze:
                    best_squeeze = metric
                    best_sq_circles = opt.copy()
            else:
                hi_eps = mid_eps
        except:
            hi_eps = mid_eps

    if best_squeeze > best_metric + 1e-15:
        best_metric = best_squeeze
        best_circles = best_sq_circles.copy()
        fp(f"  Squeeze improved: {best_metric:.10f}")

    # Step 3: More basin hopping focused on swap moves
    fp("\n=== Step 3: Targeted basin hopping (1000 trials) ===")
    rng = np.random.RandomState(11111)
    for trial in range(1000):
        c = best_circles.copy()
        n = N

        method = rng.randint(0, 6)
        if method == 0:
            # Swap 2-3 pairs
            for _ in range(rng.randint(2, 4)):
                i, j = rng.choice(n, 2, replace=False)
                c[i, 0], c[j, 0] = c[j, 0], c[i, 0]
                c[i, 1], c[j, 1] = c[j, 1], c[i, 1]
        elif method == 1:
            # Large perturbation of 3-5 circles
            indices = rng.choice(n, rng.randint(3, 6), replace=False)
            for i in indices:
                c[i, 0] += rng.normal(0, 0.1)
                c[i, 1] += rng.normal(0, 0.1)
                c[i, 2] *= (1 + rng.normal(0, 0.1))
        elif method == 2:
            # Rotate quadrant
            quadrant = rng.randint(4)
            cx, cy = 0.5, 0.5
            mask = np.zeros(n, dtype=bool)
            if quadrant == 0: mask = (c[:, 0] < cx) & (c[:, 1] < cy)
            elif quadrant == 1: mask = (c[:, 0] >= cx) & (c[:, 1] < cy)
            elif quadrant == 2: mask = (c[:, 0] < cx) & (c[:, 1] >= cy)
            else: mask = (c[:, 0] >= cx) & (c[:, 1] >= cy)
            angle = rng.uniform(-0.3, 0.3)
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            for i in np.where(mask)[0]:
                dx, dy = c[i, 0] - cx, c[i, 1] - cy
                c[i, 0] = cx + cos_a * dx - sin_a * dy
                c[i, 1] = cy + sin_a * dx + cos_a * dy
        elif method == 3:
            # Swap a circle with a random position
            i = rng.randint(n)
            c[i, 0] = rng.random() * 0.8 + 0.1
            c[i, 1] = rng.random() * 0.8 + 0.1
        elif method == 4:
            # Scale radii differentially
            c[:, 2] *= (1 + rng.normal(0, 0.03, n))
        elif method == 5:
            # Reflect half
            if rng.random() < 0.5:
                mask = c[:, 0] > 0.5
                c[mask, 0] = 1.0 - c[mask, 0]
            else:
                mask = c[:, 1] > 0.5
                c[mask, 1] = 1.0 - c[mask, 1]

        c[:, 2] = np.clip(c[:, 2], 0.005, 0.45)
        c[:, 0] = np.clip(c[:, 0], c[:, 2] + 0.001, 1 - c[:, 2] - 0.001)
        c[:, 1] = np.clip(c[:, 1], c[:, 2] + 0.001, 1 - c[:, 2] - 0.001)

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
            fp(f"    ... {trial+1}/1000, best={best_metric:.10f}")

    elapsed = time.time() - start_time
    fp(f"\n=== FINAL ===")
    fp(f"Time: {elapsed:.1f}s")
    fp(f"Best metric (n=32): {best_metric:.10f}")
    fp(f"Valid: {max_violation(best_circles) <= 1e-10}")
    fp(f"Violation: {max_violation(best_circles):.2e}")

    save_solution(best_circles, OUTPUT_DIR / "solution_n32.json")
    fp("Saved!")


if __name__ == "__main__":
    main()
