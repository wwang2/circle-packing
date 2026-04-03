#!/usr/bin/env python3
"""
Deep search for n=32: Try to find better topologies via aggressive perturbations.
Current: 2.9396. Target: 2.939+ (known best).
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
    """Random placement with collision avoidance."""
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

def main():
    start_time = time.time()

    base = load_solution(OUTPUT_DIR / "solution_n32.json")
    best_metric = sum_radii(base)
    best_circles = base.copy()
    fp(f"Starting n=32 metric: {best_metric:.10f}")

    rng = np.random.RandomState(22222)

    # Strategy 1: Very aggressive basin hopping (large moves)
    fp("\n=== Strategy 1: Aggressive basin hopping (3000 trials) ===")
    for trial in range(3000):
        c = best_circles.copy()
        n = N
        method = rng.randint(0, 12)

        if method == 0:
            # Multi-swap (3-5 pairs)
            for _ in range(rng.randint(3, 6)):
                i, j = rng.choice(n, 2, replace=False)
                c[i, 0], c[j, 0] = c[j, 0], c[i, 0]
                c[i, 1], c[j, 1] = c[j, 1], c[i, 1]
        elif method == 1:
            # Scramble a subset (5-10 circles)
            k = rng.randint(5, 11)
            idx = rng.choice(n, k, replace=False)
            perm = rng.permutation(k)
            positions = c[idx, :2].copy()
            c[idx, 0] = positions[perm, 0]
            c[idx, 1] = positions[perm, 1]
        elif method == 2:
            # Remove 2 circles, re-add randomly
            remove = rng.choice(n, 2, replace=False)
            for i in remove:
                r = c[i, 2] * (0.5 + rng.random())
                c[i, 0] = r + rng.random() * (1 - 2*r)
                c[i, 1] = r + rng.random() * (1 - 2*r)
                c[i, 2] = r
        elif method == 3:
            # Reflect entire packing
            if rng.random() < 0.33:
                c[:, 0] = 1.0 - c[:, 0]
            elif rng.random() < 0.5:
                c[:, 1] = 1.0 - c[:, 1]
            else:
                c[:, 0], c[:, 1] = c[:, 1].copy(), c[:, 0].copy()
            # Then perturb
            c[:, 0] += rng.normal(0, 0.01, n)
            c[:, 1] += rng.normal(0, 0.01, n)
        elif method == 4:
            # Scale all radii, then perturb
            scale = 1 + rng.normal(0, 0.05)
            c[:, 2] *= scale
            c[:, 0] += rng.normal(0, 0.02, n)
            c[:, 1] += rng.normal(0, 0.02, n)
        elif method == 5:
            # Rotate quadrant by 90 degrees
            quadrant = rng.randint(4)
            cx, cy = 0.5, 0.5
            mask = np.zeros(n, dtype=bool)
            if quadrant == 0: mask = (c[:, 0] < cx) & (c[:, 1] < cy)
            elif quadrant == 1: mask = (c[:, 0] >= cx) & (c[:, 1] < cy)
            elif quadrant == 2: mask = (c[:, 0] < cx) & (c[:, 1] >= cy)
            else: mask = (c[:, 0] >= cx) & (c[:, 1] >= cy)
            # 90-degree rotation
            for i in np.where(mask)[0]:
                dx, dy = c[i, 0] - cx, c[i, 1] - cy
                c[i, 0] = cx - dy
                c[i, 1] = cy + dx
        elif method == 6:
            # Merge two small circles into one
            radii_order = np.argsort(c[:, 2])
            i, j = radii_order[0], radii_order[1]
            new_r = c[i, 2] + c[j, 2]
            new_x = (c[i, 0] + c[j, 0]) / 2
            new_y = (c[i, 1] + c[j, 1]) / 2
            c[i] = [new_x, new_y, new_r * 0.8]
            # Put j somewhere random
            r = 0.02 + rng.random() * 0.04
            c[j] = [r + rng.random() * (1-2*r), r + rng.random() * (1-2*r), r]
        elif method == 7:
            # Split largest into two
            i = np.argmax(c[:, 2])
            r_old = c[i, 2]
            r_new = r_old * 0.65
            offset = r_new * 0.5
            angle = rng.uniform(0, 2*math.pi)
            c[i, 2] = r_new
            # Find smallest to replace
            j = np.argmin(c[:, 2])
            c[j] = [c[i, 0] + offset * math.cos(angle),
                     c[i, 1] + offset * math.sin(angle),
                     r_new * 0.8]
        elif method == 8:
            # Perturb with varying intensity
            intensity = rng.uniform(0.01, 0.15)
            c[:, 0] += rng.normal(0, intensity, n)
            c[:, 1] += rng.normal(0, intensity, n)
            c[:, 2] *= (1 + rng.normal(0, intensity * 0.5, n))
        elif method == 9:
            # Move corner circles toward center
            corners = []
            for i in range(n):
                if (c[i, 0] < 0.15 or c[i, 0] > 0.85) and (c[i, 1] < 0.15 or c[i, 1] > 0.85):
                    corners.append(i)
            if corners:
                for i in corners:
                    c[i, 0] += (0.5 - c[i, 0]) * rng.uniform(0.1, 0.4)
                    c[i, 1] += (0.5 - c[i, 1]) * rng.uniform(0.1, 0.4)
        elif method == 10:
            # Fresh random init (explore completely new topology)
            c = random_init(n, rng)
        elif method == 11:
            # Inversion: swap inside/outside (conceptual)
            # Sort by distance from center, reverse assignment
            dists = np.sqrt((c[:, 0] - 0.5)**2 + (c[:, 1] - 0.5)**2)
            order = np.argsort(dists)
            rev_order = order[::-1]
            new_pos = c[rev_order, :2].copy()
            c[:, :2] = new_pos

        c[:, 2] = np.clip(c[:, 2], 0.005, 0.45)
        c[:, 0] = np.clip(c[:, 0], c[:, 2] + 0.001, 1 - c[:, 2] - 0.001)
        c[:, 1] = np.clip(c[:, 1], c[:, 2] + 0.001, 1 - c[:, 2] - 0.001)

        try:
            opt, metric = optimize_slsqp(c, maxiter=5000)
            viol = max_violation(opt)
            if viol <= 1e-10 and metric > best_metric + 1e-12:
                best_metric = metric
                best_circles = opt.copy()
                fp(f"  trial {trial} (method={method}): {metric:.10f} (NEW BEST)")
        except:
            pass

        if trial % 500 == 499:
            fp(f"    ... {trial+1}/3000, best={best_metric:.10f}")

    elapsed = time.time() - start_time
    fp(f"\n=== FINAL ===")
    fp(f"Time: {elapsed:.1f}s")
    fp(f"Best metric (n=32): {best_metric:.10f}")
    fp(f"Valid: {max_violation(best_circles) <= 1e-10}")

    current = load_solution(OUTPUT_DIR / "solution_n32.json")
    if best_metric > sum_radii(current) + 1e-14:
        save_solution(best_circles, OUTPUT_DIR / "solution_n32.json")
        fp("Saved!")
    else:
        fp(f"No improvement over current ({sum_radii(current):.10f})")


if __name__ == "__main__":
    main()
