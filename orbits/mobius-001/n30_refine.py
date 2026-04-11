#!/usr/bin/env python3
"""
Intensive refinement for n=30. Current: 2.8387, known best: 2.842+.
Gap of ~0.003 means there's likely a better topology to find.
"""

import json
import math
import numpy as np
from scipy.optimize import minimize
from pathlib import Path
import time

WORKTREE = Path("/Users/wujiewang/code/circle-packing/.worktrees/mobius-001")
OUTPUT_DIR = WORKTREE / "orbits/mobius-001"
N = 30

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
    n_rings = rng.randint(3, 6)
    r_center = 0.10 + rng.random() * 0.06
    circles[0] = [0.5, 0.5, r_center]
    idx = 1
    remaining = n - 1
    for ring in range(n_rings):
        if remaining <= 0 or idx >= n:
            break
        if ring < n_rings - 1:
            sz = max(3, remaining // (n_rings - ring) + rng.randint(-2, 3))
            sz = min(sz, remaining)
        else:
            sz = remaining
        ring_r = 0.10 + 0.12 * (ring + 1) + rng.random() * 0.04
        r_c = 0.03 + rng.random() * 0.05
        phase = rng.random() * 2 * math.pi
        for k in range(sz):
            if idx >= n:
                break
            angle = phase + 2 * math.pi * k / sz
            cx = 0.5 + ring_r * math.cos(angle)
            cy = 0.5 + ring_r * math.sin(angle)
            r = r_c * (0.7 + rng.random() * 0.6)
            cx = np.clip(cx, r + 0.002, 1 - r - 0.002)
            cy = np.clip(cy, r + 0.002, 1 - r - 0.002)
            circles[idx] = [cx, cy, r]
            idx += 1
        remaining = n - idx
    for i in range(idx, n):
        circles[i] = [rng.random() * 0.8 + 0.1, rng.random() * 0.8 + 0.1, 0.02]
    return circles

def main():
    start_time = time.time()

    base = load_solution(OUTPUT_DIR / "solution_n30.json")
    best_metric = sum_radii(base)
    best_circles = base.copy()
    fp(f"Starting n=30 metric: {best_metric:.10f}")

    # Phase 1: More multi-start with different seeds
    fp("\n=== Phase 1: Additional multi-start (500 seeds) ===")
    for seed in range(500):
        rng = np.random.RandomState(seed + 200000)
        try:
            if seed % 3 == 0:
                init = random_init(N, rng)
            elif seed % 3 == 1:
                init = ring_init(N, rng)
            else:
                # Grid-like
                side = int(math.ceil(math.sqrt(N)))
                r_base = 0.45 / (side + 1)
                init = np.zeros((N, 3))
                idx = 0
                for i in range(side + 1):
                    for j in range(side + 1):
                        if idx >= N:
                            break
                        init[idx] = [(i+0.5)/(side+0.5)*0.9+0.05 + rng.normal(0, r_base*0.15),
                                    (j+0.5)/(side+0.5)*0.9+0.05 + rng.normal(0, r_base*0.15),
                                    r_base * (0.6 + rng.random() * 0.8)]
                        idx += 1
                init[:, 2] = np.clip(init[:, 2], 0.005, 0.45)
                init[:, 0] = np.clip(init[:, 0], init[:, 2] + 0.002, 1 - init[:, 2] - 0.002)
                init[:, 1] = np.clip(init[:, 1], init[:, 2] + 0.002, 1 - init[:, 2] - 0.002)

            opt, metric = optimize_slsqp(init, maxiter=5000)
            viol = max_violation(opt)
            if viol <= 1e-10 and metric > best_metric + 1e-12:
                best_metric = metric
                best_circles = opt.copy()
                fp(f"  seed {seed}: {metric:.10f} (NEW BEST)")
        except:
            pass

        if seed % 100 == 99:
            fp(f"    ... {seed+1}/500, best={best_metric:.10f}")

    # Phase 2: Aggressive basin hopping
    fp(f"\n=== Phase 2: Basin hopping (2000 trials) ===")
    rng = np.random.RandomState(33333)
    for trial in range(2000):
        c = best_circles.copy()
        n = N
        method = rng.randint(0, 8)

        if method == 0:
            for _ in range(rng.randint(1, 4)):
                i, j = rng.choice(n, 2, replace=False)
                c[i, 0], c[j, 0] = c[j, 0], c[i, 0]
                c[i, 1], c[j, 1] = c[j, 1], c[i, 1]
        elif method == 1:
            c[:, 0] += rng.normal(0, 0.03, n)
            c[:, 1] += rng.normal(0, 0.03, n)
        elif method == 2:
            c[:, 2] *= (1 + rng.normal(0, 0.08, n))
        elif method == 3:
            sz = rng.randint(3, n//2)
            idx = rng.choice(n, sz, replace=False)
            angle = rng.uniform(-0.5, 0.5)
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
            c[i, 2] = max(0.01, c[i, 2] * (0.5 + rng.random()))
        elif method == 5:
            subset = rng.choice(n, rng.randint(2, n//3), replace=False)
            if rng.random() < 0.5:
                c[subset, 0] = 1.0 - c[subset, 0]
            else:
                c[subset, 1] = 1.0 - c[subset, 1]
        elif method == 6:
            # Scramble positions of 4-8 circles
            k = rng.randint(4, 9)
            idx = rng.choice(n, k, replace=False)
            perm = rng.permutation(k)
            positions = c[idx, :2].copy()
            c[idx, 0] = positions[perm, 0]
            c[idx, 1] = positions[perm, 1]
        elif method == 7:
            # Fresh random init
            c = random_init(n, rng)

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
            fp(f"    ... {trial+1}/2000, best={best_metric:.10f}")

    elapsed = time.time() - start_time
    fp(f"\n=== FINAL ===")
    fp(f"Time: {elapsed:.1f}s")
    fp(f"Best n=30: {best_metric:.10f}")
    fp(f"Valid: {max_violation(best_circles) <= 1e-10}")

    save_solution(best_circles, OUTPUT_DIR / "solution_n30.json")
    fp("Saved!")


if __name__ == "__main__":
    main()
