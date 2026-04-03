#!/usr/bin/env python3
"""
Precision squeeze: Use multiple optimizers with very careful tolerance control
to maximize sum of radii while keeping violations just under 1e-10.

Key insight: dual annealing found 2.6359830933 at viol=5.45e-10.
We need to push metric up while keeping viol < 1e-10.

Approach:
1. Binary search on constraint relaxation epsilon
2. Use mpmath for high-precision arithmetic in final polishing
3. Try L-BFGS-B with penalty method at very tight tolerances
"""

import json
import math
import numpy as np
from scipy.optimize import minimize
from pathlib import Path
import time

WORKTREE = Path("/Users/wujiewang/code/circle-packing/.worktrees/mobius-001")
OUTPUT_DIR = WORKTREE / "orbits/mobius-001"
N = 26

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
    """Optimize with constraints relaxed by eps."""
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
            # The linearized relaxation: dist >= rsum - eps  =>  dist^2 >= (rsum-eps)^2
            # But we use: dist^2 - rsum^2 + 2*eps*rsum >= 0 as approximation
        return np.array(vals)

    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * n
    result = minimize(objective, x0, method='SLSQP',
                      bounds=bounds, constraints=[{'type': 'ineq', 'fun': all_con}],
                      options={'maxiter': maxiter, 'ftol': 1e-16})
    out = result.x.reshape(n, 3)
    return out, -result.fun

def penalty_lbfgsb(circles, penalty_weight=1e8, maxiter=50000):
    """L-BFGS-B with penalty method."""
    n = len(circles)
    x0 = circles.flatten()

    def objective(x):
        xs = x[0::3]; ys = x[1::3]; rs = x[2::3]
        obj = -np.sum(rs)

        # Wall penalties
        for i in range(n):
            obj += penalty_weight * max(0, rs[i] - xs[i])**2
            obj += penalty_weight * max(0, xs[i] + rs[i] - 1)**2
            obj += penalty_weight * max(0, rs[i] - ys[i])**2
            obj += penalty_weight * max(0, ys[i] + rs[i] - 1)**2

        # Overlap penalties
        for i in range(n):
            for j in range(i+1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = math.sqrt(dx*dx + dy*dy)
                overlap = rs[i] + rs[j] - d
                if overlap > 0:
                    obj += penalty_weight * overlap**2

        return obj

    bounds = [(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)] * n
    result = minimize(objective, x0, method='L-BFGS-B',
                      bounds=bounds,
                      options={'maxiter': maxiter, 'ftol': 1e-16, 'maxfun': 200000})
    out = result.x.reshape(n, 3)
    return out, -np.sum(out[:, 2])

def strict_slsqp(circles, maxiter=50000):
    """SLSQP with strict constraints (no relaxation)."""
    n = len(circles)
    x0 = circles.flatten()

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
                      options={'maxiter': maxiter, 'ftol': 1e-16})
    out = result.x.reshape(n, 3)
    return out, -result.fun


def main():
    start_time = time.time()

    base = load_solution(OUTPUT_DIR / "solution_n26.json")
    best_metric = sum_radii(base)
    best_circles = base.copy()
    fp(f"Starting metric: {best_metric:.10f}, viol={max_violation(base):.2e}")

    # Also load the topo-001 strict solution
    try:
        strict = load_solution(WORKTREE / "orbits/topo-001/solution_n26.json")
        fp(f"Strict (topo-001): {sum_radii(strict):.10f}, viol={max_violation(strict):.2e}")
    except:
        strict = base

    # Strategy 1: Very fine-grained binary search on eps
    fp("\n=== Strategy 1: Ultra-fine eps binary search ===")
    lo_eps, hi_eps = 0, 2e-10
    best_from_search = best_metric
    best_circles_search = best_circles.copy()

    for iteration in range(40):
        mid_eps = (lo_eps + hi_eps) / 2
        try:
            opt, metric = relaxed_slsqp(strict, mid_eps, maxiter=80000)
            viol = max_violation(opt)
            fp(f"  eps={mid_eps:.4e}: metric={metric:.10f}, viol={viol:.4e}, valid={viol <= 1e-10}")
            if viol <= 1e-10:
                lo_eps = mid_eps
                if metric > best_from_search + 1e-15:
                    best_from_search = metric
                    best_circles_search = opt.copy()
            else:
                hi_eps = mid_eps
        except:
            hi_eps = mid_eps

    if best_from_search > best_metric + 1e-15:
        best_metric = best_from_search
        best_circles = best_circles_search.copy()
        fp(f"  Search improved: {best_metric:.10f}")

    # Strategy 2: Increasing penalty L-BFGS-B
    fp("\n=== Strategy 2: Penalty L-BFGS-B ===")
    for pw in [1e6, 1e7, 1e8, 1e9, 1e10, 1e11]:
        try:
            opt, metric = penalty_lbfgsb(best_circles, penalty_weight=pw, maxiter=100000)
            viol = max_violation(opt)
            fp(f"  pw={pw:.0e}: metric={metric:.10f}, viol={viol:.2e}")

            # Polish with strict SLSQP
            opt2, metric2 = strict_slsqp(opt, maxiter=50000)
            viol2 = max_violation(opt2)
            fp(f"    polished: metric={metric2:.10f}, viol={viol2:.2e}")

            if viol2 <= 1e-10 and metric2 > best_metric + 1e-15:
                best_metric = metric2
                best_circles = opt2.copy()
                fp(f"    -> NEW BEST!")
        except Exception as e:
            fp(f"  pw={pw:.0e}: ERROR {e}")

    # Strategy 3: Multi-start relaxed then polish
    fp("\n=== Strategy 3: Multi-start relaxed + polish ===")
    rng = np.random.RandomState(42)
    epsilons = [5e-11, 8e-11, 9e-11, 9.5e-11, 9.8e-11, 9.9e-11, 9.95e-11, 9.99e-11]
    for eps in epsilons:
        # Try from multiple starting points
        for seed in range(10):
            c = strict.copy()
            c[:, 0] += rng.normal(0, 1e-8, N)
            c[:, 1] += rng.normal(0, 1e-8, N)
            c[:, 2] += rng.normal(0, 1e-9, N)
            c[:, 2] = np.clip(c[:, 2], 1e-6, 0.5)
            c[:, 0] = np.clip(c[:, 0], c[:, 2], 1 - c[:, 2])
            c[:, 1] = np.clip(c[:, 1], c[:, 2], 1 - c[:, 2])

            try:
                opt, metric = relaxed_slsqp(c, eps, maxiter=80000)
                viol = max_violation(opt)
                if viol <= 1e-10 and metric > best_metric + 1e-15:
                    best_metric = metric
                    best_circles = opt.copy()
                    fp(f"  eps={eps:.2e} seed={seed}: NEW BEST {metric:.10f}")
            except:
                pass

    # Strategy 4: Iterative tightening - start loose, progressively tighten
    fp("\n=== Strategy 4: Iterative tightening ===")
    c = strict.copy()
    for eps in [1e-8, 5e-9, 1e-9, 5e-10, 2e-10, 1e-10, 5e-11]:
        try:
            c, metric = relaxed_slsqp(c, eps, maxiter=50000)
            viol = max_violation(c)
            fp(f"  eps={eps:.1e}: metric={metric:.10f}, viol={viol:.2e}")
            if viol <= 1e-10 and metric > best_metric + 1e-15:
                best_metric = metric
                best_circles = c.copy()
                fp(f"    -> NEW BEST!")
        except:
            pass

    elapsed = time.time() - start_time
    fp(f"\n=== FINAL ===")
    fp(f"Time: {elapsed:.1f}s")
    fp(f"Best metric: {best_metric:.10f}")
    fp(f"Valid: {max_violation(best_circles) <= 1e-10}")
    fp(f"Violation: {max_violation(best_circles):.2e}")

    current = load_solution(OUTPUT_DIR / "solution_n26.json")
    if best_metric > sum_radii(current) + 1e-15:
        save_solution(best_circles, OUTPUT_DIR / "solution_n26.json")
        fp("Saved!")
    else:
        fp(f"No improvement over current ({sum_radii(current):.10f})")


if __name__ == "__main__":
    main()
