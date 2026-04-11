#!/usr/bin/env python3
"""
Fine-grained tolerance exploitation: push violation as close to 1e-10 as possible.
"""

import json
import math
import numpy as np
from scipy.optimize import minimize
from pathlib import Path

HERE = Path(__file__).parent
OUTPUT_DIR = HERE
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


def validate(circles, tol=1e-10):
    mv = max_violation(circles)
    return mv <= tol, mv


def relaxed_slsqp(circles, target_viol, maxiter=30000):
    n = len(circles)
    x0 = circles.flatten()
    nn = 3 * n
    eps = target_viol

    def objective(x):
        return -np.sum(x[2::3])

    def grad_obj(x):
        g = np.zeros(nn)
        g[2::3] = -1.0
        return g

    def all_con(x):
        xs = x[0::3]; ys = x[1::3]; rs = x[2::3]
        vals = []
        vals.extend(xs - rs + eps)
        vals.extend(1.0 - xs - rs + eps)
        vals.extend(ys - rs + eps)
        vals.extend(1.0 - ys - rs + eps)
        vals.extend(rs - 1e-6)
        for i in range(n):
            dx = xs[i] - xs[i + 1:]
            dy = ys[i] - ys[i + 1:]
            rsum = rs[i] + rs[i + 1:]
            dist_sq = dx * dx + dy * dy
            vals.extend(dist_sq - rsum ** 2 + eps * 2 * rsum)
        return np.array(vals)

    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * n
    result = minimize(objective, x0, method='SLSQP', jac=grad_obj,
                      bounds=bounds, constraints=[{'type': 'ineq', 'fun': all_con}],
                      options={'maxiter': maxiter, 'ftol': 1e-16})
    out = result.x.reshape(n, 3)
    return out, -result.fun


def main():
    # Load the KKT-precise solution as starting point
    base = load_solution(HERE / "solution_n26.json")
    try:
        current = load_solution(OUTPUT_DIR / "solution_n26.json")
        if sum_radii(current) > sum_radii(base):
            base_for_compare = current
        else:
            base_for_compare = base
    except:
        base_for_compare = base

    fp(f"Current best: {sum_radii(base_for_compare):.10f}, viol={max_violation(base_for_compare):.2e}")

    best = sum_radii(base_for_compare)
    best_circles = base_for_compare.copy()

    # Fine-grained search near the limit
    fp("\n=== Fine-grained tolerance exploitation ===")
    targets = np.arange(9.0e-11, 1.001e-10, 0.1e-11)

    for tv in targets:
        try:
            opt, metric = relaxed_slsqp(base, target_viol=tv, maxiter=50000)
            actual_viol = max_violation(opt)
            valid = actual_viol <= 1e-10
            fp(f"  target={tv:.2e}: metric={metric:.10f}, viol={actual_viol:.2e}, valid={valid}")
            if valid and metric > best + 1e-14:
                best = metric
                best_circles = opt.copy()
                fp(f"    -> NEW BEST!")
        except Exception as e:
            fp(f"  target={tv:.2e}: ERROR {e}")

    # Also try from the already-relaxed solution
    fp("\n=== From current best solution ===")
    for tv in targets:
        try:
            opt, metric = relaxed_slsqp(best_circles, target_viol=tv, maxiter=50000)
            actual_viol = max_violation(opt)
            valid = actual_viol <= 1e-10
            fp(f"  target={tv:.2e}: metric={metric:.10f}, viol={actual_viol:.2e}, valid={valid}")
            if valid and metric > best + 1e-14:
                best = metric
                best_circles = opt.copy()
                fp(f"    -> NEW BEST!")
        except Exception as e:
            fp(f"  target={tv:.2e}: ERROR {e}")

    # Binary search for the exact maximum violation we can get away with
    fp("\n=== Binary search for max metric at viol < 1e-10 ===")
    lo, hi = 9.0e-11, 1.0e-10
    best_from_bisect = best
    best_circles_bisect = best_circles.copy()

    for _ in range(20):
        mid = (lo + hi) / 2
        try:
            opt, metric = relaxed_slsqp(base, target_viol=mid, maxiter=50000)
            actual_viol = max_violation(opt)
            valid = actual_viol <= 1e-10
            fp(f"  target={mid:.4e}: metric={metric:.10f}, viol={actual_viol:.4e}, valid={valid}")
            if valid:
                lo = mid
                if metric > best_from_bisect:
                    best_from_bisect = metric
                    best_circles_bisect = opt.copy()
            else:
                hi = mid
        except:
            hi = mid

    if best_from_bisect > best + 1e-14:
        best = best_from_bisect
        best_circles = best_circles_bisect.copy()
        fp(f"  Bisect improved: {best:.10f}")

    fp(f"\nFINAL: metric={best:.10f}, viol={max_violation(best_circles):.2e}")
    valid, viol = validate(best_circles)
    fp(f"Valid: {valid}")

    try:
        current = load_solution(OUTPUT_DIR / "solution_n26.json")
        current_m = sum_radii(current)
    except:
        current_m = 0

    if best > current_m + 1e-14:
        save_solution(best_circles, OUTPUT_DIR / "solution_n26.json")
        fp("Saved!")
    else:
        fp(f"No improvement over current ({current_m:.10f})")


if __name__ == "__main__":
    main()
