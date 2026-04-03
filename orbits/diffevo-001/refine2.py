"""
Advanced refinement: systematic single-circle moves + re-optimization.
Also tries removing a circle and adding it back elsewhere.
"""

import numpy as np
from scipy.optimize import minimize
import json
import sys
import time
from pathlib import Path


def penalized_obj_grad(x, n, lam):
    """Objective + gradient for L-BFGS-B."""
    arr = x.reshape(n, 3)
    cx, cy, r = arr[:, 0], arr[:, 1], arr[:, 2]
    grad = np.zeros_like(arr)

    obj = -r.sum()
    grad[:, 2] = -1.0

    v1 = np.maximum(0, r - cx)
    v2 = np.maximum(0, cx + r - 1.0)
    v3 = np.maximum(0, r - cy)
    v4 = np.maximum(0, cy + r - 1.0)

    obj += lam * (np.dot(v1, v1) + np.dot(v2, v2) + np.dot(v3, v3) + np.dot(v4, v4))
    grad[:, 0] += lam * 2 * (-v1 + v2)
    grad[:, 1] += lam * 2 * (-v3 + v4)
    grad[:, 2] += lam * 2 * (v1 + v2 + v3 + v4)

    ii, jj = np.triu_indices(n, k=1)
    dx = cx[ii] - cx[jj]
    dy = cy[ii] - cy[jj]
    dists = np.sqrt(dx**2 + dy**2 + 1e-30)
    r_pair = r[ii] + r[jj]
    gaps = r_pair - dists
    mask = gaps > 0
    if mask.any():
        gaps_m = gaps[mask]
        obj += lam * np.dot(gaps_m, gaps_m)
        coeff = 2 * lam * gaps_m / dists[mask]
        dx_m, dy_m = dx[mask], dy[mask]
        ii_m, jj_m = ii[mask], jj[mask]
        np.add.at(grad[:, 0], ii_m, -coeff * dx_m)
        np.add.at(grad[:, 0], jj_m, coeff * dx_m)
        np.add.at(grad[:, 1], ii_m, -coeff * dy_m)
        np.add.at(grad[:, 1], jj_m, coeff * dy_m)
        r_coeff = 2 * lam * gaps_m
        np.add.at(grad[:, 2], ii_m, r_coeff)
        np.add.at(grad[:, 2], jj_m, r_coeff)

    return obj, grad.ravel()


def max_violation(x, n):
    arr = x.reshape(n, 3)
    cx, cy, r = arr[:, 0], arr[:, 1], arr[:, 2]
    viol = max(0, np.max(np.maximum(0, r - cx)))
    viol = max(viol, np.max(np.maximum(0, cx + r - 1.0)))
    viol = max(viol, np.max(np.maximum(0, r - cy)))
    viol = max(viol, np.max(np.maximum(0, cy + r - 1.0)))
    ii, jj = np.triu_indices(n, k=1)
    dists = np.sqrt((cx[ii]-cx[jj])**2 + (cy[ii]-cy[jj])**2)
    overlap = np.max(np.maximum(0, r[ii]+r[jj] - dists))
    return max(viol, overlap)


def fix_solution(x, n):
    x = x.copy()
    for i in range(n):
        r = np.clip(x[3*i+2], 0.001, 0.499)
        x[3*i+2] = r
        x[3*i] = np.clip(x[3*i], r + 1e-12, 1.0 - r - 1e-12)
        x[3*i+1] = np.clip(x[3*i+1], r + 1e-12, 1.0 - r - 1e-12)
    return x


def optimize(x0, n):
    bounds = [(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)] * n
    x = fix_solution(x0, n)
    for lam in [10, 100, 1000, 10000, 100000, 1000000, 1e7, 1e8, 1e9, 1e10]:
        result = minimize(
            penalized_obj_grad, x, args=(n, lam), method='L-BFGS-B',
            jac=True, bounds=bounds,
            options={'maxiter': 5000, 'ftol': 1e-20, 'gtol': 1e-14, 'maxcor': 20}
        )
        x = fix_solution(result.x, n)
    return x


def find_gaps(x, n):
    """Find largest gaps in the packing where a circle could grow."""
    arr = x.reshape(n, 3)
    cx, cy, r = arr[:, 0], arr[:, 1], arr[:, 2]

    # Sample grid points and find max inscribed circle at each
    grid = np.linspace(0.05, 0.95, 20)
    gaps = []
    for gx in grid:
        for gy in grid:
            # Distance to walls
            d_wall = min(gx, 1-gx, gy, 1-gy)
            # Distance to each circle
            d_circles = np.sqrt((cx - gx)**2 + (cy - gy)**2) - r
            d_min = min(d_wall, np.min(d_circles))
            if d_min > 0.01:
                gaps.append((gx, gy, d_min))

    gaps.sort(key=lambda x: -x[2])
    return gaps[:10]


def refine_systematic(solution_path, n_rounds=5, base_seed=456):
    """Systematically move each circle to gap locations and re-optimize."""
    with open(solution_path) as f:
        data = json.load(f)
    circles = data['circles']
    n = len(circles)

    x_best = np.zeros(3 * n)
    for i, (cx, cy, r) in enumerate(circles):
        x_best[3*i] = cx
        x_best[3*i+1] = cy
        x_best[3*i+2] = r

    best_metric = sum(x_best[3*i+2] for i in range(n))
    print(f"Starting: n={n}, metric={best_metric:.10f}")

    rng = np.random.RandomState(base_seed)
    total_improved = 0

    for round_idx in range(n_rounds):
        improved_this_round = 0

        # Sort circles by radius (try moving smallest first)
        radii = [(x_best[3*i+2], i) for i in range(n)]
        radii.sort()

        for r_val, ci in radii:
            # Find gaps in the current packing (excluding circle ci)
            x_temp = x_best.copy()
            # Shrink circle ci to find gaps
            old_r = x_temp[3*ci+2]
            old_cx = x_temp[3*ci]
            old_cy = x_temp[3*ci+1]

            gaps = find_gaps(x_best, n)

            for gx, gy, gr in gaps[:5]:
                # Try moving circle ci to the gap
                x_trial = x_best.copy()
                x_trial[3*ci] = gx
                x_trial[3*ci+1] = gy
                x_trial[3*ci+2] = min(gr * 0.9, 0.499)

                # Optimize
                x_opt = optimize(x_trial, n)
                metric = sum(x_opt[3*i+2] for i in range(n))
                viol = max_violation(x_opt, n)

                if viol < 1e-6 and metric > best_metric:
                    best_metric = metric
                    x_best = x_opt.copy()
                    improved_this_round += 1
                    total_improved += 1
                    print(f"  Round {round_idx+1}, circle {ci} → gap ({gx:.2f},{gy:.2f}): {metric:.10f} ***")
                    break

            # Also try random repositioning
            for _ in range(3):
                x_trial = x_best.copy()
                x_trial[3*ci] = rng.uniform(0.05, 0.95)
                x_trial[3*ci+1] = rng.uniform(0.05, 0.95)
                x_trial[3*ci+2] = old_r * rng.uniform(0.5, 1.5)

                x_opt = optimize(x_trial, n)
                metric = sum(x_opt[3*i+2] for i in range(n))
                viol = max_violation(x_opt, n)

                if viol < 1e-6 and metric > best_metric:
                    best_metric = metric
                    x_best = x_opt.copy()
                    improved_this_round += 1
                    total_improved += 1
                    print(f"  Round {round_idx+1}, circle {ci} random: {metric:.10f} ***")
                    break

        print(f"  Round {round_idx+1} complete: {improved_this_round} improvements, best={best_metric:.10f}")

        if improved_this_round == 0:
            # Try pair swaps
            for _ in range(n):
                i, j = rng.choice(n, 2, replace=False)
                x_trial = x_best.copy()
                # Swap positions
                x_trial[3*i], x_trial[3*j] = x_trial[3*j], x_trial[3*i]
                x_trial[3*i+1], x_trial[3*j+1] = x_trial[3*j+1], x_trial[3*i+1]
                # Keep own radii
                x_opt = optimize(x_trial, n)
                metric = sum(x_opt[3*k+2] for k in range(n))
                viol = max_violation(x_opt, n)
                if viol < 1e-6 and metric > best_metric:
                    best_metric = metric
                    x_best = x_opt.copy()
                    total_improved += 1
                    print(f"  Swap {i}<->{j}: {metric:.10f} ***")

        if improved_this_round == 0 and round_idx > 1:
            print("  No improvement in 2 rounds, stopping.")
            break

    # Save
    circles_out = [[float(x_best[3*i]), float(x_best[3*i+1]), float(x_best[3*i+2])] for i in range(n)]
    with open(solution_path, 'w') as f:
        json.dump({"circles": circles_out}, f, indent=2)
    print(f"\nFinal: metric={best_metric:.10f}")
    print(f"Saved to {solution_path}")
    return best_metric


if __name__ == '__main__':
    path = sys.argv[1]
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    t0 = time.time()
    metric = refine_systematic(path, n_rounds=rounds)
    print(f"Time: {time.time()-t0:.1f}s")
