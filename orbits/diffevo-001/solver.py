"""
Differential Evolution solver for circle packing in a unit square.
Maximize sum of radii for n circles.

Fast version: single DE run per seed with high penalty, then SLSQP polish.
Uses vectorized numpy for speed.
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist
import json
import sys
import time
from pathlib import Path


def objective_penalized(x, n, penalty_weight):
    """Vectorized objective: minimize -(sum of radii) + penalty * violations."""
    x_arr = x.reshape(n, 3)
    cx = x_arr[:, 0]
    cy = x_arr[:, 1]
    r = x_arr[:, 2]

    total_radius = r.sum()

    # Containment violations
    viol = np.zeros(4 * n)
    viol[:n] = np.maximum(0, r - cx)
    viol[n:2*n] = np.maximum(0, (cx + r) - 1.0)
    viol[2*n:3*n] = np.maximum(0, r - cy)
    viol[3*n:] = np.maximum(0, (cy + r) - 1.0)
    contain_penalty = np.sum(viol ** 2)

    # Non-overlap violations using pdist
    centers = x_arr[:, :2]
    dists = pdist(centers)  # pairwise distances

    # Pairwise sum of radii
    r_sums = []
    for i in range(n):
        for j in range(i+1, n):
            r_sums.append(r[i] + r[j])
    r_sums = np.array(r_sums)

    overlaps = np.maximum(0, r_sums - dists)
    overlap_penalty = np.sum(overlaps ** 2)

    return -total_radius + penalty_weight * (contain_penalty + overlap_penalty)


def get_constraints(n):
    """Build scipy constraint dicts for SLSQP."""
    constraints = []

    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i] - x[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[3*i] - x[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+1] - x[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[3*i+1] - x[3*i+2]})

    for i in range(n):
        for j in range(i+1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: np.sqrt(
                    (x[3*i] - x[3*j])**2 + (x[3*i+1] - x[3*j+1])**2
                ) - x[3*i+2] - x[3*j+2]
            })

    return constraints


def solve_de_single(n, seed=42, maxiter=1000, popsize=30, penalty=10000, verbose=True):
    """Single DE run."""
    bounds = []
    for i in range(n):
        bounds.append((0.01, 0.99))  # x
        bounds.append((0.01, 0.99))  # y
        bounds.append((0.005, 0.49))  # r

    if verbose:
        print(f"  DE: maxiter={maxiter}, popsize={popsize}, penalty={penalty}, seed={seed}")

    result = differential_evolution(
        objective_penalized,
        bounds=bounds,
        args=(n, penalty),
        strategy='best1bin',
        maxiter=maxiter,
        popsize=popsize,
        mutation=(0.5, 1.5),
        recombination=0.9,
        tol=1e-12,
        seed=seed,
        polish=False,
        disp=False,
        init='latinhypercube',
        updating='deferred',  # allows parallel-like updates
        workers=1,
    )

    actual_metric = sum(result.x[3*i+2] for i in range(n))
    if verbose:
        print(f"    metric={actual_metric:.6f}, converged={result.success}, nfev={result.nfev}")

    return result.x, actual_metric


def polish_slsqp(x0, n, verbose=True):
    """Polish solution with SLSQP."""
    constraints = get_constraints(n)

    bounds = []
    for i in range(n):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.499))

    def neg_sum_radii(x):
        return -sum(x[3*i+2] for i in range(n))

    result = minimize(
        neg_sum_radii,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 5000, 'ftol': 1e-15, 'disp': False}
    )

    metric = -result.fun
    if verbose:
        print(f"  SLSQP: metric={metric:.10f}, success={result.success}")

    return result.x, metric


def validate_and_fix(x, n):
    """Fix containment violations."""
    x = x.copy()
    for i in range(n):
        r = max(x[3*i+2], 0.001)
        x[3*i+2] = r
        x[3*i] = np.clip(x[3*i], r, 1.0 - r)
        x[3*i+1] = np.clip(x[3*i+1], r, 1.0 - r)
    return x


def solve(n, num_seeds=3, base_seed=42, maxiter=1000, popsize=30, verbose=True):
    """Multi-start DE + SLSQP polish."""
    print(f"Solving n={n} with {num_seeds} seeds, maxiter={maxiter}, popsize={popsize}")

    overall_best_x = None
    overall_best_metric = 0.0
    results_log = []

    for s in range(num_seeds):
        seed = base_seed + s * 137
        print(f"\n--- Seed {s+1}/{num_seeds} (seed={seed}) ---")
        t0 = time.time()

        # Progressive: start with lower penalty, warm-start next
        x_cur = None
        for penalty in [1000, 50000]:
            if x_cur is None:
                x_de, m = solve_de_single(n, seed=seed, maxiter=maxiter,
                                          popsize=popsize, penalty=penalty, verbose=verbose)
            else:
                # Warm start: use previous solution as part of init
                x_de, m = solve_de_single(n, seed=seed+1, maxiter=maxiter//2,
                                          popsize=popsize, penalty=penalty, verbose=verbose)
                if m < sum(x_cur[3*i+2] for i in range(n)):
                    x_de = x_cur  # keep better
            x_cur = x_de

        elapsed_de = time.time() - t0
        print(f"  DE done in {elapsed_de:.1f}s")

        # Polish
        x_fixed = validate_and_fix(x_cur, n)
        x_pol, metric_pol = polish_slsqp(x_fixed, n, verbose=verbose)
        x_pol = validate_and_fix(x_pol, n)

        # Second polish
        x_pol2, metric_pol2 = polish_slsqp(x_pol, n, verbose=verbose)
        x_pol2 = validate_and_fix(x_pol2, n)

        final_metric = sum(x_pol2[3*i+2] for i in range(n))
        elapsed = time.time() - t0
        print(f"  Final: {final_metric:.10f} ({elapsed:.1f}s)")
        results_log.append((seed, final_metric))

        if final_metric > overall_best_metric:
            overall_best_metric = final_metric
            overall_best_x = x_pol2.copy()
            print(f"  *** New best! ***")

    print(f"\nAll seeds: {results_log}")
    return overall_best_x, overall_best_metric


def save_solution(x, n, filepath):
    """Save solution as JSON."""
    circles = []
    for i in range(n):
        circles.append([float(x[3*i]), float(x[3*i+1]), float(x[3*i+2])])
    with open(filepath, 'w') as f:
        json.dump({"circles": circles}, f, indent=2)
    print(f"Saved to {filepath}")


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    num_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    maxiter = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    popsize = int(sys.argv[4]) if len(sys.argv) > 4 else 30

    outdir = Path(__file__).parent
    x_best, metric_best = solve(n, num_seeds=num_seeds, maxiter=maxiter, popsize=popsize)

    outfile = outdir / f"solution_n{n}.json"
    save_solution(x_best, n, outfile)
    print(f"\n=== FINAL: n={n}, metric={metric_best:.10f} ===")
