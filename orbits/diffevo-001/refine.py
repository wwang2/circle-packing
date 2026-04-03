"""
Basin-hopping refinement for circle packing solutions.
Takes a solution JSON, perturbs it, and re-optimizes.
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

    # Containment penalties
    v1 = np.maximum(0, r - cx)
    v2 = np.maximum(0, cx + r - 1.0)
    v3 = np.maximum(0, r - cy)
    v4 = np.maximum(0, cy + r - 1.0)

    obj += lam * (np.dot(v1, v1) + np.dot(v2, v2) + np.dot(v3, v3) + np.dot(v4, v4))

    grad[:, 0] += lam * 2 * (-v1 + v2)
    grad[:, 1] += lam * 2 * (-v3 + v4)
    grad[:, 2] += lam * 2 * (v1 + v2 + v3 + v4)

    # Overlap penalties
    ii, jj = np.triu_indices(n, k=1)
    dx = cx[ii] - cx[jj]
    dy = cy[ii] - cy[jj]
    dist_sq = dx**2 + dy**2
    dists = np.sqrt(dist_sq + 1e-30)
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

    viol = 0.0
    viol = max(viol, np.max(np.maximum(0, r - cx)))
    viol = max(viol, np.max(np.maximum(0, cx + r - 1.0)))
    viol = max(viol, np.max(np.maximum(0, r - cy)))
    viol = max(viol, np.max(np.maximum(0, cy + r - 1.0)))

    ii, jj = np.triu_indices(n, k=1)
    dx = cx[ii] - cx[jj]
    dy = cy[ii] - cy[jj]
    dists = np.sqrt(dx**2 + dy**2)
    r_pair = r[ii] + r[jj]
    overlap = np.max(np.maximum(0, r_pair - dists))
    viol = max(viol, overlap)
    return viol


def fix_solution(x, n):
    x = x.copy()
    for i in range(n):
        r = np.clip(x[3*i+2], 0.001, 0.499)
        x[3*i+2] = r
        x[3*i] = np.clip(x[3*i], r + 1e-12, 1.0 - r - 1e-12)
        x[3*i+1] = np.clip(x[3*i+1], r + 1e-12, 1.0 - r - 1e-12)
    return x


def optimize_lbfgsb(x0, n):
    """L-BFGS-B with progressive penalty."""
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


def perturb_solution(x, n, rng, perturbation_type='mixed', strength=0.05):
    """Perturb solution in various ways."""
    x = x.copy()
    arr = x.reshape(n, 3)

    if perturbation_type == 'position':
        # Perturb positions only
        arr[:, 0] += rng.uniform(-strength, strength, n)
        arr[:, 1] += rng.uniform(-strength, strength, n)

    elif perturbation_type == 'radius':
        # Perturb radii
        arr[:, 2] *= (1 + rng.uniform(-strength, strength, n))

    elif perturbation_type == 'swap':
        # Swap two random circles
        i, j = rng.choice(n, 2, replace=False)
        arr[i], arr[j] = arr[j].copy(), arr[i].copy()
        # Also perturb slightly
        arr[:, 0] += rng.uniform(-strength*0.2, strength*0.2, n)
        arr[:, 1] += rng.uniform(-strength*0.2, strength*0.2, n)

    elif perturbation_type == 'shake':
        # Large perturbation of a few circles
        num_shake = max(2, n // 6)
        indices = rng.choice(n, num_shake, replace=False)
        for idx in indices:
            arr[idx, 0] = rng.uniform(0.1, 0.9)
            arr[idx, 1] = rng.uniform(0.1, 0.9)
            arr[idx, 2] *= rng.uniform(0.5, 1.5)

    elif perturbation_type == 'mixed':
        # Mix of perturbations
        arr[:, 0] += rng.uniform(-strength, strength, n)
        arr[:, 1] += rng.uniform(-strength, strength, n)
        arr[:, 2] *= (1 + rng.uniform(-strength*0.5, strength*0.5, n))

    elif perturbation_type == 'rotate':
        # Rotate entire packing around center
        angle = rng.uniform(-0.3, 0.3)
        center_x = arr[:, 0].mean()
        center_y = arr[:, 1].mean()
        dx = arr[:, 0] - center_x
        dy = arr[:, 1] - center_y
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        arr[:, 0] = center_x + cos_a * dx - sin_a * dy
        arr[:, 1] = center_y + sin_a * dx + cos_a * dy
        # Small position noise
        arr[:, 0] += rng.uniform(-strength*0.1, strength*0.1, n)
        arr[:, 1] += rng.uniform(-strength*0.1, strength*0.1, n)

    return fix_solution(x, n)


def refine(solution_path, n_iters=200, base_seed=123):
    """Basin-hopping refinement."""
    # Load solution
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
    best_viol = max_violation(x_best, n)
    print(f"Starting: n={n}, metric={best_metric:.10f}, viol={best_viol:.2e}")

    rng = np.random.RandomState(base_seed)

    perturbation_types = ['position', 'radius', 'swap', 'shake', 'mixed', 'rotate']
    strengths = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3]

    no_improve_count = 0

    for trial in range(n_iters):
        # Choose perturbation
        ptype = perturbation_types[trial % len(perturbation_types)]
        strength = strengths[(trial // len(perturbation_types)) % len(strengths)]

        # Perturb
        x_pert = perturb_solution(x_best, n, rng, ptype, strength)

        # Optimize
        x_opt = optimize_lbfgsb(x_pert, n)

        metric = sum(x_opt[3*i+2] for i in range(n))
        viol = max_violation(x_opt, n)

        improved = (viol < 1e-6 and metric > best_metric)
        if improved:
            best_metric = metric
            best_viol = viol
            x_best = x_opt.copy()
            no_improve_count = 0

            print(f"  [{trial+1:4d}/{n_iters}] {ptype:10s} s={strength:.2f} → {metric:.10f} viol={viol:.2e} ***")
        else:
            no_improve_count += 1
            if trial < 10 or trial % 20 == 0:
                print(f"  [{trial+1:4d}/{n_iters}] {ptype:10s} s={strength:.2f} → {metric:.10f} viol={viol:.2e}")

    # Save
    circles_out = [[float(x_best[3*i]), float(x_best[3*i+1]), float(x_best[3*i+2])] for i in range(n)]
    with open(solution_path, 'w') as f:
        json.dump({"circles": circles_out}, f, indent=2)
    print(f"\nFinal: metric={best_metric:.10f}, viol={best_viol:.2e}")
    print(f"Saved to {solution_path}")

    return best_metric


if __name__ == '__main__':
    solution_path = sys.argv[1]
    n_iters = int(sys.argv[2]) if len(sys.argv) > 2 else 200

    t0 = time.time()
    metric = refine(solution_path, n_iters=n_iters)
    elapsed = time.time() - t0
    print(f"Time: {elapsed:.1f}s")
