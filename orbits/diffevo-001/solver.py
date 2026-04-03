"""
Circle packing solver: multi-start with smart init + L-BFGS-B penalty + SLSQP polish.
Maximize sum of radii for n circles in [0,1]^2.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist
import json
import sys
import time
from pathlib import Path


def penalized_obj(x, n, lam):
    """Penalty objective for L-BFGS-B: -(sum r) + lam * violations^2."""
    arr = x.reshape(n, 3)
    cx, cy, r = arr[:, 0], arr[:, 1], arr[:, 2]

    obj = -r.sum()

    # Containment
    cv = np.concatenate([
        np.maximum(0, r - cx),
        np.maximum(0, cx + r - 1.0),
        np.maximum(0, r - cy),
        np.maximum(0, cy + r - 1.0),
    ])
    obj += lam * np.dot(cv, cv)

    # Overlap: use pdist for speed
    dists = pdist(arr[:, :2])
    # Pairwise r_i + r_j via broadcasting
    r_sums = (r[:, None] + r[None, :])
    # Extract upper triangle
    ii, jj = np.triu_indices(n, k=1)
    r_pair = r_sums[ii, jj]
    overlaps = np.maximum(0, r_pair - dists)
    obj += lam * np.dot(overlaps, overlaps)

    return obj


def penalized_obj_grad(x, n, lam):
    """Objective + gradient for L-BFGS-B."""
    arr = x.reshape(n, 3)
    cx, cy, r = arr[:, 0], arr[:, 1], arr[:, 2]
    grad = np.zeros_like(arr)

    obj = -r.sum()
    grad[:, 2] = -1.0  # d/dr of -sum(r)

    # Containment penalties
    v1 = np.maximum(0, r - cx)
    v2 = np.maximum(0, cx + r - 1.0)
    v3 = np.maximum(0, r - cy)
    v4 = np.maximum(0, cy + r - 1.0)

    obj += lam * (np.dot(v1, v1) + np.dot(v2, v2) + np.dot(v3, v3) + np.dot(v4, v4))

    # Gradients for containment
    grad[:, 0] += lam * 2 * (-v1 + v2)
    grad[:, 1] += lam * 2 * (-v3 + v4)
    grad[:, 2] += lam * 2 * (v1 + v2 + v3 + v4)

    # Overlap penalties
    ii, jj = np.triu_indices(n, k=1)
    dx = cx[ii] - cx[jj]
    dy = cy[ii] - cy[jj]
    dist_sq = dx**2 + dy**2
    dists = np.sqrt(dist_sq + 1e-30)  # avoid div by zero
    r_pair = r[ii] + r[jj]
    gaps = r_pair - dists
    mask = gaps > 0
    if mask.any():
        gaps_m = gaps[mask]
        obj += lam * np.dot(gaps_m, gaps_m)

        # Gradient of overlap penalty: d/dx of lam*(r_i+r_j-dist)^2
        # d/dcx_i = 2*lam*gap * (-dx/dist) = -2*lam*gap*dx/dist
        coeff = 2 * lam * gaps_m / dists[mask]
        dx_m, dy_m = dx[mask], dy[mask]
        ii_m, jj_m = ii[mask], jj[mask]

        np.add.at(grad[:, 0], ii_m, -coeff * dx_m)
        np.add.at(grad[:, 0], jj_m, coeff * dx_m)
        np.add.at(grad[:, 1], ii_m, -coeff * dy_m)
        np.add.at(grad[:, 1], jj_m, coeff * dy_m)

        # d/dr_i = 2*lam*gap * 1
        r_coeff = 2 * lam * gaps_m
        np.add.at(grad[:, 2], ii_m, r_coeff)
        np.add.at(grad[:, 2], jj_m, r_coeff)

    return obj, grad.ravel()


def make_slsqp_constraints(n):
    """Build constraint list for SLSQP final polish."""
    cons = []
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i] - x[3*i+2]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[3*i] - x[3*i+2]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+1] - x[3*i+2]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[3*i+1] - x[3*i+2]})
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: np.sqrt(
                    (x[3*i] - x[3*j])**2 + (x[3*i+1] - x[3*j+1])**2
                ) - x[3*i+2] - x[3*j+2]
            })
    return cons


def fix_solution(x, n):
    """Clip to satisfy containment."""
    x = x.copy()
    for i in range(n):
        r = np.clip(x[3*i+2], 0.001, 0.499)
        x[3*i+2] = r
        x[3*i] = np.clip(x[3*i], r + 1e-12, 1.0 - r - 1e-12)
        x[3*i+1] = np.clip(x[3*i+1], r + 1e-12, 1.0 - r - 1e-12)
    return x


def is_feasible(x, n, tol=1e-9):
    """Check if solution is feasible (containment + non-overlap)."""
    arr = x.reshape(n, 3)
    cx, cy, r = arr[:, 0], arr[:, 1], arr[:, 2]

    # Containment
    if np.any(r - cx > tol) or np.any(cx + r - 1.0 > tol):
        return False
    if np.any(r - cy > tol) or np.any(cy + r - 1.0 > tol):
        return False

    # Non-overlap
    ii, jj = np.triu_indices(n, k=1)
    dx = cx[ii] - cx[jj]
    dy = cy[ii] - cy[jj]
    dists = np.sqrt(dx**2 + dy**2)
    r_pair = r[ii] + r[jj]
    if np.any(r_pair - dists > tol):
        return False

    return True


def max_violation(x, n):
    """Return maximum constraint violation."""
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


def hex_init(n, rng, noise=0.02):
    """Hexagonal grid init."""
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    dx = 1.0 / (cols + 1)
    dy = 1.0 / (rows + 1)
    r_est = min(dx, dy) * 0.42

    positions = []
    for row in range(rows):
        for col in range(cols):
            cx = (col + 1) * dx + (dx * 0.5 if row % 2 else 0)
            cy = (row + 1) * dy
            positions.append((cx, cy))

    x0 = np.zeros(3 * n)
    for i in range(n):
        if i < len(positions):
            cx, cy = positions[i]
        else:
            cx, cy = rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)
        cx += rng.uniform(-noise, noise)
        cy += rng.uniform(-noise, noise)
        cx = np.clip(cx, r_est + 0.01, 1.0 - r_est - 0.01)
        cy = np.clip(cy, r_est + 0.01, 1.0 - r_est - 0.01)
        x0[3*i] = cx
        x0[3*i+1] = cy
        x0[3*i+2] = r_est
    return x0


def random_init(n, rng):
    """Random init."""
    r_est = 0.35 / np.sqrt(n)
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = rng.uniform(0.1, 0.9)
        x0[3*i+1] = rng.uniform(0.1, 0.9)
        x0[3*i+2] = r_est * rng.uniform(0.5, 1.0)
    return x0


def concentric_init(n, rng, noise=0.01):
    """Concentric rings init."""
    r_est = 0.38 / np.sqrt(n)
    x0 = np.zeros(3 * n)
    x0[0] = 0.5 + rng.uniform(-noise, noise)
    x0[1] = 0.5 + rng.uniform(-noise, noise)
    x0[2] = r_est
    placed = 1
    ring = 1
    while placed < n:
        ring_r = ring * 2.2 * r_est
        num = max(1, min(int(2 * np.pi * ring_r / (2.1 * r_est)), n - placed))
        for k in range(num):
            if placed >= n:
                break
            angle = 2 * np.pi * k / num + rng.uniform(-0.1, 0.1)
            cx = 0.5 + ring_r * np.cos(angle) + rng.uniform(-noise, noise)
            cy = 0.5 + ring_r * np.sin(angle) + rng.uniform(-noise, noise)
            cx = np.clip(cx, r_est + 0.01, 1.0 - r_est - 0.01)
            cy = np.clip(cy, r_est + 0.01, 1.0 - r_est - 0.01)
            x0[3*placed] = cx
            x0[3*placed+1] = cy
            x0[3*placed+2] = r_est
            placed += 1
        ring += 1
    return x0


def optimize_one(x0, n, verbose=False):
    """Optimize one initialization: L-BFGS-B with increasing penalty, then SLSQP polish."""
    bounds = [(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)] * n
    x = fix_solution(x0, n)

    # Progressive penalty L-BFGS-B
    for lam in [10, 100, 1000, 10000, 100000, 1000000, 1e7, 1e8, 1e9, 1e10]:
        result = minimize(
            penalized_obj_grad, x, args=(n, lam), method='L-BFGS-B',
            jac=True, bounds=bounds,
            options={'maxiter': 5000, 'ftol': 1e-20, 'gtol': 1e-14, 'maxcor': 20}
        )
        x = fix_solution(result.x, n)

    # Save pre-SLSQP result
    x_penalty = x.copy()
    metric_penalty = sum(x[3*i+2] for i in range(n))

    # Check penalty result feasibility
    viol_pen = max_violation(x_penalty, n)
    if viol_pen < 1e-10:
        # Already feasible from penalty method, try quick SLSQP
        if n <= 20:
            cons = make_slsqp_constraints(n)
            result = minimize(
                lambda x: -sum(x[3*i+2] for i in range(n)),
                x, method='SLSQP', bounds=bounds, constraints=cons,
                options={'maxiter': 500, 'ftol': 1e-15, 'disp': False}
            )
            x_slsqp = fix_solution(result.x, n)
            metric_slsqp = sum(x_slsqp[3*i+2] for i in range(n))
            viol_slsqp = max_violation(x_slsqp, n)
            if viol_slsqp < 1e-10 and metric_slsqp > metric_penalty:
                return x_slsqp, metric_slsqp
        return x_penalty, metric_penalty

    # Not feasible from penalty alone -- try SLSQP to fix
    if n <= 30:
        cons = make_slsqp_constraints(n)
        slsqp_maxiter = min(500, max(100, 5000 // n))
        result = minimize(
            lambda x: -sum(x[3*i+2] for i in range(n)),
            x, method='SLSQP', bounds=bounds, constraints=cons,
            options={'maxiter': slsqp_maxiter, 'ftol': 1e-15, 'disp': False}
        )
        x_slsqp = fix_solution(result.x, n)
        metric_slsqp = sum(x_slsqp[3*i+2] for i in range(n))
        viol_slsqp = max_violation(x_slsqp, n)
        if viol_slsqp < 1e-10:
            return x_slsqp, metric_slsqp

    # Return penalty result if close to feasible
    if viol_pen < 1e-6:
        return x_penalty, metric_penalty

    return x_penalty, 0.0


def solve(n, num_starts=30, base_seed=42, verbose=True):
    """Multi-start solver."""
    print(f"Solving n={n}, {num_starts} starts")
    rng = np.random.RandomState(base_seed)

    best_x = None
    best_metric = 0.0

    for trial in range(num_starts):
        seed = base_seed + trial * 71
        trial_rng = np.random.RandomState(seed)
        t0 = time.time()

        mod = trial % 5
        if mod == 0:
            x0 = hex_init(n, trial_rng, noise=0.02)
            init_type = "hex"
        elif mod == 1:
            x0 = random_init(n, trial_rng)
            init_type = "random"
        elif mod == 2:
            x0 = concentric_init(n, trial_rng, noise=0.01)
            init_type = "concentric"
        elif mod == 3:
            x0 = hex_init(n, trial_rng, noise=0.06)
            init_type = "hex-noisy"
        else:
            x0 = concentric_init(n, trial_rng, noise=0.03)
            init_type = "conc-noisy"

        x_opt, metric = optimize_one(x0, n, verbose=False)
        elapsed = time.time() - t0
        viol = max_violation(x_opt, n)

        improved = metric > best_metric
        if improved:
            best_metric = metric
            best_x = x_opt.copy()

        if verbose:
            flag = " ***" if improved else ""
            viol_str = f" viol={viol:.2e}" if viol > 1e-10 else ""
            print(f"  [{trial+1:3d}/{num_starts}] {init_type:12s} → {metric:.10f} ({elapsed:.1f}s){viol_str}{flag}")

    return best_x, best_metric


def save_solution(x, n, filepath):
    circles = [[float(x[3*i]), float(x[3*i+1]), float(x[3*i+2])] for i in range(n)]
    with open(filepath, 'w') as f:
        json.dump({"circles": circles}, f, indent=2)
    print(f"Saved to {filepath}")


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    num_starts = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    outdir = Path(__file__).parent
    t0 = time.time()
    x_best, metric_best = solve(n, num_starts=num_starts)
    elapsed = time.time() - t0

    outfile = outdir / f"solution_n{n}.json"
    if x_best is not None:
        save_solution(x_best, n, outfile)
        print(f"\n=== FINAL: n={n}, metric={metric_best:.10f}, time={elapsed:.1f}s ===")
    else:
        print(f"\n=== FAILED: no feasible solution found for n={n} ===")
