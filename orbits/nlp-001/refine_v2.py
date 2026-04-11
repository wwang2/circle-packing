"""
Aggressive refinement with:
1. Targeted single-circle perturbation (move one circle at a time)
2. Try COBYLA solver (different search path than SLSQP)
3. Larger basin-hopping with L-BFGS-B warm-start
4. Systematic scale exploration
"""

import numpy as np
from scipy.optimize import minimize
import json
import sys
import time
import math
from pathlib import Path


def compute_objective_and_penalty(x, n, penalty_weight):
    xx = x[0::3]; yy = x[1::3]; rr = x[2::3]
    obj = -np.sum(rr)
    vl = np.maximum(0, rr - xx); vr = np.maximum(0, xx + rr - 1.0)
    vb = np.maximum(0, rr - yy); vt = np.maximum(0, yy + rr - 1.0)
    contain_pen = np.sum(vl**2 + vr**2 + vb**2 + vt**2)
    dx = xx[:, None] - xx[None, :]; dy = yy[:, None] - yy[None, :]
    dist = np.sqrt(dx**2 + dy**2 + 1e-30)
    min_dist = rr[:, None] + rr[None, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    overlap = np.maximum(0, min_dist - dist)
    overlap_pen = np.sum((overlap[mask])**2)
    obj += penalty_weight * (contain_pen + overlap_pen)
    return obj


def compute_gradient(x, n, penalty_weight):
    grad = np.zeros_like(x)
    xx = x[0::3]; yy = x[1::3]; rr = x[2::3]
    grad[2::3] = -1.0
    vl = np.maximum(0, rr - xx); vr = np.maximum(0, xx + rr - 1.0)
    vb = np.maximum(0, rr - yy); vt = np.maximum(0, yy + rr - 1.0)
    grad[0::3] += penalty_weight * (-2*vl + 2*vr)
    grad[1::3] += penalty_weight * (-2*vb + 2*vt)
    grad[2::3] += penalty_weight * (2*vl + 2*vr + 2*vb + 2*vt)
    dx = xx[:, None] - xx[None, :]; dy = yy[:, None] - yy[None, :]
    dist = np.sqrt(dx**2 + dy**2 + 1e-30)
    min_dist = rr[:, None] + rr[None, :]
    overlap = np.maximum(0, min_dist - dist)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    active = (overlap > 0) & mask
    if np.any(active):
        factor = np.zeros((n, n))
        factor[active] = 2.0 * overlap[active] / dist[active]
        for i in range(n):
            grad[3*i] += penalty_weight * (np.sum(-factor[i,:]*dx[i,:]) + np.sum(factor[:,i]*dx[:,i]))
            grad[3*i+1] += penalty_weight * (np.sum(-factor[i,:]*dy[i,:]) + np.sum(factor[:,i]*dy[:,i]))
            grad[3*i+2] += penalty_weight * (np.sum(factor[i,:]) + np.sum(factor[:,i]))
    return grad


def get_slsqp_constraints(n):
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
                'fun': lambda x, i=i, j=j: math.sqrt(
                    (x[3*i]-x[3*j])**2 + (x[3*i+1]-x[3*j+1])**2
                ) - x[3*i+2] - x[3*j+2]
            })
    return constraints


def lbfgsb_optimize(x0, n, schedule=None):
    if schedule is None:
        schedule = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
    bounds = [(1e-4, 1-1e-4), (1e-4, 1-1e-4), (1e-6, 0.5)] * n
    x = x0.copy()
    for pw in schedule:
        result = minimize(
            compute_objective_and_penalty, x, args=(n, pw),
            jac=lambda x, n=n, pw=pw: compute_gradient(x, n, pw),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-15}
        )
        x = result.x
    return x


def slsqp_polish(x, n, maxiter=8000):
    bounds = [(1e-6, 1-1e-6), (1e-6, 1-1e-6), (1e-6, 0.5)] * n
    constraints = get_slsqp_constraints(n)
    result = minimize(
        lambda x: -np.sum(x[2::3]), x,
        method='SLSQP', bounds=bounds, constraints=constraints,
        options={'maxiter': maxiter, 'ftol': 1e-15}
    )
    return result.x


def validate_and_repair(x, n, tol=1e-10):
    positions = [(x[3*i], x[3*i+1]) for i in range(n)]
    radii = [x[3*i+2] for i in range(n)]
    for _ in range(200):
        changed = False
        for i in range(n):
            xi, yi = positions[i]; r = radii[i]
            r_new = min(r, xi - tol, 1 - xi - tol, yi - tol, 1 - yi - tol)
            if r_new < r:
                radii[i] = max(r_new, 1e-8); changed = True
        for i in range(n):
            xi, yi = positions[i]; ri = radii[i]
            for j in range(i+1, n):
                xj, yj = positions[j]; rj = radii[j]
                dist = math.sqrt((xi-xj)**2 + (yi-yj)**2)
                if ri + rj > dist - tol and ri + rj > 0:
                    scale = max((dist - 2*tol) / (ri + rj), 0.01)
                    if scale < 1:
                        radii[i] *= scale; radii[j] *= scale; changed = True
        if not changed:
            break
    return positions, radii


def check_valid(positions, radii, tol=1e-10):
    n = len(positions)
    for i in range(n):
        x, y = positions[i]; r = radii[i]
        if r <= 0 or r-x > tol or x+r-1 > tol or r-y > tol or y+r-1 > tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            xi, yi = positions[i]; ri = radii[i]
            xj, yj = positions[j]; rj = radii[j]
            if ri+rj - math.sqrt((xi-xj)**2+(yi-yj)**2) > tol:
                return False
    return True


def pack_to_x(positions, radii):
    n = len(positions)
    x = np.zeros(3*n)
    for i in range(n):
        x[3*i] = positions[i][0]; x[3*i+1] = positions[i][1]; x[3*i+2] = radii[i]
    return x


def try_improve(x, n):
    """Run L-BFGS-B + SLSQP and return (metric, x) if valid."""
    x_opt = lbfgsb_optimize(x, n)
    x_pol = slsqp_polish(x_opt, n, maxiter=5000)
    pos, rad = validate_and_repair(x_pol, n)
    metric = sum(rad)
    valid = check_valid(pos, rad)
    if valid:
        return metric, pack_to_x(pos, rad)
    return 0, x


def main():
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path

    with open(input_path) as f:
        data = json.load(f)
    circles = data["circles"]
    n = len(circles)
    x0 = np.zeros(3*n)
    for i, (cx, cy, r) in enumerate(circles):
        x0[3*i] = cx; x0[3*i+1] = cy; x0[3*i+2] = r

    best_metric = np.sum(x0[2::3])
    best_x = x0.copy()
    print(f"Loaded n={n}, metric={best_metric:.10f}", flush=True)

    rng = np.random.RandomState(999)

    # Phase 1: Targeted single-circle perturbation
    # Move each circle to a new random position, re-optimize
    print(f"\nPhase 1: Single-circle relocation", flush=True)
    for i in range(n):
        for trial in range(3):
            x_pert = best_x.copy()
            # Move circle i to random position
            x_pert[3*i] = rng.uniform(0.05, 0.95)
            x_pert[3*i+1] = rng.uniform(0.05, 0.95)
            x_pert[3*i+2] = rng.uniform(0.02, 0.15)
            try:
                metric, x_new = try_improve(x_pert, n)
                if metric > best_metric + 1e-8:
                    best_metric = metric
                    best_x = x_new
                    print(f"  Circle {i}, trial {trial}: {metric:.10f} ** IMPROVED **", flush=True)
            except:
                pass

    # Phase 2: Pair swap + re-optimize
    print(f"\nPhase 2: Top-10 pair swaps (from {best_metric:.10f})", flush=True)
    # Only try swapping the smallest circles (most likely to benefit)
    radii_sorted = sorted(range(n), key=lambda i: best_x[3*i+2])
    small_circles = radii_sorted[:10]  # 10 smallest
    for i in small_circles:
        for j in small_circles:
            if i >= j:
                continue
            x_swap = best_x.copy()
            x_swap[3*i], x_swap[3*j] = x_swap[3*j], x_swap[3*i]
            x_swap[3*i+1], x_swap[3*j+1] = x_swap[3*j+1], x_swap[3*i+1]
            try:
                x_pol = slsqp_polish(x_swap, n, maxiter=5000)
                pos, rad = validate_and_repair(x_pol, n)
                metric = sum(rad)
                if check_valid(pos, rad) and metric > best_metric + 1e-8:
                    best_metric = metric
                    best_x = pack_to_x(pos, rad)
                    print(f"  Swap ({i},{j}): {metric:.10f} ** IMPROVED **", flush=True)
            except:
                pass

    # Phase 3: Large perturbation basin-hopping
    print(f"\nPhase 3: Large perturbation ({best_metric:.10f})", flush=True)
    no_improve = 0
    for attempt in range(80):
        if no_improve >= 25:
            break
        scale = rng.choice([0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10])
        x_pert = best_x.copy()
        # Perturb a random subset of circles
        n_perturb = rng.randint(1, n+1)
        indices = rng.choice(n, size=n_perturb, replace=False)
        for i in indices:
            x_pert[3*i] = np.clip(x_pert[3*i] + rng.normal(0, scale), 0.01, 0.99)
            x_pert[3*i+1] = np.clip(x_pert[3*i+1] + rng.normal(0, scale), 0.01, 0.99)
            if rng.random() < 0.4:
                x_pert[3*i+2] = np.clip(x_pert[3*i+2] * rng.uniform(0.5, 1.5), 0.001, 0.49)
        try:
            metric, x_new = try_improve(x_pert, n)
            if metric > best_metric + 1e-10:
                best_metric = metric
                best_x = x_new
                no_improve = 0
                print(f"  #{attempt+1} sc={scale:.2f} np={n_perturb}: {metric:.10f} ** IMPROVED **", flush=True)
            else:
                no_improve += 1
                if attempt % 20 == 0:
                    print(f"  #{attempt+1}: {metric:.6f} (no improve x{no_improve})", flush=True)
        except:
            no_improve += 1

    print(f"\nFinal: {best_metric:.10f}", flush=True)

    # Save best
    pos, rad = validate_and_repair(best_x, n)
    circles = [[pos[i][0], pos[i][1], rad[i]] for i in range(n)]
    with open(output_path, 'w') as f:
        json.dump({"circles": circles}, f, indent=2)
    print(f"Saved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
