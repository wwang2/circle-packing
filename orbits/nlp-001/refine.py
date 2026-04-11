"""
Refine an existing solution with:
1. SLSQP polish (multiple restarts)
2. Basin-hopping perturbations
3. Load best known solution and improve it
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
    vl = np.maximum(0, rr - xx)
    vr = np.maximum(0, xx + rr - 1.0)
    vb = np.maximum(0, rr - yy)
    vt = np.maximum(0, yy + rr - 1.0)
    contain_pen = np.sum(vl**2 + vr**2 + vb**2 + vt**2)
    dx = xx[:, None] - xx[None, :]
    dy = yy[:, None] - yy[None, :]
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
    vl = np.maximum(0, rr - xx)
    vr = np.maximum(0, xx + rr - 1.0)
    vb = np.maximum(0, rr - yy)
    vt = np.maximum(0, yy + rr - 1.0)
    grad[0::3] += penalty_weight * (-2*vl + 2*vr)
    grad[1::3] += penalty_weight * (-2*vb + 2*vt)
    grad[2::3] += penalty_weight * (2*vl + 2*vr + 2*vb + 2*vt)
    dx = xx[:, None] - xx[None, :]
    dy = yy[:, None] - yy[None, :]
    dist = np.sqrt(dx**2 + dy**2 + 1e-30)
    min_dist = rr[:, None] + rr[None, :]
    overlap = np.maximum(0, min_dist - dist)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    active = (overlap > 0) & mask
    if np.any(active):
        factor = np.zeros((n, n))
        factor[active] = 2.0 * overlap[active] / dist[active]
        for i in range(n):
            f_ij = factor[i, :]
            grad[3*i] += penalty_weight * np.sum(-f_ij * dx[i, :])
            grad[3*i+1] += penalty_weight * np.sum(-f_ij * dy[i, :])
            grad[3*i+2] += penalty_weight * np.sum(factor[i, :])
            f_ji = factor[:, i]
            grad[3*i] += penalty_weight * np.sum(f_ji * dx[:, i])
            grad[3*i+1] += penalty_weight * np.sum(f_ji * dy[:, i])
            grad[3*i+2] += penalty_weight * np.sum(factor[:, i])
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


def lbfgsb_optimize(x0, n):
    bounds = [(1e-4, 1-1e-4), (1e-4, 1-1e-4), (1e-6, 0.5)] * n
    x = x0.copy()
    for pw in [1e3, 1e4, 1e5, 1e6, 1e7, 1e8]:
        result = minimize(
            compute_objective_and_penalty, x, args=(n, pw),
            jac=lambda x, n=n, pw=pw: compute_gradient(x, n, pw),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-14}
        )
        x = result.x
    return x


def slsqp_polish(x, n, maxiter=5000):
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
            xi, yi = positions[i]
            r = radii[i]
            r_new = min(r, xi - tol, 1 - xi - tol, yi - tol, 1 - yi - tol)
            if r_new < r:
                radii[i] = max(r_new, 1e-8)
                changed = True
        for i in range(n):
            xi, yi = positions[i]
            ri = radii[i]
            for j in range(i+1, n):
                xj, yj = positions[j]
                rj = radii[j]
                dist = math.sqrt((xi-xj)**2 + (yi-yj)**2)
                if ri + rj > dist - tol and ri + rj > 0:
                    scale = max((dist - 2*tol) / (ri + rj), 0.01)
                    if scale < 1:
                        radii[i] *= scale
                        radii[j] *= scale
                        changed = True
        if not changed:
            break
    return positions, radii


def check_valid(positions, radii, tol=1e-10):
    n = len(positions)
    for i in range(n):
        x, y = positions[i]
        r = radii[i]
        if r <= 0 or r-x > tol or x+r-1 > tol or r-y > tol or y+r-1 > tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            xi, yi = positions[i]; ri = radii[i]
            xj, yj = positions[j]; rj = radii[j]
            if ri+rj - math.sqrt((xi-xj)**2+(yi-yj)**2) > tol:
                return False
    return True


def load_solution(path):
    with open(path) as f:
        data = json.load(f)
    circles = data.get("circles", data)
    n = len(circles)
    x = np.zeros(3 * n)
    for i, (cx, cy, r) in enumerate(circles):
        x[3*i] = cx; x[3*i+1] = cy; x[3*i+2] = r
    return x, n


def save_solution(x, n, path):
    circles = [[x[3*i], x[3*i+1], x[3*i+2]] for i in range(n)]
    with open(path, 'w') as f:
        json.dump({"circles": circles}, f, indent=2)


def main():
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path

    x0, n = load_solution(input_path)
    initial_metric = np.sum(x0[2::3])
    print(f"Loaded n={n}, initial metric={initial_metric:.10f}", flush=True)

    best_x = x0.copy()
    best_metric = initial_metric

    # Phase 1: Direct SLSQP polish (multiple rounds)
    print("\nPhase 1: SLSQP polish rounds", flush=True)
    x = x0.copy()
    for round_i in range(5):
        t0 = time.time()
        x = slsqp_polish(x, n, maxiter=10000)
        pos, rad = validate_and_repair(x, n)
        metric = sum(rad)
        valid = check_valid(pos, rad)
        x_repaired = np.zeros(3*n)
        for i in range(n):
            x_repaired[3*i] = pos[i][0]
            x_repaired[3*i+1] = pos[i][1]
            x_repaired[3*i+2] = rad[i]
        dt = time.time() - t0
        if valid and metric > best_metric:
            best_metric = metric
            best_x = x_repaired.copy()
            print(f"  Round {round_i+1}: {metric:.10f} ** IMPROVED ** [{dt:.1f}s]", flush=True)
        else:
            print(f"  Round {round_i+1}: {metric:.10f} {'ok' if valid else 'INV'} [{dt:.1f}s]", flush=True)
        x = x_repaired

    # Phase 2: Basin-hopping with perturbation
    print(f"\nPhase 2: Basin-hopping from best ({best_metric:.10f})", flush=True)
    rng = np.random.RandomState(42)

    for attempt in range(40):
        scale = rng.choice([0.003, 0.005, 0.01, 0.015, 0.02, 0.03])
        x_pert = best_x.copy()
        # Perturb positions
        for i in range(n):
            x_pert[3*i] += rng.normal(0, scale)
            x_pert[3*i+1] += rng.normal(0, scale)
            x_pert[3*i] = np.clip(x_pert[3*i], 0.01, 0.99)
            x_pert[3*i+1] = np.clip(x_pert[3*i+1], 0.01, 0.99)

        try:
            # L-BFGS-B to get feasible
            x_opt = lbfgsb_optimize(x_pert, n)
            # SLSQP polish
            x_pol = slsqp_polish(x_opt, n, maxiter=5000)
            pos, rad = validate_and_repair(x_pol, n)
            metric = sum(rad)
            valid = check_valid(pos, rad)

            if valid and metric > best_metric:
                x_new = np.zeros(3*n)
                for i in range(n):
                    x_new[3*i] = pos[i][0]
                    x_new[3*i+1] = pos[i][1]
                    x_new[3*i+2] = rad[i]
                best_metric = metric
                best_x = x_new
                print(f"  Attempt {attempt+1}: sc={scale:.3f} -> {metric:.10f} ** NEW BEST **", flush=True)
            elif attempt % 10 == 0:
                print(f"  Attempt {attempt+1}: sc={scale:.3f} -> {metric:.6f} {'ok' if valid else 'INV'}", flush=True)
        except Exception as e:
            if attempt % 10 == 0:
                print(f"  Attempt {attempt+1}: ERR", flush=True)

    print(f"\nFinal metric: {best_metric:.10f}", flush=True)

    # Save
    pos, rad = validate_and_repair(best_x, n)
    final_metric = sum(rad)
    save_solution(best_x if check_valid(*validate_and_repair(best_x, n)) else x0, n, output_path)
    print(f"Saved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
