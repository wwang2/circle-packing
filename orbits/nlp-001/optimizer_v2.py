"""
V2: More aggressive optimization with focus on what worked in V1.
Key changes:
- More poisson disk variants (the best performers)
- Polish top 15 instead of 5
- Multiple rounds of SLSQP polish
- Basin-hopping style perturbation of best solutions
"""

import numpy as np
from scipy.optimize import minimize
import json
import sys
import time
import math
from pathlib import Path


def compute_objective_and_penalty(x, n, penalty_weight):
    xx = x[0::3]
    yy = x[1::3]
    rr = x[2::3]
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
    xx = x[0::3]
    yy = x[1::3]
    rr = x[2::3]
    grad[2::3] = -1.0
    vl = np.maximum(0, rr - xx)
    vr = np.maximum(0, xx + rr - 1.0)
    vb = np.maximum(0, rr - yy)
    vt = np.maximum(0, yy + rr - 1.0)
    grad[0::3] += penalty_weight * (-2 * vl + 2 * vr)
    grad[1::3] += penalty_weight * (-2 * vb + 2 * vt)
    grad[2::3] += penalty_weight * (2 * vl + 2 * vr + 2 * vb + 2 * vt)
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


# ============================================================
# Initializations - focused on what works
# ============================================================

def poisson_disk_init(n, seed=42, min_dist_factor=0.8):
    rng = np.random.RandomState(seed)
    min_dist = min_dist_factor / math.sqrt(n)
    r_est = min_dist / 2.5
    margin = max(r_est, 0.02)
    positions = []
    for _ in range(50000):
        if len(positions) >= n:
            break
        x = rng.uniform(margin, 1 - margin)
        y = rng.uniform(margin, 1 - margin)
        ok = all((x-px)**2 + (y-py)**2 >= min_dist**2 for px, py in positions)
        if ok:
            positions.append((x, y))
    while len(positions) < n:
        x = rng.uniform(margin, 1 - margin)
        y = rng.uniform(margin, 1 - margin)
        positions.append((x, y))
    return positions[:n], [r_est] * n


def hex_grid_init(n, offset_x=0.0, offset_y=0.0):
    cols = int(math.ceil(math.sqrt(n * 2 / math.sqrt(3))))
    rows = int(math.ceil(n / cols))
    r_est = min(0.5 / cols, 0.5 / (rows * math.sqrt(3)/2)) * 0.9
    positions = []
    for row in range(rows + 1):
        for col in range(cols + 1):
            if len(positions) >= n:
                break
            x = (col + 0.5) / (cols + 1) + offset_x * r_est
            y = (row + 0.5) / (rows + 1) + offset_y * r_est
            if row % 2 == 1:
                x += 0.5 / (cols + 1)
            positions.append((np.clip(x, 0.02, 0.98), np.clip(y, 0.02, 0.98)))
    return positions[:n], [r_est] * n


def concentric_rings_init(n, seed=0):
    rng = np.random.RandomState(seed)
    positions = [(0.5, 0.5)]
    for ring_n, ring_rad in [(6, 0.16), (12, 0.32), (18, 0.45)]:
        offset = rng.uniform(0, 2 * math.pi)
        for k in range(ring_n):
            if len(positions) >= n:
                break
            angle = 2 * math.pi * k / ring_n + offset
            positions.append((np.clip(0.5 + ring_rad * math.cos(angle), 0.04, 0.96),
                            np.clip(0.5 + ring_rad * math.sin(angle), 0.04, 0.96)))
    while len(positions) < n:
        positions.append((rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95)))
    return positions[:n], [0.35 / math.sqrt(n)] * n


def sunflower_init(n):
    golden_angle = math.pi * (3 - math.sqrt(5))
    positions = []
    for i in range(n):
        r = 0.45 * math.sqrt((i + 0.5) / n)
        theta = i * golden_angle
        positions.append((np.clip(0.5 + r * math.cos(theta), 0.04, 0.96),
                         np.clip(0.5 + r * math.sin(theta), 0.04, 0.96)))
    return positions, [0.35 / math.sqrt(n)] * n


def random_init(n, seed=0):
    rng = np.random.RandomState(seed)
    r_est = 0.3 / math.sqrt(n)
    m = max(r_est, 0.02)
    return [(rng.uniform(m, 1-m), rng.uniform(m, 1-m)) for _ in range(n)], [r_est] * n


def perturb_solution(x, n, seed=0, scale=0.02):
    """Perturb an existing solution for basin-hopping."""
    rng = np.random.RandomState(seed)
    x_new = x.copy()
    # Perturb positions only, keep radii
    for i in range(n):
        x_new[3*i] += rng.normal(0, scale)
        x_new[3*i+1] += rng.normal(0, scale)
        x_new[3*i] = np.clip(x_new[3*i], 0.02, 0.98)
        x_new[3*i+1] = np.clip(x_new[3*i+1], 0.02, 0.98)
    return x_new


def generate_inits(n):
    inits = []
    # Hex variants
    for ox, oy in [(0,0), (0.3,0), (0,0.3), (0.3,0.3), (-0.3,0.2), (0.5,0.5)]:
        inits.append((f"hex_{ox}_{oy}", hex_grid_init(n, ox, oy)))
    # Rings
    for s in range(6):
        inits.append((f"rings_s{s}", concentric_rings_init(n, s)))
    # Sunflower
    inits.append(("sunflower", sunflower_init(n)))
    # Poisson disk - many variants (this was the best performer)
    for s in range(30):
        inits.append((f"poisson_s{s}", poisson_disk_init(n, seed=s)))
    for mdf in [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2]:
        for s in range(3):
            inits.append((f"poisson_md{mdf}_s{s}", poisson_disk_init(n, seed=100+s, min_dist_factor=mdf)))
    # Random
    for s in range(20):
        inits.append((f"rand_s{s}", random_init(n, s)))
    return inits


# ============================================================
# Optimization
# ============================================================

def pack_to_x(positions, radii):
    n = len(positions)
    x = np.zeros(3 * n)
    for i in range(n):
        x[3*i] = positions[i][0]
        x[3*i+1] = positions[i][1]
        x[3*i+2] = radii[i]
    return x


def x_to_circles(x, n):
    return [(x[3*i], x[3*i+1]) for i in range(n)], [x[3*i+2] for i in range(n)]


def lbfgsb_optimize(x0, n):
    bounds = [(1e-4, 1-1e-4), (1e-4, 1-1e-4), (1e-6, 0.5)] * n
    x = x0.copy()
    for pw in [1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]:
        result = minimize(
            compute_objective_and_penalty, x, args=(n, pw),
            jac=lambda x, n=n, pw=pw: compute_gradient(x, n, pw),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 600, 'ftol': 1e-14, 'maxfun': 15000}
        )
        x = result.x
    return x


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


def slsqp_polish(x, n, maxiter=5000):
    bounds = [(1e-6, 1-1e-6), (1e-6, 1-1e-6), (1e-6, 0.5)] * n
    constraints = get_slsqp_constraints(n)
    result = minimize(
        lambda x: -np.sum(x[2::3]), x,
        method='SLSQP', bounds=bounds, constraints=constraints,
        options={'maxiter': maxiter, 'ftol': 1e-15, 'disp': False}
    )
    return result.x


def validate_and_repair(x, n, tol=1e-10):
    positions, radii = x_to_circles(x, n)
    radii = list(radii)
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
        xi, yi = positions[i]
        ri = radii[i]
        for j in range(i+1, n):
            xj, yj = positions[j]
            rj = radii[j]
            if ri+rj - math.sqrt((xi-xj)**2+(yi-yj)**2) > tol:
                return False
    return True


def optimize_packing(n, verbose=True):
    if verbose:
        print(f"=== Optimizing n={n} ===")

    all_inits = generate_inits(n)
    if verbose:
        print(f"Phase 1: L-BFGS-B on {len(all_inits)} initializations")

    candidates = []  # (metric, x)
    best_metric = 0.0
    t_start = time.time()

    for idx, (name, (positions, radii)) in enumerate(all_inits):
        t0 = time.time()
        try:
            x0 = pack_to_x(positions, radii)
            x = lbfgsb_optimize(x0, n)
            pos, rad = validate_and_repair(x, n)
            metric = sum(rad)
            valid = check_valid(pos, rad)
            dt = time.time() - t0

            if valid and metric > 0.5:
                candidates.append((metric, pack_to_x(pos, rad)))
            if valid and metric > best_metric:
                best_metric = metric
                if verbose:
                    print(f"  [{idx+1}/{len(all_inits)}] {name}: {metric:.6f} ** BEST ** [{dt:.1f}s]")
            elif verbose and idx % 20 == 0:
                print(f"  [{idx+1}/{len(all_inits)}] {name}: {metric:.6f} {'ok' if valid else 'INV'} [{dt:.1f}s]")
        except Exception as e:
            if verbose and idx % 20 == 0:
                print(f"  [{idx+1}/{len(all_inits)}] {name}: ERR [{e}]")

    # Sort and take top candidates for SLSQP polish
    candidates.sort(key=lambda t: -t[0])
    num_polish = min(15, len(candidates))

    if verbose:
        print(f"\nPhase 2: SLSQP polish on top {num_polish} candidates")

    best_x = candidates[0][1] if candidates else None
    best_metric_final = candidates[0][0] if candidates else 0

    for rank in range(num_polish):
        metric_before, x_sol = candidates[rank]
        t0 = time.time()
        try:
            x_pol = slsqp_polish(x_sol, n, maxiter=5000)
            pos, rad = validate_and_repair(x_pol, n)
            metric = sum(rad)
            valid = check_valid(pos, rad)
            dt = time.time() - t0

            if valid and metric > best_metric_final:
                best_metric_final = metric
                best_x = pack_to_x(pos, rad)
                if verbose:
                    print(f"  #{rank+1}: {metric_before:.6f} -> {metric:.6f} ** NEW BEST ** [{dt:.1f}s]")
            elif verbose:
                print(f"  #{rank+1}: {metric_before:.6f} -> {metric:.6f} [{dt:.1f}s]")
        except Exception as e:
            if verbose:
                print(f"  #{rank+1}: ERR [{e}]")

    # Phase 3: Basin-hopping on best solution
    if best_x is not None and verbose:
        print(f"\nPhase 3: Basin-hopping perturbations on best ({best_metric_final:.6f})")

    if best_x is not None:
        for seed in range(30):
            for scale in [0.005, 0.01, 0.02, 0.03]:
                try:
                    x_pert = perturb_solution(best_x, n, seed=seed*100+int(scale*1000), scale=scale)
                    x_opt = lbfgsb_optimize(x_pert, n)
                    x_pol = slsqp_polish(x_opt, n, maxiter=3000)
                    pos, rad = validate_and_repair(x_pol, n)
                    metric = sum(rad)
                    valid = check_valid(pos, rad)

                    if valid and metric > best_metric_final:
                        best_metric_final = metric
                        best_x = pack_to_x(pos, rad)
                        if verbose:
                            print(f"  Perturb s={seed} sc={scale}: {metric:.6f} ** NEW BEST **")
                except Exception:
                    pass

    total_time = time.time() - t_start
    if verbose:
        print(f"\nFinal metric: {best_metric_final:.10f}")
        print(f"Total time: {total_time:.1f}s")

    if best_x is not None:
        positions, radii = validate_and_repair(best_x, n)
        return positions, radii, sum(radii)
    return [], [], 0.0


def save_solution(positions, radii, path):
    circles = [[positions[i][0], positions[i][1], radii[i]] for i in range(len(positions))]
    with open(path, 'w') as f:
        json.dump({"circles": circles}, f, indent=2)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 26
    output = sys.argv[2] if len(sys.argv) > 2 else str(Path(__file__).parent / f"solution_n{n}.json")
    positions, radii, metric = optimize_packing(n, verbose=True)
    if positions:
        save_solution(positions, radii, output)
        print(f"Solution saved to {output}")
    else:
        print("No valid solution found!")
        sys.exit(1)


if __name__ == "__main__":
    main()
