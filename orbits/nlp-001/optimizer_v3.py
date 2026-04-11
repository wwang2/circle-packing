"""
V3: Targeted search for n=26 with known-good topologies.

The best n=26 packings have specific structure:
- Mix of different-sized circles
- Larger circles tend to be near corners/edges
- Smaller circles fill gaps

Strategy: Start from many Poisson disk inits + known-arrangement seeds,
use fast penalty L-BFGS-B, then aggressive SLSQP polish on top candidates.
Also try: seeding from the known ~2.636 arrangement structure.
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


# ============================================================
# Diverse initializations
# ============================================================

def poisson_disk_init(n, seed=42, min_dist_factor=0.8):
    rng = np.random.RandomState(seed)
    min_dist = min_dist_factor / math.sqrt(n)
    r_est = min_dist / 2.5
    margin = max(r_est, 0.02)
    positions = []
    for _ in range(80000):
        if len(positions) >= n:
            break
        x = rng.uniform(margin, 1 - margin)
        y = rng.uniform(margin, 1 - margin)
        ok = all((x-px)**2 + (y-py)**2 >= min_dist**2 for px, py in positions)
        if ok:
            positions.append((x, y))
    while len(positions) < n:
        positions.append((rng.uniform(margin, 1-margin), rng.uniform(margin, 1-margin)))
    return positions[:n], [r_est] * n


def varied_radius_init(n, seed=0):
    """Init with varying radii -- larger near edges, smaller inside."""
    rng = np.random.RandomState(seed)
    positions, radii = poisson_disk_init(n, seed=seed)
    new_radii = []
    for x, y in positions:
        # Distance to nearest edge
        d_edge = min(x, y, 1-x, 1-y)
        # Larger radius if near center, smaller near edge (counterintuitive but
        # the optimizer will fix it)
        r = 0.03 + 0.04 * rng.random()
        new_radii.append(r)
    return positions, new_radii


def grid_5x6_init(n=26, seed=0):
    """5 columns x 6 rows = 30 slots, pick 26."""
    rng = np.random.RandomState(seed)
    cols, rows = 5, 6
    r_est = 0.08
    all_pos = []
    for row in range(rows):
        for col in range(cols):
            x = (col + 0.5) / cols
            y = (row + 0.5) / rows
            if row % 2 == 1:
                x += 0.5 / cols
            all_pos.append((np.clip(x, 0.05, 0.95), np.clip(y, 0.05, 0.95)))
    rng.shuffle(all_pos)
    return all_pos[:n], [r_est] * n


def grid_6x5_init(n=26, seed=0):
    """6 columns x 5 rows with hex offset."""
    rng = np.random.RandomState(seed)
    cols, rows = 6, 5
    r_est = 0.075
    all_pos = []
    for row in range(rows):
        for col in range(cols):
            x = (col + 0.5) / cols
            y = (row + 0.5) / rows
            if row % 2 == 1:
                x += 0.5 / cols
            all_pos.append((np.clip(x, 0.04, 0.96), np.clip(y, 0.04, 0.96)))
    rng.shuffle(all_pos)
    return all_pos[:n], [r_est] * n


def diamond_init(n=26):
    """Diamond/rhombus pattern."""
    positions = []
    layers = [1, 4, 6, 6, 5, 3, 1]  # Diamond shape summing to 26
    y_spacing = 1.0 / (len(layers) + 1)
    for row, count in enumerate(layers):
        y = (row + 1) * y_spacing
        x_spacing = 1.0 / (count + 1)
        for col in range(count):
            x = (col + 1) * x_spacing
            positions.append((np.clip(x, 0.04, 0.96), np.clip(y, 0.04, 0.96)))
    r_est = 0.06
    return positions[:n], [r_est] * n


def ring_based_init(n=26, seed=0):
    """Multiple concentric rings optimized for n=26."""
    rng = np.random.RandomState(seed)
    # Try: 1 center + 6 inner + 10 middle + 9 outer = 26
    configs = [
        (1, 0.0, 0.0),     # center
        (6, 0.17, 0.0),    # inner ring
        (10, 0.33, 0.0),   # middle ring
        (9, 0.46, 0.0),    # outer ring
    ]
    positions = []
    for count, radius, offset in configs:
        if count == 1:
            positions.append((0.5, 0.5))
        else:
            base_offset = rng.uniform(0, 2*math.pi)
            for k in range(count):
                angle = 2*math.pi*k/count + base_offset + offset
                x = 0.5 + radius * math.cos(angle)
                y = 0.5 + radius * math.sin(angle)
                positions.append((np.clip(x, 0.04, 0.96), np.clip(y, 0.04, 0.96)))
    r_est = 0.06
    return positions[:n], [r_est] * n


def generate_all_inits(n):
    inits = []
    # Poisson disk - heavy emphasis (best performers in V1)
    for s in range(50):
        inits.append((f"poisson_s{s}", poisson_disk_init(n, seed=s)))
    for mdf in [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1]:
        for s in range(5):
            inits.append((f"poisson_md{mdf}_s{s}", poisson_disk_init(n, seed=200+s, min_dist_factor=mdf)))
    # Varied radius
    for s in range(10):
        inits.append((f"varied_s{s}", varied_radius_init(n, seed=s)))
    # Grid patterns
    for s in range(5):
        inits.append((f"grid5x6_s{s}", grid_5x6_init(n, seed=s)))
        inits.append((f"grid6x5_s{s}", grid_6x5_init(n, seed=s)))
    # Diamond
    inits.append(("diamond", diamond_init(n)))
    # Ring-based
    for s in range(10):
        inits.append((f"ring_s{s}", ring_based_init(n, seed=s)))
    return inits


# ============================================================
# Optimization
# ============================================================

def pack_to_x(positions, radii):
    n = len(positions)
    x = np.zeros(3 * n)
    for i in range(n):
        x[3*i] = positions[i][0]; x[3*i+1] = positions[i][1]; x[3*i+2] = radii[i]
    return x


def lbfgsb_optimize(x0, n, penalty_schedule=None):
    if penalty_schedule is None:
        penalty_schedule = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
    bounds = [(1e-4, 1-1e-4), (1e-4, 1-1e-4), (1e-6, 0.5)] * n
    x = x0.copy()
    for pw in penalty_schedule:
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


def optimize_packing(n=26, verbose=True):
    all_inits = generate_all_inits(n)
    if verbose:
        print(f"Phase 1: L-BFGS-B on {len(all_inits)} initializations", flush=True)

    candidates = []
    best_penalty_metric = 0.0
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
            if valid and metric > best_penalty_metric:
                best_penalty_metric = metric
                if verbose:
                    print(f"  [{idx+1}/{len(all_inits)}] {name}: {metric:.6f} ** BEST ** [{dt:.1f}s]", flush=True)
            elif verbose and idx % 25 == 0:
                print(f"  [{idx+1}/{len(all_inits)}] {name}: {metric:.6f} {'ok' if valid else 'INV'} [{dt:.1f}s]", flush=True)
        except Exception as e:
            pass

    # Sort and polish top candidates
    candidates.sort(key=lambda t: -t[0])
    num_polish = min(20, len(candidates))
    if verbose:
        print(f"\nPhase 2: SLSQP polish on top {num_polish} (from {len(candidates)} valid)", flush=True)

    best_metric = 0.0
    best_x = None

    for rank in range(num_polish):
        metric_before, x_sol = candidates[rank]
        t0 = time.time()
        try:
            x_pol = slsqp_polish(x_sol, n, maxiter=8000)
            pos, rad = validate_and_repair(x_pol, n)
            metric = sum(rad)
            valid = check_valid(pos, rad)
            dt = time.time() - t0
            if valid and metric > best_metric:
                best_metric = metric
                best_x = pack_to_x(pos, rad)
                if verbose:
                    print(f"  #{rank+1}: {metric_before:.6f} -> {metric:.6f} ** BEST ** [{dt:.1f}s]", flush=True)
            elif verbose and rank < 5:
                print(f"  #{rank+1}: {metric_before:.6f} -> {metric:.6f} [{dt:.1f}s]", flush=True)
        except:
            pass

    # Phase 3: basin-hopping on best
    if best_x is not None:
        if verbose:
            print(f"\nPhase 3: Basin-hopping from {best_metric:.10f}", flush=True)
        rng = np.random.RandomState(123)
        no_improve_count = 0
        for attempt in range(50):
            if no_improve_count >= 15:
                break
            scale = rng.choice([0.002, 0.005, 0.008, 0.01, 0.015, 0.02])
            x_pert = best_x.copy()
            for i in range(n):
                x_pert[3*i] = np.clip(x_pert[3*i] + rng.normal(0, scale), 0.01, 0.99)
                x_pert[3*i+1] = np.clip(x_pert[3*i+1] + rng.normal(0, scale), 0.01, 0.99)
            try:
                x_opt = lbfgsb_optimize(x_pert, n)
                x_pol = slsqp_polish(x_opt, n, maxiter=5000)
                pos, rad = validate_and_repair(x_pol, n)
                metric = sum(rad)
                valid = check_valid(pos, rad)
                if valid and metric > best_metric + 1e-10:
                    best_metric = metric
                    best_x = pack_to_x(pos, rad)
                    no_improve_count = 0
                    if verbose:
                        print(f"  Attempt {attempt+1}: sc={scale:.3f} -> {metric:.10f} ** IMPROVED **", flush=True)
                else:
                    no_improve_count += 1
            except:
                no_improve_count += 1

    total_time = time.time() - t_start
    if verbose:
        print(f"\nFinal: {best_metric:.10f} in {total_time:.1f}s", flush=True)

    if best_x is not None:
        pos, rad = validate_and_repair(best_x, n)
        return pos, rad, sum(rad)
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
        print(f"Saved to {output}", flush=True)
    else:
        print("No valid solution!", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
