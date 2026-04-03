"""
Multi-start NLP optimizer for circle packing in unit square.
Maximizes sum of radii for n circles.

Uses vectorized numpy operations + L-BFGS-B with penalty method
for speed, then SLSQP polish on the best few solutions.
"""

import numpy as np
from scipy.optimize import minimize
import json
import sys
import time
import math
from pathlib import Path


# ============================================================
# Vectorized objective and penalty
# ============================================================

def compute_objective_and_penalty(x, n, penalty_weight):
    """Vectorized penalized objective: -sum(r) + penalty * violations."""
    xx = x[0::3]  # x coords
    yy = x[1::3]  # y coords
    rr = x[2::3]  # radii

    # Negative sum of radii
    obj = -np.sum(rr)

    # Containment violations
    viol_left = np.maximum(0, rr - xx)
    viol_right = np.maximum(0, xx + rr - 1.0)
    viol_bottom = np.maximum(0, rr - yy)
    viol_top = np.maximum(0, yy + rr - 1.0)
    contain_pen = np.sum(viol_left**2 + viol_right**2 + viol_bottom**2 + viol_top**2)

    # Non-overlap violations (vectorized)
    dx = xx[:, None] - xx[None, :]  # n x n
    dy = yy[:, None] - yy[None, :]
    dist = np.sqrt(dx**2 + dy**2 + 1e-30)  # avoid div by zero
    min_dist = rr[:, None] + rr[None, :]

    # Upper triangular only
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    overlap = np.maximum(0, min_dist - dist)
    overlap_pen = np.sum((overlap[mask])**2)

    obj += penalty_weight * (contain_pen + overlap_pen)
    return obj


def compute_gradient(x, n, penalty_weight):
    """Analytical gradient of the penalized objective."""
    grad = np.zeros_like(x)
    xx = x[0::3]
    yy = x[1::3]
    rr = x[2::3]

    # Gradient of -sum(r)
    grad[2::3] = -1.0

    # Containment gradient
    viol_left = np.maximum(0, rr - xx)
    viol_right = np.maximum(0, xx + rr - 1.0)
    viol_bottom = np.maximum(0, rr - yy)
    viol_top = np.maximum(0, yy + rr - 1.0)

    # d/dx: -2*viol_left + 2*viol_right
    grad[0::3] += penalty_weight * (-2 * viol_left + 2 * viol_right)
    # d/dy: -2*viol_bottom + 2*viol_top
    grad[1::3] += penalty_weight * (-2 * viol_bottom + 2 * viol_top)
    # d/dr: 2*viol_left + 2*viol_right + 2*viol_bottom + 2*viol_top
    grad[2::3] += penalty_weight * (2 * viol_left + 2 * viol_right + 2 * viol_bottom + 2 * viol_top)

    # Non-overlap gradient
    dx = xx[:, None] - xx[None, :]
    dy = yy[:, None] - yy[None, :]
    dist = np.sqrt(dx**2 + dy**2 + 1e-30)
    min_dist = rr[:, None] + rr[None, :]
    overlap = np.maximum(0, min_dist - dist)

    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    active = (overlap > 0) & mask

    if np.any(active):
        # d(overlap)/d(x_i) = -(x_i - x_j)/dist for pair (i,j)
        # d(penalty)/d(x_i) = 2 * overlap * d(overlap)/d(x_i)
        factor = np.zeros((n, n))
        factor[active] = 2.0 * overlap[active] / dist[active]

        for i in range(n):
            # Pairs where i < j
            f_ij = factor[i, :]  # contributions from i,j pairs
            grad[3*i] += penalty_weight * np.sum(-f_ij * dx[i, :])
            grad[3*i+1] += penalty_weight * np.sum(-f_ij * dy[i, :])
            grad[3*i+2] += penalty_weight * np.sum(factor[i, :])  # d(overlap)/d(r_i) = 1

            # Pairs where j < i (from upper triangular)
            f_ji = factor[:, i]
            grad[3*i] += penalty_weight * np.sum(f_ji * dx[:, i])
            grad[3*i+1] += penalty_weight * np.sum(f_ji * dy[:, i])
            grad[3*i+2] += penalty_weight * np.sum(factor[:, i])

    return grad


# ============================================================
# Initialization strategies
# ============================================================

def hex_grid_init(n, offset_x=0.0, offset_y=0.0, scale=1.0):
    cols = int(math.ceil(math.sqrt(n * 2 / math.sqrt(3))))
    rows = int(math.ceil(n / cols))
    r_est = min(0.5 / cols, 0.5 / (rows * math.sqrt(3)/2)) * 0.9

    positions = []
    for row in range(rows + 1):
        for col in range(cols + 1):
            if len(positions) >= n:
                break
            x = (col + 0.5) / (cols + 1)
            y = (row + 0.5) / (rows + 1)
            if row % 2 == 1:
                x += 0.5 / (cols + 1)
            x += offset_x * r_est
            y += offset_y * r_est
            x = np.clip(x, r_est + 1e-4, 1 - r_est - 1e-4)
            y = np.clip(y, r_est + 1e-4, 1 - r_est - 1e-4)
            positions.append((x, y))

    positions = positions[:n]
    radii = [r_est * scale] * n
    return positions, radii


def concentric_rings_init(n, seed=0):
    rng = np.random.RandomState(seed)
    positions = [(0.5, 0.5)]
    ring_configs = [(6, 0.16), (12, 0.32), (18, 0.45)]

    for ring_n, ring_rad in ring_configs:
        offset = rng.uniform(0, 2 * math.pi)
        for k in range(ring_n):
            if len(positions) >= n:
                break
            angle = 2 * math.pi * k / ring_n + offset
            x = 0.5 + ring_rad * math.cos(angle)
            y = 0.5 + ring_rad * math.sin(angle)
            x = np.clip(x, 0.04, 0.96)
            y = np.clip(y, 0.04, 0.96)
            positions.append((x, y))

    while len(positions) < n:
        x = rng.uniform(0.05, 0.95)
        y = rng.uniform(0.05, 0.95)
        positions.append((x, y))

    positions = positions[:n]
    r_est = 0.35 / math.sqrt(n)
    return positions, [r_est] * n


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
        ok = True
        for px, py in positions:
            if (x - px)**2 + (y - py)**2 < min_dist**2:
                ok = False
                break
        if ok:
            positions.append((x, y))

    while len(positions) < n:
        x = rng.uniform(margin, 1 - margin)
        y = rng.uniform(margin, 1 - margin)
        positions.append((x, y))

    return positions[:n], [r_est] * n


def sunflower_init(n):
    golden_angle = math.pi * (3 - math.sqrt(5))
    positions = []
    for i in range(n):
        r = 0.45 * math.sqrt((i + 0.5) / n)
        theta = i * golden_angle
        x = np.clip(0.5 + r * math.cos(theta), 0.04, 0.96)
        y = np.clip(0.5 + r * math.sin(theta), 0.04, 0.96)
        positions.append((x, y))
    r_est = 0.35 / math.sqrt(n)
    return positions, [r_est] * n


def random_init(n, seed=0):
    rng = np.random.RandomState(seed)
    r_est = 0.3 / math.sqrt(n)
    margin = max(r_est, 0.02)
    positions = [(rng.uniform(margin, 1-margin), rng.uniform(margin, 1-margin)) for _ in range(n)]
    return positions, [r_est] * n


def perturbed_hex_init(n, seed=0, perturbation=0.4):
    positions, radii = hex_grid_init(n)
    rng = np.random.RandomState(seed)
    r_est = radii[0]
    new_pos = []
    for x, y in positions:
        dx = rng.normal(0, perturbation * r_est)
        dy = rng.normal(0, perturbation * r_est)
        new_pos.append((np.clip(x+dx, 0.02, 0.98), np.clip(y+dy, 0.02, 0.98)))
    return new_pos, radii


def generate_all_inits(n, num_random=15, num_perturbed=15):
    inits = []
    # Structured
    inits.append(("hex", hex_grid_init(n)))
    for ox, oy in [(0.3, 0), (0, 0.3), (0.3, 0.3), (-0.3, 0.2)]:
        inits.append((f"hex_{ox}_{oy}", hex_grid_init(n, ox, oy)))
    for s in range(4):
        inits.append((f"rings_s{s}", concentric_rings_init(n, seed=s)))
    inits.append(("sunflower", sunflower_init(n)))
    for s in range(6):
        inits.append((f"poisson_s{s}", poisson_disk_init(n, seed=s)))
    for mdf in [0.6, 0.7, 0.9, 1.0, 1.1]:
        inits.append((f"poisson_md{mdf}", poisson_disk_init(n, seed=100, min_dist_factor=mdf)))
    # Random
    for s in range(num_random):
        inits.append((f"rand_s{s}", random_init(n, seed=s)))
    # Perturbed hex
    for s in range(num_perturbed):
        inits.append((f"phex_s{s}", perturbed_hex_init(n, seed=s)))
    return inits


# ============================================================
# Optimization pipeline
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
    positions = [(x[3*i], x[3*i+1]) for i in range(n)]
    radii = [x[3*i+2] for i in range(n)]
    return positions, radii


def optimize_single(positions, radii, n, verbose=False):
    """Optimize a single initialization through all stages."""
    x0 = pack_to_x(positions, radii)

    bounds = []
    for i in range(n):
        bounds.append((1e-4, 1 - 1e-4))  # x
        bounds.append((1e-4, 1 - 1e-4))  # y
        bounds.append((1e-6, 0.5))        # r

    # Stage 1-3: Progressive penalty with L-BFGS-B
    x = x0.copy()
    for pw in [1e2, 1e3, 1e4, 1e5, 1e6, 1e7]:
        result = minimize(
            compute_objective_and_penalty, x, args=(n, pw),
            jac=lambda x, n=n, pw=pw: compute_gradient(x, n, pw),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 800, 'ftol': 1e-14, 'maxfun': 20000}
        )
        x = result.x

    return x


def get_slsqp_constraints(n):
    """Generate SLSQP constraints."""
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
                    (x[3*i] - x[3*j])**2 + (x[3*i+1] - x[3*j+1])**2
                ) - x[3*i+2] - x[3*j+2]
            })
    return constraints


def slsqp_polish(x, n, maxiter=3000):
    """Polish with SLSQP (exact constraint handling)."""
    bounds = []
    for i in range(n):
        bounds.append((1e-6, 1 - 1e-6))
        bounds.append((1e-6, 1 - 1e-6))
        bounds.append((1e-6, 0.5))

    constraints = get_slsqp_constraints(n)

    result = minimize(
        lambda x: -np.sum(x[2::3]), x,
        method='SLSQP', bounds=bounds, constraints=constraints,
        options={'maxiter': maxiter, 'ftol': 1e-15, 'disp': False}
    )
    return result.x


def validate_and_repair(x, n, tol=1e-10):
    """Validate and repair solution."""
    positions, radii = x_to_circles(x, n)
    radii = list(radii)

    for iteration in range(200):
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
                if ri + rj > dist - tol:
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
        if r <= 0 or r - x > tol or x + r - 1 > tol or r - y > tol or y + r - 1 > tol:
            return False
    for i in range(n):
        xi, yi = positions[i]
        ri = radii[i]
        for j in range(i+1, n):
            xj, yj = positions[j]
            rj = radii[j]
            dist = math.sqrt((xi-xj)**2 + (yi-yj)**2)
            if ri + rj - dist > tol:
                return False
    return True


def optimize_packing(n, num_inits_override=None, verbose=True):
    """Main optimization pipeline."""
    if verbose:
        print(f"Optimizing circle packing for n={n}")

    num_random = 15
    num_perturbed = 15
    all_inits = generate_all_inits(n, num_random=num_random, num_perturbed=num_perturbed)

    if num_inits_override:
        all_inits = all_inits[:num_inits_override]

    if verbose:
        print(f"Using {len(all_inits)} initializations")

    best_metric = 0.0
    best_x = None
    top_solutions = []  # (metric, x) for SLSQP polishing
    t_start = time.time()

    for idx, (name, (positions, radii)) in enumerate(all_inits):
        t0 = time.time()
        try:
            x = optimize_single(positions, radii, n)
            pos, rad = validate_and_repair(x, n)
            metric = sum(rad)
            valid = check_valid(pos, rad)
            dt = time.time() - t0

            if valid:
                top_solutions.append((metric, pack_to_x(pos, rad)))
                if metric > best_metric:
                    best_metric = metric
                    best_x = pack_to_x(pos, rad)
                    if verbose:
                        print(f"  [{idx+1}/{len(all_inits)}] {name}: {metric:.6f} ** NEW BEST ** [{dt:.1f}s]")
                elif verbose and (idx % 10 == 0 or idx < 5):
                    print(f"  [{idx+1}/{len(all_inits)}] {name}: {metric:.6f} [{dt:.1f}s]")
            elif verbose and idx % 10 == 0:
                print(f"  [{idx+1}/{len(all_inits)}] {name}: {metric:.6f} INVALID [{dt:.1f}s]")

        except Exception as e:
            if verbose:
                print(f"  [{idx+1}/{len(all_inits)}] {name}: FAILED ({e})")

    # SLSQP polish on top 5 solutions
    if verbose:
        print(f"\nPolishing top solutions with SLSQP...")

    top_solutions.sort(key=lambda t: -t[0])
    for rank, (metric_before, x_sol) in enumerate(top_solutions[:5]):
        t0 = time.time()
        try:
            x_polished = slsqp_polish(x_sol, n, maxiter=5000)
            pos, rad = validate_and_repair(x_polished, n)
            metric = sum(rad)
            valid = check_valid(pos, rad)
            dt = time.time() - t0

            if valid and metric > best_metric:
                best_metric = metric
                best_x = pack_to_x(pos, rad)
                if verbose:
                    print(f"  Polish #{rank+1}: {metric_before:.6f} -> {metric:.6f} ** IMPROVED ** [{dt:.1f}s]")
            elif verbose:
                print(f"  Polish #{rank+1}: {metric_before:.6f} -> {metric:.6f} {'valid' if valid else 'INVALID'} [{dt:.1f}s]")
        except Exception as e:
            if verbose:
                print(f"  Polish #{rank+1}: FAILED ({e})")

    total_time = time.time() - t_start

    if best_x is not None:
        positions, radii = validate_and_repair(best_x, n)
        best_metric = sum(radii)
    else:
        positions, radii = [], []

    if verbose:
        print(f"\nBest metric: {best_metric:.10f}")
        print(f"Total time: {total_time:.1f}s")

    return positions, radii, best_metric


def save_solution(positions, radii, path):
    circles = [[positions[i][0], positions[i][1], radii[i]] for i in range(len(positions))]
    with open(path, 'w') as f:
        json.dump({"circles": circles}, f, indent=2)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 26
    output = sys.argv[2] if len(sys.argv) > 2 else None

    positions, radii, metric = optimize_packing(n, verbose=True)

    if positions:
        if output is None:
            output = str(Path(__file__).parent / f"solution_n{n}.json")
        save_solution(positions, radii, output)
        print(f"\nSolution saved to {output}")
    else:
        print("No valid solution found!")
        sys.exit(1)


if __name__ == "__main__":
    main()
