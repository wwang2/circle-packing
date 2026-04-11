"""
V4: Topology-aware search for n=26.

Key insight: the gap from 2.627 to 2.636 requires finding the right
topological basin. Known good n=26 packings have:
- 4 large circles (r~0.18-0.21) near corners
- ~8 medium circles (r~0.10-0.14) along edges
- ~14 smaller circles (r~0.05-0.08) filling interior

Strategy:
1. Generate arrangements with specific radius distributions
2. Use constrained optimization (COBYLA/trust-constr for diversity)
3. Much more aggressive basin-hopping with larger perturbations
4. Try "billiard" dynamics: start with small circles, grow them
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
        schedule = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
    bounds = [(1e-4, 1-1e-4), (1e-4, 1-1e-4), (1e-6, 0.5)] * n
    x = x0.copy()
    for pw in schedule:
        result = minimize(
            compute_objective_and_penalty, x, args=(n, pw),
            jac=lambda x, n=n, pw=pw: compute_gradient(x, n, pw),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 600, 'ftol': 1e-15}
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


# ============================================================
# Topology-aware initializations
# ============================================================

def topology_init_A(n=26, seed=0):
    """4 large corners + 8 edge + 14 interior."""
    rng = np.random.RandomState(seed)
    positions = []
    radii = []

    # 4 corner circles (large)
    r_corner = 0.18 + rng.uniform(-0.02, 0.02)
    for cx, cy in [(r_corner, r_corner), (1-r_corner, r_corner),
                   (r_corner, 1-r_corner), (1-r_corner, 1-r_corner)]:
        positions.append((cx + rng.normal(0, 0.01), cy + rng.normal(0, 0.01)))
        radii.append(r_corner + rng.uniform(-0.01, 0.01))

    # 8 edge circles (medium)
    r_edge = 0.10 + rng.uniform(-0.02, 0.02)
    edge_positions = [
        (0.5, r_edge), (0.5, 1-r_edge),  # top/bottom center
        (r_edge, 0.5), (1-r_edge, 0.5),  # left/right center
        (0.33, r_edge), (0.67, r_edge),   # bottom
        (0.33, 1-r_edge), (0.67, 1-r_edge),  # top
    ]
    for ex, ey in edge_positions:
        positions.append((ex + rng.normal(0, 0.02), ey + rng.normal(0, 0.02)))
        radii.append(r_edge + rng.uniform(-0.01, 0.01))

    # 14 interior circles (small)
    r_int = 0.065 + rng.uniform(-0.01, 0.01)
    for _ in range(14):
        x = rng.uniform(0.15, 0.85)
        y = rng.uniform(0.15, 0.85)
        positions.append((x, y))
        radii.append(r_int + rng.uniform(-0.01, 0.01))

    positions = [(np.clip(x, 0.02, 0.98), np.clip(y, 0.02, 0.98)) for x, y in positions]
    return positions[:n], radii[:n]


def topology_init_B(n=26, seed=0):
    """3 large + 5 medium + 18 small - asymmetric arrangement."""
    rng = np.random.RandomState(seed)
    positions = []
    radii = []

    # 3 large circles
    r_large = 0.20 + rng.uniform(-0.02, 0.02)
    for _ in range(3):
        x = rng.uniform(r_large, 1-r_large)
        y = rng.uniform(r_large, 1-r_large)
        positions.append((x, y))
        radii.append(r_large + rng.uniform(-0.02, 0.02))

    # 5 medium
    r_med = 0.12 + rng.uniform(-0.02, 0.02)
    for _ in range(5):
        x = rng.uniform(r_med, 1-r_med)
        y = rng.uniform(r_med, 1-r_med)
        positions.append((x, y))
        radii.append(r_med + rng.uniform(-0.01, 0.01))

    # 18 small
    r_small = 0.06 + rng.uniform(-0.01, 0.01)
    for _ in range(18):
        x = rng.uniform(0.05, 0.95)
        y = rng.uniform(0.05, 0.95)
        positions.append((x, y))
        radii.append(r_small + rng.uniform(-0.01, 0.01))

    return positions[:n], radii[:n]


def topology_init_C(n=26, seed=0):
    """Billiard-inspired: start with equal circles, let optimizer differentiate."""
    rng = np.random.RandomState(seed)
    # Place circles in a semi-regular pattern
    r_eq = 1.0 / (2 * math.sqrt(n) + 1)  # ~0.098 for n=26

    positions = []
    # Use Poisson disk with the equal radius
    min_dist = 2.2 * r_eq
    margin = r_eq + 0.005
    for _ in range(100000):
        if len(positions) >= n:
            break
        x = rng.uniform(margin, 1 - margin)
        y = rng.uniform(margin, 1 - margin)
        ok = all((x-px)**2 + (y-py)**2 >= min_dist**2 for px, py in positions)
        if ok:
            positions.append((x, y))
    while len(positions) < n:
        positions.append((rng.uniform(margin, 1-margin), rng.uniform(margin, 1-margin)))

    return positions[:n], [r_eq] * n


def topology_init_D(n=26, seed=0):
    """Row-based packing: rows of different sizes."""
    rng = np.random.RandomState(seed)
    # Try different row configurations
    configs = [
        [5, 6, 5, 6, 4],    # 26
        [4, 5, 6, 5, 4, 2],  # 26
        [6, 5, 6, 5, 4],    # 26
        [4, 6, 6, 6, 4],    # 26
        [3, 5, 6, 5, 4, 3], # 26
    ]
    config = configs[seed % len(configs)]
    total = sum(config)
    assert total >= n

    positions = []
    nrows = len(config)
    row_h = 1.0 / (nrows + 0.5)

    for row_idx, ncols in enumerate(config):
        y = (row_idx + 0.75) * row_h
        col_w = 1.0 / (ncols + 0.5)
        offset = 0.25 * col_w if row_idx % 2 == 1 else 0
        for col_idx in range(ncols):
            if len(positions) >= n:
                break
            x = (col_idx + 0.75) * col_w + offset
            x += rng.normal(0, 0.01)
            y_pert = y + rng.normal(0, 0.01)
            positions.append((np.clip(x, 0.04, 0.96), np.clip(y_pert, 0.04, 0.96)))

    r_est = 0.07 + rng.uniform(-0.01, 0.01)
    return positions[:n], [r_est] * n


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


def grow_from_small(n=26, seed=0):
    """Start with tiny circles and grow them iteratively."""
    rng = np.random.RandomState(seed)
    # Place n circles randomly with very small radii
    positions = []
    r_tiny = 0.005
    margin = 0.01
    for _ in range(n):
        positions.append((rng.uniform(margin, 1-margin), rng.uniform(margin, 1-margin)))

    x = pack_to_x(positions, [r_tiny]*n)

    # Grow radii in stages
    for target_r in [0.02, 0.04, 0.06, 0.08, 0.10]:
        # Set all radii to target
        for i in range(n):
            x[3*i+2] = target_r
        # Optimize positions
        x = lbfgsb_optimize(x, n, schedule=[1e3, 1e5, 1e7])

    return x


def generate_all_inits(n):
    inits = []
    # Topology-aware
    for s in range(20):
        inits.append((f"topoA_s{s}", topology_init_A(n, seed=s)))
    for s in range(15):
        inits.append((f"topoB_s{s}", topology_init_B(n, seed=s)))
    for s in range(20):
        inits.append((f"topoC_s{s}", topology_init_C(n, seed=s)))
    for s in range(10):
        inits.append((f"topoD_s{s}", topology_init_D(n, seed=s)))
    # Poisson disk (still good)
    for s in range(30):
        inits.append((f"poisson_s{s}", poisson_disk_init(n, seed=s+500)))
    for mdf in [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]:
        for s in range(3):
            inits.append((f"poisson_md{mdf}_s{s}", poisson_disk_init(n, seed=300+s, min_dist_factor=mdf)))
    return inits


def optimize_packing(n=26, verbose=True):
    all_inits = generate_all_inits(n)

    # Also try grow-from-small (returns x directly, not positions/radii)
    grow_xs = []
    if verbose:
        print(f"Phase 0: Growing from small ({5} seeds)", flush=True)
    for s in range(5):
        try:
            t0 = time.time()
            x_grown = grow_from_small(n, seed=s)
            pos, rad = validate_and_repair(x_grown, n)
            metric = sum(rad)
            valid = check_valid(pos, rad)
            dt = time.time() - t0
            if valid and metric > 0.5:
                grow_xs.append((metric, pack_to_x(pos, rad)))
            if verbose:
                print(f"  Grow s={s}: {metric:.6f} {'ok' if valid else 'INV'} [{dt:.1f}s]", flush=True)
        except Exception as e:
            if verbose:
                print(f"  Grow s={s}: ERR {e}", flush=True)

    if verbose:
        print(f"\nPhase 1: L-BFGS-B on {len(all_inits)} inits", flush=True)

    candidates = list(grow_xs)
    best_p1 = max((m for m, _ in candidates), default=0)
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
            if valid and metric > best_p1:
                best_p1 = metric
                if verbose:
                    print(f"  [{idx+1}/{len(all_inits)}] {name}: {metric:.6f} ** BEST ** [{dt:.1f}s]", flush=True)
            elif verbose and idx % 25 == 0:
                print(f"  [{idx+1}/{len(all_inits)}] {name}: {metric:.6f} [{dt:.1f}s]", flush=True)
        except:
            pass

    candidates.sort(key=lambda t: -t[0])
    num_polish = min(20, len(candidates))
    if verbose:
        print(f"\nPhase 2: SLSQP polish on top {num_polish}", flush=True)

    best_metric = 0.0
    best_x = None

    for rank in range(num_polish):
        mb, x_sol = candidates[rank]
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
                    print(f"  #{rank+1}: {mb:.6f} -> {metric:.6f} ** BEST ** [{dt:.1f}s]", flush=True)
            elif verbose and rank < 5:
                print(f"  #{rank+1}: {mb:.6f} -> {metric:.6f} [{dt:.1f}s]", flush=True)
        except:
            pass

    # Also load and try to improve previous best if it exists
    prev_best_path = Path(__file__).parent / "solution_n26.json"
    if prev_best_path.exists():
        try:
            with open(prev_best_path) as f:
                data = json.load(f)
            circles = data["circles"]
            x_prev = np.zeros(3*n)
            for i, (cx, cy, r) in enumerate(circles):
                x_prev[3*i] = cx; x_prev[3*i+1] = cy; x_prev[3*i+2] = r
            prev_metric = np.sum(x_prev[2::3])
            if verbose:
                print(f"\nLoaded previous best: {prev_metric:.10f}", flush=True)
            # Polish it too
            x_pol = slsqp_polish(x_prev, n, maxiter=10000)
            pos, rad = validate_and_repair(x_pol, n)
            metric = sum(rad)
            if check_valid(pos, rad) and metric > best_metric:
                best_metric = metric
                best_x = pack_to_x(pos, rad)
                if verbose:
                    print(f"  Previous best polished: {metric:.10f} ** NEW BEST **", flush=True)
        except:
            pass

    # Phase 3: Aggressive basin-hopping
    if best_x is not None:
        if verbose:
            print(f"\nPhase 3: Basin-hopping from {best_metric:.10f}", flush=True)
        rng = np.random.RandomState(777)
        no_improve = 0
        for attempt in range(60):
            if no_improve >= 20:
                break
            scale = rng.choice([0.002, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05])
            x_pert = best_x.copy()
            # Randomly perturb positions (and sometimes radii too)
            for i in range(n):
                x_pert[3*i] = np.clip(x_pert[3*i] + rng.normal(0, scale), 0.01, 0.99)
                x_pert[3*i+1] = np.clip(x_pert[3*i+1] + rng.normal(0, scale), 0.01, 0.99)
                if rng.random() < 0.3:  # 30% chance to perturb radius
                    x_pert[3*i+2] = np.clip(x_pert[3*i+2] + rng.normal(0, scale*0.5), 0.001, 0.49)
            try:
                x_opt = lbfgsb_optimize(x_pert, n)
                x_pol = slsqp_polish(x_opt, n, maxiter=5000)
                pos, rad = validate_and_repair(x_pol, n)
                metric = sum(rad)
                if check_valid(pos, rad) and metric > best_metric + 1e-10:
                    best_metric = metric
                    best_x = pack_to_x(pos, rad)
                    no_improve = 0
                    if verbose:
                        print(f"  #{attempt+1} sc={scale:.3f}: {metric:.10f} ** IMPROVED **", flush=True)
                else:
                    no_improve += 1
            except:
                no_improve += 1

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
    output = sys.argv[2] if len(sys.argv) > 2 else str(Path(__file__).parent / f"solution_n{n}_v4.json")
    positions, radii, metric = optimize_packing(n, verbose=True)
    if positions:
        save_solution(positions, radii, output)
        print(f"Saved to {output}", flush=True)
    else:
        print("No valid solution!", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
