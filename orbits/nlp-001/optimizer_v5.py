"""
V5: Massive parallel search with 500+ initializations.
Uses multiprocessing to run in parallel.
Focus: find the 2.636 basin through sheer diversity.
"""

import numpy as np
from scipy.optimize import minimize
import json
import sys
import time
import math
from pathlib import Path
from multiprocessing import Pool, cpu_count


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


def generate_init(n, seed):
    """Generate one random initialization with seed-determined strategy."""
    rng = np.random.RandomState(seed)
    strategy = seed % 8

    if strategy == 0:
        # Poisson disk
        min_dist = rng.uniform(0.5, 1.2) / math.sqrt(n)
        r_est = min_dist / 2.5
        margin = max(r_est, 0.02)
        positions = []
        for _ in range(80000):
            if len(positions) >= n: break
            x = rng.uniform(margin, 1-margin)
            y = rng.uniform(margin, 1-margin)
            if all((x-px)**2+(y-py)**2 >= min_dist**2 for px, py in positions):
                positions.append((x, y))
        while len(positions) < n:
            positions.append((rng.uniform(margin, 1-margin), rng.uniform(margin, 1-margin)))
        return positions[:n], [r_est]*n

    elif strategy == 1:
        # Random uniform
        r_est = rng.uniform(0.03, 0.10)
        m = max(r_est, 0.02)
        return [(rng.uniform(m, 1-m), rng.uniform(m, 1-m)) for _ in range(n)], [r_est]*n

    elif strategy == 2:
        # Hex grid with random offset
        cols = rng.randint(4, 8)
        rows = int(math.ceil(n / cols)) + 1
        r_est = min(0.5/cols, 0.5/(rows*0.87)) * 0.85
        positions = []
        ox = rng.uniform(-0.5, 0.5)
        oy = rng.uniform(-0.5, 0.5)
        for row in range(rows):
            for col in range(cols+1):
                if len(positions) >= n: break
                x = (col + 0.5)/(cols+1) + ox*r_est
                y = (row + 0.5)/(rows+1) + oy*r_est
                if row % 2 == 1: x += 0.5/(cols+1)
                positions.append((np.clip(x, 0.02, 0.98), np.clip(y, 0.02, 0.98)))
        return positions[:n], [r_est]*n

    elif strategy == 3:
        # Concentric rings
        positions = [(0.5, 0.5)]
        for ring_n, ring_r in [(rng.randint(5,8), rng.uniform(0.12,0.20)),
                               (rng.randint(8,14), rng.uniform(0.25,0.38)),
                               (rng.randint(10,20), rng.uniform(0.40,0.48))]:
            off = rng.uniform(0, 2*math.pi)
            for k in range(ring_n):
                if len(positions) >= n: break
                a = 2*math.pi*k/ring_n + off
                positions.append((np.clip(0.5+ring_r*math.cos(a), 0.04, 0.96),
                                 np.clip(0.5+ring_r*math.sin(a), 0.04, 0.96)))
        while len(positions) < n:
            positions.append((rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95)))
        return positions[:n], [0.35/math.sqrt(n)]*n

    elif strategy == 4:
        # Corner-biased topology
        r_large = rng.uniform(0.15, 0.22)
        r_small = rng.uniform(0.04, 0.08)
        positions = []; radii = []
        # 4 corners
        for cx, cy in [(r_large, r_large), (1-r_large, r_large),
                       (r_large, 1-r_large), (1-r_large, 1-r_large)]:
            positions.append((cx+rng.normal(0, 0.01), cy+rng.normal(0, 0.01)))
            radii.append(r_large+rng.normal(0, 0.01))
        for _ in range(n-4):
            positions.append((rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95)))
            radii.append(r_small+rng.uniform(-0.01, 0.01))
        return [(np.clip(x, 0.02, 0.98), np.clip(y, 0.02, 0.98)) for x, y in positions[:n]], radii[:n]

    elif strategy == 5:
        # Row-based
        nrows = rng.randint(4, 7)
        row_counts = []
        remaining = n
        for r in range(nrows):
            if r == nrows - 1:
                row_counts.append(remaining)
            else:
                c = rng.randint(max(1, remaining//(nrows-r)-2), min(remaining, remaining//(nrows-r)+3))
                row_counts.append(c)
                remaining -= c
        positions = []
        for row_idx, nc in enumerate(row_counts):
            y = (row_idx + 0.5) / nrows
            for col_idx in range(nc):
                x = (col_idx + 0.5) / nc
                if row_idx % 2 == 1: x += 0.5/nc * rng.uniform(-0.3, 0.3)
                positions.append((np.clip(x+rng.normal(0,0.01), 0.03, 0.97),
                                 np.clip(y+rng.normal(0,0.01), 0.03, 0.97)))
        r_est = 0.07+rng.uniform(-0.02, 0.02)
        return positions[:n], [r_est]*n

    elif strategy == 6:
        # Sunflower spiral
        golden = math.pi * (3 - math.sqrt(5))
        positions = []
        scale = rng.uniform(0.40, 0.48)
        for i in range(n):
            r = scale * math.sqrt((i + 0.5) / n)
            theta = i * golden + rng.uniform(0, 0.5)
            positions.append((np.clip(0.5+r*math.cos(theta), 0.04, 0.96),
                             np.clip(0.5+r*math.sin(theta), 0.04, 0.96)))
        return positions, [0.35/math.sqrt(n)]*n

    else:  # strategy == 7
        # Mixed-size: some big, some small
        n_big = rng.randint(2, 6)
        r_big = rng.uniform(0.15, 0.25)
        r_small = rng.uniform(0.04, 0.08)
        positions = []; radii = []
        for _ in range(n_big):
            positions.append((rng.uniform(r_big, 1-r_big), rng.uniform(r_big, 1-r_big)))
            radii.append(r_big + rng.normal(0, 0.02))
        for _ in range(n - n_big):
            positions.append((rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95)))
            radii.append(r_small + rng.normal(0, 0.01))
        return positions[:n], [max(r, 0.01) for r in radii[:n]]


def optimize_one(args):
    """Optimize one initialization. Returns (metric, x_flat) or (0, None)."""
    seed, n = args
    try:
        positions, radii = generate_init(n, seed)
        x0 = pack_to_x(positions, radii)

        bounds = [(1e-4, 1-1e-4), (1e-4, 1-1e-4), (1e-6, 0.5)] * n
        x = x0.copy()
        for pw in [1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]:
            result = minimize(
                compute_objective_and_penalty, x, args=(n, pw),
                jac=lambda x, n=n, pw=pw: compute_gradient(x, n, pw),
                method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-15}
            )
            x = result.x

        pos, rad = validate_and_repair(x, n)
        metric = sum(rad)
        if check_valid(pos, rad):
            return (metric, pack_to_x(pos, rad))
    except:
        pass
    return (0.0, None)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 26
    output = sys.argv[2] if len(sys.argv) > 2 else str(Path(__file__).parent / f"solution_n{n}_v5.json")
    num_inits = int(sys.argv[3]) if len(sys.argv) > 3 else 400

    print(f"V5: Parallel search for n={n} with {num_inits} inits", flush=True)
    t_start = time.time()

    # Phase 1: Parallel L-BFGS-B
    ncpu = min(cpu_count(), 8)
    print(f"Using {ncpu} workers", flush=True)

    args = [(seed, n) for seed in range(num_inits)]
    with Pool(ncpu) as pool:
        results = pool.map(optimize_one, args)

    candidates = [(m, x) for m, x in results if m > 0.5 and x is not None]
    candidates.sort(key=lambda t: -t[0])

    print(f"Phase 1 done: {len(candidates)} valid, top={candidates[0][0]:.6f} [{time.time()-t_start:.1f}s]", flush=True)

    # Phase 2: SLSQP polish top candidates
    num_polish = min(25, len(candidates))
    print(f"Phase 2: SLSQP polish top {num_polish}", flush=True)

    best_metric = 0.0
    best_x = None
    constraints = get_slsqp_constraints(n)
    bounds_slsqp = [(1e-6, 1-1e-6), (1e-6, 1-1e-6), (1e-6, 0.5)] * n

    for rank in range(num_polish):
        mb, x_sol = candidates[rank]
        t0 = time.time()
        try:
            result = minimize(
                lambda x: -np.sum(x[2::3]), x_sol,
                method='SLSQP', bounds=bounds_slsqp, constraints=constraints,
                options={'maxiter': 8000, 'ftol': 1e-15}
            )
            pos, rad = validate_and_repair(result.x, n)
            metric = sum(rad)
            dt = time.time() - t0
            if check_valid(pos, rad) and metric > best_metric:
                best_metric = metric
                best_x = pack_to_x(pos, rad)
                print(f"  #{rank+1}: {mb:.6f} -> {metric:.6f} ** BEST ** [{dt:.1f}s]", flush=True)
            elif rank < 5:
                print(f"  #{rank+1}: {mb:.6f} -> {metric:.6f} [{dt:.1f}s]", flush=True)
        except:
            pass

    # Load previous best
    prev_path = Path(__file__).parent / "solution_n26.json"
    if prev_path.exists() and n == 26:
        try:
            with open(prev_path) as f:
                data = json.load(f)
            x_prev = np.zeros(3*n)
            for i, (cx, cy, r) in enumerate(data["circles"]):
                x_prev[3*i] = cx; x_prev[3*i+1] = cy; x_prev[3*i+2] = r
            prev_m = np.sum(x_prev[2::3])
            if prev_m > best_metric:
                best_metric = prev_m
                best_x = x_prev
                print(f"Previous best {prev_m:.10f} is still better", flush=True)
        except:
            pass

    # Phase 3: basin-hopping
    if best_x is not None:
        print(f"\nPhase 3: Basin-hopping from {best_metric:.10f}", flush=True)
        rng = np.random.RandomState(42)
        no_imp = 0
        for att in range(80):
            if no_imp >= 25:
                break
            scale = rng.choice([0.005, 0.01, 0.02, 0.03, 0.05, 0.07])
            x_pert = best_x.copy()
            n_pert = rng.randint(1, n+1)
            idx = rng.choice(n, n_pert, replace=False)
            for i in idx:
                x_pert[3*i] = np.clip(x_pert[3*i]+rng.normal(0,scale), 0.01, 0.99)
                x_pert[3*i+1] = np.clip(x_pert[3*i+1]+rng.normal(0,scale), 0.01, 0.99)
                if rng.random() < 0.3:
                    x_pert[3*i+2] = np.clip(x_pert[3*i+2]*rng.uniform(0.5,1.5), 0.001, 0.49)
            try:
                bounds = [(1e-4, 1-1e-4), (1e-4, 1-1e-4), (1e-6, 0.5)] * n
                x = x_pert
                for pw in [1e3, 1e5, 1e7, 1e8]:
                    result = minimize(
                        compute_objective_and_penalty, x, args=(n, pw),
                        jac=lambda x, n=n, pw=pw: compute_gradient(x, n, pw),
                        method='L-BFGS-B', bounds=bounds,
                        options={'maxiter': 300, 'ftol': 1e-14}
                    )
                    x = result.x
                result = minimize(
                    lambda x: -np.sum(x[2::3]), x,
                    method='SLSQP', bounds=bounds_slsqp, constraints=constraints,
                    options={'maxiter': 5000, 'ftol': 1e-15}
                )
                pos, rad = validate_and_repair(result.x, n)
                metric = sum(rad)
                if check_valid(pos, rad) and metric > best_metric + 1e-10:
                    best_metric = metric
                    best_x = pack_to_x(pos, rad)
                    no_imp = 0
                    print(f"  #{att+1} sc={scale:.3f}: {metric:.10f} ** IMPROVED **", flush=True)
                else:
                    no_imp += 1
            except:
                no_imp += 1

    total = time.time() - t_start
    print(f"\nFinal: {best_metric:.10f} in {total:.1f}s", flush=True)

    if best_x is not None:
        pos, rad = validate_and_repair(best_x, n)
        circles = [[pos[i][0], pos[i][1], rad[i]] for i in range(n)]
        with open(output, 'w') as f:
            json.dump({"circles": circles}, f, indent=2)
        print(f"Saved to {output}", flush=True)


if __name__ == "__main__":
    main()
