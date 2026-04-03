"""
Circle packing solver v2: maximize sum of radii for n circles in [0,1]^2.

Strategy: Fast penalty-based L-BFGS-B on joint (x,y,r) variables,
followed by greedy refinement. No nested optimization.
"""

import numpy as np
from scipy.optimize import minimize
import json
import os
import time
import sys


def joint_penalty(vec, n, mu):
    """Penalized objective: -sum(r) + mu * violations^2."""
    xs = vec[:n]
    ys = vec[n:2*n]
    rs = vec[2*n:]

    obj = -np.sum(rs)

    # Containment penalties
    p = np.sum(np.maximum(0, rs - xs)**2)
    p += np.sum(np.maximum(0, xs + rs - 1)**2)
    p += np.sum(np.maximum(0, rs - ys)**2)
    p += np.sum(np.maximum(0, ys + rs - 1)**2)

    # Non-overlap: vectorized upper triangle
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist_sq = dx**2 + dy**2
    r_sum = rs[:, None] + rs[None, :]
    # Overlap where r_sum > dist
    # Use squared comparison to avoid sqrt when possible
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    # Where r_sum^2 > dist_sq, there might be overlap
    potential = mask & (r_sum**2 > dist_sq)
    if np.any(potential):
        dist_vals = np.sqrt(dist_sq[potential])
        r_sum_vals = r_sum[potential]
        overlaps = np.maximum(0, r_sum_vals - dist_vals)
        p += np.sum(overlaps**2)

    # Negative radius penalty
    p += 100 * np.sum(np.maximum(0, -rs)**2)

    return obj + mu * p


def joint_penalty_grad(vec, n, mu):
    """Gradient of penalized objective."""
    xs = vec[:n]
    ys = vec[n:2*n]
    rs = vec[2*n:]

    grad_x = np.zeros(n)
    grad_y = np.zeros(n)
    grad_r = -np.ones(n)  # d(-sum(r))/dr_i = -1

    # Containment gradients
    # rs - xs > 0 => penalty += (rs-xs)^2, d/dx = -2*(rs-xs), d/dr = 2*(rs-xs)
    v = np.maximum(0, rs - xs)
    grad_x -= 2 * mu * v
    grad_r += 2 * mu * v

    v = np.maximum(0, xs + rs - 1)
    grad_x += 2 * mu * v
    grad_r += 2 * mu * v

    v = np.maximum(0, rs - ys)
    grad_y -= 2 * mu * v
    grad_r += 2 * mu * v

    v = np.maximum(0, ys + rs - 1)
    grad_y += 2 * mu * v
    grad_r += 2 * mu * v

    # Non-overlap gradients
    dx = xs[:, None] - xs[None, :]  # dx[i,j] = xi - xj
    dy = ys[:, None] - ys[None, :]
    dist_sq = dx**2 + dy**2
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    r_sum = rs[:, None] + rs[None, :]

    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    overlap = np.maximum(0, r_sum - dist) * mask

    # For overlap[i,j] > 0:
    # d(overlap^2)/dx_i = 2*overlap * dx[i,j]/dist[i,j]
    # d(overlap^2)/dx_j = -2*overlap * dx[i,j]/dist[i,j]
    # d(overlap^2)/dr_i = -2*overlap
    # d(overlap^2)/dr_j = -2*overlap

    inv_dist = np.where(dist > 1e-15, 1.0/dist, 0.0)

    # Gradient contributions from overlap
    # For each pair (i,j) with i<j and overlap>0:
    olap_factor = 2 * mu * overlap * inv_dist  # shape (n,n)

    # dx_i += olap_factor[i,j] * dx[i,j] for j>i
    # dx_j -= olap_factor[i,j] * dx[i,j] for j>i (but j is column)
    grad_x += np.sum(olap_factor * dx, axis=1)   # contributions where i is row
    grad_x -= np.sum(olap_factor * dx, axis=0)   # contributions where i is column

    grad_y += np.sum(olap_factor * dy, axis=1)
    grad_y -= np.sum(olap_factor * dy, axis=0)

    # dr_i: -2*mu*overlap for each pair involving i
    olap_r = 2 * mu * overlap
    grad_r -= np.sum(olap_r, axis=1)  # pairs (i, j>i)
    grad_r -= np.sum(olap_r, axis=0)  # pairs (j<i, i)

    # Negative radius
    neg_v = np.maximum(0, -rs)
    grad_r -= 200 * mu * neg_v

    grad = np.concatenate([grad_x, grad_y, grad_r])
    return grad


def optimize_penalty(xs, ys, rs, n, max_iter=500):
    """Progressive penalty optimization with analytical gradients."""
    vec = np.concatenate([xs, ys, rs])
    bounds = ([(1e-4, 1-1e-4)] * n +
              [(1e-4, 1-1e-4)] * n +
              [(1e-6, 0.5)] * n)

    for mu in [10, 100, 1000, 10000, 100000]:
        result = minimize(
            lambda v: joint_penalty(v, n, mu),
            vec,
            jac=lambda v: joint_penalty_grad(v, n, mu),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12}
        )
        vec = result.x

    return vec[:n], vec[n:2*n], vec[2*n:]


def repair_and_grow(xs, ys, rs, n):
    """Repair constraint violations, then grow radii to maximum."""
    # Containment
    for i in range(n):
        wall_max = min(xs[i], 1-xs[i], ys[i], 1-ys[i])
        rs[i] = min(rs[i], wall_max)

    # Fix overlaps
    for _ in range(200):
        fixed = True
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                overlap = rs[i] + rs[j] - dist
                if overlap > 1e-13:
                    total = rs[i] + rs[j]
                    if total > 0:
                        shrink = overlap + 1e-14
                        rs[i] -= shrink * rs[i] / total
                        rs[j] -= shrink * rs[j] / total
                    fixed = False
        if fixed:
            break

    rs = np.maximum(rs, 1e-15)

    # Grow radii
    for _ in range(10):
        changed = False
        for i in range(n):
            max_r = min(xs[i], 1-xs[i], ys[i], 1-ys[i])
            for j in range(n):
                if j == i:
                    continue
                dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                max_r = min(max_r, dist - rs[j])
            if max_r > rs[i] + 1e-15:
                rs[i] = max_r
                changed = True
        if not changed:
            break

    return rs


def local_search_move(xs, ys, rs, n, step=0.01, grid_pts=9, iters=3):
    """Greedy local search: move each circle to maximize total radii."""
    for it in range(iters):
        improved = False
        order = np.random.permutation(n)
        offsets = np.linspace(-step, step, grid_pts)

        for i in order:
            best_x, best_y = xs[i], ys[i]
            best_gain = 0

            old_r = rs[i]
            for dx in offsets:
                for dy in offsets:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = xs[i] + dx, ys[i] + dy
                    if nx < 0.002 or nx > 0.998 or ny < 0.002 or ny > 0.998:
                        continue

                    max_r = min(nx, 1-nx, ny, 1-ny)
                    for j in range(n):
                        if j == i:
                            continue
                        dist = np.sqrt((nx - xs[j])**2 + (ny - ys[j])**2)
                        max_r = min(max_r, dist - rs[j])

                    if max_r <= 0:
                        continue

                    gain = max_r - old_r
                    if gain > best_gain + 1e-15:
                        best_x, best_y = nx, ny
                        best_gain = gain
                        best_r = max_r

            if best_gain > 1e-15:
                xs[i], ys[i] = best_x, best_y
                rs[i] = best_r
                improved = True

        if not improved:
            break
        step *= 0.6

    return xs, ys, rs


def hex_init(n, noise=0.0):
    """Hexagonal initial positions."""
    side = int(np.ceil(np.sqrt(n * 2 / np.sqrt(3)))) + 1
    pts = []
    for row in range(side + 3):
        for col in range(side + 3):
            x = (col + 0.5 * (row % 2) + 0.5) / (side + 2)
            y = (row * np.sqrt(3) / 2 + 0.5) / (side + 2)
            if 0.02 < x < 0.98 and 0.02 < y < 0.98:
                pts.append((x, y))
    pts = np.array(pts)

    if len(pts) >= n:
        selected = [len(pts) // 2]
        for _ in range(n - 1):
            min_d = np.min([np.sum((pts - pts[s])**2, axis=1) for s in selected], axis=0)
            min_d[selected] = -1
            selected.append(np.argmax(min_d))
        pts = pts[selected]
    else:
        extra = np.random.uniform(0.05, 0.95, (n - len(pts), 2))
        pts = np.vstack([pts, extra])

    xs, ys = pts[:n, 0].copy(), pts[:n, 1].copy()
    if noise > 0:
        xs += np.random.randn(n) * noise
        ys += np.random.randn(n) * noise
        xs = np.clip(xs, 0.02, 0.98)
        ys = np.clip(ys, 0.02, 0.98)
    return xs, ys


def grid_init(n, noise=0.0):
    side = int(np.ceil(np.sqrt(n)))
    pts = []
    for i in range(side):
        for j in range(side):
            pts.append(((i+0.5)/side, (j+0.5)/side))
    pts = np.array(pts[:n])
    xs, ys = pts[:, 0].copy(), pts[:, 1].copy()
    if noise > 0:
        xs += np.random.randn(n) * noise
        ys += np.random.randn(n) * noise
        xs = np.clip(xs, 0.02, 0.98)
        ys = np.clip(ys, 0.02, 0.98)
    return xs, ys


def random_init(n):
    margin = 0.4 / np.sqrt(n)
    xs = np.random.uniform(margin, 1-margin, n)
    ys = np.random.uniform(margin, 1-margin, n)
    return xs, ys


def validate(xs, ys, rs, n, tol=1e-10):
    max_viol = 0.0
    for i in range(n):
        max_viol = max(max_viol, rs[i] - xs[i])
        max_viol = max(max_viol, xs[i] + rs[i] - 1.0)
        max_viol = max(max_viol, rs[i] - ys[i])
        max_viol = max(max_viol, ys[i] + rs[i] - 1.0)
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
            max_viol = max(max_viol, rs[i]+rs[j] - dist)
    return max_viol <= tol, np.sum(rs), max_viol


def single_start(n, init_type='hex', noise=0.0):
    """One complete optimization run."""
    if init_type == 'hex':
        xs, ys = hex_init(n, noise)
    elif init_type == 'grid':
        xs, ys = grid_init(n, noise)
    else:
        xs, ys = random_init(n)

    r_est = 0.4 / np.sqrt(n)
    rs = np.full(n, r_est)

    # Penalty optimization
    xs, ys, rs = optimize_penalty(xs, ys, rs, n, max_iter=400)

    # Repair and grow
    rs = repair_and_grow(xs, ys, rs, n)

    # Local search
    xs, ys, rs = local_search_move(xs, ys, rs, n, step=0.02, grid_pts=7, iters=3)
    rs = repair_and_grow(xs, ys, rs, n)

    valid, metric, viol = validate(xs, ys, rs, n)
    if not valid:
        # Aggressive repair
        for i in range(n):
            rs[i] *= 0.999
        rs = repair_and_grow(xs, ys, rs, n)
        valid, metric, viol = validate(xs, ys, rs, n)

    if valid:
        return xs, ys, rs, metric
    return None


def solve_n(n, num_starts=60, bh_hops=150, verbose=True):
    """Full solver."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Solving n={n}")
        print(f"{'='*60}")

    best = None
    best_metric = -1
    t0 = time.time()

    inits = ['hex', 'grid', 'random']

    if verbose:
        print(f"\nPhase 1: Multi-start ({num_starts} starts)")

    for s in range(num_starts):
        init = inits[s % len(inits)]
        noise = 0.003 * (s // len(inits))

        try:
            result = single_start(n, init, noise)
            if result is not None:
                xs, ys, rs, metric = result
                if metric > best_metric:
                    best_metric = metric
                    best = (xs.copy(), ys.copy(), rs.copy())
                    if verbose:
                        print(f"  Start {s:3d} ({init:6s}, noise={noise:.3f}): {metric:.10f} BEST")
        except Exception as e:
            if verbose and s < 3:
                print(f"  Start {s}: FAILED {e}")

    if verbose:
        print(f"Phase 1 best: {best_metric:.10f} ({time.time()-t0:.1f}s)")

    if best is None:
        return None

    # Phase 2: Basin hopping
    if verbose:
        print(f"\nPhase 2: Basin hopping ({bh_hops} hops)")

    bxs, bys, brs = best
    cur_xs, cur_ys, cur_rs = bxs.copy(), bys.copy(), brs.copy()
    cur_metric = best_metric

    for hop in range(bh_hops):
        txs, tys = cur_xs.copy(), cur_ys.copy()

        strat = np.random.randint(5)
        if strat == 0:
            k = np.random.randint(1, max(2, n//4))
            idx = np.random.choice(n, k, replace=False)
            step = 0.01 + 0.04 * np.random.rand()
            txs[idx] += np.random.randn(k) * step
            tys[idx] += np.random.randn(k) * step
        elif strat == 1:
            i, j = np.random.choice(n, 2, replace=False)
            txs[i], txs[j] = txs[j], txs[i]
            tys[i], tys[j] = tys[j], tys[i]
        elif strat == 2:
            step = 0.005 + 0.015 * np.random.rand()
            txs += np.random.randn(n) * step
            tys += np.random.randn(n) * step
        elif strat == 3:
            worst = np.argmin(cur_rs)
            txs[worst] = np.random.uniform(0.05, 0.95)
            tys[worst] = np.random.uniform(0.05, 0.95)
        else:
            # Move 2 worst circles
            worst2 = np.argsort(cur_rs)[:2]
            for w in worst2:
                txs[w] = np.random.uniform(0.05, 0.95)
                tys[w] = np.random.uniform(0.05, 0.95)

        txs = np.clip(txs, 0.02, 0.98)
        tys = np.clip(tys, 0.02, 0.98)

        r_est = 0.4 / np.sqrt(n)
        trs = np.full(n, r_est)

        txs, tys, trs = optimize_penalty(txs, tys, trs, n, max_iter=250)
        trs = repair_and_grow(txs, tys, trs, n)
        txs, tys, trs = local_search_move(txs, tys, trs, n, step=0.015, grid_pts=5, iters=2)
        trs = repair_and_grow(txs, tys, trs, n)

        valid, metric, _ = validate(txs, tys, trs, n)
        if valid:
            if metric > best_metric:
                bxs, bys, brs = txs.copy(), tys.copy(), trs.copy()
                best_metric = metric
                cur_xs, cur_ys, cur_rs = txs.copy(), tys.copy(), trs.copy()
                cur_metric = metric
                if verbose:
                    print(f"    BH {hop:3d}: {metric:.10f} NEW BEST")
            else:
                delta = metric - cur_metric
                if delta > 0 or np.random.rand() < np.exp(delta / 0.005):
                    cur_xs, cur_ys, cur_rs = txs.copy(), tys.copy(), trs.copy()
                    cur_metric = metric

    best = (bxs, bys, brs)

    # Phase 3: Fine refinement
    if verbose:
        print(f"\nPhase 3: Fine refinement")

    xs, ys, rs = best
    for step in [0.008, 0.004, 0.002, 0.001, 0.0005, 0.0002]:
        xs, ys, rs = local_search_move(xs, ys, rs, n, step=step, grid_pts=11, iters=5)
        rs = repair_and_grow(xs, ys, rs, n)

    valid, metric, _ = validate(xs, ys, rs, n)
    if valid and metric > best_metric:
        best_metric = metric
        best = (xs, ys, rs)

    elapsed = time.time() - t0
    if verbose:
        print(f"\nFinal n={n}: {best_metric:.10f} ({elapsed:.1f}s)")

    return best + (best_metric,)


def save_solution(xs, ys, rs, n, filepath):
    circles = [[float(xs[i]), float(ys[i]), float(rs[i])] for i in range(n)]
    with open(filepath, 'w') as f:
        json.dump({"circles": circles, "n": n, "metric": float(np.sum(rs))}, f, indent=2)
    print(f"Saved {filepath}")


if __name__ == "__main__":
    targets = {24: 2.530, 25: 2.587, 27: 2.685, 29: 2.790, 31: 2.889}

    if len(sys.argv) > 1:
        n_values = [int(x) for x in sys.argv[1:]]
    else:
        n_values = [29, 31, 24, 25, 27]

    out_dir = os.path.dirname(os.path.abspath(__file__))
    results = {}

    for n in n_values:
        np.random.seed(42 + n)
        sota = targets.get(n, 0)
        result = solve_n(n, num_starts=60, bh_hops=150)
        if result:
            xs, ys, rs, metric = result
            fp = os.path.join(out_dir, f"solution_n{n}.json")
            save_solution(xs, ys, rs, n, fp)
            results[n] = metric
            print(f"\n>>> n={n}: {metric:.10f}  SOTA={sota:.3f}  ({metric/sota*100:.1f}%)")
        else:
            results[n] = 0

    print("\n" + "="*60)
    for n in n_values:
        s = targets.get(n, 0)
        m = results.get(n, 0)
        print(f"  n={n}: {m:.10f}  SOTA={s:.3f}  ({m/s*100:.1f}%)" if s else f"  n={n}: {m:.10f}")
