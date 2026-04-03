"""
Circle packing solver v3: maximize sum of radii for n circles in [0,1]^2.

Improvements over v2:
- Much more basin hopping iterations
- Smarter perturbation strategies
- Multiple rounds of refinement
- Restart from perturbed best when stuck
"""

import numpy as np
from scipy.optimize import minimize
import json
import os
import time
import sys


def joint_penalty_and_grad(vec, n, mu):
    """Combined objective + gradient computation (more efficient)."""
    xs = vec[:n]
    ys = vec[n:2*n]
    rs = vec[2*n:]

    obj = -np.sum(rs)
    grad_x = np.zeros(n)
    grad_y = np.zeros(n)
    grad_r = -np.ones(n)

    # Containment
    v = np.maximum(0, rs - xs)
    obj += mu * np.sum(v**2)
    grad_x -= 2 * mu * v
    grad_r += 2 * mu * v

    v = np.maximum(0, xs + rs - 1)
    obj += mu * np.sum(v**2)
    grad_x += 2 * mu * v
    grad_r += 2 * mu * v

    v = np.maximum(0, rs - ys)
    obj += mu * np.sum(v**2)
    grad_y -= 2 * mu * v
    grad_r += 2 * mu * v

    v = np.maximum(0, ys + rs - 1)
    obj += mu * np.sum(v**2)
    grad_y += 2 * mu * v
    grad_r += 2 * mu * v

    # Non-overlap
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist_sq = dx**2 + dy**2
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    r_sum = rs[:, None] + rs[None, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    overlap = np.maximum(0, r_sum - dist) * mask

    obj += mu * np.sum(overlap**2)

    inv_dist = np.where(dist > 1e-15, 1.0/dist, 0.0)
    olap_factor = 2 * mu * overlap * inv_dist

    grad_x += np.sum(olap_factor * dx, axis=1) - np.sum(olap_factor * dx, axis=0)
    grad_y += np.sum(olap_factor * dy, axis=1) - np.sum(olap_factor * dy, axis=0)

    olap_r = 2 * mu * overlap
    grad_r -= np.sum(olap_r, axis=1) + np.sum(olap_r, axis=0)

    # Negative radius
    neg_v = np.maximum(0, -rs)
    obj += 100 * mu * np.sum(neg_v**2)
    grad_r -= 200 * mu * neg_v

    return obj, np.concatenate([grad_x, grad_y, grad_r])


def optimize_penalty(xs, ys, rs, n, max_iter=400):
    """Progressive penalty with analytical gradients."""
    vec = np.concatenate([xs, ys, rs])
    bounds = ([(1e-4, 1-1e-4)] * n +
              [(1e-4, 1-1e-4)] * n +
              [(1e-6, 0.5)] * n)

    for mu in [10, 100, 1000, 10000, 100000]:
        result = minimize(
            lambda v: joint_penalty_and_grad(v, n, mu),
            vec, jac=True,
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12}
        )
        vec = result.x

    return vec[:n], vec[n:2*n], vec[2*n:]


def repair_and_grow(xs, ys, rs, n):
    """Repair violations, then grow radii maximally."""
    for i in range(n):
        wall_max = min(xs[i], 1-xs[i], ys[i], 1-ys[i])
        rs[i] = min(rs[i], wall_max)

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


def local_search(xs, ys, rs, n, step=0.01, grid_pts=9, iters=3):
    """Move each circle to maximize its radius."""
    for it in range(iters):
        improved = False
        order = np.random.permutation(n)
        offsets = np.linspace(-step, step, grid_pts)

        for i in order:
            best_x, best_y = xs[i], ys[i]
            best_gain = 0.0

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
                        d = np.sqrt((nx - xs[j])**2 + (ny - ys[j])**2)
                        max_r = min(max_r, d - rs[j])

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
        step *= 0.65

    return xs, ys, rs


def hex_init(n, noise=0.0):
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


def full_optimize(xs, ys, n, r_est=None):
    """Full optimization pipeline from initial positions."""
    if r_est is None:
        r_est = 0.35 / np.sqrt(n)
    rs = np.full(n, r_est)

    xs, ys, rs = optimize_penalty(xs, ys, rs, n, max_iter=400)
    rs = repair_and_grow(xs, ys, rs, n)
    xs, ys, rs = local_search(xs, ys, rs, n, step=0.02, grid_pts=7, iters=3)
    rs = repair_and_grow(xs, ys, rs, n)

    valid, metric, viol = validate(xs, ys, rs, n)
    if not valid:
        rs *= 0.9999
        rs = repair_and_grow(xs, ys, rs, n)
        valid, metric, viol = validate(xs, ys, rs, n)

    return xs, ys, rs, valid, metric


def solve_n(n, time_budget=300, verbose=True):
    """Full solver with time budget in seconds."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Solving n={n} (budget={time_budget}s)")
        print(f"{'='*60}")

    best_xs, best_ys, best_rs = None, None, None
    best_metric = -1
    t0 = time.time()

    inits = ['hex', 'grid', 'random']

    # Phase 1: Multi-start
    if verbose:
        print(f"\nPhase 1: Multi-start")

    s = 0
    while time.time() - t0 < time_budget * 0.15:  # 15% of budget
        init = inits[s % len(inits)]
        noise = 0.003 * (s // len(inits))

        try:
            if init == 'hex':
                xs, ys = hex_init(n, noise)
            elif init == 'grid':
                xs, ys = grid_init(n, noise)
            else:
                xs, ys = random_init(n)

            xs, ys, rs, valid, metric = full_optimize(xs, ys, n)
            if valid and metric > best_metric:
                best_metric = metric
                best_xs, best_ys, best_rs = xs.copy(), ys.copy(), rs.copy()
                if verbose:
                    print(f"  Start {s:3d}: {metric:.10f} BEST")
        except Exception as e:
            pass
        s += 1

    if verbose:
        print(f"Phase 1: {s} starts, best={best_metric:.10f} ({time.time()-t0:.1f}s)")

    if best_xs is None:
        return None

    # Phase 2: Basin hopping (main phase - 70% of budget)
    if verbose:
        print(f"\nPhase 2: Basin hopping")

    cur_xs, cur_ys, cur_rs = best_xs.copy(), best_ys.copy(), best_rs.copy()
    cur_metric = best_metric
    stale_count = 0
    hop = 0

    while time.time() - t0 < time_budget * 0.85:
        txs, tys = cur_xs.copy(), cur_ys.copy()

        # Diversify perturbation based on staleness
        if stale_count > 30:
            # Big restart from best with large perturbation
            txs, tys = best_xs.copy(), best_ys.copy()
            txs += np.random.randn(n) * 0.05
            tys += np.random.randn(n) * 0.05
            stale_count = 0
        elif stale_count > 15:
            # Medium perturbation
            k = np.random.randint(n//3, n//2 + 1)
            idx = np.random.choice(n, k, replace=False)
            txs[idx] += np.random.randn(k) * 0.04
            tys[idx] += np.random.randn(k) * 0.04
        else:
            strat = np.random.randint(7)
            if strat == 0:
                # Shift small subset
                k = np.random.randint(1, max(2, n//5))
                idx = np.random.choice(n, k, replace=False)
                step = 0.01 + 0.03 * np.random.rand()
                txs[idx] += np.random.randn(k) * step
                tys[idx] += np.random.randn(k) * step
            elif strat == 1:
                # Swap two circles
                i, j = np.random.choice(n, 2, replace=False)
                txs[i], txs[j] = txs[j], txs[i]
                tys[i], tys[j] = tys[j], tys[i]
            elif strat == 2:
                # Small global shake
                step = 0.005 + 0.015 * np.random.rand()
                txs += np.random.randn(n) * step
                tys += np.random.randn(n) * step
            elif strat == 3:
                # Move worst circle
                worst = np.argmin(cur_rs)
                txs[worst] = np.random.uniform(0.05, 0.95)
                tys[worst] = np.random.uniform(0.05, 0.95)
            elif strat == 4:
                # Move 2-3 worst circles
                k = min(3, n)
                worst_k = np.argsort(cur_rs)[:k]
                for w in worst_k:
                    txs[w] = np.random.uniform(0.05, 0.95)
                    tys[w] = np.random.uniform(0.05, 0.95)
            elif strat == 5:
                # Rotate a subset around center
                angle = np.random.uniform(-0.3, 0.3)
                k = np.random.randint(2, max(3, n//3))
                idx = np.random.choice(n, k, replace=False)
                cx, cy = np.mean(txs[idx]), np.mean(tys[idx])
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                for ii in idx:
                    dx, dy = txs[ii] - cx, tys[ii] - cy
                    txs[ii] = cx + cos_a * dx - sin_a * dy
                    tys[ii] = cy + sin_a * dx + cos_a * dy
            else:
                # Mirror a subset
                k = np.random.randint(1, max(2, n//4))
                idx = np.random.choice(n, k, replace=False)
                if np.random.rand() < 0.5:
                    txs[idx] = 1.0 - txs[idx]
                else:
                    tys[idx] = 1.0 - tys[idx]

        txs = np.clip(txs, 0.02, 0.98)
        tys = np.clip(tys, 0.02, 0.98)

        txs, tys, trs, valid, metric = full_optimize(txs, tys, n)

        if valid:
            if metric > best_metric:
                best_xs, best_ys, best_rs = txs.copy(), tys.copy(), trs.copy()
                best_metric = metric
                cur_xs, cur_ys, cur_rs = txs.copy(), tys.copy(), trs.copy()
                cur_metric = metric
                stale_count = 0
                if verbose:
                    print(f"    BH {hop:4d}: {metric:.10f} BEST ({time.time()-t0:.0f}s)")
            else:
                delta = metric - cur_metric
                temp = 0.003
                if delta > 0 or np.random.rand() < np.exp(delta / temp):
                    cur_xs, cur_ys, cur_rs = txs.copy(), tys.copy(), trs.copy()
                    cur_metric = metric
                stale_count += 1
        else:
            stale_count += 1

        hop += 1

    if verbose:
        print(f"Phase 2: {hop} hops, best={best_metric:.10f} ({time.time()-t0:.1f}s)")

    # Phase 3: Fine refinement (15% of budget)
    if verbose:
        print(f"\nPhase 3: Fine refinement")

    xs, ys, rs = best_xs.copy(), best_ys.copy(), best_rs.copy()
    while time.time() - t0 < time_budget:
        step = 0.005 * np.random.rand() + 0.0005
        grid_pts = np.random.choice([9, 11, 13])
        xs, ys, rs = local_search(xs, ys, rs, n, step=step, grid_pts=grid_pts, iters=2)
        rs = repair_and_grow(xs, ys, rs, n)

        valid, metric, _ = validate(xs, ys, rs, n)
        if valid and metric > best_metric:
            best_metric = metric
            best_xs, best_ys, best_rs = xs.copy(), ys.copy(), rs.copy()
            if verbose:
                print(f"    Refine: {metric:.10f} BEST")
        else:
            xs, ys, rs = best_xs.copy(), best_ys.copy(), best_rs.copy()

    elapsed = time.time() - t0
    if verbose:
        print(f"\nFinal n={n}: {best_metric:.10f} ({elapsed:.1f}s)")

    return best_xs, best_ys, best_rs, best_metric


def save_solution(xs, ys, rs, n, filepath):
    circles = [[float(xs[i]), float(ys[i]), float(rs[i])] for i in range(n)]
    with open(filepath, 'w') as f:
        json.dump({"circles": circles, "n": n, "metric": float(np.sum(rs))}, f, indent=2)
    print(f"Saved {filepath}")


if __name__ == "__main__":
    targets = {24: 2.530, 25: 2.587, 27: 2.685, 29: 2.790, 31: 2.889}

    if len(sys.argv) > 1:
        n_values = [int(sys.argv[1])]
        budget = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    else:
        n_values = [29, 31, 24, 25, 27]
        budget = 300

    out_dir = os.path.dirname(os.path.abspath(__file__))
    results = {}

    for n_val in n_values:
        np.random.seed(42 + n_val)
        sota = targets.get(n_val, 0)
        result = solve_n(n_val, time_budget=budget)
        if result:
            xs, ys, rs, metric = result
            fp = os.path.join(out_dir, f"solution_n{n_val}.json")
            save_solution(xs, ys, rs, n_val, fp)
            results[n_val] = metric
            print(f"\n>>> n={n_val}: {metric:.10f}  SOTA={sota:.3f}  ({metric/sota*100:.1f}%)")
        else:
            results[n_val] = 0

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for n_val in n_values:
        s = targets.get(n_val, 0)
        m = results.get(n_val, 0)
        print(f"  n={n_val}: {m:.10f}  SOTA={s:.3f}  ({m/s*100:.1f}%)" if s else f"  n={n_val}: {m:.10f}")
