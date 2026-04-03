"""
Fast circle packing solver: maximize sum of radii for n circles in [0,1]^2.

Key insight: Separate position optimization from radius assignment.
1. Optimize positions to spread circles well (maximize minimum distance)
2. Assign maximum feasible radii given positions
3. Jointly refine with penalty method
4. Basin-hopping with position perturbations
"""

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import json
import os
import time


def max_radius_at(x, y, other_xs, other_ys, other_rs, idx_skip=-1):
    """Maximum feasible radius for a circle at (x,y) given others."""
    # Wall constraints
    r = min(x, 1.0 - x, y, 1.0 - y)
    # Other circle constraints
    for j in range(len(other_xs)):
        if j == idx_skip:
            continue
        dist = np.sqrt((x - other_xs[j])**2 + (y - other_ys[j])**2)
        r = min(r, dist - other_rs[j])
    return max(r, 0.0)


def assign_radii(xs, ys, n):
    """Greedily assign maximum radii to circles.

    Process circles from center outward, assigning max feasible radius.
    Then iterate to improve.
    """
    rs = np.zeros(n)

    # Sort by distance from center (process center first)
    center_dist = (xs - 0.5)**2 + (ys - 0.5)**2
    order = np.argsort(center_dist)

    for i in order:
        r = min(xs[i], 1 - xs[i], ys[i], 1 - ys[i])
        for j in range(n):
            if j == i or rs[j] == 0:
                continue
            dist = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
            r = min(r, dist - rs[j])
        rs[i] = max(r, 1e-15)

    # Iterate: reassign radii in order of smallest first (to let them grow)
    for iteration in range(10):
        improved = False
        order = np.argsort(rs)
        for i in order:
            r = min(xs[i], 1 - xs[i], ys[i], 1 - ys[i])
            for j in range(n):
                if j == i:
                    continue
                dist = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                r = min(r, dist - rs[j])
            new_r = max(r, 1e-15)
            if new_r > rs[i] + 1e-15:
                rs[i] = new_r
                improved = True
        if not improved:
            break

    return rs


def spread_objective(pos_vec, n):
    """Objective for spreading circles: maximize sum of feasible radii.

    Uses negative of a proxy metric.
    """
    xs = pos_vec[:n]
    ys = pos_vec[n:]

    # Assign radii and return negative sum
    rs = assign_radii(xs, ys, n)
    return -np.sum(rs)


def penalty_spread_objective(pos_vec, n, mu=100):
    """Spread objective + penalty for being too close to walls/each other."""
    xs = pos_vec[:n]
    ys = pos_vec[n:]

    rs = assign_radii(xs, ys, n)
    obj = -np.sum(rs)

    # Penalty for negative radii (means overlapping)
    penalty = mu * np.sum(np.maximum(0, -rs)**2)

    return obj + penalty


def joint_objective(vec, n, mu):
    """Joint optimization of positions and radii with penalty."""
    xs = vec[:n]
    ys = vec[n:2*n]
    rs = vec[2*n:]

    obj = -np.sum(rs)

    penalty = 0.0
    # Containment
    penalty += np.sum(np.maximum(0, rs - xs)**2)
    penalty += np.sum(np.maximum(0, xs + rs - 1)**2)
    penalty += np.sum(np.maximum(0, rs - ys)**2)
    penalty += np.sum(np.maximum(0, ys + rs - 1)**2)

    # Non-overlap (vectorized)
    positions = np.column_stack([xs, ys])
    dists = pdist(positions)
    idx = 0
    overlaps = np.zeros(len(dists))
    r_sums = np.zeros(len(dists))
    k = 0
    for i in range(n):
        for j in range(i+1, n):
            r_sums[k] = rs[i] + rs[j]
            k += 1
    overlaps = np.maximum(0, r_sums - dists)
    penalty += np.sum(overlaps**2)

    # Negative radius penalty
    penalty += 10 * np.sum(np.maximum(0, -rs)**2)

    return obj + mu * penalty


def joint_objective_fast(vec, n, mu):
    """Faster joint optimization using precomputed pair indices."""
    xs = vec[:n]
    ys = vec[n:2*n]
    rs = vec[2*n:]

    obj = -np.sum(rs)

    # Containment penalties
    p = np.sum(np.maximum(0, rs - xs)**2)
    p += np.sum(np.maximum(0, xs + rs - 1)**2)
    p += np.sum(np.maximum(0, rs - ys)**2)
    p += np.sum(np.maximum(0, ys + rs - 1)**2)

    # Non-overlap
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist_sq = dx**2 + dy**2
    r_sum = rs[:, None] + rs[None, :]

    # Upper triangle only
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    dist_vals = np.sqrt(dist_sq[mask])
    r_sum_vals = r_sum[mask]
    overlaps = np.maximum(0, r_sum_vals - dist_vals)
    p += np.sum(overlaps**2)

    p += 10 * np.sum(np.maximum(0, -rs)**2)

    return obj + mu * p


def hex_positions(n):
    """Generate hexagonal grid positions for n circles."""
    side = int(np.ceil(np.sqrt(n * 2 / np.sqrt(3))))
    pts = []
    for row in range(side + 3):
        for col in range(side + 3):
            x = (col + 0.5 * (row % 2) + 0.5) / (side + 2)
            y = (row * np.sqrt(3) / 2 + 0.5) / (side + 2)
            if 0.02 < x < 0.98 and 0.02 < y < 0.98:
                pts.append((x, y))
    pts = np.array(pts)

    if len(pts) >= n:
        # Farthest point sampling
        selected = [len(pts) // 2]  # Start from center-ish
        for _ in range(n - 1):
            min_dists = np.min([np.sum((pts - pts[s])**2, axis=1) for s in selected], axis=0)
            min_dists[selected] = -1
            selected.append(np.argmax(min_dists))
        pts = pts[selected]
    else:
        extra = np.random.uniform(0.05, 0.95, (n - len(pts), 2))
        pts = np.vstack([pts, extra])

    return pts[:, 0].copy(), pts[:, 1].copy()


def random_positions(n):
    """Random positions avoiding walls."""
    margin = 0.5 / np.sqrt(n)
    xs = np.random.uniform(margin, 1 - margin, n)
    ys = np.random.uniform(margin, 1 - margin, n)
    return xs, ys


def grid_positions(n):
    """Grid positions."""
    side = int(np.ceil(np.sqrt(n)))
    xs, ys = [], []
    for i in range(side):
        for j in range(side):
            xs.append((i + 0.5) / side)
            ys.append((j + 0.5) / side)
    xs, ys = np.array(xs[:n]), np.array(ys[:n])
    return xs, ys


def optimize_positions(xs, ys, n, max_iter=300):
    """Optimize positions to maximize sum of feasible radii."""
    pos_vec = np.concatenate([xs, ys])
    margin = 0.02
    bounds = [(margin, 1-margin)] * n + [(margin, 1-margin)] * n

    result = minimize(
        spread_objective, pos_vec, args=(n,),
        method='L-BFGS-B', bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-12}
    )

    xs_opt = result.x[:n]
    ys_opt = result.x[n:]
    return xs_opt, ys_opt


def joint_optimize(xs, ys, rs, n, max_iter=500):
    """Joint optimization of positions and radii via penalty method."""
    vec = np.concatenate([xs, ys, rs])

    bounds = ([(1e-4, 1-1e-4)] * n +   # xs
              [(1e-4, 1-1e-4)] * n +   # ys
              [(1e-6, 0.5)] * n)        # rs

    for mu in [10, 100, 1000, 10000, 100000]:
        result = minimize(
            joint_objective_fast, vec, args=(n, mu),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12}
        )
        vec = result.x

    return vec[:n], vec[n:2*n], vec[2*n:]


def repair_and_maximize(xs, ys, rs, n):
    """Repair violations and maximize radii."""
    # First ensure containment
    for i in range(n):
        max_r = min(xs[i], 1 - xs[i], ys[i], 1 - ys[i])
        rs[i] = min(rs[i], max_r)

    # Fix overlaps by shrinking
    for _ in range(100):
        fixed = True
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                overlap = rs[i] + rs[j] - dist
                if overlap > 1e-12:
                    # Shrink proportionally
                    total = rs[i] + rs[j]
                    if total > 0:
                        shrink = overlap + 1e-13
                        rs[i] -= shrink * rs[i] / total
                        rs[j] -= shrink * rs[j] / total
                    fixed = False
        if fixed:
            break

    rs = np.maximum(rs, 1e-15)

    # Now try to grow each radius
    for _ in range(5):
        for i in range(n):
            max_r = min(xs[i], 1 - xs[i], ys[i], 1 - ys[i])
            for j in range(n):
                if j == i:
                    continue
                dist = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                max_r = min(max_r, dist - rs[j])
            if max_r > rs[i]:
                rs[i] = max_r - 1e-14  # tiny safety margin

    return rs


def local_search(xs, ys, rs, n, step=0.01, grid_size=7, iterations=3):
    """Move each circle to maximize its radius."""
    for it in range(iterations):
        improved = False
        order = np.random.permutation(n)

        for i in order:
            best_x, best_y, best_total = xs[i], ys[i], np.sum(rs)

            offsets = np.linspace(-step, step, grid_size)
            for dx in offsets:
                for dy in offsets:
                    nx = xs[i] + dx
                    ny = ys[i] + dy
                    if nx <= 0.005 or nx >= 0.995 or ny <= 0.005 or ny >= 0.995:
                        continue

                    # Max radius at new position
                    max_r = min(nx, 1-nx, ny, 1-ny)
                    for j in range(n):
                        if j == i:
                            continue
                        dist = np.sqrt((nx - xs[j])**2 + (ny - ys[j])**2)
                        max_r = min(max_r, dist - rs[j])

                    if max_r <= 0:
                        continue

                    new_total = np.sum(rs) - rs[i] + max_r
                    if new_total > best_total + 1e-14:
                        best_x, best_y = nx, ny
                        best_total = new_total
                        best_r = max_r

            if best_total > np.sum(rs) + 1e-14:
                xs[i] = best_x
                ys[i] = best_y
                rs[i] = best_r - 1e-14
                improved = True

        if not improved:
            break
        step *= 0.7  # Reduce step size

    return xs, ys, rs


def validate(xs, ys, rs, n, tol=1e-10):
    """Validate solution. Returns (valid, metric, max_violation)."""
    max_viol = 0.0
    for i in range(n):
        max_viol = max(max_viol, rs[i] - xs[i])
        max_viol = max(max_viol, xs[i] + rs[i] - 1.0)
        max_viol = max(max_viol, rs[i] - ys[i])
        max_viol = max(max_viol, ys[i] + rs[i] - 1.0)

    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
            max_viol = max(max_viol, rs[i] + rs[j] - dist)

    metric = np.sum(rs)
    return max_viol <= tol, metric, max_viol


def solve_single_start(n, init_type, noise=0.0):
    """Single optimization start. Returns (xs, ys, rs, metric) or None."""
    # Initialize positions
    if init_type == 'hex':
        xs, ys = hex_positions(n)
    elif init_type == 'grid':
        xs, ys = grid_positions(n)
    elif init_type == 'random':
        xs, ys = random_positions(n)
    else:
        xs, ys = hex_positions(n)

    if noise > 0:
        xs += np.random.randn(n) * noise
        ys += np.random.randn(n) * noise
        xs = np.clip(xs, 0.02, 0.98)
        ys = np.clip(ys, 0.02, 0.98)

    # Phase 1: Optimize positions for max spread
    xs, ys = optimize_positions(xs, ys, n, max_iter=200)

    # Phase 2: Assign radii
    rs = assign_radii(xs, ys, n)

    # Phase 3: Joint optimization
    xs, ys, rs = joint_optimize(xs, ys, rs, n, max_iter=300)

    # Phase 4: Repair
    rs = repair_and_maximize(xs, ys, rs, n)

    # Phase 5: Local search
    xs, ys, rs = local_search(xs, ys, rs, n, step=0.02, grid_size=7, iterations=3)
    rs = repair_and_maximize(xs, ys, rs, n)

    valid, metric, max_viol = validate(xs, ys, rs, n)
    if valid:
        return xs, ys, rs, metric
    else:
        # Try harder repair
        rs = repair_and_maximize(xs, ys, rs, n)
        valid, metric, max_viol = validate(xs, ys, rs, n)
        if valid:
            return xs, ys, rs, metric
    return None


def basin_hop(xs, ys, rs, n, n_hops=200, temperature=0.005):
    """Basin hopping on a good solution."""
    best_xs, best_ys, best_rs = xs.copy(), ys.copy(), rs.copy()
    best_metric = np.sum(rs)

    cur_xs, cur_ys, cur_rs = xs.copy(), ys.copy(), rs.copy()
    cur_metric = best_metric

    for hop in range(n_hops):
        # Perturb
        trial_xs, trial_ys = cur_xs.copy(), cur_ys.copy()

        strategy = np.random.randint(4)
        if strategy == 0:
            # Shift random subset
            k = np.random.randint(1, max(2, n//4))
            idx = np.random.choice(n, k, replace=False)
            step = 0.01 + 0.04 * np.random.rand()
            trial_xs[idx] += np.random.randn(k) * step
            trial_ys[idx] += np.random.randn(k) * step
        elif strategy == 1:
            # Swap two circles
            i, j = np.random.choice(n, 2, replace=False)
            trial_xs[i], trial_xs[j] = trial_xs[j], trial_xs[i]
            trial_ys[i], trial_ys[j] = trial_ys[j], trial_ys[i]
        elif strategy == 2:
            # Shake all
            step = 0.005 + 0.02 * np.random.rand()
            trial_xs += np.random.randn(n) * step
            trial_ys += np.random.randn(n) * step
        else:
            # Move worst circle to random position
            worst = np.argmin(cur_rs)
            trial_xs[worst] = np.random.uniform(0.05, 0.95)
            trial_ys[worst] = np.random.uniform(0.05, 0.95)

        trial_xs = np.clip(trial_xs, 0.02, 0.98)
        trial_ys = np.clip(trial_ys, 0.02, 0.98)

        # Re-optimize
        trial_xs, trial_ys = optimize_positions(trial_xs, trial_ys, n, max_iter=100)
        trial_rs = assign_radii(trial_xs, trial_ys, n)
        trial_xs, trial_ys, trial_rs = joint_optimize(trial_xs, trial_ys, trial_rs, n, max_iter=200)
        trial_rs = repair_and_maximize(trial_xs, trial_ys, trial_rs, n)
        trial_xs, trial_ys, trial_rs = local_search(trial_xs, trial_ys, trial_rs, n, step=0.015, grid_size=5, iterations=2)
        trial_rs = repair_and_maximize(trial_xs, trial_ys, trial_rs, n)

        valid, metric, _ = validate(trial_xs, trial_ys, trial_rs, n)

        if valid:
            if metric > best_metric:
                best_xs, best_ys, best_rs = trial_xs.copy(), trial_ys.copy(), trial_rs.copy()
                best_metric = metric
                cur_xs, cur_ys, cur_rs = trial_xs.copy(), trial_ys.copy(), trial_rs.copy()
                cur_metric = metric
                print(f"    BH hop {hop}: NEW BEST {metric:.10f}")
            else:
                delta = metric - cur_metric
                if delta > 0 or np.random.rand() < np.exp(delta / temperature):
                    cur_xs, cur_ys, cur_rs = trial_xs.copy(), trial_ys.copy(), trial_rs.copy()
                    cur_metric = metric

    return best_xs, best_ys, best_rs, best_metric


def solve_n(n, num_starts=60, bh_hops=200, verbose=True):
    """Full solver for given n."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Solving n={n}")
        print(f"{'='*60}")

    best_result = None
    best_metric = -1
    t0 = time.time()

    # Phase 1: Multi-start
    if verbose:
        print(f"\nPhase 1: Multi-start ({num_starts} starts)")

    init_types = ['hex', 'grid', 'random']

    for s in range(num_starts):
        init = init_types[s % len(init_types)]
        noise = 0.005 * (s // len(init_types))

        try:
            result = solve_single_start(n, init, noise=noise)
            if result is not None:
                xs, ys, rs, metric = result
                if metric > best_metric:
                    best_metric = metric
                    best_result = (xs.copy(), ys.copy(), rs.copy())
                    if verbose and (s < 10 or s % 10 == 0):
                        print(f"  Start {s} ({init}): {metric:.10f} NEW BEST")
        except Exception as e:
            if verbose and s < 3:
                print(f"  Start {s}: FAILED ({e})")

    if verbose:
        print(f"Phase 1 best: {best_metric:.10f} ({time.time()-t0:.1f}s)")

    if best_result is None:
        print("No valid solution found!")
        return None

    # Phase 2: Basin hopping
    if verbose:
        print(f"\nPhase 2: Basin hopping ({bh_hops} hops)")

    xs, ys, rs = best_result
    xs, ys, rs, bh_metric = basin_hop(xs, ys, rs, n, n_hops=bh_hops)
    if bh_metric > best_metric:
        best_metric = bh_metric
        best_result = (xs.copy(), ys.copy(), rs.copy())
        if verbose:
            print(f"Basin hopping improved to: {best_metric:.10f}")

    # Phase 3: Fine local search
    if verbose:
        print(f"\nPhase 3: Fine refinement")

    xs, ys, rs = best_result
    for step in [0.01, 0.005, 0.002, 0.001, 0.0005]:
        xs, ys, rs = local_search(xs, ys, rs, n, step=step, grid_size=9, iterations=3)
        rs = repair_and_maximize(xs, ys, rs, n)

    valid, metric, _ = validate(xs, ys, rs, n)
    if valid and metric > best_metric:
        best_metric = metric
        best_result = (xs.copy(), ys.copy(), rs.copy())

    elapsed = time.time() - t0
    if verbose:
        print(f"\nFinal n={n}: metric={best_metric:.10f} ({elapsed:.1f}s)")

    return best_result + (best_metric,)


def save_solution(xs, ys, rs, n, filepath):
    """Save solution as JSON."""
    circles = [[float(xs[i]), float(ys[i]), float(rs[i])] for i in range(n)]
    data = {"circles": circles, "n": n, "metric": float(np.sum(rs))}
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved to {filepath}")


def main():
    import sys

    targets = {
        24: 2.530,
        25: 2.587,
        27: 2.685,
        29: 2.790,
        31: 2.889,
    }

    # Parse command line for specific n
    if len(sys.argv) > 1:
        n_values = [int(x) for x in sys.argv[1:]]
    else:
        n_values = [29, 31, 24, 25, 27]

    out_dir = os.path.dirname(os.path.abspath(__file__))
    results = {}

    for n in n_values:
        np.random.seed(42 + n)
        sota = targets.get(n, 0)

        result = solve_n(n, num_starts=60, bh_hops=200)

        if result is not None:
            xs, ys, rs, metric = result
            filepath = os.path.join(out_dir, f"solution_n{n}.json")
            save_solution(xs, ys, rs, n, filepath)
            results[n] = metric
            print(f"\n>>> n={n}: metric={metric:.10f}, SOTA={sota:.3f}, ratio={metric/sota:.4f}")
        else:
            results[n] = 0
            print(f"\n>>> n={n}: FAILED")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for n in n_values:
        sota = targets.get(n, 0)
        m = results.get(n, 0)
        pct = m/sota*100 if sota > 0 else 0
        print(f"  n={n}: metric={m:.10f}  SOTA={sota:.3f}  ({pct:.1f}%)")


if __name__ == "__main__":
    main()
