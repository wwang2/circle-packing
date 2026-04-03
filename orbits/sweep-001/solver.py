"""
Circle packing solver: maximize sum of radii for n circles in [0,1]^2.

Strategy:
1. Multi-start with diverse initializations (hex, grid, ring, random)
2. L-BFGS-B with progressive penalty method
3. SLSQP polish for constraint satisfaction
4. Basin-hopping on best solution
5. Single-circle repositioning refinement
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist, squareform
import json
import sys
import time
import os

# ─── Initialization strategies ───────────────────────────────────────

def hex_init(n, noise=0.0):
    """Hexagonal grid initialization."""
    # Estimate grid size
    side = int(np.ceil(np.sqrt(n)))
    pts = []
    for row in range(side + 2):
        for col in range(side + 2):
            x = (col + 0.5 * (row % 2)) / (side + 1)
            y = row * np.sqrt(3) / 2 / (side + 1)
            if 0.05 < x < 0.95 and 0.05 < y < 0.95:
                pts.append((x, y))
    pts = np.array(pts)
    if len(pts) < n:
        # pad with random
        extra = np.random.uniform(0.1, 0.9, (n - len(pts), 2))
        pts = np.vstack([pts, extra])
    # Select n points spread out
    if len(pts) > n:
        # greedy farthest-point selection
        selected = [0]
        for _ in range(n - 1):
            dists = np.min([np.sum((pts - pts[s])**2, axis=1) for s in selected], axis=0)
            dists[selected] = -1
            selected.append(np.argmax(dists))
        pts = pts[selected]
    r_est = 0.4 / np.sqrt(n)
    radii = np.full(n, r_est)
    if noise > 0:
        pts += np.random.randn(*pts.shape) * noise
    return pts, radii


def grid_init(n, noise=0.0):
    """Regular grid initialization."""
    side = int(np.ceil(np.sqrt(n)))
    xs = np.linspace(1/(2*side), 1 - 1/(2*side), side)
    ys = np.linspace(1/(2*side), 1 - 1/(2*side), side)
    pts = np.array([(x, y) for x in xs for y in ys])[:n]
    if len(pts) < n:
        extra = np.random.uniform(0.1, 0.9, (n - len(pts), 2))
        pts = np.vstack([pts, extra])
    r_est = 0.4 / np.sqrt(n)
    radii = np.full(n, r_est)
    if noise > 0:
        pts += np.random.randn(*pts.shape) * noise
    return pts, radii


def random_init(n):
    """Random initialization."""
    r_est = 0.35 / np.sqrt(n)
    pts = np.random.uniform(r_est + 0.01, 1 - r_est - 0.01, (n, 2))
    radii = np.full(n, r_est * np.random.uniform(0.5, 1.0))
    return pts, radii


def ring_init(n, noise=0.0):
    """Concentric rings initialization."""
    pts = []
    # Outer ring, middle ring, center
    layers = []
    remaining = n
    r_outer = 0.4
    layer_idx = 0
    while remaining > 0:
        if layer_idx == 0:
            count = min(remaining, max(6, int(n * 0.5)))
        elif layer_idx == 1:
            count = min(remaining, max(3, int(n * 0.3)))
        else:
            count = remaining
        layers.append(count)
        remaining -= count
        layer_idx += 1

    for li, count in enumerate(layers):
        if li == len(layers) - 1 and count <= 2:
            # Center circles
            for i in range(count):
                angle = i * np.pi
                r = 0.05 * li
                pts.append((0.5 + r * np.cos(angle), 0.5 + r * np.sin(angle)))
        else:
            radius = 0.35 - li * 0.12
            for i in range(count):
                angle = 2 * np.pi * i / count + li * 0.3
                pts.append((0.5 + radius * np.cos(angle), 0.5 + radius * np.sin(angle)))

    pts = np.array(pts[:n])
    pts = np.clip(pts, 0.05, 0.95)
    r_est = 0.35 / np.sqrt(n)
    radii = np.full(n, r_est)
    if noise > 0:
        pts += np.random.randn(*pts.shape) * noise
    return pts, radii


# ─── Optimization core ───────────────────────────────────────────────

def pack_to_vec(pts, radii):
    """Pack (n,2) positions and (n,) radii into flat vector [x0,y0,r0,x1,y1,r1,...]."""
    n = len(radii)
    vec = np.empty(3 * n)
    vec[0::3] = pts[:, 0]
    vec[1::3] = pts[:, 1]
    vec[2::3] = radii
    return vec


def vec_to_pack(vec):
    """Unpack flat vector to positions and radii."""
    n = len(vec) // 3
    xs = vec[0::3]
    ys = vec[1::3]
    rs = vec[2::3]
    return xs, ys, rs


def penalty_objective(vec, n, mu):
    """Negative sum of radii + penalty for constraint violations."""
    xs, ys, rs = vec_to_pack(vec)

    # Objective: minimize negative sum of radii
    obj = -np.sum(rs)

    penalty = 0.0
    # Containment: r <= x, x+r <= 1, r <= y, y+r <= 1
    viol_left = np.maximum(0, rs - xs)
    viol_right = np.maximum(0, xs + rs - 1.0)
    viol_bottom = np.maximum(0, rs - ys)
    viol_top = np.maximum(0, ys + rs - 1.0)
    penalty += np.sum(viol_left**2 + viol_right**2 + viol_bottom**2 + viol_top**2)

    # Non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
            overlap = rs[i] + rs[j] - dist
            if overlap > 0:
                penalty += overlap**2

    return obj + mu * penalty


def penalty_objective_vectorized(vec, n, mu):
    """Vectorized penalty objective - faster for larger n."""
    xs, ys, rs = vec_to_pack(vec)

    obj = -np.sum(rs)

    # Containment penalties
    viol = np.maximum(0, rs - xs)
    penalty = np.sum(viol**2)
    viol = np.maximum(0, xs + rs - 1.0)
    penalty += np.sum(viol**2)
    viol = np.maximum(0, rs - ys)
    penalty += np.sum(viol**2)
    viol = np.maximum(0, ys + rs - 1.0)
    penalty += np.sum(viol**2)

    # Non-overlap (vectorized via pdist)
    positions = np.column_stack([xs, ys])
    dists = pdist(positions)
    # sum of radii pairs
    r_sums = np.array([rs[i] + rs[j] for i in range(n) for j in range(i+1, n)])
    overlaps = np.maximum(0, r_sums - dists)
    penalty += np.sum(overlaps**2)

    # Negative radii penalty
    neg_r = np.maximum(0, -rs)
    penalty += 100 * np.sum(neg_r**2)

    return obj + mu * penalty


def get_bounds(n):
    """Get variable bounds for L-BFGS-B."""
    bounds = []
    for i in range(n):
        bounds.append((1e-6, 1.0 - 1e-6))  # x
        bounds.append((1e-6, 1.0 - 1e-6))  # y
        bounds.append((1e-6, 0.5))           # r
    return bounds


def get_slsqp_constraints(n):
    """Build SLSQP constraints for the circle packing problem."""
    constraints = []

    # Containment constraints: x_i - r_i >= 0, 1 - x_i - r_i >= 0, etc.
    for i in range(n):
        xi, yi, ri = 3*i, 3*i+1, 3*i+2
        constraints.append({'type': 'ineq', 'fun': lambda v, idx=xi, ridx=ri: v[idx] - v[ridx]})
        constraints.append({'type': 'ineq', 'fun': lambda v, idx=xi, ridx=ri: 1.0 - v[idx] - v[ridx]})
        constraints.append({'type': 'ineq', 'fun': lambda v, idx=yi, ridx=ri: v[idx] - v[ridx]})
        constraints.append({'type': 'ineq', 'fun': lambda v, idx=yi, ridx=ri: 1.0 - v[idx] - v[ridx]})
        # Positive radius
        constraints.append({'type': 'ineq', 'fun': lambda v, ridx=ri: v[ridx] - 1e-8})

    # Non-overlap constraints: dist(i,j) - r_i - r_j >= 0
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = 3*i, 3*i+1, 3*i+2
            xj, yj, rj = 3*j, 3*j+1, 3*j+2
            def nonoverlap(v, _xi=xi, _yi=yi, _ri=ri, _xj=xj, _yj=yj, _rj=rj):
                dist = np.sqrt((v[_xi] - v[_xj])**2 + (v[_yi] - v[_yj])**2)
                return dist - v[_ri] - v[_rj]
            constraints.append({'type': 'ineq', 'fun': nonoverlap})

    return constraints


def slsqp_objective(vec):
    """Simple objective for SLSQP: minimize -sum(radii)."""
    return -np.sum(vec[2::3])


def progressive_penalty_optimize(pts, radii, n, max_iter=300):
    """Progressive penalty method with increasing mu."""
    vec = pack_to_vec(pts, radii)
    bounds = get_bounds(n)

    for mu in [1e1, 1e2, 1e3, 1e4, 1e5]:
        result = minimize(
            penalty_objective_vectorized, vec, args=(n, mu),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-10}
        )
        vec = result.x

    return vec


def slsqp_polish(vec, n, max_iter=500):
    """Polish solution with SLSQP to satisfy constraints exactly."""
    bounds = get_bounds(n)
    constraints = get_slsqp_constraints(n)

    result = minimize(
        slsqp_objective, vec,
        method='SLSQP', bounds=bounds, constraints=constraints,
        options={'maxiter': max_iter, 'ftol': 1e-15}
    )
    return result.x, result.fun


def validate_solution(vec, n, tol=1e-10):
    """Check if solution is valid. Returns (valid, metric, max_violation)."""
    xs, ys, rs = vec_to_pack(vec)
    max_viol = 0.0

    for i in range(n):
        max_viol = max(max_viol, rs[i] - xs[i])
        max_viol = max(max_viol, xs[i] + rs[i] - 1.0)
        max_viol = max(max_viol, rs[i] - ys[i])
        max_viol = max(max_viol, ys[i] + rs[i] - 1.0)
        if rs[i] <= 0:
            max_viol = max(max_viol, -rs[i])

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
            overlap = rs[i] + rs[j] - dist
            max_viol = max(max_viol, overlap)

    metric = np.sum(rs)
    valid = max_viol <= tol
    return valid, metric, max_viol


def repair_solution(vec, n, tol=1e-10, max_rounds=50):
    """Attempt to repair a solution by shrinking radii to eliminate violations."""
    xs, ys, rs = vec_to_pack(vec)

    for _ in range(max_rounds):
        changed = False
        # Containment
        for i in range(n):
            max_r = min(xs[i], 1 - xs[i], ys[i], 1 - ys[i])
            if rs[i] > max_r + tol/10:
                rs[i] = max_r - tol
                changed = True

        # Non-overlap
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                if rs[i] + rs[j] > dist + tol/10:
                    # Shrink both proportionally
                    excess = rs[i] + rs[j] - dist + tol
                    shrink_i = excess * rs[i] / (rs[i] + rs[j])
                    shrink_j = excess * rs[j] / (rs[i] + rs[j])
                    rs[i] -= shrink_i
                    rs[j] -= shrink_j
                    changed = True

        if not changed:
            break

    rs = np.maximum(rs, 1e-12)
    vec_out = pack_to_vec(np.column_stack([xs, ys]), rs)
    return vec_out


def single_circle_refine(vec, n, iterations=3):
    """Refine by re-optimizing each circle individually."""
    xs, ys, rs = vec_to_pack(vec)

    for _ in range(iterations):
        improved = False
        order = np.random.permutation(n)
        for i in order:
            # Find maximum radius for circle i at its current position
            # given all other circles and walls
            best_x, best_y, best_r = xs[i], ys[i], rs[i]

            # Try a grid of positions near current
            for dx in np.linspace(-0.05, 0.05, 7):
                for dy in np.linspace(-0.05, 0.05, 7):
                    nx, ny = xs[i] + dx, ys[i] + dy
                    if nx <= 0.001 or nx >= 0.999 or ny <= 0.001 or ny >= 0.999:
                        continue
                    # Max radius from walls
                    max_r = min(nx, 1 - nx, ny, 1 - ny)
                    # Max radius from other circles
                    for j in range(n):
                        if j == i:
                            continue
                        dist = np.sqrt((nx - xs[j])**2 + (ny - ys[j])**2)
                        max_r = min(max_r, dist - rs[j])
                    if max_r > best_r + 1e-12:
                        best_x, best_y, best_r = nx, ny, max_r
                        improved = True

            xs[i], ys[i], rs[i] = best_x, best_y, max(best_r, 1e-12)

        if not improved:
            break

    return pack_to_vec(np.column_stack([xs, ys]), rs)


def fine_single_circle_refine(vec, n, iterations=5):
    """Fine-grained single circle refinement with smaller steps."""
    xs, ys, rs = vec_to_pack(vec)

    for iteration in range(iterations):
        improved = False
        step = 0.02 / (iteration + 1)
        order = np.random.permutation(n)
        for i in order:
            best_x, best_y, best_r = xs[i], ys[i], rs[i]

            for dx in np.linspace(-step, step, 11):
                for dy in np.linspace(-step, step, 11):
                    nx, ny = xs[i] + dx, ys[i] + dy
                    if nx <= 0.001 or nx >= 0.999 or ny <= 0.001 or ny >= 0.999:
                        continue
                    max_r = min(nx, 1 - nx, ny, 1 - ny)
                    for j in range(n):
                        if j == i:
                            continue
                        dist = np.sqrt((nx - xs[j])**2 + (ny - ys[j])**2)
                        max_r = min(max_r, dist - rs[j])
                    if max_r > best_r + 1e-14:
                        best_x, best_y, best_r = nx, ny, max_r
                        improved = True

            xs[i], ys[i], rs[i] = best_x, best_y, max(best_r, 1e-12)

        if not improved:
            break

    return pack_to_vec(np.column_stack([xs, ys]), rs)


def basin_hopping_optimize(vec, n, n_hops=300, temperature=0.01):
    """Basin hopping: random perturbation + local optimization."""
    best_vec = vec.copy()
    valid, best_metric, _ = validate_solution(vec, n)
    if not valid:
        best_metric = -999

    current_vec = vec.copy()
    current_metric = best_metric

    for hop in range(n_hops):
        # Random perturbation
        trial = current_vec.copy()
        xs, ys, rs = vec_to_pack(trial)

        # Perturbation strategy
        strategy = np.random.choice(['shift', 'swap', 'shake', 'grow'])

        if strategy == 'shift':
            # Shift a random subset of circles
            k = np.random.randint(1, max(2, n // 3))
            indices = np.random.choice(n, k, replace=False)
            step = 0.02 + 0.05 * np.random.rand()
            xs[indices] += np.random.randn(k) * step
            ys[indices] += np.random.randn(k) * step
        elif strategy == 'swap':
            # Swap two circles
            i, j = np.random.choice(n, 2, replace=False)
            xs[i], xs[j] = xs[j], xs[i]
            ys[i], ys[j] = ys[j], ys[i]
            rs[i], rs[j] = rs[j], rs[i]
        elif strategy == 'shake':
            # Shake all circles a little
            step = 0.01 + 0.03 * np.random.rand()
            xs += np.random.randn(n) * step
            ys += np.random.randn(n) * step
        elif strategy == 'grow':
            # Try to grow radii slightly
            rs *= (1 + np.random.rand(n) * 0.05)

        # Clip to bounds
        xs = np.clip(xs, 0.01, 0.99)
        ys = np.clip(ys, 0.01, 0.99)
        rs = np.clip(rs, 0.001, 0.5)

        trial = pack_to_vec(np.column_stack([xs, ys]), rs)

        # Local optimization
        trial = progressive_penalty_optimize(
            np.column_stack([trial[0::3], trial[1::3]]), trial[2::3], n, max_iter=150
        )
        trial, _ = slsqp_polish(trial, n, max_iter=300)
        trial = repair_solution(trial, n)

        valid, metric, max_viol = validate_solution(trial, n)

        if valid and metric > best_metric:
            best_vec = trial.copy()
            best_metric = metric
            current_vec = trial.copy()
            current_metric = metric
            print(f"  BH hop {hop}: NEW BEST metric={metric:.10f}")
        elif valid:
            # Metropolis criterion
            delta = metric - current_metric
            if delta > 0 or np.random.rand() < np.exp(delta / temperature):
                current_vec = trial.copy()
                current_metric = metric

    return best_vec, best_metric


def solve_n(n, num_starts=80, bh_hops=200, verbose=True):
    """Full solver for a given n."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Solving n={n}")
        print(f"{'='*60}")

    best_vec = None
    best_metric = -1
    start_time = time.time()

    # Phase 1: Multi-start
    if verbose:
        print(f"\nPhase 1: Multi-start ({num_starts} starts)")

    for s in range(num_starts):
        # Choose initialization
        init_type = s % 5
        try:
            if init_type == 0:
                pts, radii = hex_init(n, noise=0.01 * (s // 5))
            elif init_type == 1:
                pts, radii = grid_init(n, noise=0.01 * (s // 5))
            elif init_type == 2:
                pts, radii = ring_init(n, noise=0.01 * (s // 5))
            elif init_type == 3:
                pts, radii = random_init(n)
            else:
                pts, radii = hex_init(n, noise=0.03 + 0.01 * (s // 5))

            # Progressive penalty
            vec = progressive_penalty_optimize(pts, radii, n, max_iter=200)

            # SLSQP polish
            vec, _ = slsqp_polish(vec, n, max_iter=400)

            # Repair
            vec = repair_solution(vec, n)

            # Single circle refine
            vec = single_circle_refine(vec, n, iterations=2)
            vec = repair_solution(vec, n)

            valid, metric, max_viol = validate_solution(vec, n)

            if valid and metric > best_metric:
                best_metric = metric
                best_vec = vec.copy()
                if verbose and (s < 10 or s % 10 == 0 or metric > best_metric - 0.001):
                    print(f"  Start {s}: metric={metric:.10f} (NEW BEST)")
        except Exception as e:
            if verbose and s < 5:
                print(f"  Start {s}: FAILED ({e})")

    if verbose:
        print(f"Phase 1 best: {best_metric:.10f} (time: {time.time()-start_time:.1f}s)")

    if best_vec is None:
        print("ERROR: No valid solution found in Phase 1!")
        return None, -1

    # Phase 2: Basin-hopping on best
    if verbose:
        print(f"\nPhase 2: Basin-hopping ({bh_hops} hops)")
    bh_vec, bh_metric = basin_hopping_optimize(best_vec, n, n_hops=bh_hops)
    if bh_metric > best_metric:
        best_vec = bh_vec
        best_metric = bh_metric
        if verbose:
            print(f"Basin-hopping improved to: {best_metric:.10f}")

    # Phase 3: Fine refinement
    if verbose:
        print(f"\nPhase 3: Fine refinement")
    vec = fine_single_circle_refine(best_vec, n, iterations=5)
    vec = repair_solution(vec, n)
    valid, metric, _ = validate_solution(vec, n)
    if valid and metric > best_metric:
        best_vec = vec
        best_metric = metric
        if verbose:
            print(f"Fine refinement improved to: {best_metric:.10f}")

    # Phase 4: Final SLSQP polish
    if verbose:
        print(f"\nPhase 4: Final polish")
    vec, _ = slsqp_polish(best_vec, n, max_iter=1000)
    vec = repair_solution(vec, n)
    valid, metric, max_viol = validate_solution(vec, n)
    if valid and metric > best_metric:
        best_vec = vec
        best_metric = metric

    # Final fine refinement
    for _ in range(3):
        vec = fine_single_circle_refine(best_vec, n, iterations=5)
        vec = repair_solution(vec, n)
        valid, metric, _ = validate_solution(vec, n)
        if valid and metric > best_metric:
            best_vec = vec
            best_metric = metric

    elapsed = time.time() - start_time
    if verbose:
        print(f"\nFinal result for n={n}: metric={best_metric:.10f} (time: {elapsed:.1f}s)")

    return best_vec, best_metric


def save_solution(vec, n, filepath):
    """Save solution to JSON."""
    xs, ys, rs = vec_to_pack(vec)
    circles = [[float(xs[i]), float(ys[i]), float(rs[i])] for i in range(n)]
    data = {"circles": circles, "n": n, "metric": float(np.sum(rs))}
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved to {filepath}")


def main():
    np.random.seed(42)

    targets = {
        24: 2.530,
        25: 2.587,
        27: 2.685,
        29: 2.790,
        31: 2.889,
    }

    # Order: 29, 31, 24, 25, 27
    order = [29, 31, 24, 25, 27]

    out_dir = os.path.dirname(os.path.abspath(__file__))
    results = {}

    for n in order:
        sota = targets[n]
        vec, metric = solve_n(n, num_starts=100, bh_hops=300)

        if vec is not None:
            filepath = os.path.join(out_dir, f"solution_n{n}.json")
            save_solution(vec, n, filepath)
            results[n] = metric
            print(f"\nn={n}: metric={metric:.10f}, SOTA={sota:.3f}, ratio={metric/sota:.4f}")
        else:
            print(f"\nn={n}: FAILED")
            results[n] = 0

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for n in order:
        sota = targets[n]
        m = results.get(n, 0)
        print(f"  n={n}: metric={m:.10f}  SOTA={sota:.3f}  ratio={m/sota:.4f}")


if __name__ == "__main__":
    main()
