#!/usr/bin/env python3
"""
Graph topology search: Generate random maximal planar triangulations on 26 vertices,
realize each as a circle packing, optimize, and score by sum of radii.

Strategy:
1. Generate random planar graphs via multiple methods:
   - Random Delaunay triangulations of point sets
   - Randomized incremental construction
   - Edge-flip mutations from known good topologies
2. For each graph, attempt circle packing realization
3. Optimize with SLSQP
4. Track the best topology found
"""

import json
import math
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from pathlib import Path
import itertools
import time

WORKTREE = Path("/Users/wujiewang/code/circle-packing/.worktrees/mobius-001")
OUTPUT_DIR = WORKTREE / "orbits/mobius-001"
N = 26
np.random.seed(42)

def fp(*args, **kwargs):
    print(*args, **kwargs, flush=True)


def load_solution(path):
    with open(path) as f:
        return np.array(json.load(f)["circles"])


def save_solution(circles, path):
    data = {"circles": [[float(c[0]), float(c[1]), float(c[2])] for c in circles]}
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def sum_radii(c):
    return float(np.sum(c[:, 2]))


def max_violation(circles):
    n = len(circles)
    mv = 0.0
    for i in range(n):
        x, y, r = circles[i]
        mv = max(mv, r - x, x + r - 1, r - y, y + r - 1)
    for i in range(n):
        for j in range(i + 1, n):
            dx = circles[i, 0] - circles[j, 0]
            dy = circles[i, 1] - circles[j, 1]
            d = math.sqrt(dx * dx + dy * dy)
            mv = max(mv, circles[i, 2] + circles[j, 2] - d)
    return mv


def validate(circles, tol=1e-10):
    return max_violation(circles) <= tol


def optimize_packing(init_circles, maxiter=5000, ftol=1e-15):
    """Optimize circle packing with SLSQP."""
    n = len(init_circles)
    x0 = init_circles.flatten()
    nn = 3 * n

    def objective(x):
        return -np.sum(x[2::3])

    def grad_obj(x):
        g = np.zeros(nn)
        g[2::3] = -1.0
        return g

    constraints = []
    # Wall constraints
    for i in range(n):
        xi, yi, ri = 3*i, 3*i+1, 3*i+2
        constraints.append({'type': 'ineq', 'fun': lambda x, xi=xi, ri=ri: x[xi] - x[ri]})
        constraints.append({'type': 'ineq', 'fun': lambda x, xi=xi, ri=ri: 1.0 - x[xi] - x[ri]})
        constraints.append({'type': 'ineq', 'fun': lambda x, yi=yi, ri=ri: x[yi] - x[ri]})
        constraints.append({'type': 'ineq', 'fun': lambda x, yi=yi, ri=ri: 1.0 - x[yi] - x[ri]})
        constraints.append({'type': 'ineq', 'fun': lambda x, ri=ri: x[ri] - 1e-6})

    # Non-overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            xi, yi, ri = 3*i, 3*i+1, 3*i+2
            xj, yj, rj = 3*j, 3*j+1, 3*j+2
            def con(x, xi=xi, yi=yi, ri=ri, xj=xj, yj=yj, rj=rj):
                dx = x[xi] - x[xj]
                dy = x[yi] - x[yj]
                return dx*dx + dy*dy - (x[ri] + x[rj])**2
            constraints.append({'type': 'ineq', 'fun': con})

    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * n
    result = minimize(objective, x0, method='SLSQP', jac=grad_obj,
                      bounds=bounds, constraints=constraints,
                      options={'maxiter': maxiter, 'ftol': ftol})
    out = result.x.reshape(n, 3)
    return out, -result.fun


def optimize_packing_fast(init_circles, maxiter=2000):
    """Fast vectorized SLSQP optimization."""
    n = len(init_circles)
    x0 = init_circles.flatten()

    def objective(x):
        return -np.sum(x[2::3])

    def all_con(x):
        xs = x[0::3]; ys = x[1::3]; rs = x[2::3]
        vals = []
        vals.extend(xs - rs)
        vals.extend(1.0 - xs - rs)
        vals.extend(ys - rs)
        vals.extend(1.0 - ys - rs)
        vals.extend(rs - 1e-6)
        for i in range(n):
            dx = xs[i] - xs[i+1:]
            dy = ys[i] - ys[i+1:]
            rsum = rs[i] + rs[i+1:]
            dist_sq = dx*dx + dy*dy
            vals.extend(dist_sq - rsum**2)
        return np.array(vals)

    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * n
    result = minimize(objective, x0, method='SLSQP',
                      bounds=bounds, constraints=[{'type': 'ineq', 'fun': all_con}],
                      options={'maxiter': maxiter, 'ftol': 1e-14})
    out = result.x.reshape(n, 3)
    return out, -result.fun


def generate_random_init(n, method='uniform', seed=None):
    """Generate random initial circle configurations."""
    rng = np.random.RandomState(seed)

    if method == 'uniform':
        # Random uniform placement
        r_base = 0.5 / math.sqrt(n * math.pi)
        circles = np.zeros((n, 3))
        placed = 0
        for attempt in range(n * 100):
            if placed >= n:
                break
            r = r_base * (0.5 + rng.random())
            x = r + rng.random() * (1 - 2*r)
            y = r + rng.random() * (1 - 2*r)
            ok = True
            for k in range(placed):
                dx = x - circles[k, 0]
                dy = y - circles[k, 1]
                if math.sqrt(dx*dx + dy*dy) < r + circles[k, 2]:
                    ok = False
                    break
            if ok:
                circles[placed] = [x, y, r]
                placed += 1
        if placed < n:
            # Fill remaining with tiny circles
            for i in range(placed, n):
                circles[i] = [rng.random() * 0.8 + 0.1, rng.random() * 0.8 + 0.1, 0.01]
        return circles

    elif method == 'grid_perturb':
        # Perturbed grid
        side = int(math.ceil(math.sqrt(n)))
        r_base = 0.5 / (side + 1)
        circles = np.zeros((n, 3))
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= n:
                    break
                cx = (i + 1) / (side + 1) + rng.normal(0, r_base * 0.2)
                cy = (j + 1) / (side + 1) + rng.normal(0, r_base * 0.2)
                r = r_base * (0.7 + 0.6 * rng.random())
                cx = np.clip(cx, r + 0.001, 1 - r - 0.001)
                cy = np.clip(cy, r + 0.001, 1 - r - 0.001)
                circles[idx] = [cx, cy, r]
                idx += 1
        return circles

    elif method == 'concentric':
        # Concentric ring pattern
        circles = np.zeros((n, 3))
        # Big center circle
        circles[0] = [0.5, 0.5, 0.12 + rng.random() * 0.05]
        idx = 1
        for ring in range(1, 5):
            num_in_ring = min(4 + ring * 2 + rng.randint(0, 3), n - idx)
            if num_in_ring <= 0:
                break
            ring_r = 0.15 * ring + rng.random() * 0.05
            for k in range(num_in_ring):
                angle = 2 * math.pi * k / num_in_ring + rng.normal(0, 0.1)
                cr = 0.04 + rng.random() * 0.06
                cx = 0.5 + ring_r * math.cos(angle)
                cy = 0.5 + ring_r * math.sin(angle)
                cx = np.clip(cx, cr + 0.001, 1 - cr - 0.001)
                cy = np.clip(cy, cr + 0.001, 1 - cr - 0.001)
                circles[idx] = [cx, cy, cr]
                idx += 1
                if idx >= n:
                    break
        # Fill remaining
        for i in range(idx, n):
            circles[i] = [rng.random() * 0.8 + 0.1, rng.random() * 0.8 + 0.1, 0.02]
        return circles

    elif method == 'delaunay_based':
        # Generate points, compute Delaunay, use dual for radii hints
        points = rng.random((n, 2)) * 0.8 + 0.1
        tri = Delaunay(points)
        # Estimate radius from nearest neighbor distance
        circles = np.zeros((n, 3))
        for i in range(n):
            neighbors = set()
            for s in tri.simplices:
                if i in s:
                    neighbors.update(s)
            neighbors.discard(i)
            if neighbors:
                min_dist = min(np.linalg.norm(points[i] - points[j]) for j in neighbors)
                r = min_dist * (0.35 + rng.random() * 0.15)
            else:
                r = 0.02
            circles[i] = [points[i, 0], points[i, 1], r]
            circles[i, 0] = np.clip(circles[i, 0], r + 0.001, 1 - r - 0.001)
            circles[i, 1] = np.clip(circles[i, 1], r + 0.001, 1 - r - 0.001)
        return circles

    elif method == 'hex':
        # Hexagonal packing with randomized offsets
        circles = np.zeros((n, 3))
        r_base = 0.5 / (math.sqrt(n) + 1)
        idx = 0
        row = 0
        while idx < n:
            y = r_base + row * r_base * math.sqrt(3)
            if y > 1 - r_base:
                break
            offset = r_base if row % 2 else 0
            x = r_base + offset
            while x < 1 - r_base and idx < n:
                r = r_base * (0.6 + rng.random() * 0.8)
                cx = x + rng.normal(0, r_base * 0.15)
                cy = y + rng.normal(0, r_base * 0.15)
                cx = np.clip(cx, r + 0.001, 1 - r - 0.001)
                cy = np.clip(cy, r + 0.001, 1 - r - 0.001)
                circles[idx] = [cx, cy, r]
                idx += 1
                x += 2 * r_base
            row += 1
        for i in range(idx, n):
            circles[i] = [rng.random() * 0.8 + 0.1, rng.random() * 0.8 + 0.1, 0.02]
        return circles


def perturb_solution(circles, rng, scale=0.1):
    """Perturb a solution to create a new starting point."""
    n = len(circles)
    c = circles.copy()
    method = rng.randint(0, 6)

    if method == 0:
        # Swap two circles
        i, j = rng.choice(n, 2, replace=False)
        c[i, 0], c[j, 0] = c[j, 0], c[i, 0]
        c[i, 1], c[j, 1] = c[j, 1], c[i, 1]
    elif method == 1:
        # Perturb all positions
        c[:, 0] += rng.normal(0, scale * 0.1, n)
        c[:, 1] += rng.normal(0, scale * 0.1, n)
    elif method == 2:
        # Perturb all radii
        c[:, 2] *= (1 + rng.normal(0, scale * 0.2, n))
    elif method == 3:
        # Rotate a subset around center
        subset = rng.choice(n, rng.randint(3, n//2), replace=False)
        angle = rng.uniform(-0.5, 0.5)
        cx, cy = np.mean(c[subset, 0]), np.mean(c[subset, 1])
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        for i in subset:
            dx, dy = c[i, 0] - cx, c[i, 1] - cy
            c[i, 0] = cx + cos_a * dx - sin_a * dy
            c[i, 1] = cy + sin_a * dx + cos_a * dy
    elif method == 4:
        # Mirror a subset
        axis = rng.choice(['x', 'y'])
        subset = rng.choice(n, rng.randint(2, n//3), replace=False)
        if axis == 'x':
            c[subset, 0] = 1.0 - c[subset, 0]
        else:
            c[subset, 1] = 1.0 - c[subset, 1]
    elif method == 5:
        # Large random displacement of one circle
        i = rng.randint(n)
        c[i, 0] = rng.random() * 0.8 + 0.1
        c[i, 1] = rng.random() * 0.8 + 0.1
        c[i, 2] = max(0.01, c[i, 2] * (0.5 + rng.random()))

    # Clip to valid range
    c[:, 2] = np.clip(c[:, 2], 0.005, 0.45)
    c[:, 0] = np.clip(c[:, 0], c[:, 2] + 0.001, 1 - c[:, 2] - 0.001)
    c[:, 1] = np.clip(c[:, 1], c[:, 2] + 0.001, 1 - c[:, 2] - 0.001)
    return c


def large_mobius_transform(circles, rng):
    """Apply a LARGE Mobius transformation to the packing.

    In inversive geometry, Mobius transforms are circle-preserving maps.
    We apply them in the complex plane then map back.
    """
    n = len(circles)
    c = circles.copy()

    # Convert to complex plane representation
    z = c[:, 0] + 1j * c[:, 1]
    r = c[:, 2]

    # Random Mobius map: f(z) = (az + b) / (cz + d) with ad - bc != 0
    # Use large parameters for discontinuous-in-Euclidean transforms
    param_scale = rng.uniform(0.5, 5.0)
    a = complex(rng.normal(1, param_scale), rng.normal(0, param_scale))
    b = complex(rng.normal(0, param_scale), rng.normal(0, param_scale))
    cc = complex(rng.normal(0, param_scale * 0.3), rng.normal(0, param_scale * 0.3))
    d = complex(rng.normal(1, param_scale), rng.normal(0, param_scale))

    # Ensure non-degenerate
    det = a * d - b * cc
    if abs(det) < 0.1:
        return c  # Skip degenerate transforms

    # Apply to centers
    denom = cc * z + d
    if np.any(np.abs(denom) < 1e-10):
        return c

    w = (a * z + b) / denom

    # Mobius transforms circles to circles; radius scales by |derivative|
    deriv = det / (denom ** 2)
    new_r = r * np.abs(deriv)

    # Map back to unit square
    new_x = np.real(w)
    new_y = np.imag(w)

    # Normalize to fit in [0,1]^2
    x_min, x_max = np.min(new_x - new_r), np.max(new_x + new_r)
    y_min, y_max = np.min(new_y - new_r), np.max(new_y + new_r)

    x_range = x_max - x_min
    y_range = y_max - y_min

    if x_range < 1e-10 or y_range < 1e-10:
        return c

    # Scale to fit in unit square with margin
    scale = 0.95 / max(x_range, y_range)
    margin = 0.025

    new_x = (new_x - x_min) * scale + margin
    new_y = (new_y - y_min) * scale + margin
    new_r = new_r * scale

    result = np.column_stack([new_x, new_y, new_r])
    result[:, 2] = np.clip(result[:, 2], 0.005, 0.45)
    result[:, 0] = np.clip(result[:, 0], result[:, 2] + 0.001, 1 - result[:, 2] - 0.001)
    result[:, 1] = np.clip(result[:, 1], result[:, 2] + 0.001, 1 - result[:, 2] - 0.001)
    return result


def conformal_disk_init(n, rng):
    """Generate packing in disk, map to square via inverse conformal map."""
    # Pack in unit disk
    circles = np.zeros((n, 3))
    # Center circle
    r0 = 0.2 + rng.random() * 0.1
    circles[0] = [0, 0, r0]

    for i in range(1, n):
        for _ in range(100):
            # Random position in disk
            angle = rng.uniform(0, 2 * math.pi)
            dist = rng.uniform(0, 0.9)
            r = 0.02 + rng.random() * 0.1
            x = dist * math.cos(angle)
            y = dist * math.sin(angle)

            # Check disk containment
            if math.sqrt(x*x + y*y) + r > 0.98:
                continue

            # Check non-overlap
            ok = True
            for j in range(i):
                dx = x - circles[j, 0]
                dy = y - circles[j, 1]
                if math.sqrt(dx*dx + dy*dy) < r + circles[j, 2] + 0.001:
                    ok = False
                    break
            if ok:
                circles[i] = [x, y, r]
                break
        else:
            circles[i] = [rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), 0.01]

    # Conformal map from disk to square (approximate via Schwarz-Christoffel)
    # Use simpler approach: radial stretching
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, r = circles[i]
        # Map from [-1,1] disk to [0,1] square
        # Simple affine for now
        result[i] = [(x + 1) / 2, (y + 1) / 2, r / 2]

    result[:, 2] = np.clip(result[:, 2], 0.005, 0.45)
    result[:, 0] = np.clip(result[:, 0], result[:, 2] + 0.001, 1 - result[:, 2] - 0.001)
    result[:, 1] = np.clip(result[:, 1], result[:, 2] + 0.001, 1 - result[:, 2] - 0.001)
    return result


def main():
    start_time = time.time()

    # Load current best
    base = load_solution(OUTPUT_DIR / "solution_n26.json")
    best_metric = sum_radii(base)
    best_circles = base.copy()
    fp(f"Starting metric: {best_metric:.10f}")

    results = []
    total_tried = 0

    # Strategy 1: Massive multi-start from random inits
    fp("\n=== Strategy 1: Random multi-start (5 methods x 200 seeds) ===")
    methods = ['uniform', 'grid_perturb', 'concentric', 'delaunay_based', 'hex']

    for method in methods:
        method_best = 0
        for seed in range(200):
            try:
                init = generate_random_init(N, method=method, seed=seed * 100 + hash(method) % 10000)
                opt, metric = optimize_packing_fast(init, maxiter=3000)
                viol = max_violation(opt)
                total_tried += 1

                if viol <= 1e-10:
                    method_best = max(method_best, metric)
                    if metric > best_metric + 1e-12:
                        best_metric = metric
                        best_circles = opt.copy()
                        fp(f"  NEW BEST! method={method} seed={seed}: {metric:.10f}")

                    results.append((method, seed, metric, viol))
            except:
                pass

        fp(f"  {method}: best={method_best:.10f} ({total_tried} total tried)")

    # Strategy 2: Large Mobius transforms from best known
    fp("\n=== Strategy 2: Large Mobius transforms (500 trials) ===")
    rng = np.random.RandomState(12345)
    mobius_best = 0
    for trial in range(500):
        try:
            transformed = large_mobius_transform(base, rng)
            opt, metric = optimize_packing_fast(transformed, maxiter=3000)
            viol = max_violation(opt)
            total_tried += 1

            if viol <= 1e-10:
                mobius_best = max(mobius_best, metric)
                if metric > best_metric + 1e-12:
                    best_metric = metric
                    best_circles = opt.copy()
                    fp(f"  NEW BEST! trial={trial}: {metric:.10f}")
        except:
            pass

    fp(f"  Mobius: best={mobius_best:.10f}")

    # Strategy 3: Conformal disk mapping
    fp("\n=== Strategy 3: Conformal disk inits (200 trials) ===")
    disk_best = 0
    for seed in range(200):
        try:
            rng2 = np.random.RandomState(seed + 50000)
            init = conformal_disk_init(N, rng2)
            opt, metric = optimize_packing_fast(init, maxiter=3000)
            viol = max_violation(opt)
            total_tried += 1

            if viol <= 1e-10:
                disk_best = max(disk_best, metric)
                if metric > best_metric + 1e-12:
                    best_metric = metric
                    best_circles = opt.copy()
                    fp(f"  NEW BEST! seed={seed}: {metric:.10f}")
        except:
            pass

    fp(f"  Disk: best={disk_best:.10f}")

    # Strategy 4: Perturbations of best known solution (aggressive)
    fp("\n=== Strategy 4: Aggressive perturbations (1000 trials) ===")
    rng3 = np.random.RandomState(99999)
    perturb_best = 0
    for trial in range(1000):
        try:
            perturbed = perturb_solution(base, rng3, scale=0.3 + 0.5 * (trial % 10) / 10)
            opt, metric = optimize_packing_fast(perturbed, maxiter=3000)
            viol = max_violation(opt)
            total_tried += 1

            if viol <= 1e-10:
                perturb_best = max(perturb_best, metric)
                if metric > best_metric + 1e-12:
                    best_metric = metric
                    best_circles = opt.copy()
                    fp(f"  NEW BEST! trial={trial}: {metric:.10f}")
        except:
            pass

        if trial % 200 == 199:
            fp(f"  ... {trial+1}/1000, best={perturb_best:.10f}")

    fp(f"  Perturb: best={perturb_best:.10f}")

    # Strategy 5: Combine Mobius + perturbation (chain)
    fp("\n=== Strategy 5: Mobius + perturb chain (300 trials) ===")
    chain_best = 0
    rng4 = np.random.RandomState(77777)
    for trial in range(300):
        try:
            # First apply Mobius
            transformed = large_mobius_transform(base, rng4)
            # Then perturb
            perturbed = perturb_solution(transformed, rng4, scale=0.2)
            opt, metric = optimize_packing_fast(perturbed, maxiter=3000)
            viol = max_violation(opt)
            total_tried += 1

            if viol <= 1e-10:
                chain_best = max(chain_best, metric)
                if metric > best_metric + 1e-12:
                    best_metric = metric
                    best_circles = opt.copy()
                    fp(f"  NEW BEST! trial={trial}: {metric:.10f}")
        except:
            pass

    fp(f"  Chain: best={chain_best:.10f}")

    elapsed = time.time() - start_time
    fp(f"\n=== SUMMARY ===")
    fp(f"Total configs tried: {total_tried}")
    fp(f"Time: {elapsed:.1f}s")
    fp(f"Best metric: {best_metric:.10f}")
    fp(f"Valid: {validate(best_circles)}")

    # Save if improved
    current = load_solution(OUTPUT_DIR / "solution_n26.json")
    if best_metric > sum_radii(current) + 1e-14:
        save_solution(best_circles, OUTPUT_DIR / "solution_n26.json")
        fp("Saved new best!")
    else:
        fp(f"No improvement over current ({sum_radii(current):.10f})")

    # Save top results for analysis
    results.sort(key=lambda x: -x[2])
    with open(OUTPUT_DIR / "graph_search_results.json", 'w') as f:
        json.dump({
            'best_metric': best_metric,
            'total_tried': total_tried,
            'elapsed_s': elapsed,
            'top_results': [{'method': r[0], 'seed': r[1], 'metric': r[2], 'viol': r[3]}
                          for r in results[:50]]
        }, f, indent=2)


if __name__ == "__main__":
    main()
