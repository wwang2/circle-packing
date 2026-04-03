"""
Contact Graph Topology Search for Circle Packing n=26
Strategy: Analyze the contact graph of the best known solution,
then systematically explore topology variations to find new basins.
"""

import json
import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import sys
import os
import time

WORKDIR = os.path.dirname(os.path.abspath(__file__))

def load_solution(path):
    with open(path) as f:
        data = json.load(f)
    circles = np.array(data["circles"])
    return circles[:, 0], circles[:, 1], circles[:, 2]

def save_solution(x, y, r, path):
    circles = [[float(x[i]), float(y[i]), float(r[i])] for i in range(len(x))]
    with open(path, 'w') as f:
        json.dump({"circles": circles}, f, indent=2)

def compute_contacts(x, y, r, tol=1e-4):
    """Build contact graph: which circles touch each other or walls."""
    n = len(x)
    circle_contacts = []  # (i, j, gap)
    wall_contacts = []    # (i, wall_name, gap)

    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            gap = dist - (r[i] + r[j])
            if gap < tol:
                circle_contacts.append((i, j, gap))

        # Wall contacts
        gaps = {
            'left': x[i] - r[i],
            'right': (1 - x[i]) - r[i],
            'bottom': y[i] - r[i],
            'top': (1 - y[i]) - r[i],
        }
        for wall, gap in gaps.items():
            if gap < tol:
                wall_contacts.append((i, wall, gap))

    return circle_contacts, wall_contacts

def print_contacts(x, y, r, circle_contacts, wall_contacts):
    """Print contact graph analysis."""
    n = len(x)
    print(f"\n=== Contact Graph Analysis (n={n}) ===")
    print(f"Sum of radii: {sum(r):.10f}")
    print(f"\nCircle-circle contacts ({len(circle_contacts)}):")
    for i, j, gap in sorted(circle_contacts, key=lambda t: t[2]):
        print(f"  {i:2d}-{j:2d}: gap={gap:.2e}, r_i={r[i]:.6f}, r_j={r[j]:.6f}")

    print(f"\nWall contacts ({len(wall_contacts)}):")
    for i, wall, gap in sorted(wall_contacts, key=lambda t: t[2]):
        print(f"  {i:2d}-{wall:6s}: gap={gap:.2e}, r={r[i]:.6f}")

    # Degree of each circle
    degree = np.zeros(n, dtype=int)
    for i, j, _ in circle_contacts:
        degree[i] += 1
        degree[j] += 1
    for i, _, _ in wall_contacts:
        degree[i] += 1

    print(f"\nDegree distribution:")
    for i in range(n):
        print(f"  Circle {i:2d}: degree={degree[i]}, r={r[i]:.6f}, pos=({x[i]:.4f}, {y[i]:.4f})")

    return degree

def objective_and_constraints(params, n):
    """Returns negative sum of radii and constraint violations."""
    x = params[:n]
    y = params[n:2*n]
    r = params[2*n:3*n]
    return -np.sum(r)

def build_constraints(n):
    """Build scipy constraints for circle packing."""
    constraints = []

    # Containment: r <= x <= 1-r, r <= y <= 1-r
    # Equivalently: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[i] - p[2*n+i]})           # x_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[i] - p[2*n+i]})      # 1 - x_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[n+i] - p[2*n+i]})         # y_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[n+i] - p[2*n+i]})    # 1 - y_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[2*n+i] - 1e-6})           # r_i > 0

    # Non-overlap: dist(i,j) >= r_i + r_j
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: (
                    (p[i]-p[j])**2 + (p[n+i]-p[n+j])**2 - (p[2*n+i]+p[2*n+j])**2
                )
            })

    return constraints

def optimize_packing(x0, y0, r0, maxiter=5000):
    """Run SLSQP optimization from given initial positions."""
    n = len(x0)
    params0 = np.concatenate([x0, y0, r0])
    constraints = build_constraints(n)

    result = minimize(
        lambda p: -np.sum(p[2*n:3*n]),
        params0,
        method='SLSQP',
        constraints=constraints,
        options={'maxiter': maxiter, 'ftol': 1e-15, 'disp': False}
    )

    x = result.x[:n]
    y = result.x[n:2*n]
    r = result.x[2*n:3*n]

    return x, y, r, -result.fun, result.success

def is_feasible(x, y, r, tol=1e-10):
    """Check if a solution is feasible."""
    n = len(x)
    for i in range(n):
        if r[i] < -tol:
            return False
        if x[i] - r[i] < -tol or 1 - x[i] - r[i] < -tol:
            return False
        if y[i] - r[i] < -tol or 1 - y[i] - r[i] < -tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            dist2 = (x[i]-x[j])**2 + (y[i]-y[j])**2
            sum_r = r[i] + r[j]
            if dist2 < sum_r**2 - tol:
                return False
    return True

def perturb_swap(x, y, r, i, j):
    """Swap positions of circles i and j, keeping radii."""
    x2, y2, r2 = x.copy(), y.copy(), r.copy()
    x2[i], x2[j] = x[j], x[i]
    y2[i], y2[j] = y[j], y[i]
    # Keep radii where they are (don't swap) - this changes topology
    return x2, y2, r2

def perturb_displace(x, y, r, idx, dx, dy):
    """Displace circle idx by (dx, dy)."""
    x2, y2, r2 = x.copy(), y.copy(), r.copy()
    x2[idx] += dx
    y2[idx] += dy
    # Reduce radius to avoid overlap
    r2[idx] *= 0.8
    return x2, y2, r2

def perturb_split(x, y, r, idx):
    """Split circle idx into two smaller circles."""
    n = len(x)
    x2 = np.zeros(n + 1)
    y2 = np.zeros(n + 1)
    r2 = np.zeros(n + 1)

    x2[:n] = x
    y2[:n] = y
    r2[:n] = r

    # Split: two circles of 0.6*r at offset positions
    new_r = r[idx] * 0.6
    offset = r[idx] * 0.5

    r2[idx] = new_r
    x2[n] = x[idx] + offset
    y2[n] = y[idx]
    r2[n] = new_r

    x2[idx] = x[idx] - offset

    # Clamp to bounds
    for i in [idx, n]:
        x2[i] = np.clip(x2[i], r2[i] + 0.001, 1 - r2[i] - 0.001)
        y2[i] = np.clip(y2[i], r2[i] + 0.001, 1 - r2[i] - 0.001)

    return x2, y2, r2

def perturb_random_positions(n, seed=42):
    """Generate completely random initial positions."""
    rng = np.random.RandomState(seed)
    r = np.full(n, 0.05)
    x = rng.uniform(0.1, 0.9, n)
    y = rng.uniform(0.1, 0.9, n)
    return x, y, r

def generate_hex_init(n, seed=0):
    """Generate hexagonal close-packing initialization."""
    rng = np.random.RandomState(seed)
    # Estimate radius for hex packing
    r_est = 0.5 / (np.sqrt(n) + 1)

    positions = []
    row = 0
    y = r_est
    while y < 1 - r_est and len(positions) < n * 2:
        x = r_est + (row % 2) * r_est
        while x < 1 - r_est and len(positions) < n * 2:
            positions.append((x, y))
            x += 2 * r_est
        y += r_est * np.sqrt(3)
        row += 1

    if len(positions) < n:
        # Fill randomly
        while len(positions) < n:
            positions.append((rng.uniform(0.15, 0.85), rng.uniform(0.15, 0.85)))

    # Take n positions, add some noise
    positions = positions[:n]
    x = np.array([p[0] for p in positions]) + rng.uniform(-0.01, 0.01, n)
    y = np.array([p[1] for p in positions]) + rng.uniform(-0.01, 0.01, n)
    r = np.full(n, r_est * 0.8)

    x = np.clip(x, r + 0.001, 1 - r - 0.001)
    y = np.clip(y, r + 0.001, 1 - r - 0.001)

    return x, y, r

def generate_mixed_size_init(n, n_large=4, seed=0):
    """Few large circles + many small circles."""
    rng = np.random.RandomState(seed)

    # Place large circles at strategic positions
    large_r = 0.15
    small_r = 0.05

    x = np.zeros(n)
    y = np.zeros(n)
    r = np.zeros(n)

    # Large circles in a grid
    large_positions = [
        (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75),
        (0.5, 0.5), (0.5, 0.15), (0.5, 0.85), (0.15, 0.5), (0.85, 0.5)
    ]

    for i in range(min(n_large, n)):
        x[i], y[i] = large_positions[i % len(large_positions)]
        r[i] = large_r

    # Small circles in remaining space
    for i in range(n_large, n):
        x[i] = rng.uniform(0.1, 0.9)
        y[i] = rng.uniform(0.1, 0.9)
        r[i] = small_r

    return x, y, r

def systematic_topology_search(parent_path, output_dir):
    """Main topology search routine."""
    x0, y0, r0 = load_solution(parent_path)
    n = len(x0)
    parent_metric = np.sum(r0)

    print(f"Parent solution: n={n}, sum_r={parent_metric:.10f}")

    # Analyze contact graph
    cc, wc = compute_contacts(x0, y0, r0)
    degree = print_contacts(x0, y0, r0, cc, wc)

    best_metric = parent_metric
    best_x, best_y, best_r = x0.copy(), y0.copy(), r0.copy()
    results = []

    # =============================================
    # Strategy 1: Position swaps
    # =============================================
    print("\n\n=== Strategy 1: Position Swaps ===")
    # Sort circles by radius to identify size classes
    r_order = np.argsort(r0)[::-1]

    # Try swapping circles of different sizes
    swap_count = 0
    for i in range(n):
        for j in range(i+1, n):
            if abs(r0[i] - r0[j]) < 0.01:
                continue  # Skip similar-size circles

            x2, y2, r2 = perturb_swap(x0, y0, r0, i, j)
            x2, y2, r2, metric, success = optimize_packing(x2, y2, r2, maxiter=3000)

            if success and is_feasible(x2, y2, r2) and metric > best_metric + 1e-8:
                print(f"  IMPROVED: swap({i},{j}) -> {metric:.10f} (delta={metric-parent_metric:.2e})")
                best_metric = metric
                best_x, best_y, best_r = x2.copy(), y2.copy(), r2.copy()
                results.append(('swap', i, j, metric))

            swap_count += 1
            if swap_count % 50 == 0:
                print(f"  ... {swap_count} swaps tried, best={best_metric:.10f}")

    print(f"  Total swaps: {swap_count}, best after swaps: {best_metric:.10f}")

    # =============================================
    # Strategy 2: Single-circle displacement
    # =============================================
    print("\n\n=== Strategy 2: Single-Circle Displacement ===")
    rng = np.random.RandomState(42)

    for trial in range(5):
        for i in range(n):
            for _ in range(8):
                angle = rng.uniform(0, 2*np.pi)
                dist = rng.uniform(0.02, 0.15)
                dx = dist * np.cos(angle)
                dy = dist * np.sin(angle)

                x2, y2, r2 = perturb_displace(x0, y0, r0, i, dx, dy)
                x2 = np.clip(x2, r2 + 0.001, 1 - r2 - 0.001)
                y2 = np.clip(y2, r2 + 0.001, 1 - r2 - 0.001)

                x2, y2, r2, metric, success = optimize_packing(x2, y2, r2, maxiter=3000)

                if success and is_feasible(x2, y2, r2) and metric > best_metric + 1e-8:
                    print(f"  IMPROVED: displace({i}, angle={angle:.2f}, dist={dist:.3f}) -> {metric:.10f}")
                    best_metric = metric
                    best_x, best_y, best_r = x2.copy(), y2.copy(), r2.copy()
                    results.append(('displace', i, metric))

        print(f"  Trial {trial+1}/5 done, best={best_metric:.10f}")

    # =============================================
    # Strategy 3: Fresh topology initializations
    # =============================================
    print("\n\n=== Strategy 3: Fresh Topology Initializations ===")

    for seed in range(50):
        # Random positions
        x2, y2, r2 = perturb_random_positions(n, seed=seed)
        x2, y2, r2, metric, success = optimize_packing(x2, y2, r2, maxiter=5000)

        if success and is_feasible(x2, y2, r2) and metric > best_metric + 1e-8:
            print(f"  IMPROVED: random(seed={seed}) -> {metric:.10f}")
            best_metric = metric
            best_x, best_y, best_r = x2.copy(), y2.copy(), r2.copy()
            results.append(('random', seed, metric))

        if seed % 10 == 0:
            print(f"  Random seed {seed}, this={metric:.6f}, best={best_metric:.10f}")

    # Hex initializations
    for seed in range(20):
        x2, y2, r2 = generate_hex_init(n, seed=seed)
        x2, y2, r2, metric, success = optimize_packing(x2, y2, r2, maxiter=5000)

        if success and is_feasible(x2, y2, r2) and metric > best_metric + 1e-8:
            print(f"  IMPROVED: hex(seed={seed}) -> {metric:.10f}")
            best_metric = metric
            best_x, best_y, best_r = x2.copy(), y2.copy(), r2.copy()
            results.append(('hex', seed, metric))

        if seed % 5 == 0:
            print(f"  Hex seed {seed}, this={metric:.6f}, best={best_metric:.10f}")

    # Mixed size initializations
    for n_large in [2, 3, 4, 5, 6, 7, 8]:
        for seed in range(10):
            x2, y2, r2 = generate_mixed_size_init(n, n_large=n_large, seed=seed)
            x2, y2, r2, metric, success = optimize_packing(x2, y2, r2, maxiter=5000)

            if success and is_feasible(x2, y2, r2) and metric > best_metric + 1e-8:
                print(f"  IMPROVED: mixed(n_large={n_large}, seed={seed}) -> {metric:.10f}")
                best_metric = metric
                best_x, best_y, best_r = x2.copy(), y2.copy(), r2.copy()
                results.append(('mixed', n_large, seed, metric))

        print(f"  Mixed n_large={n_large} done, best={best_metric:.10f}")

    # =============================================
    # Strategy 4: Perturb from parent with noise
    # =============================================
    print("\n\n=== Strategy 4: Parent + Noise (Basin Hopping) ===")

    for scale in [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]:
        for seed in range(20):
            rng = np.random.RandomState(seed + 1000)
            x2 = x0 + rng.normal(0, scale, n)
            y2 = y0 + rng.normal(0, scale, n)
            r2 = r0 * (1 + rng.normal(0, scale*0.5, n))
            r2 = np.maximum(r2, 0.01)
            x2 = np.clip(x2, r2 + 0.001, 1 - r2 - 0.001)
            y2 = np.clip(y2, r2 + 0.001, 1 - r2 - 0.001)

            x2, y2, r2, metric, success = optimize_packing(x2, y2, r2, maxiter=5000)

            if success and is_feasible(x2, y2, r2) and metric > best_metric + 1e-8:
                print(f"  IMPROVED: noise(scale={scale}, seed={seed}) -> {metric:.10f}")
                best_metric = metric
                best_x, best_y, best_r = x2.copy(), y2.copy(), r2.copy()
                results.append(('noise', scale, seed, metric))

        print(f"  Scale {scale} done, best={best_metric:.10f}")

    # Save best result
    save_solution(best_x, best_y, best_r, os.path.join(output_dir, 'solution_n26.json'))

    print(f"\n\n=== FINAL RESULT ===")
    print(f"Parent metric: {parent_metric:.10f}")
    print(f"Best metric:   {best_metric:.10f}")
    print(f"Improvement:   {best_metric - parent_metric:.2e}")
    print(f"Improvements found: {len(results)}")
    for r in results:
        print(f"  {r}")

    return best_x, best_y, best_r, best_metric

if __name__ == '__main__':
    parent_path = os.path.join(WORKDIR, '..', 'nlp-001', 'solution_n26.json')
    systematic_topology_search(parent_path, WORKDIR)
