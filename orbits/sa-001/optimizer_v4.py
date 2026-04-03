"""
V4: Multi-stage optimization inspired by OpenEvolve approach.
Key insights from SOTA:
1. Multiple specialized initialization patterns
2. Three-stage optimization: positions -> radii -> joint
3. Penalty-based approach with gradual hardening
4. Many more random restarts with best pattern selection
"""

import json
import math
import random
import sys
import time
import numpy as np
from scipy.optimize import minimize
from pathlib import Path


def sum_radii(circles):
    return sum(c[2] for c in circles)


def is_valid(circles, tol=1e-10):
    n = len(circles)
    for x, y, r in circles:
        if r <= 0 or x - r < -tol or x + r > 1 + tol or y - r < -tol or y + r > 1 + tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist_sq = dx*dx + dy*dy
            min_dist = circles[i][2] + circles[j][2]
            if dist_sq < min_dist * min_dist - tol:
                return False
    return True


# ============================================================
# Initialization patterns from SOTA approaches
# ============================================================

def pattern_corners_edges_center(n, seed):
    """4 corners + 4 edge midpoints + center + rings."""
    rng = random.Random(seed)
    circles = []

    # 4 corner circles
    r_corner = 0.118 + rng.uniform(-0.01, 0.01)
    for x, y in [(r_corner, r_corner), (1-r_corner, r_corner),
                 (r_corner, 1-r_corner), (1-r_corner, 1-r_corner)]:
        circles.append([x, y, r_corner])

    # 4 edge midpoint circles
    r_edge = 0.103 + rng.uniform(-0.01, 0.01)
    for x, y in [(0.5, r_edge), (0.5, 1-r_edge),
                 (r_edge, 0.5), (1-r_edge, 0.5)]:
        circles.append([x, y, r_edge])

    # Center circle
    r_center = 0.124 + rng.uniform(-0.01, 0.01)
    circles.append([0.5, 0.5, r_center])

    # Inner ring
    r_inner = 0.098 + rng.uniform(-0.01, 0.01)
    ring_r = 0.28 + rng.uniform(-0.02, 0.02)
    for i in range(8):
        angle = 2 * math.pi * i / 8 + rng.uniform(-0.1, 0.1)
        x = 0.5 + ring_r * math.cos(angle)
        y = 0.5 + ring_r * math.sin(angle)
        circles.append([max(r_inner, min(1-r_inner, x)),
                       max(r_inner, min(1-r_inner, y)), r_inner])
        if len(circles) >= n:
            break

    # Fill remaining with outer ring
    while len(circles) < n:
        r_fill = 0.075 + rng.uniform(-0.01, 0.01)
        angle = rng.uniform(0, 2 * math.pi)
        ring_r2 = 0.4 + rng.uniform(-0.03, 0.03)
        x = 0.5 + ring_r2 * math.cos(angle)
        y = 0.5 + ring_r2 * math.sin(angle)
        x = max(r_fill, min(1-r_fill, x))
        y = max(r_fill, min(1-r_fill, y))
        circles.append([x, y, r_fill])

    return circles[:n]


def pattern_hybrid(n, seed):
    """Large center + inner/middle/corner rings."""
    rng = random.Random(seed)
    circles = []

    # Large center
    r_c = 0.128 + rng.uniform(-0.015, 0.015)
    circles.append([0.5, 0.5, r_c])

    # 6 inner ring
    r_in = 0.103 + rng.uniform(-0.01, 0.01)
    ring_r = 0.25 + rng.uniform(-0.02, 0.02)
    for i in range(6):
        angle = 2 * math.pi * i / 6 + rng.uniform(-0.15, 0.15)
        x = 0.5 + ring_r * math.cos(angle)
        y = 0.5 + ring_r * math.sin(angle)
        circles.append([max(r_in, min(1-r_in, x)),
                       max(r_in, min(1-r_in, y)), r_in])

    # 8 middle ring
    r_mid = 0.093 + rng.uniform(-0.01, 0.01)
    ring_r2 = 0.38 + rng.uniform(-0.02, 0.02)
    for i in range(8):
        angle = 2 * math.pi * i / 8 + math.pi/8 + rng.uniform(-0.1, 0.1)
        x = 0.5 + ring_r2 * math.cos(angle)
        y = 0.5 + ring_r2 * math.sin(angle)
        circles.append([max(r_mid, min(1-r_mid, x)),
                       max(r_mid, min(1-r_mid, y)), r_mid])

    # 4 corner circles
    r_corn = 0.113 + rng.uniform(-0.01, 0.01)
    offset = r_corn + 0.01
    for x, y in [(offset, offset), (1-offset, offset),
                 (offset, 1-offset), (1-offset, 1-offset)]:
        circles.append([x, y, r_corn])

    # Fill remaining
    while len(circles) < n:
        r_fill = 0.073 + rng.uniform(-0.01, 0.01)
        x = r_fill + rng.random() * (1 - 2*r_fill)
        y = r_fill + rng.random() * (1 - 2*r_fill)
        circles.append([x, y, r_fill])

    return circles[:n]


def pattern_billiard(n, seed):
    """Edge-focused with central rings."""
    rng = random.Random(seed)
    circles = []

    # Edge circles (2 per edge, offset from center)
    r_e = 0.10 + rng.uniform(-0.01, 0.01)
    for pos in [0.3, 0.7]:
        circles.append([pos, r_e, r_e])  # bottom
        circles.append([pos, 1-r_e, r_e])  # top
        circles.append([r_e, pos, r_e])  # left
        circles.append([1-r_e, pos, r_e])  # right

    # 4 corners
    r_c = 0.085 + rng.uniform(-0.01, 0.01)
    off = r_c + 0.01
    for x, y in [(off, off), (1-off, off), (off, 1-off), (1-off, 1-off)]:
        circles.append([x, y, r_c])

    # Center region
    r_center = 0.11 + rng.uniform(-0.01, 0.01)
    circles.append([0.5, 0.5, r_center])

    # Fill
    while len(circles) < n:
        r_fill = 0.065 + rng.uniform(-0.01, 0.01)
        x = 0.15 + rng.random() * 0.7
        y = 0.15 + rng.random() * 0.7
        circles.append([x, y, r_fill])

    return circles[:n]


def pattern_greedy_largest_gap(n, seed):
    """Place circles one by one in the largest available gap."""
    rng = random.Random(seed)
    circles = []

    # Target radii distribution (sorted large to small)
    target_sum = 2.64
    radii = []
    for i in range(n):
        if i < 4:
            radii.append(0.14 + rng.uniform(-0.02, 0.02))
        elif i < 10:
            radii.append(0.11 + rng.uniform(-0.02, 0.02))
        elif i < 18:
            radii.append(0.085 + rng.uniform(-0.02, 0.02))
        else:
            radii.append(0.06 + rng.uniform(-0.02, 0.02))
    radii.sort(reverse=True)

    for target_r in radii:
        best_pos = None
        best_gap = -1
        n_tries = 3000 if len(circles) < 10 else 2000

        for _ in range(n_tries):
            x = 0.01 + rng.random() * 0.98
            y = 0.01 + rng.random() * 0.98
            gap = min(x, 1-x, y, 1-y)
            for cx, cy, cr in circles:
                d = math.sqrt((x-cx)**2 + (y-cy)**2)
                gap = min(gap, d - cr)

            if gap > best_gap:
                best_gap = gap
                best_pos = (x, y)

        if best_pos:
            r = min(target_r, max(0.005, best_gap - 1e-6))
            circles.append([best_pos[0], best_pos[1], r])

    return circles[:n]


def pattern_diagonal_symmetric(n, seed):
    """Exploit diagonal symmetry (y=x line)."""
    rng = random.Random(seed)
    circles = []
    half = n // 2

    # Generate half the circles
    for i in range(half):
        if i < 2:
            r = 0.14 + rng.uniform(-0.02, 0.02)
        elif i < 5:
            r = 0.11 + rng.uniform(-0.02, 0.02)
        elif i < 9:
            r = 0.085 + rng.uniform(-0.02, 0.02)
        else:
            r = 0.06 + rng.uniform(-0.02, 0.02)

        x = r + rng.random() * (1 - 2*r)
        y = r + rng.random() * (min(x, 1-r))  # Below diagonal
        y = max(r, min(1-r, y))
        circles.append([x, y, r])

    # Mirror across diagonal
    for i in range(half):
        mx, my, mr = circles[i][1], circles[i][0], circles[i][2]
        circles.append([mx, my, mr])

    # If odd, add one on diagonal
    if n % 2 == 1:
        r = 0.10 + rng.uniform(-0.02, 0.02)
        t = 0.3 + rng.random() * 0.4
        circles.append([t, t, r])

    return circles[:n]


def pattern_grid_optimized(n, seed):
    """Grid with location-dependent sizes."""
    rng = random.Random(seed)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))

    circles = []
    cell_w = 1.0 / cols
    cell_h = 1.0 / rows

    for i in range(rows):
        for j in range(cols):
            if len(circles) >= n:
                break
            cx = (j + 0.5) * cell_w
            cy = (i + 0.5) * cell_h

            # Larger radii near edges/corners
            dist_to_center = math.sqrt((cx-0.5)**2 + (cy-0.5)**2)
            r_base = min(cell_w, cell_h) * 0.45

            # Adjust based on position
            if dist_to_center > 0.3:
                r = r_base * (1.1 + rng.uniform(-0.05, 0.05))
            else:
                r = r_base * (0.95 + rng.uniform(-0.05, 0.05))

            cx += rng.gauss(0, cell_w * 0.08)
            cy += rng.gauss(0, cell_h * 0.08)
            cx = max(r, min(1-r, cx))
            cy = max(r, min(1-r, cy))
            circles.append([cx, cy, r])

    return circles[:n]


# ============================================================
# Three-stage optimization
# ============================================================

def optimize_positions(circles, maxiter=300, penalty_weight=300):
    """Stage 1: Optimize only positions with fixed-ish radii."""
    n = len(circles)
    # Flatten to position-only vector, keep radii separate
    positions = np.array([c[:2] for c in circles]).flatten()
    radii = np.array([c[2] for c in circles])

    def objective(pos):
        total_penalty = 0
        total_radius = 0

        for i in range(n):
            x, y = pos[2*i], pos[2*i+1]
            r = radii[i]
            total_radius += r

            # Wall penalties
            total_penalty += max(0, r - x)**2
            total_penalty += max(0, x + r - 1)**2
            total_penalty += max(0, r - y)**2
            total_penalty += max(0, y + r - 1)**2

        for i in range(n):
            for j in range(i+1, n):
                dx = pos[2*i] - pos[2*j]
                dy = pos[2*i+1] - pos[2*j+1]
                dist_sq = dx*dx + dy*dy
                min_dist = radii[i] + radii[j]
                overlap = max(0, min_dist**2 - dist_sq)
                total_penalty += overlap * penalty_weight

        return -total_radius + total_penalty

    bounds = [(0.0, 1.0)] * (2 * n)

    result = minimize(objective, positions, method='L-BFGS-B',
                     bounds=bounds,
                     options={'maxiter': maxiter, 'ftol': 1e-12})

    new_circles = []
    for i in range(n):
        new_circles.append([result.x[2*i], result.x[2*i+1], radii[i]])
    return new_circles


def optimize_radii(circles, maxiter=500):
    """Stage 2: Optimize only radii with fixed positions."""
    n = len(circles)
    positions = [(c[0], c[1]) for c in circles]
    r0 = np.array([c[2] for c in circles])

    def objective(r):
        return -np.sum(r)

    def grad_obj(r):
        return -np.ones(n)

    constraints = []

    for i in range(n):
        x, y = positions[i]
        # r_i <= x_i
        constraints.append({'type': 'ineq', 'fun': lambda r, i=i: x - r[i]})
        # r_i <= 1 - x_i
        constraints.append({'type': 'ineq', 'fun': lambda r, i=i: (1-positions[i][0]) - r[i]})
        # r_i <= y_i
        constraints.append({'type': 'ineq', 'fun': lambda r, i=i: positions[i][1] - r[i]})
        # r_i <= 1 - y_i
        constraints.append({'type': 'ineq', 'fun': lambda r, i=i: (1-positions[i][1]) - r[i]})

    for i in range(n):
        for j in range(i+1, n):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            dist = math.sqrt(dx*dx + dy*dy)
            constraints.append({'type': 'ineq',
                'fun': lambda r, i=i, j=j, d=dist: d - r[i] - r[j]})

    bounds = [(0.001, 0.5)] * n

    result = minimize(objective, r0, method='SLSQP',
                     jac=grad_obj, constraints=constraints,
                     bounds=bounds,
                     options={'maxiter': maxiter, 'ftol': 1e-15})

    new_circles = [[positions[i][0], positions[i][1], max(0.001, result.x[i])]
                   for i in range(n)]
    return new_circles


def optimize_joint(circles, maxiter=2000):
    """Stage 3: Joint optimization of positions and radii."""
    n = len(circles)
    x0 = np.array([v for c in circles for v in c], dtype=np.float64)

    def objective(x):
        return -sum(x[3*i+2] for i in range(n))

    def grad_objective(x):
        g = np.zeros_like(x)
        for i in range(n):
            g[3*i+2] = -1.0
        return g

    constraints = []
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i] - x[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[3*i] - x[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+1] - x[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[3*i+1] - x[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+2] - 1e-6})

    for i in range(n):
        for j in range(i+1, n):
            constraints.append({'type': 'ineq',
                'fun': lambda x, i=i, j=j: (
                    (x[3*i]-x[3*j])**2 + (x[3*i+1]-x[3*j+1])**2
                    - (x[3*i+2]+x[3*j+2])**2
                )})

    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * n

    result = minimize(objective, x0, method='SLSQP',
                     jac=grad_objective, constraints=constraints,
                     bounds=bounds,
                     options={'maxiter': maxiter, 'ftol': 1e-15})

    polished = [[result.x[3*i], result.x[3*i+1], result.x[3*i+2]] for i in range(n)]
    return polished


def grow_radii(circles):
    n = len(circles)
    circles = [list(c) for c in circles]
    for _ in range(200):
        changed = False
        for i in range(n):
            x, y, r = circles[i]
            max_r = min(x, 1 - x, y, 1 - y)
            for j in range(n):
                if j == i:
                    continue
                dx = x - circles[j][0]
                dy = y - circles[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                max_r = min(max_r, dist - circles[j][2])
            new_r = max(0.001, max_r - 1e-12)
            if abs(new_r - r) > 1e-12:
                circles[i][2] = new_r
                changed = True
        if not changed:
            break
    return circles


def push_apart(circles, iterations=200):
    n = len(circles)
    circles = [list(c) for c in circles]
    for _ in range(iterations):
        moved = False
        for i in range(n):
            for j in range(i+1, n):
                dx = circles[i][0] - circles[j][0]
                dy = circles[i][1] - circles[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                min_dist = circles[i][2] + circles[j][2]
                if dist < min_dist - 1e-10:
                    if dist < 1e-12:
                        dx = random.random() - 0.5
                        dy = random.random() - 0.5
                        dist = math.sqrt(dx*dx + dy*dy) + 1e-12
                    push = (min_dist - dist) / dist * 0.55
                    circles[i][0] += dx * push
                    circles[i][1] += dy * push
                    circles[j][0] -= dx * push
                    circles[j][1] -= dy * push
                    moved = True
            circles[i][0] = max(circles[i][2], min(1 - circles[i][2], circles[i][0]))
            circles[i][1] = max(circles[i][2], min(1 - circles[i][2], circles[i][1]))
        if not moved:
            break
    for _ in range(50):
        any_overlap = False
        for i in range(n):
            for j in range(i+1, n):
                dx = circles[i][0] - circles[j][0]
                dy = circles[i][1] - circles[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < circles[i][2] + circles[j][2] - 1e-10:
                    shrink = (circles[i][2] + circles[j][2] - dist) / 2 + 1e-11
                    circles[i][2] = max(0.001, circles[i][2] - shrink)
                    circles[j][2] = max(0.001, circles[j][2] - shrink)
                    any_overlap = True
        if not any_overlap:
            break
    for c in circles:
        c[0] = max(c[2], min(1 - c[2], c[0]))
        c[1] = max(c[2], min(1 - c[2], c[1]))
    return circles


def three_stage_optimize(circles, verbose=False):
    """Run the three-stage optimization pipeline."""
    # Stage 1: Optimize positions
    circles = optimize_positions(circles, maxiter=300, penalty_weight=300)

    # Fix any remaining issues
    circles = push_apart(circles)
    circles = grow_radii(circles)

    # Stage 2: Optimize radii
    circles = optimize_radii(circles, maxiter=500)

    # Stage 3: Joint optimization (multiple rounds)
    for round_num in range(3):
        circles = optimize_joint(circles, maxiter=3000)
        circles = grow_radii(circles)

    return circles


def main():
    script_dir = Path(__file__).parent
    solution_path = script_dir / "solution_n26.json"
    n = 26

    # Load current best
    if solution_path.exists():
        with open(solution_path) as f:
            data = json.load(f)
        best_circles = [list(c) for c in data["circles"]]
        best_metric = sum_radii(best_circles)
    else:
        best_circles = None
        best_metric = 0.0

    seed = 42424
    num_trials = 100
    if len(sys.argv) > 1:
        num_trials = int(sys.argv[1])
    if len(sys.argv) > 2:
        seed = int(sys.argv[2])

    print(f"Starting best: {best_metric:.10f}")
    print(f"Running {num_trials} three-stage optimization trials...")

    start_time = time.time()

    patterns = [
        pattern_corners_edges_center,
        pattern_hybrid,
        pattern_billiard,
        pattern_greedy_largest_gap,
        pattern_diagonal_symmetric,
        pattern_grid_optimized,
    ]

    for idx in range(num_trials):
        pattern_fn = patterns[idx % len(patterns)]
        trial_seed = seed + idx * 7

        try:
            circles = pattern_fn(n, trial_seed)
            circles = push_apart(circles)
            circles = grow_radii(circles)

            # Three-stage optimization
            optimized = three_stage_optimize(circles)

            if is_valid(optimized):
                metric = sum_radii(optimized)

                if metric > best_metric:
                    best_metric = metric
                    best_circles = optimized
                    elapsed = time.time() - start_time
                    print(f"  [{idx}] NEW BEST: {best_metric:.10f} (pattern={pattern_fn.__name__}) [{elapsed:.1f}s]")
                elif idx % 15 == 0:
                    elapsed = time.time() - start_time
                    print(f"  [{idx}] metric={metric:.6f} [{elapsed:.1f}s]")
        except Exception as e:
            if idx % 20 == 0:
                print(f"  [{idx}] Error: {e}")

    elapsed = time.time() - start_time
    print(f"\nFinal best: {best_metric:.10f} ({elapsed:.1f}s)")

    if best_circles is not None:
        with open(solution_path, 'w') as f:
            json.dump({"circles": best_circles}, f, indent=2)
        print(f"Saved to {solution_path}")


if __name__ == "__main__":
    main()
