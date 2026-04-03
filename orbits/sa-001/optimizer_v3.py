"""
V3: Targeted approach for n=26.
Key insight: The SOTA topology has specific circle size distribution.
From literature, the best n=26 packing has circles with sizes roughly:
  ~4 circles r~0.135-0.155
  ~6 circles r~0.09-0.12
  ~8 circles r~0.06-0.085
  ~8 circles r~0.035-0.055

Strategy:
1. Generate candidate radius distributions
2. For each distribution, optimize positions via SLSQP
3. Use LP to assign radii to positions optimally
4. Multiple rounds of position perturbation + re-optimization
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


def scipy_polish(circles, maxiter=10000):
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
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+2] - 1e-8})

    for i in range(n):
        for j in range(i+1, n):
            constraints.append({'type': 'ineq',
                'fun': lambda x, i=i, j=j: (
                    (x[3*i]-x[3*j])**2 + (x[3*i+1]-x[3*j+1])**2
                    - (x[3*i+2]+x[3*j+2])**2
                )})

    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-8, 0.5)] * n

    result = minimize(objective, x0, method='SLSQP',
                     jac=grad_objective, constraints=constraints,
                     bounds=bounds,
                     options={'maxiter': maxiter, 'ftol': 1e-15})

    polished = [[result.x[3*i], result.x[3*i+1], result.x[3*i+2]] for i in range(n)]
    return polished


def greedy_placement(n, radii_hint=None, seed=42):
    """Place circles greedily: largest first, in the biggest available gap."""
    rng = random.Random(seed)

    if radii_hint is None:
        # Default radius distribution for n=26
        radii_hint = sorted([
            0.155, 0.145, 0.140, 0.135,  # 4 large
            0.115, 0.110, 0.105, 0.100, 0.095, 0.090,  # 6 medium
            0.080, 0.075, 0.070, 0.065, 0.060, 0.058, 0.055, 0.052,  # 8 small-med
            0.048, 0.045, 0.042, 0.040, 0.038, 0.035, 0.033, 0.030,  # 8 small
        ], reverse=True)[:n]

    circles = []
    for idx, target_r in enumerate(radii_hint):
        # Find best position for this circle
        best_pos = None
        best_gap = -1

        n_samples = 2000 if idx < 6 else 1000
        for _ in range(n_samples):
            x = target_r + rng.random() * (1 - 2 * target_r)
            y = target_r + rng.random() * (1 - 2 * target_r)

            # Compute max radius at this position
            max_r = min(x, 1-x, y, 1-y)
            for cx, cy, cr in circles:
                d = math.sqrt((x-cx)**2 + (y-cy)**2)
                max_r = min(max_r, d - cr)

            if max_r > best_gap:
                best_gap = max_r
                best_pos = (x, y)

        if best_pos is not None:
            r = min(target_r, max(0.005, best_gap - 1e-10))
            circles.append([best_pos[0], best_pos[1], r])
        else:
            circles.append([0.5, 0.5, 0.005])

    return circles


def generate_radius_distributions(n, num_dists, seed=42):
    """Generate diverse radius distributions that sum to ~2.6."""
    rng = random.Random(seed)
    distributions = []

    # Type 1: Based on known good structure
    for i in range(num_dists // 4):
        radii = []
        # Large circles
        n_large = rng.randint(3, 6)
        for _ in range(n_large):
            radii.append(rng.uniform(0.11, 0.17))
        # Medium
        n_med = rng.randint(4, 8)
        for _ in range(n_med):
            radii.append(rng.uniform(0.06, 0.11))
        # Fill rest with small
        while len(radii) < n:
            radii.append(rng.uniform(0.02, 0.06))
        radii = sorted(radii[:n], reverse=True)
        distributions.append(radii)

    # Type 2: More uniform
    for i in range(num_dists // 4):
        base = 2.6 / n  # ~0.1 each
        radii = [base + rng.gauss(0, 0.03) for _ in range(n)]
        radii = [max(0.02, r) for r in radii]
        radii = sorted(radii, reverse=True)
        distributions.append(radii)

    # Type 3: Power-law-ish
    for i in range(num_dists // 4):
        alpha = rng.uniform(0.3, 0.8)
        radii = [0.2 * ((j+1) / n) ** (-alpha) for j in range(n)]
        radii = [max(0.02, min(0.2, r)) for r in radii]
        radii = sorted(radii, reverse=True)
        distributions.append(radii)

    # Type 4: Specific known-good-ish distributions
    for i in range(num_dists - len(distributions)):
        # Mimic AlphaEvolve-style: a few big, many medium-small
        radii = []
        for j in range(n):
            if j < 2:
                radii.append(rng.uniform(0.14, 0.17))
            elif j < 6:
                radii.append(rng.uniform(0.10, 0.14))
            elif j < 14:
                radii.append(rng.uniform(0.07, 0.10))
            else:
                radii.append(rng.uniform(0.03, 0.07))
        radii = sorted(radii, reverse=True)
        distributions.append(radii)

    return distributions


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

    num_trials = 60
    seed = 54321
    if len(sys.argv) > 1:
        num_trials = int(sys.argv[1])
    if len(sys.argv) > 2:
        seed = int(sys.argv[2])

    print(f"Starting best: {best_metric:.10f}")
    print(f"Running {num_trials} greedy placement trials...")

    start_time = time.time()
    distributions = generate_radius_distributions(n, num_trials, seed=seed)

    for idx, radii in enumerate(distributions):
        circles = greedy_placement(n, radii_hint=radii, seed=seed + idx * 100)
        circles = push_apart(circles)
        circles = grow_radii(circles)

        if not is_valid(circles):
            continue

        pre_metric = sum_radii(circles)

        try:
            polished = scipy_polish(circles, maxiter=8000)
            polished = grow_radii(polished)

            if is_valid(polished):
                metric = sum_radii(polished)

                if metric > best_metric:
                    best_metric = metric
                    best_circles = polished
                    elapsed = time.time() - start_time
                    print(f"  [{idx}] NEW BEST: {best_metric:.10f} [{elapsed:.1f}s]")
                elif idx % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"  [{idx}] metric={metric:.6f} [{elapsed:.1f}s]")
        except Exception:
            pass

    # Now try intensive improvement on best
    if best_circles is not None:
        print(f"\nStarting intensive improvement from {best_metric:.10f}...")
        rng = random.Random(seed + 999999)

        no_improve = 0
        for trial in range(100):
            trial_circles = [list(c) for c in best_circles]

            # Perturbation
            ptype = rng.random()
            scale = 0.03 * (0.7 ** (trial / 30))

            if ptype < 0.3:
                i = rng.randint(0, n-1)
                trial_circles[i][0] += rng.gauss(0, scale)
                trial_circles[i][1] += rng.gauss(0, scale)
            elif ptype < 0.5:
                i, j = rng.sample(range(n), 2)
                trial_circles[i][0], trial_circles[j][0] = trial_circles[j][0], trial_circles[i][0]
                trial_circles[i][1], trial_circles[j][1] = trial_circles[j][1], trial_circles[i][1]
            elif ptype < 0.7:
                k = rng.randint(2, 5)
                indices = rng.sample(range(n), k)
                for i in indices:
                    trial_circles[i][0] += rng.gauss(0, scale)
                    trial_circles[i][1] += rng.gauss(0, scale)
            elif ptype < 0.85:
                # Relocate smallest
                radii = [c[2] for c in trial_circles]
                i = radii.index(min(radii))
                best_gap, best_pos = 0, (trial_circles[i][0], trial_circles[i][1])
                for _ in range(200):
                    x = 0.02 + rng.random() * 0.96
                    y = 0.02 + rng.random() * 0.96
                    gap = min(x, 1-x, y, 1-y)
                    for j in range(n):
                        if j == i: continue
                        d = math.sqrt((x-trial_circles[j][0])**2 + (y-trial_circles[j][1])**2)
                        gap = min(gap, d - trial_circles[j][2])
                    if gap > best_gap:
                        best_gap, best_pos = gap, (x, y)
                trial_circles[i][0], trial_circles[i][1] = best_pos
            else:
                # Rotate subset
                k = rng.randint(3, 8)
                indices = rng.sample(range(n), k)
                angle = rng.gauss(0, 0.15)
                cx = sum(trial_circles[i][0] for i in indices) / k
                cy = sum(trial_circles[i][1] for i in indices) / k
                cos_a, sin_a = math.cos(angle), math.sin(angle)
                for i in indices:
                    dx = trial_circles[i][0] - cx
                    dy = trial_circles[i][1] - cy
                    trial_circles[i][0] = cx + dx * cos_a - dy * sin_a
                    trial_circles[i][1] = cy + dx * sin_a + dy * cos_a

            for c in trial_circles:
                c[2] = max(0.005, c[2])
                c[0] = max(c[2], min(1 - c[2], c[0]))
                c[1] = max(c[2], min(1 - c[2], c[1]))

            trial_circles = grow_radii(trial_circles)

            try:
                polished = scipy_polish(trial_circles, maxiter=5000)
                polished = grow_radii(polished)
                if is_valid(polished):
                    metric = sum_radii(polished)
                    if metric > best_metric:
                        improvement = metric - best_metric
                        best_metric = metric
                        best_circles = polished
                        no_improve = 0
                        elapsed = time.time() - start_time
                        print(f"  Improve [{trial}]: {best_metric:.10f} (+{improvement:.8f}) [{elapsed:.1f}s]")
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            except:
                no_improve += 1

            if no_improve >= 30:
                print(f"  No improvement in {no_improve} rounds.")
                break

    elapsed = time.time() - start_time
    print(f"\nFinal best: {best_metric:.10f} ({elapsed:.1f}s)")

    if best_circles is not None:
        with open(solution_path, 'w') as f:
            json.dump({"circles": best_circles}, f, indent=2)
        print(f"Saved to {solution_path}")


if __name__ == "__main__":
    main()
