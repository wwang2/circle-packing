"""
Targeted refinement of best solution.
Focus on:
1. Relocating smallest circles to better positions
2. Symmetric perturbations (exploit diagonal symmetry)
3. Multiple SLSQP restarts with slightly different starting points
"""

import json
import math
import random
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


def scipy_polish(circles, maxiter=15000):
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


def find_best_position(circles, idx, n_samples=5000):
    """Find the best position for circle idx given all others fixed."""
    n = len(circles)
    rng = random.Random(42 + idx)

    best_gap = 0
    best_pos = (circles[idx][0], circles[idx][1])

    for _ in range(n_samples):
        x = 0.01 + rng.random() * 0.98
        y = 0.01 + rng.random() * 0.98
        gap = min(x, 1-x, y, 1-y)
        for j in range(n):
            if j == idx:
                continue
            d = math.sqrt((x - circles[j][0])**2 + (y - circles[j][1])**2)
            gap = min(gap, d - circles[j][2])
        if gap > best_gap:
            best_gap = gap
            best_pos = (x, y)

    return best_pos, best_gap


def refine_by_relocating(circles, n_rounds=5):
    """Repeatedly relocate the smallest circles to better positions."""
    n = len(circles)
    circles = [list(c) for c in circles]
    best_metric = sum_radii(circles)

    for round_num in range(n_rounds):
        # Sort by radius to find smallest
        radii = [c[2] for c in circles]
        order = sorted(range(n), key=lambda i: radii[i])

        improved = False
        # Try relocating each of the smallest 10 circles
        for idx in order[:10]:
            trial = [list(c) for c in circles]
            pos, gap = find_best_position(trial, idx, n_samples=3000)

            if gap > trial[idx][2] * 1.05:  # Only if significantly better
                trial[idx][0], trial[idx][1] = pos
                trial = grow_radii(trial)

                if is_valid(trial) and sum_radii(trial) > best_metric:
                    circles = trial
                    best_metric = sum_radii(trial)
                    improved = True

        if not improved:
            break

    return circles


def symmetric_perturbation(circles, rng, pair_map=None):
    """Apply symmetric perturbation exploiting diagonal symmetry."""
    n = len(circles)
    trial = [list(c) for c in circles]

    # Find approximate diagonal symmetry pairs
    if pair_map is None:
        pair_map = {}
        used = set()
        for i in range(n):
            if i in used:
                continue
            # Look for mirror partner along y=x diagonal
            mirror_x = trial[i][1]  # swap x,y
            mirror_y = trial[i][0]
            best_j = -1
            best_dist = 0.05
            for j in range(n):
                if j == i or j in used:
                    continue
                d = math.sqrt((trial[j][0] - mirror_x)**2 + (trial[j][1] - mirror_y)**2)
                if d < best_dist:
                    best_dist = d
                    best_j = j
            if best_j >= 0:
                pair_map[i] = best_j
                pair_map[best_j] = i
                used.add(i)
                used.add(best_j)

    # Apply symmetric perturbation to a random pair
    paired = [(i, j) for i, j in pair_map.items() if i < j]
    if not paired:
        return trial, pair_map

    i, j = rng.choice(paired)
    scale = rng.uniform(0.005, 0.03)
    dx = rng.gauss(0, scale)
    dy = rng.gauss(0, scale)

    trial[i][0] += dx
    trial[i][1] += dy
    trial[j][0] += dy  # Mirror: swap dx/dy
    trial[j][1] += dx

    for c in trial:
        c[0] = max(c[2], min(1 - c[2], c[0]))
        c[1] = max(c[2], min(1 - c[2], c[1]))

    return trial, pair_map


def main():
    script_dir = Path(__file__).parent
    solution_path = script_dir / "solution_n26.json"

    with open(solution_path) as f:
        data = json.load(f)
    circles = [list(c) for c in data["circles"]]
    n = len(circles)

    best_metric = sum_radii(circles)
    best_circles = circles
    print(f"Starting: {best_metric:.10f}")

    start_time = time.time()
    rng = random.Random(77777)

    # Phase 1: Relocate smallest circles
    print("\nPhase 1: Relocating smallest circles...")
    relocated = refine_by_relocating(circles, n_rounds=5)
    relocated = grow_radii(relocated)

    try:
        polished = scipy_polish(relocated, maxiter=15000)
        polished = grow_radii(polished)
        if is_valid(polished):
            metric = sum_radii(polished)
            if metric > best_metric:
                best_metric = metric
                best_circles = polished
                print(f"  Improved by relocation: {best_metric:.10f}")
    except:
        pass

    # Phase 2: Symmetric perturbations + polish
    print(f"\nPhase 2: Symmetric perturbations (from {best_metric:.10f})...")
    pair_map = None
    no_improve = 0

    for trial_num in range(150):
        trial, pair_map = symmetric_perturbation(best_circles, rng, pair_map)
        trial = grow_radii(trial)

        try:
            polished = scipy_polish(trial, maxiter=8000)
            polished = grow_radii(polished)
            if is_valid(polished):
                metric = sum_radii(polished)
                if metric > best_metric:
                    improvement = metric - best_metric
                    best_metric = metric
                    best_circles = polished
                    no_improve = 0
                    elapsed = time.time() - start_time
                    print(f"  [{trial_num}] {best_metric:.10f} (+{improvement:.8f}) [{elapsed:.1f}s]")
                else:
                    no_improve += 1
            else:
                no_improve += 1
        except:
            no_improve += 1

        if no_improve >= 40:
            print(f"  Stopping after {no_improve} non-improving rounds.")
            break

    # Phase 3: Random aggressive perturbations
    print(f"\nPhase 3: Aggressive perturbations (from {best_metric:.10f})...")
    no_improve = 0
    for trial_num in range(100):
        trial = [list(c) for c in best_circles]

        # Pick 1-3 circles and move them significantly
        k = rng.randint(1, 3)
        indices = rng.sample(range(n), k)
        for i in indices:
            trial[i][0] += rng.gauss(0, 0.05)
            trial[i][1] += rng.gauss(0, 0.05)
            trial[i][0] = max(0.02, min(0.98, trial[i][0]))
            trial[i][1] = max(0.02, min(0.98, trial[i][1]))

        trial = grow_radii(trial)

        try:
            polished = scipy_polish(trial, maxiter=8000)
            polished = grow_radii(polished)
            if is_valid(polished):
                metric = sum_radii(polished)
                if metric > best_metric:
                    improvement = metric - best_metric
                    best_metric = metric
                    best_circles = polished
                    no_improve = 0
                    elapsed = time.time() - start_time
                    print(f"  [{trial_num}] {best_metric:.10f} (+{improvement:.8f}) [{elapsed:.1f}s]")
                else:
                    no_improve += 1
            else:
                no_improve += 1
        except:
            no_improve += 1

        if no_improve >= 30:
            break

    elapsed = time.time() - start_time
    print(f"\nFinal: {best_metric:.10f} ({elapsed:.1f}s)")

    with open(solution_path, 'w') as f:
        json.dump({"circles": best_circles}, f, indent=2)
    print(f"Saved to {solution_path}")


if __name__ == "__main__":
    main()
