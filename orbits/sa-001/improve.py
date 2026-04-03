"""
Intensively improve an existing circle packing solution.
- Load best solution
- Run many rounds of perturbation + SLSQP polish
- Try different perturbation strategies
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


def perturb_and_polish(circles, rng, scale=0.02):
    """Perturb positions and re-optimize."""
    n = len(circles)
    trial = [list(c) for c in circles]

    ptype = rng.random()

    if ptype < 0.25:
        # Move one circle
        i = rng.randint(0, n-1)
        trial[i][0] += rng.gauss(0, scale)
        trial[i][1] += rng.gauss(0, scale)

    elif ptype < 0.45:
        # Swap two circles
        i, j = rng.sample(range(n), 2)
        trial[i][0], trial[j][0] = trial[j][0], trial[i][0]
        trial[i][1], trial[j][1] = trial[j][1], trial[i][1]

    elif ptype < 0.6:
        # Move 2-4 circles
        k = rng.randint(2, 4)
        indices = rng.sample(range(n), k)
        for i in indices:
            trial[i][0] += rng.gauss(0, scale)
            trial[i][1] += rng.gauss(0, scale)

    elif ptype < 0.75:
        # Rotate a subset around center
        k = rng.randint(3, min(8, n))
        indices = rng.sample(range(n), k)
        angle = rng.gauss(0, 0.1)
        cx = sum(trial[i][0] for i in indices) / k
        cy = sum(trial[i][1] for i in indices) / k
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        for i in indices:
            dx = trial[i][0] - cx
            dy = trial[i][1] - cy
            trial[i][0] = cx + dx * cos_a - dy * sin_a
            trial[i][1] = cy + dx * sin_a + dy * cos_a

    elif ptype < 0.85:
        # Move smallest circle to best empty region
        radii = [c[2] for c in trial]
        i = radii.index(min(radii))
        # Try random positions
        best_gap = 0
        best_pos = (trial[i][0], trial[i][1])
        for _ in range(100):
            x = 0.02 + rng.random() * 0.96
            y = 0.02 + rng.random() * 0.96
            min_gap = min(x, 1-x, y, 1-y)
            for j in range(n):
                if j == i:
                    continue
                dist = math.sqrt((x-trial[j][0])**2 + (y-trial[j][1])**2)
                min_gap = min(min_gap, dist - trial[j][2])
            if min_gap > best_gap:
                best_gap = min_gap
                best_pos = (x, y)
        trial[i][0], trial[i][1] = best_pos

    else:
        # Shrink all radii slightly and re-grow (escape local optimum)
        for i in range(n):
            trial[i][2] *= (0.9 + rng.random() * 0.08)
        # Also perturb positions
        for i in range(n):
            trial[i][0] += rng.gauss(0, scale * 0.5)
            trial[i][1] += rng.gauss(0, scale * 0.5)

    # Clamp positions
    for c in trial:
        c[2] = max(0.005, c[2])
        c[0] = max(c[2], min(1 - c[2], c[0]))
        c[1] = max(c[2], min(1 - c[2], c[1]))

    # Grow radii
    trial = grow_radii(trial)

    # Polish
    try:
        polished = scipy_polish(trial, maxiter=5000)
        polished = grow_radii(polished)
        if is_valid(polished):
            return polished, sum_radii(polished)
    except Exception:
        pass

    if is_valid(trial):
        return trial, sum_radii(trial)

    return None, 0.0


def main():
    script_dir = Path(__file__).parent

    # Load current best
    solution_path = script_dir / "solution_n26.json"
    if not solution_path.exists():
        print("No solution_n26.json found. Run optimizer.py first.")
        sys.exit(1)

    with open(solution_path) as f:
        data = json.load(f)
    circles = [list(c) for c in data["circles"]]

    n_rounds = 100
    seed = 123
    if len(sys.argv) > 1:
        n_rounds = int(sys.argv[1])
    if len(sys.argv) > 2:
        seed = int(sys.argv[2])

    rng = random.Random(seed)

    best = circles
    best_metric = sum_radii(best)
    print(f"Starting metric: {best_metric:.10f}")

    no_improve = 0
    start_time = time.time()

    for round_num in range(n_rounds):
        # Adaptive scale: start larger, get smaller
        scale = 0.05 * (0.5 ** (round_num / 30))
        scale = max(scale, 0.005)

        result, metric = perturb_and_polish(best, rng, scale=scale)

        if result is not None and metric > best_metric:
            improvement = metric - best_metric
            best = result
            best_metric = metric
            no_improve = 0
            elapsed = time.time() - start_time
            print(f"  Round {round_num}: {best_metric:.10f} (+{improvement:.8f}) [{elapsed:.1f}s]")
        else:
            no_improve += 1

        if no_improve >= 30 and round_num > 50:
            print(f"No improvement in {no_improve} rounds, stopping.")
            break

    elapsed = time.time() - start_time
    print(f"\nFinal metric: {best_metric:.10f} (elapsed: {elapsed:.1f}s)")

    # Save
    solution = {"circles": best}
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)
    print(f"Saved to {solution_path}")


if __name__ == "__main__":
    main()
