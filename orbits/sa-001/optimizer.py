"""
Fast circle packing optimizer using multi-start + scipy SLSQP.
Focus on generating many good initializations and polishing with scipy.
SA is too slow in pure Python; instead use greedy growth + scipy.
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
# Initialization strategies
# ============================================================

def init_hex_grid(n, scale=1.0, offset_x=0.0, offset_y=0.0):
    """Hex grid with variable scale/offset."""
    cols = int(math.ceil(math.sqrt(n * 2 / math.sqrt(3))))
    rows = int(math.ceil(n / cols))
    r_est = 0.45 / max(cols, rows)

    circles = []
    for i in range(rows + 1):
        for j in range(cols + 1):
            if len(circles) >= n:
                break
            x = (j + 0.5) / (cols + 1) + offset_x
            y = (i + 0.5) / (rows + 1) + offset_y
            if i % 2 == 1:
                x += 0.5 / (cols + 1)
            x = max(0.02, min(0.98, x))
            y = max(0.02, min(0.98, y))
            circles.append([x, y, r_est * scale])
    return circles[:n]


def init_random_placement(n, seed):
    """Random placement with rejection sampling."""
    rng = random.Random(seed)
    r_base = 0.3 / math.sqrt(n)
    circles = []
    for _ in range(n):
        placed = False
        for _ in range(2000):
            r = r_base * (0.5 + rng.random() * 1.0)
            x = r + rng.random() * (1 - 2*r)
            y = r + rng.random() * (1 - 2*r)
            ok = True
            for cx, cy, cr in circles:
                if math.sqrt((x-cx)**2 + (y-cy)**2) < r + cr + 0.002:
                    ok = False
                    break
            if ok:
                circles.append([x, y, r])
                placed = True
                break
        if not placed:
            r = 0.005
            x = r + rng.random() * (1 - 2*r)
            y = r + rng.random() * (1 - 2*r)
            circles.append([x, y, r])
    return circles


def init_grid_varied_radii(n, seed):
    """Grid with varied radii - larger circles get more space."""
    rng = random.Random(seed)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))

    cell_w = 1.0 / cols
    cell_h = 1.0 / rows

    circles = []
    for i in range(rows):
        for j in range(cols):
            if len(circles) >= n:
                break
            cx = (j + 0.5) * cell_w + rng.gauss(0, cell_w * 0.05)
            cy = (i + 0.5) * cell_h + rng.gauss(0, cell_h * 0.05)
            r = min(cell_w, cell_h) * 0.45 * (0.7 + rng.random() * 0.6)
            cx = max(r + 0.001, min(1 - r - 0.001, cx))
            cy = max(r + 0.001, min(1 - r - 0.001, cy))
            circles.append([cx, cy, r])
    return circles[:n]


def init_mixed_sizes(n, seed):
    """Mix of large and small circles."""
    rng = random.Random(seed)
    n_large = max(2, n // 5)
    n_small = n - n_large

    circles = []
    # Place large circles first
    r_large = 0.12 + rng.random() * 0.05
    for _ in range(n_large):
        for _ in range(2000):
            r = r_large * (0.8 + rng.random() * 0.4)
            x = r + rng.random() * (1 - 2*r)
            y = r + rng.random() * (1 - 2*r)
            ok = True
            for cx, cy, cr in circles:
                if math.sqrt((x-cx)**2 + (y-cy)**2) < r + cr + 0.005:
                    ok = False
                    break
            if ok:
                circles.append([x, y, r])
                break
        else:
            r = 0.03
            x = r + rng.random() * (1 - 2*r)
            y = r + rng.random() * (1 - 2*r)
            circles.append([x, y, r])

    # Fill with small circles
    r_small = 0.04 + rng.random() * 0.03
    for _ in range(n_small):
        for _ in range(2000):
            r = r_small * (0.5 + rng.random() * 1.0)
            x = r + rng.random() * (1 - 2*r)
            y = r + rng.random() * (1 - 2*r)
            ok = True
            for cx, cy, cr in circles:
                if math.sqrt((x-cx)**2 + (y-cy)**2) < r + cr + 0.002:
                    ok = False
                    break
            if ok:
                circles.append([x, y, r])
                break
        else:
            r = 0.005
            x = r + rng.random() * (1 - 2*r)
            y = r + rng.random() * (1 - 2*r)
            circles.append([x, y, r])

    return circles[:n]


# ============================================================
# Greedy radius growth
# ============================================================

def grow_radii(circles):
    """Greedily maximize each circle's radius given positions."""
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


def push_apart_and_grow(circles, iterations=100):
    """Push overlapping circles apart, then grow radii."""
    n = len(circles)
    circles = [list(c) for c in circles]

    for _ in range(iterations):
        # Push apart
        for _ in range(50):
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
                            dist = math.sqrt(dx*dx + dy*dy)
                        overlap = min_dist - dist
                        push = overlap / dist * 0.55
                        circles[i][0] += dx * push
                        circles[i][1] += dy * push
                        circles[j][0] -= dx * push
                        circles[j][1] -= dy * push
                        moved = True
                # Clamp
                circles[i][0] = max(circles[i][2], min(1 - circles[i][2], circles[i][0]))
                circles[i][1] = max(circles[i][2], min(1 - circles[i][2], circles[i][1]))
            if not moved:
                break

        # Shrink if still overlapping
        for i in range(n):
            for j in range(i+1, n):
                dx = circles[i][0] - circles[j][0]
                dy = circles[i][1] - circles[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                min_dist = circles[i][2] + circles[j][2]
                if dist < min_dist - 1e-10:
                    shrink = (min_dist - dist) / 2 + 1e-11
                    circles[i][2] = max(0.001, circles[i][2] - shrink)
                    circles[j][2] = max(0.001, circles[j][2] - shrink)

        # Clamp
        for c in circles:
            c[0] = max(c[2], min(1 - c[2], c[0]))
            c[1] = max(c[2], min(1 - c[2], c[1]))

        # Grow
        circles = grow_radii(circles)

    return circles


# ============================================================
# Scipy SLSQP polish
# ============================================================

def scipy_polish(circles, maxiter=10000):
    """Polish solution with SLSQP."""
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
        # x_i - r_i >= 0
        constraints.append({'type': 'ineq',
            'fun': lambda x, i=i: x[3*i] - x[3*i+2]})
        # 1 - x_i - r_i >= 0
        constraints.append({'type': 'ineq',
            'fun': lambda x, i=i: 1.0 - x[3*i] - x[3*i+2]})
        # y_i - r_i >= 0
        constraints.append({'type': 'ineq',
            'fun': lambda x, i=i: x[3*i+1] - x[3*i+2]})
        # 1 - y_i - r_i >= 0
        constraints.append({'type': 'ineq',
            'fun': lambda x, i=i: 1.0 - x[3*i+1] - x[3*i+2]})
        # r_i > 0
        constraints.append({'type': 'ineq',
            'fun': lambda x, i=i: x[3*i+2] - 1e-8})

    for i in range(n):
        for j in range(i+1, n):
            constraints.append({'type': 'ineq',
                'fun': lambda x, i=i, j=j: (
                    (x[3*i]-x[3*j])**2 + (x[3*i+1]-x[3*j+1])**2
                    - (x[3*i+2]+x[3*j+2])**2
                )})

    bounds = []
    for i in range(n):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (1e-8, 0.5)])

    result = minimize(objective, x0, method='SLSQP',
                     jac=grad_objective, constraints=constraints,
                     bounds=bounds,
                     options={'maxiter': maxiter, 'ftol': 1e-15})

    polished = [[result.x[3*i], result.x[3*i+1], result.x[3*i+2]] for i in range(n)]
    return polished


# ============================================================
# Local search: perturb positions and re-optimize
# ============================================================

def local_search(circles, n_trials=200, seed=42):
    """Perturb positions and re-optimize radii, keep improvements."""
    rng = random.Random(seed)
    n = len(circles)
    best = [list(c) for c in circles]
    best_metric = sum_radii(best)

    for trial in range(n_trials):
        trial_circles = [list(c) for c in best]

        # Choose perturbation type
        ptype = rng.random()

        if ptype < 0.4:
            # Move one circle
            i = rng.randint(0, n-1)
            scale = 0.05 * (1 - trial / n_trials)  # Decrease over time
            trial_circles[i][0] += rng.gauss(0, scale)
            trial_circles[i][1] += rng.gauss(0, scale)
            trial_circles[i][0] = max(0.01, min(0.99, trial_circles[i][0]))
            trial_circles[i][1] = max(0.01, min(0.99, trial_circles[i][1]))

        elif ptype < 0.7:
            # Swap two circles' positions
            i, j = rng.sample(range(n), 2)
            trial_circles[i][0], trial_circles[j][0] = trial_circles[j][0], trial_circles[i][0]
            trial_circles[i][1], trial_circles[j][1] = trial_circles[j][1], trial_circles[i][1]

        elif ptype < 0.85:
            # Move multiple circles slightly
            k = rng.randint(2, min(5, n))
            indices = rng.sample(range(n), k)
            scale = 0.03 * (1 - trial / n_trials)
            for i in indices:
                trial_circles[i][0] += rng.gauss(0, scale)
                trial_circles[i][1] += rng.gauss(0, scale)
                trial_circles[i][0] = max(0.01, min(0.99, trial_circles[i][0]))
                trial_circles[i][1] = max(0.01, min(0.99, trial_circles[i][1]))

        else:
            # Move one circle to a random empty spot
            i = rng.randint(0, n-1)
            # Find circle with smallest radius - it's probably in a bad spot
            radii = [c[2] for c in trial_circles]
            i = radii.index(min(radii))
            trial_circles[i][0] = 0.01 + rng.random() * 0.98
            trial_circles[i][1] = 0.01 + rng.random() * 0.98

        # Re-grow radii
        trial_circles = push_apart_and_grow(trial_circles, iterations=5)
        trial_metric = sum_radii(trial_circles)

        if trial_metric > best_metric and is_valid(trial_circles):
            best = trial_circles
            best_metric = trial_metric

    return best, best_metric


# ============================================================
# Main optimization pipeline
# ============================================================

def run_optimization(n=26, num_starts=50, seed=42, verbose=True):
    """Multi-start optimization: init -> grow -> local search -> scipy polish."""
    random.seed(seed)
    np.random.seed(seed)

    best_circles = None
    best_metric = 0.0
    all_results = []

    start_time = time.time()

    if verbose:
        print(f"Generating {num_starts} initializations for n={n}...")

    inits = []

    # Hex grid variants
    for i in range(8):
        inits.append(("hex", init_hex_grid(n, scale=0.7 + i*0.05,
                       offset_x=random.uniform(-0.02, 0.02),
                       offset_y=random.uniform(-0.02, 0.02))))

    # Random placements
    for i in range(20):
        inits.append(("random", init_random_placement(n, seed=seed + 200 + i)))

    # Grid with varied radii
    for i in range(10):
        inits.append(("grid_var", init_grid_varied_radii(n, seed=seed + 400 + i)))

    # Mixed sizes
    for i in range(12):
        inits.append(("mixed", init_mixed_sizes(n, seed=seed + 600 + i)))

    if verbose:
        print(f"Generated {len(inits)} initializations")

    # Phase 1: Push apart + grow radii on all inits
    scored = []
    for name, circles in inits:
        try:
            circles = push_apart_and_grow(circles, iterations=20)
            metric = sum_radii(circles)
            if is_valid(circles):
                scored.append((metric, name, circles))
        except Exception:
            pass

    scored.sort(reverse=True)

    if verbose:
        print(f"Valid inits: {len(scored)}/{len(inits)}")
        if scored:
            print(f"Top 5: {[f'{s[1]}={s[0]:.4f}' for s in scored[:5]]}")

    # Phase 2: Local search on top candidates
    top_n = min(10, len(scored))
    phase2_results = []

    for rank, (init_metric, name, circles) in enumerate(scored[:top_n]):
        if verbose:
            print(f"\nLocal search {rank+1}/{top_n} (init={name}, metric={init_metric:.4f})...")

        improved, improved_metric = local_search(circles, n_trials=300, seed=seed + rank * 1000)

        if is_valid(improved):
            phase2_results.append((improved_metric, name, improved))
            if verbose:
                print(f"  After local search: {improved_metric:.6f}")
        else:
            phase2_results.append((init_metric, name, circles))

    phase2_results.sort(reverse=True)

    # Phase 3: Scipy polish on top candidates
    top_polish = min(5, len(phase2_results))

    for rank, (pre_metric, name, circles) in enumerate(phase2_results[:top_polish]):
        if verbose:
            print(f"\nPolishing {rank+1}/{top_polish} (metric={pre_metric:.4f})...")

        try:
            polished = scipy_polish(circles, maxiter=10000)
            polished = grow_radii(polished)

            if is_valid(polished):
                polished_metric = sum_radii(polished)
                if verbose:
                    print(f"  Polished: {polished_metric:.6f}")
                all_results.append((polished_metric, polished))

                if polished_metric > best_metric:
                    best_metric = polished_metric
                    best_circles = polished
            else:
                # Try just the pre-polish version
                if pre_metric > best_metric and is_valid(circles):
                    best_metric = pre_metric
                    best_circles = circles
                    all_results.append((pre_metric, circles))
        except Exception as e:
            if verbose:
                print(f"  Polish failed: {e}")
            if pre_metric > best_metric and is_valid(circles):
                best_metric = pre_metric
                best_circles = circles

    elapsed = time.time() - start_time
    if verbose:
        print(f"\n=== Best metric: {best_metric:.6f} (elapsed: {elapsed:.1f}s) ===")

    return best_circles, best_metric


def main():
    n = 26
    seed = 42
    num_starts = 50

    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    if len(sys.argv) > 2:
        num_starts = int(sys.argv[2])
    if len(sys.argv) > 3:
        seed = int(sys.argv[3])

    print(f"Circle packing optimizer: n={n}, starts={num_starts}, seed={seed}")

    circles, metric = run_optimization(n=n, num_starts=num_starts, seed=seed)

    if circles is None:
        print("ERROR: No valid solution found!")
        sys.exit(1)

    output_dir = Path(__file__).parent
    solution = {"circles": circles}
    solution_path = output_dir / f"solution_n{n}.json"
    with open(solution_path, 'w') as f:
        json.dump(solution, f, indent=2)

    print(f"\nSolution saved to {solution_path}")
    print(f"Final metric: {metric:.10f}")
    print(f"Valid: {is_valid(circles)}")


if __name__ == "__main__":
    main()
