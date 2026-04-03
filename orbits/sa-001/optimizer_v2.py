"""
V2 optimizer: Focus on finding the right topology/basin.
Key insight: SLSQP does great within a basin, so we need to explore MORE basins.

Strategy:
1. Generate many more diverse inits (100+)
2. Use basin-hopping style: large perturbations + SLSQP
3. Track contact graph topology to avoid revisiting same basin
4. Use known SOTA structure hints (from literature)
"""

import json
import math
import random
import sys
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from pathlib import Path
from itertools import combinations


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
    """Push overlapping circles apart."""
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

    # Shrink if still overlapping
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


def get_contact_signature(circles, tol=0.001):
    """Get a hashable signature of the contact graph."""
    n = len(circles)
    contacts = []
    for i in range(n):
        for j in range(i+1, n):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = math.sqrt(dx*dx + dy*dy)
            gap = dist - circles[i][2] - circles[j][2]
            if gap < tol:
                contacts.append((i, j))
        # Wall contacts
        x, y, r = circles[i]
        if x - r < tol: contacts.append(('L', i))
        if 1 - x - r < tol: contacts.append(('R', i))
        if y - r < tol: contacts.append(('B', i))
        if 1 - y - r < tol: contacts.append(('T', i))

    return frozenset(contacts)


# ============================================================
# Initialization strategies for n=26
# ============================================================

def init_structured_26(seed, variant=0):
    """Create structured initializations inspired by known good packings.

    Good n=26 packings tend to have:
    - 4-6 large circles (r ~ 0.12-0.16)
    - ~10 medium circles (r ~ 0.06-0.10)
    - ~10 small circles filling gaps (r ~ 0.03-0.06)
    """
    rng = random.Random(seed)
    circles = []

    if variant == 0:
        # 5 large + fill pattern
        large_positions = [
            (0.16, 0.16), (0.84, 0.16), (0.16, 0.84), (0.84, 0.84), (0.5, 0.5)
        ]
        for x, y in large_positions:
            r = 0.13 + rng.uniform(-0.02, 0.02)
            circles.append([x + rng.gauss(0, 0.02), y + rng.gauss(0, 0.02), r])

    elif variant == 1:
        # 6 large in 2x3 grid
        for i in range(2):
            for j in range(3):
                x = (j + 0.5) / 3 + rng.gauss(0, 0.02)
                y = (i + 0.5) / 2 + rng.gauss(0, 0.02)
                r = 0.12 + rng.uniform(-0.02, 0.02)
                circles.append([x, y, r])

    elif variant == 2:
        # 4 large corners + 2 large center row
        for x, y in [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]:
            circles.append([x + rng.gauss(0, 0.01), y + rng.gauss(0, 0.01),
                          0.14 + rng.uniform(-0.01, 0.01)])
        for x, y in [(0.35, 0.5), (0.65, 0.5)]:
            circles.append([x + rng.gauss(0, 0.02), y + rng.gauss(0, 0.02),
                          0.12 + rng.uniform(-0.01, 0.01)])

    elif variant == 3:
        # Triangle arrangement: rows of 7, 6, 5, 4, 3, 1
        row_sizes = [7, 6, 5, 4, 3, 1]
        r_base = 0.065
        y = r_base
        for row_n in row_sizes:
            if len(circles) >= 26:
                break
            y_step = r_base * 1.8
            x_start = 0.5 - (row_n - 1) * r_base
            for j in range(row_n):
                if len(circles) >= 26:
                    break
                x = x_start + j * r_base * 2
                r = r_base + rng.uniform(-0.01, 0.01)
                circles.append([max(r, min(1-r, x)), max(r, min(1-r, y)), r])
            y += y_step

    elif variant == 4:
        # Diagonal arrangement
        for i in range(6):
            r = 0.075 + rng.uniform(-0.01, 0.01)
            x = 0.1 + i * 0.16 + rng.gauss(0, 0.01)
            y = 0.1 + i * 0.16 + rng.gauss(0, 0.01)
            circles.append([max(r, min(1-r, x)), max(r, min(1-r, y)), r])
        for i in range(5):
            r = 0.065 + rng.uniform(-0.01, 0.01)
            x = 0.18 + i * 0.16 + rng.gauss(0, 0.01)
            y = 0.9 - i * 0.16 + rng.gauss(0, 0.01)
            circles.append([max(r, min(1-r, x)), max(r, min(1-r, y)), r])

    else:
        # Random structured: place circles greedily in largest gap
        for _ in range(min(6, 26)):
            if not circles:
                circles.append([0.5, 0.5, 0.15])
                continue
            # Find largest gap
            best_r, best_x, best_y = 0, 0.5, 0.5
            for _ in range(500):
                x = 0.02 + rng.random() * 0.96
                y = 0.02 + rng.random() * 0.96
                max_r = min(x, 1-x, y, 1-y)
                for cx, cy, cr in circles:
                    d = math.sqrt((x-cx)**2 + (y-cy)**2)
                    max_r = min(max_r, d - cr)
                if max_r > best_r:
                    best_r, best_x, best_y = max_r, x, y
            circles.append([best_x, best_y, max(0.01, best_r - 0.001)])

    # Fill remaining with greedy gap-finding
    while len(circles) < 26:
        best_r, best_x, best_y = 0, 0.5, 0.5
        for _ in range(1000):
            x = 0.01 + rng.random() * 0.98
            y = 0.01 + rng.random() * 0.98
            max_r = min(x, 1-x, y, 1-y)
            for cx, cy, cr in circles:
                d = math.sqrt((x-cx)**2 + (y-cy)**2)
                max_r = min(max_r, d - cr)
            if max_r > best_r:
                best_r, best_x, best_y = max_r, x, y
        circles.append([best_x, best_y, max(0.005, best_r - 0.001)])

    return circles[:26]


def init_from_best_perturbed(best_circles, rng, scale=0.1):
    """Large perturbation of best known solution."""
    n = len(best_circles)
    circles = [list(c) for c in best_circles]

    ptype = rng.random()

    if ptype < 0.3:
        # Shuffle positions of subset
        k = rng.randint(3, n // 2)
        indices = rng.sample(range(n), k)
        positions = [(circles[i][0], circles[i][1]) for i in indices]
        rng.shuffle(positions)
        for idx, (x, y) in zip(indices, positions):
            circles[idx][0] = x
            circles[idx][1] = y

    elif ptype < 0.5:
        # Mirror/flip
        axis = rng.choice(['x', 'y', 'diag', 'adiag'])
        for c in circles:
            if axis == 'x':
                c[1] = 1 - c[1]
            elif axis == 'y':
                c[0] = 1 - c[0]
            elif axis == 'diag':
                c[0], c[1] = c[1], c[0]
            else:
                c[0], c[1] = 1 - c[1], 1 - c[0]
        # Plus perturbation
        for c in circles:
            c[0] += rng.gauss(0, scale * 0.3)
            c[1] += rng.gauss(0, scale * 0.3)

    elif ptype < 0.7:
        # Move all circles by random amount
        for c in circles:
            c[0] += rng.gauss(0, scale)
            c[1] += rng.gauss(0, scale)

    else:
        # Replace worst circles with random new positions
        radii = [c[2] for c in circles]
        n_replace = rng.randint(2, min(8, n))
        # Get indices of smallest circles
        indices = sorted(range(n), key=lambda i: radii[i])[:n_replace]
        for i in indices:
            circles[i][0] = 0.05 + rng.random() * 0.9
            circles[i][1] = 0.05 + rng.random() * 0.9
            circles[i][2] = 0.01

    # Clamp
    for c in circles:
        c[2] = max(0.005, c[2])
        c[0] = max(c[2], min(1 - c[2], c[0]))
        c[1] = max(c[2], min(1 - c[2], c[1]))

    return circles


# ============================================================
# Main basin-hopping loop
# ============================================================

def main():
    script_dir = Path(__file__).parent
    solution_path = script_dir / "solution_n26.json"
    n = 26

    # Load current best if exists
    if solution_path.exists():
        with open(solution_path) as f:
            data = json.load(f)
        best_circles = [list(c) for c in data["circles"]]
        best_metric = sum_radii(best_circles)
    else:
        best_circles = None
        best_metric = 0.0

    n_inits = 80
    seed = 7777
    if len(sys.argv) > 1:
        n_inits = int(sys.argv[1])
    if len(sys.argv) > 2:
        seed = int(sys.argv[2])

    rng = random.Random(seed)
    np.random.seed(seed)

    print(f"Starting best: {best_metric:.10f}")
    print(f"Generating {n_inits} diverse initializations...")

    start_time = time.time()
    seen_signatures = set()
    all_results = []

    for idx in range(n_inits):
        # Generate initialization
        if idx < 30:
            # Structured variants
            circles = init_structured_26(seed + idx, variant=idx % 6)
        elif idx < 50 and best_circles is not None:
            # Perturbations of best
            circles = init_from_best_perturbed(best_circles, rng, scale=0.05 + 0.1 * (idx / n_inits))
        else:
            # Random structured greedy
            circles = init_structured_26(seed + idx + 1000, variant=5)

        # Fix and grow
        circles = push_apart(circles)
        circles = grow_radii(circles)

        if not is_valid(circles):
            continue

        pre_metric = sum_radii(circles)

        # Polish
        try:
            polished = scipy_polish(circles, maxiter=8000)
            polished = grow_radii(polished)

            if is_valid(polished):
                metric = sum_radii(polished)
                sig = get_contact_signature(polished)

                if sig not in seen_signatures:
                    seen_signatures.add(sig)
                    all_results.append((metric, polished))

                    if metric > best_metric:
                        best_metric = metric
                        best_circles = polished
                        elapsed = time.time() - start_time
                        print(f"  [{idx}] NEW BEST: {best_metric:.10f} (from {pre_metric:.4f}) [{elapsed:.1f}s]")
                    elif idx % 10 == 0:
                        elapsed = time.time() - start_time
                        print(f"  [{idx}] metric={metric:.6f}, unique_basins={len(seen_signatures)} [{elapsed:.1f}s]")
        except Exception:
            pass

    elapsed = time.time() - start_time

    # Sort all results
    all_results.sort(reverse=True)
    print(f"\nTop 5 results:")
    for i, (m, _) in enumerate(all_results[:5]):
        print(f"  {i+1}. {m:.10f}")

    print(f"\nBest metric: {best_metric:.10f}")
    print(f"Unique basins explored: {len(seen_signatures)}")
    print(f"Elapsed: {elapsed:.1f}s")

    # Save best
    if best_circles is not None:
        solution = {"circles": best_circles}
        with open(solution_path, 'w') as f:
            json.dump(solution, f, indent=2)
        print(f"Saved to {solution_path}")


if __name__ == "__main__":
    main()
