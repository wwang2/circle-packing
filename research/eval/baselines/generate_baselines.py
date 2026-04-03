"""Generate baseline circle packing solutions for sanity checking.

Produces:
- trivial_n1.json: 1 circle, optimal (r=0.5)
- trivial_n4.json: 4 equal circles in corners (r=0.25 each)
- bad_overlap.json: 2 overlapping circles (should be invalid)
- bad_outside.json: circle extends outside square (should be invalid)
- baseline_n10.json: simple grid-based packing for n=10
- baseline_n26.json: multi-start optimized packing for n=26
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

# Add parent to path for evaluator import
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluator import validate_packing


OUT = Path(__file__).parent


def save(name: str, circles: list[tuple[float, float, float]]):
    data = {"circles": [[x, y, r] for x, y, r in circles]}
    path = OUT / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    result = validate_packing(circles)
    status = "VALID" if result["valid"] else "INVALID"
    print(f"  {name}: n={len(circles)}, metric={result['raw_metric']:.6f}, {status}")
    return result


def trivial_n1():
    """Optimal: single circle with r=0.5 centered at (0.5, 0.5)."""
    return [(0.5, 0.5, 0.5)]


def trivial_n4():
    """4 equal circles in corners, each r=0.25."""
    r = 0.25
    return [(r, r, r), (1-r, r, r), (r, 1-r, r), (1-r, 1-r, r)]


def bad_overlap():
    """Two overlapping circles."""
    return [(0.3, 0.5, 0.25), (0.5, 0.5, 0.25)]


def bad_outside():
    """Circle extends outside the square."""
    return [(0.1, 0.5, 0.2)]


def grid_packing(n: int) -> list[tuple[float, float, float]]:
    """Simple grid-based packing: arrange circles in a grid with equal radii."""
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    r = 0.5 / max(cols, rows)
    circles = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x = r + col * (1.0 / cols) + (0.5 / cols - r)
        y = r + row * (1.0 / rows) + (0.5 / rows - r)
        # Clamp to valid range
        x = max(r, min(1 - r, x))
        y = max(r, min(1 - r, y))
        circles.append((x, y, r))
    return circles


def optimize_packing(circles: list[tuple[float, float, float]], iterations: int = 2000) -> list[tuple[float, float, float]]:
    """Simple local optimization: alternating position and radius adjustments."""
    try:
        from scipy.optimize import minimize
        import numpy as np
    except ImportError:
        print("  scipy not available, returning unoptimized packing")
        return circles

    n = len(circles)

    def pack_to_vec(c):
        v = []
        for x, y, r in c:
            v.extend([x, y, r])
        return np.array(v)

    def vec_to_pack(v):
        return [(v[3*i], v[3*i+1], v[3*i+2]) for i in range(n)]

    def neg_objective(v):
        """Negative sum of radii + penalty for constraint violations."""
        total_r = 0.0
        penalty = 0.0
        for i in range(n):
            x, y, r = v[3*i], v[3*i+1], v[3*i+2]
            total_r += r
            # Containment penalties
            penalty += max(0, r - x) ** 2
            penalty += max(0, x + r - 1) ** 2
            penalty += max(0, r - y) ** 2
            penalty += max(0, y + r - 1) ** 2
            # Positive radius
            penalty += max(0, -r) ** 2

        # Non-overlap penalties
        for i in range(n):
            xi, yi, ri = v[3*i], v[3*i+1], v[3*i+2]
            for j in range(i+1, n):
                xj, yj, rj = v[3*j], v[3*j+1], v[3*j+2]
                dist = math.sqrt((xi-xj)**2 + (yi-yj)**2)
                min_dist = ri + rj
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2

        return -total_r + 1e4 * penalty

    v0 = pack_to_vec(circles)

    # Bounds: x in [0,1], y in [0,1], r in [1e-6, 0.5]
    bounds = []
    for _ in range(n):
        bounds.extend([(1e-6, 1 - 1e-6), (1e-6, 1 - 1e-6), (1e-6, 0.5)])

    result = minimize(neg_objective, v0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': iterations, 'ftol': 1e-15})

    optimized = vec_to_pack(result.x)

    # Verify validity — if invalid, try SLSQP with constraints
    check = validate_packing(optimized)
    if not check["valid"]:
        # Try with tighter penalty
        def neg_objective_tight(v):
            total_r = 0.0
            penalty = 0.0
            for i in range(n):
                x, y, r = v[3*i], v[3*i+1], v[3*i+2]
                total_r += r
                penalty += max(0, r - x) ** 2
                penalty += max(0, x + r - 1) ** 2
                penalty += max(0, r - y) ** 2
                penalty += max(0, y + r - 1) ** 2
                penalty += max(0, -r) ** 2
            for i in range(n):
                xi, yi, ri = v[3*i], v[3*i+1], v[3*i+2]
                for j in range(i+1, n):
                    xj, yj, rj = v[3*j], v[3*j+1], v[3*j+2]
                    dist = math.sqrt((xi-xj)**2 + (yi-yj)**2)
                    min_dist = ri + rj
                    if dist < min_dist:
                        penalty += (min_dist - dist) ** 2
            return -total_r + 1e8 * penalty

        result2 = minimize(neg_objective_tight, result.x, method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': iterations, 'ftol': 1e-15})
        optimized = vec_to_pack(result2.x)

    return optimized


def multi_start_packing(n: int, starts: int = 20) -> list[tuple[float, float, float]]:
    """Multi-start optimization: try many random initializations."""
    best_circles = None
    best_metric = -1.0

    for s in range(starts):
        random.seed(42 + s)
        # Random initialization
        circles = []
        for _ in range(n):
            r = random.uniform(0.02, 0.15)
            x = random.uniform(r, 1 - r)
            y = random.uniform(r, 1 - r)
            circles.append((x, y, r))

        optimized = optimize_packing(circles, iterations=3000)
        result = validate_packing(optimized)

        if result["valid"] and result["metric"] > best_metric:
            best_metric = result["metric"]
            best_circles = optimized
            print(f"    start {s}: metric={result['metric']:.6f} (new best)")
        elif result["valid"]:
            print(f"    start {s}: metric={result['metric']:.6f}")
        else:
            print(f"    start {s}: invalid (violation={result['max_violation']:.2e})")

    if best_circles is None:
        # Fallback to grid
        print("  WARNING: all starts invalid, falling back to grid")
        best_circles = grid_packing(n)

    return best_circles


def main():
    print("Generating baseline solutions...\n")

    print("Trivial solutions:")
    save("trivial_n1", trivial_n1())
    save("trivial_n4", trivial_n4())

    print("\nInvalid solutions (for sanity checks):")
    save("bad_overlap", bad_overlap())
    save("bad_outside", bad_outside())

    print("\nGrid baselines:")
    grid10 = grid_packing(10)
    save("grid_n10", grid10)
    grid26 = grid_packing(26)
    save("grid_n26", grid26)

    print("\nOptimized baselines (multi-start):")
    print("  n=10:")
    opt10 = multi_start_packing(10, starts=10)
    res10 = save("baseline_n10", opt10)

    print("  n=26:")
    opt26 = multi_start_packing(26, starts=20)
    res26 = save("baseline_n26", opt26)

    print("\nDone! Baseline summary:")
    print(f"  n=10 best: {res10['metric']:.6f}")
    print(f"  n=26 best: {res26['metric']:.6f}")
    print(f"  n=26 SOTA: ~2.6359 (gap to close)")


if __name__ == "__main__":
    main()
