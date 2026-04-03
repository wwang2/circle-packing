"""
Aggressive topology search for n=32 circle packing.
The current best (2.9365) is at a rigid local optimum.
We need to find a DIFFERENT contact graph topology.

Strategy:
- Generate 500+ diverse initializations
- Polish each with SLSQP
- Track all distinct local optima
"""

import json
import math
import numpy as np
from scipy.optimize import minimize
from pathlib import Path
import sys
import time

WORKDIR = Path(__file__).parent
N = 32
TOL = 1e-10


def save_solution(circles, path):
    data = {"circles": circles.tolist()}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_solution(path):
    with open(path) as f:
        data = json.load(f)
    return np.array(data.get("circles", data))


def validate(circles):
    n = len(circles)
    max_viol = 0.0
    for i in range(n):
        x, y, r = circles[i]
        if r <= 0:
            return False, 0.0, abs(r)
        for v in [r - x, x + r - 1.0, r - y, y + r - 1.0]:
            if v > TOL:
                max_viol = max(max_viol, v)
    for i in range(n):
        xi, yi, ri = circles[i]
        for j in range(i+1, n):
            xj, yj, rj = circles[j]
            dist = math.sqrt((xi-xj)**2 + (yi-yj)**2)
            overlap = ri + rj - dist
            if overlap > TOL:
                max_viol = max(max_viol, overlap)
    valid = max_viol <= TOL
    metric = sum(c[2] for c in circles) if valid else 0.0
    return valid, metric, max_viol


def sum_radii_neg(x):
    n = len(x) // 3
    return -sum(x[3*i+2] for i in range(n))


def constraints_list(n):
    cons = []
    for i in range(n):
        ix, iy, ir = 3*i, 3*i+1, 3*i+2
        cons.append({"type": "ineq", "fun": lambda x, ix=ix, ir=ir: x[ix] - x[ir]})
        cons.append({"type": "ineq", "fun": lambda x, ix=ix, ir=ir: 1.0 - x[ix] - x[ir]})
        cons.append({"type": "ineq", "fun": lambda x, iy=iy, ir=ir: x[iy] - x[ir]})
        cons.append({"type": "ineq", "fun": lambda x, iy=iy, ir=ir: 1.0 - x[iy] - x[ir]})
        cons.append({"type": "ineq", "fun": lambda x, ir=ir: x[ir] - 1e-12})
    for i in range(n):
        for j in range(i+1, n):
            ix, iy, ir = 3*i, 3*i+1, 3*i+2
            jx, jy, jr = 3*j, 3*j+1, 3*j+2
            cons.append({
                "type": "ineq",
                "fun": lambda x, ix=ix, iy=iy, ir=ir, jx=jx, jy=jy, jr=jr:
                    math.sqrt((x[ix]-x[jx])**2 + (x[iy]-x[jy])**2) - x[ir] - x[jr]
            })
    return cons


CONS = constraints_list(N)


def run_slsqp(circles, ftol=1e-14, maxiter=15000):
    x0 = circles.flatten()
    result = minimize(sum_radii_neg, x0, method="SLSQP",
                     constraints=CONS,
                     options={"ftol": ftol, "maxiter": maxiter, "disp": False})
    new_circles = result.x.reshape(-1, 3)
    valid, metric, viol = validate(new_circles)
    return new_circles, valid, metric


def greedy_constructive(rng, n=32, n_candidates=500):
    """Greedy constructive heuristic: place circles one at a time in largest gap."""
    circles = []
    for i in range(n):
        best_r = 0
        best_pos = None

        for _ in range(n_candidates):
            cx = rng.uniform(0.02, 0.98)
            cy = rng.uniform(0.02, 0.98)
            max_r = min(cx, 1-cx, cy, 1-cy)

            for (xk, yk, rk) in circles:
                dist = math.sqrt((cx-xk)**2 + (cy-yk)**2)
                max_r = min(max_r, dist - rk)

            if max_r > best_r:
                best_r = max_r
                best_pos = (cx, cy)

        if best_pos is None or best_r < 0.001:
            return None
        circles.append((best_pos[0], best_pos[1], best_r * 0.999))

    return np.array(circles)


def hex_grid_init(spacing, offset_x=0, offset_y=0, n=32):
    """Hexagonal grid initialization."""
    circles = []
    row = 0
    y = spacing/2 + offset_y
    while y < 1.0 and len(circles) < n * 2:
        x_off = (spacing/2 if row % 2 else 0) + offset_x
        x = spacing/2 + x_off
        while x < 1.0 and len(circles) < n * 2:
            r = min(x, 1-x, y, 1-y, spacing * 0.48)
            if r > 0.01:
                circles.append((x, y, r))
            x += spacing
        y += spacing * math.sqrt(3)/2
        row += 1

    if len(circles) < n:
        return None
    # Take the n with largest radii
    circles.sort(key=lambda c: -c[2])
    return np.array(circles[:n])


def ring_init(config, n=32):
    """Concentric ring initialization with given configuration."""
    rings = config  # list of (n_circles, ring_radius, circle_radius, angle_offset)
    circles = []
    for nc, rr, cr, ao in rings:
        if rr == 0:
            circles.append((0.5, 0.5, cr))
        else:
            for k in range(nc):
                angle = 2*math.pi*k/nc + ao
                cx = 0.5 + rr * math.cos(angle)
                cy = 0.5 + rr * math.sin(angle)
                r = min(cr, cx, 1-cx, cy, 1-cy)
                circles.append((cx, cy, max(r, 0.01)))
    assert len(circles) == n
    return np.array(circles)


def perturbed_solution(base, rng, sigma=0.03):
    """Create a perturbed version of a solution."""
    perturbed = base.copy()
    n = len(base)
    # Perturb random subset
    k = rng.randint(2, min(10, n))
    indices = rng.choice(n, k, replace=False)
    for idx in indices:
        perturbed[idx, 0] += rng.normal(0, sigma)
        perturbed[idx, 1] += rng.normal(0, sigma)
        perturbed[idx, 2] *= (1 + rng.normal(0, sigma*0.5))
    # Clip
    for i in range(n):
        perturbed[i, 2] = max(perturbed[i, 2], 0.005)
        perturbed[i, 0] = np.clip(perturbed[i, 0], perturbed[i, 2]+0.001, 1-perturbed[i, 2]-0.001)
        perturbed[i, 1] = np.clip(perturbed[i, 1], perturbed[i, 2]+0.001, 1-perturbed[i, 2]-0.001)
    return perturbed


def generate_diverse_starts(base_solution, n_starts=500, seed=42):
    """Generate diverse starting configurations."""
    rng = np.random.RandomState(seed)
    starts = []

    # 1. Greedy constructive (100 starts)
    print("Generating greedy constructive starts...")
    for i in range(100):
        init = greedy_constructive(np.random.RandomState(seed + i), n=N, n_candidates=800)
        if init is not None:
            starts.append(("greedy", init))

    # 2. Hex grid variations (30 starts)
    print("Generating hex grid starts...")
    for spacing in np.linspace(0.14, 0.22, 10):
        for ox in [0, 0.02, -0.02]:
            init = hex_grid_init(spacing, offset_x=ox, offset_y=ox, n=N)
            if init is not None:
                starts.append(("hex", init))

    # 3. Ring patterns (40 starts)
    print("Generating ring pattern starts...")
    ring_configs = [
        # (n_circles, ring_radius, circle_radius, angle_offset)
        [(1, 0, 0.09, 0), (7, 0.18, 0.07, 0), (11, 0.33, 0.065, 0.15), (13, 0.44, 0.05, 0)],
        [(1, 0, 0.09, 0), (8, 0.19, 0.068, 0), (12, 0.35, 0.06, 0.13), (11, 0.44, 0.048, 0)],
        [(1, 0, 0.08, 0), (6, 0.16, 0.07, 0), (11, 0.31, 0.063, 0.14), (14, 0.44, 0.048, 0)],
        [(1, 0, 0.08, 0), (8, 0.20, 0.07, 0), (10, 0.34, 0.062, 0.16), (13, 0.45, 0.047, 0)],
        [(1, 0, 0.09, 0), (9, 0.20, 0.065, 0), (11, 0.35, 0.06, 0.14), (11, 0.46, 0.045, 0)],
        [(1, 0, 0.085, 0), (7, 0.17, 0.072, 0), (12, 0.33, 0.06, 0.13), (12, 0.45, 0.047, 0)],
    ]
    for config in ring_configs:
        for angle_shift in np.linspace(0, 0.5, 7):
            shifted = [(nc, rr, cr, ao + angle_shift) for nc, rr, cr, ao in config]
            try:
                init = ring_init(shifted, N)
                starts.append(("ring", init))
            except:
                pass

    # 4. Perturbations of best known (200 starts)
    print("Generating perturbation starts...")
    for i in range(200):
        sigma = rng.choice([0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15])
        init = perturbed_solution(base_solution, rng, sigma=sigma)
        starts.append(("perturb", init))

    # 5. Random uniform (80 starts)
    print("Generating random uniform starts...")
    for i in range(80):
        init = greedy_constructive(np.random.RandomState(seed + 1000 + i), n=N, n_candidates=300)
        if init is not None:
            starts.append(("random", init))

    # 6. Symmetric patterns
    print("Generating symmetric starts...")
    # 4-fold symmetric: 8 circles per quadrant
    for trial in range(20):
        quarter = greedy_constructive(np.random.RandomState(seed + 2000 + trial), n=8, n_candidates=500)
        if quarter is None:
            continue
        circles = []
        for x, y, r in quarter:
            # Scale to quarter
            sx, sy = x * 0.5, y * 0.5
            sr = r * 0.5 * 0.9
            # 4-fold reflection
            circles.extend([
                (sx, sy, sr),
                (1-sx, sy, sr),
                (sx, 1-sy, sr),
                (1-sx, 1-sy, sr),
            ])
        if len(circles) == N:
            starts.append(("sym4", np.array(circles)))

    print(f"Total starts: {len(starts)}")
    return starts


def main():
    print("=" * 60)
    print("Topology Search for N=32")
    print("=" * 60)

    base = load_solution(WORKDIR / "solution_n32_initial.json")
    _, base_metric, _ = validate(base)
    print(f"Base metric: {base_metric:.10f}")

    best_circles = base.copy()
    best_metric = base_metric

    starts = generate_diverse_starts(base, n_starts=500, seed=42)

    # Track all valid optima
    optima = []
    t0 = time.time()

    for idx, (stype, init) in enumerate(starts):
        polished, valid, metric = run_slsqp(init)
        if valid:
            optima.append((metric, stype, idx))
            if metric > best_metric:
                best_metric = metric
                best_circles = polished.copy()
                save_solution(best_circles, WORKDIR / "solution_n32_topo_best.json")
                elapsed = time.time() - t0
                print(f"  *** NEW BEST at start {idx} ({stype}): {metric:.10f} (elapsed {elapsed:.0f}s)")

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            n_valid = len(optima)
            print(f"  Progress: {idx+1}/{len(starts)}, valid={n_valid}, best={best_metric:.10f}, elapsed={elapsed:.0f}s")
            sys.stdout.flush()

    # Summary
    print(f"\n{'='*60}")
    print(f"Topology Search Complete")
    print(f"Total starts: {len(starts)}")
    print(f"Valid solutions: {len(optima)}")
    print(f"Best metric: {best_metric:.10f}")

    # Show top 10 distinct optima
    optima.sort(key=lambda x: -x[0])
    seen = set()
    print("\nTop distinct optima:")
    for metric, stype, idx in optima:
        key = round(metric, 6)
        if key not in seen:
            seen.add(key)
            print(f"  {metric:.10f} ({stype}, start {idx})")
            if len(seen) >= 10:
                break

    save_solution(best_circles, WORKDIR / "solution_n32.json")
    print(f"\nFinal: {best_metric:.10f}")

    return best_metric


if __name__ == "__main__":
    main()
