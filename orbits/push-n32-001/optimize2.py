"""
Focused optimization for n=32 circle packing.
Strategy: High-precision SLSQP with many restarts from small perturbations
of the current best, plus force-directed relaxation.
"""

import json
import math
import numpy as np
from scipy.optimize import minimize
from pathlib import Path
import time

N = 32
BEST_FILE = Path(__file__).parent / "solution_n32.json"
LOG_FILE = Path(__file__).parent / "log.md"

def load_solution(path):
    with open(path) as f:
        data = json.load(f)
    circles = data.get("circles", data)
    return np.array(circles)

def save_solution(circles, path):
    data = {"circles": [[float(x), float(y), float(r)] for x, y, r in circles]}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def circles_to_x(circles):
    return circles.flatten()

def x_to_circles(x):
    return x.reshape(-1, 3)

def sum_radii_neg(x):
    return -np.sum(x.reshape(-1, 3)[:, 2])

def sum_radii_neg_jac(x):
    """Analytic Jacobian of -sum(r_i)."""
    n = len(x) // 3
    jac = np.zeros(len(x))
    for i in range(n):
        jac[3*i + 2] = -1.0  # d/dr_i of -sum(r) = -1
    return jac

def make_constraint_dicts():
    """Create individual constraint dicts with analytic Jacobians."""
    n = N
    constraints = []

    # Containment constraints
    for i in range(n):
        # x_i - r_i >= 0
        def c_left(x, i=i):
            return x[3*i] - x[3*i+2]
        def c_left_jac(x, i=i):
            j = np.zeros(len(x))
            j[3*i] = 1.0; j[3*i+2] = -1.0
            return j
        constraints.append({"type": "ineq", "fun": c_left, "jac": c_left_jac})

        # 1 - x_i - r_i >= 0
        def c_right(x, i=i):
            return 1.0 - x[3*i] - x[3*i+2]
        def c_right_jac(x, i=i):
            j = np.zeros(len(x))
            j[3*i] = -1.0; j[3*i+2] = -1.0
            return j
        constraints.append({"type": "ineq", "fun": c_right, "jac": c_right_jac})

        # y_i - r_i >= 0
        def c_bottom(x, i=i):
            return x[3*i+1] - x[3*i+2]
        def c_bottom_jac(x, i=i):
            j = np.zeros(len(x))
            j[3*i+1] = 1.0; j[3*i+2] = -1.0
            return j
        constraints.append({"type": "ineq", "fun": c_bottom, "jac": c_bottom_jac})

        # 1 - y_i - r_i >= 0
        def c_top(x, i=i):
            return 1.0 - x[3*i+1] - x[3*i+2]
        def c_top_jac(x, i=i):
            j = np.zeros(len(x))
            j[3*i+1] = -1.0; j[3*i+2] = -1.0
            return j
        constraints.append({"type": "ineq", "fun": c_top, "jac": c_top_jac})

        # r_i >= eps
        def c_rpos(x, i=i):
            return x[3*i+2] - 1e-6
        def c_rpos_jac(x, i=i):
            j = np.zeros(len(x))
            j[3*i+2] = 1.0
            return j
        constraints.append({"type": "ineq", "fun": c_rpos, "jac": c_rpos_jac})

    # Non-overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            def c_nooverlap(x, i=i, j=j):
                dx = x[3*i] - x[3*j]
                dy = x[3*i+1] - x[3*j+1]
                dist = math.sqrt(dx*dx + dy*dy)
                return dist - x[3*i+2] - x[3*j+2]

            def c_nooverlap_jac(x, i=i, j=j):
                dx = x[3*i] - x[3*j]
                dy = x[3*i+1] - x[3*j+1]
                dist = math.sqrt(dx*dx + dy*dy)
                jac = np.zeros(len(x))
                if dist > 1e-15:
                    jac[3*i] = dx / dist
                    jac[3*i+1] = dy / dist
                    jac[3*j] = -dx / dist
                    jac[3*j+1] = -dy / dist
                jac[3*i+2] = -1.0
                jac[3*j+2] = -1.0
                return jac

            constraints.append({"type": "ineq", "fun": c_nooverlap, "jac": c_nooverlap_jac})

    return constraints

def make_bounds():
    bounds = []
    for _ in range(N):
        bounds.append((1e-4, 1.0 - 1e-4))
        bounds.append((1e-4, 1.0 - 1e-4))
        bounds.append((1e-6, 0.5))
    return bounds

def is_valid(x, tol=1e-10):
    c = x.reshape(-1, 3)
    for i in range(len(c)):
        xi, yi, ri = c[i]
        if ri <= 0: return False
        if ri - xi > tol: return False
        if xi + ri - 1 > tol: return False
        if ri - yi > tol: return False
        if yi + ri - 1 > tol: return False
    for i in range(len(c)):
        for j in range(i+1, len(c)):
            dx = c[i,0]-c[j,0]; dy = c[i,1]-c[j,1]
            dist = math.sqrt(dx*dx + dy*dy)
            if c[i,2] + c[j,2] - dist > tol:
                return False
    return True

def get_metric(x):
    return np.sum(x.reshape(-1, 3)[:, 2])

def slsqp_optimize(x0, constraints, maxiter=5000):
    bounds = make_bounds()
    result = minimize(
        sum_radii_neg, x0, method='SLSQP', jac=sum_radii_neg_jac,
        bounds=bounds, constraints=constraints,
        options={'maxiter': maxiter, 'ftol': 1e-16, 'disp': False}
    )
    return result.x, -result.fun

def force_directed_step(x, step_size=0.001, iterations=100):
    """Push circles apart using force-directed approach, then grow radii."""
    c = x.reshape(-1, 3).copy()
    n = len(c)

    for _ in range(iterations):
        forces = np.zeros((n, 2))

        # Repulsion from walls
        for i in range(n):
            xi, yi, ri = c[i]
            # Left wall
            gap = xi - ri
            if gap < 0.01:
                forces[i, 0] += 0.01 / max(gap, 1e-6)
            # Right wall
            gap = 1 - xi - ri
            if gap < 0.01:
                forces[i, 0] -= 0.01 / max(gap, 1e-6)
            # Bottom wall
            gap = yi - ri
            if gap < 0.01:
                forces[i, 1] += 0.01 / max(gap, 1e-6)
            # Top wall
            gap = 1 - yi - ri
            if gap < 0.01:
                forces[i, 1] -= 0.01 / max(gap, 1e-6)

        # Repulsion between overlapping circles
        for i in range(n):
            for j in range(i+1, n):
                dx = c[i,0] - c[j,0]
                dy = c[i,1] - c[j,1]
                dist = math.sqrt(dx*dx + dy*dy)
                min_dist = c[i,2] + c[j,2]
                if dist < min_dist + 0.001:
                    if dist < 1e-10:
                        dx, dy, dist = 0.001, 0.001, 0.00141
                    overlap = min_dist - dist + 0.001
                    fx = overlap * dx / dist
                    fy = overlap * dy / dist
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy

        # Apply forces
        for i in range(n):
            c[i, 0] += forces[i, 0] * step_size
            c[i, 1] += forces[i, 1] * step_size
            # Clamp
            c[i, 0] = max(c[i,2] + 1e-4, min(1 - c[i,2] - 1e-4, c[i,0]))
            c[i, 1] = max(c[i,2] + 1e-4, min(1 - c[i,2] - 1e-4, c[i,1]))

        # Try to grow radii
        for i in range(n):
            max_r = min(c[i,0], 1-c[i,0], c[i,1], 1-c[i,1])
            for j in range(n):
                if j == i: continue
                dx = c[i,0] - c[j,0]
                dy = c[i,1] - c[j,1]
                dist = math.sqrt(dx*dx + dy*dy)
                max_r = min(max_r, dist - c[j,2])
            if max_r > c[i,2]:
                c[i,2] = c[i,2] + (max_r - c[i,2]) * 0.1  # Grow slowly

    return c.flatten()

def targeted_perturbation(x, rng, mode='single'):
    """Targeted perturbation of specific circles."""
    c = x.reshape(-1, 3).copy()
    n = len(c)

    if mode == 'single':
        # Perturb one circle
        idx = rng.integers(n)
        c[idx, 0] += rng.normal(0, 0.01)
        c[idx, 1] += rng.normal(0, 0.01)
        c[idx, 0] = max(c[idx,2]+1e-4, min(1-c[idx,2]-1e-4, c[idx,0]))
        c[idx, 1] = max(c[idx,2]+1e-4, min(1-c[idx,2]-1e-4, c[idx,1]))
    elif mode == 'smallest':
        # Perturb the smallest circles more
        radii = c[:, 2]
        idx = np.argsort(radii)[:5]
        for i in idx:
            c[i, 0] += rng.normal(0, 0.02)
            c[i, 1] += rng.normal(0, 0.02)
            c[i, 0] = max(c[i,2]+1e-4, min(1-c[i,2]-1e-4, c[i,0]))
            c[i, 1] = max(c[i,2]+1e-4, min(1-c[i,2]-1e-4, c[i,1]))
    elif mode == 'cluster':
        # Pick a random circle, perturb it and its neighbors
        idx = rng.integers(n)
        cx, cy = c[idx, 0], c[idx, 1]
        dists = np.sqrt((c[:, 0] - cx)**2 + (c[:, 1] - cy)**2)
        neighbors = np.argsort(dists)[:4]
        for i in neighbors:
            c[i, 0] += rng.normal(0, 0.015)
            c[i, 1] += rng.normal(0, 0.015)
            c[i, 0] = max(c[i,2]+1e-4, min(1-c[i,2]-1e-4, c[i,0]))
            c[i, 1] = max(c[i,2]+1e-4, min(1-c[i,2]-1e-4, c[i,1]))
    elif mode == 'radius_shuffle':
        # Slightly randomize radii and let SLSQP fix it
        c[:, 2] *= (1 + rng.normal(0, 0.02, n))
        c[:, 2] = np.maximum(c[:, 2], 0.005)
        for i in range(n):
            max_r = min(c[i,0], 1-c[i,0], c[i,1], 1-c[i,1])
            c[i, 2] = min(c[i, 2], max_r - 1e-4)

    return c.flatten()


def run():
    best_circles = load_solution(BEST_FILE)
    best_x = circles_to_x(best_circles)
    best_metric = get_metric(best_x)
    print(f"Starting metric: {best_metric:.10f}")

    constraints = make_constraint_dicts()
    print(f"Built {len(constraints)} constraints with analytic Jacobians")

    improvements = []

    def try_update(x, source):
        nonlocal best_x, best_metric
        if is_valid(x):
            m = get_metric(x)
            if m > best_metric + 1e-12:
                improvement = m - best_metric
                best_metric = m
                best_x = x.copy()
                save_solution(x_to_circles(x), BEST_FILE)
                improvements.append((source, m))
                print(f"  *** NEW BEST: {m:.10f} (+{improvement:.2e}) from {source}")
                return True
        return False

    # Phase A: Re-optimize current best with analytic Jacobians
    print("\n=== Phase A: Re-optimize with analytic Jacobians ===")
    x_opt, m = slsqp_optimize(best_x, constraints, maxiter=10000)
    try_update(x_opt, "analytic-jac-reopt")

    # Phase B: Force-directed + SLSQP
    print("\n=== Phase B: Force-directed relaxation + SLSQP ===")
    for trial in range(20):
        rng = np.random.default_rng(trial * 43 + 100)
        x0 = best_x.copy()
        # Small random perturbation
        noise = rng.normal(0, 0.005, len(x0))
        # Only perturb positions, not radii
        for i in range(N):
            noise[3*i+2] = 0
        x0 = x0 + noise
        # Clamp
        c = x0.reshape(-1, 3)
        for i in range(N):
            c[i, 0] = max(c[i,2]+1e-4, min(1-c[i,2]-1e-4, c[i,0]))
            c[i, 1] = max(c[i,2]+1e-4, min(1-c[i,2]-1e-4, c[i,1]))
        x0 = c.flatten()

        x0 = force_directed_step(x0, step_size=0.0005, iterations=50)
        x_opt, m = slsqp_optimize(x0, constraints, maxiter=5000)
        try_update(x_opt, f"force-directed-{trial}")

    print(f"\nAfter Phase B: best = {best_metric:.10f}")

    # Phase C: Intensive targeted perturbation
    print("\n=== Phase C: Targeted perturbation ===")
    modes = ['single', 'smallest', 'cluster', 'radius_shuffle']

    for trial in range(500):
        if trial % 100 == 0:
            print(f"  Targeted perturbation trial {trial}/500, best={best_metric:.10f}")
        rng = np.random.default_rng(trial * 13 + 3000)
        mode = modes[trial % len(modes)]
        x0 = targeted_perturbation(best_x, rng, mode=mode)
        x_opt, m = slsqp_optimize(x0, constraints, maxiter=3000)
        try_update(x_opt, f"targeted-{mode}-{trial}")

    print(f"\nAfter Phase C: best = {best_metric:.10f}")

    # Phase D: Coordinate descent on individual circles
    print("\n=== Phase D: Coordinate descent ===")
    for outer in range(5):
        for i in range(N):
            # Optimize just circle i's position while keeping others fixed
            x0 = best_x.copy()
            c = x0.reshape(-1, 3)
            # Try nudging this circle in various directions
            best_local = best_metric
            best_local_x = best_x.copy()
            for dx, dy in [(0.005,0), (-0.005,0), (0,0.005), (0,-0.005),
                           (0.003,0.003), (-0.003,0.003), (0.003,-0.003), (-0.003,-0.003)]:
                x_try = best_x.copy()
                c_try = x_try.reshape(-1, 3)
                c_try[i, 0] += dx
                c_try[i, 1] += dy
                c_try[i, 0] = max(c_try[i,2]+1e-4, min(1-c_try[i,2]-1e-4, c_try[i,0]))
                c_try[i, 1] = max(c_try[i,2]+1e-4, min(1-c_try[i,2]-1e-4, c_try[i,1]))
                x_try = c_try.flatten()
                x_opt, m = slsqp_optimize(x_try, constraints, maxiter=2000)
                if is_valid(x_opt) and get_metric(x_opt) > best_local + 1e-12:
                    best_local = get_metric(x_opt)
                    best_local_x = x_opt.copy()
            try_update(best_local_x, f"coord-descent-outer{outer}-circle{i}")
        print(f"  Coord descent outer {outer}: best = {best_metric:.10f}")

    print(f"\n{'='*60}")
    print(f"FINAL BEST: {best_metric:.10f}")
    print(f"Improvements: {len(improvements)}")
    for src, m in improvements:
        print(f"  {src}: {m:.10f}")

    return best_metric, improvements


if __name__ == "__main__":
    start = time.time()
    best_metric, improvements = run()
    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")

    with open(LOG_FILE, "a") as f:
        f.write(f"\n## Optimize2 run {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"- Final metric: {best_metric:.10f}\n")
        f.write(f"- Time: {elapsed:.1f}s\n")
        f.write(f"- Improvements: {len(improvements)}\n")
        for src, m in improvements:
            f.write(f"  - {src}: {m:.10f}\n")
