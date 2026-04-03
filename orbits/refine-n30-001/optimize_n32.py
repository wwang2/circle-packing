"""
Comprehensive optimization for n=32 circle packing.
Strategy:
1. SLSQP refinement with very tight tolerances
2. Single-circle repositioning
3. Basin-hopping with SLSQP local minimizer
4. Multi-start from structured patterns
5. CMA-ES refinement
"""

import json
import math
import numpy as np
from scipy.optimize import minimize, basinhopping
from pathlib import Path
import sys
import time

WORKDIR = Path(__file__).parent
N = 32
TOL = 1e-10


def load_solution(path):
    with open(path) as f:
        data = json.load(f)
    circles = data.get("circles", data)
    return np.array(circles)


def save_solution(circles, path):
    data = {"circles": circles.tolist()}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def sum_radii(x):
    """Negative sum of radii (for minimization)."""
    n = len(x) // 3
    return -sum(x[3*i+2] for i in range(n))


def constraints_list(n):
    """Build scipy constraint dicts for SLSQP."""
    cons = []
    # Containment: r <= x, x <= 1-r, r <= y, y <= 1-r
    for i in range(n):
        ix, iy, ir = 3*i, 3*i+1, 3*i+2
        # x - r >= 0
        cons.append({"type": "ineq", "fun": lambda x, ix=ix, ir=ir: x[ix] - x[ir]})
        # 1 - x - r >= 0
        cons.append({"type": "ineq", "fun": lambda x, ix=ix, ir=ir: 1.0 - x[ix] - x[ir]})
        # y - r >= 0
        cons.append({"type": "ineq", "fun": lambda x, iy=iy, ir=ir: x[iy] - x[ir]})
        # 1 - y - r >= 0
        cons.append({"type": "ineq", "fun": lambda x, iy=iy, ir=ir: 1.0 - x[iy] - x[ir]})
        # r > 0
        cons.append({"type": "ineq", "fun": lambda x, ir=ir: x[ir] - 1e-12})

    # Non-overlap: dist(i,j) >= r_i + r_j
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


def validate(circles):
    """Quick validation. Returns (valid, metric, max_violation)."""
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


def circles_to_x(circles):
    return circles.flatten()


def x_to_circles(x):
    return x.reshape(-1, 3)


def run_slsqp(circles, ftol=1e-15, maxiter=10000):
    """Run SLSQP refinement."""
    x0 = circles_to_x(circles)
    n = len(circles)
    cons = constraints_list(n)

    result = minimize(
        sum_radii, x0, method="SLSQP",
        constraints=cons,
        options={"ftol": ftol, "maxiter": maxiter, "disp": False}
    )

    new_circles = x_to_circles(result.x)
    valid, metric, viol = validate(new_circles)
    return new_circles, valid, metric


def penalty_objective(x, alpha=1e4):
    """Penalty-based objective for unconstrained optimizers."""
    n = len(x) // 3
    obj = 0.0
    penalty = 0.0

    for i in range(n):
        xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
        obj -= ri
        # Containment penalties
        for v in [ri - xi, xi + ri - 1.0, ri - yi, yi + ri - 1.0]:
            if v > 0:
                penalty += v**2
        if ri < 0:
            penalty += ri**2

    # Non-overlap penalties
    for i in range(n):
        xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
        for j in range(i+1, n):
            xj, yj, rj = x[3*j], x[3*j+1], x[3*j+2]
            dist = math.sqrt((xi-xj)**2 + (yi-yj)**2)
            overlap = ri + rj - dist
            if overlap > 0:
                penalty += overlap**2

    return obj + alpha * penalty


def run_basin_hopping(circles, seed=42, niter=200, T=0.01):
    """Basin-hopping with penalty + SLSQP polish."""
    x0 = circles_to_x(circles)
    n = len(circles)
    rng = np.random.RandomState(seed)

    best_circles = circles.copy()
    best_valid = True
    _, best_metric, _ = validate(circles)

    class StepTaker:
        def __init__(self, stepsize=0.02):
            self.stepsize = stepsize
        def __call__(self, x):
            x_new = x + rng.normal(0, self.stepsize, size=x.shape)
            # Keep radii positive
            for i in range(n):
                x_new[3*i+2] = max(x_new[3*i+2], 0.005)
            return x_new

    def callback(x, f, accepted):
        nonlocal best_circles, best_metric
        c = x_to_circles(x)
        valid, metric, viol = validate(c)
        if valid and metric > best_metric:
            best_metric = metric
            best_circles = c.copy()
            print(f"  BH improved: {metric:.10f}")

    result = basinhopping(
        lambda x: penalty_objective(x, alpha=1e5),
        x0, niter=niter, T=T,
        take_step=StepTaker(0.02),
        callback=callback,
        seed=int(seed),
        minimizer_kwargs={
            "method": "L-BFGS-B",
            "options": {"maxiter": 500}
        }
    )

    # Polish best with SLSQP
    if best_metric > 0:
        polished, valid, metric = run_slsqp(best_circles)
        if valid and metric > best_metric:
            best_circles = polished
            best_metric = metric

    return best_circles, best_metric


def single_circle_reposition(circles, n_grid=50):
    """For each circle, remove it, optimize the rest, find best reinsert."""
    n = len(circles)
    best_circles = circles.copy()
    _, best_metric, _ = validate(circles)

    for idx in range(n):
        # Remove circle idx
        remaining = np.delete(circles, idx, axis=0)

        # Grid search for best position for a new circle
        best_r = 0
        best_pos = None

        for gx in range(n_grid+1):
            for gy in range(n_grid+1):
                cx = gx / n_grid
                cy = gy / n_grid

                # Max radius at this position
                max_r = min(cx, 1-cx, cy, 1-cy)
                if max_r < 0.01:
                    continue

                # Check distance to all other circles
                for k in range(len(remaining)):
                    xk, yk, rk = remaining[k]
                    dist = math.sqrt((cx-xk)**2 + (cy-yk)**2)
                    max_r = min(max_r, dist - rk)

                if max_r > best_r:
                    best_r = max_r
                    best_pos = (cx, cy)

        if best_pos is not None and best_r > 0.01:
            new_circles = np.vstack([remaining, [[best_pos[0], best_pos[1], best_r]]])
            # Polish with SLSQP
            polished, valid, metric = run_slsqp(new_circles)
            if valid and metric > best_metric:
                best_metric = metric
                best_circles = polished.copy()
                print(f"  Reposition circle {idx}: {metric:.10f}")

    return best_circles, best_metric


def multi_start_slsqp(n_starts=100, seed_base=0):
    """Generate diverse initial configurations and polish with SLSQP."""
    best_circles = None
    best_metric = 0
    rng = np.random.RandomState(seed_base)

    for trial in range(n_starts):
        circles = generate_random_packing(N, rng)
        if circles is None:
            continue

        polished, valid, metric = run_slsqp(circles)
        if valid and metric > best_metric:
            best_metric = metric
            best_circles = polished.copy()
            print(f"  Multi-start {trial}: {metric:.10f}")

    return best_circles, best_metric


def generate_random_packing(n, rng):
    """Generate a random valid packing using greedy placement."""
    circles = []
    for i in range(n):
        best_r = 0
        best_pos = None

        for _ in range(200):
            cx = rng.uniform(0.05, 0.95)
            cy = rng.uniform(0.05, 0.95)
            max_r = min(cx, 1-cx, cy, 1-cy)

            for (xk, yk, rk) in circles:
                dist = math.sqrt((cx-xk)**2 + (cy-yk)**2)
                max_r = min(max_r, dist - rk)

            if max_r > best_r:
                best_r = max_r
                best_pos = (cx, cy)

        if best_pos is None or best_r < 0.001:
            return None
        circles.append((best_pos[0], best_pos[1], best_r))

    return np.array(circles)


def generate_ring_packing(n):
    """Generate concentric ring pattern for n circles."""
    configs = [
        # (center, inner_ring, middle_ring, outer_ring)
        (1, 7, 11, 13),   # 1+7+11+13 = 32
        (1, 8, 12, 11),   # 1+8+12+11 = 32
        (1, 6, 11, 14),   # 1+6+11+14 = 32
        (0, 8, 12, 12),   # 0+8+12+12 = 32
        (1, 8, 10, 13),   # 1+8+10+13 = 32
        (1, 9, 11, 11),   # 1+9+11+11 = 32
        (4, 8, 12, 8),    # 4+8+12+8 = 32
        (1, 7, 12, 12),   # 1+7+12+12 = 32
    ]

    results = []
    for nc, n1, n2, n3 in configs:
        circles = []

        # Center circle(s)
        if nc == 1:
            circles.append((0.5, 0.5, 0.08))
        elif nc == 4:
            for dx, dy in [(0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7)]:
                circles.append((dx, dy, 0.06))

        # Inner ring
        r1 = 0.18
        for k in range(n1):
            angle = 2 * math.pi * k / n1
            cx = 0.5 + r1 * math.cos(angle)
            cy = 0.5 + r1 * math.sin(angle)
            circles.append((cx, cy, 0.07))

        # Middle ring
        r2 = 0.33
        for k in range(n2):
            angle = 2 * math.pi * k / n2 + math.pi / n2
            cx = 0.5 + r2 * math.cos(angle)
            cy = 0.5 + r2 * math.sin(angle)
            circles.append((cx, cy, 0.06))

        # Outer ring
        r3 = 0.44
        for k in range(n3):
            angle = 2 * math.pi * k / n3
            cx = 0.5 + r3 * math.cos(angle)
            cy = 0.5 + r3 * math.sin(angle)
            circles.append((cx, cy, 0.05))

        assert len(circles) == n, f"Expected {n}, got {len(circles)}"
        results.append(np.array(circles))

    return results


def generate_grid_packings(n):
    """Generate grid-based initial packings."""
    results = []

    # Try various grid arrangements
    for rows in range(4, 8):
        cols_base = n // rows
        remainder = n % rows

        circles = []
        idx = 0
        for row in range(rows):
            cols = cols_base + (1 if row < remainder else 0)
            for col in range(cols):
                if idx >= n:
                    break
                cx = (col + 0.5) / max(cols, 1)
                cy = (row + 0.5) / rows
                r = min(0.5/max(cols,1), 0.5/rows) * 0.9
                r = min(r, cx, 1-cx, cy, 1-cy)
                circles.append((cx, cy, max(r, 0.01)))
                idx += 1

        if len(circles) == n:
            results.append(np.array(circles))

    # Hex grid
    for spacing in [0.16, 0.17, 0.18, 0.19, 0.20]:
        circles = []
        row = 0
        y = spacing
        while y < 1.0 - spacing/2 and len(circles) < n:
            x = spacing + (spacing/2 if row % 2 else 0)
            while x < 1.0 - spacing/2 and len(circles) < n:
                circles.append((x, y, spacing * 0.45))
                x += spacing
            y += spacing * math.sqrt(3)/2
            row += 1

        if len(circles) >= n:
            circles = circles[:n]
            results.append(np.array(circles))

    return results


def perturb_and_optimize(circles, n_tries=50, seed=42):
    """Perturb the solution and re-optimize multiple times."""
    rng = np.random.RandomState(seed)
    best_circles = circles.copy()
    _, best_metric, _ = validate(circles)

    for trial in range(n_tries):
        # Random perturbation
        sigma = rng.choice([0.001, 0.005, 0.01, 0.02, 0.05])
        perturbed = circles.copy()

        # Perturb a subset of circles
        n_perturb = rng.randint(1, min(8, N) + 1)
        indices = rng.choice(N, n_perturb, replace=False)

        for idx in indices:
            perturbed[idx, 0] += rng.normal(0, sigma)
            perturbed[idx, 1] += rng.normal(0, sigma)
            perturbed[idx, 2] *= (1 + rng.normal(0, sigma))

        # Clip to valid range
        for i in range(N):
            perturbed[i, 2] = max(perturbed[i, 2], 0.005)
            perturbed[i, 0] = np.clip(perturbed[i, 0], perturbed[i, 2], 1 - perturbed[i, 2])
            perturbed[i, 1] = np.clip(perturbed[i, 1], perturbed[i, 2], 1 - perturbed[i, 2])

        polished, valid, metric = run_slsqp(perturbed)
        if valid and metric > best_metric:
            best_metric = metric
            best_circles = polished.copy()
            print(f"  Perturb trial {trial} (sigma={sigma}): {metric:.10f}")

    return best_circles, best_metric


def swap_and_optimize(circles, n_tries=100, seed=42):
    """Swap pairs of circles and re-optimize."""
    rng = np.random.RandomState(seed)
    best_circles = circles.copy()
    _, best_metric, _ = validate(circles)

    for trial in range(n_tries):
        swapped = circles.copy()
        i, j = rng.choice(N, 2, replace=False)
        # Swap positions but keep radii
        swapped[i, 0], swapped[j, 0] = swapped[j, 0], swapped[i, 0]
        swapped[i, 1], swapped[j, 1] = swapped[j, 1], swapped[i, 1]

        polished, valid, metric = run_slsqp(swapped)
        if valid and metric > best_metric:
            best_metric = metric
            best_circles = polished.copy()
            print(f"  Swap {i},{j} trial {trial}: {metric:.10f}")

    return best_circles, best_metric


def subgroup_optimize(circles, group_size=5, n_iters=50, seed=42):
    """Optimize subgroups of circles while keeping others fixed."""
    rng = np.random.RandomState(seed)
    best_circles = circles.copy()
    _, best_metric, _ = validate(circles)

    for iteration in range(n_iters):
        indices = rng.choice(N, group_size, replace=False)

        # Build sub-problem
        fixed = [i for i in range(N) if i not in indices]

        def sub_objective(params):
            full = best_circles.copy()
            for k, idx in enumerate(indices):
                full[idx] = params[3*k:3*k+3]
            return -sum(full[i, 2] for i in range(N))

        def sub_constraints():
            cons = []
            for k, idx in enumerate(indices):
                ki = 3*k
                # Containment
                cons.append({"type": "ineq", "fun": lambda x, ki=ki: x[ki] - x[ki+2]})
                cons.append({"type": "ineq", "fun": lambda x, ki=ki: 1 - x[ki] - x[ki+2]})
                cons.append({"type": "ineq", "fun": lambda x, ki=ki: x[ki+1] - x[ki+2]})
                cons.append({"type": "ineq", "fun": lambda x, ki=ki: 1 - x[ki+1] - x[ki+2]})
                cons.append({"type": "ineq", "fun": lambda x, ki=ki: x[ki+2] - 1e-12})

                # Non-overlap with fixed circles
                for fi in fixed:
                    fx, fy, fr = best_circles[fi]
                    cons.append({
                        "type": "ineq",
                        "fun": lambda x, ki=ki, fx=fx, fy=fy, fr=fr:
                            math.sqrt((x[ki]-fx)**2 + (x[ki+1]-fy)**2) - x[ki+2] - fr
                    })

                # Non-overlap with other optimized circles
                for k2, idx2 in enumerate(indices):
                    if k2 > k:
                        ki2 = 3*k2
                        cons.append({
                            "type": "ineq",
                            "fun": lambda x, ki=ki, ki2=ki2:
                                math.sqrt((x[ki]-x[ki2])**2 + (x[ki+1]-x[ki2+1])**2) - x[ki+2] - x[ki2+2]
                        })
            return cons

        x0 = np.concatenate([best_circles[idx] for idx in indices])
        result = minimize(sub_objective, x0, method="SLSQP",
                         constraints=sub_constraints(),
                         options={"ftol": 1e-15, "maxiter": 5000})

        candidate = best_circles.copy()
        for k, idx in enumerate(indices):
            candidate[idx] = result.x[3*k:3*k+3]

        valid, metric, viol = validate(candidate)
        if valid and metric > best_metric:
            best_metric = metric
            best_circles = candidate.copy()
            print(f"  Subgroup iter {iteration}: {metric:.10f}")

    return best_circles, best_metric


def main():
    print("=" * 60)
    print("N=32 Circle Packing Optimization")
    print("=" * 60)

    # Load best known solution
    initial = load_solution(WORKDIR / "solution_n32_initial.json")
    valid, metric, viol = validate(initial)
    print(f"\nInitial solution: metric={metric:.10f}, valid={valid}")

    best_circles = initial.copy()
    best_metric = metric

    # Step 1: SLSQP with tight tolerances
    print("\n--- Step 1: SLSQP refinement ---")
    for ftol in [1e-10, 1e-12, 1e-13, 1e-14, 1e-15]:
        refined, valid, metric = run_slsqp(best_circles, ftol=ftol, maxiter=20000)
        if valid and metric > best_metric:
            best_metric = metric
            best_circles = refined.copy()
            print(f"  SLSQP ftol={ftol}: {metric:.10f}")
        else:
            print(f"  SLSQP ftol={ftol}: no improvement (metric={metric:.10f}, valid={valid})")

    save_solution(best_circles, WORKDIR / "solution_n32_slsqp.json")
    print(f"After SLSQP: {best_metric:.10f}")

    # Step 2: Perturbation + SLSQP
    print("\n--- Step 2: Perturbation + SLSQP ---")
    perturbed, metric = perturb_and_optimize(best_circles, n_tries=80, seed=42)
    if metric > best_metric:
        best_metric = metric
        best_circles = perturbed.copy()
    save_solution(best_circles, WORKDIR / "solution_n32_perturb.json")
    print(f"After perturbation: {best_metric:.10f}")

    # Step 3: Single-circle repositioning
    print("\n--- Step 3: Single-circle repositioning ---")
    repositioned, metric = single_circle_reposition(best_circles, n_grid=80)
    if metric > best_metric:
        best_metric = metric
        best_circles = repositioned.copy()
    save_solution(best_circles, WORKDIR / "solution_n32_reposition.json")
    print(f"After repositioning: {best_metric:.10f}")

    # Step 4: Swap optimization
    print("\n--- Step 4: Swap optimization ---")
    swapped, metric = swap_and_optimize(best_circles, n_tries=100, seed=42)
    if metric > best_metric:
        best_metric = metric
        best_circles = swapped.copy()
    print(f"After swap: {best_metric:.10f}")

    # Step 5: Subgroup optimization
    print("\n--- Step 5: Subgroup optimization ---")
    for gs in [3, 4, 5, 6]:
        subopt, metric = subgroup_optimize(best_circles, group_size=gs, n_iters=30, seed=42)
        if metric > best_metric:
            best_metric = metric
            best_circles = subopt.copy()
    save_solution(best_circles, WORKDIR / "solution_n32_subgroup.json")
    print(f"After subgroup: {best_metric:.10f}")

    # Step 6: Basin-hopping
    print("\n--- Step 6: Basin-hopping ---")
    for seed in [42, 123, 456, 789, 1337]:
        bh_result, metric = run_basin_hopping(best_circles, seed=seed, niter=150, T=0.005)
        if metric > best_metric:
            best_metric = metric
            best_circles = bh_result.copy()
    save_solution(best_circles, WORKDIR / "solution_n32_bh.json")
    print(f"After basin-hopping: {best_metric:.10f}")

    # Step 7: Multi-start from ring patterns
    print("\n--- Step 7: Ring pattern starts ---")
    ring_packings = generate_ring_packing(N)
    for idx, rp in enumerate(ring_packings):
        polished, valid, metric = run_slsqp(rp, ftol=1e-13, maxiter=20000)
        if valid and metric > best_metric:
            best_metric = metric
            best_circles = polished.copy()
            print(f"  Ring pattern {idx}: {metric:.10f}")
        elif valid:
            print(f"  Ring pattern {idx}: {metric:.10f} (not better)")
        else:
            print(f"  Ring pattern {idx}: invalid")
    print(f"After ring patterns: {best_metric:.10f}")

    # Step 8: Grid pattern starts
    print("\n--- Step 8: Grid pattern starts ---")
    grid_packings = generate_grid_packings(N)
    for idx, gp in enumerate(grid_packings):
        polished, valid, metric = run_slsqp(gp, ftol=1e-13, maxiter=20000)
        if valid and metric > best_metric:
            best_metric = metric
            best_circles = polished.copy()
            print(f"  Grid pattern {idx}: {metric:.10f}")
        elif valid:
            print(f"  Grid pattern {idx}: {metric:.10f} (not better)")
        else:
            print(f"  Grid pattern {idx}: invalid")
    print(f"After grid patterns: {best_metric:.10f}")

    # Step 9: Random multi-start
    print("\n--- Step 9: Random multi-start (100 starts) ---")
    ms_result, metric = multi_start_slsqp(n_starts=100, seed_base=42)
    if ms_result is not None and metric > best_metric:
        best_metric = metric
        best_circles = ms_result.copy()
    print(f"After multi-start: {best_metric:.10f}")

    # Final save
    save_solution(best_circles, WORKDIR / "solution_n32.json")
    print(f"\n{'=' * 60}")
    print(f"FINAL RESULT: {best_metric:.10f}")
    print(f"{'=' * 60}")

    return best_metric


if __name__ == "__main__":
    main()
