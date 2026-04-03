"""CMA-ES optimizer for circle packing in unit square.

Strategy:
1. Warm-start from parent sa-001 solution
2. Use CMA-ES with various sigma values to explore/exploit
3. Penalty-based constraint handling
4. SLSQP polish after CMA-ES convergence
"""

import json
import math
import sys
import os
import numpy as np
import cma
from scipy.optimize import minimize

WORKDIR = os.path.dirname(os.path.abspath(__file__))


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
    """Flatten circles array to 1D vector [x0,y0,r0, x1,y1,r1, ...]."""
    return circles.flatten()


def x_to_circles(x, n):
    """Reshape 1D vector to (n, 3) circles array."""
    return x.reshape(n, 3)


def compute_violation(circles):
    """Compute total constraint violation."""
    n = len(circles)
    total_viol = 0.0
    max_viol = 0.0

    for i in range(n):
        x, y, r = circles[i]
        # Containment
        for v in [r - x, x + r - 1.0, r - y, y + r - 1.0, -r]:
            if v > 0:
                total_viol += v
                max_viol = max(max_viol, v)

    # Non-overlap
    for i in range(n):
        xi, yi, ri = circles[i]
        for j in range(i + 1, n):
            xj, yj, rj = circles[j]
            dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)
            overlap = (ri + rj) - dist
            if overlap > 0:
                total_viol += overlap
                max_viol = max(max_viol, overlap)

    return total_viol, max_viol


def objective(x, n, penalty_weight):
    """CMA-ES objective: minimize -(sum_radii) + penalty * violation."""
    circles = x_to_circles(x, n)
    sum_radii = np.sum(circles[:, 2])
    total_viol, _ = compute_violation(circles)
    return -sum_radii + penalty_weight * total_viol


def slsqp_polish(circles, maxiter=5000):
    """Polish solution with SLSQP."""
    n = len(circles)
    x0 = circles_to_x(circles)

    def neg_sum_radii(x):
        return -np.sum(x.reshape(n, 3)[:, 2])

    constraints = []
    # Containment
    for i in range(n):
        ri = 3 * i + 2
        xi = 3 * i
        yi = 3 * i + 1
        # r > 0
        constraints.append({"type": "ineq", "fun": lambda x, idx=ri: x[idx] - 1e-12})
        # x - r >= 0
        constraints.append({"type": "ineq", "fun": lambda x, ix=xi, ir=ri: x[ix] - x[ir]})
        # 1 - x - r >= 0
        constraints.append({"type": "ineq", "fun": lambda x, ix=xi, ir=ri: 1.0 - x[ix] - x[ir]})
        # y - r >= 0
        constraints.append({"type": "ineq", "fun": lambda x, iy=yi, ir=ri: x[iy] - x[ir]})
        # 1 - y - r >= 0
        constraints.append({"type": "ineq", "fun": lambda x, iy=yi, ir=ri: 1.0 - x[iy] - x[ir]})

    # Non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            def overlap_con(x, ii=i, jj=j):
                xi, yi, ri = x[3*ii], x[3*ii+1], x[3*ii+2]
                xj, yj, rj = x[3*jj], x[3*jj+1], x[3*jj+2]
                return math.sqrt((xi-xj)**2 + (yi-yj)**2) - ri - rj
            constraints.append({"type": "ineq", "fun": overlap_con})

    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])

    res = minimize(neg_sum_radii, x0, method="SLSQP",
                   constraints=constraints, bounds=bounds,
                   options={"maxiter": maxiter, "ftol": 1e-15, "disp": False})

    result_circles = x_to_circles(res.x, n)
    viol, max_v = compute_violation(result_circles)
    sr = np.sum(result_circles[:, 2])

    return result_circles, sr, viol, max_v


def run_cmaes(init_circles, sigma, popsize, maxiter, penalty_schedule, seed=42):
    """Run CMA-ES optimization."""
    n = len(init_circles)
    x0 = circles_to_x(init_circles)
    dim = len(x0)

    best_valid_sr = 0.0
    best_valid_circles = None

    for penalty_weight in penalty_schedule:
        print(f"  CMA-ES: sigma={sigma:.4f}, pop={popsize}, penalty={penalty_weight:.0f}, maxiter={maxiter}")

        opts = cma.CMAOptions()
        opts['seed'] = seed
        opts['maxiter'] = maxiter
        opts['popsize'] = popsize
        opts['verbose'] = -9  # quiet
        opts['tolfun'] = 1e-14
        opts['tolx'] = 1e-14
        # Bounds
        lower = []
        upper = []
        for i in range(n):
            lower.extend([0.001, 0.001, 0.001])
            upper.extend([0.999, 0.999, 0.499])
        opts['bounds'] = [lower, upper]

        es = cma.CMAEvolutionStrategy(x0, sigma, opts)

        gen = 0
        while not es.stop():
            solutions = es.ask()
            fitnesses = [objective(x, n, penalty_weight) for x in solutions]
            es.tell(solutions, fitnesses)
            gen += 1

            if gen % 200 == 0:
                best_x = es.result.xbest
                c = x_to_circles(best_x, n)
                sr = np.sum(c[:, 2])
                v, mv = compute_violation(c)
                print(f"    gen={gen}: sum_r={sr:.10f}, viol={v:.2e}, max_v={mv:.2e}")

        best_x = es.result.xbest
        c = x_to_circles(best_x, n)
        sr = np.sum(c[:, 2])
        v, mv = compute_violation(c)
        print(f"  CMA-ES done: sum_r={sr:.10f}, viol={v:.2e}, max_v={mv:.2e}")

        # Use best as starting point for next penalty level
        x0 = best_x.copy()

        # Track best valid
        if mv < 1e-10 and sr > best_valid_sr:
            best_valid_sr = sr
            best_valid_circles = c.copy()

    return best_valid_circles, best_valid_sr, x0, n


def perturb_and_polish(circles, n_attempts=20, perturb_scale=0.02, seed=12345):
    """Perturb solution and re-polish to escape local optima."""
    rng = np.random.RandomState(seed)
    best_sr = np.sum(circles[:, 2])
    best_circles = circles.copy()

    for attempt in range(n_attempts):
        perturbed = circles.copy()
        # Perturb a random subset of circles
        n_perturb = rng.randint(1, max(2, len(circles) // 3))
        indices = rng.choice(len(circles), n_perturb, replace=False)
        for idx in indices:
            perturbed[idx, 0] += rng.normal(0, perturb_scale)
            perturbed[idx, 1] += rng.normal(0, perturb_scale)
            perturbed[idx, 2] *= (1 + rng.normal(0, perturb_scale * 0.5))

        # Clip to valid range
        perturbed[:, 2] = np.clip(perturbed[:, 2], 0.005, 0.495)
        perturbed[:, 0] = np.clip(perturbed[:, 0], perturbed[:, 2], 1 - perturbed[:, 2])
        perturbed[:, 1] = np.clip(perturbed[:, 1], perturbed[:, 2], 1 - perturbed[:, 2])

        polished, sr, viol, max_v = slsqp_polish(perturbed, maxiter=5000)
        if max_v < 1e-10 and sr > best_sr:
            print(f"  Perturb {attempt}: IMPROVED {best_sr:.10f} -> {sr:.10f}")
            best_sr = sr
            best_circles = polished.copy()
        elif attempt % 5 == 0:
            print(f"  Perturb {attempt}: sr={sr:.10f}, max_v={max_v:.2e} (best={best_sr:.10f})")

    return best_circles, best_sr


def main():
    # Load parent solution
    parent_path = os.path.join(WORKDIR, "..", "sa-001", "solution_n26.json")
    init_circles = load_solution(parent_path)
    n = len(init_circles)
    init_sr = np.sum(init_circles[:, 2])
    print(f"Loaded parent solution: n={n}, sum_radii={init_sr:.10f}")

    best_circles = init_circles.copy()
    best_sr = init_sr

    # === Strategy 1: Small sigma refinement around known optimum ===
    print("\n=== Strategy 1: Small sigma CMA-ES (basin refinement) ===")
    for sigma in [0.005, 0.01, 0.02]:
        print(f"\n--- sigma={sigma} ---")
        valid_circles, valid_sr, raw_x, _ = run_cmaes(
            best_circles, sigma=sigma, popsize=100, maxiter=1000,
            penalty_schedule=[1e4, 1e6], seed=42
        )
        if valid_circles is not None and valid_sr > best_sr:
            best_sr = valid_sr
            best_circles = valid_circles
            print(f"  ** New best (valid): {best_sr:.10f}")

        # Always try to polish the raw CMA-ES output
        raw_circles = x_to_circles(raw_x, n)
        polished, pol_sr, pol_v, pol_mv = slsqp_polish(raw_circles, maxiter=5000)
        print(f"  Polished: sr={pol_sr:.10f}, max_v={pol_mv:.2e}")
        if pol_mv < 1e-10 and pol_sr > best_sr:
            best_sr = pol_sr
            best_circles = polished
            print(f"  ** New best (polished): {best_sr:.10f}")

    # === Strategy 2: Medium sigma exploration ===
    print("\n=== Strategy 2: Medium sigma CMA-ES (nearby basins) ===")
    for sigma in [0.05, 0.1]:
        for seed in [42, 137, 2024]:
            print(f"\n--- sigma={sigma}, seed={seed} ---")
            valid_circles, valid_sr, raw_x, _ = run_cmaes(
                best_circles, sigma=sigma, popsize=200, maxiter=800,
                penalty_schedule=[1e3, 1e5, 1e7], seed=seed
            )
            if valid_circles is not None and valid_sr > best_sr:
                best_sr = valid_sr
                best_circles = valid_circles
                print(f"  ** New best (valid): {best_sr:.10f}")

            raw_circles = x_to_circles(raw_x, n)
            polished, pol_sr, pol_v, pol_mv = slsqp_polish(raw_circles, maxiter=5000)
            print(f"  Polished: sr={pol_sr:.10f}, max_v={pol_mv:.2e}")
            if pol_mv < 1e-10 and pol_sr > best_sr:
                best_sr = pol_sr
                best_circles = polished
                print(f"  ** New best (polished): {best_sr:.10f}")

    # === Strategy 3: Large sigma exploration ===
    print("\n=== Strategy 3: Large sigma CMA-ES (new basins) ===")
    for sigma in [0.2, 0.3]:
        for seed in [42, 99, 314]:
            print(f"\n--- sigma={sigma}, seed={seed} ---")
            valid_circles, valid_sr, raw_x, _ = run_cmaes(
                best_circles, sigma=sigma, popsize=300, maxiter=600,
                penalty_schedule=[1e2, 1e4, 1e6, 1e8], seed=seed
            )
            if valid_circles is not None and valid_sr > best_sr:
                best_sr = valid_sr
                best_circles = valid_circles
                print(f"  ** New best (valid): {best_sr:.10f}")

            raw_circles = x_to_circles(raw_x, n)
            polished, pol_sr, pol_v, pol_mv = slsqp_polish(raw_circles, maxiter=5000)
            print(f"  Polished: sr={pol_sr:.10f}, max_v={pol_mv:.2e}")
            if pol_mv < 1e-10 and pol_sr > best_sr:
                best_sr = pol_sr
                best_circles = polished
                print(f"  ** New best (polished): {best_sr:.10f}")

    # === Strategy 4: Perturb best + re-polish ===
    print("\n=== Strategy 4: Perturbation + Polish ===")
    perturbed, perturbed_sr = perturb_and_polish(best_circles, n_attempts=30, seed=88888)
    if perturbed_sr > best_sr:
        best_sr = perturbed_sr
        best_circles = perturbed
        print(f"  ** New best (perturbed): {best_sr:.10f}")

    # === Final result ===
    print(f"\n{'='*60}")
    print(f"FINAL RESULT: sum_radii = {best_sr:.10f}")
    print(f"Parent was:   sum_radii = {init_sr:.10f}")
    print(f"Improvement:  {best_sr - init_sr:.12f}")

    # Validate
    viol, max_v = compute_violation(best_circles)
    print(f"Violation: total={viol:.2e}, max={max_v:.2e}")

    # Save
    out_path = os.path.join(WORKDIR, "solution_n26.json")
    save_solution(best_circles, out_path)
    print(f"Saved to {out_path}")

    return best_sr


if __name__ == "__main__":
    best = main()
