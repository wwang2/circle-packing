"""
CMA-ES optimization for n=32 circle packing.
Uses penalty method with CMA-ES, then polishes with SLSQP.
"""

import json
import math
import numpy as np
from scipy.optimize import minimize
from pathlib import Path
import sys

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


def run_slsqp(circles, ftol=1e-15, maxiter=10000):
    x0 = circles.flatten()
    n = len(circles)
    cons = constraints_list(n)
    result = minimize(
        lambda x: -sum(x[3*i+2] for i in range(n)),
        x0, method="SLSQP",
        constraints=cons,
        options={"ftol": ftol, "maxiter": maxiter, "disp": False}
    )
    new_circles = result.x.reshape(-1, 3)
    valid, metric, viol = validate(new_circles)
    return new_circles, valid, metric


def penalty_objective(x, alpha=1e5):
    n = len(x) // 3
    obj = 0.0
    penalty = 0.0
    for i in range(n):
        xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
        obj -= ri
        for v in [ri - xi, xi + ri - 1.0, ri - yi, yi + ri - 1.0]:
            if v > 0:
                penalty += v**2
        if ri < 0:
            penalty += ri**2
    for i in range(n):
        xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
        for j in range(i+1, n):
            xj, yj, rj = x[3*j], x[3*j+1], x[3*j+2]
            dist = math.sqrt((xi-xj)**2 + (yi-yj)**2 + 1e-30)
            overlap = ri + rj - dist
            if overlap > 0:
                penalty += overlap**2
    return obj + alpha * penalty


def run_cmaes(initial, sigma=0.01, maxiter=5000, seed=42, popsize=None):
    """Run CMA-ES with penalty method."""
    try:
        import cma
    except ImportError:
        print("CMA-ES not available, installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "cma"])
        import cma

    x0 = initial.flatten()
    n = len(initial)

    opts = {
        "maxiter": maxiter,
        "seed": seed,
        "verbose": -1,
        "tolfun": 1e-15,
        "tolx": 1e-15,
    }
    if popsize:
        opts["popsize"] = popsize

    # Bounds
    lower = np.zeros(3*n)
    upper = np.ones(3*n)
    for i in range(n):
        lower[3*i+2] = 0.001  # min radius
        upper[3*i+2] = 0.5    # max radius
    opts["bounds"] = [lower.tolist(), upper.tolist()]

    best_valid_metric = 0
    best_valid_circles = None

    # Progressive alpha schedule
    for alpha in [1e3, 1e4, 1e5, 1e6]:
        es = cma.CMAEvolutionStrategy(x0, sigma, opts)

        while not es.stop():
            solutions = es.ask()
            fits = [penalty_objective(s, alpha=alpha) for s in solutions]
            es.tell(solutions, fits)

        x0 = es.result.xbest.copy()

        # Try SLSQP polish
        circles = x0.reshape(-1, 3)
        polished, valid, metric = run_slsqp(circles)
        if valid and metric > best_valid_metric:
            best_valid_metric = metric
            best_valid_circles = polished.copy()
            x0 = polished.flatten()
            print(f"  CMA-ES alpha={alpha}: valid metric={metric:.10f}")

    return best_valid_circles, best_valid_metric


def main():
    print("=" * 60)
    print("CMA-ES Optimization for N=32")
    print("=" * 60)

    initial = load_solution(WORKDIR / "solution_n32_initial.json")
    valid, metric, _ = validate(initial)
    print(f"Initial: metric={metric:.10f}")

    best_circles = initial.copy()
    best_metric = metric

    # Small sigma refinement around current best
    print("\n--- CMA-ES small sigma refinement ---")
    for sigma in [0.001, 0.003, 0.005, 0.01]:
        for seed in [42, 123, 456]:
            result, m = run_cmaes(best_circles, sigma=sigma, maxiter=3000, seed=seed, popsize=64)
            if result is not None and m > best_metric:
                best_metric = m
                best_circles = result.copy()
                print(f"  Improved with sigma={sigma}, seed={seed}: {m:.10f}")

    # Larger sigma exploration
    print("\n--- CMA-ES larger sigma exploration ---")
    for sigma in [0.02, 0.05, 0.1]:
        for seed in [42, 123, 456, 789]:
            result, m = run_cmaes(best_circles, sigma=sigma, maxiter=5000, seed=seed, popsize=128)
            if result is not None and m > best_metric:
                best_metric = m
                best_circles = result.copy()
                print(f"  Improved with sigma={sigma}, seed={seed}: {m:.10f}")

    save_solution(best_circles, WORKDIR / "solution_n32_cmaes.json")
    print(f"\nFinal CMA-ES result: {best_metric:.10f}")


if __name__ == "__main__":
    main()
