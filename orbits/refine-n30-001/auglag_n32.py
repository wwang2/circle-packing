"""
Augmented Lagrangian method for n=32 circle packing.
This can sometimes find better solutions than SLSQP by using
a different optimization trajectory through the constraint landscape.

Also includes: trust-constr, COBYLA, and dual annealing approaches.
"""

import json
import math
import numpy as np
from scipy.optimize import minimize, dual_annealing, differential_evolution
from pathlib import Path
import sys
import time

WORKDIR = Path(__file__).parent
N = 32
TOL = 1e-10


def load_solution(path):
    with open(path) as f:
        data = json.load(f)
    return np.array(data.get("circles", data))


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


CONS = constraints_list(N)


def run_slsqp(circles, ftol=1e-15, maxiter=15000):
    x0 = circles.flatten()
    result = minimize(lambda x: -sum(x[3*i+2] for i in range(N)),
                     x0, method="SLSQP", constraints=CONS,
                     options={"ftol": ftol, "maxiter": maxiter, "disp": False})
    new_circles = result.x.reshape(-1, 3)
    valid, metric, viol = validate(new_circles)
    return new_circles, valid, metric


def augmented_lagrangian(circles, max_outer=50, seed=42):
    """
    Augmented Lagrangian method:
    min -sum(r_i) + sum(lambda_k * g_k(x)) + (rho/2) * sum(max(0, g_k(x))^2)

    where g_k(x) <= 0 are constraint violations.
    """
    x = circles.flatten().copy()
    n = N

    # Compute constraint violations
    def get_violations(x):
        viols = []
        for i in range(n):
            xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
            viols.append(ri - xi)      # left
            viols.append(xi + ri - 1)  # right
            viols.append(ri - yi)      # bottom
            viols.append(yi + ri - 1)  # top
            viols.append(-ri + 1e-12)  # positive radius
        for i in range(n):
            for j in range(i+1, n):
                xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
                xj, yj, rj = x[3*j], x[3*j+1], x[3*j+2]
                dist = math.sqrt((xi-xj)**2 + (yi-yj)**2 + 1e-30)
                viols.append(ri + rj - dist)
        return np.array(viols)

    n_cons = 5*n + n*(n-1)//2
    lam = np.zeros(n_cons)
    rho = 10.0

    best_circles = circles.copy()
    _, best_metric, _ = validate(circles)

    for outer in range(max_outer):
        def aug_obj(x):
            obj = -sum(x[3*i+2] for i in range(n))
            viols = get_violations(x)
            for k in range(n_cons):
                gk = viols[k]
                # Augmented Lagrangian term
                obj += lam[k] * max(0, gk) + (rho/2) * max(0, gk)**2
            return obj

        result = minimize(aug_obj, x, method="L-BFGS-B",
                         options={"maxiter": 2000, "ftol": 1e-15})
        x = result.x.copy()

        # Update multipliers
        viols = get_violations(x)
        for k in range(n_cons):
            lam[k] = max(0, lam[k] + rho * viols[k])

        # Increase penalty
        max_viol = max(0, max(viols))
        if max_viol > 1e-6:
            rho *= 2.0

        # Check feasibility
        c = x.reshape(-1, 3)
        valid, metric, viol = validate(c)
        if valid and metric > best_metric:
            best_metric = metric
            best_circles = c.copy()
            print(f"  AugLag iter {outer}: {metric:.10f} (rho={rho:.0f}, maxviol={max_viol:.2e})")

        if max_viol < 1e-12 and outer > 5:
            break

    # Polish with SLSQP
    polished, valid, metric = run_slsqp(best_circles)
    if valid and metric > best_metric:
        best_metric = metric
        best_circles = polished.copy()

    return best_circles, best_metric


def trust_constr_optimize(circles):
    """Use trust-constr method."""
    from scipy.optimize import NonlinearConstraint

    x0 = circles.flatten()
    n = N

    def objective(x):
        return -sum(x[3*i+2] for i in range(n))

    # Build bounds
    bounds = []
    for i in range(n):
        bounds.extend([(0, 1), (0, 1), (0.001, 0.5)])

    result = minimize(objective, x0, method="trust-constr",
                     constraints=[{"type": "ineq", "fun": c["fun"]} for c in CONS],
                     bounds=bounds,
                     options={"maxiter": 10000, "gtol": 1e-15})

    new_circles = result.x.reshape(-1, 3)
    valid, metric, viol = validate(new_circles)

    # Polish
    if valid:
        polished, v2, m2 = run_slsqp(new_circles)
        if v2 and m2 > metric:
            return polished, v2, m2

    return new_circles, valid, metric


def differential_evolution_search(seed=42, maxiter=1000, popsize=30):
    """Differential evolution with penalty method."""
    def penalty_obj(x):
        n = N
        obj = -sum(x[3*i+2] for i in range(n))
        penalty = 0
        for i in range(n):
            xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
            for v in [ri-xi, xi+ri-1, ri-yi, yi+ri-1]:
                if v > 0: penalty += v**2 * 1e6
        for i in range(n):
            xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
            for j in range(i+1, n):
                xj, yj, rj = x[3*j], x[3*j+1], x[3*j+2]
                dist = math.sqrt((xi-xj)**2 + (yi-yj)**2 + 1e-30)
                overlap = ri + rj - dist
                if overlap > 0: penalty += overlap**2 * 1e6
        return obj + penalty

    bounds = []
    for i in range(N):
        bounds.extend([(0.01, 0.99), (0.01, 0.99), (0.005, 0.5)])

    # Use the initial solution to seed the population
    initial = load_solution(WORKDIR / "solution_n32_initial.json")
    x0 = initial.flatten()

    result = differential_evolution(penalty_obj, bounds, seed=seed,
                                   maxiter=maxiter, popsize=popsize,
                                   x0=x0, tol=1e-15,
                                   mutation=(0.5, 1.5), recombination=0.9)

    circles = result.x.reshape(-1, 3)
    # Polish
    polished, valid, metric = run_slsqp(circles)
    return polished, valid, metric


def dual_annealing_search(circles, seed=42, maxiter=500):
    """Dual annealing with SLSQP local minimizer."""
    def penalty_obj(x):
        n = N
        obj = -sum(x[3*i+2] for i in range(n))
        penalty = 0
        for i in range(n):
            xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
            for v in [ri-xi, xi+ri-1, ri-yi, yi+ri-1]:
                if v > 0: penalty += v**2 * 1e5
        for i in range(n):
            xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
            for j in range(i+1, n):
                xj, yj, rj = x[3*j], x[3*j+1], x[3*j+2]
                dist = math.sqrt((xi-xj)**2 + (yi-yj)**2 + 1e-30)
                overlap = ri + rj - dist
                if overlap > 0: penalty += overlap**2 * 1e5
        return obj + penalty

    bounds = []
    for i in range(N):
        bounds.extend([(0.01, 0.99), (0.01, 0.99), (0.005, 0.5)])

    x0 = circles.flatten()

    result = dual_annealing(penalty_obj, bounds, seed=seed,
                           maxiter=maxiter, x0=x0,
                           local_search_options={"method": "L-BFGS-B", "options": {"maxiter": 500}})

    new_circles = result.x.reshape(-1, 3)
    polished, valid, metric = run_slsqp(new_circles)
    return polished, valid, metric


def main():
    print("=" * 60)
    print("Augmented Lagrangian + Alternative Methods for N=32")
    print("=" * 60)

    initial = load_solution(WORKDIR / "solution_n32_initial.json")
    _, base_metric, _ = validate(initial)
    print(f"Base metric: {base_metric:.10f}")

    best_circles = initial.copy()
    best_metric = base_metric

    # 1. Augmented Lagrangian
    print("\n--- Augmented Lagrangian ---")
    al_circles, al_metric = augmented_lagrangian(initial, max_outer=50)
    print(f"AugLag result: {al_metric:.10f}")
    if al_metric > best_metric:
        best_metric = al_metric
        best_circles = al_circles.copy()

    # 2. trust-constr
    print("\n--- trust-constr ---")
    tc_circles, tc_valid, tc_metric = trust_constr_optimize(initial)
    print(f"trust-constr result: {tc_metric:.10f}, valid={tc_valid}")
    if tc_valid and tc_metric > best_metric:
        best_metric = tc_metric
        best_circles = tc_circles.copy()

    # 3. Differential evolution (shorter run as DE is slow for 96D)
    print("\n--- Differential Evolution ---")
    for seed in [42, 123]:
        de_circles, de_valid, de_metric = differential_evolution_search(seed=seed, maxiter=300, popsize=20)
        print(f"DE seed={seed}: {de_metric:.10f}, valid={de_valid}")
        if de_valid and de_metric > best_metric:
            best_metric = de_metric
            best_circles = de_circles.copy()

    # 4. Dual annealing
    print("\n--- Dual Annealing ---")
    for seed in [42, 123, 456]:
        da_circles, da_valid, da_metric = dual_annealing_search(initial, seed=seed, maxiter=200)
        print(f"DA seed={seed}: {da_metric:.10f}, valid={da_valid}")
        if da_valid and da_metric > best_metric:
            best_metric = da_metric
            best_circles = da_circles.copy()

    save_solution(best_circles, WORKDIR / "solution_n32_auglag.json")
    print(f"\n{'='*60}")
    print(f"FINAL: {best_metric:.10f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
