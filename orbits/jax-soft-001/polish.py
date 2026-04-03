"""
Strict polishing script: takes a near-feasible solution and makes it
strictly feasible (violations < 1e-10) while maximizing sum of radii.

Strategy:
1. First shrink radii slightly to make feasible
2. Then re-optimize with SLSQP with very tight tolerances
3. Multiple rounds with decreasing perturbation
"""

import json
import math
import numpy as np
from scipy.optimize import minimize
from pathlib import Path

N = 26


def load_solution(path):
    with open(path) as f:
        data = json.load(f)
    circles = data["circles"] if "circles" in data else data
    x = np.array([c[0] for c in circles])
    y = np.array([c[1] for c in circles])
    r = np.array([c[2] for c in circles])
    return np.concatenate([x, y, r])


def save_solution(params, path):
    circles = []
    for i in range(N):
        circles.append([float(params[i]), float(params[N+i]), float(params[2*N+i])])
    with open(path, 'w') as f:
        json.dump({"circles": circles}, f, indent=2)


def check_feasibility(params, tol=1e-10):
    x = params[:N]
    y = params[N:2*N]
    r = params[2*N:]

    max_viol = 0.0

    for i in range(N):
        max_viol = max(max_viol, r[i] - x[i])
        max_viol = max(max_viol, x[i] + r[i] - 1.0)
        max_viol = max(max_viol, r[i] - y[i])
        max_viol = max(max_viol, y[i] + r[i] - 1.0)

    for i in range(N):
        for j in range(i+1, N):
            dist = math.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            overlap = r[i] + r[j] - dist
            max_viol = max(max_viol, overlap)

    return max_viol


def make_feasible(params, margin=1e-9):
    """Shrink radii slightly to eliminate all violations."""
    x = params[:N].copy()
    y = params[N:2*N].copy()
    r = params[2*N:].copy()

    # Shrink for wall violations
    for i in range(N):
        r[i] = min(r[i], x[i] - margin)
        r[i] = min(r[i], 1.0 - x[i] - margin)
        r[i] = min(r[i], y[i] - margin)
        r[i] = min(r[i], 1.0 - y[i] - margin)

    # Shrink for overlap violations
    for _ in range(10):  # iterate to handle cascading
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                dist = math.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                overlap = r[i] + r[j] - dist + margin
                if overlap > 0:
                    # Shrink both proportionally
                    total = r[i] + r[j]
                    if total > 0:
                        r[i] -= overlap * r[i] / total
                        r[j] -= overlap * r[j] / total
                        changed = True
        if not changed:
            break

    return np.concatenate([x, y, r])


def polish_slsqp(params, ftol=1e-15, maxiter=20000):
    """Polish with SLSQP."""
    def objective(p):
        return -np.sum(p[2*N:])

    def obj_jac(p):
        g = np.zeros_like(p)
        g[2*N:] = -1.0
        return g

    constraints = []

    # Wall containment
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[i] - p[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1.0 - p[i] - p[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[N+i] - p[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1.0 - p[N+i] - p[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[2*N+i] - 1e-10})

    # Non-overlap
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: math.sqrt(
                    (p[i]-p[j])**2 + (p[N+i]-p[N+j])**2
                ) - p[2*N+i] - p[2*N+j]
            })

    result = minimize(
        objective, params, jac=obj_jac,
        method='SLSQP', constraints=constraints,
        options={'maxiter': maxiter, 'ftol': ftol, 'disp': False}
    )

    return result.x, -result.fun, result.success


def main():
    workdir = Path(__file__).parent
    sol_path = workdir / "solution_n26.json"

    if not sol_path.exists():
        print("No solution found to polish!")
        return

    params = load_solution(str(sol_path))
    metric_raw = np.sum(params[2*N:])
    viol = check_feasibility(params)
    print(f"Input: metric={metric_raw:.10f}, max_violation={viol:.2e}")

    # Step 1: Make strictly feasible
    params_feas = make_feasible(params, margin=1e-9)
    metric_feas = np.sum(params_feas[2*N:])
    viol_feas = check_feasibility(params_feas)
    print(f"After feasibility fix: metric={metric_feas:.10f}, max_violation={viol_feas:.2e}")

    # Step 2: Polish with SLSQP (multiple rounds)
    best_metric = 0.0
    best_params = params_feas

    for round_idx in range(5):
        polished, metric, success = polish_slsqp(best_params, ftol=1e-15, maxiter=20000)
        viol = check_feasibility(polished)
        print(f"Polish round {round_idx+1}: metric={metric:.10f}, max_violation={viol:.2e}, success={success}")

        if viol < 1e-10 and metric > best_metric:
            best_metric = metric
            best_params = polished
            print(f"  -> Valid! New best: {best_metric:.10f}")

    # Step 3: Try starting from topo-001 solution too
    topo_path = workdir.parent / "topo-001" / "solution_n26.json"
    if topo_path.exists():
        topo_params = load_solution(str(topo_path))
        for round_idx in range(3):
            polished, metric, success = polish_slsqp(topo_params, ftol=1e-15, maxiter=20000)
            viol = check_feasibility(polished)
            print(f"Topo polish round {round_idx+1}: metric={metric:.10f}, max_violation={viol:.2e}")

            if viol < 1e-10 and metric > best_metric:
                best_metric = metric
                best_params = polished
                print(f"  -> Valid! New best from topo: {best_metric:.10f}")
            topo_params = polished

    # Save best valid solution
    if best_metric > 0:
        save_solution(best_params, str(workdir / "solution_n26.json"))
        print(f"\nFinal: metric={best_metric:.10f}")
        print(f"Saved to {workdir / 'solution_n26.json'}")
    else:
        print("\nNo valid solution found after polishing!")


if __name__ == "__main__":
    main()
