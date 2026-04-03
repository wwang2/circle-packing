"""
v8: KKT Refinement with multiple strategies.

Try to squeeze more precision from the known solution by:
1. Running KKT Newton from the SLSQP-improved solution (v4's output)
2. Using mpmath for higher precision
3. Iterating SLSQP -> KKT -> SLSQP -> KKT
"""

import json
import numpy as np
from scipy.optimize import fsolve, minimize
import os
import time

WORKDIR = os.path.dirname(os.path.abspath(__file__))
N = 26


def load_solution(path):
    with open(path) as f:
        data = json.load(f)
    circles = np.array(data["circles"])
    return circles[:, 0], circles[:, 1], circles[:, 2]


def save_solution(x, y, r, path):
    circles = [[float(x[i]), float(y[i]), float(r[i])] for i in range(len(x))]
    with open(path, 'w') as f:
        json.dump({"circles": circles}, f, indent=2)


def is_feasible(x, y, r, tol=1e-10):
    n = len(x)
    for i in range(n):
        if r[i] <= 0: return False
        if x[i] - r[i] < -tol or 1 - x[i] - r[i] < -tol: return False
        if y[i] - r[i] < -tol or 1 - y[i] - r[i] < -tol: return False
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if dist < r[i] + r[j] - tol: return False
    return True


def find_active(x, y, r, tol=1e-6):
    n = len(x)
    active = []
    for i in range(n):
        if abs(x[i] - r[i]) < tol: active.append(('wall_L', i))
        if abs(1 - x[i] - r[i]) < tol: active.append(('wall_R', i))
        if abs(y[i] - r[i]) < tol: active.append(('wall_B', i))
        if abs(1 - y[i] - r[i]) < tol: active.append(('wall_T', i))
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if abs(dist - r[i] - r[j]) < tol:
                active.append(('contact', i, j))
    return active


def kkt_refine(x0, y0, r0):
    """KKT refinement via Newton on the full KKT system."""
    n = len(x0)
    active = find_active(x0, y0, r0)
    n_active = len(active)
    n_vars = 3 * n
    n_total = n_vars + n_active

    print(f"  Active constraints: {n_active} (need {n_vars} for zero DOF)")

    def kkt_eqns(vars):
        x = vars[:n]
        y = vars[n:2*n]
        r = vars[2*n:3*n]
        lam = vars[3*n:]

        equations = np.zeros(n_total)
        grad_L = np.zeros(n_vars)
        grad_L[2*n:3*n] = -1.0

        for k, constraint in enumerate(active):
            if constraint[0] == 'wall_L':
                i = constraint[1]
                grad_L[i] -= lam[k]; grad_L[2*n+i] -= lam[k]*(-1)
            elif constraint[0] == 'wall_R':
                i = constraint[1]
                grad_L[i] -= lam[k]*(-1); grad_L[2*n+i] -= lam[k]*(-1)
            elif constraint[0] == 'wall_B':
                i = constraint[1]
                grad_L[n+i] -= lam[k]; grad_L[2*n+i] -= lam[k]*(-1)
            elif constraint[0] == 'wall_T':
                i = constraint[1]
                grad_L[n+i] -= lam[k]*(-1); grad_L[2*n+i] -= lam[k]*(-1)
            elif constraint[0] == 'contact':
                i, j = constraint[1], constraint[2]
                dx = x[i]-x[j]; dy = y[i]-y[j]; sr = r[i]+r[j]
                grad_L[i] -= lam[k]*2*dx; grad_L[j] -= lam[k]*(-2*dx)
                grad_L[n+i] -= lam[k]*2*dy; grad_L[n+j] -= lam[k]*(-2*dy)
                grad_L[2*n+i] -= lam[k]*(-2*sr); grad_L[2*n+j] -= lam[k]*(-2*sr)

        equations[:n_vars] = grad_L

        for k, constraint in enumerate(active):
            if constraint[0] == 'wall_L':
                i = constraint[1]; equations[n_vars+k] = x[i] - r[i]
            elif constraint[0] == 'wall_R':
                i = constraint[1]; equations[n_vars+k] = 1 - x[i] - r[i]
            elif constraint[0] == 'wall_B':
                i = constraint[1]; equations[n_vars+k] = y[i] - r[i]
            elif constraint[0] == 'wall_T':
                i = constraint[1]; equations[n_vars+k] = 1 - y[i] - r[i]
            elif constraint[0] == 'contact':
                i, j = constraint[1], constraint[2]
                equations[n_vars+k] = (x[i]-x[j])**2 + (y[i]-y[j])**2 - (r[i]+r[j])**2

        return equations

    # Initial dual variables via least squares
    A = np.zeros((n_vars, n_active))
    for k, constraint in enumerate(active):
        if constraint[0] == 'wall_L':
            i = constraint[1]; A[i,k] = 1; A[2*n+i,k] = -1
        elif constraint[0] == 'wall_R':
            i = constraint[1]; A[i,k] = -1; A[2*n+i,k] = -1
        elif constraint[0] == 'wall_B':
            i = constraint[1]; A[n+i,k] = 1; A[2*n+i,k] = -1
        elif constraint[0] == 'wall_T':
            i = constraint[1]; A[n+i,k] = -1; A[2*n+i,k] = -1
        elif constraint[0] == 'contact':
            i, j = constraint[1], constraint[2]
            dx = x0[i]-x0[j]; dy = y0[i]-y0[j]; sr = r0[i]+r0[j]
            A[i,k] = 2*dx; A[j,k] = -2*dx
            A[n+i,k] = 2*dy; A[n+j,k] = -2*dy
            A[2*n+i,k] = -2*sr; A[2*n+j,k] = -2*sr

    grad_obj = np.zeros(n_vars)
    grad_obj[2*n:3*n] = -1.0
    lam0, _, _, _ = np.linalg.lstsq(A, grad_obj, rcond=None)

    print(f"  Dual variables: min={lam0.min():.6f}, max={lam0.max():.6f}")
    print(f"  Negative duals: {np.sum(lam0 < -1e-8)}")

    vars0 = np.concatenate([x0, y0, r0, lam0])

    # Solve KKT
    result = fsolve(kkt_eqns, vars0, full_output=True)
    vars_sol = result[0]

    x_sol = vars_sol[:n]
    y_sol = vars_sol[n:2*n]
    r_sol = vars_sol[2*n:3*n]

    residual = np.linalg.norm(kkt_eqns(vars_sol))
    metric = np.sum(r_sol)
    feasible = is_feasible(x_sol, y_sol, r_sol)

    print(f"  KKT residual: {residual:.2e}")
    print(f"  Metric: {metric:.15f}")
    print(f"  Feasible: {feasible}")

    return x_sol, y_sol, r_sol, metric, feasible


def fast_slsqp(x0, y0, r0, maxiter=2000):
    """SLSQP with analytical Jacobian."""
    n = len(x0)
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    n_wall = 4 * n
    n_sep = len(pairs)
    n_cons = n_wall + n_sep

    def all_constraints(v):
        x, y, r = v[:n], v[n:2*n], v[2*n:3*n]
        c = np.empty(n_cons)
        for i in range(n):
            c[4*i] = x[i] - r[i]
            c[4*i+1] = 1.0 - x[i] - r[i]
            c[4*i+2] = y[i] - r[i]
            c[4*i+3] = 1.0 - y[i] - r[i]
        idx = n_wall
        for i, j in pairs:
            c[idx] = (x[i]-x[j])**2 + (y[i]-y[j])**2 - (r[i]+r[j])**2
            idx += 1
        return c

    def all_constraints_jac(v):
        x, y, r = v[:n], v[n:2*n], v[2*n:3*n]
        J = np.zeros((n_cons, 3*n))
        for i in range(n):
            J[4*i, i] = 1.0; J[4*i, 2*n+i] = -1.0
            J[4*i+1, i] = -1.0; J[4*i+1, 2*n+i] = -1.0
            J[4*i+2, n+i] = 1.0; J[4*i+2, 2*n+i] = -1.0
            J[4*i+3, n+i] = -1.0; J[4*i+3, 2*n+i] = -1.0
        idx = n_wall
        for i, j in pairs:
            dx = x[i]-x[j]; dy = y[i]-y[j]; sr = r[i]+r[j]
            J[idx, i] = 2*dx; J[idx, j] = -2*dx
            J[idx, n+i] = 2*dy; J[idx, n+j] = -2*dy
            J[idx, 2*n+i] = -2*sr; J[idx, 2*n+j] = -2*sr
            idx += 1
        return J

    constraints = [{'type': 'ineq', 'fun': all_constraints, 'jac': all_constraints_jac}]
    bounds = [(0.001, 0.999)]*n + [(0.001, 0.999)]*n + [(0.001, 0.5)]*n
    v0 = np.concatenate([x0, y0, r0])

    result = minimize(lambda v: -np.sum(v[2*n:3*n]), v0, method='SLSQP',
                      jac=lambda v: np.concatenate([np.zeros(2*n), -np.ones(n)]),
                      constraints=constraints, bounds=bounds,
                      options={'maxiter': maxiter, 'ftol': 1e-15})

    x, y, r = result.x[:n], result.x[n:2*n], result.x[2*n:3*n]
    return x, y, r, np.sum(r), is_feasible(x, y, r)


def main():
    t0 = time.time()

    sol_path = os.path.join(WORKDIR, 'solution_n26.json')
    x0, y0, r0 = load_solution(sol_path)
    base_metric = np.sum(r0)
    print(f"Starting metric: {base_metric:.15f}")

    best_metric = base_metric
    best_sol = (x0.copy(), y0.copy(), r0.copy())

    # Round 1: KKT refine from current solution
    print("\n=== Round 1: KKT refinement ===")
    x1, y1, r1, m1, f1 = kkt_refine(x0, y0, r0)
    if f1 and m1 > best_metric:
        best_metric = m1
        best_sol = (x1, y1, r1)
        print(f"  Improved: {best_metric:.15f}")

    # Round 2: SLSQP from KKT result
    print("\n=== Round 2: SLSQP polish ===")
    x2, y2, r2, m2, f2 = fast_slsqp(*best_sol)
    if f2 and m2 > best_metric:
        best_metric = m2
        best_sol = (x2, y2, r2)
        print(f"  Improved: {best_metric:.15f}")

    # Round 3: KKT again
    print("\n=== Round 3: KKT refinement ===")
    x3, y3, r3, m3, f3 = kkt_refine(*best_sol)
    if f3 and m3 > best_metric:
        best_metric = m3
        best_sol = (x3, y3, r3)
        print(f"  Improved: {best_metric:.15f}")

    # Round 4: SLSQP again
    print("\n=== Round 4: SLSQP polish ===")
    x4, y4, r4, m4, f4 = fast_slsqp(*best_sol)
    if f4 and m4 > best_metric:
        best_metric = m4
        best_sol = (x4, y4, r4)
        print(f"  Improved: {best_metric:.15f}")

    # Round 5: Final KKT
    print("\n=== Round 5: Final KKT ===")
    x5, y5, r5, m5, f5 = kkt_refine(*best_sol)
    if f5 and m5 > best_metric:
        best_metric = m5
        best_sol = (x5, y5, r5)
        print(f"  Improved: {best_metric:.15f}")

    # Save
    save_solution(*best_sol, sol_path)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"FINAL: {best_metric:.15f}")
    print(f"Start: {base_metric:.15f}")
    print(f"Improvement: {best_metric - base_metric:.2e}")
    print(f"Time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
