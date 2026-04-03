"""
KKT-based refinement: At the optimal solution, all KKT conditions must hold.
We can use Newton's method on the KKT system to get higher precision.

The idea: identify active constraints, set up the KKT system as a
nonlinear system of equations, and solve with scipy.optimize.fsolve.

For an optimal circle packing:
- Gradient of objective = sum of dual variables * gradient of constraints
- Active constraints hold with equality
- Dual variables for active constraints >= 0
"""

import json
import numpy as np
from scipy.optimize import fsolve, minimize
import os
import time

WORKDIR = os.path.dirname(os.path.abspath(__file__))

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

def find_active_constraints(x, y, r, tol=1e-6):
    """Identify active constraints and their types."""
    n = len(x)
    active = []

    for i in range(n):
        if abs(x[i] - r[i]) < tol:
            active.append(('wall_L', i))
        if abs(1 - x[i] - r[i]) < tol:
            active.append(('wall_R', i))
        if abs(y[i] - r[i]) < tol:
            active.append(('wall_B', i))
        if abs(1 - y[i] - r[i]) < tol:
            active.append(('wall_T', i))

    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            gap = dist - (r[i] + r[j])
            if abs(gap) < tol:
                active.append(('contact', i, j))

    return active

def build_kkt_system(x0, y0, r0):
    """Build the KKT system for the optimal circle packing."""
    n = len(x0)
    active = find_active_constraints(x0, y0, r0)
    n_active = len(active)
    n_vars = 3 * n  # x, y, r for each circle
    n_total = n_vars + n_active  # primal + dual variables

    print(f"  n={n}, active constraints={n_active}, total variables={n_total}")

    # Objective gradient: d(-sum r)/d(x,y,r) = [0,...,0, -1,...,-1]
    # grad_obj[2*n+i] = -1 for all i

    # For each active constraint g_k(x,y,r) = 0:
    #   grad_obj = sum_k lambda_k * grad(g_k)

    def kkt_equations(vars):
        """
        vars = [x_0,...,x_{n-1}, y_0,...,y_{n-1}, r_0,...,r_{n-1}, lam_0,...,lam_{m-1}]

        KKT conditions:
        1. Stationarity: nabla f = sum_k lambda_k * nabla g_k
        2. Primal feasibility: g_k(x) = 0 for active constraints
        """
        x = vars[:n]
        y = vars[n:2*n]
        r = vars[2*n:3*n]
        lam = vars[3*n:]

        equations = np.zeros(n_total)

        # Stationarity: grad_obj - sum lambda_k * grad(g_k) = 0
        # grad_obj for x: all zeros
        # grad_obj for y: all zeros
        # grad_obj for r: all -1
        grad_L = np.zeros(n_vars)
        grad_L[2*n:3*n] = -1.0  # objective gradient

        for k, constraint in enumerate(active):
            if constraint[0] == 'wall_L':
                i = constraint[1]
                # g = x_i - r_i = 0
                # grad_g: d/dx_i = 1, d/dr_i = -1
                grad_L[i] -= lam[k] * 1.0
                grad_L[2*n+i] -= lam[k] * (-1.0)

            elif constraint[0] == 'wall_R':
                i = constraint[1]
                # g = 1 - x_i - r_i = 0
                # grad_g: d/dx_i = -1, d/dr_i = -1
                grad_L[i] -= lam[k] * (-1.0)
                grad_L[2*n+i] -= lam[k] * (-1.0)

            elif constraint[0] == 'wall_B':
                i = constraint[1]
                # g = y_i - r_i = 0
                grad_L[n+i] -= lam[k] * 1.0
                grad_L[2*n+i] -= lam[k] * (-1.0)

            elif constraint[0] == 'wall_T':
                i = constraint[1]
                # g = 1 - y_i - r_i = 0
                grad_L[n+i] -= lam[k] * (-1.0)
                grad_L[2*n+i] -= lam[k] * (-1.0)

            elif constraint[0] == 'contact':
                i, j = constraint[1], constraint[2]
                # g = (x_i-x_j)^2 + (y_i-y_j)^2 - (r_i+r_j)^2 = 0
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                sr = r[i] + r[j]
                grad_L[i] -= lam[k] * 2 * dx
                grad_L[j] -= lam[k] * (-2 * dx)
                grad_L[n+i] -= lam[k] * 2 * dy
                grad_L[n+j] -= lam[k] * (-2 * dy)
                grad_L[2*n+i] -= lam[k] * (-2 * sr)
                grad_L[2*n+j] -= lam[k] * (-2 * sr)

        equations[:n_vars] = grad_L

        # Primal feasibility
        for k, constraint in enumerate(active):
            if constraint[0] == 'wall_L':
                i = constraint[1]
                equations[n_vars+k] = x[i] - r[i]
            elif constraint[0] == 'wall_R':
                i = constraint[1]
                equations[n_vars+k] = 1 - x[i] - r[i]
            elif constraint[0] == 'wall_B':
                i = constraint[1]
                equations[n_vars+k] = y[i] - r[i]
            elif constraint[0] == 'wall_T':
                i = constraint[1]
                equations[n_vars+k] = 1 - y[i] - r[i]
            elif constraint[0] == 'contact':
                i, j = constraint[1], constraint[2]
                equations[n_vars+k] = (x[i]-x[j])**2 + (y[i]-y[j])**2 - (r[i]+r[j])**2

        return equations

    # Initial dual variables: solve least-squares for stationarity
    # grad_obj = A @ lambda, where A is the Jacobian of active constraints
    A = np.zeros((n_vars, n_active))
    for k, constraint in enumerate(active):
        if constraint[0] == 'wall_L':
            i = constraint[1]
            A[i, k] = 1.0; A[2*n+i, k] = -1.0
        elif constraint[0] == 'wall_R':
            i = constraint[1]
            A[i, k] = -1.0; A[2*n+i, k] = -1.0
        elif constraint[0] == 'wall_B':
            i = constraint[1]
            A[n+i, k] = 1.0; A[2*n+i, k] = -1.0
        elif constraint[0] == 'wall_T':
            i = constraint[1]
            A[n+i, k] = -1.0; A[2*n+i, k] = -1.0
        elif constraint[0] == 'contact':
            i, j = constraint[1], constraint[2]
            dx = x0[i] - x0[j]
            dy = y0[i] - y0[j]
            sr = r0[i] + r0[j]
            A[i, k] = 2*dx; A[j, k] = -2*dx
            A[n+i, k] = 2*dy; A[n+j, k] = -2*dy
            A[2*n+i, k] = -2*sr; A[2*n+j, k] = -2*sr

    # Objective gradient
    grad_obj = np.zeros(n_vars)
    grad_obj[2*n:3*n] = -1.0

    # Solve A @ lambda = grad_obj  (least squares)
    lam0, residual, rank, sv = np.linalg.lstsq(A, grad_obj, rcond=None)
    print(f"  Dual variable LS residual: {np.linalg.norm(A @ lam0 - grad_obj):.2e}")
    print(f"  Dual variables range: [{lam0.min():.6f}, {lam0.max():.6f}]")
    print(f"  Negative duals: {np.sum(lam0 < -1e-8)}")

    vars0 = np.concatenate([x0, y0, r0, lam0])

    # Solve KKT system
    result = fsolve(kkt_equations, vars0, full_output=True)
    vars_sol = result[0]
    info = result[1]

    x_sol = vars_sol[:n]
    y_sol = vars_sol[n:2*n]
    r_sol = vars_sol[2*n:3*n]
    lam_sol = vars_sol[3*n:]

    residual_norm = np.linalg.norm(kkt_equations(vars_sol))
    print(f"  KKT residual: {residual_norm:.2e}")
    print(f"  Sum of radii: {np.sum(r_sol):.15f}")
    print(f"  Feasible: {is_feasible(x_sol, y_sol, r_sol)}")

    return x_sol, y_sol, r_sol

def main():
    t0 = time.time()

    # Load best known solution
    sol_path = os.path.join(WORKDIR, 'solution_n26.json')
    if os.path.exists(sol_path):
        x0, y0, r0 = load_solution(sol_path)
    else:
        parent_path = os.path.join(WORKDIR, '..', 'nlp-001', 'solution_n26.json')
        x0, y0, r0 = load_solution(parent_path)

    parent_metric = np.sum(r0)
    print(f"Starting metric: {parent_metric:.15f}")

    print("\n=== KKT Refinement ===")
    x1, y1, r1 = build_kkt_system(x0, y0, r0)

    m1 = np.sum(r1)
    if is_feasible(x1, y1, r1) and m1 > parent_metric:
        print(f"\n  IMPROVED: {parent_metric:.15f} -> {m1:.15f}")
        save_solution(x1, y1, r1, os.path.join(WORKDIR, 'solution_n26.json'))
    else:
        print(f"\n  No improvement: {m1:.15f}, feasible={is_feasible(x1, y1, r1)}")

    elapsed = time.time() - t0
    print(f"\nTime: {elapsed:.1f}s")

if __name__ == '__main__':
    main()
