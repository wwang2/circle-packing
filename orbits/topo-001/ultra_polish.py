"""
Ultra-high-precision polish of the known solution.
Uses equality constraints for active contacts and iterative refinement.
"""

import json
import numpy as np
from scipy.optimize import minimize
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

def find_active_contacts(x, y, r, tol=1e-4):
    """Find active circle-circle and wall contacts."""
    n = len(x)
    cc = []
    wc = []
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            gap = dist - (r[i] + r[j])
            if abs(gap) < tol:
                cc.append((i, j))
        if abs(x[i] - r[i]) < tol:
            wc.append((i, 'L'))
        if abs(1 - x[i] - r[i]) < tol:
            wc.append((i, 'R'))
        if abs(y[i] - r[i]) < tol:
            wc.append((i, 'B'))
        if abs(1 - y[i] - r[i]) < tol:
            wc.append((i, 'T'))
    return cc, wc

def optimize_with_active_contacts(x0, y0, r0, maxiter=30000):
    """Optimize using equality constraints for active contacts."""
    n = len(x0)
    cc, wc = find_active_contacts(x0, y0, r0)
    print(f"  Active contacts: {len(cc)} circle-circle, {len(wc)} wall")

    params0 = np.concatenate([x0, y0, r0])
    constraints = []

    # Inequality constraints (all)
    for i in range(n):
        def cf_xl(p, i=i): return p[i] - p[2*n+i]
        def cj_xl(p, i=i):
            g = np.zeros(3*n); g[i] = 1.0; g[2*n+i] = -1.0; return g
        constraints.append({'type': 'ineq', 'fun': cf_xl, 'jac': cj_xl})

        def cf_xr(p, i=i): return 1 - p[i] - p[2*n+i]
        def cj_xr(p, i=i):
            g = np.zeros(3*n); g[i] = -1.0; g[2*n+i] = -1.0; return g
        constraints.append({'type': 'ineq', 'fun': cf_xr, 'jac': cj_xr})

        def cf_yl(p, i=i): return p[n+i] - p[2*n+i]
        def cj_yl(p, i=i):
            g = np.zeros(3*n); g[n+i] = 1.0; g[2*n+i] = -1.0; return g
        constraints.append({'type': 'ineq', 'fun': cf_yl, 'jac': cj_yl})

        def cf_yr(p, i=i): return 1 - p[n+i] - p[2*n+i]
        def cj_yr(p, i=i):
            g = np.zeros(3*n); g[n+i] = -1.0; g[2*n+i] = -1.0; return g
        constraints.append({'type': 'ineq', 'fun': cf_yr, 'jac': cj_yr})

        def cf_rp(p, i=i): return p[2*n+i] - 1e-6
        def cj_rp(p, i=i):
            g = np.zeros(3*n); g[2*n+i] = 1.0; return g
        constraints.append({'type': 'ineq', 'fun': cf_rp, 'jac': cj_rp})

    for i in range(n):
        for j in range(i+1, n):
            def cf_no(p, i=i, j=j):
                return (p[i]-p[j])**2 + (p[n+i]-p[n+j])**2 - (p[2*n+i]+p[2*n+j])**2
            def cj_no(p, i=i, j=j):
                g = np.zeros(3*n)
                dx = p[i]-p[j]; dy = p[n+i]-p[n+j]; sr = p[2*n+i]+p[2*n+j]
                g[i]=2*dx; g[j]=-2*dx; g[n+i]=2*dy; g[n+j]=-2*dy
                g[2*n+i]=-2*sr; g[2*n+j]=-2*sr
                return g
            constraints.append({'type': 'ineq', 'fun': cf_no, 'jac': cj_no})

    # Add EQUALITY constraints for active contacts
    for ci, cj in cc:
        def cf_eq(p, i=ci, j=cj):
            return (p[i]-p[j])**2 + (p[n+i]-p[n+j])**2 - (p[2*n+i]+p[2*n+j])**2
        def cj_eq(p, i=ci, j=cj):
            g = np.zeros(3*n)
            dx = p[i]-p[j]; dy = p[n+i]-p[n+j]; sr = p[2*n+i]+p[2*n+j]
            g[i]=2*dx; g[j]=-2*dx; g[n+i]=2*dy; g[n+j]=-2*dy
            g[2*n+i]=-2*sr; g[2*n+j]=-2*sr
            return g
        constraints.append({'type': 'eq', 'fun': cf_eq, 'jac': cj_eq})

    for ci, wall in wc:
        if wall == 'L':
            def cf_eq(p, i=ci): return p[i] - p[2*n+i]
            def cj_eq(p, i=ci):
                g = np.zeros(3*n); g[i] = 1.0; g[2*n+i] = -1.0; return g
        elif wall == 'R':
            def cf_eq(p, i=ci): return 1 - p[i] - p[2*n+i]
            def cj_eq(p, i=ci):
                g = np.zeros(3*n); g[i] = -1.0; g[2*n+i] = -1.0; return g
        elif wall == 'B':
            def cf_eq(p, i=ci): return p[n+i] - p[2*n+i]
            def cj_eq(p, i=ci):
                g = np.zeros(3*n); g[n+i] = 1.0; g[2*n+i] = -1.0; return g
        elif wall == 'T':
            def cf_eq(p, i=ci): return 1 - p[n+i] - p[2*n+i]
            def cj_eq(p, i=ci):
                g = np.zeros(3*n); g[n+i] = -1.0; g[2*n+i] = -1.0; return g
        constraints.append({'type': 'eq', 'fun': cf_eq, 'jac': cj_eq})

    result = minimize(
        lambda p: (-np.sum(p[2*n:3*n]),
                   np.concatenate([np.zeros(2*n), -np.ones(n)])),
        params0, method='SLSQP', jac=True, constraints=constraints,
        options={'maxiter': maxiter, 'ftol': 1e-16, 'disp': False}
    )

    x, y, r = result.x[:n], result.x[n:2*n], result.x[2*n:3*n]
    return x, y, r, np.sum(r), result.success

def iterative_polish(x0, y0, r0, rounds=10):
    """Alternating SLSQP optimization: fix positions -> optimize radii -> fix radii -> optimize positions."""
    n = len(x0)
    best_x, best_y, best_r = x0.copy(), y0.copy(), r0.copy()
    best_metric = np.sum(r0)

    for rnd in range(rounds):
        # Phase A: Fix positions, optimize radii
        params_r = best_r.copy()
        x_fixed, y_fixed = best_x.copy(), best_y.copy()

        constraints_r = []
        for i in range(n):
            constraints_r.append({'type': 'ineq', 'fun': lambda p, i=i: x_fixed[i] - p[i]})
            constraints_r.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - x_fixed[i] - p[i]})
            constraints_r.append({'type': 'ineq', 'fun': lambda p, i=i: y_fixed[i] - p[i]})
            constraints_r.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - y_fixed[i] - p[i]})
            constraints_r.append({'type': 'ineq', 'fun': lambda p, i=i: p[i] - 1e-6})
        for i in range(n):
            for j in range(i+1, n):
                dist2 = (x_fixed[i]-x_fixed[j])**2 + (y_fixed[i]-y_fixed[j])**2
                constraints_r.append({
                    'type': 'ineq',
                    'fun': lambda p, d2=dist2, i=i, j=j: d2 - (p[i]+p[j])**2
                })

        result_r = minimize(
            lambda p: -np.sum(p),
            params_r, method='SLSQP', constraints=constraints_r,
            options={'maxiter': 5000, 'ftol': 1e-16}
        )

        if result_r.success:
            r_new = result_r.x
            if is_feasible(x_fixed, y_fixed, r_new) and np.sum(r_new) > best_metric:
                best_r = r_new
                best_metric = np.sum(r_new)
                print(f"  Round {rnd} phase A: {best_metric:.12f}")

        # Phase B: Optimize all jointly
        x_all, y_all, r_all, m_all, s_all = optimize_with_active_contacts(
            best_x, best_y, best_r, maxiter=15000)
        if s_all and is_feasible(x_all, y_all, r_all) and m_all > best_metric:
            best_x, best_y, best_r = x_all, y_all, r_all
            best_metric = m_all
            print(f"  Round {rnd} phase B: {best_metric:.12f}")

    return best_x, best_y, best_r, best_metric

def main():
    t0 = time.time()
    parent_path = os.path.join(WORKDIR, '..', 'nlp-001', 'solution_n26.json')
    x0, y0, r0 = load_solution(parent_path)
    parent_metric = np.sum(r0)
    print(f"Parent metric: {parent_metric:.12f}")

    # Also load solution_n26.json if exists (from previous runs)
    sol_path = os.path.join(WORKDIR, 'solution_n26.json')
    if os.path.exists(sol_path):
        x1, y1, r1 = load_solution(sol_path)
        if np.sum(r1) > np.sum(r0):
            x0, y0, r0 = x1, y1, r1
            parent_metric = np.sum(r0)
            print(f"Using better existing solution: {parent_metric:.12f}")

    print(f"\n=== Ultra Polish ===")
    best_x, best_y, best_r, best_metric = iterative_polish(x0, y0, r0, rounds=5)

    print(f"\n=== Standard SLSQP with high iterations ===")
    n = len(best_x)
    params0 = np.concatenate([best_x, best_y, best_r])
    constraints = []
    for i in range(n):
        def cf_xl(p, i=i): return p[i] - p[2*n+i]
        def cj_xl(p, i=i):
            g = np.zeros(3*n); g[i] = 1.0; g[2*n+i] = -1.0; return g
        constraints.append({'type': 'ineq', 'fun': cf_xl, 'jac': cj_xl})
        def cf_xr(p, i=i): return 1 - p[i] - p[2*n+i]
        def cj_xr(p, i=i):
            g = np.zeros(3*n); g[i] = -1.0; g[2*n+i] = -1.0; return g
        constraints.append({'type': 'ineq', 'fun': cf_xr, 'jac': cj_xr})
        def cf_yl(p, i=i): return p[n+i] - p[2*n+i]
        def cj_yl(p, i=i):
            g = np.zeros(3*n); g[n+i] = 1.0; g[2*n+i] = -1.0; return g
        constraints.append({'type': 'ineq', 'fun': cf_yl, 'jac': cj_yl})
        def cf_yr(p, i=i): return 1 - p[n+i] - p[2*n+i]
        def cj_yr(p, i=i):
            g = np.zeros(3*n); g[n+i] = -1.0; g[2*n+i] = -1.0; return g
        constraints.append({'type': 'ineq', 'fun': cf_yr, 'jac': cj_yr})
        def cf_rp(p, i=i): return p[2*n+i] - 1e-6
        def cj_rp(p, i=i):
            g = np.zeros(3*n); g[2*n+i] = 1.0; return g
        constraints.append({'type': 'ineq', 'fun': cf_rp, 'jac': cj_rp})
    for i in range(n):
        for j in range(i+1, n):
            def cf_no(p, i=i, j=j):
                return (p[i]-p[j])**2 + (p[n+i]-p[n+j])**2 - (p[2*n+i]+p[2*n+j])**2
            def cj_no(p, i=i, j=j):
                g = np.zeros(3*n)
                dx = p[i]-p[j]; dy = p[n+i]-p[n+j]; sr = p[2*n+i]+p[2*n+j]
                g[i]=2*dx; g[j]=-2*dx; g[n+i]=2*dy; g[n+j]=-2*dy
                g[2*n+i]=-2*sr; g[2*n+j]=-2*sr
                return g
            constraints.append({'type': 'ineq', 'fun': cf_no, 'jac': cj_no})

    for attempt in range(5):
        result = minimize(
            lambda p: (-np.sum(p[2*n:3*n]),
                       np.concatenate([np.zeros(2*n), -np.ones(n)])),
            np.concatenate([best_x, best_y, best_r]),
            method='SLSQP', jac=True, constraints=constraints,
            options={'maxiter': 50000, 'ftol': 1e-16, 'disp': False}
        )
        x, y, r = result.x[:n], result.x[n:2*n], result.x[2*n:3*n]
        m = np.sum(r)
        if is_feasible(x, y, r) and m > best_metric:
            best_metric = m
            best_x, best_y, best_r = x, y, r
            print(f"  Attempt {attempt}: {best_metric:.12f}")

    save_solution(best_x, best_y, best_r, os.path.join(WORKDIR, 'solution_n26.json'))

    elapsed = time.time() - t0
    print(f"\n=== FINAL ===")
    print(f"Parent:  {parent_metric:.12f}")
    print(f"Best:    {best_metric:.12f}")
    print(f"Delta:   {best_metric - parent_metric:.2e}")
    print(f"Time:    {elapsed:.0f}s")

if __name__ == '__main__':
    main()
