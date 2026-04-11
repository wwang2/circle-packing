#!/usr/bin/env python3
"""
High-precision KKT refinement for n=26 circle packing.

The current solution has 78 active constraints = 78 variables (fully rigid).
The KKT system is:
  grad(objective) = sum_k lambda_k * grad(g_k)
  g_k(x) = 0 for active constraints

We solve this 78+78 = 156 dimensional system using Newton's method
with careful identification of active constraints.
"""

import json
import math
import numpy as np
from scipy.optimize import fsolve, minimize
from pathlib import Path
import sys

WORKTREE = Path("/Users/wujiewang/code/circle-packing/.worktrees/mobius-001")
OUTPUT_DIR = WORKTREE / "orbits/mobius-001"
N = 26


def flush_print(*args, **kwargs):
    print(*args, **kwargs, flush=True)


def load_solution(path):
    with open(path) as f:
        return np.array(json.load(f)["circles"])


def save_solution(circles, path):
    data = {"circles": [[float(c[0]), float(c[1]), float(c[2])] for c in circles]}
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def sum_radii(c):
    return float(np.sum(c[:, 2]))


def validate(circles, tol=1e-10):
    n = len(circles)
    mv = 0.0
    for i in range(n):
        x, y, r = circles[i]
        if r <= 0: return False, abs(r)
        mv = max(mv, r - x, x + r - 1, r - y, y + r - 1)
    for i in range(n):
        for j in range(i + 1, n):
            dx = circles[i, 0] - circles[j, 0]
            dy = circles[i, 1] - circles[j, 1]
            d = math.sqrt(dx * dx + dy * dy)
            mv = max(mv, circles[i, 2] + circles[j, 2] - d)
    return mv <= tol, mv


def identify_active_constraints(circles, tol=1e-6):
    """Identify active constraints and return their info."""
    n = len(circles)
    active = []

    for i in range(n):
        x, y, r = circles[i]
        # Wall constraints: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
        if abs(x - r) < tol:
            active.append(('wall_left', i))
        if abs(1 - x - r) < tol:
            active.append(('wall_right', i))
        if abs(y - r) < tol:
            active.append(('wall_bottom', i))
        if abs(1 - y - r) < tol:
            active.append(('wall_top', i))

    for i in range(n):
        for j in range(i + 1, n):
            dx = circles[i, 0] - circles[j, 0]
            dy = circles[i, 1] - circles[j, 1]
            d = math.sqrt(dx * dx + dy * dy)
            gap = d - circles[i, 2] - circles[j, 2]
            if abs(gap) < tol:
                active.append(('contact', i, j))

    return active


def build_kkt_system(circles, active_constraints):
    """Build the KKT system F(x, lambda) = 0.

    Variables: x = (x1,y1,r1,...,xn,yn,rn) (78 vars)
    Multipliers: lambda_k for each active constraint (78 multipliers)

    KKT conditions:
    1. grad L = 0: -grad(sum ri) + sum_k lambda_k * grad(gk) = 0
    2. gk(x) = 0 for each active constraint k
    """
    n = len(circles)
    n_vars = 3 * n
    n_con = len(active_constraints)

    def kkt_residual(z):
        x = z[:n_vars]
        lam = z[n_vars:]

        xs = x[0::3]; ys = x[1::3]; rs = x[2::3]

        # Gradient of objective: -grad(sum ri)
        grad_obj = np.zeros(n_vars)
        grad_obj[2::3] = -1.0  # d/dri of -sum(ri)

        # For each active constraint, compute gradient and constraint value
        grad_L = grad_obj.copy()  # Will add lambda * grad(g)
        con_vals = np.zeros(n_con)

        for k, con in enumerate(active_constraints):
            if con[0] == 'wall_left':
                i = con[1]
                # g = x_i - r_i >= 0 (active: x_i - r_i = 0)
                con_vals[k] = xs[i] - rs[i]
                # grad_g: d/dxi = 1, d/dri = -1
                grad_L[3 * i] += lam[k] * 1.0      # d/dxi
                grad_L[3 * i + 2] += lam[k] * (-1.0)  # d/dri

            elif con[0] == 'wall_right':
                i = con[1]
                # g = 1 - x_i - r_i >= 0
                con_vals[k] = 1.0 - xs[i] - rs[i]
                grad_L[3 * i] += lam[k] * (-1.0)
                grad_L[3 * i + 2] += lam[k] * (-1.0)

            elif con[0] == 'wall_bottom':
                i = con[1]
                # g = y_i - r_i >= 0
                con_vals[k] = ys[i] - rs[i]
                grad_L[3 * i + 1] += lam[k] * 1.0
                grad_L[3 * i + 2] += lam[k] * (-1.0)

            elif con[0] == 'wall_top':
                i = con[1]
                # g = 1 - y_i - r_i >= 0
                con_vals[k] = 1.0 - ys[i] - rs[i]
                grad_L[3 * i + 1] += lam[k] * (-1.0)
                grad_L[3 * i + 2] += lam[k] * (-1.0)

            elif con[0] == 'contact':
                i, j = con[1], con[2]
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist_sq = dx * dx + dy * dy
                dist = math.sqrt(dist_sq)
                sr = rs[i] + rs[j]
                # g = dist^2 - (ri+rj)^2 >= 0 (using squared form for smoothness)
                con_vals[k] = dist_sq - sr * sr

                # Gradients of dist^2 - sr^2
                grad_L[3 * i] += lam[k] * 2 * dx       # d/dxi
                grad_L[3 * i + 1] += lam[k] * 2 * dy   # d/dyi
                grad_L[3 * j] += lam[k] * (-2 * dx)     # d/dxj
                grad_L[3 * j + 1] += lam[k] * (-2 * dy) # d/dyj
                grad_L[3 * i + 2] += lam[k] * (-2 * sr) # d/dri
                grad_L[3 * j + 2] += lam[k] * (-2 * sr) # d/drj

        residual = np.concatenate([grad_L, con_vals])
        return residual

    return kkt_residual


def main():
    # Load best solution
    base = load_solution(WORKTREE / "orbits/topo-001/solution_n26.json")
    try:
        current = load_solution(OUTPUT_DIR / "solution_n26.json")
        if sum_radii(current) > sum_radii(base):
            base = current
    except:
        pass

    base_metric = sum_radii(base)
    flush_print(f"Starting metric: {base_metric:.10f}")
    valid, viol = validate(base)
    flush_print(f"Valid: {valid}, violation: {viol:.2e}")

    # Identify active constraints
    active = identify_active_constraints(base, tol=1e-6)
    flush_print(f"Active constraints: {len(active)}")
    n_wall = sum(1 for c in active if c[0].startswith('wall'))
    n_contact = sum(1 for c in active if c[0] == 'contact')
    flush_print(f"  Wall: {n_wall}, Contact: {n_contact}")
    flush_print(f"  Variables: {3 * N} = {3 * N}")
    flush_print(f"  DOF: {3 * N - len(active)}")

    if len(active) != 3 * N:
        flush_print(f"WARNING: system not fully determined ({len(active)} constraints "
                     f"vs {3*N} variables)")
        # Try different tolerances
        for tol in [1e-5, 1e-4, 1e-3, 5e-4]:
            active_t = identify_active_constraints(base, tol=tol)
            if len(active_t) >= 3 * N:
                flush_print(f"  Using tol={tol}: {len(active_t)} constraints")
                active = active_t[:3 * N]  # Take exactly 78
                break

    # Build and solve KKT system
    kkt_fun = build_kkt_system(base, active)

    # Initial guess for multipliers
    x0 = base.flatten()
    # Estimate multipliers from constraint activity
    lam0 = np.ones(len(active)) * 0.1

    z0 = np.concatenate([x0, lam0])

    flush_print(f"\nSolving KKT system ({len(z0)} variables)...")
    flush_print(f"Initial KKT residual: {np.max(np.abs(kkt_fun(z0))):.2e}")

    # Try fsolve
    z_sol, info, ier, msg = fsolve(kkt_fun, z0, full_output=True, maxfev=50000)
    flush_print(f"fsolve status: {ier}, message: {msg}")
    flush_print(f"Final KKT residual: {np.max(np.abs(info['fvec'])):.2e}")

    x_sol = z_sol[:3 * N].reshape(N, 3)
    lam_sol = z_sol[3 * N:]

    sol_metric = sum_radii(x_sol)
    sol_valid, sol_viol = validate(x_sol)
    flush_print(f"\nKKT solution: metric={sol_metric:.10f}, valid={sol_valid}, viol={sol_viol:.2e}")
    flush_print(f"Multipliers range: [{lam_sol.min():.4f}, {lam_sol.max():.4f}]")
    flush_print(f"Positive multipliers: {np.sum(lam_sol > 0)}/{len(lam_sol)}")

    if sol_valid and sol_metric > base_metric + 1e-14:
        flush_print(f"IMPROVEMENT: {base_metric:.10f} -> {sol_metric:.10f}")
    else:
        flush_print(f"No improvement (diff: {sol_metric - base_metric:.2e})")

    # Try SLSQP polish on KKT solution
    flush_print("\nSLSQP polish on KKT solution...")
    n = N
    nn = 3 * n
    def objective(x): return -np.sum(x[2::3])
    def grad_obj(x):
        g = np.zeros(nn); g[2::3] = -1.0; return g
    def all_con(x):
        xs = x[0::3]; ys = x[1::3]; rs = x[2::3]
        vals = []
        vals.extend(xs - rs); vals.extend(1.0 - xs - rs)
        vals.extend(ys - rs); vals.extend(1.0 - ys - rs)
        vals.extend(rs - 1e-6)
        for i in range(n):
            dx = xs[i] - xs[i+1:]; dy = ys[i] - ys[i+1:]
            vals.extend(dx*dx + dy*dy - (rs[i] + rs[i+1:])**2)
        return np.array(vals)

    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * n
    result = minimize(objective, x_sol.flatten(), method='SLSQP', jac=grad_obj,
                      bounds=bounds, constraints=[{'type': 'ineq', 'fun': all_con}],
                      options={'maxiter': 20000, 'ftol': 1e-15})
    polished = result.x.reshape(n, 3)
    p_valid, p_viol = validate(polished)
    p_metric = -result.fun if p_valid else 0.0
    flush_print(f"Polished: metric={p_metric:.10f}, valid={p_valid}, viol={p_viol:.2e}")

    # Also try SLSQP directly from original with more iterations
    flush_print("\nDirect SLSQP from original (maxiter=50000)...")
    result2 = minimize(objective, base.flatten(), method='SLSQP', jac=grad_obj,
                       bounds=bounds, constraints=[{'type': 'ineq', 'fun': all_con}],
                       options={'maxiter': 50000, 'ftol': 1e-16})
    direct = result2.x.reshape(n, 3)
    d_valid, d_viol = validate(direct)
    d_metric = -result2.fun if d_valid else 0.0
    flush_print(f"Direct: metric={d_metric:.10f}, valid={d_valid}, viol={d_viol:.2e}")

    # Pick best
    best = base_metric
    best_circles = base.copy()
    for m, c, v in [(sol_metric, x_sol, sol_valid),
                     (p_metric, polished, p_valid),
                     (d_metric, direct, d_valid)]:
        if v and m > best:
            best = m
            best_circles = c.copy()

    flush_print(f"\nFINAL: {best:.10f}")
    flush_print(f"Improvement: {best - base_metric:.2e}")

    try:
        current = load_solution(OUTPUT_DIR / "solution_n26.json")
        current_m = sum_radii(current)
    except:
        current_m = 0

    if best > current_m:
        save_solution(best_circles, OUTPUT_DIR / "solution_n26.json")
        flush_print("Saved!")
    else:
        flush_print(f"No improvement over current ({current_m:.10f})")


if __name__ == "__main__":
    main()
