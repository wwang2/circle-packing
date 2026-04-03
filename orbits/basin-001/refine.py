"""Refine an existing solution with intensive perturbation + SLSQP polish.

Takes the best known solution and applies:
1. Many small perturbations + SLSQP
2. Swap-based perturbations (exchange positions of circles)
3. Squeeze perturbations (try to grow small circles)
4. Topology changes (move a small circle to a gap)
"""

import json
import math
import numpy as np
from scipy.optimize import minimize
from pathlib import Path
import time
import sys
import functools

print = functools.partial(print, flush=True)

def build_slsqp_constraints(n):
    constraints = []
    for i in range(n):
        ix, iy, ir = 3*i, 3*i+1, 3*i+2
        constraints.append({'type': 'ineq', 'fun': lambda x, a=ix, b=ir: x[a] - x[b]})
        constraints.append({'type': 'ineq', 'fun': lambda x, a=ix, b=ir: 1.0 - x[a] - x[b]})
        constraints.append({'type': 'ineq', 'fun': lambda x, a=iy, b=ir: x[a] - x[b]})
        constraints.append({'type': 'ineq', 'fun': lambda x, a=iy, b=ir: 1.0 - x[a] - x[b]})
        constraints.append({'type': 'ineq', 'fun': lambda x, b=ir: x[b] - 1e-8})
    for i in range(n):
        for j in range(i+1, n):
            ai, bi, ci = 3*i, 3*i+1, 3*i+2
            aj, bj, cj = 3*j, 3*j+1, 3*j+2
            def sep(x, ai=ai, bi=bi, ci=ci, aj=aj, bj=bj, cj=cj):
                dx = x[ai]-x[aj]; dy = x[bi]-x[bj]
                return math.sqrt(dx*dx+dy*dy) - x[ci] - x[cj]
            constraints.append({'type': 'ineq', 'fun': sep})
    return constraints

def make_penalty_obj(n, pw):
    def f(x):
        c = x.reshape(n, 3)
        xs, ys, rs = c[:, 0], c[:, 1], c[:, 2]
        obj = -np.sum(rs)
        p = 0.0
        p += np.sum(np.maximum(0, rs - xs)**2)
        p += np.sum(np.maximum(0, xs + rs - 1)**2)
        p += np.sum(np.maximum(0, rs - ys)**2)
        p += np.sum(np.maximum(0, ys + rs - 1)**2)
        p += np.sum(np.maximum(0, -rs)**2)
        for i in range(n):
            dx = xs[i] - xs[i+1:]
            dy = ys[i] - ys[i+1:]
            dists = np.sqrt(dx*dx + dy*dy + 1e-30)
            overlaps = np.maximum(0, rs[i] + rs[i+1:] - dists)
            p += np.sum(overlaps**2)
        return obj + pw * p
    return f

def validate(circles_list, tol=1e-9):
    n = len(circles_list)
    for x, y, r in circles_list:
        if r <= 0 or x-r < -tol or x+r > 1+tol or y-r < -tol or y+r > 1+tol:
            return False
    for i in range(n):
        xi, yi, ri = circles_list[i]
        for j in range(i+1, n):
            xj, yj, rj = circles_list[j]
            if math.sqrt((xi-xj)**2+(yi-yj)**2) < ri+rj-tol:
                return False
    return True

def slsqp_polish(x, n, constraints, bounds, maxiter=3000):
    def obj(x):
        return -np.sum(x.reshape(n, 3)[:, 2])
    res = minimize(obj, x, method='SLSQP', constraints=constraints, bounds=bounds,
                  options={'maxiter': maxiter, 'ftol': 1e-15})
    return res.x, -res.fun

def penalty_optimize(x, n, bounds):
    for pw in [500, 2000, 10000, 50000]:
        obj = make_penalty_obj(n, pw)
        try:
            res = minimize(obj, x, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 500, 'ftol': 1e-14})
            x = res.x
        except:
            pass
    return x

def find_gaps(circles, n):
    """Find the largest empty gaps in the packing."""
    # Sample grid and find points farthest from all circles and walls
    gaps = []
    for gx in np.linspace(0.02, 0.98, 30):
        for gy in np.linspace(0.02, 0.98, 30):
            max_r = min(gx, 1-gx, gy, 1-gy)
            for cx, cy, cr in circles:
                d = math.sqrt((gx-cx)**2 + (gy-cy)**2)
                max_r = min(max_r, d - cr)
            if max_r > 0.01:
                gaps.append((max_r, gx, gy))
    gaps.sort(reverse=True)
    return gaps[:5]

def refine(solution_path, n_perturb=100, timeout=300, seed=42):
    rng = np.random.default_rng(seed)

    with open(solution_path) as f:
        data = json.load(f)
    circles = data['circles']
    n = len(circles)

    x_best = np.array(circles).flatten()
    best_metric = sum(c[2] for c in circles)

    constraints = build_slsqp_constraints(n)
    slsqp_bounds = []
    for _ in range(n):
        slsqp_bounds.extend([(1e-6, 1-1e-6), (1e-6, 1-1e-6), (1e-8, 0.5)])
    lbfgsb_bounds = [(0.001, 0.999)] * (3*n)
    for i in range(n):
        lbfgsb_bounds[3*i+2] = (0.001, 0.499)

    t0 = time.time()
    no_improve = 0

    print(f"Starting refinement from metric={best_metric:.6f}, n={n}")

    for pi in range(n_perturb):
        if time.time() - t0 > timeout:
            print(f"Timeout at perturbation {pi}")
            break
        if no_improve >= 25:
            print(f"No improvement for 25 perturbations, stopping")
            break

        x_pert = x_best.copy()

        # Choose perturbation type
        action = rng.choice(['small_shift', 'medium_shift', 'grow_small',
                            'relocate', 'swap_pos', 'squeeze_all'],
                           p=[0.3, 0.2, 0.15, 0.15, 0.1, 0.1])

        if action == 'small_shift':
            k = rng.integers(1, 4)
            idx = rng.choice(n, k, replace=False)
            for i in idx:
                x_pert[3*i] += rng.normal(0, 0.01)
                x_pert[3*i+1] += rng.normal(0, 0.01)
                x_pert[3*i+2] += rng.normal(0, 0.003)

        elif action == 'medium_shift':
            k = rng.integers(1, 6)
            idx = rng.choice(n, k, replace=False)
            for i in idx:
                x_pert[3*i] += rng.normal(0, 0.03)
                x_pert[3*i+1] += rng.normal(0, 0.03)
                x_pert[3*i+2] += rng.normal(0, 0.01)

        elif action == 'grow_small':
            # Try to grow the smallest circles
            radii = [(x_pert[3*i+2], i) for i in range(n)]
            radii.sort()
            for r, i in radii[:3]:
                x_pert[3*i+2] *= rng.uniform(1.05, 1.3)

        elif action == 'relocate':
            # Move smallest circle to largest gap
            circles_curr = x_pert.reshape(n, 3).tolist()
            gaps = find_gaps(circles_curr, n)
            radii = [(x_pert[3*i+2], i) for i in range(n)]
            radii.sort()
            if gaps:
                _, gx, gy = gaps[0]
                _, i = radii[0]
                x_pert[3*i] = gx
                x_pert[3*i+1] = gy

        elif action == 'swap_pos':
            i, j = rng.choice(n, 2, replace=False)
            x_pert[3*i], x_pert[3*j] = x_pert[3*j], x_pert[3*i]
            x_pert[3*i+1], x_pert[3*j+1] = x_pert[3*j+1], x_pert[3*i+1]

        elif action == 'squeeze_all':
            # Try to grow all radii slightly
            for i in range(n):
                x_pert[3*i+2] *= rng.uniform(1.001, 1.02)

        # Clip
        for i in range(n):
            x_pert[3*i+2] = np.clip(x_pert[3*i+2], 0.002, 0.499)
            x_pert[3*i] = np.clip(x_pert[3*i], x_pert[3*i+2]+0.001, 1-x_pert[3*i+2]-0.001)
            x_pert[3*i+1] = np.clip(x_pert[3*i+1], x_pert[3*i+2]+0.001, 1-x_pert[3*i+2]-0.001)

        # Quick penalty optimization
        x_opt = penalty_optimize(x_pert, n, lbfgsb_bounds)

        # SLSQP polish
        try:
            x_pol, metric = slsqp_polish(x_opt, n, constraints, slsqp_bounds)
            circles_pol = x_pol.reshape(n, 3).tolist()
            valid = validate(circles_pol)

            if valid and metric > best_metric:
                improvement = metric - best_metric
                best_metric = metric
                x_best = x_pol.copy()
                no_improve = 0
                print(f"  Perturb {pi+1} ({action}): NEW BEST {best_metric:.6f} (+{improvement:.6f})")
            else:
                no_improve += 1
        except:
            no_improve += 1

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Best metric: {best_metric:.6f}")

    # Save
    best_circles = x_best.reshape(n, 3).tolist()
    out_path = Path(solution_path)
    with open(out_path, 'w') as f:
        json.dump({"circles": best_circles}, f, indent=2)
    print(f"Saved to {out_path}")

    return best_metric

if __name__ == "__main__":
    solution_path = sys.argv[1] if len(sys.argv) > 1 else "orbits/basin-001/solution_n26.json"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    timeout = int(sys.argv[3]) if len(sys.argv) > 3 else 300

    refine(solution_path, n_perturb=200, timeout=timeout, seed=seed)
