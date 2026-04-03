"""v4: Differential evolution + targeted refinement from best known solution.

Two strategies:
1. DE for global exploration (finds new topologies)
2. Intensive local refinement from best known (many fast penalty perturbations,
   batch SLSQP polish on top candidates)
"""

import json
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution
from pathlib import Path
import time
import sys
import functools

print = functools.partial(print, flush=True)

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

def slsqp_polish(x, n, constraints, bounds):
    def obj(x):
        return -np.sum(x.reshape(n, 3)[:, 2])
    res = minimize(obj, x, method='SLSQP', constraints=constraints, bounds=bounds,
                  options={'maxiter': 5000, 'ftol': 1e-15})
    return res.x, -res.fun

def penalty_optimize(x, n, bounds):
    for pw in [200, 1000, 5000, 20000]:
        obj = make_penalty_obj(n, pw)
        try:
            res = minimize(obj, x, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 500, 'ftol': 1e-14})
            x = res.x
        except:
            pass
    return x

def refine_from_best(solution_path, n, timeout=400, seed=42):
    """Intensive refinement: many fast penalty perturbations, batch SLSQP."""
    rng = np.random.default_rng(seed)

    with open(solution_path) as f:
        data = json.load(f)
    circles = data['circles']
    x_best = np.array(circles).flatten()
    best_metric = sum(c[2] for c in circles)

    lbfgsb_bounds = [(0.001, 0.999)] * (3*n)
    for i in range(n):
        lbfgsb_bounds[3*i+2] = (0.001, 0.499)

    slsqp_constraints = build_slsqp_constraints(n)
    slsqp_bounds = [(1e-6, 1-1e-6)] * (3*n)
    for i in range(n):
        slsqp_bounds[3*i+2] = (1e-8, 0.5)

    t0 = time.time()
    candidates = []  # (penalty_metric, x)

    print(f"Phase 1: Fast penalty perturbations from {best_metric:.6f}")

    # Generate many perturbations quickly (penalty only, no SLSQP)
    n_fast = 0
    while time.time() - t0 < timeout * 0.5:
        x_pert = x_best.copy()

        # Random perturbation strategy
        strategy = rng.integers(0, 6)
        if strategy == 0:  # Small shift few
            k = rng.integers(1, 4)
            idx = rng.choice(n, k, replace=False)
            for i in idx:
                x_pert[3*i] += rng.normal(0, 0.015)
                x_pert[3*i+1] += rng.normal(0, 0.015)
                x_pert[3*i+2] += rng.normal(0, 0.005)
        elif strategy == 1:  # Medium shift many
            k = rng.integers(3, n//2)
            idx = rng.choice(n, k, replace=False)
            for i in idx:
                x_pert[3*i] += rng.normal(0, 0.04)
                x_pert[3*i+1] += rng.normal(0, 0.04)
                x_pert[3*i+2] += rng.normal(0, 0.015)
        elif strategy == 2:  # Grow all slightly
            for i in range(n):
                x_pert[3*i+2] *= rng.uniform(1.001, 1.03)
        elif strategy == 3:  # Swap two positions
            i, j = rng.choice(n, 2, replace=False)
            x_pert[3*i], x_pert[3*j] = x_pert[3*j], x_pert[3*i]
            x_pert[3*i+1], x_pert[3*j+1] = x_pert[3*j+1], x_pert[3*i+1]
        elif strategy == 4:  # Relocate smallest to random
            radii = [(x_pert[3*i+2], i) for i in range(n)]
            radii.sort()
            _, i = radii[0]
            x_pert[3*i] = rng.uniform(0.05, 0.95)
            x_pert[3*i+1] = rng.uniform(0.05, 0.95)
        elif strategy == 5:  # Shift all slightly
            x_pert += rng.normal(0, 0.008, len(x_pert))

        # Clip
        for i in range(n):
            x_pert[3*i+2] = np.clip(x_pert[3*i+2], 0.002, 0.499)
            x_pert[3*i] = np.clip(x_pert[3*i], x_pert[3*i+2]+0.001, 1-x_pert[3*i+2]-0.001)
            x_pert[3*i+1] = np.clip(x_pert[3*i+1], x_pert[3*i+2]+0.001, 1-x_pert[3*i+2]-0.001)

        # Quick penalty optimize
        x_opt = penalty_optimize(x_pert, n, lbfgsb_bounds)
        metric = sum(x_opt.reshape(n, 3)[:, 2])
        candidates.append((metric, x_opt.copy()))
        n_fast += 1

        if n_fast % 10 == 0:
            candidates.sort(key=lambda c: -c[0])
            print(f"  {n_fast} perturbations, top penalty metric: {candidates[0][0]:.6f}")

    # Sort and take top candidates for SLSQP
    candidates.sort(key=lambda c: -c[0])
    print(f"\nPhase 2: SLSQP polish on top {min(15, len(candidates))} of {len(candidates)} candidates")

    for ci in range(min(15, len(candidates))):
        if time.time() - t0 > timeout * 0.95:
            print(f"  Timeout at candidate {ci}")
            break
        _, x_cand = candidates[ci]
        try:
            x_pol, metric = slsqp_polish(x_cand, n, slsqp_constraints, slsqp_bounds)
            circles_pol = x_pol.reshape(n, 3).tolist()
            valid = validate(circles_pol)
            if valid and metric > best_metric:
                improvement = metric - best_metric
                best_metric = metric
                x_best = x_pol.copy()
                print(f"  Candidate {ci+1}: NEW BEST {best_metric:.6f} (+{improvement:.6f})")
            else:
                print(f"  Candidate {ci+1}: {metric:.6f} valid={valid}")
        except Exception as e:
            print(f"  Candidate {ci+1}: error {e}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Best: {best_metric:.6f}")
    return x_best.reshape(n, 3).tolist(), best_metric

def save_solution(circles, path):
    with open(path, 'w') as f:
        json.dump({"circles": circles}, f, indent=2)

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 26
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    timeout = int(sys.argv[3]) if len(sys.argv) > 3 else 400

    sol_path = Path(__file__).parent / f"solution_n{n}_best.json"
    if not sol_path.exists():
        sol_path = Path(__file__).parent / f"solution_n{n}.json"

    print(f"Refining n={n} from {sol_path}, seed={seed}, timeout={timeout}s")

    solution, metric = refine_from_best(str(sol_path), n, timeout=timeout, seed=seed)

    if solution is not None:
        out = Path(__file__).parent / f"solution_n{n}.json"
        out_best = Path(__file__).parent / f"solution_n{n}_best.json"
        # Only overwrite if better
        if out_best.exists():
            with open(out_best) as f:
                old = json.load(f)
            old_metric = sum(c[2] for c in old['circles'])
            if metric > old_metric:
                save_solution(solution, out)
                save_solution(solution, out_best)
                print(f"Saved new best: {metric:.6f} > {old_metric:.6f}")
            else:
                print(f"No improvement: {metric:.6f} <= {old_metric:.6f}")
        else:
            save_solution(solution, out)
            save_solution(solution, out_best)
            print(f"Saved: {metric:.6f}")
