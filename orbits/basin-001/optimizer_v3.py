"""Circle packing optimizer v3 - Fast multi-start with penalty + SLSQP polish.

Strategy:
1. Generate many initializations quickly
2. For each: quick penalty-based optimization with L-BFGS-B
3. SLSQP polish on top candidates
4. Perturbation + re-optimize on best solution
"""

import json
import math
import numpy as np
from scipy.optimize import minimize
from pathlib import Path
import time
import sys
import functools

# Force unbuffered output
print = functools.partial(print, flush=True)

def make_penalty_obj(n, pw):
    """Fast penalty objective."""
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
        # Vectorized overlap
        for i in range(n):
            dx = xs[i] - xs[i+1:]
            dy = ys[i] - ys[i+1:]
            dists = np.sqrt(dx*dx + dy*dy + 1e-30)
            overlaps = np.maximum(0, rs[i] + rs[i+1:] - dists)
            p += np.sum(overlaps**2)
        return obj + pw * p
    return f

def make_bounds(n):
    bounds = []
    for _ in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
    return bounds

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

# ─── Initialization ───

def init_hex(n, rng):
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    r = 0.42 / max(cols, rows)
    circles = []
    for row in range(rows+1):
        for col in range(cols+1):
            if len(circles) >= n: break
            x = (col + 0.5*(row%2) + 0.5) / (cols+1)
            y = (row + 0.5) / (rows+1)
            circles.append([np.clip(x, 0.05, 0.95), np.clip(y, 0.05, 0.95), r])
        if len(circles) >= n: break
    arr = np.array(circles[:n]).flatten()
    arr += rng.normal(0, 0.005, len(arr))
    for i in range(n):
        arr[3*i+2] = np.clip(abs(arr[3*i+2]), 0.005, 0.45)
        arr[3*i] = np.clip(arr[3*i], arr[3*i+2]+0.002, 1-arr[3*i+2]-0.002)
        arr[3*i+1] = np.clip(arr[3*i+1], arr[3*i+2]+0.002, 1-arr[3*i+2]-0.002)
    return arr

def init_greedy_fast(n, rng, n_cand=100):
    """Fast greedy: fewer candidates per circle."""
    circles = []
    for _ in range(n):
        best_r, best_x, best_y = 0, 0.5, 0.5
        for _ in range(n_cand):
            x = rng.uniform(0.02, 0.98)
            y = rng.uniform(0.02, 0.98)
            r_max = min(x, 1-x, y, 1-y)
            for cx, cy, cr in circles:
                d = math.sqrt((x-cx)**2 + (y-cy)**2)
                r_max = min(r_max, d - cr)
            if r_max > best_r:
                best_r, best_x, best_y = r_max, x, y
        circles.append([best_x, best_y, max(best_r, 0.001)])
    return np.array(circles).flatten()

def init_random(n, rng):
    r = 0.35 / math.sqrt(n)
    c = []
    for _ in range(n):
        x = rng.uniform(r+0.01, 1-r-0.01)
        y = rng.uniform(r+0.01, 1-r-0.01)
        c.append([x, y, r * rng.uniform(0.5, 1.2)])
    return np.array(c).flatten()

def init_mixed_sizes(n, rng):
    """Some large, some small circles."""
    circles = []
    # 4-6 large circles
    n_large = min(n // 4, 6)
    r_large = 0.5 / (math.sqrt(n_large) + 1)
    for i in range(n_large):
        angle = 2 * math.pi * i / n_large
        x = 0.5 + 0.3 * math.cos(angle)
        y = 0.5 + 0.3 * math.sin(angle)
        circles.append([x, y, r_large * rng.uniform(0.8, 1.0)])
    # Rest smaller, in gaps
    r_small = r_large * 0.4
    for _ in range(n - n_large):
        x = rng.uniform(0.05, 0.95)
        y = rng.uniform(0.05, 0.95)
        circles.append([x, y, r_small * rng.uniform(0.5, 1.5)])
    return np.array(circles[:n]).flatten()

# ─── Validation ───

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

def max_violation(circles_list):
    n = len(circles_list)
    mv = 0.0
    for x, y, r in circles_list:
        mv = max(mv, r-x, x+r-1, r-y, y+r-1, -r)
    for i in range(n):
        xi, yi, ri = circles_list[i]
        for j in range(i+1, n):
            xj, yj, rj = circles_list[j]
            d = math.sqrt((xi-xj)**2+(yi-yj)**2)
            mv = max(mv, ri+rj-d)
    return mv

# ─── Perturbation for local search ───

def perturb(x, n, rng, scale=0.03):
    """Small perturbation of solution."""
    x_new = x.copy()
    # Pick 1-3 circles to perturb
    k = rng.integers(1, min(4, n+1))
    idx = rng.choice(n, k, replace=False)
    for i in idx:
        x_new[3*i] += rng.normal(0, scale)
        x_new[3*i+1] += rng.normal(0, scale)
        x_new[3*i+2] += rng.normal(0, scale * 0.3)
    # Clip
    for i in range(n):
        x_new[3*i+2] = np.clip(x_new[3*i+2], 0.002, 0.499)
        x_new[3*i] = np.clip(x_new[3*i], x_new[3*i+2]+0.001, 1-x_new[3*i+2]-0.001)
        x_new[3*i+1] = np.clip(x_new[3*i+1], x_new[3*i+2]+0.001, 1-x_new[3*i+2]-0.001)
    return x_new

# ─── Main solver ───

def solve(n=26, n_starts=20, n_perturb=30, seed=42, verbose=True, timeout=400):
    rng = np.random.default_rng(seed)
    bounds = make_bounds(n)
    t0 = time.time()

    best_metric = 0.0
    best_x = None
    all_results = []

    inits = [init_hex, init_greedy_fast, init_random, init_mixed_sizes]

    # Phase 1: Multi-start with progressive penalty
    if verbose: print("=== Phase 1: Multi-start optimization ===")

    for si in range(n_starts):
        if time.time() - t0 > timeout * 0.6:
            if verbose: print(f"  Phase 1 timeout at start {si}")
            break

        init_fn = inits[si % len(inits)]
        x0 = init_fn(n, rng)

        # Progressive penalty optimization
        x = x0.copy()
        for pw in [50, 200, 1000, 5000, 20000]:
            obj = make_penalty_obj(n, pw)
            try:
                res = minimize(obj, x, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 500, 'ftol': 1e-14})
                x = res.x
            except:
                pass

        circles = x.reshape(n, 3).tolist()
        metric = sum(c[2] for c in circles)
        valid = validate(circles)
        mv = max_violation(circles)

        if verbose and si < 10 or si % 5 == 0:
            print(f"  Start {si+1}: metric={metric:.4f}, valid={valid}, viol={mv:.2e}")

        all_results.append((metric, valid, x.copy()))

        if valid and metric > best_metric:
            best_metric = metric
            best_x = x.copy()
            if verbose: print(f"    *** NEW BEST: {best_metric:.6f} ***")

    # Phase 2: SLSQP polish on top candidates
    if verbose: print(f"\n=== Phase 2: SLSQP polish (best so far: {best_metric:.6f}) ===")

    # Sort by metric, take top candidates (valid or nearly valid)
    all_results.sort(key=lambda r: r[0], reverse=True)
    n_polish = min(8, len(all_results))
    slsqp_constraints = build_slsqp_constraints(n)
    slsqp_bounds = []
    for _ in range(n):
        slsqp_bounds.extend([(1e-6, 1-1e-6), (1e-6, 1-1e-6), (1e-8, 0.5)])

    def slsqp_obj(x):
        return -np.sum(x.reshape(n, 3)[:, 2])

    for pi in range(n_polish):
        if time.time() - t0 > timeout * 0.8:
            if verbose: print(f"  Phase 2 timeout at {pi}")
            break

        _, _, x_cand = all_results[pi]
        try:
            res = minimize(slsqp_obj, x_cand, method='SLSQP',
                         constraints=slsqp_constraints, bounds=slsqp_bounds,
                         options={'maxiter': 3000, 'ftol': 1e-15})
            circles = res.x.reshape(n, 3).tolist()
            metric = -res.fun
            valid = validate(circles)
            if verbose:
                print(f"  Polish {pi+1}: metric={metric:.6f}, valid={valid}")
            if valid and metric > best_metric:
                best_metric = metric
                best_x = res.x.copy()
                if verbose: print(f"    *** NEW BEST: {best_metric:.6f} ***")
        except Exception as e:
            if verbose: print(f"  Polish {pi+1}: error {e}")

    # Phase 3: Perturbation + re-optimize on best
    if best_x is not None and verbose:
        print(f"\n=== Phase 3: Perturbation search (best: {best_metric:.6f}) ===")

    no_improve = 0
    for pi in range(n_perturb):
        if time.time() - t0 > timeout * 0.95:
            if verbose: print(f"  Phase 3 timeout at {pi}")
            break
        if no_improve >= 10:
            if verbose: print(f"  No improvement for 10 perturbations, stopping")
            break

        x_pert = perturb(best_x, n, rng, scale=0.02 + 0.02 * rng.random())

        # Quick penalty opt
        for pw in [1000, 10000]:
            obj = make_penalty_obj(n, pw)
            try:
                res = minimize(obj, x_pert, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 300, 'ftol': 1e-14})
                x_pert = res.x
            except:
                pass

        # SLSQP polish
        try:
            res = minimize(slsqp_obj, x_pert, method='SLSQP',
                         constraints=slsqp_constraints, bounds=slsqp_bounds,
                         options={'maxiter': 2000, 'ftol': 1e-15})
            circles = res.x.reshape(n, 3).tolist()
            metric = -res.fun
            valid = validate(circles)
            if valid and metric > best_metric:
                best_metric = metric
                best_x = res.x.copy()
                no_improve = 0
                if verbose: print(f"  Perturb {pi+1}: *** NEW BEST: {best_metric:.6f} ***")
            else:
                no_improve += 1
                if verbose and pi < 5:
                    print(f"  Perturb {pi+1}: metric={metric:.6f}, valid={valid}")
        except:
            no_improve += 1

    elapsed = time.time() - t0
    if verbose:
        print(f"\n=== DONE in {elapsed:.1f}s ===")
        print(f"Best metric: {best_metric:.6f}")

    return best_x.reshape(n, 3).tolist() if best_x is not None else None, best_metric, all_results

def save_solution(circles, path):
    with open(path, 'w') as f:
        json.dump({"circles": circles}, f, indent=2)

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 26
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    n_starts = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    timeout = int(sys.argv[4]) if len(sys.argv) > 4 else 400

    print(f"Circle packing v3: n={n}, seed={seed}, starts={n_starts}, timeout={timeout}s")

    solution, metric, _ = solve(n=n, n_starts=n_starts, seed=seed,
                                 verbose=True, timeout=timeout)

    if solution is not None:
        out_path = Path(__file__).parent / f"solution_n{n}.json"
        save_solution(solution, out_path)
        print(f"\nSaved to {out_path}")
    else:
        print("No valid solution found!")
