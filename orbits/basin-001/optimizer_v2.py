"""Basin-hopping optimizer v2 for circle packing in unit square.

Key change from v1: Use penalty-based objective with L-BFGS-B for speed,
then final SLSQP polish on best solution only.
"""

import json
import math
import numpy as np
from scipy.optimize import minimize, basinhopping, differential_evolution
from pathlib import Path
import time
import sys

# ─── Problem encoding ───
# Decision vector: [x0, y0, r0, x1, y1, r1, ..., x_{n-1}, y_{n-1}, r_{n-1}]

def penalty_objective(x, n, penalty_weight):
    """Objective: minimize -(sum of radii) + penalty * violations."""
    circles = x.reshape(n, 3)
    xs, ys, rs = circles[:, 0], circles[:, 1], circles[:, 2]

    obj = -np.sum(rs)

    penalty = 0.0
    # Containment violations
    penalty += np.sum(np.maximum(0, rs - xs)**2)
    penalty += np.sum(np.maximum(0, xs + rs - 1)**2)
    penalty += np.sum(np.maximum(0, rs - ys)**2)
    penalty += np.sum(np.maximum(0, ys + rs - 1)**2)
    # Positive radius
    penalty += np.sum(np.maximum(0, 1e-6 - rs)**2)

    # Non-overlap: vectorized
    for i in range(n):
        dx = xs[i] - xs[i+1:]
        dy = ys[i] - ys[i+1:]
        dists = np.sqrt(dx*dx + dy*dy)
        min_dists = rs[i] + rs[i+1:]
        overlaps = np.maximum(0, min_dists - dists)
        penalty += np.sum(overlaps**2)

    return obj + penalty_weight * penalty

def penalty_objective_grad(x, n, penalty_weight):
    """Gradient of penalty objective via finite differences (or analytic)."""
    # Use analytic gradient for speed
    circles = x.reshape(n, 3)
    xs, ys, rs = circles[:, 0], circles[:, 1], circles[:, 2]

    grad = np.zeros_like(x)

    # Gradient of -sum(r)
    for i in range(n):
        grad[3*i+2] = -1.0

    # Containment gradients
    for i in range(n):
        ix, iy, ir = 3*i, 3*i+1, 3*i+2
        # rs - xs > 0 => penalty += (rs-xs)^2 => d/dxs = -2*(rs-xs), d/drs = 2*(rs-xs)
        v = rs[i] - xs[i]
        if v > 0:
            grad[ix] += penalty_weight * (-2*v)
            grad[ir] += penalty_weight * (2*v)
        # xs + rs - 1 > 0
        v = xs[i] + rs[i] - 1
        if v > 0:
            grad[ix] += penalty_weight * (2*v)
            grad[ir] += penalty_weight * (2*v)
        # rs - ys > 0
        v = rs[i] - ys[i]
        if v > 0:
            grad[iy] += penalty_weight * (-2*v)
            grad[ir] += penalty_weight * (2*v)
        # ys + rs - 1 > 0
        v = ys[i] + rs[i] - 1
        if v > 0:
            grad[iy] += penalty_weight * (2*v)
            grad[ir] += penalty_weight * (2*v)

    # Non-overlap gradients
    for i in range(n):
        for j in range(i+1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = math.sqrt(dx*dx + dy*dy)
            min_dist = rs[i] + rs[j]
            overlap = min_dist - dist
            if overlap > 0 and dist > 1e-12:
                # d(overlap)/d(xi) = -(dx/dist), etc.
                factor = penalty_weight * 2 * overlap
                grad[3*i]   += factor * (dx / dist)
                grad[3*i+1] += factor * (dy / dist)
                grad[3*j]   -= factor * (dx / dist)
                grad[3*j+1] -= factor * (dy / dist)
                grad[3*i+2] += factor * (-1)  # wait, d(overlap)/d(ri) = 1 -> factor * 1
                grad[3*j+2] += factor * (-1)  # same for rj
                # Correction: overlap = ri + rj - dist
                # d(overlap)/d(ri) = 1, d/d(rj) = 1
                # penalty += overlap^2, d/d(ri) = 2*overlap * 1
                # But we want to PUSH apart so gradient should increase dist
                # Actually: penalty = overlap^2, grad wrt ri = 2*overlap * d(overlap)/d(ri) = 2*overlap*1
                # This means grad[ir] += 2*overlap (positive, discouraging large r)
                # Recalculate properly:
                grad[3*i+2] += -factor  # undo wrong
                grad[3*j+2] += -factor  # undo wrong
                grad[3*i+2] += factor   # d/dri of overlap^2 = 2*overlap*1
                grad[3*j+2] += factor   # d/drj of overlap^2 = 2*overlap*1
                # d/dxi of overlap^2 = 2*overlap*(-dx/dist) (since d(dist)/dxi = dx/dist)
                # Already handled above with factor*(dx/dist) ... but sign:
                # overlap = ri+rj-dist, d(overlap)/dxi = -dx/dist
                # d(overlap^2)/dxi = 2*overlap*(-dx/dist)
                # So: grad[3*i] += pw * 2*overlap*(-dx/dist)
                # But above I wrote: grad[3*i] += factor * (dx/dist) = pw*2*overlap*(dx/dist)
                # That's WRONG sign! Let me fix.
                grad[3*i]   -= 2 * factor * (dx / dist)  # undo and redo with correct sign
                grad[3*i+1] -= 2 * factor * (dy / dist)
                grad[3*j]   += 2 * factor * (dx / dist)
                grad[3*j+1] += 2 * factor * (dy / dist)

    return grad

# Actually, let me just use numerical gradients via L-BFGS-B. The analytic gradient
# above has bugs. For n=26 with L-BFGS-B, numerical diffs are fast enough.

def make_objective(n, penalty_weight):
    """Create objective closure."""
    def f(x):
        circles = x.reshape(n, 3)
        xs, ys, rs = circles[:, 0], circles[:, 1], circles[:, 2]
        obj = -np.sum(rs)
        penalty = 0.0
        penalty += np.sum(np.maximum(0, rs - xs)**2)
        penalty += np.sum(np.maximum(0, xs + rs - 1)**2)
        penalty += np.sum(np.maximum(0, rs - ys)**2)
        penalty += np.sum(np.maximum(0, ys + rs - 1)**2)
        penalty += np.sum(np.maximum(0, 1e-6 - rs)**2)
        for i in range(n):
            dx = xs[i] - xs[i+1:]
            dy = ys[i] - ys[i+1:]
            dists = np.sqrt(dx*dx + dy*dy + 1e-30)
            min_dists = rs[i] + rs[i+1:]
            overlaps = np.maximum(0, min_dists - dists)
            penalty += np.sum(overlaps**2)
        return obj + penalty_weight * penalty
    return f

# ─── Initialization strategies ───

def init_hexagonal_grid(n, rng):
    cols = int(math.ceil(math.sqrt(n * 2 / math.sqrt(3))))
    rows = int(math.ceil(n / cols))
    r_est = 0.45 / max(cols, rows * math.sqrt(3)/2)
    circles = []
    for row in range(rows + 2):
        for col in range(cols + 2):
            if len(circles) >= n:
                break
            x = (col + 0.5 * (row % 2) + 0.5) / (cols + 1)
            y = (row * math.sqrt(3) / 2 + 0.5) / (rows * math.sqrt(3)/2 + 1)
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            circles.append([x, y, r_est])
        if len(circles) >= n:
            break
    circles = circles[:n]
    result = np.array(circles).flatten()
    result += rng.normal(0, 0.003, len(result))
    for i in range(n):
        result[3*i+2] = max(abs(result[3*i+2]), 0.005)
        result[3*i] = np.clip(result[3*i], result[3*i+2]+0.001, 1-result[3*i+2]-0.001)
        result[3*i+1] = np.clip(result[3*i+1], result[3*i+2]+0.001, 1-result[3*i+2]-0.001)
    return result

def init_random_greedy(n, rng):
    circles = []
    for i in range(n):
        best_r = 0
        best_x, best_y = 0.5, 0.5
        for _ in range(500):
            x = rng.uniform(0.02, 0.98)
            y = rng.uniform(0.02, 0.98)
            r_max = min(x, 1-x, y, 1-y)
            for cx, cy, cr in circles:
                dist = math.sqrt((x-cx)**2 + (y-cy)**2)
                r_max = min(r_max, dist - cr)
            if r_max > best_r:
                best_r = r_max
                best_x, best_y = x, y
        if best_r < 1e-6:
            best_r = 1e-4
        circles.append([best_x, best_y, best_r])
    return np.array(circles).flatten()

def init_poisson_disk(n, rng):
    min_dist = 0.8 / math.sqrt(n)
    points = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        x = rng.uniform(0.05, 0.95)
        y = rng.uniform(0.05, 0.95)
        ok = all(math.sqrt((x-px)**2 + (y-py)**2) >= min_dist for px, py in points)
        if ok:
            points.append((x, y))
        attempts += 1
    while len(points) < n:
        points.append((rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95)))
    r_est = min_dist / 2.5
    circles = [[px, py, r_est] for px, py in points[:n]]
    return np.array(circles).flatten()

def init_random_uniform(n, rng):
    r_est = 0.4 / math.sqrt(n)
    circles = []
    for _ in range(n):
        x = rng.uniform(r_est + 0.01, 1 - r_est - 0.01)
        y = rng.uniform(r_est + 0.01, 1 - r_est - 0.01)
        circles.append([x, y, r_est])
    return np.array(circles).flatten()

def init_concentric(n, rng):
    rings = []
    remaining = n
    ring_idx = 0
    while remaining > 0:
        count = 1 if ring_idx == 0 else min(6 * ring_idx, remaining)
        rings.append(count)
        remaining -= count
        ring_idx += 1
    r_est = 1.0 / (2 * (len(rings) + 1))
    circles = []
    for ring_i, count in enumerate(rings):
        if ring_i == 0:
            circles.append([0.5, 0.5, r_est])
        else:
            ring_r = ring_i * 2 * r_est
            for k in range(count):
                angle = 2 * math.pi * k / count
                x = np.clip(0.5 + ring_r * math.cos(angle), r_est+0.01, 1-r_est-0.01)
                y = np.clip(0.5 + ring_r * math.sin(angle), r_est+0.01, 1-r_est-0.01)
                circles.append([x, y, r_est * 0.8])
    return np.array(circles[:n]).flatten()

# ─── Custom step for basin-hopping ───

class PackingStep:
    def __init__(self, n, stepsize=0.05, rng=None):
        self.n = n
        self.stepsize = stepsize
        self.rng = rng or np.random.default_rng(42)

    def __call__(self, x):
        n = self.n
        x_new = x.copy()
        action = self.rng.choice(['shift_one', 'shift_few', 'swap', 'resize', 'big_move'],
                                  p=[0.3, 0.25, 0.1, 0.2, 0.15])

        if action == 'shift_one':
            i = self.rng.integers(0, n)
            x_new[3*i] += self.rng.normal(0, self.stepsize)
            x_new[3*i+1] += self.rng.normal(0, self.stepsize)
            x_new[3*i+2] += self.rng.normal(0, self.stepsize * 0.3)
        elif action == 'shift_few':
            k = self.rng.integers(2, max(3, n//4))
            indices = self.rng.choice(n, k, replace=False)
            for i in indices:
                x_new[3*i] += self.rng.normal(0, self.stepsize * 0.5)
                x_new[3*i+1] += self.rng.normal(0, self.stepsize * 0.5)
                x_new[3*i+2] += self.rng.normal(0, self.stepsize * 0.15)
        elif action == 'swap':
            i, j = self.rng.choice(n, 2, replace=False)
            x_new[3*i], x_new[3*j] = x_new[3*j], x_new[3*i]
            x_new[3*i+1], x_new[3*j+1] = x_new[3*j+1], x_new[3*i+1]
        elif action == 'resize':
            k = self.rng.integers(1, max(2, n//4))
            indices = self.rng.choice(n, k, replace=False)
            for i in indices:
                x_new[3*i+2] *= self.rng.uniform(0.6, 1.4)
        elif action == 'big_move':
            i = self.rng.integers(0, n)
            x_new[3*i] = self.rng.uniform(0.05, 0.95)
            x_new[3*i+1] = self.rng.uniform(0.05, 0.95)
            x_new[3*i+2] = self.rng.uniform(0.01, 0.15)

        # Clip
        for i in range(n):
            x_new[3*i+2] = np.clip(x_new[3*i+2], 0.001, 0.499)
            x_new[3*i] = np.clip(x_new[3*i], x_new[3*i+2]+0.001, 1-x_new[3*i+2]-0.001)
            x_new[3*i+1] = np.clip(x_new[3*i+1], x_new[3*i+2]+0.001, 1-x_new[3*i+2]-0.001)
        return x_new

# ─── SLSQP constraint-based polish ───

def build_constraints(n):
    constraints = []
    for i in range(n):
        ix, iy, ir = 3*i, 3*i+1, 3*i+2
        constraints.append({'type': 'ineq', 'fun': lambda x, ix=ix, ir=ir: x[ix] - x[ir]})
        constraints.append({'type': 'ineq', 'fun': lambda x, ix=ix, ir=ir: 1.0 - x[ix] - x[ir]})
        constraints.append({'type': 'ineq', 'fun': lambda x, iy=iy, ir=ir: x[iy] - x[ir]})
        constraints.append({'type': 'ineq', 'fun': lambda x, iy=iy, ir=ir: 1.0 - x[iy] - x[ir]})
        constraints.append({'type': 'ineq', 'fun': lambda x, ir=ir: x[ir] - 1e-8})
    for i in range(n):
        for j in range(i+1, n):
            ixi, iyi, iri = 3*i, 3*i+1, 3*i+2
            ixj, iyj, irj = 3*j, 3*j+1, 3*j+2
            def sep(x, ixi=ixi, iyi=iyi, iri=iri, ixj=ixj, iyj=iyj, irj=irj):
                dx = x[ixi] - x[ixj]
                dy = x[iyi] - x[iyj]
                return math.sqrt(dx*dx + dy*dy) - x[iri] - x[irj]
            constraints.append({'type': 'ineq', 'fun': sep})
    return constraints

def slsqp_polish(x, n):
    """Final polish with SLSQP constraints."""
    constraints = build_constraints(n)
    bounds = []
    for i in range(n):
        bounds.append((1e-6, 1.0 - 1e-6))
        bounds.append((1e-6, 1.0 - 1e-6))
        bounds.append((1e-8, 0.5))

    def obj(x):
        return -np.sum(x.reshape(n, 3)[:, 2])

    result = minimize(obj, x, method='SLSQP', constraints=constraints, bounds=bounds,
                     options={'maxiter': 5000, 'ftol': 1e-15, 'disp': False})
    return result.x, -result.fun

# ─── Validation ───

def validate_solution(circles_list, tol=1e-9):
    n = len(circles_list)
    for i, (x, y, r) in enumerate(circles_list):
        if r <= 0: return False
        if x - r < -tol or x + r > 1 + tol: return False
        if y - r < -tol or y + r > 1 + tol: return False
    for i in range(n):
        xi, yi, ri = circles_list[i]
        for j in range(i+1, n):
            xj, yj, rj = circles_list[j]
            dist = math.sqrt((xi-xj)**2 + (yi-yj)**2)
            if dist < ri + rj - tol: return False
    return True

def save_solution(circles_list, path):
    with open(path, 'w') as f:
        json.dump({"circles": circles_list}, f, indent=2)

# ─── Main solver ───

def solve(n=26, n_basin_iter=150, n_starts=20, seed=42, verbose=True, timeout=500):
    rng = np.random.default_rng(seed)

    best_metric = 0.0
    best_x = None
    start_time = time.time()

    init_funcs = [
        ("hex_grid", init_hexagonal_grid),
        ("greedy", init_random_greedy),
        ("poisson", init_poisson_disk),
        ("random", init_random_uniform),
        ("concentric", init_concentric),
    ]

    # Progressive penalty schedule
    penalty_weights = [100, 500, 2000, 10000]
    temperatures = [1.0, 0.5, 0.2, 0.05]

    results_log = []

    for start_i in range(n_starts):
        elapsed = time.time() - start_time
        if elapsed > timeout:
            if verbose: print(f"  Timeout after {start_i} starts ({elapsed:.0f}s)")
            break

        init_name, init_func = init_funcs[start_i % len(init_funcs)]
        T = temperatures[start_i % len(temperatures)]

        if verbose:
            print(f"\n--- Start {start_i+1}/{n_starts}: init={init_name}, T={T}, elapsed={elapsed:.0f}s ---")

        x0 = init_func(n, rng)
        step_func = PackingStep(n, stepsize=0.04 + 0.03 * rng.random(), rng=rng)

        # Progressive penalty: start loose, tighten
        best_x_run = x0.copy()
        best_metric_run = 0.0

        for pw in penalty_weights:
            obj = make_objective(n, pw)
            bounds = [(0.001, 0.999)] * (3*n)
            for i in range(n):
                bounds[3*i+2] = (0.001, 0.499)

            try:
                result = basinhopping(
                    obj, best_x_run,
                    minimizer_kwargs={
                        'method': 'L-BFGS-B',
                        'bounds': bounds,
                        'options': {'maxiter': 300, 'ftol': 1e-12},
                    },
                    niter=n_basin_iter // len(penalty_weights),
                    T=T,
                    take_step=step_func,
                    seed=int(rng.integers(0, 2**31)),
                    disp=False,
                    niter_success=30,
                )
                best_x_run = result.x
            except Exception as e:
                if verbose: print(f"  Error at pw={pw}: {e}")
                break

        # Extract metric (before polish)
        circles_raw = best_x_run.reshape(n, 3).tolist()
        raw_metric = sum(c[2] for c in circles_raw)
        raw_valid = validate_solution(circles_raw)

        if verbose:
            print(f"  Raw: metric={raw_metric:.6f}, valid={raw_valid}")

        # SLSQP polish if promising (metric > 80% of best known ~2.6)
        if raw_metric > max(best_metric * 0.9, 1.5):
            try:
                polished_x, polished_metric = slsqp_polish(best_x_run, n)
                polished_circles = polished_x.reshape(n, 3).tolist()
                polished_valid = validate_solution(polished_circles)
                if verbose:
                    print(f"  Polished: metric={polished_metric:.6f}, valid={polished_valid}")
                if polished_valid and polished_metric > best_metric:
                    best_metric = polished_metric
                    best_x = polished_x.copy()
                    if verbose: print(f"  *** NEW BEST: {best_metric:.6f} ***")
                elif raw_valid and raw_metric > best_metric:
                    best_metric = raw_metric
                    best_x = best_x_run.copy()
                    if verbose: print(f"  *** NEW BEST (raw): {best_metric:.6f} ***")
            except Exception as e:
                if verbose: print(f"  Polish error: {e}")
                if raw_valid and raw_metric > best_metric:
                    best_metric = raw_metric
                    best_x = best_x_run.copy()
        elif raw_valid and raw_metric > best_metric:
            best_metric = raw_metric
            best_x = best_x_run.copy()
            if verbose: print(f"  *** NEW BEST (raw): {best_metric:.6f} ***")

        results_log.append({
            'start': start_i, 'init': init_name, 'T': T,
            'metric': best_metric, 'raw_metric': raw_metric,
        })

    elapsed = time.time() - start_time

    # Final extra polish on best solution
    if best_x is not None:
        if verbose: print(f"\n--- Final SLSQP polish ---")
        try:
            final_x, final_metric = slsqp_polish(best_x, n)
            final_circles = final_x.reshape(n, 3).tolist()
            if validate_solution(final_circles) and final_metric > best_metric:
                best_metric = final_metric
                best_x = final_x
                if verbose: print(f"  Final polish: {final_metric:.6f}")
        except Exception as e:
            if verbose: print(f"  Final polish error: {e}")

    if verbose:
        print(f"\n=== Done in {time.time()-start_time:.1f}s, best={best_metric:.6f} ===")

    best_circles = best_x.reshape(n, 3).tolist() if best_x is not None else None
    return best_circles, best_metric, results_log

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 26
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    n_starts = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    timeout = int(sys.argv[4]) if len(sys.argv) > 4 else 500

    print(f"Circle packing v2: n={n}, seed={seed}, starts={n_starts}, timeout={timeout}s")

    solution, metric, log = solve(n=n, n_basin_iter=150, n_starts=n_starts,
                                   seed=seed, verbose=True, timeout=timeout)

    if solution is not None:
        out_path = Path(__file__).parent / f"solution_n{n}.json"
        save_solution(solution, out_path)
        print(f"\nSaved to {out_path}")
        log_path = Path(__file__).parent / f"run_log_n{n}.json"
        with open(log_path, 'w') as f:
            json.dump(log, f, indent=2)
    else:
        print("No valid solution found!")
