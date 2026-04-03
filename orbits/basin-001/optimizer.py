"""Basin-hopping optimizer for circle packing in unit square.

Maximizes sum of radii for n circles packed in [0,1]^2.
Uses scipy.optimize.basinhopping with SLSQP local minimizer.
"""

import json
import math
import numpy as np
from scipy.optimize import basinhopping, minimize
from pathlib import Path
import time
import sys

# ─── Problem encoding ───
# Decision vector: [x0, y0, r0, x1, y1, r1, ..., x_{n-1}, y_{n-1}, r_{n-1}]
# Length: 3*n

def decode(x, n):
    """Decode flat vector to (centers, radii)."""
    x = x.reshape(n, 3)
    return x[:, 0], x[:, 1], x[:, 2]

def objective(x, n):
    """Minimize negative sum of radii."""
    return -np.sum(x.reshape(n, 3)[:, 2])

def build_constraints(n):
    """Build SLSQP inequality constraints (>=0 convention)."""
    constraints = []

    # Containment: r <= x, x <= 1-r, r <= y, y <= 1-r
    # Rewrite: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    for i in range(n):
        ix, iy, ir = 3*i, 3*i+1, 3*i+2
        constraints.append({'type': 'ineq', 'fun': lambda x, ix=ix, ir=ir: x[ix] - x[ir]})
        constraints.append({'type': 'ineq', 'fun': lambda x, ix=ix, ir=ir: 1.0 - x[ix] - x[ir]})
        constraints.append({'type': 'ineq', 'fun': lambda x, iy=iy, ir=ir: x[iy] - x[ir]})
        constraints.append({'type': 'ineq', 'fun': lambda x, iy=iy, ir=ir: 1.0 - x[iy] - x[ir]})
        # Positive radius
        constraints.append({'type': 'ineq', 'fun': lambda x, ir=ir: x[ir] - 1e-8})

    # Non-overlap: sqrt((xi-xj)^2 + (yi-yj)^2) - (ri+rj) >= 0
    for i in range(n):
        for j in range(i+1, n):
            ixi, iyi, iri = 3*i, 3*i+1, 3*i+2
            ixj, iyj, irj = 3*j, 3*j+1, 3*j+2
            def sep_constraint(x, ixi=ixi, iyi=iyi, iri=iri, ixj=ixj, iyj=iyj, irj=irj):
                dx = x[ixi] - x[ixj]
                dy = x[iyi] - x[iyj]
                dist = math.sqrt(dx*dx + dy*dy)
                return dist - x[iri] - x[irj]
            constraints.append({'type': 'ineq', 'fun': sep_constraint})

    return constraints

def build_bounds(n):
    """Build variable bounds."""
    bounds = []
    for i in range(n):
        bounds.append((1e-6, 1.0 - 1e-6))  # x
        bounds.append((1e-6, 1.0 - 1e-6))  # y
        bounds.append((1e-8, 0.5))           # r
    return bounds

# ─── Initialization strategies ───

def init_hexagonal_grid(n, rng):
    """Place circles on a hex grid with slight jitter."""
    # Estimate grid size
    cols = int(math.ceil(math.sqrt(n * 2 / math.sqrt(3))))
    rows = int(math.ceil(n / cols))

    r_est = 0.5 / max(cols, rows * math.sqrt(3)/2)

    circles = []
    for row in range(rows + 2):
        for col in range(cols + 2):
            if len(circles) >= n:
                break
            x = (col + 0.5 * (row % 2)) / (cols + 1)
            y = row * math.sqrt(3) / 2 / (rows + 1)
            x = np.clip(x, r_est + 0.01, 1 - r_est - 0.01)
            y = np.clip(y, r_est + 0.01, 1 - r_est - 0.01)
            circles.append([x, y, r_est * 0.9])
        if len(circles) >= n:
            break

    circles = circles[:n]
    result = np.array(circles).flatten()
    # Add small jitter
    result[:] += rng.normal(0, 0.005, len(result))
    # Clip radii to positive
    for i in range(n):
        result[3*i+2] = max(result[3*i+2], 1e-4)
        result[3*i] = np.clip(result[3*i], result[3*i+2] + 0.001, 1 - result[3*i+2] - 0.001)
        result[3*i+1] = np.clip(result[3*i+1], result[3*i+2] + 0.001, 1 - result[3*i+2] - 0.001)
    return result

def init_concentric_rings(n, rng):
    """Place circles in concentric rings."""
    # Ring layout: 1 center + ring of ~6 + ring of ~12 + ...
    rings = []
    remaining = n
    ring_idx = 0
    while remaining > 0:
        if ring_idx == 0:
            count = 1
        else:
            count = min(6 * ring_idx, remaining)
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
                angle = 2 * math.pi * k / count + rng.uniform(0, 0.1)
                x = 0.5 + ring_r * math.cos(angle)
                y = 0.5 + ring_r * math.sin(angle)
                x = np.clip(x, r_est + 0.01, 1 - r_est - 0.01)
                y = np.clip(y, r_est + 0.01, 1 - r_est - 0.01)
                circles.append([x, y, r_est * 0.8])

    circles = circles[:n]
    return np.array(circles).flatten()

def init_random_greedy(n, rng):
    """Greedy random placement: place each circle as large as possible."""
    circles = []
    for i in range(n):
        best_r = 0
        best_x, best_y = 0.5, 0.5
        for _ in range(200):
            x = rng.uniform(0.02, 0.98)
            y = rng.uniform(0.02, 0.98)
            # Max radius from containment
            r_max = min(x, 1-x, y, 1-y)
            # Max radius from existing circles
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
    """Poisson disk sampling to get well-separated initial points."""
    min_dist = 0.8 / math.sqrt(n)
    points = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        x = rng.uniform(0.05, 0.95)
        y = rng.uniform(0.05, 0.95)
        ok = True
        for px, py in points:
            if math.sqrt((x-px)**2 + (y-py)**2) < min_dist:
                ok = False
                break
        if ok:
            points.append((x, y))
        attempts += 1

    # Fill remaining randomly if needed
    while len(points) < n:
        x = rng.uniform(0.05, 0.95)
        y = rng.uniform(0.05, 0.95)
        points.append((x, y))

    r_est = min_dist / 2.5
    circles = [[px, py, r_est] for px, py in points[:n]]
    return np.array(circles).flatten()

def init_from_best_known(n, rng):
    """Start from a known good solution with perturbation."""
    # For n=26, use a structure inspired by known good packings:
    # Mixed sizes with some large circles and many small ones
    if n == 26:
        # Try a 5x5 grid + 1 center approach with varied sizes
        circles = []
        for i in range(5):
            for j in range(5):
                x = 0.1 + 0.2 * i
                y = 0.1 + 0.2 * j
                r = 0.08 + rng.uniform(-0.01, 0.01)
                circles.append([x, y, r])
        # Add center circle
        circles.append([0.5, 0.5, 0.05])
        return np.array(circles[:n]).flatten()
    else:
        return init_random_greedy(n, rng)

# ─── Custom step function ───

class AdaptiveStep:
    """Custom step function for basin-hopping that respects problem structure."""
    def __init__(self, n, stepsize=0.05, rng=None):
        self.n = n
        self.stepsize = stepsize
        self.rng = rng or np.random.default_rng(42)

    def __call__(self, x):
        n = self.n
        x_new = x.copy()

        # Choose perturbation type
        action = self.rng.choice(['shift_one', 'shift_all', 'swap', 'resize', 'big_move'])

        if action == 'shift_one':
            # Shift one random circle
            i = self.rng.integers(0, n)
            x_new[3*i] += self.rng.normal(0, self.stepsize)
            x_new[3*i+1] += self.rng.normal(0, self.stepsize)
            x_new[3*i+2] += self.rng.normal(0, self.stepsize * 0.3)

        elif action == 'shift_all':
            # Small shift to all circles
            for i in range(n):
                x_new[3*i] += self.rng.normal(0, self.stepsize * 0.3)
                x_new[3*i+1] += self.rng.normal(0, self.stepsize * 0.3)
                x_new[3*i+2] += self.rng.normal(0, self.stepsize * 0.1)

        elif action == 'swap':
            # Swap two circles
            i, j = self.rng.choice(n, 2, replace=False)
            x_new[3*i], x_new[3*j] = x_new[3*j], x_new[3*i]
            x_new[3*i+1], x_new[3*j+1] = x_new[3*j+1], x_new[3*i+1]
            # Keep radii, don't swap

        elif action == 'resize':
            # Resize a few circles
            k = self.rng.integers(1, max(2, n//3))
            indices = self.rng.choice(n, k, replace=False)
            for i in indices:
                x_new[3*i+2] *= self.rng.uniform(0.7, 1.3)

        elif action == 'big_move':
            # Big move: relocate one circle to a random spot
            i = self.rng.integers(0, n)
            x_new[3*i] = self.rng.uniform(0.05, 0.95)
            x_new[3*i+1] = self.rng.uniform(0.05, 0.95)
            x_new[3*i+2] = self.rng.uniform(0.01, 0.15)

        # Clip to bounds
        for i in range(n):
            x_new[3*i+2] = np.clip(x_new[3*i+2], 1e-4, 0.499)
            x_new[3*i] = np.clip(x_new[3*i], x_new[3*i+2] + 0.001, 1 - x_new[3*i+2] - 0.001)
            x_new[3*i+1] = np.clip(x_new[3*i+1], x_new[3*i+2] + 0.001, 1 - x_new[3*i+2] - 0.001)

        return x_new

# ─── Penalty-based objective for faster local search ───

def penalty_objective(x, n, penalty_weight=1000.0):
    """Objective with penalty for constraint violations (for methods without constraint support)."""
    circles = x.reshape(n, 3)
    xs, ys, rs = circles[:, 0], circles[:, 1], circles[:, 2]

    obj = -np.sum(rs)  # Maximize sum of radii

    penalty = 0.0
    # Containment
    penalty += np.sum(np.maximum(0, rs - xs)**2)
    penalty += np.sum(np.maximum(0, xs + rs - 1)**2)
    penalty += np.sum(np.maximum(0, rs - ys)**2)
    penalty += np.sum(np.maximum(0, ys + rs - 1)**2)
    # Positive radius
    penalty += np.sum(np.maximum(0, -rs)**2)

    # Non-overlap
    for i in range(n):
        for j in range(i+1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = math.sqrt(dx*dx + dy*dy)
            overlap = rs[i] + rs[j] - dist
            if overlap > 0:
                penalty += overlap**2

    return obj + penalty_weight * penalty

# ─── Main solver ───

def solve(n=26, n_basin_iter=300, n_starts=10, seed=42, verbose=True, timeout=300):
    """Run basin-hopping optimization.

    Args:
        n: number of circles
        n_basin_iter: basin-hopping iterations per start
        n_starts: number of random restarts
        seed: random seed
        verbose: print progress
        timeout: max seconds
    """
    rng = np.random.default_rng(seed)
    constraints = build_constraints(n)
    bounds = build_bounds(n)

    best_metric = 0.0
    best_solution = None
    start_time = time.time()

    init_funcs = [
        ("hex_grid", init_hexagonal_grid),
        ("concentric", init_concentric_rings),
        ("greedy", init_random_greedy),
        ("poisson", init_poisson_disk),
        ("best_known", init_from_best_known),
    ]

    temperatures = [1.0, 0.5, 0.1]

    results_log = []

    for start_i in range(n_starts):
        if time.time() - start_time > timeout:
            if verbose:
                print(f"  Timeout after {start_i} starts")
            break

        # Pick initialization
        init_name, init_func = init_funcs[start_i % len(init_funcs)]
        x0 = init_func(n, rng)

        # Pick temperature
        T = temperatures[start_i % len(temperatures)]

        if verbose:
            print(f"\n--- Start {start_i+1}/{n_starts}: init={init_name}, T={T} ---")

        step_func = AdaptiveStep(n, stepsize=0.05 + 0.02 * rng.random(), rng=rng)

        # Wrap objective to bind n
        def obj_fn(x):
            return objective(x, n)

        minimizer_kwargs = {
            'method': 'SLSQP',
            'constraints': constraints,
            'bounds': bounds,
            'options': {'maxiter': 500, 'ftol': 1e-12, 'disp': False},
        }

        try:
            result = basinhopping(
                obj_fn,
                x0,
                minimizer_kwargs=minimizer_kwargs,
                niter=n_basin_iter,
                T=T,
                stepsize=0.05,
                take_step=step_func,
                seed=int(rng.integers(0, 2**31)),
                disp=False,
                niter_success=50,  # Stop if no improvement for 50 iters
                callback=None,
            )

            metric = -result.fun

            # Polish with tight tolerance
            polish = minimize(
                obj_fn,
                result.x,
                method='SLSQP',
                constraints=constraints,
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-15, 'disp': False},
            )

            metric_polished = -polish.fun

            if verbose:
                print(f"  Basin-hop: {metric:.6f}, polished: {metric_polished:.6f}")

            if metric_polished > metric:
                metric = metric_polished
                result_x = polish.x
            else:
                result_x = result.x

            # Validate
            circles_list = result_x.reshape(n, 3).tolist()
            valid = validate_solution(circles_list)

            results_log.append({
                'start': start_i,
                'init': init_name,
                'T': T,
                'metric': metric,
                'valid': valid,
            })

            if valid and metric > best_metric:
                best_metric = metric
                best_solution = circles_list
                if verbose:
                    print(f"  *** NEW BEST: {metric:.6f} (valid) ***")
            elif verbose:
                print(f"  metric={metric:.6f}, valid={valid}")

        except Exception as e:
            if verbose:
                print(f"  Error: {e}")
            results_log.append({
                'start': start_i,
                'init': init_name,
                'T': T,
                'metric': 0,
                'valid': False,
                'error': str(e),
            })

    elapsed = time.time() - start_time
    if verbose:
        print(f"\n=== Done in {elapsed:.1f}s ===")
        print(f"Best metric: {best_metric:.6f}")

    return best_solution, best_metric, results_log

def validate_solution(circles_list):
    """Quick validation of a solution."""
    n = len(circles_list)
    tol = 1e-9
    for i, (x, y, r) in enumerate(circles_list):
        if r <= 0:
            return False
        if x - r < -tol or x + r > 1 + tol:
            return False
        if y - r < -tol or y + r > 1 + tol:
            return False
    for i in range(n):
        xi, yi, ri = circles_list[i]
        for j in range(i+1, n):
            xj, yj, rj = circles_list[j]
            dist = math.sqrt((xi-xj)**2 + (yi-yj)**2)
            if dist < ri + rj - tol:
                return False
    return True

def save_solution(circles_list, path):
    """Save solution in evaluator format."""
    data = {"circles": circles_list}
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 26
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    n_starts = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    timeout = int(sys.argv[4]) if len(sys.argv) > 4 else 600

    print(f"Circle packing: n={n}, seed={seed}, starts={n_starts}, timeout={timeout}s")

    solution, metric, log = solve(
        n=n,
        n_basin_iter=200,
        n_starts=n_starts,
        seed=seed,
        verbose=True,
        timeout=timeout,
    )

    if solution is not None:
        out_path = Path(__file__).parent / f"solution_n{n}.json"
        save_solution(solution, out_path)
        print(f"\nSaved to {out_path}")

        # Also save log
        log_path = Path(__file__).parent / f"run_log_n{n}.json"
        with open(log_path, 'w') as f:
            json.dump(log, f, indent=2)
    else:
        print("No valid solution found!")
