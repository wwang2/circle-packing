"""
Simulated Annealing with Topology-Aware Perturbations for n=26.
Key idea: Use SA to explore different topological basins,
with SLSQP polishing of promising candidates.
"""

import json
import numpy as np
from scipy.optimize import minimize
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

def project_feasible(x, y, r):
    """Project solution to be approximately feasible."""
    n = len(x)
    r = np.maximum(r, 1e-4)
    x = np.clip(x, r, 1 - r)
    y = np.clip(y, r, 1 - r)

    # Resolve overlaps by pushing circles apart
    for iteration in range(100):
        moved = False
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx**2 + dy**2)
                min_dist = r[i] + r[j]
                if dist < min_dist and dist > 1e-10:
                    overlap = min_dist - dist
                    # Push apart proportionally
                    push = overlap * 0.55 / dist
                    x[i] += dx * push
                    y[i] += dy * push
                    x[j] -= dx * push
                    y[j] -= dy * push
                    moved = True
                elif dist < 1e-10:
                    # Coincident - push randomly
                    angle = np.random.uniform(0, 2*np.pi)
                    x[i] += min_dist * 0.55 * np.cos(angle)
                    y[i] += min_dist * 0.55 * np.sin(angle)
                    moved = True
        # Re-clamp
        x = np.clip(x, r, 1 - r)
        y = np.clip(y, r, 1 - r)
        if not moved:
            break

    return x, y, r

def quick_optimize(x0, y0, r0, maxiter=2000):
    """Quick SLSQP optimization."""
    n = len(x0)
    params0 = np.concatenate([x0, y0, r0])

    constraints = []
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[i] - p[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[i] - p[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[n+i] - p[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[n+i] - p[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[2*n+i] - 1e-6})

    for i in range(n):
        for j in range(i+1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: (
                    (p[i]-p[j])**2 + (p[n+i]-p[n+j])**2 - (p[2*n+i]+p[2*n+j])**2
                )
            })

    result = minimize(
        lambda p: -np.sum(p[2*n:3*n]),
        params0,
        method='SLSQP',
        constraints=constraints,
        options={'maxiter': maxiter, 'ftol': 1e-15, 'disp': False}
    )

    x = result.x[:n]
    y = result.x[n:2*n]
    r = result.x[2*n:3*n]
    return x, y, r, np.sum(r), result.success

def compute_metric_fast(x, y, r):
    """Fast feasibility check + metric."""
    n = len(x)
    for i in range(n):
        if r[i] < 0: return -1
        if x[i] - r[i] < -1e-8 or 1 - x[i] - r[i] < -1e-8: return -1
        if y[i] - r[i] < -1e-8 or 1 - y[i] - r[i] < -1e-8: return -1
    for i in range(n):
        for j in range(i+1, n):
            if (x[i]-x[j])**2 + (y[i]-y[j])**2 < (r[i]+r[j])**2 - 1e-8:
                return -1
    return np.sum(r)

def sa_topology_search(x0, y0, r0, T_init=0.02, T_min=1e-6, cooling=0.995,
                       steps_per_temp=20, seed=42):
    """Simulated annealing that explores topology space."""
    rng = np.random.RandomState(seed)
    n = len(x0)

    # Start from parent
    x_cur, y_cur, r_cur = x0.copy(), y0.copy(), r0.copy()
    metric_cur = np.sum(r_cur)

    best_x, best_y, best_r = x_cur.copy(), y_cur.copy(), r_cur.copy()
    best_metric = metric_cur

    T = T_init
    step = 0
    accepts = 0
    improves = 0

    while T > T_min:
        for _ in range(steps_per_temp):
            step += 1

            # Generate neighbor via topology-changing perturbation
            x_new, y_new, r_new = x_cur.copy(), y_cur.copy(), r_cur.copy()

            move_type = rng.randint(0, 8)

            if move_type == 0:
                # Swap two circles' positions
                i, j = rng.choice(n, 2, replace=False)
                x_new[i], x_new[j] = x_new[j], x_new[i]
                y_new[i], y_new[j] = y_new[j], y_new[i]

            elif move_type == 1:
                # Large displacement of one circle
                i = rng.randint(n)
                scale = T * 5 + 0.02
                x_new[i] += rng.normal(0, scale)
                y_new[i] += rng.normal(0, scale)
                r_new[i] *= rng.uniform(0.7, 1.3)

            elif move_type == 2:
                # Move circle to a random new position
                i = rng.randint(n)
                x_new[i] = rng.uniform(0.05, 0.95)
                y_new[i] = rng.uniform(0.05, 0.95)
                r_new[i] *= rng.uniform(0.5, 1.2)

            elif move_type == 3:
                # Rotate entire packing by small angle
                angle = rng.normal(0, T * 2 + 0.01)
                cx, cy = 0.5, 0.5
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                for i in range(n):
                    dx, dy = x_new[i] - cx, y_new[i] - cy
                    x_new[i] = cx + cos_a * dx - sin_a * dy
                    y_new[i] = cy + sin_a * dx + cos_a * dy

            elif move_type == 4:
                # Scale radii uniformly
                factor = 1 + rng.normal(0, T * 0.5 + 0.001)
                r_new *= factor

            elif move_type == 5:
                # Swap radii (but not positions) of two circles
                i, j = rng.choice(n, 2, replace=False)
                r_new[i], r_new[j] = r_new[j], r_new[i]

            elif move_type == 6:
                # Perturb multiple circles' radii
                k = rng.randint(1, n//2)
                idxs = rng.choice(n, k, replace=False)
                for idx in idxs:
                    r_new[idx] *= (1 + rng.normal(0, T + 0.005))

            elif move_type == 7:
                # Mirror a subset about center
                k = rng.randint(2, n//3 + 1)
                idxs = rng.choice(n, k, replace=False)
                axis = rng.randint(0, 3)
                if axis == 0:
                    x_new[idxs] = 1 - x_new[idxs]
                elif axis == 1:
                    y_new[idxs] = 1 - y_new[idxs]
                else:
                    for idx in idxs:
                        x_new[idx], y_new[idx] = y_new[idx], x_new[idx]

            # Project to feasibility
            r_new = np.maximum(r_new, 0.005)
            x_new, y_new, r_new = project_feasible(x_new, y_new, r_new)

            # Quick optimize
            x_new, y_new, r_new, metric_new, success = quick_optimize(
                x_new, y_new, r_new, maxiter=1500)

            if not success:
                continue

            metric_new_check = compute_metric_fast(x_new, y_new, r_new)
            if metric_new_check < 0:
                continue

            # SA acceptance criterion
            delta = metric_new - metric_cur
            if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
                x_cur, y_cur, r_cur = x_new.copy(), y_new.copy(), r_new.copy()
                metric_cur = metric_new
                accepts += 1

                if metric_new > best_metric:
                    best_metric = metric_new
                    best_x, best_y, best_r = x_new.copy(), y_new.copy(), r_new.copy()
                    improves += 1
                    print(f"  Step {step}, T={T:.6f}: NEW BEST {best_metric:.10f} (move={move_type})")

        T *= cooling

        if step % 500 == 0:
            print(f"  Step {step}, T={T:.6f}, metric={metric_cur:.10f}, "
                  f"best={best_metric:.10f}, accepts={accepts}, improves={improves}")

    return best_x, best_y, best_r, best_metric

def main():
    t0 = time.time()
    parent_path = os.path.join(WORKDIR, '..', 'nlp-001', 'solution_n26.json')
    x0, y0, r0 = load_solution(parent_path)
    parent_metric = np.sum(r0)
    print(f"Parent metric: {parent_metric:.10f}")

    overall_best = parent_metric
    overall_x, overall_y, overall_r = x0.copy(), y0.copy(), r0.copy()

    # Run SA with different seeds and temperatures
    configs = [
        (0.05, 0.9985, 15, 42),    # High T, slow cooling
        (0.02, 0.997, 20, 123),     # Medium T
        (0.01, 0.998, 25, 456),     # Low T, more steps
        (0.1, 0.996, 10, 789),      # Very high T
        (0.03, 0.999, 12, 1001),    # Slow cooling
    ]

    for i, (T_init, cooling, spt, seed) in enumerate(configs):
        print(f"\n=== SA Run {i+1}: T0={T_init}, cool={cooling}, spt={spt}, seed={seed} ===")
        xb, yb, rb, mb = sa_topology_search(
            x0, y0, r0,
            T_init=T_init, T_min=1e-6, cooling=cooling,
            steps_per_temp=spt, seed=seed
        )
        print(f"  Result: {mb:.10f}")

        if mb > overall_best:
            overall_best = mb
            overall_x, overall_y, overall_r = xb.copy(), yb.copy(), rb.copy()
            save_solution(overall_x, overall_y, overall_r,
                        os.path.join(WORKDIR, 'solution_n26_sa.json'))

    # Final polish
    print(f"\n=== Final Polish: {overall_best:.10f} ===")
    xf, yf, rf, mf, _ = quick_optimize(overall_x, overall_y, overall_r, maxiter=20000)
    mf_check = compute_metric_fast(xf, yf, rf)
    if mf_check > overall_best:
        overall_best = mf_check
        overall_x, overall_y, overall_r = xf, yf, rf

    save_solution(overall_x, overall_y, overall_r,
                os.path.join(WORKDIR, 'solution_n26_sa.json'))

    elapsed = time.time() - t0
    print(f"\n=== SA FINAL ===")
    print(f"Parent:  {parent_metric:.10f}")
    print(f"Best:    {overall_best:.10f}")
    print(f"Delta:   {overall_best - parent_metric:.2e}")
    print(f"Time:    {elapsed:.0f}s")

if __name__ == '__main__':
    main()
