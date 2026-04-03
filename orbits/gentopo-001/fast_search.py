"""
Fast topology search using penalty method + basin hopping.

Key changes from topo_jump.py:
1. Penalty method instead of constrained SLSQP (faster, can cross barriers)
2. L-BFGS-B with analytical gradient (much faster per iteration)
3. More aggressive perturbations
4. Simulated annealing acceptance criterion
"""

import json
import numpy as np
from scipy.optimize import minimize, differential_evolution
import os
import time

WORKDIR = os.path.dirname(os.path.abspath(__file__))
N = 26
SEED = 42


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
            dist2 = (x[i]-x[j])**2 + (y[i]-y[j])**2
            if dist2 < (r[i]+r[j])**2 - 2*tol*(r[i]+r[j]):
                return False
    return True


def penalty_objective(v, n, mu):
    """
    Penalized objective: -sum(r) + mu * sum(constraint_violations^2)
    """
    x = v[:n]
    y = v[n:2*n]
    r = v[2*n:3*n]

    obj = -np.sum(r)
    penalty = 0.0

    # Containment
    for i in range(n):
        viol = r[i] - x[i]
        if viol > 0: penalty += viol**2
        viol = x[i] + r[i] - 1.0
        if viol > 0: penalty += viol**2
        viol = r[i] - y[i]
        if viol > 0: penalty += viol**2
        viol = y[i] + r[i] - 1.0
        if viol > 0: penalty += viol**2

    # Non-overlap
    for i in range(n):
        for j in range(i+1, n):
            dist2 = (x[i]-x[j])**2 + (y[i]-y[j])**2
            sum_r = r[i] + r[j]
            viol = sum_r**2 - dist2
            if viol > 0:
                penalty += viol

    return obj + mu * penalty


def penalty_gradient(v, n, mu):
    """Analytical gradient of penalty objective."""
    x = v[:n]
    y = v[n:2*n]
    r = v[2*n:3*n]

    grad = np.zeros(3*n)
    grad[2*n:3*n] = -1.0  # objective gradient

    # Containment gradients
    for i in range(n):
        # x_i - r_i >= 0
        viol = r[i] - x[i]
        if viol > 0:
            grad[i] += mu * (-2*viol)
            grad[2*n+i] += mu * (2*viol)
        # 1 - x_i - r_i >= 0
        viol = x[i] + r[i] - 1.0
        if viol > 0:
            grad[i] += mu * (2*viol)
            grad[2*n+i] += mu * (2*viol)
        # y_i - r_i >= 0
        viol = r[i] - y[i]
        if viol > 0:
            grad[n+i] += mu * (-2*viol)
            grad[2*n+i] += mu * (2*viol)
        # 1 - y_i - r_i >= 0
        viol = y[i] + r[i] - 1.0
        if viol > 0:
            grad[n+i] += mu * (2*viol)
            grad[2*n+i] += mu * (2*viol)

    # Non-overlap gradients
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist2 = dx**2 + dy**2
            sum_r = r[i] + r[j]
            viol = sum_r**2 - dist2
            if viol > 0:
                # d(viol)/dx_i = -2*dx, d(viol)/dx_j = 2*dx
                grad[i] += mu * (-2*dx)
                grad[j] += mu * (2*dx)
                grad[n+i] += mu * (-2*dy)
                grad[n+j] += mu * (2*dy)
                grad[2*n+i] += mu * (2*sum_r)
                grad[2*n+j] += mu * (2*sum_r)

    return grad


def optimize_penalty(x0, y0, r0, mu_schedule=None):
    """Optimize using increasing penalty parameter."""
    n = len(x0)

    if mu_schedule is None:
        mu_schedule = [1, 10, 100, 1000, 10000, 100000, 1000000]

    v = np.concatenate([x0, y0, r0])
    bounds = [(0.001, 0.999)]*n + [(0.001, 0.999)]*n + [(0.001, 0.5)]*n

    for mu in mu_schedule:
        result = minimize(
            penalty_objective, v, args=(n, mu),
            jac=lambda v, n=n, mu=mu: penalty_gradient(v, n, mu),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-15}
        )
        v = result.x

    x = v[:n]
    y = v[n:2*n]
    r = v[2*n:3*n]

    # Repair: shrink radii slightly to ensure feasibility
    for _ in range(50):
        violated = False
        for i in range(n):
            if x[i] - r[i] < 0: r[i] = x[i] - 1e-6; violated = True
            if 1 - x[i] - r[i] < 0: r[i] = 1 - x[i] - 1e-6; violated = True
            if y[i] - r[i] < 0: r[i] = y[i] - 1e-6; violated = True
            if 1 - y[i] - r[i] < 0: r[i] = 1 - y[i] - 1e-6; violated = True
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if dist < r[i] + r[j]:
                    excess = (r[i] + r[j] - dist) / 2 + 1e-6
                    r[i] -= excess
                    r[j] -= excess
                    violated = True
        if not violated:
            break

    r = np.maximum(r, 0.001)
    metric = np.sum(r)
    feasible = is_feasible(x, y, r, tol=1e-8)

    return x, y, r, metric, feasible


def optimize_slsqp_polish(x0, y0, r0):
    """Final SLSQP polish for high precision."""
    n = len(x0)

    def objective(v):
        return -np.sum(v[2*n:3*n])

    def obj_jac(v):
        g = np.zeros(3*n)
        g[2*n:3*n] = -1.0
        return g

    constraints = []
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[i] - v[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[n+i] - v[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[n+i] - v[2*n+i]})

    for i in range(n):
        for j in range(i+1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j: (
                    (v[i]-v[j])**2 + (v[n+i]-v[n+j])**2 - (v[2*n+i]+v[2*n+j])**2
                )
            })

    bounds = [(0.001, 0.999)]*n + [(0.001, 0.999)]*n + [(0.001, 0.5)]*n
    v0 = np.concatenate([x0, y0, r0])

    result = minimize(
        objective, v0, method='SLSQP',
        jac=obj_jac,
        constraints=constraints, bounds=bounds,
        options={'maxiter': 2000, 'ftol': 1e-15}
    )

    x, y, r = result.x[:n], result.x[n:2*n], result.x[2*n:3*n]
    return x, y, r, np.sum(r), is_feasible(x, y, r)


def topology_fingerprint(x, y, r, tol=1e-5):
    """Fingerprint based on sorted radii and contact pattern."""
    n = len(x)
    contacts = set()
    walls = set()

    for i in range(n):
        if abs(x[i] - r[i]) < tol: walls.add((i, 'L'))
        if abs(1 - x[i] - r[i]) < tol: walls.add((i, 'R'))
        if abs(y[i] - r[i]) < tol: walls.add((i, 'B'))
        if abs(1 - y[i] - r[i]) < tol: walls.add((i, 'T'))

    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if abs(dist - r[i] - r[j]) < tol:
                contacts.add((i, j))

    return (frozenset(contacts), frozenset(walls))


def generate_ring_init(n, ring_config, seed=42):
    """Generate ring-based initialization.
    ring_config: list of (n_circles, radius, distance_from_center)
    """
    rng = np.random.RandomState(seed)
    x, y, r = [], [], []

    for n_in_ring, ring_r, ring_dist in ring_config:
        for k in range(n_in_ring):
            angle = k * 2 * np.pi / n_in_ring + rng.uniform(-0.1, 0.1)
            cx = 0.5 + ring_dist * np.cos(angle)
            cy = 0.5 + ring_dist * np.sin(angle)
            x.append(cx)
            y.append(cy)
            r.append(ring_r)

    x, y, r = np.array(x[:n]), np.array(y[:n]), np.array(r[:n])
    r = np.clip(r, 0.01, 0.49)
    x = np.clip(x, r+0.001, 1-r-0.001)
    y = np.clip(y, r+0.001, 1-r-0.001)
    return x, y, r


def main():
    t0 = time.time()
    rng = np.random.RandomState(SEED)

    # Load known best
    known_path = os.path.join(WORKDIR, '..', 'topo-001', 'solution_n26.json')
    xk, yk, rk = load_solution(known_path)
    known_metric = np.sum(rk)
    print(f"Known best: {known_metric:.10f}")

    best_metric = known_metric
    best_sol = (xk.copy(), yk.copy(), rk.copy())
    seen_fps = set()
    seen_fps.add(topology_fingerprint(xk, yk, rk))

    # ===== Strategy A: Penalty-based basin hopping from known best =====
    print("\n" + "="*60)
    print("STRATEGY A: Penalty basin hopping (500 attempts)")
    print("="*60)

    for attempt in range(500):
        # Varying perturbation strengths
        if attempt < 100:
            strength = rng.uniform(0.05, 0.20)  # moderate
        elif attempt < 300:
            strength = rng.uniform(0.10, 0.40)  # large
        else:
            strength = rng.uniform(0.20, 0.60)  # very large

        x0 = xk.copy() + rng.uniform(-strength, strength, N)
        y0 = yk.copy() + rng.uniform(-strength, strength, N)
        r0 = rk.copy() * rng.uniform(max(0.3, 1-strength*2), min(1.7, 1+strength*2), N)

        r0 = np.clip(r0, 0.01, 0.49)
        x0 = np.clip(x0, r0+0.001, 1-r0-0.001)
        y0 = np.clip(y0, r0+0.001, 1-r0-0.001)

        x2, y2, r2, metric, feasible = optimize_penalty(x0, y0, r0)

        if feasible and metric > 2.5:
            # Polish with SLSQP
            x2, y2, r2, metric, feasible = optimize_slsqp_polish(x2, y2, r2)

            if feasible and metric > 2.60:
                fp = topology_fingerprint(x2, y2, r2)
                is_new = fp not in seen_fps
                seen_fps.add(fp)

                if metric > best_metric:
                    best_metric = metric
                    best_sol = (x2.copy(), y2.copy(), r2.copy())
                    print(f"  [{attempt+1}] *** NEW BEST: {metric:.10f} *** "
                          f"(strength={strength:.2f}, new_topo={is_new})")
                elif is_new and metric > 2.63:
                    print(f"  [{attempt+1}] New topology: {metric:.10f} "
                          f"(strength={strength:.2f})")

        if (attempt + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  Progress: {attempt+1}/500, best: {best_metric:.10f}, "
                  f"unique topos: {len(seen_fps)}, time: {elapsed:.0f}s")

    # ===== Strategy B: Diverse ring initializations =====
    print("\n" + "="*60)
    print("STRATEGY B: Ring-based initializations")
    print("="*60)

    ring_configs = [
        # (n_circles, approx_radius, distance_from_center)
        [(1, 0.14, 0.0), (7, 0.10, 0.24), (12, 0.08, 0.42), (6, 0.06, 0.44)],
        [(1, 0.15, 0.0), (6, 0.11, 0.26), (12, 0.08, 0.42), (7, 0.05, 0.44)],
        [(1, 0.13, 0.0), (8, 0.10, 0.24), (12, 0.07, 0.40), (5, 0.06, 0.44)],
        [(2, 0.13, 0.15), (6, 0.10, 0.28), (12, 0.08, 0.42), (6, 0.05, 0.44)],
        [(1, 0.16, 0.0), (5, 0.12, 0.28), (10, 0.08, 0.40), (10, 0.05, 0.44)],
        [(3, 0.12, 0.14), (7, 0.10, 0.30), (10, 0.08, 0.42), (6, 0.06, 0.44)],
        [(1, 0.12, 0.0), (6, 0.10, 0.22), (6, 0.09, 0.36), (6, 0.07, 0.42), (7, 0.04, 0.46)],
        [(4, 0.14, 0.18), (8, 0.10, 0.35), (14, 0.06, 0.44)],
        [(1, 0.14, 0.0), (6, 0.11, 0.25), (8, 0.09, 0.40), (4, 0.085, 0.44), (7, 0.04, 0.46)],
        [(1, 0.15, 0.0), (4, 0.12, 0.27), (8, 0.09, 0.38), (8, 0.07, 0.44), (5, 0.05, 0.45)],
    ]

    for config_idx, config in enumerate(ring_configs):
        for seed_offset in range(20):
            x0, y0, r0 = generate_ring_init(N, config, seed=SEED + config_idx*100 + seed_offset)

            # Add some noise
            x0 += rng.uniform(-0.02, 0.02, N)
            y0 += rng.uniform(-0.02, 0.02, N)
            r0 *= rng.uniform(0.9, 1.1, N)
            r0 = np.clip(r0, 0.01, 0.49)
            x0 = np.clip(x0, r0+0.001, 1-r0-0.001)
            y0 = np.clip(y0, r0+0.001, 1-r0-0.001)

            x2, y2, r2, metric, feasible = optimize_penalty(x0, y0, r0)

            if feasible and metric > 2.5:
                x2, y2, r2, metric, feasible = optimize_slsqp_polish(x2, y2, r2)

                if feasible and metric > 2.60:
                    fp = topology_fingerprint(x2, y2, r2)
                    is_new = fp not in seen_fps
                    seen_fps.add(fp)

                    if metric > best_metric:
                        best_metric = metric
                        best_sol = (x2.copy(), y2.copy(), r2.copy())
                        print(f"  [config {config_idx}, seed {seed_offset}] "
                              f"*** NEW BEST: {metric:.10f} *** (new_topo={is_new})")
                    elif is_new and metric > 2.63:
                        print(f"  [config {config_idx}, seed {seed_offset}] "
                              f"New topology: {metric:.10f}")

        elapsed = time.time() - t0
        print(f"  Config {config_idx+1}/{len(ring_configs)} done, "
              f"best: {best_metric:.10f}, time: {elapsed:.0f}s")

    # ===== Strategy C: Completely random starts (fast penalty) =====
    print("\n" + "="*60)
    print("STRATEGY C: Random starts with penalty method (500 attempts)")
    print("="*60)

    for attempt in range(500):
        # Random positions and radii
        r0 = rng.uniform(0.02, 0.12, N)
        x0 = rng.uniform(0.05, 0.95, N)
        y0 = rng.uniform(0.05, 0.95, N)
        r0 = np.clip(r0, 0.01, 0.49)
        x0 = np.clip(x0, r0+0.001, 1-r0-0.001)
        y0 = np.clip(y0, r0+0.001, 1-r0-0.001)

        x2, y2, r2, metric, feasible = optimize_penalty(x0, y0, r0)

        if feasible and metric > 2.5:
            x2, y2, r2, metric, feasible = optimize_slsqp_polish(x2, y2, r2)

            if feasible and metric > 2.60:
                fp = topology_fingerprint(x2, y2, r2)
                is_new = fp not in seen_fps
                seen_fps.add(fp)

                if metric > best_metric:
                    best_metric = metric
                    best_sol = (x2.copy(), y2.copy(), r2.copy())
                    print(f"  [{attempt+1}] *** NEW BEST: {metric:.10f} *** "
                          f"(new_topo={is_new})")
                elif is_new and metric > 2.63:
                    print(f"  [{attempt+1}] New topology: {metric:.10f}")

        if (attempt + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  Progress: {attempt+1}/500, best: {best_metric:.10f}, "
                  f"unique topos: {len(seen_fps)}, time: {elapsed:.0f}s")

    # ===== Save =====
    sol_path = os.path.join(WORKDIR, 'solution_n26.json')
    save_solution(*best_sol, sol_path)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"FINAL: metric={best_metric:.10f} (known={known_metric:.10f})")
    print(f"Improvement: {best_metric - known_metric:.2e}")
    print(f"Unique topologies: {len(seen_fps)}")
    print(f"Time: {elapsed:.0f}s")

    return best_metric


if __name__ == '__main__':
    main()
