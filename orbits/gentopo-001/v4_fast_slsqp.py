"""
v4: Fast SLSQP with analytical Jacobian for constraints.

The key insight: SLSQP is slow because scipy evaluates constraint Jacobians
via finite differences. Providing analytical Jacobians makes it 5-10x faster.
"""

import json
import numpy as np
from scipy.optimize import minimize
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
    if np.any(r <= 0): return False
    if np.any(x - r < -tol) or np.any(1 - x - r < -tol): return False
    if np.any(y - r < -tol) or np.any(1 - y - r < -tol): return False
    for i in range(n):
        for j in range(i+1, n):
            dist2 = (x[i]-x[j])**2 + (y[i]-y[j])**2
            sr = r[i] + r[j]
            if dist2 < sr*sr - 2*tol*sr:
                return False
    return True


def fast_slsqp(x0, y0, r0, maxiter=1000):
    """SLSQP with analytical constraint Jacobian."""
    n = len(x0)
    n_sep = n * (n - 1) // 2  # separation constraints
    n_wall = 4 * n  # wall constraints
    n_cons = n_sep + n_wall

    def objective(v):
        return -np.sum(v[2*n:3*n])

    def obj_jac(v):
        g = np.zeros(3*n)
        g[2*n:3*n] = -1.0
        return g

    # Build pairs list once
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((i, j))

    def all_constraints(v):
        """Return all constraint values as a single array."""
        x, y, r = v[:n], v[n:2*n], v[2*n:3*n]
        c = np.empty(n_cons)

        # Wall constraints: [L0, R0, B0, T0, L1, R1, B1, T1, ...]
        for i in range(n):
            c[4*i] = x[i] - r[i]          # left
            c[4*i+1] = 1.0 - x[i] - r[i]  # right
            c[4*i+2] = y[i] - r[i]         # bottom
            c[4*i+3] = 1.0 - y[i] - r[i]  # top

        # Separation constraints
        idx = n_wall
        for i, j in pairs:
            c[idx] = (x[i]-x[j])**2 + (y[i]-y[j])**2 - (r[i]+r[j])**2
            idx += 1

        return c

    def all_constraints_jac(v):
        """Jacobian of all constraints."""
        x, y, r = v[:n], v[n:2*n], v[2*n:3*n]
        J = np.zeros((n_cons, 3*n))

        # Wall constraints Jacobian
        for i in range(n):
            # x_i - r_i >= 0
            J[4*i, i] = 1.0; J[4*i, 2*n+i] = -1.0
            # 1 - x_i - r_i >= 0
            J[4*i+1, i] = -1.0; J[4*i+1, 2*n+i] = -1.0
            # y_i - r_i >= 0
            J[4*i+2, n+i] = 1.0; J[4*i+2, 2*n+i] = -1.0
            # 1 - y_i - r_i >= 0
            J[4*i+3, n+i] = -1.0; J[4*i+3, 2*n+i] = -1.0

        # Separation constraints Jacobian
        idx = n_wall
        for i, j in pairs:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            sr = r[i] + r[j]

            J[idx, i] = 2*dx
            J[idx, j] = -2*dx
            J[idx, n+i] = 2*dy
            J[idx, n+j] = -2*dy
            J[idx, 2*n+i] = -2*sr
            J[idx, 2*n+j] = -2*sr
            idx += 1

        return J

    constraints = [{
        'type': 'ineq',
        'fun': all_constraints,
        'jac': all_constraints_jac,
    }]

    bounds = [(0.001, 0.999)]*n + [(0.001, 0.999)]*n + [(0.001, 0.5)]*n
    v0 = np.concatenate([x0, y0, r0])

    result = minimize(
        objective, v0, method='SLSQP',
        jac=obj_jac,
        constraints=constraints,
        bounds=bounds,
        options={'maxiter': maxiter, 'ftol': 1e-15}
    )

    x, y, r = result.x[:n], result.x[n:2*n], result.x[2*n:3*n]
    return x, y, r, np.sum(r), is_feasible(x, y, r)


def topo_fp(x, y, r, tol=1e-5):
    """Quick topology fingerprint based on contact pattern."""
    n = len(x)
    contacts = []
    for i in range(n):
        if abs(x[i]-r[i]) < tol: contacts.append(f"L{i}")
        if abs(1-x[i]-r[i]) < tol: contacts.append(f"R{i}")
        if abs(y[i]-r[i]) < tol: contacts.append(f"B{i}")
        if abs(1-y[i]-r[i]) < tol: contacts.append(f"T{i}")
    for i in range(n):
        for j in range(i+1, n):
            if abs(np.sqrt((x[i]-x[j])**2+(y[i]-y[j])**2) - r[i]-r[j]) < tol:
                contacts.append(f"C{i}_{j}")
    return hash(tuple(sorted(contacts)))


def main():
    t0 = time.time()
    rng = np.random.RandomState(SEED)

    known_path = os.path.join(WORKDIR, '..', 'topo-001', 'solution_n26.json')
    xk, yk, rk = load_solution(known_path)
    known_metric = np.sum(rk)
    print(f"Known best: {known_metric:.10f}")

    # Speed test
    print("\nSpeed test...")
    st = time.time()
    x2, y2, r2, m, f = fast_slsqp(xk, yk, rk, maxiter=100)
    print(f"  From known: metric={m:.10f}, feasible={f}, time={time.time()-st:.2f}s")

    # Test with perturbation
    x0 = xk + rng.uniform(-0.03, 0.03, N)
    y0 = yk + rng.uniform(-0.03, 0.03, N)
    r0 = rk * rng.uniform(0.95, 1.05, N)
    r0 = np.clip(r0, 0.01, 0.49)
    x0 = np.clip(x0, r0+0.001, 1-r0-0.001)
    y0 = np.clip(y0, r0+0.001, 1-r0-0.001)
    st = time.time()
    x2, y2, r2, m, f = fast_slsqp(x0, y0, r0)
    print(f"  Perturbed:  metric={m:.10f}, feasible={f}, time={time.time()-st:.2f}s")

    best_metric = known_metric
    best_sol = (xk.copy(), yk.copy(), rk.copy())
    seen = {topo_fp(xk, yk, rk)}
    n_new = 0

    # ===== Main loop: perturbed starts =====
    print(f"\n{'='*60}")
    print("SEARCH: Perturbed starts from known best")
    print(f"{'='*60}")

    for att in range(2000):
        # Varying perturbation strength
        if att < 500:
            strength = rng.uniform(0.02, 0.10)
        elif att < 1000:
            strength = rng.uniform(0.05, 0.20)
        elif att < 1500:
            strength = rng.uniform(0.10, 0.35)
        else:
            strength = rng.uniform(0.15, 0.50)

        x0 = xk + rng.uniform(-strength, strength, N)
        y0 = yk + rng.uniform(-strength, strength, N)
        r0 = rk * rng.uniform(max(0.5, 1-strength*2), min(1.5, 1+strength*2), N)
        r0 = np.clip(r0, 0.01, 0.49)
        x0 = np.clip(x0, r0+0.001, 1-r0-0.001)
        y0 = np.clip(y0, r0+0.001, 1-r0-0.001)

        x2, y2, r2, metric, feasible = fast_slsqp(x0, y0, r0)

        if feasible and metric > 2.60:
            fp = topo_fp(x2, y2, r2)
            if fp not in seen:
                seen.add(fp)
                n_new += 1
                if metric > 2.635:
                    print(f"  [{att+1}] New topology: {metric:.10f}")

            if metric > best_metric:
                best_metric = metric
                best_sol = (x2.copy(), y2.copy(), r2.copy())
                print(f"  [{att+1}] *** NEW BEST: {metric:.10f} ***")

        if (att + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  [{att+1}/2000] best={best_metric:.10f}, "
                  f"new_topos={n_new}, time={elapsed:.0f}s")

    # Save
    sol_path = os.path.join(WORKDIR, 'solution_n26.json')
    save_solution(*best_sol, sol_path)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"FINAL: {best_metric:.10f} (known={known_metric:.10f})")
    print(f"Improvement: {best_metric - known_metric:.2e}")
    print(f"Unique topologies (>2.60): {n_new}")
    print(f"Time: {elapsed:.0f}s")

    return best_metric


if __name__ == '__main__':
    main()
