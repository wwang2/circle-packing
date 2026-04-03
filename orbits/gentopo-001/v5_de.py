"""
v5: Differential Evolution + Local Refinement.

DE is a global optimizer that maintains a population of solutions and
uses mutation/crossover to explore. It can jump between basins that
gradient methods cannot escape.

Strategy:
1. Run DE with custom initialization (seeded with known best + perturbations)
2. Use penalty formulation for DE (it handles bounds but not constraints)
3. After DE converges, polish top solutions with fast SLSQP
"""

import json
import numpy as np
from scipy.optimize import differential_evolution, minimize
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


def penalty_de(v):
    """Penalty objective for DE: -sum(r) + penalties."""
    n = N
    x = v[:n]
    y = v[n:2*n]
    r = v[2*n:3*n]

    obj = -np.sum(r)
    mu = 1000.0  # Fixed penalty weight for DE

    # Containment
    vL = np.maximum(0, r - x)
    vR = np.maximum(0, x + r - 1)
    vB = np.maximum(0, r - y)
    vT = np.maximum(0, y + r - 1)
    obj += mu * (np.sum(vL**2) + np.sum(vR**2) + np.sum(vB**2) + np.sum(vT**2))

    # Non-overlap
    for i in range(n):
        dx = x[i] - x[i+1:]
        dy = y[i] - y[i+1:]
        dist2 = dx**2 + dy**2
        sr = r[i] + r[i+1:]
        viol = np.maximum(0, sr**2 - dist2)
        obj += mu * np.sum(viol)

    return obj


def fast_slsqp(x0, y0, r0, maxiter=1000):
    """SLSQP with analytical Jacobian."""
    n = len(x0)

    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    n_wall = 4 * n
    n_sep = len(pairs)
    n_cons = n_wall + n_sep

    def objective(v):
        return -np.sum(v[2*n:3*n])

    def obj_jac(v):
        g = np.zeros(3*n)
        g[2*n:3*n] = -1.0
        return g

    def all_constraints(v):
        x, y, r = v[:n], v[n:2*n], v[2*n:3*n]
        c = np.empty(n_cons)
        for i in range(n):
            c[4*i] = x[i] - r[i]
            c[4*i+1] = 1.0 - x[i] - r[i]
            c[4*i+2] = y[i] - r[i]
            c[4*i+3] = 1.0 - y[i] - r[i]
        idx = n_wall
        for i, j in pairs:
            c[idx] = (x[i]-x[j])**2 + (y[i]-y[j])**2 - (r[i]+r[j])**2
            idx += 1
        return c

    def all_constraints_jac(v):
        x, y, r = v[:n], v[n:2*n], v[2*n:3*n]
        J = np.zeros((n_cons, 3*n))
        for i in range(n):
            J[4*i, i] = 1.0; J[4*i, 2*n+i] = -1.0
            J[4*i+1, i] = -1.0; J[4*i+1, 2*n+i] = -1.0
            J[4*i+2, n+i] = 1.0; J[4*i+2, 2*n+i] = -1.0
            J[4*i+3, n+i] = -1.0; J[4*i+3, 2*n+i] = -1.0
        idx = n_wall
        for i, j in pairs:
            dx = x[i] - x[j]; dy = y[i] - y[j]; sr = r[i] + r[j]
            J[idx, i] = 2*dx; J[idx, j] = -2*dx
            J[idx, n+i] = 2*dy; J[idx, n+j] = -2*dy
            J[idx, 2*n+i] = -2*sr; J[idx, 2*n+j] = -2*sr
            idx += 1
        return J

    constraints = [{'type': 'ineq', 'fun': all_constraints, 'jac': all_constraints_jac}]
    bounds = [(0.001, 0.999)]*n + [(0.001, 0.999)]*n + [(0.001, 0.5)]*n
    v0 = np.concatenate([x0, y0, r0])

    result = minimize(objective, v0, method='SLSQP', jac=obj_jac,
                      constraints=constraints, bounds=bounds,
                      options={'maxiter': maxiter, 'ftol': 1e-15})

    x, y, r = result.x[:n], result.x[n:2*n], result.x[2*n:3*n]
    return x, y, r, np.sum(r), is_feasible(x, y, r)


class DECallback:
    """Track DE progress and collect promising solutions."""
    def __init__(self):
        self.best = float('inf')
        self.history = []
        self.gen = 0
        self.candidates = []

    def __call__(self, xk, convergence):
        self.gen += 1
        val = penalty_de(xk)
        if val < self.best:
            self.best = val
            metric = np.sum(xk[2*N:3*N])
            self.history.append((self.gen, metric))
            if self.gen % 50 == 0 or metric > 2.60:
                print(f"  DE gen {self.gen}: best_penalty={val:.6f}, sum_r={metric:.6f}")
        elif self.gen % 100 == 0:
            metric = np.sum(xk[2*N:3*N])
            print(f"  DE gen {self.gen}: best_penalty={self.best:.6f}, sum_r={metric:.6f}")

        # Save promising candidates
        n = N
        x, y, r = xk[:n], xk[n:2*n], xk[2*n:3*n]
        if is_feasible(x, y, r, tol=1e-4):
            self.candidates.append(xk.copy())

        return False  # Don't stop


def main():
    t0 = time.time()

    known_path = os.path.join(WORKDIR, '..', 'topo-001', 'solution_n26.json')
    xk, yk, rk = load_solution(known_path)
    known_metric = np.sum(rk)
    print(f"Known best: {known_metric:.10f}")

    best_metric = known_metric
    best_sol = (xk.copy(), yk.copy(), rk.copy())

    # ===== DE with custom init =====
    print(f"\n{'='*60}")
    print("Differential Evolution (popsize=100, maxiter=500)")
    print(f"{'='*60}")

    n = N
    bounds = [(0.001, 0.999)]*n + [(0.001, 0.999)]*n + [(0.001, 0.5)]*n

    # Custom initialization: mix of known best perturbations + random
    rng = np.random.RandomState(SEED)
    popsize = 100
    init_pop = np.zeros((popsize, 3*n))

    # 20% from known best with perturbations
    for i in range(20):
        strength = rng.uniform(0.01, 0.15)
        x0 = xk + rng.uniform(-strength, strength, n)
        y0 = yk + rng.uniform(-strength, strength, n)
        r0 = rk * rng.uniform(0.9, 1.1, n)
        r0 = np.clip(r0, 0.01, 0.49)
        x0 = np.clip(x0, r0+0.001, 1-r0-0.001)
        y0 = np.clip(y0, r0+0.001, 1-r0-0.001)
        init_pop[i] = np.concatenate([x0, y0, r0])

    # 80% random
    for i in range(20, popsize):
        r0 = rng.uniform(0.02, 0.12, n)
        x0 = rng.uniform(0.05, 0.95, n)
        y0 = rng.uniform(0.05, 0.95, n)
        r0 = np.clip(r0, 0.001, 0.499)
        x0 = np.clip(x0, r0+0.001, 1-r0-0.001)
        y0 = np.clip(y0, r0+0.001, 1-r0-0.001)
        init_pop[i] = np.concatenate([x0, y0, r0])

    callback = DECallback()

    result = differential_evolution(
        penalty_de,
        bounds=bounds,
        init=init_pop,
        maxiter=500,
        popsize=popsize,
        mutation=(0.5, 1.5),
        recombination=0.9,
        seed=SEED,
        tol=1e-12,
        callback=callback,
        workers=1,
        updating='deferred',
        polish=False,
    )

    print(f"\nDE finished: {result.message}")
    x_de, y_de, r_de = result.x[:n], result.x[n:2*n], result.x[2*n:3*n]
    de_metric = np.sum(r_de)
    print(f"DE raw metric: {de_metric:.10f}, feasible: {is_feasible(x_de, y_de, r_de, 1e-4)}")

    # Polish DE result with SLSQP
    print("\nPolishing DE result with SLSQP...")
    x3, y3, r3, metric3, f3 = fast_slsqp(x_de, y_de, r_de)
    print(f"Polished: {metric3:.10f}, feasible: {f3}")

    if f3 and metric3 > best_metric:
        best_metric = metric3
        best_sol = (x3.copy(), y3.copy(), r3.copy())
        print(f"*** NEW BEST: {best_metric:.10f} ***")

    # Polish all collected candidates
    print(f"\nPolishing {len(callback.candidates)} DE candidates...")
    for i, cand in enumerate(callback.candidates):
        x0, y0, r0 = cand[:n], cand[n:2*n], cand[2*n:3*n]
        x3, y3, r3, metric3, f3 = fast_slsqp(x0, y0, r0)
        if f3 and metric3 > best_metric:
            best_metric = metric3
            best_sol = (x3.copy(), y3.copy(), r3.copy())
            print(f"  Candidate {i}: *** NEW BEST: {best_metric:.10f} ***")
        if (i+1) % 10 == 0:
            print(f"  Polished {i+1}/{len(callback.candidates)}")

    # Save
    sol_path = os.path.join(WORKDIR, 'solution_n26.json')
    save_solution(*best_sol, sol_path)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"FINAL: {best_metric:.10f} (known={known_metric:.10f})")
    print(f"Improvement: {best_metric - known_metric:.2e}")
    print(f"Time: {elapsed:.0f}s")

    return best_metric


if __name__ == '__main__':
    main()
