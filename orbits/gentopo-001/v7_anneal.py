"""
v7: Dual Annealing + Basin Hopping with tight time budget.

dual_annealing combines classical simulated annealing with a local search.
basin_hopping is designed specifically for escaping local minima.

Both use the penalty formulation since they handle bounds but not constraints.
After finding promising candidates, polish with fast SLSQP.
"""

import json
import numpy as np
from scipy.optimize import dual_annealing, basinhopping, minimize
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
            if dist2 < sr*sr - 2*tol*sr: return False
    return True


def penalty_obj(v, mu=10000):
    """Penalty objective for global optimizers."""
    n = N
    x, y, r = v[:n], v[n:2*n], v[2*n:3*n]

    obj = -np.sum(r)

    # Containment
    obj += mu * (np.sum(np.maximum(0, r-x)**2) + np.sum(np.maximum(0, x+r-1)**2) +
                 np.sum(np.maximum(0, r-y)**2) + np.sum(np.maximum(0, y+r-1)**2))

    # Non-overlap
    for i in range(n):
        dx = x[i] - x[i+1:]
        dy = y[i] - y[i+1:]
        dist2 = dx**2 + dy**2
        sr = r[i] + r[i+1:]
        viol = np.maximum(0, sr**2 - dist2)
        obj += mu * np.sum(viol)

    return obj


def fast_slsqp(x0, y0, r0, maxiter=2000):
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
            dx = x[i]-x[j]; dy = y[i]-y[j]; sr = r[i]+r[j]
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


class BHCallback:
    def __init__(self):
        self.best = float('inf')
        self.n_calls = 0
        self.candidates = []

    def __call__(self, x, f, accepted):
        self.n_calls += 1
        if f < self.best:
            self.best = f
            n = N
            xc, yc, rc = x[:n], x[n:2*n], x[2*n:3*n]
            metric = np.sum(rc)
            if is_feasible(xc, yc, rc, 1e-4):
                self.candidates.append(x.copy())
            if self.n_calls % 50 == 0:
                print(f"  BH step {self.n_calls}: best_penalty={f:.4f}, metric~{metric:.6f}, accepted={accepted}")
        elif self.n_calls % 100 == 0:
            print(f"  BH step {self.n_calls}: current penalty best={self.best:.4f}")


def main():
    t0 = time.time()
    rng = np.random.RandomState(SEED)

    known_path = os.path.join(WORKDIR, '..', 'topo-001', 'solution_n26.json')
    xk, yk, rk = load_solution(known_path)
    known_metric = np.sum(rk)
    print(f"Known best: {known_metric:.10f}")

    best_metric = known_metric
    best_sol = (xk.copy(), yk.copy(), rk.copy())

    # ===== Basin Hopping =====
    print(f"\n{'='*60}")
    print("Basin Hopping (from known best, 300 iterations)")
    print(f"{'='*60}")

    n = N
    bounds_list = [(0.001, 0.999)]*n + [(0.001, 0.999)]*n + [(0.001, 0.5)]*n
    v_known = np.concatenate([xk, yk, rk])

    callback = BHCallback()

    # Custom step: larger steps in position, smaller in radius
    class CustomStep:
        def __init__(self, stepsize=0.05):
            self.stepsize = stepsize
            self.rng = np.random.RandomState(SEED + 1)

        def __call__(self, x):
            n = N
            s = self.stepsize
            x_new = x.copy()
            # Larger position perturbation
            x_new[:n] += self.rng.uniform(-s, s, n)
            x_new[n:2*n] += self.rng.uniform(-s, s, n)
            # Smaller radius perturbation
            x_new[2*n:3*n] *= self.rng.uniform(1-s*0.5, 1+s*0.5, n)
            # Clamp
            x_new[:n] = np.clip(x_new[:n], 0.001, 0.999)
            x_new[n:2*n] = np.clip(x_new[n:2*n], 0.001, 0.999)
            x_new[2*n:3*n] = np.clip(x_new[2*n:3*n], 0.001, 0.5)
            return x_new

    for stepsize in [0.03, 0.05, 0.10, 0.20]:
        print(f"\n  stepsize={stepsize}:")
        result = basinhopping(
            penalty_obj, v_known,
            niter=75,
            T=1.0,
            stepsize=stepsize,
            minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': bounds_list,
                              'options': {'maxiter': 200}},
            callback=callback,
            take_step=CustomStep(stepsize),
            seed=SEED,
        )

        x_bh, y_bh, r_bh = result.x[:n], result.x[n:2*n], result.x[2*n:3*n]
        bh_metric = np.sum(r_bh)
        print(f"  BH result: penalty={result.fun:.4f}, metric~{bh_metric:.6f}")

    # Polish all candidates
    print(f"\nPolishing {len(callback.candidates)} candidates...")
    for i, cand in enumerate(callback.candidates):
        x0, y0, r0 = cand[:n], cand[n:2*n], cand[2*n:3*n]
        x3, y3, r3, metric3, f3 = fast_slsqp(x0, y0, r0)
        if f3 and metric3 > best_metric:
            best_metric = metric3
            best_sol = (x3, y3, r3)
            print(f"  Candidate {i}: *** NEW BEST: {best_metric:.10f} ***")
        if (i+1) % 20 == 0:
            print(f"  Polished {i+1}/{len(callback.candidates)}")

    # ===== Dual Annealing (reduced dimension) =====
    print(f"\n{'='*60}")
    print("Dual Annealing (maxiter=500)")
    print(f"{'='*60}")

    bounds_da = list(zip([0.001]*n + [0.001]*n + [0.001]*n,
                         [0.999]*n + [0.999]*n + [0.5]*n))

    da_result = dual_annealing(
        penalty_obj, bounds_da,
        x0=v_known,
        maxiter=500,
        seed=SEED,
        initial_temp=5230,
        restart_temp_ratio=2e-5,
        visit=2.62,
        accept=-5.0,
        maxfun=100000,
    )

    x_da, y_da, r_da = da_result.x[:n], da_result.x[n:2*n], da_result.x[2*n:3*n]
    da_metric = np.sum(r_da)
    print(f"DA raw: metric~{da_metric:.6f}, penalty={da_result.fun:.4f}")

    # Polish
    x3, y3, r3, metric3, f3 = fast_slsqp(x_da, y_da, r_da)
    print(f"DA polished: {metric3:.10f}, feasible={f3}")
    if f3 and metric3 > best_metric:
        best_metric = metric3
        best_sol = (x3, y3, r3)
        print(f"*** NEW BEST: {best_metric:.10f} ***")

    # Save
    sol_path = os.path.join(WORKDIR, 'solution_n26.json')
    save_solution(*best_sol, sol_path)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"FINAL: {best_metric:.10f} (known={known_metric:.10f})")
    print(f"Time: {elapsed:.0f}s")

    return best_metric


if __name__ == '__main__':
    main()
