"""
Global optimization approaches for circle packing n=26.
Uses scipy.optimize.dual_annealing and basinhopping with custom moves.
These are true global optimizers that can escape deep local basins.
"""

import json
import numpy as np
from scipy.optimize import minimize, dual_annealing, basinhopping
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

def is_feasible(x, y, r, tol=1e-10):
    n = len(x)
    for i in range(n):
        if r[i] <= 0: return False
        if x[i] - r[i] < -tol or 1 - x[i] - r[i] < -tol: return False
        if y[i] - r[i] < -tol or 1 - y[i] - r[i] < -tol: return False
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if dist < r[i] + r[j] - tol: return False
    return True

def penalty_obj(params):
    """Penalized objective for unconstrained optimizers."""
    n = N
    x = params[:n]
    y = params[n:2*n]
    r = params[2*n:3*n]

    obj = -np.sum(r)
    penalty = 0.0
    pw = 1e4

    for i in range(n):
        v = max(0, r[i] - x[i]); penalty += v**2
        v = max(0, x[i] + r[i] - 1); penalty += v**2
        v = max(0, r[i] - y[i]); penalty += v**2
        v = max(0, y[i] + r[i] - 1); penalty += v**2
        v = max(0, -r[i]); penalty += v**2

    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            v = max(0, r[i] + r[j] - dist)
            penalty += v**2

    return obj + pw * penalty

def optimize_slsqp_jac(x0, y0, r0, maxiter=15000):
    """SLSQP with analytical Jacobian."""
    n = len(x0)
    params0 = np.concatenate([x0, y0, r0])

    constraints = []
    for i in range(n):
        def cf_xl(p, i=i): return p[i] - p[2*n+i]
        def cj_xl(p, i=i):
            g = np.zeros(3*n); g[i] = 1.0; g[2*n+i] = -1.0; return g
        constraints.append({'type': 'ineq', 'fun': cf_xl, 'jac': cj_xl})

        def cf_xr(p, i=i): return 1 - p[i] - p[2*n+i]
        def cj_xr(p, i=i):
            g = np.zeros(3*n); g[i] = -1.0; g[2*n+i] = -1.0; return g
        constraints.append({'type': 'ineq', 'fun': cf_xr, 'jac': cj_xr})

        def cf_yl(p, i=i): return p[n+i] - p[2*n+i]
        def cj_yl(p, i=i):
            g = np.zeros(3*n); g[n+i] = 1.0; g[2*n+i] = -1.0; return g
        constraints.append({'type': 'ineq', 'fun': cf_yl, 'jac': cj_yl})

        def cf_yr(p, i=i): return 1 - p[n+i] - p[2*n+i]
        def cj_yr(p, i=i):
            g = np.zeros(3*n); g[n+i] = -1.0; g[2*n+i] = -1.0; return g
        constraints.append({'type': 'ineq', 'fun': cf_yr, 'jac': cj_yr})

        def cf_rp(p, i=i): return p[2*n+i] - 1e-6
        def cj_rp(p, i=i):
            g = np.zeros(3*n); g[2*n+i] = 1.0; return g
        constraints.append({'type': 'ineq', 'fun': cf_rp, 'jac': cj_rp})

    for i in range(n):
        for j in range(i+1, n):
            def cf_no(p, i=i, j=j):
                return (p[i]-p[j])**2 + (p[n+i]-p[n+j])**2 - (p[2*n+i]+p[2*n+j])**2
            def cj_no(p, i=i, j=j):
                g = np.zeros(3*n)
                dx = p[i]-p[j]; dy = p[n+i]-p[n+j]; sr = p[2*n+i]+p[2*n+j]
                g[i]=2*dx; g[j]=-2*dx; g[n+i]=2*dy; g[n+j]=-2*dy
                g[2*n+i]=-2*sr; g[2*n+j]=-2*sr
                return g
            constraints.append({'type': 'ineq', 'fun': cf_no, 'jac': cj_no})

    result = minimize(
        lambda p: (-np.sum(p[2*n:3*n]),
                   np.concatenate([np.zeros(2*n), -np.ones(n)])),
        params0, method='SLSQP', jac=True, constraints=constraints,
        options={'maxiter': maxiter, 'ftol': 1e-15, 'disp': False}
    )
    x, y, r = result.x[:n], result.x[n:2*n], result.x[2*n:3*n]
    return x, y, r, np.sum(r), result.success

class TopologyStepTaker:
    """Custom step taker for basinhopping that makes topology-changing moves."""
    def __init__(self, n, stepsize=0.1, seed=42):
        self.n = n
        self.stepsize = stepsize
        self.rng = np.random.RandomState(seed)

    def __call__(self, x):
        n = self.n
        params = x.copy()
        xc = params[:n]
        yc = params[n:2*n]
        rc = params[2*n:3*n]

        move = self.rng.randint(0, 6)

        if move == 0:
            # Relocate one circle
            i = self.rng.randint(n)
            xc[i] = self.rng.uniform(0.05, 0.95)
            yc[i] = self.rng.uniform(0.05, 0.95)
            rc[i] = self.rng.uniform(0.03, 0.15)
        elif move == 1:
            # Swap two circles
            i, j = self.rng.choice(n, 2, replace=False)
            xc[i], xc[j] = xc[j], xc[i]
            yc[i], yc[j] = yc[j], yc[i]
        elif move == 2:
            # Random perturbation
            scale = self.stepsize
            xc += self.rng.normal(0, scale, n)
            yc += self.rng.normal(0, scale, n)
            rc *= (1 + self.rng.normal(0, scale*0.5, n))
        elif move == 3:
            # Cycle 3 circles
            i, j, k = self.rng.choice(n, 3, replace=False)
            xc[i], xc[j], xc[k] = xc[k], xc[i], xc[j]
            yc[i], yc[j], yc[k] = yc[k], yc[i], yc[j]
        elif move == 4:
            # Reflect subset
            k = self.rng.randint(2, n//2)
            idxs = self.rng.choice(n, k, replace=False)
            if self.rng.random() < 0.5:
                xc[idxs] = 1 - xc[idxs]
            else:
                yc[idxs] = 1 - yc[idxs]
        elif move == 5:
            # Grow/shrink random subset
            k = self.rng.randint(1, n//3)
            idxs = self.rng.choice(n, k, replace=False)
            factor = self.rng.uniform(0.6, 1.4)
            rc[idxs] *= factor

        rc = np.maximum(rc, 0.005)
        xc = np.clip(xc, rc + 0.001, 1 - rc - 0.001)
        yc = np.clip(yc, rc + 0.001, 1 - rc - 0.001)

        params[:n] = xc
        params[n:2*n] = yc
        params[2*n:3*n] = rc
        return params

def main():
    t0 = time.time()
    parent_path = os.path.join(WORKDIR, '..', 'nlp-001', 'solution_n26.json')
    x0, y0, r0 = load_solution(parent_path)
    parent_metric = np.sum(r0)
    print(f"Parent metric: {parent_metric:.10f}")

    best_metric = parent_metric
    best_x, best_y, best_r = x0.copy(), y0.copy(), r0.copy()

    # ====== Approach 1: dual_annealing ======
    print("\n=== Dual Annealing ===")
    bounds = [(0.01, 0.99)]*N + [(0.01, 0.99)]*N + [(0.005, 0.25)]*N

    for seed in range(3):
        print(f"  DA seed={seed}...")
        t1 = time.time()
        result = dual_annealing(
            penalty_obj, bounds,
            seed=seed, maxiter=500,
            initial_temp=5230, restart_temp_ratio=2e-5,
            visit=2.62, accept=-5.0,
            x0=np.concatenate([x0, y0, r0])
        )
        params = result.x
        x, y, r = params[:N], params[N:2*N], params[2*N:3*N]

        # Polish with SLSQP
        r = np.maximum(r, 0.005)
        x = np.clip(x, r+0.001, 1-r-0.001)
        y = np.clip(y, r+0.001, 1-r-0.001)
        x, y, r, m, s = optimize_slsqp_jac(x, y, r, maxiter=10000)

        elapsed = time.time() - t1
        if s and is_feasible(x, y, r):
            print(f"  DA seed={seed}: {m:.10f} ({elapsed:.0f}s)")
            if m > best_metric:
                print(f"  IMPROVED!")
                best_metric, best_x, best_y, best_r = m, x, y, r
                save_solution(best_x, best_y, best_r,
                            os.path.join(WORKDIR, 'solution_n26_global.json'))
        else:
            print(f"  DA seed={seed}: infeasible after polish ({elapsed:.0f}s)")

    # ====== Approach 2: scipy.optimize.basinhopping ======
    print("\n=== Basin Hopping (scipy) ===")

    for seed in range(3):
        for stepsize in [0.05, 0.1, 0.2]:
            print(f"  BH seed={seed}, step={stepsize}...")
            t1 = time.time()

            step_taker = TopologyStepTaker(N, stepsize=stepsize, seed=seed)

            result = basinhopping(
                penalty_obj,
                np.concatenate([x0, y0, r0]),
                minimizer_kwargs={'method': 'L-BFGS-B',
                                  'bounds': bounds,
                                  'options': {'maxiter': 1000}},
                niter=100,
                T=0.5,
                stepsize=stepsize,
                take_step=step_taker,
                seed=seed
            )

            params = result.x
            x, y, r = params[:N], params[N:2*N], params[2*N:3*N]
            r = np.maximum(r, 0.005)
            x = np.clip(x, r+0.001, 1-r-0.001)
            y = np.clip(y, r+0.001, 1-r-0.001)
            x, y, r, m, s = optimize_slsqp_jac(x, y, r, maxiter=10000)

            elapsed = time.time() - t1
            if s and is_feasible(x, y, r):
                print(f"    Result: {m:.10f} ({elapsed:.0f}s)")
                if m > best_metric:
                    print(f"    IMPROVED!")
                    best_metric, best_x, best_y, best_r = m, x, y, r
                    save_solution(best_x, best_y, best_r,
                                os.path.join(WORKDIR, 'solution_n26_global.json'))
            else:
                print(f"    Infeasible ({elapsed:.0f}s)")

    # ====== Approach 3: Multi-start from truly random ======
    print("\n=== Multi-Start Random (with penalty -> SLSQP) ===")
    rng = np.random.RandomState(9999)

    for trial in range(100):
        # Generate random packing
        x = rng.uniform(0.06, 0.94, N)
        y = rng.uniform(0.06, 0.94, N)
        r = rng.uniform(0.03, 0.12, N)

        # Quick penalty optimization
        params = np.concatenate([x, y, r])
        result = minimize(penalty_obj, params, method='L-BFGS-B',
                         bounds=bounds, options={'maxiter': 2000})
        params = result.x
        x, y, r = params[:N], params[N:2*N], params[2*N:3*N]

        # SLSQP polish
        r = np.maximum(r, 0.005)
        x = np.clip(x, r+0.001, 1-r-0.001)
        y = np.clip(y, r+0.001, 1-r-0.001)
        x, y, r, m, s = optimize_slsqp_jac(x, y, r, maxiter=8000)

        if s and is_feasible(x, y, r) and m > best_metric:
            print(f"  Trial {trial}: IMPROVED to {m:.10f}")
            best_metric, best_x, best_y, best_r = m, x, y, r

        if trial % 20 == 0:
            elapsed = time.time() - t0
            print(f"  Trial {trial}/100, {elapsed:.0f}s, best={best_metric:.10f}")

    # Save
    save_solution(best_x, best_y, best_r, os.path.join(WORKDIR, 'solution_n26_global.json'))

    elapsed = time.time() - t0
    print(f"\n=== FINAL ===")
    print(f"Parent:  {parent_metric:.10f}")
    print(f"Best:    {best_metric:.10f}")
    print(f"Delta:   {best_metric - parent_metric:.2e}")
    print(f"Time:    {elapsed:.0f}s")

if __name__ == '__main__':
    main()
