"""
v3: Fast vectorized topology search.

Key optimizations:
1. Penalty method only (no SLSQP polish unless very promising)
2. Vectorized constraint evaluation
3. More penalty iterations with higher mu
4. Only polish candidates with penalty-metric > 2.62
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


def penalty_obj_grad(v, n, mu):
    """Combined objective + gradient for penalty method."""
    x = v[:n]
    y = v[n:2*n]
    r = v[2*n:3*n]

    obj = -np.sum(r)
    grad = np.zeros(3*n)
    grad[2*n:3*n] = -1.0

    # Containment violations
    vL = r - x; mask = vL > 0
    obj += mu * np.sum(vL[mask]**2)
    grad[:n][mask] += mu * (-2*vL[mask])
    grad[2*n:3*n][mask] += mu * (2*vL[mask])

    vR = x + r - 1; mask = vR > 0
    obj += mu * np.sum(vR[mask]**2)
    grad[:n][mask] += mu * (2*vR[mask])
    grad[2*n:3*n][mask] += mu * (2*vR[mask])

    vB = r - y; mask = vB > 0
    obj += mu * np.sum(vB[mask]**2)
    grad[n:2*n][mask] += mu * (-2*vB[mask])
    grad[2*n:3*n][mask] += mu * (2*vB[mask])

    vT = y + r - 1; mask = vT > 0
    obj += mu * np.sum(vT[mask]**2)
    grad[n:2*n][mask] += mu * (2*vT[mask])
    grad[2*n:3*n][mask] += mu * (2*vT[mask])

    # Non-overlap violations (vectorized upper triangle)
    for i in range(n):
        dx = x[i] - x[i+1:]
        dy = y[i] - y[i+1:]
        dist2 = dx**2 + dy**2
        sr = r[i] + r[i+1:]
        viol = sr**2 - dist2
        mask = viol > 0
        if np.any(mask):
            obj += mu * np.sum(viol[mask])
            js = np.arange(i+1, n)[mask]
            dx_m = dx[mask]
            dy_m = dy[mask]
            sr_m = sr[mask]

            grad[i] += mu * np.sum(-2*dx_m)
            grad[js] += mu * (2*dx_m)
            grad[n+i] += mu * np.sum(-2*dy_m)
            grad[n+js] += mu * (2*dy_m)
            grad[2*n+i] += mu * np.sum(2*sr_m)
            grad[2*n+js] += mu * (2*sr_m)

    return obj, grad


def optimize_penalty_fast(x0, y0, r0):
    """Fast penalty optimization with increasing mu."""
    n = len(x0)
    v = np.concatenate([x0, y0, r0])
    bounds = [(0.001, 0.999)]*n + [(0.001, 0.999)]*n + [(0.001, 0.5)]*n

    for mu in [1, 10, 100, 1000, 10000, 100000, 1000000]:
        result = minimize(
            lambda v: penalty_obj_grad(v, n, mu),
            v, method='L-BFGS-B', bounds=bounds,
            jac=True,
            options={'maxiter': 200, 'ftol': 1e-14}
        )
        v = result.x

    x, y, r = v[:n], v[n:2*n], v[2*n:3*n]

    # Repair feasibility
    for _ in range(100):
        ok = True
        for i in range(n):
            if x[i] < r[i]: r[i] = x[i] - 1e-8; ok = False
            if x[i] + r[i] > 1: r[i] = 1 - x[i] - 1e-8; ok = False
            if y[i] < r[i]: r[i] = y[i] - 1e-8; ok = False
            if y[i] + r[i] > 1: r[i] = 1 - y[i] - 1e-8; ok = False
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if dist < r[i] + r[j]:
                    excess = (r[i]+r[j]-dist)/2 + 1e-8
                    r[i] -= excess; r[j] -= excess
                    ok = False
        if ok: break

    r = np.maximum(r, 0.001)
    return x, y, r, np.sum(r), is_feasible(x, y, r, 1e-8)


def slsqp_polish(x0, y0, r0):
    """SLSQP polish for final precision."""
    n = len(x0)

    # Use vectorized constraints for speed
    def make_constraint(typ, i, j=None):
        if typ == 'wall_L':
            return {'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2*n+i]}
        elif typ == 'wall_R':
            return {'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[i] - v[2*n+i]}
        elif typ == 'wall_B':
            return {'type': 'ineq', 'fun': lambda v, i=i: v[n+i] - v[2*n+i]}
        elif typ == 'wall_T':
            return {'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[n+i] - v[2*n+i]}
        elif typ == 'sep':
            return {'type': 'ineq', 'fun': lambda v, i=i, j=j: (
                (v[i]-v[j])**2 + (v[n+i]-v[n+j])**2 - (v[2*n+i]+v[2*n+j])**2
            )}

    constraints = []
    for i in range(n):
        constraints.extend([
            make_constraint('wall_L', i),
            make_constraint('wall_R', i),
            make_constraint('wall_B', i),
            make_constraint('wall_T', i),
        ])
    for i in range(n):
        for j in range(i+1, n):
            constraints.append(make_constraint('sep', i, j))

    bounds = [(0.001, 0.999)]*n + [(0.001, 0.999)]*n + [(0.001, 0.5)]*n

    def obj(v): return -np.sum(v[2*n:3*n])
    def obj_jac(v):
        g = np.zeros(3*n)
        g[2*n:3*n] = -1.0
        return g

    v0 = np.concatenate([x0, y0, r0])
    result = minimize(obj, v0, method='SLSQP', jac=obj_jac,
                      constraints=constraints, bounds=bounds,
                      options={'maxiter': 2000, 'ftol': 1e-15})

    x, y, r = result.x[:n], result.x[n:2*n], result.x[2*n:3*n]
    return x, y, r, np.sum(r), is_feasible(x, y, r)


def topo_fp(x, y, r, tol=1e-5):
    """Quick topology fingerprint."""
    n = len(x)
    parts = []
    for i in range(n):
        if abs(x[i]-r[i]) < tol: parts.append(f"L{i}")
        if abs(1-x[i]-r[i]) < tol: parts.append(f"R{i}")
        if abs(y[i]-r[i]) < tol: parts.append(f"B{i}")
        if abs(1-y[i]-r[i]) < tol: parts.append(f"T{i}")
    for i in range(n):
        for j in range(i+1, n):
            if abs(np.sqrt((x[i]-x[j])**2+(y[i]-y[j])**2) - r[i]-r[j]) < tol:
                parts.append(f"C{i}_{j}")
    return hash(tuple(sorted(parts)))


def main():
    t0 = time.time()
    rng = np.random.RandomState(SEED)

    known_path = os.path.join(WORKDIR, '..', 'topo-001', 'solution_n26.json')
    xk, yk, rk = load_solution(known_path)
    known_metric = np.sum(rk)
    print(f"Known best: {known_metric:.10f}")

    best_metric = known_metric
    best_sol = (xk.copy(), yk.copy(), rk.copy())
    seen = {topo_fp(xk, yk, rk)}
    n_new = 0
    metrics_found = []

    # ===== Phase 1: Basin hopping with penalty method (fast) =====
    print(f"\n{'='*60}")
    print("PHASE 1: Fast penalty basin hopping (1000 attempts)")
    print(f"{'='*60}")

    for att in range(1000):
        # Varying perturbation
        if att < 200:
            strength = rng.uniform(0.03, 0.15)
        elif att < 500:
            strength = rng.uniform(0.10, 0.35)
        else:
            strength = rng.uniform(0.20, 0.60)

        x0 = xk + rng.uniform(-strength, strength, N)
        y0 = yk + rng.uniform(-strength, strength, N)
        r0 = rk * rng.uniform(max(0.3, 1-strength*2), min(1.7, 1+strength*2), N)
        r0 = np.clip(r0, 0.01, 0.49)
        x0 = np.clip(x0, r0+0.001, 1-r0-0.001)
        y0 = np.clip(y0, r0+0.001, 1-r0-0.001)

        x2, y2, r2, metric, feasible = optimize_penalty_fast(x0, y0, r0)

        if feasible and metric > 2.55:
            metrics_found.append(metric)
            fp = topo_fp(x2, y2, r2)
            if fp not in seen:
                seen.add(fp)
                n_new += 1

            # Only SLSQP polish if very promising
            if metric > 2.62:
                x2, y2, r2, metric, feasible = slsqp_polish(x2, y2, r2)
                if feasible and metric > best_metric:
                    best_metric = metric
                    best_sol = (x2.copy(), y2.copy(), r2.copy())
                    print(f"  [{att+1}] *** NEW BEST: {metric:.10f} ***")

        if (att + 1) % 200 == 0:
            elapsed = time.time() - t0
            n_good = sum(1 for m in metrics_found if m > 2.60)
            print(f"  [{att+1}/1000] best={best_metric:.10f}, "
                  f"new_topos={n_new}, >2.60={n_good}, "
                  f"time={elapsed:.0f}s")

    # ===== Phase 2: Random starts (no known-best bias) =====
    print(f"\n{'='*60}")
    print("PHASE 2: Pure random starts (1000 attempts)")
    print(f"{'='*60}")

    for att in range(1000):
        # Completely random initialization
        r0 = rng.uniform(0.02, 0.12, N)
        x0 = rng.uniform(0.05, 0.95, N)
        y0 = rng.uniform(0.05, 0.95, N)

        # Make sure circles fit
        r0 = np.clip(r0, 0.01, 0.49)
        x0 = np.clip(x0, r0+0.001, 1-r0-0.001)
        y0 = np.clip(y0, r0+0.001, 1-r0-0.001)

        x2, y2, r2, metric, feasible = optimize_penalty_fast(x0, y0, r0)

        if feasible and metric > 2.55:
            metrics_found.append(metric)

            if metric > 2.62:
                x2, y2, r2, metric, feasible = slsqp_polish(x2, y2, r2)
                if feasible and metric > best_metric:
                    best_metric = metric
                    best_sol = (x2.copy(), y2.copy(), r2.copy())
                    print(f"  [{att+1}] *** NEW BEST: {metric:.10f} ***")

        if (att + 1) % 200 == 0:
            elapsed = time.time() - t0
            n_good = sum(1 for m in metrics_found if m > 2.60)
            print(f"  [{att+1}/1000] best={best_metric:.10f}, "
                  f">2.60={n_good}, time={elapsed:.0f}s")

    # ===== Phase 3: Structured starts (rings, grids, etc.) =====
    print(f"\n{'='*60}")
    print("PHASE 3: Structured starts")
    print(f"{'='*60}")

    def make_ring(n_per_ring, radii, dists, seed):
        """Generate ring layout."""
        rng2 = np.random.RandomState(seed)
        x, y, r = [], [], []
        for nr, rad, dist in zip(n_per_ring, radii, dists):
            for k in range(nr):
                angle = k * 2 * np.pi / nr + rng2.uniform(-0.2, 0.2)
                cx = 0.5 + dist * np.cos(angle)
                cy = 0.5 + dist * np.sin(angle)
                x.append(cx); y.append(cy); r.append(rad)
        x, y, r = np.array(x[:N]), np.array(y[:N]), np.array(r[:N])
        r = np.clip(r, 0.01, 0.49)
        x = np.clip(x, r+0.001, 1-r-0.001)
        y = np.clip(y, r+0.001, 1-r-0.001)
        return x, y, r

    configs = [
        ([1, 6, 12, 7], [0.14, 0.10, 0.08, 0.06], [0, 0.24, 0.40, 0.46]),
        ([1, 7, 12, 6], [0.14, 0.10, 0.08, 0.06], [0, 0.24, 0.40, 0.46]),
        ([1, 8, 12, 5], [0.13, 0.10, 0.07, 0.06], [0, 0.24, 0.40, 0.46]),
        ([1, 5, 10, 10], [0.15, 0.11, 0.08, 0.05], [0, 0.26, 0.40, 0.46]),
        ([1, 6, 8, 6, 5], [0.14, 0.10, 0.08, 0.07, 0.05], [0, 0.24, 0.36, 0.44, 0.47]),
        ([2, 6, 12, 6], [0.12, 0.10, 0.07, 0.06], [0.12, 0.26, 0.40, 0.46]),
        ([4, 8, 14], [0.12, 0.10, 0.06], [0.18, 0.34, 0.44]),
        ([1, 4, 8, 8, 5], [0.15, 0.12, 0.09, 0.07, 0.04], [0, 0.27, 0.38, 0.44, 0.47]),
        ([1, 6, 6, 6, 7], [0.14, 0.10, 0.09, 0.07, 0.04], [0, 0.24, 0.36, 0.44, 0.47]),
        ([3, 6, 10, 7], [0.12, 0.10, 0.08, 0.06], [0.14, 0.28, 0.40, 0.46]),
    ]

    for ci, (nring, rads, dists) in enumerate(configs):
        for seed_off in range(50):
            x0, y0, r0 = make_ring(nring, rads, dists, SEED+ci*100+seed_off)
            x0 += rng.uniform(-0.03, 0.03, N)
            y0 += rng.uniform(-0.03, 0.03, N)
            r0 *= rng.uniform(0.85, 1.15, N)
            r0 = np.clip(r0, 0.01, 0.49)
            x0 = np.clip(x0, r0+0.001, 1-r0-0.001)
            y0 = np.clip(y0, r0+0.001, 1-r0-0.001)

            x2, y2, r2, metric, feasible = optimize_penalty_fast(x0, y0, r0)

            if feasible and metric > 2.62:
                x2, y2, r2, metric, feasible = slsqp_polish(x2, y2, r2)
                if feasible and metric > best_metric:
                    best_metric = metric
                    best_sol = (x2.copy(), y2.copy(), r2.copy())
                    print(f"  [config {ci}, seed {seed_off}] *** NEW BEST: {metric:.10f} ***")

        elapsed = time.time() - t0
        print(f"  Config {ci+1}/{len(configs)}: best={best_metric:.10f}, time={elapsed:.0f}s")

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
