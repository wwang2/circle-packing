"""
Differential Evolution + aggressive multi-start for n=30.
Uses scipy.optimize.differential_evolution which is a good global optimizer.
Also tries lattice-based initializations with various symmetries.
"""

import json
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution
from pathlib import Path
import time

N = 30
DIR = Path(__file__).parent
BEST_FILE = DIR / "solution_n30.json"
DE_BEST = DIR / "de_best_n30.json"

def load_solution(path):
    with open(path) as f:
        data = json.load(f)
    return np.array(data.get("circles", data), dtype=np.float64)

def save_solution(circles, path):
    data = {"circles": [[float(x), float(y), float(r)] for x, y, r in circles]}
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def is_feasible_strict(c, tol=1e-10):
    x, y, r = c[:, 0], c[:, 1], c[:, 2]
    if np.any(r <= 0) or np.any(r - x > tol) or np.any(x + r - 1 > tol): return False
    if np.any(r - y > tol) or np.any(y + r - 1 > tol): return False
    for i in range(len(c)):
        for j in range(i+1, len(c)):
            dx = x[i]-x[j]; dy = y[i]-y[j]
            if dx*dx + dy*dy < (r[i]+r[j]-tol)**2: return False
    return True

def penalty_obj(v, penalty=1e5):
    c = v.reshape(-1, 3); n = len(c)
    x, y, r = c[:,0], c[:,1], c[:,2]
    obj = -np.sum(r)
    pen = np.sum(np.maximum(0, r-x)**2 + np.maximum(0, x+r-1)**2 +
                 np.maximum(0, r-y)**2 + np.maximum(0, y+r-1)**2 +
                 np.maximum(0, -r)**2)
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i]-x[j]; dy = y[i]-y[j]; dsq = dx*dx+dy*dy
            md = r[i]+r[j]
            if dsq < md*md:
                ov = md - math.sqrt(dsq+1e-30)
                if ov > 0: pen += ov*ov
    return obj + penalty * pen

def slsqp_refine_strict(circles, maxiter=5000):
    """SLSQP refine, then shrink to ensure strict feasibility."""
    n = len(circles); v0 = circles.flatten()
    bounds = [(0.001,0.999) if i%3!=2 else (0.001,0.499) for i in range(len(v0))]
    constraints = []
    for i in range(n):
        ix,iy,ir = 3*i,3*i+1,3*i+2
        constraints.append({'type':'ineq','fun':lambda v,a=ix,b=ir: v[a]-v[b]})
        constraints.append({'type':'ineq','fun':lambda v,a=ix,b=ir: 1-v[a]-v[b]})
        constraints.append({'type':'ineq','fun':lambda v,a=iy,b=ir: v[a]-v[b]})
        constraints.append({'type':'ineq','fun':lambda v,a=iy,b=ir: 1-v[a]-v[b]})
        constraints.append({'type':'ineq','fun':lambda v,b=ir: v[b]-1e-6})
    for i in range(n):
        for j in range(i+1, n):
            ix,iy,ir = 3*i,3*i+1,3*i+2; jx,jy,jr = 3*j,3*j+1,3*j+2
            constraints.append({'type':'ineq','fun':lambda v,a=ix,b=iy,c=ir,d=jx,e=jy,f=jr:
                math.sqrt((v[a]-v[d])**2+(v[b]-v[e])**2)-v[c]-v[f]})
    res = minimize(lambda v: -np.sum(v[2::3]), v0, method='SLSQP',
                   bounds=bounds, constraints=constraints, options={'maxiter':maxiter,'ftol':1e-15})
    r = res.x.reshape(-1, 3)

    if is_feasible_strict(r):
        return r, np.sum(r[:,2])

    # Shrink radii slightly and re-refine
    for factor in [1-2e-9, 1-5e-9, 1-1e-8, 1-2e-8, 1-5e-8]:
        c2 = r.copy(); c2[:,2] *= factor
        res2 = minimize(lambda v: -np.sum(v[2::3]), c2.flatten(), method='SLSQP',
                       bounds=bounds, constraints=constraints, options={'maxiter':maxiter,'ftol':1e-15})
        r2 = res2.x.reshape(-1, 3)
        if is_feasible_strict(r2):
            return r2, np.sum(r2[:,2])

    return circles, np.sum(circles[:,2])

def de_optimize(seed, maxiter=200, popsize=50):
    """Differential evolution."""
    n = N * 3
    bounds = []
    for i in range(N):
        bounds.extend([(0.01, 0.99), (0.01, 0.99), (0.01, 0.2)])

    result = differential_evolution(
        penalty_obj, bounds, seed=seed,
        maxiter=maxiter, popsize=popsize,
        mutation=(0.5, 1.5), recombination=0.9,
        tol=1e-14, atol=1e-14,
        args=(1e5,)
    )
    return result.x.reshape(-1, 3)

def grid_init(rows, cols, rng):
    """Grid-based initialization."""
    circles = []
    r = 0.5 / max(rows, cols)
    for i in range(rows):
        for j in range(cols):
            x = (j + 0.5) / cols
            y = (i + 0.5) / rows
            circles.append([x, y, r * 0.9])
            if len(circles) >= N:
                break
        if len(circles) >= N:
            break
    while len(circles) < N:
        circles.append([rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95), 0.02])
    c = np.array(circles[:N])
    c[:, :2] += rng.randn(N, 2) * 0.01
    c[:, 2] = np.clip(c[:, 2], 0.005, 0.49)
    c[:, 0] = np.clip(c[:, 0], c[:, 2]+0.001, 1-c[:, 2]-0.001)
    c[:, 1] = np.clip(c[:, 1], c[:, 2]+0.001, 1-c[:, 2]-0.001)
    return c

def main():
    t0 = time.time()
    try:
        gb = load_solution(BEST_FILE); gm = np.sum(gb[:,2])
    except:
        gb = None; gm = 0
    print(f"=== DE Search n={N}, best={gm:.10f} ===", flush=True)

    # 1) DE with various seeds
    print("\n--- Differential Evolution ---", flush=True)
    for seed in range(20):
        print(f"  DE seed={seed}...", flush=True)
        c = de_optimize(seed, maxiter=300, popsize=40)
        ref, m = slsqp_refine_strict(c)
        if m > gm + 1e-12:
            gm = m; gb = ref.copy()
            save_solution(gb, DE_BEST)
            print(f"  DE seed={seed} NEW BEST: {gm:.10f}", flush=True)
        print(f"  DE seed={seed} m={m:.10f} best={gm:.10f} ({time.time()-t0:.0f}s)", flush=True)

    # 2) Grid inits with SLSQP
    print("\n--- Grid initializations ---", flush=True)
    grid_configs = [(5,6), (6,5), (4,8), (8,4), (3,10), (10,3), (5,7), (7,5), (6,6)]
    for rows, cols in grid_configs:
        for seed in range(10):
            rng = np.random.RandomState(rows*100+cols*10+seed)
            c = grid_init(rows, cols, rng)
            ref, m = slsqp_refine_strict(c)
            if m > gm + 1e-12:
                gm = m; gb = ref.copy()
                save_solution(gb, DE_BEST)
                print(f"  Grid {rows}x{cols} seed={seed} NEW BEST: {gm:.10f}", flush=True)
        print(f"  Grid {rows}x{cols} done, best={gm:.10f} ({time.time()-t0:.0f}s)", flush=True)

    # 3) Quasi-random (Halton) starts
    print("\n--- Halton starts ---", flush=True)
    for trial in range(100):
        rng = np.random.RandomState(50000 + trial)
        # Halton-like: use different bases for x, y
        c = []
        for i in range(N):
            # Van der Corput sequences
            x = 0; y = 0; r_init = rng.uniform(0.03, 0.12)
            f = 0.5; n = i + trial * N + 1
            while n > 0:
                x += f * (n % 2); n //= 2; f *= 0.5
            f = 1/3; n = i + trial * N + 1
            while n > 0:
                y += f * (n % 3); n //= 3; f /= 3
            x = np.clip(x, r_init + 0.001, 1 - r_init - 0.001)
            y = np.clip(y, r_init + 0.001, 1 - r_init - 0.001)
            c.append([x, y, r_init])
        c = np.array(c)
        ref, m = slsqp_refine_strict(c)
        if m > gm + 1e-12:
            gm = m; gb = ref.copy()
            save_solution(gb, DE_BEST)
            print(f"  Halton {trial} NEW BEST: {gm:.10f}", flush=True)
        if trial % 20 == 0:
            print(f"  Halton {trial} m={m:.10f} best={gm:.10f} ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n=== DE FINAL: {gm:.10f} ({time.time()-t0:.0f}s) ===", flush=True)

if __name__ == "__main__":
    main()
