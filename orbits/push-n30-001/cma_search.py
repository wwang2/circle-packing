"""
CMA-ES search for n=30 circle packing.
Runs independently from main optimizer.
"""

import json
import math
import numpy as np
from scipy.optimize import minimize
from pathlib import Path
import time
import cma

N = 30
DIR = Path(__file__).parent
BEST_FILE = DIR / "solution_n30.json"
CMA_BEST = DIR / "cma_best_n30.json"

def load_solution(path):
    with open(path) as f:
        data = json.load(f)
    return np.array(data.get("circles", data), dtype=np.float64)

def save_solution(circles, path):
    data = {"circles": [[float(x), float(y), float(r)] for x, y, r in circles]}
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def is_feasible(c, tol=1e-9):
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
    pen = 0.0
    pen += np.sum(np.maximum(0, r-x)**2 + np.maximum(0, x+r-1)**2)
    pen += np.sum(np.maximum(0, r-y)**2 + np.maximum(0, y+r-1)**2)
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i]-x[j]; dy = y[i]-y[j]; dsq = dx*dx+dy*dy; md = r[i]+r[j]
            if dsq < md*md:
                ov = md - math.sqrt(dsq+1e-30)
                if ov > 0: pen += ov*ov
    return obj + penalty * pen

def slsqp_refine(circles, maxiter=2000):
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
    return (r, np.sum(r[:,2])) if is_feasible(r) else (circles, np.sum(circles[:,2]))

def main():
    t0 = time.time()
    init = load_solution(BEST_FILE)
    gb = init.copy(); gm = np.sum(init[:, 2])
    print(f"CMA-ES search, start={gm:.10f}", flush=True)

    configs = [
        (0.05, 80, 40000, 42),
        (0.1, 100, 40000, 137),
        (0.15, 120, 40000, 271),
        (0.2, 150, 40000, 314),
        (0.3, 200, 30000, 577),
        (0.1, 80, 40000, 719),
        (0.15, 100, 40000, 997),
        (0.2, 120, 40000, 1234),
        (0.05, 60, 50000, 1777),
        (0.1, 150, 40000, 2021),
    ]

    for sigma, popsize, maxevals, seed in configs:
        print(f"\n  CMA sigma={sigma} pop={popsize} seed={seed}", flush=True)
        v0 = gb.flatten()
        n = len(v0)
        opts = {
            'seed': seed, 'popsize': popsize, 'maxfevals': maxevals,
            'tolfun': 1e-14, 'tolx': 1e-14, 'verbose': -1,
            'bounds': [[0.001]*n, [0.999 if i%3!=2 else 0.499 for i in range(n)]],
        }

        best_m = gm; best_c = gb.copy()
        es = cma.CMAEvolutionStrategy(v0, sigma, opts)
        gen = 0
        while not es.stop():
            sols = es.ask()
            fits = [penalty_obj(s, 1e5) for s in sols]
            es.tell(sols, fits)

            # Check if any are feasible and better
            bi = np.argmin(fits)
            cand = sols[bi].reshape(-1, 3)
            if is_feasible(cand):
                m = np.sum(cand[:, 2])
                if m > best_m + 1e-12:
                    best_m = m; best_c = cand.copy()
                    print(f"    gen={gen} NEW: {best_m:.10f}", flush=True)

            gen += 1
            if gen % 200 == 0:
                print(f"    gen={gen} best={best_m:.10f}", flush=True)

        # SLSQP refine CMA result
        cma_res = es.result.xbest.reshape(-1, 3)
        ref, rm = slsqp_refine(cma_res, maxiter=3000)
        if rm > best_m:
            best_m = rm; best_c = ref
            print(f"    +SLSQP: {best_m:.10f}", flush=True)

        if best_m > gm + 1e-12:
            gm = best_m; gb = best_c.copy()
            save_solution(gb, CMA_BEST)
            print(f"  *** CMA IMPROVEMENT: {gm:.10f} ***", flush=True)

        print(f"  Done sigma={sigma} seed={seed} best={gm:.10f} ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n=== CMA FINAL: {gm:.10f} ({time.time()-t0:.0f}s) ===", flush=True)

if __name__ == "__main__":
    main()
