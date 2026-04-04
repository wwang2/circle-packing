"""
Fast multi-strategy optimizer for n=30 circle packing in [0,1]^2.
Two-phase approach: penalty-based pre-filter + SLSQP for promising candidates.
"""

import json
import math
import numpy as np
from scipy.optimize import minimize
from pathlib import Path
import time

N = 30
DIR = Path(__file__).parent
BEST_FILE = DIR / "solution_n30.json"

def load_solution(path):
    with open(path) as f:
        data = json.load(f)
    return np.array(data.get("circles", data), dtype=np.float64)

def save_solution(circles, path):
    data = {"circles": [[float(x), float(y), float(r)] for x, y, r in circles]}
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def sum_radii(c): return np.sum(c[:, 2])

def is_feasible(c, tol=1e-9):
    x, y, r = c[:, 0], c[:, 1], c[:, 2]
    if np.any(r <= 0) or np.any(r - x > tol) or np.any(x + r - 1 > tol): return False
    if np.any(r - y > tol) or np.any(y + r - 1 > tol): return False
    for i in range(len(c)):
        for j in range(i+1, len(c)):
            dx = x[i]-x[j]; dy = y[i]-y[j]
            if dx*dx + dy*dy < (r[i]+r[j]-tol)**2: return False
    return True

def penalty_obj_grad(v, penalty):
    c = v.reshape(-1, 3)
    n = len(c); x, y, r = c[:,0], c[:,1], c[:,2]
    obj = -np.sum(r)
    grad = np.zeros((n, 3))
    grad[:, 2] = -1.0
    pen = 0.0
    vl = np.maximum(0, r-x); vr = np.maximum(0, x+r-1)
    vb = np.maximum(0, r-y); vt = np.maximum(0, y+r-1)
    pen += np.sum(vl**2 + vr**2 + vb**2 + vt**2)
    grad[:,0] += penalty*2*(-vl+vr); grad[:,1] += penalty*2*(-vb+vt)
    grad[:,2] += penalty*2*(vl+vr+vb+vt)
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i]-x[j]; dy = y[i]-y[j]; dsq = dx*dx+dy*dy; md = r[i]+r[j]
            if dsq < md*md:
                d = math.sqrt(dsq)+1e-30; ov = md-d
                if ov > 0:
                    pen += ov*ov; f = penalty*2*ov
                    gx = -dx/d; gy = -dy/d
                    grad[i,0] += f*gx; grad[i,1] += f*gy
                    grad[j,0] -= f*gx; grad[j,1] -= f*gy
                    grad[i,2] += f; grad[j,2] += f
    return obj + penalty*pen, grad.flatten()

def fast_opt(circles, penalty=1e5, maxiter=100):
    v0 = circles.flatten()
    bounds = [(0.001,0.999) if i%3!=2 else (0.001,0.499) for i in range(len(v0))]
    res = minimize(lambda v: penalty_obj_grad(v, penalty), v0, method='L-BFGS-B',
                   jac=True, bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-15})
    return res.x.reshape(-1, 3)

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
    return (r, np.sum(r[:,2])) if is_feasible(r) else (circles, sum_radii(circles))

def basin_hopping(init, n_hops, step, seed, temp=0.5, threshold=0.0):
    """Basin-hopping: penalty pre-filter, SLSQP for promising."""
    rng = np.random.RandomState(seed)
    best = init.copy(); best_m = sum_radii(best)
    cur = best.copy(); cur_m = best_m
    slsqp_calls = 0

    for hop in range(n_hops):
        trial = cur.copy()
        p = rng.randint(0, 7)
        if p == 0:
            trial += rng.randn(N, 3) * step * 0.2
        elif p == 1:
            i, j = rng.choice(N, 2, replace=False); trial[[i,j]] = trial[[j,i]]
        elif p == 2:
            k = rng.randint(2, 8); idx = rng.choice(N, k, replace=False)
            trial[idx,:2] += rng.randn(k, 2)*step; trial[idx,2] *= 1+rng.randn(k)*0.15
        elif p == 3:
            i,j = rng.choice(N, 2, replace=False); d = rng.uniform(0.005, 0.04)
            trial[i,2] -= d; trial[j,2] += d
        elif p == 4:
            i = rng.randint(N)
            trial[i] = [rng.uniform(0.05,0.95), rng.uniform(0.05,0.95), rng.uniform(0.02,0.15)]
        elif p == 5:
            k = rng.randint(3, 6); idx = rng.choice(N, k, replace=False)
            for i in idx: trial[i] = [rng.uniform(0.05,0.95), rng.uniform(0.05,0.95), rng.uniform(0.02,0.13)]
        else:
            # Mirror a circle through center
            i = rng.randint(N)
            trial[i, 0] = 1 - trial[i, 0]; trial[i, 1] = 1 - trial[i, 1]

        trial[:,2] = np.clip(trial[:,2], 0.005, 0.49)
        trial[:,0] = np.clip(trial[:,0], trial[:,2]+0.001, 1-trial[:,2]-0.001)
        trial[:,1] = np.clip(trial[:,1], trial[:,2]+0.001, 1-trial[:,2]-0.001)

        # Quick penalty pre-filter
        opt = fast_opt(trial, penalty=1e5, maxiter=80)
        raw_m = np.sum(opt[:, 2])

        # Only SLSQP if raw metric is promising (within threshold of best)
        if raw_m < best_m - threshold:
            continue

        # SLSQP to get feasible
        ref, m = slsqp_refine(opt, maxiter=500)
        slsqp_calls += 1

        if not is_feasible(ref):
            continue

        de = m - cur_m
        if de > 0 or rng.random() < math.exp(min(de/temp, 0)):
            cur = ref.copy(); cur_m = m

        if m > best_m + 1e-12:
            best = ref.copy(); best_m = m
            print(f"  [BH hop={hop} s={seed}] NEW: {best_m:.10f} (slsqp_calls={slsqp_calls})", flush=True)

        if hop % 100 == 0:
            print(f"  [BH {hop}/{n_hops} s={seed}] best={best_m:.10f} slsqp={slsqp_calls}", flush=True)

    return best, best_m

def random_packing(rng):
    circles = []
    for _ in range(N):
        placed = False
        for _ in range(2000):
            r = rng.uniform(0.02, 0.13)
            x = rng.uniform(r+0.001, 1-r-0.001); y = rng.uniform(r+0.001, 1-r-0.001)
            if all(math.sqrt((x-cx)**2+(y-cy)**2) >= r+cr for cx,cy,cr in circles):
                circles.append((x,y,r)); placed = True; break
        if not placed:
            circles.append((rng.uniform(0.01,0.99), rng.uniform(0.01,0.99), 0.005))
    return np.array(circles[:N])

def multi_start(n_starts, seed_base):
    best_c = None; best_m = 0
    for i in range(n_starts):
        rng = np.random.RandomState(seed_base+i)
        init = random_packing(rng)
        opt = fast_opt(init, penalty=1e5, maxiter=150)
        raw_m = np.sum(opt[:,2])
        if raw_m > best_m - 0.05:
            ref, m = slsqp_refine(opt, maxiter=500)
            if is_feasible(ref) and m > best_m:
                best_m = m; best_c = ref.copy()
                print(f"  [MS {i}/{n_starts}] NEW: {best_m:.10f}", flush=True)
        if i % 50 == 0:
            print(f"  [MS {i}/{n_starts}] best={best_m:.10f}", flush=True)
    return best_c, best_m

def remove_reinsert(circles):
    best = circles.copy(); best_m = sum_radii(best)
    for idx in range(N):
        remaining = np.delete(circles, idx, axis=0)
        # Quick optimize 29
        opt29 = fast_opt(remaining, penalty=1e5, maxiter=150)
        rng = np.random.RandomState(idx*137)
        best_r = 0; best_ins = None
        for _ in range(5000):
            x = rng.uniform(0.01,0.99); y = rng.uniform(0.01,0.99)
            r_max = min(x, 1-x, y, 1-y)
            for c in opt29:
                d = math.sqrt((x-c[0])**2+(y-c[1])**2); r_max = min(r_max, d-c[2])
            if r_max > best_r: best_r = r_max; best_ins = [x, y, r_max]
        if best_ins is None or best_r <= 0.001: continue
        full = np.vstack([opt29, [best_ins]])
        ref, m = slsqp_refine(full, maxiter=1000)
        if is_feasible(ref) and m > best_m + 1e-12:
            best = ref.copy(); best_m = m
            print(f"  [R&R idx={idx}] NEW: {best_m:.10f}", flush=True)
        if idx % 10 == 0:
            print(f"  [R&R {idx}/30] best={best_m:.10f}", flush=True)
    return best, best_m

def update(c, m, gb, gm):
    if m > gm + 1e-12:
        print(f"\n*** IMPROVEMENT: {gm:.10f} -> {m:.10f} ***\n", flush=True)
        save_solution(c, BEST_FILE); save_solution(c, DIR/"best_n30.json")
        return c.copy(), m
    return gb, gm

def main():
    t0 = time.time()
    print(f"=== n={N} Optimizer ===", flush=True)
    init = load_solution(BEST_FILE)
    gb = init.copy(); gm = sum_radii(init)
    print(f"Start: {gm:.10f}", flush=True)

    # SLSQP refinement
    ref, rm = slsqp_refine(gb, maxiter=5000)
    gb, gm = update(ref, rm, gb, gm)
    print(f"SLSQP: {gm:.10f} ({time.time()-t0:.0f}s)", flush=True)

    # Basin-hopping with various seeds
    print("\n--- BH (step=0.15) ---", flush=True)
    for seed in [42, 137, 271, 314, 577, 719, 997, 1234, 1777, 2021]:
        bh, bm = basin_hopping(gb, 500, 0.15, seed, 0.5, threshold=0.03)
        if bm >= gm - 0.001:
            ref, rm = slsqp_refine(bh, maxiter=3000)
            if rm > bm: bm, bh = rm, ref
        gb, gm = update(bh, bm, gb, gm)
        print(f"  s={seed} best={gm:.10f} ({time.time()-t0:.0f}s)", flush=True)

    # BH large steps
    print("\n--- BH (step=0.25) ---", flush=True)
    for seed in [42, 137, 271, 314, 577]:
        bh, bm = basin_hopping(gb, 400, 0.25, seed, 1.0, threshold=0.05)
        if bm >= gm - 0.001:
            ref, rm = slsqp_refine(bh, maxiter=3000)
            if rm > bm: bm, bh = rm, ref
        gb, gm = update(bh, bm, gb, gm)
        print(f"  s={seed} best={gm:.10f} ({time.time()-t0:.0f}s)", flush=True)

    # R&R
    print("\n--- R&R ---", flush=True)
    rr, rrm = remove_reinsert(gb)
    gb, gm = update(rr, rrm, gb, gm)
    print(f"  R&R best={gm:.10f} ({time.time()-t0:.0f}s)", flush=True)

    # Multi-start
    print("\n--- Multi-start ---", flush=True)
    ms, msm = multi_start(200, 5000)
    if ms is not None: gb, gm = update(ms, msm, gb, gm)
    print(f"  MS best={gm:.10f} ({time.time()-t0:.0f}s)", flush=True)

    # Final aggressive BH
    print("\n--- Final BH ---", flush=True)
    for seed in range(3000, 3015):
        bh, bm = basin_hopping(gb, 300, 0.2, seed, 2.0, threshold=0.05)
        if bm >= gm - 0.001:
            ref, rm = slsqp_refine(bh, maxiter=3000)
            if rm > bm: bm, bh = rm, ref
        gb, gm = update(bh, bm, gb, gm)

    print(f"\n=== FINAL: {gm:.10f} ({time.time()-t0:.0f}s) ===", flush=True)
    save_solution(gb, BEST_FILE)

if __name__ == "__main__":
    main()
