"""
Radical topology search for n=30 circle packing.
Instead of perturbing existing solution, generates completely different
configurations and optimizes them. Tries structured layouts.
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
RAD_BEST = DIR / "radical_best_n30.json"

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

def slsqp_refine(circles, maxiter=3000):
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

def hex_grid(n, jitter=0.0, rng=None):
    """Generate hex grid packing with n circles."""
    # Estimate grid size
    k = int(math.ceil(math.sqrt(n)))
    circles = []
    r_est = 0.5 / (k + 0.5)
    for row in range(k + 2):
        for col in range(k + 2):
            x = r_est + col * 2 * r_est
            y = r_est + row * 2 * r_est * math.sqrt(3)/2
            if row % 2 == 1:
                x += r_est
            if 0 < x < 1 and 0 < y < 1:
                circles.append([x, y, r_est * 0.95])
    # Take closest n to center
    circles = np.array(circles)
    if len(circles) > n:
        dists = (circles[:, 0] - 0.5)**2 + (circles[:, 1] - 0.5)**2
        idx = np.argsort(dists)[:n]
        circles = circles[idx]
    elif len(circles) < n:
        # Pad with tiny circles
        if rng is None: rng = np.random.RandomState(0)
        while len(circles) < n:
            x, y = rng.uniform(0.02, 0.98, 2)
            circles = np.vstack([circles, [[x, y, 0.01]]])
    if jitter > 0 and rng is not None:
        circles[:, :2] += rng.randn(n, 2) * jitter
        circles[:, 2] *= (1 + rng.randn(n) * jitter * 0.5)
    circles[:, 2] = np.clip(circles[:, 2], 0.005, 0.49)
    circles[:, 0] = np.clip(circles[:, 0], circles[:, 2]+0.001, 1-circles[:, 2]-0.001)
    circles[:, 1] = np.clip(circles[:, 1], circles[:, 2]+0.001, 1-circles[:, 2]-0.001)
    return circles

def k_big_rest_small(k, rng):
    """k big circles + (N-k) small ones filling gaps."""
    circles = []
    # Place k big circles
    big_r = 0.5 / (math.sqrt(k) + 0.5)
    for _ in range(k):
        for _ in range(1000):
            r = rng.uniform(big_r * 0.7, big_r * 1.3)
            x = rng.uniform(r+0.001, 1-r-0.001)
            y = rng.uniform(r+0.001, 1-r-0.001)
            if all(math.sqrt((x-cx)**2+(y-cy)**2) >= r+cr for cx,cy,cr in circles):
                circles.append((x, y, r)); break
        else:
            circles.append((rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9), 0.01))
    # Fill remaining with largest-fit
    for _ in range(N - k):
        best_r = 0; best_pos = None
        for _ in range(5000):
            x = rng.uniform(0.01, 0.99); y = rng.uniform(0.01, 0.99)
            r_max = min(x, 1-x, y, 1-y)
            for cx, cy, cr in circles:
                d = math.sqrt((x-cx)**2+(y-cy)**2)
                r_max = min(r_max, d - cr)
            if r_max > best_r:
                best_r = r_max; best_pos = (x, y, r_max)
        if best_pos and best_r > 0.001:
            circles.append(best_pos)
        else:
            circles.append((rng.uniform(0.01, 0.99), rng.uniform(0.01, 0.99), 0.005))
    return np.array(circles[:N])

def greedy_largest_first(rng):
    """Greedily place circles, always finding the largest available gap."""
    circles = []
    for i in range(N):
        best_r = 0; best_pos = None
        n_tries = 10000 if i < 10 else 5000
        for _ in range(n_tries):
            x = rng.uniform(0.01, 0.99); y = rng.uniform(0.01, 0.99)
            r_max = min(x, 1-x, y, 1-y)
            for cx, cy, cr in circles:
                d = math.sqrt((x-cx)**2+(y-cy)**2)
                r_max = min(r_max, d - cr)
            if r_max > best_r:
                best_r = r_max; best_pos = (x, y, r_max)
        if best_pos and best_r > 0.001:
            circles.append(best_pos)
        else:
            circles.append((rng.uniform(0.01, 0.99), rng.uniform(0.01, 0.99), 0.005))
    return np.array(circles)

def random_uniform(rng):
    """Uniform random radii."""
    circles = []
    for _ in range(N):
        for _ in range(2000):
            r = rng.uniform(0.02, 0.13)
            x = rng.uniform(r+0.001, 1-r-0.001)
            y = rng.uniform(r+0.001, 1-r-0.001)
            if all(math.sqrt((x-cx)**2+(y-cy)**2) >= r+cr for cx,cy,cr in circles):
                circles.append((x,y,r)); break
        else:
            circles.append((rng.uniform(0.01, 0.99), rng.uniform(0.01, 0.99), 0.005))
    return np.array(circles[:N])

def main():
    t0 = time.time()
    gb = None; gm = 0

    # Try to load current best for comparison
    try:
        from optimize import load_solution
        ref = load_solution(BEST_FILE)
        gm = np.sum(ref[:, 2])
        gb = ref.copy()
    except:
        pass

    print(f"=== Radical Search n={N}, current best={gm:.10f} ===", flush=True)

    trial_num = 0

    # 1) Hex grid with various jitters
    print("\n--- Hex grids ---", flush=True)
    for seed in range(50):
        rng = np.random.RandomState(seed)
        init = hex_grid(N, jitter=0.02 * (seed % 10), rng=rng)
        ref, m = slsqp_refine(init, maxiter=3000)
        trial_num += 1
        if m > gm + 1e-12:
            gm = m; gb = ref.copy()
            save_solution(gb, RAD_BEST)
            print(f"  [{trial_num}] HEX seed={seed} NEW BEST: {gm:.10f}", flush=True)
        if seed % 10 == 0:
            print(f"  [{trial_num}] hex seed={seed} m={m:.10f} best={gm:.10f} ({time.time()-t0:.0f}s)", flush=True)

    # 2) K big + rest small
    print("\n--- K big + rest small ---", flush=True)
    for k in range(2, 12):
        for seed in range(20):
            rng = np.random.RandomState(k * 100 + seed)
            init = k_big_rest_small(k, rng)
            ref, m = slsqp_refine(init, maxiter=3000)
            trial_num += 1
            if m > gm + 1e-12:
                gm = m; gb = ref.copy()
                save_solution(gb, RAD_BEST)
                print(f"  [{trial_num}] K={k} seed={seed} NEW BEST: {gm:.10f}", flush=True)
        print(f"  K={k} done, best={gm:.10f} ({time.time()-t0:.0f}s)", flush=True)

    # 3) Greedy largest-first
    print("\n--- Greedy ---", flush=True)
    for seed in range(100):
        rng = np.random.RandomState(10000 + seed)
        init = greedy_largest_first(rng)
        ref, m = slsqp_refine(init, maxiter=3000)
        trial_num += 1
        if m > gm + 1e-12:
            gm = m; gb = ref.copy()
            save_solution(gb, RAD_BEST)
            print(f"  [{trial_num}] GREEDY seed={seed} NEW BEST: {gm:.10f}", flush=True)
        if seed % 20 == 0:
            print(f"  [{trial_num}] greedy seed={seed} m={m:.10f} best={gm:.10f} ({time.time()-t0:.0f}s)", flush=True)

    # 4) Random uniform
    print("\n--- Random ---", flush=True)
    for seed in range(200):
        rng = np.random.RandomState(20000 + seed)
        init = random_uniform(rng)
        ref, m = slsqp_refine(init, maxiter=3000)
        trial_num += 1
        if m > gm + 1e-12:
            gm = m; gb = ref.copy()
            save_solution(gb, RAD_BEST)
            print(f"  [{trial_num}] RANDOM seed={seed} NEW BEST: {gm:.10f}", flush=True)
        if seed % 50 == 0:
            print(f"  [{trial_num}] random seed={seed} m={m:.10f} best={gm:.10f} ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n=== RADICAL FINAL: {gm:.10f} ({time.time()-t0:.0f}s) ===", flush=True)
    if gb is not None:
        save_solution(gb, RAD_BEST)

if __name__ == "__main__":
    main()
