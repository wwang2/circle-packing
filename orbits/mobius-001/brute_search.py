#!/usr/bin/env python3
"""
Brute-force topology exploration with forced output flushing.

Strategy: Generate thousands of random configurations via fast penalty
optimization, track ALL distinct basins found, then deeply polish
the best ones.

Key difference from other scripts:
1. Use COBYLA as alternative optimizer (different convergence path)
2. Try trust-constr method
3. Force-flush all output
4. Explicit topology tracking via sorted-radii fingerprint
"""

import json
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution
from pathlib import Path
import time
import sys

SEED = 99999
N = 26
WORKTREE = Path("/Users/wujiewang/code/circle-packing/.worktrees/mobius-001")
OUTPUT_DIR = WORKTREE / "orbits/mobius-001"


def flush_print(*args, **kwargs):
    print(*args, **kwargs, flush=True)


def load_solution(path):
    with open(path) as f:
        return np.array(json.load(f)["circles"])


def save_solution(circles, path):
    data = {"circles": [[float(c[0]), float(c[1]), float(c[2])] for c in circles]}
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def sum_radii(c):
    return float(np.sum(c[:, 2]))


def validate(circles, tol=1e-10):
    n = len(circles)
    mv = 0.0
    for i in range(n):
        x, y, r = circles[i]
        if r <= 0: return False, abs(r)
        mv = max(mv, r - x, x + r - 1, r - y, y + r - 1)
    for i in range(n):
        for j in range(i + 1, n):
            dx = circles[i, 0] - circles[j, 0]
            dy = circles[i, 1] - circles[j, 1]
            d = math.sqrt(dx * dx + dy * dy)
            mv = max(mv, circles[i, 2] + circles[j, 2] - d)
    return mv <= tol, mv


def fast_penalty(circles, stages=None):
    """Fast L-BFGS-B penalty optimization."""
    if stages is None:
        stages = [1, 100, 10000, 1e6, 1e8]
    n = len(circles)
    x = circles.flatten().copy()
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-5, 0.5)] * n

    for pw in stages:
        def obj_grad(x, pw=pw):
            xs = x[0::3]; ys = x[1::3]; rs = x[2::3]
            obj = -np.sum(rs)
            grad = np.zeros_like(x)
            grad[2::3] = -1.0
            vl = np.maximum(0, rs - xs); vr = np.maximum(0, xs + rs - 1)
            vb = np.maximum(0, rs - ys); vt = np.maximum(0, ys + rs - 1)
            obj += pw * np.sum(vl**2 + vr**2 + vb**2 + vt**2)
            grad[0::3] += pw * (-2*vl + 2*vr)
            grad[1::3] += pw * (-2*vb + 2*vt)
            grad[2::3] += pw * (2*vl + 2*vr + 2*vb + 2*vt)
            for i in range(n):
                js = np.arange(i+1, n)
                if len(js) == 0: continue
                dx = xs[i]-xs[js]; dy = ys[i]-ys[js]
                dsq = dx*dx+dy*dy
                md = rs[i]+rs[js]; mdsq = md**2
                active = mdsq > dsq
                if not np.any(active): continue
                a_js = js[active]
                factor = mdsq[active]-dsq[active]
                obj += pw*np.sum(factor**2)
                f2 = 2*factor
                a_dx=dx[active]; a_dy=dy[active]; a_md=md[active]
                grad[3*i] += pw*np.sum(f2*(-2*a_dx))
                grad[3*i+1] += pw*np.sum(f2*(-2*a_dy))
                for k, j in enumerate(a_js):
                    grad[3*j] += pw*f2[k]*2*a_dx[k]
                    grad[3*j+1] += pw*f2[k]*2*a_dy[k]
                    grad[3*i+2] += pw*f2[k]*2*a_md[k]
                    grad[3*j+2] += pw*f2[k]*2*a_md[k]
            return obj, grad
        result = minimize(obj_grad, x, jac=True, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 1000, 'ftol': 1e-15})
        x = result.x.copy()
    out = x.reshape(n, 3)
    valid, viol = validate(out, tol=1e-8)
    return out, sum_radii(out) if valid else 0.0, valid


def slsqp_polish(circles, maxiter=10000):
    n = len(circles)
    x0 = circles.flatten()
    nn = 3*n
    def objective(x): return -np.sum(x[2::3])
    def grad_obj(x):
        g = np.zeros(nn); g[2::3] = -1.0; return g
    def all_con(x):
        xs=x[0::3]; ys=x[1::3]; rs=x[2::3]
        vals = []
        vals.extend(xs-rs); vals.extend(1.0-xs-rs)
        vals.extend(ys-rs); vals.extend(1.0-ys-rs)
        vals.extend(rs-1e-6)
        for i in range(n):
            dx=xs[i]-xs[i+1:]; dy=ys[i]-ys[i+1:]
            vals.extend(dx*dx+dy*dy-(rs[i]+rs[i+1:])**2)
        return np.array(vals)
    bounds = [(0.0,1.0),(0.0,1.0),(1e-6,0.5)]*n
    result = minimize(objective, x0, method='SLSQP', jac=grad_obj,
                     bounds=bounds, constraints=[{'type':'ineq','fun':all_con}],
                     options={'maxiter':maxiter,'ftol':1e-15})
    out = result.x.reshape(n, 3)
    valid, _ = validate(out)
    return out, -result.fun if valid else 0.0, valid


def radii_fingerprint(circles):
    """Fingerprint based on sorted radii (4 decimal places)."""
    radii = sorted(circles[:, 2], reverse=True)
    return tuple(round(r, 4) for r in radii)


def generate_random_init(rng, method='greedy'):
    """Generate diverse random initialization."""
    if method == 'greedy':
        radii = rng.uniform(0.04, 0.14, N)
        radii = np.sort(radii)[::-1]
        circles = np.zeros((N, 3))
        for i in range(N):
            r = radii[i]
            best = None
            for _ in range(150):
                x = rng.uniform(r+0.001, 1-r-0.001)
                y = rng.uniform(r+0.001, 1-r-0.001)
                ok = True
                for j in range(i):
                    dx = x-circles[j,0]; dy = y-circles[j,1]
                    if math.sqrt(dx*dx+dy*dy) < r+circles[j,2]+0.001:
                        ok = False; break
                if ok:
                    best = (x, y); break
            if best is None:
                best = (rng.uniform(r, 1-r), rng.uniform(r, 1-r))
            circles[i] = [best[0], best[1], r]
        return circles

    elif method == 'uniform':
        r = rng.uniform(0.06, 0.10)
        circles = np.zeros((N, 3))
        for i in range(N):
            for _ in range(200):
                x = rng.uniform(r+0.001, 1-r-0.001)
                y = rng.uniform(r+0.001, 1-r-0.001)
                ok = True
                for j in range(i):
                    dx = x-circles[j,0]; dy = y-circles[j,1]
                    if math.sqrt(dx*dx+dy*dy) < 2*r+0.001:
                        ok = False; break
                if ok:
                    circles[i] = [x, y, r]; break
            else:
                circles[i] = [rng.uniform(r, 1-r), rng.uniform(r, 1-r), r]
        return circles

    elif method == 'bimodal':
        n_big = rng.randint(2, 8)
        big_r = rng.uniform(0.10, 0.16)
        small_r = rng.uniform(0.04, 0.08)
        radii = np.concatenate([np.full(n_big, big_r), np.full(N-n_big, small_r)])
        radii += rng.uniform(-0.01, 0.01, N)
        radii = np.maximum(radii, 0.02)
        radii = np.sort(radii)[::-1]
        circles = np.zeros((N, 3))
        for i in range(N):
            r = radii[i]
            for _ in range(200):
                x = rng.uniform(r+0.001, 1-r-0.001)
                y = rng.uniform(r+0.001, 1-r-0.001)
                ok = True
                for j in range(i):
                    dx = x-circles[j,0]; dy = y-circles[j,1]
                    if math.sqrt(dx*dx+dy*dy) < r+circles[j,2]+0.001:
                        ok = False; break
                if ok:
                    circles[i] = [x, y, r]; break
            else:
                circles[i] = [rng.uniform(r, 1-r), rng.uniform(r, 1-r), r]
        return circles

    elif method == 'grid':
        # 5x5 + 1 or 6x4+2
        config = rng.choice(['5x5+1', '6x4+2', '4x6+2', '5x5+1_v2'])
        if config == '5x5+1':
            rows, cols = 5, 5
        elif config == '6x4+2':
            rows, cols = 6, 4
        elif config == '4x6+2':
            rows, cols = 4, 6
        else:
            rows, cols = 5, 5

        r = rng.uniform(0.07, 0.10)
        circles = []
        dy = 1.0 / (rows + 1)
        for row in range(rows):
            dx_grid = 1.0 / (cols + 1)
            y = dy * (row + 1)
            for col in range(cols):
                x = dx_grid * (col + 1)
                circles.append([x + rng.normal(0, 0.01),
                                y + rng.normal(0, 0.01),
                                r + rng.uniform(-0.01, 0.01)])
        # Fill remaining
        while len(circles) < N:
            circles.append([rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9),
                            rng.uniform(0.03, 0.06)])
        result = np.array(circles[:N])
        for i in range(N):
            result[i, 2] = max(result[i, 2], 0.02)
            result[i, 0] = max(result[i, 2], min(1-result[i, 2], result[i, 0]))
            result[i, 1] = max(result[i, 2], min(1-result[i, 2], result[i, 1]))
        return result


def main():
    base = load_solution(WORKTREE / "orbits/topo-001/solution_n26.json")
    try:
        current = load_solution(OUTPUT_DIR / "solution_n26.json")
        if sum_radii(current) > sum_radii(base):
            base = current
    except:
        pass
    base_metric = sum_radii(base)
    flush_print(f"Starting metric: {base_metric:.10f}")

    rng = np.random.RandomState(SEED)
    best = base_metric
    best_circles = base.copy()
    basins = {}  # fingerprint -> (metric, circles)
    methods = ['greedy', 'uniform', 'bimodal', 'grid']

    # Phase 1: Rapid screening (penalty only)
    flush_print("\n=== Phase 1: Rapid screening (1000 inits, penalty only) ===")
    t0 = time.time()

    for trial in range(1000):
        method = methods[trial % len(methods)]
        try:
            init = generate_random_init(rng, method)
            opt, metric, valid = fast_penalty(init)
            if valid and metric > 2.0:
                fp = radii_fingerprint(opt)
                if fp not in basins or metric > basins[fp][0]:
                    basins[fp] = (metric, opt.copy())
        except:
            pass

        if (trial + 1) % 200 == 0:
            n_basins = len(basins)
            top5 = sorted([m for m, _ in basins.values()], reverse=True)[:5]
            flush_print(f"  Trial {trial+1}/1000: basins={n_basins}, "
                        f"top5={[f'{m:.6f}' for m in top5]}")

    flush_print(f"Phase 1: {time.time()-t0:.1f}s, {len(basins)} basins")

    # Phase 2: SLSQP polish on top basins
    flush_print(f"\n=== Phase 2: SLSQP polish on top 50 basins ===")
    t0 = time.time()

    top_basins = sorted(basins.items(), key=lambda x: -x[1][0])[:50]
    for bi, (fp, (metric, circles)) in enumerate(top_basins):
        try:
            polished, pm, pv = slsqp_polish(circles, maxiter=10000)
            if pv and pm > best + 1e-12:
                flush_print(f"  Basin {bi}: {metric:.6f} -> {pm:.10f} NEW BEST!")
                best = pm
                best_circles = polished.copy()
            elif pv and pm > metric:
                basins[fp] = (pm, polished.copy())
        except:
            pass

        if (bi + 1) % 10 == 0:
            flush_print(f"  Polished {bi+1}/50: best={best:.10f}")

    flush_print(f"Phase 2: {time.time()-t0:.1f}s, best={best:.10f}")

    # Phase 3: Deep SLSQP from diverse greedy starts
    flush_print(f"\n=== Phase 3: Full pipeline on 200 diverse starts ===")
    t0 = time.time()

    for trial in range(200):
        method = methods[trial % len(methods)]
        try:
            init = generate_random_init(rng, method)
            # Full pipeline: penalty then SLSQP
            pen, pen_m, pen_v = fast_penalty(init)
            if pen_v:
                pol, pol_m, pol_v = slsqp_polish(pen, maxiter=8000)
                if pol_v and pol_m > best + 1e-12:
                    flush_print(f"  Trial {trial}: {pol_m:.10f} NEW BEST!")
                    best = pol_m
                    best_circles = pol.copy()
        except:
            pass

        if (trial + 1) % 50 == 0:
            flush_print(f"  Trial {trial+1}/200: best={best:.10f}")

    flush_print(f"Phase 3: {time.time()-t0:.1f}s, best={best:.10f}")

    # Phase 4: Perturbations of the best known to try different SLSQP paths
    flush_print(f"\n=== Phase 4: Perturbation + SLSQP (200 trials) ===")
    t0 = time.time()

    for trial in range(200):
        new = best_circles.copy()
        # Random perturbation type
        ptype = trial % 5
        if ptype == 0:
            # Small position noise
            new[:, :2] += rng.normal(0, 0.005, (N, 2))
        elif ptype == 1:
            # Swap two circles
            i, j = rng.choice(N, 2, replace=False)
            new[i], new[j] = new[j].copy(), new[i].copy()
        elif ptype == 2:
            # Scale all radii slightly
            scale = 1 + rng.normal(0, 0.01)
            new[:, 2] *= scale
        elif ptype == 3:
            # Perturb radii individually
            new[:, 2] *= (1 + rng.normal(0, 0.02, N))
        else:
            # Move one circle to a new random position
            i = rng.randint(N)
            r = new[i, 2]
            new[i, 0] = rng.uniform(r, 1 - r)
            new[i, 1] = rng.uniform(r, 1 - r)

        for i in range(N):
            new[i, 2] = max(new[i, 2], 0.01)
            new[i, 0] = max(new[i, 2], min(1 - new[i, 2], new[i, 0]))
            new[i, 1] = max(new[i, 2], min(1 - new[i, 2], new[i, 1]))

        try:
            opt, metric, valid = slsqp_polish(new, maxiter=5000)
            if valid and metric > best + 1e-12:
                flush_print(f"  Trial {trial}: {best:.10f} -> {metric:.10f} NEW BEST!")
                best = metric
                best_circles = opt.copy()
        except:
            pass

        if (trial + 1) % 50 == 0:
            flush_print(f"  Trial {trial+1}/200: best={best:.10f}")

    flush_print(f"Phase 4: {time.time()-t0:.1f}s, best={best:.10f}")

    # Final output
    valid, viol = validate(best_circles)
    flush_print(f"\nFINAL: {best:.10f} (valid={valid}, viol={viol:.2e})")

    try:
        current = load_solution(OUTPUT_DIR / "solution_n26.json")
        current_m = sum_radii(current)
    except:
        current_m = 0

    if best > current_m:
        save_solution(best_circles, OUTPUT_DIR / "solution_n26.json")
        flush_print("Saved new best!")
    else:
        flush_print(f"No improvement over current ({current_m:.10f})")


if __name__ == "__main__":
    main()
