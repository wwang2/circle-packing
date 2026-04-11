"""
Fast n=32 optimizer with vectorized constraints and flushed output.
Focus on what works: many quick SLSQP restarts from small perturbations.
"""

import json
import math
import numpy as np
from scipy.optimize import minimize
from pathlib import Path
import time
import sys

N = 32
BEST_FILE = Path(__file__).parent / "solution_n32.json"
LOG_FILE = Path(__file__).parent / "log.md"

def log(msg):
    print(msg, flush=True)

def load_solution(path):
    with open(path) as f:
        data = json.load(f)
    circles = data.get("circles", data)
    return np.array(circles)

def save_solution(circles, path):
    data = {"circles": [[float(x), float(y), float(r)] for x, y, r in circles]}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def neg_sum_radii(x):
    return -np.sum(x[2::3])

def neg_sum_radii_jac(x):
    j = np.zeros(len(x))
    j[2::3] = -1.0
    return j

def all_ineq(x):
    """Vectorized: all inequality constraints >= 0."""
    c = x.reshape(-1, 3)
    n = len(c)
    xs, ys, rs = c[:,0], c[:,1], c[:,2]

    # Containment: 4n constraints
    cont = np.concatenate([xs - rs, 1.0 - xs - rs, ys - rs, 1.0 - ys - rs, rs - 1e-6])

    # Non-overlap: n*(n-1)/2 constraints
    overlap = np.empty(n*(n-1)//2)
    k = 0
    for i in range(n):
        for j in range(i+1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            overlap[k] = math.sqrt(dx*dx + dy*dy) - rs[i] - rs[j]
            k += 1

    return np.concatenate([cont, overlap])

def all_ineq_jac(x):
    """Analytic Jacobian of all_ineq."""
    c = x.reshape(-1, 3)
    n = len(c)
    xs, ys, rs = c[:,0], c[:,1], c[:,2]

    n_cont = 5 * n
    n_overlap = n * (n-1) // 2
    total = n_cont + n_overlap
    dim = 3 * n

    jac = np.zeros((total, dim))

    # x - r >= 0
    for i in range(n):
        jac[i, 3*i] = 1.0
        jac[i, 3*i+2] = -1.0

    # 1 - x - r >= 0
    for i in range(n):
        jac[n+i, 3*i] = -1.0
        jac[n+i, 3*i+2] = -1.0

    # y - r >= 0
    for i in range(n):
        jac[2*n+i, 3*i+1] = 1.0
        jac[2*n+i, 3*i+2] = -1.0

    # 1 - y - r >= 0
    for i in range(n):
        jac[3*n+i, 3*i+1] = -1.0
        jac[3*n+i, 3*i+2] = -1.0

    # r - eps >= 0
    for i in range(n):
        jac[4*n+i, 3*i+2] = 1.0

    # Non-overlap
    row = 5 * n
    for i in range(n):
        for j in range(i+1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist > 1e-15:
                jac[row, 3*i] = dx / dist
                jac[row, 3*i+1] = dy / dist
                jac[row, 3*j] = -dx / dist
                jac[row, 3*j+1] = -dy / dist
            jac[row, 3*i+2] = -1.0
            jac[row, 3*j+2] = -1.0
            row += 1

    return jac

def make_bounds():
    bounds = []
    for _ in range(N):
        bounds.append((1e-4, 1.0 - 1e-4))
        bounds.append((1e-4, 1.0 - 1e-4))
        bounds.append((1e-6, 0.5))
    return bounds

def is_valid(x, tol=1e-10):
    return np.all(all_ineq(x) >= -tol)

def get_metric(x):
    return np.sum(x[2::3])

def slsqp_opt(x0, maxiter=5000):
    constraints = [{"type": "ineq", "fun": all_ineq, "jac": all_ineq_jac}]
    bounds = make_bounds()
    result = minimize(
        neg_sum_radii, x0, method='SLSQP', jac=neg_sum_radii_jac,
        bounds=bounds, constraints=constraints,
        options={'maxiter': maxiter, 'ftol': 1e-16, 'disp': False}
    )
    return result.x, -result.fun

def perturb(x, rng, scale=0.01, mode='all'):
    x_new = x.copy()
    c = x_new.reshape(-1, 3)
    n = len(c)

    if mode == 'all':
        c[:, 0] += rng.normal(0, scale, n)
        c[:, 1] += rng.normal(0, scale, n)
    elif mode == 'single':
        i = rng.integers(n)
        c[i, 0] += rng.normal(0, scale * 3)
        c[i, 1] += rng.normal(0, scale * 3)
    elif mode == 'smallest':
        idx = np.argsort(c[:, 2])[:5]
        c[idx, 0] += rng.normal(0, scale * 2, len(idx))
        c[idx, 1] += rng.normal(0, scale * 2, len(idx))
    elif mode == 'cluster':
        pivot = rng.integers(n)
        dists = np.sqrt((c[:,0]-c[pivot,0])**2 + (c[:,1]-c[pivot,1])**2)
        neighbors = np.argsort(dists)[:5]
        c[neighbors, 0] += rng.normal(0, scale * 2, len(neighbors))
        c[neighbors, 1] += rng.normal(0, scale * 2, len(neighbors))
    elif mode == 'radii':
        c[:, 2] *= (1 + rng.normal(0, scale, n))
        c[:, 2] = np.maximum(c[:, 2], 0.005)

    # Clamp
    for i in range(n):
        r = max(0.001, c[i, 2])
        c[i, 2] = r
        c[i, 0] = np.clip(c[i, 0], r + 1e-4, 1 - r - 1e-4)
        c[i, 1] = np.clip(c[i, 1], r + 1e-4, 1 - r - 1e-4)

    return x_new

def run():
    best_circles = load_solution(BEST_FILE)
    best_x = best_circles.flatten()
    best_metric = get_metric(best_x)
    log(f"Starting metric: {best_metric:.10f}")

    improvements = []
    start_time = time.time()

    def try_update(x, source):
        nonlocal best_x, best_metric
        if is_valid(x):
            m = get_metric(x)
            if m > best_metric + 1e-12:
                imp = m - best_metric
                best_metric = m
                best_x = x.copy()
                save_solution(x.reshape(-1, 3), BEST_FILE)
                improvements.append((source, m))
                log(f"  *** NEW BEST: {m:.10f} (+{imp:.2e}) [{source}]")
                return True
        return False

    # Phase 1: Re-optimize with vectorized analytic Jacobians
    log("\n=== Phase 1: Re-optimize current best ===")
    x_opt, m = slsqp_opt(best_x, maxiter=10000)
    try_update(x_opt, "reopt")
    log(f"  After reopt: {best_metric:.10f}")

    # Phase 2: Massive perturbation campaign
    log("\n=== Phase 2: Perturbation campaign (2000 trials) ===")
    modes = ['all', 'single', 'smallest', 'cluster', 'radii']
    scales = [0.001, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05]

    for trial in range(2000):
        if trial % 200 == 0:
            elapsed = time.time() - start_time
            log(f"  Trial {trial}/2000, best={best_metric:.10f}, time={elapsed:.0f}s")

        rng = np.random.default_rng(trial * 17 + 42)
        mode = modes[trial % len(modes)]
        scale = scales[trial % len(scales)]

        x0 = perturb(best_x, rng, scale=scale, mode=mode)
        x_opt, m = slsqp_opt(x0, maxiter=3000)
        try_update(x_opt, f"perturb-{mode}-s{scale}-t{trial}")

    log(f"\nAfter Phase 2: best = {best_metric:.10f}")

    # Phase 3: Remove-reinsert
    log("\n=== Phase 3: Remove-reinsert ===")
    for trial in range(50):
        if trial % 10 == 0:
            log(f"  Remove-reinsert trial {trial}/50")

        rng = np.random.default_rng(trial * 31 + 9000)
        c = best_x.reshape(-1, 3).copy()
        n = len(c)

        # Remove 1-3 circles
        num_remove = rng.integers(1, 4)
        if rng.random() < 0.5:
            remove_idx = np.argsort(c[:, 2])[:num_remove]  # smallest
        else:
            remove_idx = rng.choice(n, size=num_remove, replace=False)

        keep_mask = np.ones(n, dtype=bool)
        keep_mask[remove_idx] = False
        kept = c[keep_mask]

        # Optimize kept circles
        x_kept = kept.flatten()
        n_kept = len(kept)

        def neg_sr_k(xx):
            return -np.sum(xx[2::3])
        def neg_sr_k_jac(xx):
            j = np.zeros(len(xx))
            j[2::3] = -1.0
            return j

        def all_ineq_k(xx):
            cc = xx.reshape(-1, 3)
            nk = len(cc)
            xs, ys, rs = cc[:,0], cc[:,1], cc[:,2]
            cont = np.concatenate([xs-rs, 1-xs-rs, ys-rs, 1-ys-rs, rs-1e-6])
            ov = []
            for i in range(nk):
                for j in range(i+1, nk):
                    dx = xs[i]-xs[j]; dy = ys[i]-ys[j]
                    ov.append(math.sqrt(dx*dx+dy*dy) - rs[i] - rs[j])
            return np.concatenate([cont, np.array(ov)])

        bounds_k = []
        for _ in range(n_kept):
            bounds_k.extend([(1e-4, 0.9999), (1e-4, 0.9999), (1e-6, 0.5)])

        res = minimize(neg_sr_k, x_kept, method='SLSQP', jac=neg_sr_k_jac,
                       bounds=bounds_k,
                       constraints=[{"type":"ineq","fun":all_ineq_k}],
                       options={'maxiter':2000, 'ftol':1e-16})
        kept = res.x.reshape(-1, 3)

        # Greedily reinsert
        all_c = list(kept)
        for _ in range(num_remove):
            best_r = 0
            best_pos = None
            for _ in range(5000):
                cx = rng.uniform(0.01, 0.99)
                cy = rng.uniform(0.01, 0.99)
                max_r = min(cx, 1-cx, cy, 1-cy)
                for (ox, oy, orr) in all_c:
                    d = math.sqrt((cx-ox)**2 + (cy-oy)**2)
                    max_r = min(max_r, d - orr)
                if max_r > best_r:
                    best_r = max_r
                    best_pos = (cx, cy, max_r)
            if best_pos and best_pos[2] > 0.001:
                all_c.append(best_pos)
            else:
                r = 0.005
                all_c.append((rng.uniform(r+0.01, 1-r-0.01), rng.uniform(r+0.01, 1-r-0.01), r))

        x0 = np.array(all_c).flatten()
        x_opt, m = slsqp_opt(x0, maxiter=5000)
        try_update(x_opt, f"remove-reinsert-{trial}")

    log(f"\nAfter Phase 3: best = {best_metric:.10f}")

    # Phase 4: CMA-ES
    log("\n=== Phase 4: CMA-ES ===")
    try:
        import cma

        def penalty_obj(x, mu=1e5):
            c = x.reshape(-1, 3)
            obj = -np.sum(c[:, 2])
            ineqs = all_ineq(x)
            violations = np.minimum(ineqs, 0)
            obj += mu * np.sum(violations**2)
            return obj

        for sigma in [0.03, 0.05, 0.1]:
            log(f"  CMA-ES sigma={sigma}")
            lower = []
            upper = []
            for _ in range(N):
                lower.extend([0.001, 0.001, 0.001])
                upper.extend([0.999, 0.999, 0.5])

            opts = cma.CMAOptions()
            opts['maxiter'] = 300
            opts['popsize'] = 80
            opts['bounds'] = [lower, upper]
            opts['verbose'] = -9
            opts['tolfun'] = 1e-15
            opts['seed'] = int(sigma * 10000)

            es = cma.CMAEvolutionStrategy(best_x.tolist(), sigma, opts)
            while not es.stop():
                solutions = es.ask()
                fitnesses = [penalty_obj(np.array(s)) for s in solutions]
                es.tell(solutions, fitnesses)

            x_cma = np.array(es.result.xbest)
            x_opt, m = slsqp_opt(x_cma, maxiter=5000)
            try_update(x_opt, f"cma-sigma{sigma}")
            log(f"    CMA sigma={sigma}: metric after polish = {get_metric(x_opt) if is_valid(x_opt) else 'invalid'}")

    except Exception as e:
        log(f"  CMA-ES error: {e}")

    log(f"\nAfter Phase 4: best = {best_metric:.10f}")

    # Final report
    elapsed = time.time() - start_time
    log(f"\n{'='*60}")
    log(f"FINAL BEST: {best_metric:.10f}")
    log(f"Improvements: {len(improvements)}")
    log(f"Time: {elapsed:.1f}s")
    for src, m in improvements:
        log(f"  {src}: {m:.10f}")

    # Update log
    with open(LOG_FILE, "a") as f:
        f.write(f"\n## Optimize3 run {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"- Final metric: {best_metric:.10f}\n")
        f.write(f"- Time: {elapsed:.1f}s\n")
        f.write(f"- Improvements: {len(improvements)}\n")
        for src, m in improvements:
            f.write(f"  - {src}: {m:.10f}\n")

    return best_metric

if __name__ == "__main__":
    run()
