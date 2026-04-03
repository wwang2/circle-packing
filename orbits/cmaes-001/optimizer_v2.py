"""CMA-ES optimizer v2 - faster, focused, with flushed output.

Strategy: Run CMA-ES with progressive strategies, SLSQP polish each result.
"""

import json
import math
import sys
import os
import numpy as np
import cma
from scipy.optimize import minimize

WORKDIR = os.path.dirname(os.path.abspath(__file__))

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

def compute_violation(circles):
    n = len(circles)
    total_viol = 0.0
    max_viol = 0.0
    for i in range(n):
        x, y, r = circles[i]
        for v in [r - x, x + r - 1.0, r - y, y + r - 1.0, -r]:
            if v > 0:
                total_viol += v
                max_viol = max(max_viol, v)
    for i in range(n):
        xi, yi, ri = circles[i]
        for j in range(i + 1, n):
            xj, yj, rj = circles[j]
            dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)
            overlap = (ri + rj) - dist
            if overlap > 0:
                total_viol += overlap
                max_viol = max(max_viol, overlap)
    return total_viol, max_viol

def objective(x, n, penalty_weight):
    circles = x.reshape(n, 3)
    sum_radii = np.sum(circles[:, 2])
    total_viol, _ = compute_violation(circles)
    return -sum_radii + penalty_weight * total_viol

def slsqp_polish(circles, maxiter=5000):
    n = len(circles)
    x0 = circles.flatten()

    def neg_sum_radii(x):
        return -np.sum(x.reshape(n, 3)[:, 2])

    constraints = []
    for i in range(n):
        ri = 3 * i + 2
        xi = 3 * i
        yi = 3 * i + 1
        constraints.append({"type": "ineq", "fun": lambda x, idx=ri: x[idx] - 1e-12})
        constraints.append({"type": "ineq", "fun": lambda x, ix=xi, ir=ri: x[ix] - x[ir]})
        constraints.append({"type": "ineq", "fun": lambda x, ix=xi, ir=ri: 1.0 - x[ix] - x[ir]})
        constraints.append({"type": "ineq", "fun": lambda x, iy=yi, ir=ri: x[iy] - x[ir]})
        constraints.append({"type": "ineq", "fun": lambda x, iy=yi, ir=ri: 1.0 - x[iy] - x[ir]})

    for i in range(n):
        for j in range(i + 1, n):
            def overlap_con(x, ii=i, jj=j):
                xi, yi, ri = x[3*ii], x[3*ii+1], x[3*ii+2]
                xj, yj, rj = x[3*jj], x[3*jj+1], x[3*jj+2]
                return math.sqrt((xi-xj)**2 + (yi-yj)**2) - ri - rj
            constraints.append({"type": "ineq", "fun": overlap_con})

    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])

    res = minimize(neg_sum_radii, x0, method="SLSQP",
                   constraints=constraints, bounds=bounds,
                   options={"maxiter": maxiter, "ftol": 1e-15, "disp": False})

    result_circles = res.x.reshape(n, 3)
    viol, max_v = compute_violation(result_circles)
    sr = np.sum(result_circles[:, 2])
    return result_circles, sr, viol, max_v

def run_cmaes_single(init_circles, sigma, popsize, maxiter, penalty_weight, seed=42):
    """Single CMA-ES run with fixed penalty."""
    n = len(init_circles)
    x0 = init_circles.flatten()

    lower, upper = [], []
    for i in range(n):
        lower.extend([0.001, 0.001, 0.001])
        upper.extend([0.999, 0.999, 0.499])

    opts = cma.CMAOptions()
    opts['seed'] = seed
    opts['maxiter'] = maxiter
    opts['popsize'] = popsize
    opts['verbose'] = -9
    opts['tolfun'] = 1e-14
    opts['tolx'] = 1e-14
    opts['bounds'] = [lower, upper]

    es = cma.CMAEvolutionStrategy(x0, sigma, opts)
    while not es.stop():
        solutions = es.ask()
        fitnesses = [objective(x, n, penalty_weight) for x in solutions]
        es.tell(solutions, fitnesses)

    best_x = es.result.xbest
    c = best_x.reshape(n, 3)
    sr = np.sum(c[:, 2])
    v, mv = compute_violation(c)
    return c, sr, v, mv, es.result.iterations

def main():
    parent_path = os.path.join(WORKDIR, "..", "sa-001", "solution_n26.json")
    init_circles = load_solution(parent_path)
    n = len(init_circles)
    init_sr = np.sum(init_circles[:, 2])
    log(f"Parent solution: n={n}, sum_radii={init_sr:.10f}")

    best_circles = init_circles.copy()
    best_sr = init_sr

    def try_update(circles, sr, mv, label):
        nonlocal best_circles, best_sr
        if mv < 1e-10 and sr > best_sr:
            best_sr = sr
            best_circles = circles.copy()
            log(f"  *** NEW BEST ({label}): {best_sr:.10f} ***")
            save_solution(best_circles, os.path.join(WORKDIR, "solution_n26.json"))
            return True
        return False

    # === Phase 1: Small sigma refinement ===
    log("\n=== Phase 1: Small sigma refinement ===")
    configs = [
        (0.005, 100, 500, 1e5, 42),
        (0.01,  100, 500, 1e5, 42),
        (0.01,  100, 500, 1e6, 42),
        (0.02,  100, 300, 1e5, 42),
    ]
    for sigma, pop, iters, pen, seed in configs:
        log(f"\n  sigma={sigma}, pop={pop}, iters={iters}, pen={pen:.0e}, seed={seed}")
        c, sr, v, mv, gens = run_cmaes_single(best_circles, sigma, pop, iters, pen, seed)
        log(f"  CMA-ES: sr={sr:.10f}, viol={v:.2e}, max_v={mv:.2e}, gens={gens}")
        try_update(c, sr, mv, "cmaes-raw")

        # Polish
        pol_c, pol_sr, pol_v, pol_mv = slsqp_polish(c, maxiter=3000)
        log(f"  Polish: sr={pol_sr:.10f}, max_v={pol_mv:.2e}")
        try_update(pol_c, pol_sr, pol_mv, "polished")

    # === Phase 2: Medium sigma nearby basin search ===
    log("\n=== Phase 2: Medium sigma exploration ===")
    configs2 = [
        (0.05, 150, 400, 1e4, 42),
        (0.05, 150, 400, 1e4, 137),
        (0.1,  200, 300, 1e4, 42),
        (0.1,  200, 300, 1e4, 2024),
    ]
    for sigma, pop, iters, pen, seed in configs2:
        log(f"\n  sigma={sigma}, pop={pop}, iters={iters}, pen={pen:.0e}, seed={seed}")
        c, sr, v, mv, gens = run_cmaes_single(best_circles, sigma, pop, iters, pen, seed)
        log(f"  CMA-ES: sr={sr:.10f}, viol={v:.2e}, max_v={mv:.2e}, gens={gens}")

        # Progressive penalty polish: first CMA-ES with high penalty, then SLSQP
        c2, sr2, v2, mv2, _ = run_cmaes_single(c, 0.005, 50, 200, 1e7, seed)
        log(f"  Tighten: sr={sr2:.10f}, max_v={mv2:.2e}")

        pol_c, pol_sr, pol_v, pol_mv = slsqp_polish(c2, maxiter=5000)
        log(f"  Polish: sr={pol_sr:.10f}, max_v={pol_mv:.2e}")
        try_update(pol_c, pol_sr, pol_mv, "med-polished")

    # === Phase 3: Large sigma new basin search ===
    log("\n=== Phase 3: Large sigma exploration ===")
    configs3 = [
        (0.2, 200, 300, 1e3, 42),
        (0.2, 200, 300, 1e3, 99),
        (0.3, 300, 200, 1e3, 314),
    ]
    for sigma, pop, iters, pen, seed in configs3:
        log(f"\n  sigma={sigma}, pop={pop}, iters={iters}, pen={pen:.0e}, seed={seed}")
        c, sr, v, mv, gens = run_cmaes_single(init_circles, sigma, pop, iters, pen, seed)
        log(f"  CMA-ES: sr={sr:.10f}, viol={v:.2e}, max_v={mv:.2e}, gens={gens}")

        # Tighten penalty
        c2, sr2, v2, mv2, _ = run_cmaes_single(c, 0.01, 100, 300, 1e6, seed)
        log(f"  Tighten: sr={sr2:.10f}, max_v={mv2:.2e}")

        pol_c, pol_sr, pol_v, pol_mv = slsqp_polish(c2, maxiter=5000)
        log(f"  Polish: sr={pol_sr:.10f}, max_v={pol_mv:.2e}")
        try_update(pol_c, pol_sr, pol_mv, "large-polished")

    # === Phase 4: Perturbation + Polish ===
    log("\n=== Phase 4: Perturbation + Polish ===")
    rng = np.random.RandomState(88888)
    no_improve_count = 0
    for attempt in range(20):
        perturbed = best_circles.copy()
        n_perturb = rng.randint(1, max(2, n // 3))
        indices = rng.choice(n, n_perturb, replace=False)
        scale = rng.choice([0.01, 0.02, 0.05])
        for idx in indices:
            perturbed[idx, 0] += rng.normal(0, scale)
            perturbed[idx, 1] += rng.normal(0, scale)
            perturbed[idx, 2] *= (1 + rng.normal(0, scale * 0.3))
        perturbed[:, 2] = np.clip(perturbed[:, 2], 0.005, 0.495)
        perturbed[:, 0] = np.clip(perturbed[:, 0], perturbed[:, 2], 1 - perturbed[:, 2])
        perturbed[:, 1] = np.clip(perturbed[:, 1], perturbed[:, 2], 1 - perturbed[:, 2])

        pol_c, pol_sr, pol_v, pol_mv = slsqp_polish(perturbed, maxiter=5000)
        improved = try_update(pol_c, pol_sr, pol_mv, f"perturb-{attempt}")
        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1
        if attempt % 5 == 0:
            log(f"  Perturb {attempt}: sr={pol_sr:.10f}, max_v={pol_mv:.2e}, best={best_sr:.10f}")
        if no_improve_count >= 10:
            log(f"  No improvement in 10 attempts, stopping perturbation")
            break

    # === Final ===
    log(f"\n{'='*60}")
    log(f"FINAL: sum_radii = {best_sr:.10f}")
    log(f"Parent:           {init_sr:.10f}")
    log(f"Delta:            {best_sr - init_sr:+.12f}")

    viol, max_v = compute_violation(best_circles)
    log(f"Validation: total_viol={viol:.2e}, max_viol={max_v:.2e}")

    out_path = os.path.join(WORKDIR, "solution_n26.json")
    save_solution(best_circles, out_path)
    log(f"Saved: {out_path}")
    return best_sr

if __name__ == "__main__":
    main()
