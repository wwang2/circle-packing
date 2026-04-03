"""CMA-ES v3: Position-only optimization with optimal radius computation.

Key insight: Given fixed circle positions, the optimal radii can be computed
by solving a linear program (or a simple iterative scheme). This separates
the combinatorial topology search (CMA-ES on positions) from the continuous
radius optimization.

For each set of positions, the max radii satisfy:
  maximize sum(r_i)
  s.t. r_i + r_j <= dist(i,j) for all pairs
       r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
       r_i >= 0

This is a linear program in r_i!
"""

import json
import math
import sys
import os
import numpy as np
import cma
from scipy.optimize import linprog, minimize

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

def optimal_radii_lp(positions):
    """Given n circle positions, find radii maximizing sum(r_i) via LP.

    maximize sum(r_i)
    s.t. r_i + r_j <= dist(i,j)   for all i < j
         r_i <= x_i
         r_i <= 1 - x_i
         r_i <= y_i
         r_i <= 1 - y_i
         r_i >= 0
    """
    n = len(positions)
    # linprog minimizes c^T x, so we minimize -sum(r_i)
    c = -np.ones(n)

    # Inequality constraints: A_ub @ r <= b_ub
    A_rows = []
    b_rows = []

    # Pairwise: r_i + r_j <= dist(i,j)
    for i in range(n):
        for j in range(i + 1, n):
            dist = math.sqrt((positions[i][0] - positions[j][0])**2 +
                           (positions[i][1] - positions[j][1])**2)
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_rows.append(row)
            b_rows.append(dist)

    # Containment: r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
    for i in range(n):
        xi, yi = positions[i]
        row = np.zeros(n)
        row[i] = 1.0
        for bound in [xi, 1.0 - xi, yi, 1.0 - yi]:
            A_rows.append(row.copy())
            b_rows.append(bound)

    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)

    bounds = [(0, None) for _ in range(n)]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if result.success:
        return result.x, -result.fun
    else:
        return np.zeros(n), 0.0

def positions_to_sum_radii(pos_flat, n):
    """Given flattened positions [x0,y0,x1,y1,...], compute optimal sum of radii."""
    positions = pos_flat.reshape(n, 2)
    # Clip positions to valid range
    positions = np.clip(positions, 0.001, 0.999)
    radii, sr = optimal_radii_lp(positions)
    return -sr  # Minimize negative sum

def slsqp_polish(circles, maxiter=5000):
    """Polish full solution with SLSQP."""
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

def run_position_cmaes(init_positions, sigma, popsize, maxiter, seed=42):
    """Run CMA-ES on positions only, computing optimal radii for fitness."""
    n = len(init_positions)
    x0 = init_positions.flatten()  # [x0,y0,x1,y1,...]

    lower = [0.005] * (2 * n)
    upper = [0.995] * (2 * n)

    opts = cma.CMAOptions()
    opts['seed'] = seed
    opts['maxiter'] = maxiter
    opts['popsize'] = popsize
    opts['verbose'] = -9
    opts['tolfun'] = 1e-14
    opts['tolx'] = 1e-14
    opts['bounds'] = [lower, upper]

    es = cma.CMAEvolutionStrategy(x0, sigma, opts)

    best_sr = 0.0
    gen = 0
    while not es.stop():
        solutions = es.ask()
        fitnesses = [positions_to_sum_radii(np.array(s), n) for s in solutions]
        es.tell(solutions, fitnesses)
        gen += 1

        if gen % 100 == 0:
            best_x = es.result.xbest
            positions = best_x.reshape(n, 2)
            radii, sr = optimal_radii_lp(positions)
            log(f"    gen={gen}: sum_r={sr:.10f}")

    best_x = es.result.xbest
    positions = best_x.reshape(n, 2)
    radii, sr = optimal_radii_lp(positions)

    # Construct full solution
    circles = np.column_stack([positions, radii])
    return circles, sr, gen

def main():
    parent_path = os.path.join(WORKDIR, "..", "sa-001", "solution_n26.json")
    init_circles = load_solution(parent_path)
    n = len(init_circles)
    init_sr = np.sum(init_circles[:, 2])
    log(f"Parent solution: n={n}, sum_radii={init_sr:.10f}")

    # Load current best if exists
    cur_path = os.path.join(WORKDIR, "solution_n26.json")
    if os.path.exists(cur_path):
        cur_circles = load_solution(cur_path)
        cur_sr = np.sum(cur_circles[:, 2])
        if cur_sr > init_sr:
            init_circles = cur_circles
            init_sr = cur_sr
            log(f"Using current best: {init_sr:.10f}")

    best_circles = init_circles.copy()
    best_sr = init_sr

    init_positions = init_circles[:, :2]

    def try_update(circles, sr, label):
        nonlocal best_circles, best_sr
        viol, mv = compute_violation(circles)
        if mv < 1e-10 and sr > best_sr:
            best_sr = sr
            best_circles = circles.copy()
            log(f"  *** NEW BEST ({label}): {best_sr:.10f} (viol={mv:.2e}) ***")
            save_solution(best_circles, os.path.join(WORKDIR, "solution_n26.json"))
            return True
        return False

    # First, verify LP gives same result for parent positions
    radii, lp_sr = optimal_radii_lp(init_positions)
    log(f"LP radii for parent positions: sum_r={lp_sr:.10f} (parent had {init_sr:.10f})")

    # === Phase 1: Small sigma position search ===
    log("\n=== Phase 1: Small sigma position-CMA-ES ===")
    for sigma, pop, iters, seed in [
        (0.005, 80, 500, 42),
        (0.01,  80, 500, 42),
        (0.01,  80, 500, 137),
        (0.02,  100, 400, 42),
        (0.02,  100, 400, 2024),
    ]:
        log(f"\n  sigma={sigma}, pop={pop}, iters={iters}, seed={seed}")
        c, sr, gens = run_position_cmaes(init_positions, sigma, pop, iters, seed)
        log(f"  Position-CMA-ES: sr={sr:.10f}, gens={gens}")

        # Polish with SLSQP
        pol_c, pol_sr, pol_v, pol_mv = slsqp_polish(c, maxiter=5000)
        log(f"  Polish: sr={pol_sr:.10f}, max_v={pol_mv:.2e}")
        try_update(pol_c, pol_sr, "pos-cmaes-polish")

    # === Phase 2: Medium sigma position search ===
    log("\n=== Phase 2: Medium sigma position-CMA-ES ===")
    for sigma, pop, iters, seed in [
        (0.05, 150, 400, 42),
        (0.05, 150, 400, 137),
        (0.05, 150, 400, 999),
        (0.1,  200, 300, 42),
        (0.1,  200, 300, 7777),
    ]:
        log(f"\n  sigma={sigma}, pop={pop}, iters={iters}, seed={seed}")
        c, sr, gens = run_position_cmaes(init_positions, sigma, pop, iters, seed)
        log(f"  Position-CMA-ES: sr={sr:.10f}, gens={gens}")

        pol_c, pol_sr, pol_v, pol_mv = slsqp_polish(c, maxiter=5000)
        log(f"  Polish: sr={pol_sr:.10f}, max_v={pol_mv:.2e}")
        try_update(pol_c, pol_sr, "med-pos-cmaes")

    # === Phase 3: Random restarts with position CMA-ES ===
    log("\n=== Phase 3: Random restart position-CMA-ES ===")
    rng = np.random.RandomState(54321)
    for trial in range(5):
        # Random positions
        positions = rng.uniform(0.05, 0.95, size=(n, 2))
        log(f"\n  Random trial {trial}")
        c, sr, gens = run_position_cmaes(positions, 0.1, 150, 300, seed=trial*100+1)
        log(f"  Position-CMA-ES: sr={sr:.10f}, gens={gens}")

        pol_c, pol_sr, pol_v, pol_mv = slsqp_polish(c, maxiter=5000)
        log(f"  Polish: sr={pol_sr:.10f}, max_v={pol_mv:.2e}")
        try_update(pol_c, pol_sr, f"random-restart-{trial}")

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
