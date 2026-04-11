"""
Massive optimization campaign for n=32 circle packing in [0,1]^2.
Maximize sum of radii.

Strategies:
1. Basin-hopping with SLSQP from current best
2. Multi-start from diverse initializations
3. Perturbation campaigns (remove/reinsert, swap, explode)
4. CMA-ES global search
"""

import json
import math
import numpy as np
from scipy.optimize import minimize, basinhopping, differential_evolution
from pathlib import Path
import time
import sys
import copy

N = 32
BEST_FILE = Path(__file__).parent / "solution_n32.json"
LOG_FILE = Path(__file__).parent / "log.md"

def load_solution(path):
    with open(path) as f:
        data = json.load(f)
    circles = data.get("circles", data)
    return np.array(circles)

def save_solution(circles, path):
    data = {"circles": [[float(x), float(y), float(r)] for x, y, r in circles]}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def circles_to_x(circles):
    """Flatten circles array to optimization vector [x0,y0,r0, x1,y1,r1, ...]"""
    return circles.flatten()

def x_to_circles(x):
    return x.reshape(-1, 3)

def sum_radii(x):
    """Negative sum of radii (for minimization)."""
    circles = x_to_circles(x)
    return -np.sum(circles[:, 2])

def constraint_containment(x):
    """All containment constraints: r <= x, x+r <= 1, r <= y, y+r <= 1, r > 0"""
    c = x_to_circles(x)
    cons = []
    for i in range(len(c)):
        xi, yi, ri = c[i]
        cons.append(xi - ri)       # x - r >= 0
        cons.append(1.0 - xi - ri) # 1 - x - r >= 0
        cons.append(yi - ri)       # y - r >= 0
        cons.append(1.0 - yi - ri) # 1 - y - r >= 0
        cons.append(ri - 1e-6)     # r > 0
    return np.array(cons)

def constraint_nooverlap(x):
    """All non-overlap constraints: dist(i,j) - ri - rj >= 0"""
    c = x_to_circles(x)
    n = len(c)
    cons = []
    for i in range(n):
        for j in range(i+1, n):
            dx = c[i,0] - c[j,0]
            dy = c[i,1] - c[j,1]
            dist = math.sqrt(dx*dx + dy*dy)
            cons.append(dist - c[i,2] - c[j,2])
    return np.array(cons)

def all_constraints(x):
    return np.concatenate([constraint_containment(x), constraint_nooverlap(x)])

def is_valid(x, tol=1e-10):
    cons = all_constraints(x)
    return np.all(cons >= -tol)

def get_metric(x):
    circles = x_to_circles(x)
    return np.sum(circles[:, 2])

def make_scipy_constraints():
    """Create scipy constraint dicts for SLSQP."""
    return [
        {"type": "ineq", "fun": constraint_containment},
        {"type": "ineq", "fun": constraint_nooverlap},
    ]

def make_bounds(n=N):
    """Bounds for each variable."""
    bounds = []
    for _ in range(n):
        bounds.append((0.001, 0.999))  # x
        bounds.append((0.001, 0.999))  # y
        bounds.append((0.001, 0.5))    # r
    return bounds

def penalty_objective(x, mu=1e4):
    """Objective with penalty for constraint violations."""
    circles = x_to_circles(x)
    obj = -np.sum(circles[:, 2])

    # Containment penalties
    for i in range(len(circles)):
        xi, yi, ri = circles[i]
        obj += mu * max(0, ri - xi)**2
        obj += mu * max(0, xi + ri - 1)**2
        obj += mu * max(0, ri - yi)**2
        obj += mu * max(0, yi + ri - 1)**2
        obj += mu * max(0, -ri + 1e-6)**2

    # Overlap penalties
    n = len(circles)
    for i in range(n):
        for j in range(i+1, n):
            dx = circles[i,0] - circles[j,0]
            dy = circles[i,1] - circles[j,1]
            dist = math.sqrt(dx*dx + dy*dy)
            overlap = circles[i,2] + circles[j,2] - dist
            if overlap > 0:
                obj += mu * overlap**2
    return obj

def penalty_objective_grad(x, mu=1e4):
    """Numerical gradient of penalty objective."""
    n = len(x)
    grad = np.zeros(n)
    h = 1e-7
    f0 = penalty_objective(x, mu)
    for i in range(n):
        x1 = x.copy()
        x1[i] += h
        grad[i] = (penalty_objective(x1, mu) - f0) / h
    return grad

def slsqp_polish(x0, maxiter=2000):
    """Polish solution with SLSQP."""
    constraints = make_scipy_constraints()
    bounds = make_bounds()
    result = minimize(
        sum_radii, x0, method='SLSQP',
        bounds=bounds, constraints=constraints,
        options={'maxiter': maxiter, 'ftol': 1e-15, 'disp': False}
    )
    return result.x, -result.fun

def progressive_penalty_optimize(x0, maxiter=500):
    """L-BFGS-B with progressively increasing penalty, then SLSQP polish."""
    x = x0.copy()
    bounds = make_bounds()

    for mu in [1e2, 1e3, 1e4, 1e5, 1e6]:
        result = minimize(
            lambda xx: penalty_objective(xx, mu),
            x, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-15}
        )
        x = result.x

    # Polish with SLSQP
    x, metric = slsqp_polish(x)
    return x, metric

# --- Initialization strategies ---

def init_from_solution(path):
    circles = load_solution(path)
    return circles_to_x(circles)

def init_random(n=N, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    x = np.zeros(3*n)
    circles = []
    for i in range(n):
        for _ in range(1000):
            r = rng.uniform(0.02, 0.12)
            cx = rng.uniform(r, 1-r)
            cy = rng.uniform(r, 1-r)
            ok = True
            for (ox, oy, orr) in circles:
                dist = math.sqrt((cx-ox)**2 + (cy-oy)**2)
                if dist < r + orr:
                    ok = False
                    break
            if ok:
                circles.append((cx, cy, r))
                x[3*i] = cx
                x[3*i+1] = cy
                x[3*i+2] = r
                break
        else:
            # Fallback: tiny circle
            r = 0.005
            cx = rng.uniform(r, 1-r)
            cy = rng.uniform(r, 1-r)
            circles.append((cx, cy, r))
            x[3*i] = cx
            x[3*i+1] = cy
            x[3*i+2] = r
    return x

def init_hex_grid(n=N):
    """Hex grid initialization."""
    # Estimate grid size
    cols = int(math.ceil(math.sqrt(n * 2 / math.sqrt(3))))
    rows = int(math.ceil(n / cols))
    r = min(1.0 / (2*cols), 1.0 / (2*rows * math.sqrt(3)/2)) * 0.9

    circles = []
    for row in range(rows):
        for col in range(cols):
            if len(circles) >= n:
                break
            cx = (col + 0.5) / cols
            cy = (row + 0.5) / rows
            if row % 2 == 1:
                cx += 0.5 / cols
            if cx - r < 0: cx = r + 0.001
            if cx + r > 1: cx = 1 - r - 0.001
            if cy - r < 0: cy = r + 0.001
            if cy + r > 1: cy = 1 - r - 0.001
            circles.append((cx, cy, r))

    x = np.zeros(3*n)
    for i, (cx, cy, rr) in enumerate(circles[:n]):
        x[3*i] = cx
        x[3*i+1] = cy
        x[3*i+2] = rr
    return x

def init_ring(n=N):
    """Concentric ring initialization."""
    # 1 center + ring of 6 + ring of 12 + ring of 13
    layers = []
    remaining = n
    k = 0
    while remaining > 0:
        if k == 0:
            count = 1
        else:
            count = min(6*k, remaining)
        layers.append(count)
        remaining -= count
        k += 1

    circles = []
    r_est = 0.08
    for layer_idx, count in enumerate(layers):
        if layer_idx == 0:
            circles.append((0.5, 0.5, r_est))
        else:
            ring_r = layer_idx * 2.2 * r_est
            for j in range(count):
                angle = 2 * math.pi * j / count
                cx = 0.5 + ring_r * math.cos(angle)
                cy = 0.5 + ring_r * math.sin(angle)
                cx = max(r_est+0.001, min(1-r_est-0.001, cx))
                cy = max(r_est+0.001, min(1-r_est-0.001, cy))
                circles.append((cx, cy, r_est))

    x = np.zeros(3*n)
    for i, (cx, cy, rr) in enumerate(circles[:n]):
        x[3*i] = cx
        x[3*i+1] = cy
        x[3*i+2] = rr
    return x

# --- Perturbation strategies ---

def perturb_solution(x, rng, scale=0.05):
    """Random perturbation of all positions and radii."""
    x_new = x.copy()
    n = len(x) // 3
    for i in range(n):
        x_new[3*i] += rng.normal(0, scale)
        x_new[3*i+1] += rng.normal(0, scale)
        x_new[3*i+2] += rng.normal(0, scale * 0.5)
        # Clamp
        r = max(0.001, x_new[3*i+2])
        x_new[3*i+2] = r
        x_new[3*i] = max(r+0.001, min(1-r-0.001, x_new[3*i]))
        x_new[3*i+1] = max(r+0.001, min(1-r-0.001, x_new[3*i+1]))
    return x_new

def remove_reinsert(x, num_remove=2, rng=None):
    """Remove some circles, re-optimize, then greedily reinsert."""
    if rng is None:
        rng = np.random.default_rng()

    circles = x_to_circles(x.copy())
    n = len(circles)

    # Remove circles with smallest radii (or random)
    if rng.random() < 0.5:
        # Remove smallest
        radii = circles[:, 2]
        remove_idx = np.argsort(radii)[:num_remove]
    else:
        # Remove random
        remove_idx = rng.choice(n, size=num_remove, replace=False)

    keep_idx = [i for i in range(n) if i not in remove_idx]
    kept = circles[keep_idx]

    # Optimize the kept circles
    x_kept = circles_to_x(kept)
    n_kept = len(kept)

    bounds_kept = []
    for _ in range(n_kept):
        bounds_kept.append((0.001, 0.999))
        bounds_kept.append((0.001, 0.999))
        bounds_kept.append((0.001, 0.5))

    # Quick SLSQP on kept
    def obj_kept(xx):
        return -np.sum(xx.reshape(-1,3)[:,2])

    def cont_kept(xx):
        c = xx.reshape(-1,3)
        cons = []
        for i in range(len(c)):
            cons.extend([c[i,0]-c[i,2], 1-c[i,0]-c[i,2], c[i,1]-c[i,2], 1-c[i,1]-c[i,2], c[i,2]-1e-6])
        for i in range(len(c)):
            for j in range(i+1,len(c)):
                dx = c[i,0]-c[j,0]; dy = c[i,1]-c[j,1]
                cons.append(math.sqrt(dx*dx+dy*dy) - c[i,2] - c[j,2])
        return np.array(cons)

    res = minimize(obj_kept, x_kept, method='SLSQP', bounds=bounds_kept,
                   constraints=[{"type":"ineq","fun":cont_kept}],
                   options={'maxiter':500, 'ftol':1e-15})
    kept = res.x.reshape(-1,3)

    # Greedily reinsert circles
    all_circles = list(kept)
    for _ in range(num_remove):
        best_r = 0
        best_pos = None
        # Try many random positions
        for _ in range(2000):
            cx = rng.uniform(0.01, 0.99)
            cy = rng.uniform(0.01, 0.99)
            # Max radius at this position
            max_r = min(cx, 1-cx, cy, 1-cy)
            for (ox, oy, orr) in all_circles:
                dist = math.sqrt((cx-ox)**2 + (cy-oy)**2)
                max_r = min(max_r, dist - orr)
            if max_r > best_r:
                best_r = max_r
                best_pos = (cx, cy, max_r)
        if best_pos and best_pos[2] > 0.001:
            all_circles.append(best_pos)
        else:
            # Tiny fallback
            r = 0.005
            cx = rng.uniform(r+0.01, 1-r-0.01)
            cy = rng.uniform(r+0.01, 1-r-0.01)
            all_circles.append((cx, cy, r))

    return circles_to_x(np.array(all_circles))

def swap_neighbors(x, rng):
    """Swap positions of two random circles."""
    circles = x_to_circles(x.copy())
    n = len(circles)
    i, j = rng.choice(n, size=2, replace=False)
    circles[i, 0], circles[j, 0] = circles[j, 0].copy(), circles[i, 0].copy()
    circles[i, 1], circles[j, 1] = circles[j, 1].copy(), circles[i, 1].copy()
    return circles_to_x(circles)

# --- Main campaign ---

def run_campaign():
    best_x = init_from_solution(BEST_FILE)
    best_metric = get_metric(best_x)
    print(f"Starting metric: {best_metric:.10f}")

    improvements = []
    no_improve_count = 0

    def try_update(x, source):
        nonlocal best_x, best_metric, no_improve_count
        if is_valid(x):
            m = get_metric(x)
            if m > best_metric + 1e-12:
                improvement = m - best_metric
                best_metric = m
                best_x = x.copy()
                save_solution(x_to_circles(x), BEST_FILE)
                improvements.append((source, m))
                no_improve_count = 0
                print(f"  *** NEW BEST: {m:.10f} (+{improvement:.2e}) from {source}")
                return True
        return False

    # === Phase 1: Basin-hopping from current best ===
    print("\n=== Phase 1: Basin-hopping from current best ===")
    rng = np.random.default_rng(42)

    for seed in range(20):
        print(f"\nBasin-hop seed {seed}...")
        rng_local = np.random.default_rng(seed * 137 + 42)

        for step_size in [0.02, 0.05, 0.1, 0.15]:
            x0 = perturb_solution(best_x, rng_local, scale=step_size)
            x_opt, metric = progressive_penalty_optimize(x0)
            try_update(x_opt, f"basin-hop-s{seed}-step{step_size}")

            # Also try SLSQP directly from perturbation
            x_opt2, metric2 = slsqp_polish(x0)
            try_update(x_opt2, f"slsqp-direct-s{seed}-step{step_size}")

    print(f"\nAfter Phase 1: best = {best_metric:.10f}")

    # === Phase 2: Multi-start from scratch ===
    print("\n=== Phase 2: Multi-start from diverse initializations ===")

    for trial in range(100):
        if trial % 10 == 0:
            print(f"  Multi-start trial {trial}/100...")
        rng_local = np.random.default_rng(trial * 31 + 1000)

        # Choose initialization
        if trial % 4 == 0:
            x0 = init_random(N, rng_local)
        elif trial % 4 == 1:
            x0 = init_hex_grid(N)
            x0 = perturb_solution(x0, rng_local, scale=0.03)
        elif trial % 4 == 2:
            x0 = init_ring(N)
            x0 = perturb_solution(x0, rng_local, scale=0.03)
        else:
            # Heavily perturbed best
            x0 = perturb_solution(best_x, rng_local, scale=0.2)

        x_opt, metric = progressive_penalty_optimize(x0, maxiter=300)
        try_update(x_opt, f"multi-start-{trial}")

    print(f"\nAfter Phase 2: best = {best_metric:.10f}")

    # === Phase 3: Perturbation campaigns ===
    print("\n=== Phase 3: Perturbation campaigns ===")

    for trial in range(50):
        if trial % 10 == 0:
            print(f"  Perturbation trial {trial}/50...")
        rng_local = np.random.default_rng(trial * 17 + 2000)

        if trial % 3 == 0:
            # Remove-reinsert
            num_remove = rng_local.integers(1, 4)
            x0 = remove_reinsert(best_x, num_remove=num_remove, rng=rng_local)
        elif trial % 3 == 1:
            # Swap + optimize
            x0 = swap_neighbors(best_x, rng_local)
        else:
            # Small perturbation
            x0 = perturb_solution(best_x, rng_local, scale=0.01)

        x_opt, metric = slsqp_polish(x0)
        try_update(x_opt, f"perturb-{trial}")

        # Also try progressive penalty
        x_opt2, metric2 = progressive_penalty_optimize(x0, maxiter=300)
        try_update(x_opt2, f"perturb-penalty-{trial}")

    print(f"\nAfter Phase 3: best = {best_metric:.10f}")

    # === Phase 4: CMA-ES ===
    print("\n=== Phase 4: CMA-ES ===")
    try:
        import cma

        for sigma in [0.05, 0.1, 0.2]:
            print(f"\n  CMA-ES sigma={sigma}...")
            x0_cma = best_x.copy()

            opts = {
                'maxiter': 500,
                'popsize': 100,
                'bounds': [
                    [0.001]*N + [0.001]*N + [0.001]*N,  # This is wrong shape, fix below
                    [0.999]*N + [0.999]*N + [0.5]*N,
                ],
                'verbose': -9,
                'tolfun': 1e-15,
                'seed': int(sigma * 1000),
            }

            # Correct bounds for interleaved x,y,r format
            lower = []
            upper = []
            for _ in range(N):
                lower.extend([0.001, 0.001, 0.001])
                upper.extend([0.999, 0.999, 0.5])
            opts['bounds'] = [lower, upper]

            es = cma.CMAEvolutionStrategy(x0_cma, sigma, opts)

            while not es.stop():
                solutions = es.ask()
                fitnesses = [penalty_objective(s, mu=1e5) for s in solutions]
                es.tell(solutions, fitnesses)

            x_cma = es.result.xbest
            # Polish with SLSQP
            x_polished, m = slsqp_polish(x_cma)
            try_update(x_polished, f"cma-es-sigma{sigma}")

    except Exception as e:
        print(f"  CMA-ES error: {e}")

    print(f"\nAfter Phase 4: best = {best_metric:.10f}")

    # === Phase 5: Differential Evolution ===
    print("\n=== Phase 5: Differential Evolution ===")
    bounds = make_bounds()

    try:
        # Use current best to seed the population
        def de_callback(xk, convergence):
            return False  # Don't stop early

        result = differential_evolution(
            lambda xx: penalty_objective(xx, mu=1e5),
            bounds=bounds,
            maxiter=200,
            popsize=30,
            seed=42,
            tol=1e-15,
            init='sobol',
            callback=de_callback,
        )
        x_de = result.x
        x_polished, m = slsqp_polish(x_de)
        try_update(x_polished, "diff-evolution")
    except Exception as e:
        print(f"  DE error: {e}")

    print(f"\nAfter Phase 5: best = {best_metric:.10f}")

    # === Phase 6: Intensive local search around best ===
    print("\n=== Phase 6: Intensive local search ===")

    for trial in range(200):
        if trial % 50 == 0:
            print(f"  Local search trial {trial}/200...")
        rng_local = np.random.default_rng(trial * 7 + 5000)

        # Very small perturbations
        scale = rng_local.choice([0.001, 0.003, 0.005, 0.01])
        x0 = perturb_solution(best_x, rng_local, scale=scale)
        x_opt, metric = slsqp_polish(x0, maxiter=5000)
        try_update(x_opt, f"local-{trial}-s{scale}")

    print(f"\n{'='*60}")
    print(f"FINAL BEST: {best_metric:.10f}")
    print(f"Improvements found: {len(improvements)}")
    for source, metric in improvements:
        print(f"  {source}: {metric:.10f}")

    return best_metric, improvements

if __name__ == "__main__":
    start = time.time()
    best_metric, improvements = run_campaign()
    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")

    # Update log
    with open(LOG_FILE, "a") as f:
        f.write(f"\n## Optimization run {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"- Final metric: {best_metric:.10f}\n")
        f.write(f"- Time: {elapsed:.1f}s\n")
        f.write(f"- Improvements: {len(improvements)}\n")
        for source, metric in improvements:
            f.write(f"  - {source}: {metric:.10f}\n")
