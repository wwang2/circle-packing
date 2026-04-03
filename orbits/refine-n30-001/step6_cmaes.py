"""Step 6: CMA-ES exploration + coordinate descent on circle subgroups."""
import json, math, sys, time
import numpy as np
import cma
from scipy.optimize import minimize
from pathlib import Path

WORK = Path(__file__).parent
INPUT = WORK / "solution_n30.json"
OUTPUT = WORK / "solution_n30.json"

def load(path):
    with open(path) as f:
        return np.array(json.load(f)["circles"])

def save(circles, path):
    data = {"circles": [[float(x), float(y), float(r)] for x, y, r in circles]}
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def validate(circles, tol=1e-10):
    n = len(circles)
    max_viol = 0.0
    for i, (x, y, r) in enumerate(circles):
        if r <= 0: return False, abs(r)
        for v in [r - x, x + r - 1.0, r - y, y + r - 1.0]:
            if v > tol: max_viol = max(max_viol, v)
    for i in range(n):
        xi, yi, ri = circles[i]
        for j in range(i+1, n):
            xj, yj, rj = circles[j]
            dist = math.sqrt((xi-xj)**2 + (yi-yj)**2)
            overlap = (ri + rj) - dist
            if overlap > tol: max_viol = max(max_viol, overlap)
    return max_viol <= tol, max_viol

def penalty_obj(vec, n, lam=1e8):
    circles = vec.reshape(n, 3)
    obj = -np.sum(circles[:, 2])
    penalty = 0.0
    for i in range(n):
        x, y, r = circles[i]
        penalty += max(0, r-x)**2 + max(0, x+r-1)**2
        penalty += max(0, r-y)**2 + max(0, y+r-1)**2
        penalty += max(0, -r)**2
    for i in range(n):
        xi, yi, ri = circles[i]
        for j in range(i+1, n):
            xj, yj, rj = circles[j]
            dist = math.sqrt((xi-xj)**2 + (yi-yj)**2)
            overlap = (ri+rj) - dist
            if overlap > 0: penalty += overlap**2
    return obj + lam * penalty

def build_constraints(n):
    constraints = []
    for i in range(n):
        ix, iy, ir = 3*i, 3*i+1, 3*i+2
        constraints.append({'type': 'ineq', 'fun': lambda v, ix=ix, ir=ir: v[ix] - v[ir]})
        constraints.append({'type': 'ineq', 'fun': lambda v, ix=ix, ir=ir: 1.0 - v[ix] - v[ir]})
        constraints.append({'type': 'ineq', 'fun': lambda v, iy=iy, ir=ir: v[iy] - v[ir]})
        constraints.append({'type': 'ineq', 'fun': lambda v, iy=iy, ir=ir: 1.0 - v[iy] - v[ir]})
        constraints.append({'type': 'ineq', 'fun': lambda v, ir=ir: v[ir] - 1e-12})
    for i in range(n):
        for j in range(i+1, n):
            ix, iy, ir = 3*i, 3*i+1, 3*i+2
            jx, jy, jr = 3*j, 3*j+1, 3*j+2
            def overlap_con(v, ix=ix, iy=iy, ir=ir, jx=jx, jy=jy, jr=jr):
                dx = v[ix] - v[jx]; dy = v[iy] - v[jy]
                return math.sqrt(dx*dx + dy*dy) - v[ir] - v[jr]
            constraints.append({'type': 'ineq', 'fun': overlap_con})
    return constraints

def slsqp_refine(circles, ftol=1e-15, maxiter=10000):
    n = len(circles)
    vec = circles.flatten()
    constraints = build_constraints(n)
    result = minimize(
        lambda v, n=n: -np.sum(v.reshape(n,3)[:,2]), vec, args=(n,),
        method='SLSQP', constraints=constraints,
        options={'ftol': ftol, 'maxiter': maxiter, 'disp': False}
    )
    new_c = result.x.reshape(n, 3)
    met = np.sum(new_c[:, 2])
    val, viol = validate(new_c)
    return new_c, met, val, viol

def subgroup_optimize(circles, indices, lam=1e8):
    """Optimize a subset of circles while keeping others fixed."""
    n = len(circles)
    fixed_mask = np.ones(n, dtype=bool)
    fixed_mask[indices] = False
    fixed = circles[fixed_mask]
    moving = circles[indices]

    k = len(indices)

    def sub_penalty(sub_vec):
        sub_circles = sub_vec.reshape(k, 3)
        obj = -np.sum(sub_circles[:, 2])
        penalty = 0.0

        # Containment for moving circles
        for i in range(k):
            x, y, r = sub_circles[i]
            penalty += max(0, r-x)**2 + max(0, x+r-1)**2
            penalty += max(0, r-y)**2 + max(0, y+r-1)**2
            penalty += max(0, -r)**2

        # Overlap between moving circles
        for i in range(k):
            xi, yi, ri = sub_circles[i]
            for j in range(i+1, k):
                xj, yj, rj = sub_circles[j]
                dist = math.sqrt((xi-xj)**2 + (yi-yj)**2)
                overlap = (ri+rj) - dist
                if overlap > 0: penalty += overlap**2

        # Overlap between moving and fixed circles
        for i in range(k):
            xi, yi, ri = sub_circles[i]
            for j in range(len(fixed)):
                xj, yj, rj = fixed[j]
                dist = math.sqrt((xi-xj)**2 + (yi-yj)**2)
                overlap = (ri+rj) - dist
                if overlap > 0: penalty += overlap**2

        return obj + lam * penalty

    vec = moving.flatten()
    bounds = []
    for i in range(k):
        bounds.extend([(1e-6, 1-1e-6), (1e-6, 1-1e-6), (1e-6, 0.5)])

    result = minimize(sub_penalty, vec, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 5000, 'ftol': 1e-15})

    new_circles = circles.copy()
    new_circles[indices] = result.x.reshape(k, 3)
    return new_circles

# Load
circles = load(INPUT)
n = len(circles)
initial_metric = np.sum(circles[:, 2])
print(f"Initial: metric={initial_metric:.10f}", flush=True)

best = circles.copy()
best_metric = initial_metric
t0 = time.time()

# Part A: CMA-ES with small sigma (refine within basin)
print("\n--- CMA-ES Small Sigma (Basin Refinement) ---", flush=True)
for sigma in [0.001, 0.005, 0.01]:
    for seed in [42, 123, 456]:
        vec = best.flatten()
        opts = cma.CMAOptions()
        opts['seed'] = seed
        opts['maxiter'] = 500
        opts['tolx'] = 1e-14
        opts['tolfun'] = 1e-14
        opts['popsize'] = 50
        opts['verbose'] = -9
        opts['bounds'] = [[1e-6]*3*n, [1-1e-6 if i%3!=2 else 0.5 for i in range(3*n)]]

        es = cma.CMAEvolutionStrategy(vec, sigma, opts)
        es.optimize(lambda v: penalty_obj(v, n, 1e8))
        trial = es.result.xbest.reshape(n, 3)
        # Polish
        trial, met, val, viol = slsqp_refine(trial)
        if val and met > best_metric:
            best = trial
            best_metric = met
            print(f"  CMA sigma={sigma} seed={seed}: metric={met:.10f} NEW BEST!", flush=True)
            save(best, OUTPUT)

    print(f"  CMA sigma={sigma}: best so far {best_metric:.10f} ({time.time()-t0:.0f}s)", flush=True)

# Part B: CMA-ES with larger sigma (explore new basins)
print("\n--- CMA-ES Large Sigma (Exploration) ---", flush=True)
for sigma in [0.05, 0.1, 0.2]:
    for seed in [42, 123, 789]:
        if time.time() - t0 > 300:
            break
        vec = best.flatten()
        opts = cma.CMAOptions()
        opts['seed'] = seed
        opts['maxiter'] = 300
        opts['popsize'] = 100
        opts['verbose'] = -9
        opts['bounds'] = [[1e-6]*3*n, [1-1e-6 if i%3!=2 else 0.5 for i in range(3*n)]]

        es = cma.CMAEvolutionStrategy(vec, sigma, opts)
        es.optimize(lambda v: penalty_obj(v, n, 1e8))
        trial = es.result.xbest.reshape(n, 3)
        trial, met, val, viol = slsqp_refine(trial)
        if val and met > best_metric:
            best = trial
            best_metric = met
            print(f"  CMA sigma={sigma} seed={seed}: metric={met:.10f} NEW BEST!", flush=True)
            save(best, OUTPUT)

    print(f"  CMA sigma={sigma}: best so far {best_metric:.10f} ({time.time()-t0:.0f}s)", flush=True)

# Part C: Coordinate descent on subgroups of 3-5 circles
print("\n--- Subgroup Coordinate Descent ---", flush=True)
rng = np.random.RandomState(42)
no_improve = 0
for iteration in range(20):
    if time.time() - t0 > 480:
        break

    improved = False
    # Try random subgroups of size 3-6
    for _ in range(10):
        k = rng.randint(3, 7)
        indices = rng.choice(n, k, replace=False)

        trial = subgroup_optimize(best, indices, lam=1e10)
        trial, met, val, viol = slsqp_refine(trial)

        if val and met > best_metric + 1e-13:
            best = trial
            best_metric = met
            improved = True
            print(f"  Iter {iteration}, group {list(indices)}: metric={met:.10f}", flush=True)

    if not improved:
        no_improve += 1
        if no_improve >= 3:
            print(f"  No improvement for 3 iterations, stopping", flush=True)
            break
    else:
        no_improve = 0

    if iteration % 5 == 0:
        print(f"  Iteration {iteration}: best={best_metric:.10f} ({time.time()-t0:.0f}s)", flush=True)

save(best, OUTPUT)
print(f"\nFinal: metric={best_metric:.10f}", flush=True)
print(f"Improvement: {best_metric - initial_metric:+.2e}", flush=True)
print(f"Total time: {time.time()-t0:.0f}s", flush=True)
