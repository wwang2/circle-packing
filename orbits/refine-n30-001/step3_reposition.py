"""Step 3: Single-circle repositioning + aggressive penalty refinement."""
import json, math, sys
import numpy as np
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
                dx = v[ix] - v[jx]
                dy = v[iy] - v[jy]
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

def max_radius_at(x, y, others):
    """Max radius for a circle at (x,y) given other circles."""
    max_r = min(x, 1-x, y, 1-y)
    for ox, oy, orr in others:
        dist = math.sqrt((x-ox)**2 + (y-oy)**2)
        max_r = min(max_r, dist - orr)
    return max_r

def single_circle_optimize(idx, circles):
    """Remove circle idx, find optimal placement, return new circles."""
    n = len(circles)
    others = np.delete(circles, idx, axis=0)
    others_sum = np.sum(others[:, 2])

    best_x, best_y, best_r = circles[idx]
    best_total = others_sum + best_r

    # Fine grid search
    for gx in np.linspace(0.01, 0.99, 50):
        for gy in np.linspace(0.01, 0.99, 50):
            r = max_radius_at(gx, gy, others)
            if r > 0.001 and others_sum + r > best_total:
                best_x, best_y, best_r = gx, gy, r
                best_total = others_sum + r

    # Refine around best candidate with finer grid
    for dx in np.linspace(-0.02, 0.02, 21):
        for dy in np.linspace(-0.02, 0.02, 21):
            gx, gy = best_x + dx, best_y + dy
            if 0.001 < gx < 0.999 and 0.001 < gy < 0.999:
                r = max_radius_at(gx, gy, others)
                if r > 0.001 and others_sum + r > best_total:
                    best_x, best_y, best_r = gx, gy, r
                    best_total = others_sum + r

    # Even finer refinement
    for dx in np.linspace(-0.003, 0.003, 21):
        for dy in np.linspace(-0.003, 0.003, 21):
            gx, gy = best_x + dx, best_y + dy
            if 0.001 < gx < 0.999 and 0.001 < gy < 0.999:
                r = max_radius_at(gx, gy, others)
                if r > 0.001 and others_sum + r > best_total:
                    best_x, best_y, best_r = gx, gy, r
                    best_total = others_sum + r

    # Scipy optimize for the single circle placement
    def neg_radius(pos):
        x, y = pos
        return -max_radius_at(x, y, others)

    from scipy.optimize import minimize as sp_min
    res = sp_min(neg_radius, [best_x, best_y], method='Nelder-Mead',
                 options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 10000})
    rx, ry = res.x
    rr = max_radius_at(rx, ry, others)
    if rr > 0 and others_sum + rr > best_total:
        best_x, best_y, best_r = rx, ry, rr
        best_total = others_sum + rr

    new_circles = np.insert(others, idx, [best_x, best_y, best_r], axis=0)
    return new_circles, best_total

# Load
circles = load(INPUT)
n = len(circles)
initial_metric = np.sum(circles[:, 2])
print(f"Initial: metric={initial_metric:.10f}", flush=True)

best = circles.copy()
best_metric = initial_metric

# Single-circle repositioning
print("\n--- Single-Circle Repositioning ---", flush=True)
no_improve_count = 0
for round_num in range(5):
    improved_this_round = False
    order = np.random.RandomState(round_num + 100).permutation(n)

    for idx in order:
        trial, trial_metric = single_circle_optimize(idx, best)

        if trial_metric > best_metric + 1e-12:
            # Validate
            val, viol = validate(trial)
            if val:
                best = trial
                best_metric = trial_metric
                improved_this_round = True
                print(f"  Round {round_num}, circle {idx}: metric={best_metric:.10f} (direct)", flush=True)
            else:
                # Try SLSQP polish
                polished, met, val, viol = slsqp_refine(trial)
                if val and met > best_metric:
                    best = polished
                    best_metric = met
                    improved_this_round = True
                    print(f"  Round {round_num}, circle {idx}: metric={best_metric:.10f} (polished)", flush=True)

    if not improved_this_round:
        no_improve_count += 1
        print(f"  Round {round_num}: no improvement ({no_improve_count}/3)", flush=True)
        if no_improve_count >= 2:
            break
    else:
        no_improve_count = 0

# Final SLSQP polish
print("\n--- Final Polish ---", flush=True)
for _ in range(3):
    new_c, met, val, viol = slsqp_refine(best, ftol=1e-15, maxiter=20000)
    if val and met > best_metric + 1e-14:
        best = new_c
        best_metric = met
        print(f"  Polish: metric={met:.10f}", flush=True)
    else:
        break

save(best, OUTPUT)
print(f"\nFinal: metric={best_metric:.10f}", flush=True)
print(f"Improvement over input: {best_metric - initial_metric:+.2e}", flush=True)
