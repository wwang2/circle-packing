"""Step 1: SLSQP refinement of n=30 solution."""
import json, math, sys
import numpy as np
from scipy.optimize import minimize
from pathlib import Path

WORK = Path(__file__).parent
INPUT = WORK.parent / "diffevo-001" / "solution_n30.json"
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
    return new_c, met, val, viol, result.nit

# Load
circles = load(INPUT)
initial_metric = np.sum(circles[:, 2])
val, viol = validate(circles)
print(f"Initial: metric={initial_metric:.10f} valid={val} viol={viol:.2e}", flush=True)

best = circles.copy()
best_metric = initial_metric

# Multiple SLSQP passes
for ftol in [1e-10, 1e-12, 1e-13, 1e-14, 1e-15]:
    new_c, met, val, viol, nit = slsqp_refine(best, ftol=ftol)
    print(f"SLSQP ftol={ftol:.0e}: metric={met:.10f} valid={val} viol={viol:.2e} nit={nit}", flush=True)
    if val and met > best_metric:
        best = new_c
        best_metric = met

# Try from slightly perturbed starting points
rng = np.random.RandomState(42)
for trial in range(10):
    perturbed = best.copy()
    perturbed += rng.normal(0, 0.001, perturbed.shape)
    # Clamp
    for i in range(len(perturbed)):
        perturbed[i,2] = max(1e-4, perturbed[i,2])
        perturbed[i,0] = np.clip(perturbed[i,0], perturbed[i,2], 1-perturbed[i,2])
        perturbed[i,1] = np.clip(perturbed[i,1], perturbed[i,2], 1-perturbed[i,2])
    new_c, met, val, viol, nit = slsqp_refine(perturbed, ftol=1e-15)
    if val and met > best_metric:
        print(f"Perturbed trial {trial}: metric={met:.10f} NEW BEST!", flush=True)
        best = new_c
        best_metric = met
    elif trial % 5 == 0:
        print(f"Perturbed trial {trial}: metric={met:.10f} valid={val}", flush=True)

save(best, OUTPUT)
print(f"\nFinal: metric={best_metric:.10f}", flush=True)
print(f"Improvement: {best_metric - initial_metric:+.2e}", flush=True)
