"""Step 5: Careful penalty method to push metric higher while staying feasible.

The basin-hopping found solutions at 2.84266877 that were invalid by tiny margins.
Let's try to find the true optimum by:
1. Very careful L-BFGS-B with high penalty, starting from best
2. Alternating penalty + SLSQP with smaller steps
3. Trust-constr method (handles constraints better)
4. COBYLA as alternative
"""
import json, math, sys
import numpy as np
from scipy.optimize import minimize, differential_evolution
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

def build_constraints_dict(n):
    """Build constraints for trust-constr."""
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
    constraints = build_constraints_dict(n)
    result = minimize(
        lambda v, n=n: -np.sum(v.reshape(n,3)[:,2]), vec, args=(n,),
        method='SLSQP', constraints=constraints,
        options={'ftol': ftol, 'maxiter': maxiter, 'disp': False}
    )
    new_c = result.x.reshape(n, 3)
    met = np.sum(new_c[:, 2])
    val, viol = validate(new_c)
    return new_c, met, val, viol

def cobyla_refine(circles, rhobeg=1e-6, maxiter=50000):
    """COBYLA - derivative-free constrained optimizer."""
    n = len(circles)
    vec = circles.flatten()
    constraints = build_constraints_dict(n)
    result = minimize(
        lambda v, n=n: -np.sum(v.reshape(n,3)[:,2]), vec, args=(n,),
        method='COBYLA', constraints=constraints,
        options={'rhobeg': rhobeg, 'maxiter': maxiter, 'catol': 1e-12}
    )
    new_c = result.x.reshape(n, 3)
    met = np.sum(new_c[:, 2])
    val, viol = validate(new_c)
    return new_c, met, val, viol

def trust_constr_refine(circles):
    """trust-constr method with LinearConstraint/NonlinearConstraint."""
    from scipy.optimize import NonlinearConstraint
    n = len(circles)
    vec = circles.flatten()

    # Build constraint function that returns all constraints at once
    def all_constraints(v):
        c = []
        circles_v = v.reshape(n, 3)
        for i in range(n):
            x, y, r = circles_v[i]
            c.append(x - r)      # x - r >= 0
            c.append(1 - x - r)  # 1 - x - r >= 0
            c.append(y - r)      # y - r >= 0
            c.append(1 - y - r)  # 1 - y - r >= 0
            c.append(r - 1e-12)  # r > 0
        for i in range(n):
            xi, yi, ri = circles_v[i]
            for j in range(i+1, n):
                xj, yj, rj = circles_v[j]
                dx, dy = xi-xj, yi-yj
                c.append(math.sqrt(dx*dx + dy*dy) - ri - rj)
        return np.array(c)

    nc = NonlinearConstraint(all_constraints, 0, np.inf)
    bounds = []
    for i in range(n):
        bounds.extend([(1e-6, 1-1e-6), (1e-6, 1-1e-6), (1e-6, 0.5)])

    result = minimize(
        lambda v: -np.sum(v.reshape(n,3)[:,2]), vec,
        method='trust-constr',
        constraints=nc,
        bounds=bounds,
        options={'maxiter': 10000, 'gtol': 1e-12, 'xtol': 1e-14}
    )
    new_c = result.x.reshape(n, 3)
    met = np.sum(new_c[:, 2])
    val, viol = validate(new_c)
    return new_c, met, val, viol

# Load best
circles = load(INPUT)
n = len(circles)
initial_metric = np.sum(circles[:, 2])
print(f"Initial: metric={initial_metric:.10f}", flush=True)

best = circles.copy()
best_metric = initial_metric

# Try trust-constr
print("\n--- trust-constr ---", flush=True)
try:
    new_c, met, val, viol = trust_constr_refine(best)
    print(f"  trust-constr: metric={met:.10f} valid={val} viol={viol:.2e}", flush=True)
    if val and met > best_metric:
        best = new_c
        best_metric = met
        print(f"  NEW BEST!", flush=True)
except Exception as e:
    print(f"  Error: {e}", flush=True)

# Try COBYLA with various rhobeg
print("\n--- COBYLA ---", flush=True)
for rho in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
    new_c, met, val, viol = cobyla_refine(best, rhobeg=rho, maxiter=100000)
    print(f"  COBYLA rho={rho:.0e}: metric={met:.10f} valid={val} viol={viol:.2e}", flush=True)
    if val and met > best_metric:
        best = new_c
        best_metric = met
        print(f"  NEW BEST!", flush=True)

# Try alternating penalty levels with SLSQP polish
print("\n--- Alternating Penalty + SLSQP ---", flush=True)
from scipy.optimize import minimize as sp_min

def penalty_obj_grad(vec, n, lam):
    circles_v = vec.reshape(n, 3)
    obj = -np.sum(circles_v[:, 2])
    grad = np.zeros_like(vec)
    for i in range(n):
        grad[3*i+2] = -1.0
    penalty = 0.0
    for i in range(n):
        x, y, r = circles_v[i]
        ix, iy, ir = 3*i, 3*i+1, 3*i+2
        for val_v, g_pos, g_neg in [
            (r-x, [(ir,1),(ix,-1)], None),
            (x+r-1, [(ix,1),(ir,1)], None),
            (r-y, [(ir,1),(iy,-1)], None),
            (y+r-1, [(iy,1),(ir,1)], None),
        ]:
            if val_v > 0:
                penalty += val_v**2
                for idx, sign in g_pos:
                    grad[idx] += lam*2*val_v*sign
        if r < 0:
            penalty += r**2
            grad[ir] -= lam*2*r
    for i in range(n):
        xi, yi, ri = circles_v[i]
        for j in range(i+1, n):
            xj, yj, rj = circles_v[j]
            dx, dy = xi-xj, yi-yj
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < 1e-15: continue
            overlap = (ri+rj) - dist
            if overlap > 0:
                penalty += overlap**2
                ddx, ddy = dx/dist, dy/dist
                grad[3*i] -= lam*2*overlap*ddx
                grad[3*i+1] -= lam*2*overlap*ddy
                grad[3*i+2] += lam*2*overlap
                grad[3*j] += lam*2*overlap*ddx
                grad[3*j+1] += lam*2*overlap*ddy
                grad[3*j+2] += lam*2*overlap
    return obj + lam*penalty, grad

for cycle in range(5):
    vec = best.flatten()
    bounds = [(1e-6, 1-1e-6) if i%3 != 2 else (1e-6, 0.5) for i in range(3*n)]

    for lam in [1e6, 1e8, 1e10, 1e12]:
        result = sp_min(
            penalty_obj_grad, vec, args=(n, lam),
            jac=True, method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14}
        )
        vec = result.x

    trial = vec.reshape(n, 3)
    trial, met, val, viol = slsqp_refine(trial)
    if val and met > best_metric:
        best = trial
        best_metric = met
        print(f"  Cycle {cycle}: metric={met:.10f} NEW BEST!", flush=True)
    else:
        print(f"  Cycle {cycle}: metric={met:.10f} valid={val}", flush=True)

# Multiple SLSQP with different maxiter
print("\n--- Extended SLSQP ---", flush=True)
for maxiter in [20000, 50000, 100000]:
    new_c, met, val, viol = slsqp_refine(best, ftol=1e-15, maxiter=maxiter)
    print(f"  SLSQP maxiter={maxiter}: metric={met:.10f} valid={val}", flush=True)
    if val and met > best_metric:
        best = new_c
        best_metric = met

save(best, OUTPUT)
print(f"\nFinal: metric={best_metric:.10f}", flush=True)
print(f"Improvement: {best_metric - initial_metric:+.2e}", flush=True)
