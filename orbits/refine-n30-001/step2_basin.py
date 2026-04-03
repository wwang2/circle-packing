"""Step 2: Basin-hopping + penalty refinement."""
import json, math, sys
import numpy as np
from scipy.optimize import minimize, basinhopping
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

def penalty_obj(vec, n, lam):
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

def penalty_grad(vec, n, lam):
    circles = vec.reshape(n, 3)
    grad = np.zeros_like(vec)
    for i in range(n):
        grad[3*i+2] = -1.0
    for i in range(n):
        x, y, r = circles[i]
        ix, iy, ir = 3*i, 3*i+1, 3*i+2
        if r > x:
            grad[ir] += lam*2*(r-x); grad[ix] -= lam*2*(r-x)
        if x+r > 1:
            grad[ix] += lam*2*(x+r-1); grad[ir] += lam*2*(x+r-1)
        if r > y:
            grad[ir] += lam*2*(r-y); grad[iy] -= lam*2*(r-y)
        if y+r > 1:
            grad[iy] += lam*2*(y+r-1); grad[ir] += lam*2*(y+r-1)
        if r < 0:
            grad[ir] -= lam*2*r
    for i in range(n):
        xi, yi, ri = circles[i]
        for j in range(i+1, n):
            xj, yj, rj = circles[j]
            dx, dy = xi-xj, yi-yj
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < 1e-15: continue
            overlap = (ri+rj) - dist
            if overlap > 0:
                ddx, ddy = dx/dist, dy/dist
                grad[3*i] -= lam*2*overlap*ddx
                grad[3*i+1] -= lam*2*overlap*ddy
                grad[3*i+2] += lam*2*overlap
                grad[3*j] += lam*2*overlap*ddx
                grad[3*j+1] += lam*2*overlap*ddy
                grad[3*j+2] += lam*2*overlap
    return grad

class Perturbation:
    def __init__(self, n, stepsize=0.01, seed=42):
        self.n = n
        self.stepsize = stepsize
        self.rng = np.random.RandomState(seed)

    def __call__(self, x):
        xnew = x.copy()
        mode = self.rng.randint(5)
        if mode == 0:  # global small perturbation
            xnew += self.rng.normal(0, self.stepsize, len(xnew))
        elif mode == 1:  # single circle big move
            i = self.rng.randint(self.n)
            xnew[3*i:3*i+3] += self.rng.normal(0, self.stepsize*5, 3)
        elif mode == 2:  # swap two circles
            i, j = self.rng.choice(self.n, 2, replace=False)
            xnew[3*i:3*i+3], xnew[3*j:3*j+3] = xnew[3*j:3*j+3].copy(), xnew[3*i:3*i+3].copy()
        elif mode == 3:  # radius redistribution
            for i in range(self.n):
                xnew[3*i+2] += self.rng.normal(0, self.stepsize*0.3)
        elif mode == 4:  # move 3 random circles
            for _ in range(3):
                i = self.rng.randint(self.n)
                xnew[3*i:3*i+2] += self.rng.normal(0, self.stepsize*2, 2)
        # Clamp
        for i in range(self.n):
            r = max(1e-5, min(0.49, xnew[3*i+2]))
            xnew[3*i+2] = r
            xnew[3*i] = np.clip(xnew[3*i], r+1e-6, 1-r-1e-6)
            xnew[3*i+1] = np.clip(xnew[3*i+1], r+1e-6, 1-r-1e-6)
        return xnew

# Load
circles = load(INPUT)
n = len(circles)
initial_metric = np.sum(circles[:, 2])
print(f"Initial: metric={initial_metric:.10f}", flush=True)

best = circles.copy()
best_metric = initial_metric

# Basin-hopping with L-BFGS-B penalty
print("\n--- Basin-Hopping ---", flush=True)
configs = [
    (42, 0.3, 0.01, 80),
    (123, 0.5, 0.02, 80),
    (456, 0.1, 0.005, 80),
    (789, 1.0, 0.03, 60),
    (1337, 0.2, 0.015, 80),
]

for seed, temp, ss, niter in configs:
    lam = 1e8
    vec = best.flatten()
    bounds = []
    for i in range(n):
        bounds.extend([(1e-6, 1-1e-6), (1e-6, 1-1e-6), (1e-6, 0.5)])

    minimizer_kwargs = {
        'method': 'L-BFGS-B',
        'jac': lambda v, n=n, lam=lam: penalty_grad(v, n, lam),
        'args': (n, lam),
        'bounds': bounds,
        'options': {'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12}
    }

    perturb = Perturbation(n, ss, seed)
    result = basinhopping(
        penalty_obj, vec,
        minimizer_kwargs=minimizer_kwargs,
        niter=niter, T=temp, take_step=perturb, seed=seed, disp=False
    )

    trial = result.x.reshape(n, 3)
    # Polish with SLSQP
    trial, met, val, viol = slsqp_refine(trial)
    tag = f"seed={seed} T={temp} ss={ss}"
    if val and met > best_metric:
        best = trial
        best_metric = met
        print(f"  BH {tag}: metric={met:.10f} NEW BEST!", flush=True)
    else:
        print(f"  BH {tag}: metric={met:.10f} valid={val}", flush=True)

save(best, OUTPUT)
print(f"\nFinal: metric={best_metric:.10f}", flush=True)
print(f"Improvement: {best_metric - initial_metric:+.2e}", flush=True)
