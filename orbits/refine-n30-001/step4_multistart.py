"""Step 4: Multi-start from diverse topologies for n=30.
Try hex grids, concentric rings, random constructive placement.
Then refine each with progressive penalty + SLSQP.
"""
import json, math, sys, time
import numpy as np
from scipy.optimize import minimize
from pathlib import Path

WORK = Path(__file__).parent
BEST_PATH = WORK / "solution_n30.json"

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
        if r > x: grad[ir] += lam*2*(r-x); grad[ix] -= lam*2*(r-x)
        if x+r > 1: grad[ix] += lam*2*(x+r-1); grad[ir] += lam*2*(x+r-1)
        if r > y: grad[ir] += lam*2*(r-y); grad[iy] -= lam*2*(r-y)
        if y+r > 1: grad[iy] += lam*2*(y+r-1); grad[ir] += lam*2*(y+r-1)
        if r < 0: grad[ir] -= lam*2*r
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
                grad[3*i] -= lam*2*overlap*ddx; grad[3*i+1] -= lam*2*overlap*ddy
                grad[3*i+2] += lam*2*overlap
                grad[3*j] += lam*2*overlap*ddx; grad[3*j+1] += lam*2*overlap*ddy
                grad[3*j+2] += lam*2*overlap
    return grad

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

def progressive_optimize(circles):
    """Progressive penalty + SLSQP polish."""
    n = len(circles)
    vec = circles.flatten()
    bounds = [(1e-6, 1-1e-6) if i%3 != 2 else (1e-6, 0.5) for i in range(3*n)]

    for lam in [10, 100, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]:
        result = minimize(
            penalty_obj, vec, args=(n, lam),
            jac=lambda v, n=n, lam=lam: penalty_grad(v, n, lam),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12}
        )
        vec = result.x

    circles_out = vec.reshape(n, 3)
    # SLSQP polish
    circles_out, met, val, viol = slsqp_refine(circles_out)
    return circles_out, met, val, viol

def hex_init(n, noise=0.0, seed=0):
    """Hex grid initialization."""
    rng = np.random.RandomState(seed)
    # Estimate grid size
    cols = int(math.ceil(math.sqrt(n * 2/math.sqrt(3))))
    rows = int(math.ceil(n / cols))
    r_est = 0.5 / (cols + 0.5)

    circles = []
    for row in range(rows + 2):
        for col in range(cols + 2):
            if len(circles) >= n:
                break
            x = (col + 0.5 * (row % 2)) / (cols + 0.5) + 0.5/(cols+0.5)
            y = row * math.sqrt(3)/2 / (rows + 0.5) + 0.5/(rows+0.5)
            if 0.02 < x < 0.98 and 0.02 < y < 0.98:
                circles.append([x, y, r_est * 0.8])
        if len(circles) >= n:
            break

    # Pad if needed
    while len(circles) < n:
        circles.append([rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9), 0.01])

    circles = np.array(circles[:n])
    if noise > 0:
        circles[:, :2] += rng.normal(0, noise, (n, 2))
        circles[:, 0] = np.clip(circles[:, 0], 0.02, 0.98)
        circles[:, 1] = np.clip(circles[:, 1], 0.02, 0.98)
    return circles

def random_greedy_init(n, seed=0):
    """Greedy constructive placement."""
    rng = np.random.RandomState(seed)
    circles = []

    for _ in range(n):
        best_x, best_y, best_r = 0.5, 0.5, 0.001
        # Try many random positions
        for _ in range(500):
            x = rng.uniform(0.02, 0.98)
            y = rng.uniform(0.02, 0.98)
            r = min(x, 1-x, y, 1-y)
            for cx, cy, cr in circles:
                dist = math.sqrt((x-cx)**2 + (y-cy)**2)
                r = min(r, dist - cr)
            if r > best_r:
                best_x, best_y, best_r = x, y, r
        circles.append([best_x, best_y, max(best_r, 0.001)])

    return np.array(circles)

def ring_init(n, seed=0):
    """Concentric rings initialization."""
    rng = np.random.RandomState(seed)
    # Place circles on concentric rings
    circles = []
    rings = [1, 6, 12, 11]  # center + rings, adjust to sum to n
    while sum(rings) < n:
        rings[-1] += 1
    while sum(rings) > n:
        if rings[-1] > 1:
            rings[-1] -= 1
        else:
            rings.pop()

    cx, cy = 0.5, 0.5
    radii = np.linspace(0, 0.4, len(rings) + 1)[1:]

    for ring_idx, count in enumerate(rings):
        r_ring = radii[ring_idx] if ring_idx > 0 else 0
        for j in range(count):
            angle = 2 * math.pi * j / count + rng.normal(0, 0.02)
            if ring_idx == 0:
                x, y = cx, cy
            else:
                x = cx + r_ring * math.cos(angle)
                y = cy + r_ring * math.sin(angle)
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            circles.append([x, y, 0.03])

    return np.array(circles[:n])

# Load current best
current_best = load(BEST_PATH)
best_metric = np.sum(current_best[:, 2])
best_circles = current_best.copy()
n = 30

print(f"Current best: {best_metric:.10f}", flush=True)
print(f"\n--- Multi-Start Optimization ---", flush=True)

total_starts = 0
t0 = time.time()

# Try diverse initializations
inits = []
for seed in range(20):
    inits.append(("hex", hex_init(n, noise=0.01*seed, seed=seed)))
for seed in range(20):
    inits.append(("greedy", random_greedy_init(n, seed=seed)))
for seed in range(10):
    inits.append(("ring", ring_init(n, seed=seed)))

for name, init_circles in inits:
    total_starts += 1
    if time.time() - t0 > 420:  # 7 minute timeout
        print(f"Time limit reached after {total_starts} starts", flush=True)
        break

    try:
        trial, met, val, viol = progressive_optimize(init_circles)
        if val and met > best_metric:
            best_circles = trial
            best_metric = met
            print(f"  {name} #{total_starts}: metric={met:.10f} NEW BEST!", flush=True)
            save(best_circles, BEST_PATH)
        elif total_starts % 10 == 0:
            print(f"  {name} #{total_starts}: metric={met:.10f} valid={val} (elapsed {time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        pass

# Also try perturbing current best more aggressively
print(f"\n--- Aggressive Perturbations ---", flush=True)
rng = np.random.RandomState(2024)
for trial_idx in range(30):
    if time.time() - t0 > 540:  # 9 min
        break
    perturbed = best_circles.copy()
    # Random perturbation strength
    strength = rng.uniform(0.01, 0.1)
    perturbed[:, :2] += rng.normal(0, strength, (n, 2))
    perturbed[:, 2] += rng.normal(0, strength*0.3, n)
    perturbed[:, 2] = np.maximum(perturbed[:, 2], 0.001)
    perturbed[:, 0] = np.clip(perturbed[:, 0], perturbed[:, 2]+1e-4, 1-perturbed[:, 2]-1e-4)
    perturbed[:, 1] = np.clip(perturbed[:, 1], perturbed[:, 2]+1e-4, 1-perturbed[:, 2]-1e-4)

    try:
        trial, met, val, viol = progressive_optimize(perturbed)
        if val and met > best_metric:
            best_circles = trial
            best_metric = met
            print(f"  Perturb #{trial_idx} s={strength:.3f}: metric={met:.10f} NEW BEST!", flush=True)
            save(best_circles, BEST_PATH)
        elif trial_idx % 10 == 0:
            print(f"  Perturb #{trial_idx}: metric={met:.10f} valid={val}", flush=True)
    except Exception:
        pass

save(best_circles, BEST_PATH)
elapsed = time.time() - t0
print(f"\nFinal: metric={best_metric:.10f} ({total_starts} starts, {elapsed:.0f}s)", flush=True)
print(f"Improvement: {best_metric - np.sum(current_best[:,2]):+.2e}", flush=True)
