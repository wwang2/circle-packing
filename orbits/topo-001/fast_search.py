"""
Fast Topology Search: Use penalty-based optimization (L-BFGS-B) for quick
exploration, then polish winners with SLSQP.
Much faster than pure SLSQP for the exploration phase.
"""

import json
import numpy as np
from scipy.optimize import minimize
import os
import time

WORKDIR = os.path.dirname(os.path.abspath(__file__))
N = 26

def load_solution(path):
    with open(path) as f:
        data = json.load(f)
    circles = np.array(data["circles"])
    return circles[:, 0], circles[:, 1], circles[:, 2]

def save_solution(x, y, r, path):
    circles = [[float(x[i]), float(y[i]), float(r[i])] for i in range(len(x))]
    with open(path, 'w') as f:
        json.dump({"circles": circles}, f, indent=2)

def penalty_objective(params, n, penalty_weight):
    """Penalty-based objective: -sum(r) + penalty * violations."""
    x = params[:n]
    y = params[n:2*n]
    r = params[2*n:3*n]

    obj = -np.sum(r)
    penalty = 0.0

    # Containment penalties
    for i in range(n):
        v = max(0, r[i] - x[i])
        penalty += v**2
        v = max(0, r[i] - (1 - x[i]))
        penalty += v**2
        v = max(0, r[i] - y[i])
        penalty += v**2
        v = max(0, r[i] - (1 - y[i]))
        penalty += v**2
        v = max(0, -r[i])
        penalty += v**2

    # Non-overlap penalties
    for i in range(n):
        for j in range(i+1, n):
            dist2 = (x[i]-x[j])**2 + (y[i]-y[j])**2
            sum_r = r[i] + r[j]
            if dist2 < sum_r**2:
                gap = sum_r - np.sqrt(dist2)
                penalty += gap**2

    return obj + penalty_weight * penalty

def penalty_objective_grad(params, n, penalty_weight):
    """Penalty objective with analytical gradient."""
    x = params[:n]
    y = params[n:2*n]
    r = params[2*n:3*n]

    grad = np.zeros(3*n)
    obj = -np.sum(r)
    grad[2*n:3*n] = -1.0  # d(-sum r)/dr_i = -1
    penalty = 0.0

    # Containment
    for i in range(n):
        # r_i - x_i <= 0
        v = r[i] - x[i]
        if v > 0:
            penalty += v**2
            grad[i] += penalty_weight * (-2*v)      # d/dx_i
            grad[2*n+i] += penalty_weight * (2*v)    # d/dr_i

        # r_i - (1-x_i) <= 0  =>  x_i + r_i - 1 <= 0
        v = x[i] + r[i] - 1
        if v > 0:
            penalty += v**2
            grad[i] += penalty_weight * (2*v)
            grad[2*n+i] += penalty_weight * (2*v)

        # r_i - y_i <= 0
        v = r[i] - y[i]
        if v > 0:
            penalty += v**2
            grad[n+i] += penalty_weight * (-2*v)
            grad[2*n+i] += penalty_weight * (2*v)

        # y_i + r_i - 1 <= 0
        v = y[i] + r[i] - 1
        if v > 0:
            penalty += v**2
            grad[n+i] += penalty_weight * (2*v)
            grad[2*n+i] += penalty_weight * (2*v)

        # -r_i <= 0
        if r[i] < 0:
            penalty += r[i]**2
            grad[2*n+i] += penalty_weight * (2*r[i])

    # Non-overlap
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist2 = dx**2 + dy**2
            sum_r = r[i] + r[j]
            if dist2 < sum_r**2:
                dist = np.sqrt(dist2) if dist2 > 1e-20 else 1e-10
                gap = sum_r - dist
                penalty += gap**2
                # d(gap)/d(x_i) = -(x_i-x_j)/dist = -dx/dist
                dgap_dxi = -dx/dist
                dgap_dxj = dx/dist
                dgap_dyi = -dy/dist
                dgap_dyj = dy/dist
                dgap_dri = 1.0
                dgap_drj = 1.0

                coeff = penalty_weight * 2 * gap
                grad[i] += coeff * dgap_dxi
                grad[j] += coeff * dgap_dxj
                grad[n+i] += coeff * dgap_dyi
                grad[n+j] += coeff * dgap_dyj
                grad[2*n+i] += coeff * dgap_dri
                grad[2*n+j] += coeff * dgap_drj

    return obj + penalty_weight * penalty, grad

def optimize_penalty(x0, y0, r0, maxiter=3000):
    """Progressive penalty optimization with L-BFGS-B."""
    n = len(x0)
    params = np.concatenate([x0, y0, r0])

    bounds = ([(0.001, 0.999)]*n + [(0.001, 0.999)]*n +
              [(0.005, 0.499)]*n)

    for pw in [10, 100, 1000, 10000, 100000]:
        result = minimize(
            lambda p: penalty_objective_grad(p, n, pw),
            params,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-15}
        )
        params = result.x

    x = params[:n]
    y = params[n:2*n]
    r = params[2*n:3*n]
    return x, y, r

def optimize_slsqp(x0, y0, r0, maxiter=8000):
    """SLSQP polish."""
    n = len(x0)
    params0 = np.concatenate([x0, y0, r0])

    constraints = []
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[i] - p[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[i] - p[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[n+i] - p[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[n+i] - p[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[2*n+i] - 1e-6})

    for i in range(n):
        for j in range(i+1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: (
                    (p[i]-p[j])**2 + (p[n+i]-p[n+j])**2 - (p[2*n+i]+p[2*n+j])**2
                )
            })

    result = minimize(
        lambda p: -np.sum(p[2*n:3*n]),
        params0,
        method='SLSQP',
        constraints=constraints,
        options={'maxiter': maxiter, 'ftol': 1e-15, 'disp': False}
    )

    x = result.x[:n]
    y = result.x[n:2*n]
    r = result.x[2*n:3*n]
    return x, y, r, np.sum(r), result.success

def is_feasible(x, y, r, tol=1e-8):
    n = len(x)
    for i in range(n):
        if r[i] < -tol: return False
        if x[i] - r[i] < -tol or 1 - x[i] - r[i] < -tol: return False
        if y[i] - r[i] < -tol or 1 - y[i] - r[i] < -tol: return False
    for i in range(n):
        for j in range(i+1, n):
            if (x[i]-x[j])**2 + (y[i]-y[j])**2 < (r[i]+r[j])**2 - tol:
                return False
    return True

# ---- Initialization generators ----

def init_ring(n, seed):
    """OpenEvolve-style concentric ring."""
    rng = np.random.RandomState(seed)
    x, y, r = [], [], []

    # Center
    x.append(0.5); y.append(0.5); r.append(0.14)

    # Inner ring
    n_inner = 8
    for i in range(n_inner):
        angle = 2*np.pi*i/n_inner + rng.uniform(-0.15, 0.15)
        rad = 0.22 + rng.uniform(-0.02, 0.02)
        x.append(0.5 + rad*np.cos(angle))
        y.append(0.5 + rad*np.sin(angle))
        r.append(0.105 + rng.uniform(-0.02, 0.02))

    # Middle ring
    n_mid = n - 1 - n_inner - 4
    for i in range(n_mid):
        angle = 2*np.pi*i/n_mid + rng.uniform(-0.15, 0.15)
        rad = 0.40 + rng.uniform(-0.03, 0.03)
        x.append(0.5 + rad*np.cos(angle))
        y.append(0.5 + rad*np.sin(angle))
        r.append(0.09 + rng.uniform(-0.02, 0.02))

    # Corners
    for cx, cy in [(0.09, 0.09), (0.91, 0.09), (0.09, 0.91), (0.91, 0.91)]:
        x.append(cx); y.append(cy); r.append(0.085)

    x, y, r = np.array(x[:n]), np.array(y[:n]), np.array(r[:n])
    r = np.maximum(r, 0.02)
    x = np.clip(x, r+0.001, 1-r-0.001)
    y = np.clip(y, r+0.001, 1-r-0.001)
    return x, y, r

def init_hex(n, seed):
    rng = np.random.RandomState(seed)
    r_est = 1.0 / (2 * np.sqrt(n/0.9069))  # hex packing fraction
    positions = []
    row = 0
    yy = r_est
    while yy < 1 - r_est:
        xx = r_est + (row % 2) * r_est
        while xx < 1 - r_est:
            positions.append((xx, yy))
            xx += 2 * r_est
        yy += r_est * np.sqrt(3)
        row += 1
    rng.shuffle(positions)
    positions = positions[:n]
    while len(positions) < n:
        positions.append((rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)))
    x = np.array([p[0] for p in positions]) + rng.normal(0, 0.01, n)
    y = np.array([p[1] for p in positions]) + rng.normal(0, 0.01, n)
    r = np.full(n, r_est * 0.85) + rng.uniform(-0.005, 0.005, n)
    r = np.maximum(r, 0.02)
    x = np.clip(x, r+0.001, 1-r-0.001)
    y = np.clip(y, r+0.001, 1-r-0.001)
    return x, y, r

def init_random(n, seed):
    rng = np.random.RandomState(seed)
    x = rng.uniform(0.08, 0.92, n)
    y = rng.uniform(0.08, 0.92, n)
    r = np.full(n, 0.05) + rng.uniform(-0.01, 0.02, n)
    r = np.maximum(r, 0.02)
    x = np.clip(x, r+0.001, 1-r-0.001)
    y = np.clip(y, r+0.001, 1-r-0.001)
    return x, y, r

def init_perturbed_parent(x0, y0, r0, scale, seed):
    rng = np.random.RandomState(seed)
    n = len(x0)
    x = x0 + rng.normal(0, scale, n)
    y = y0 + rng.normal(0, scale, n)
    r = r0 * (1 + rng.normal(0, scale*0.3, n))
    r = np.maximum(r, 0.02)
    x = np.clip(x, r+0.001, 1-r-0.001)
    y = np.clip(y, r+0.001, 1-r-0.001)
    return x, y, r

def init_modified_ring(n, n_inner, n_mid, seed):
    """Ring with different number of circles per ring."""
    rng = np.random.RandomState(seed)
    x, y, r = [], [], []

    # Center
    x.append(0.5 + rng.normal(0, 0.01))
    y.append(0.5 + rng.normal(0, 0.01))
    r.append(0.14 + rng.uniform(-0.02, 0.02))

    # Inner ring
    r_ring1 = 0.22 + rng.uniform(-0.03, 0.03)
    for i in range(n_inner):
        angle = 2*np.pi*i/n_inner + rng.uniform(-0.2, 0.2)
        x.append(0.5 + r_ring1*np.cos(angle))
        y.append(0.5 + r_ring1*np.sin(angle))
        r.append(0.10 + rng.uniform(-0.02, 0.02))

    # Middle ring
    actual_mid = min(n_mid, n - 1 - n_inner - 4)
    r_ring2 = 0.40 + rng.uniform(-0.04, 0.04)
    for i in range(max(0, actual_mid)):
        angle = 2*np.pi*i/max(actual_mid, 1) + rng.uniform(-0.2, 0.2)
        x.append(0.5 + r_ring2*np.cos(angle))
        y.append(0.5 + r_ring2*np.sin(angle))
        r.append(0.09 + rng.uniform(-0.02, 0.02))

    # Corners
    corners = [(0.09, 0.09), (0.91, 0.09), (0.09, 0.91), (0.91, 0.91)]
    for cx, cy in corners:
        if len(x) >= n:
            break
        x.append(cx + rng.normal(0, 0.01))
        y.append(cy + rng.normal(0, 0.01))
        r.append(0.085 + rng.uniform(-0.01, 0.01))

    # Fill remainder
    while len(x) < n:
        x.append(rng.uniform(0.15, 0.85))
        y.append(rng.uniform(0.15, 0.85))
        r.append(0.06)

    x, y, r = np.array(x[:n]), np.array(y[:n]), np.array(r[:n])
    r = np.maximum(r, 0.02)
    x = np.clip(x, r+0.001, 1-r-0.001)
    y = np.clip(y, r+0.001, 1-r-0.001)
    return x, y, r

def main():
    t0 = time.time()
    parent_path = os.path.join(WORKDIR, '..', 'nlp-001', 'solution_n26.json')
    x0, y0, r0 = load_solution(parent_path)
    parent_metric = np.sum(r0)
    print(f"Parent metric: {parent_metric:.10f}")

    best_metric = parent_metric
    best_x, best_y, best_r = x0.copy(), y0.copy(), r0.copy()

    candidates = []  # (metric_approx, x, y, r, name)

    # ====== Phase 1: Fast penalty-based exploration ======
    print("\n=== Phase 1: Fast Penalty Exploration ===")

    inits = []

    # Ring variants
    for seed in range(40):
        inits.append(('ring', seed, lambda s=seed: init_ring(N, s)))

    # Modified rings with different ring sizes
    for n_inner in [6, 7, 8, 9, 10]:
        for n_mid in [8, 9, 10, 11, 12, 13]:
            for seed in range(3):
                inits.append(('mring', f"{n_inner}_{n_mid}_s{seed}",
                             lambda ni=n_inner, nm=n_mid, s=seed: init_modified_ring(N, ni, nm, s)))

    # Hex
    for seed in range(30):
        inits.append(('hex', seed, lambda s=seed: init_hex(N, s)))

    # Random
    for seed in range(50):
        inits.append(('random', seed, lambda s=seed: init_random(N, s)))

    # Perturbed parent
    for scale in [0.02, 0.05, 0.1, 0.15, 0.2, 0.3]:
        for seed in range(15):
            inits.append(('perturb', f"s{scale}_sd{seed}",
                         lambda sc=scale, sd=seed: init_perturbed_parent(x0, y0, r0, sc, sd)))

    print(f"Total initializations: {len(inits)}")

    for idx, (name, param, gen_fn) in enumerate(inits):
        x, y, r = gen_fn()

        # Fast penalty optimization
        x, y, r = optimize_penalty(x, y, r, maxiter=2000)
        metric = np.sum(r) if is_feasible(x, y, r, tol=1e-4) else 0

        # Keep promising candidates for SLSQP polish
        if metric > 2.5:
            candidates.append((metric, x.copy(), y.copy(), r.copy(), f"{name}({param})"))

        if idx % 50 == 0:
            elapsed = time.time() - t0
            n_good = len([c for c in candidates if c[0] > 2.60])
            print(f"  [{idx}/{len(inits)}] {elapsed:.0f}s, candidates>2.60: {n_good}")

    # Sort candidates
    candidates.sort(key=lambda c: c[0], reverse=True)
    print(f"\nTop 15 penalty candidates:")
    for metric, _, _, _, name in candidates[:15]:
        print(f"  {name}: {metric:.8f}")

    # ====== Phase 2: SLSQP polish of top candidates ======
    print(f"\n=== Phase 2: SLSQP Polish (top {min(40, len(candidates))} candidates) ===")

    for idx, (metric_approx, x, y, r, name) in enumerate(candidates[:40]):
        x2, y2, r2, metric, success = optimize_slsqp(x, y, r, maxiter=10000)

        if success and is_feasible(x2, y2, r2) and metric > best_metric:
            print(f"  IMPROVED: {name} -> {metric:.10f} (+{metric-parent_metric:.2e})")
            best_metric = metric
            best_x, best_y, best_r = x2.copy(), y2.copy(), r2.copy()
            save_solution(best_x, best_y, best_r,
                        os.path.join(WORKDIR, 'solution_n26.json'))

        if idx % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{idx}/{min(40, len(candidates))}] {elapsed:.0f}s, best={best_metric:.10f}")

    # ====== Phase 3: Deep basin hopping from best ======
    print(f"\n=== Phase 3: Basin Hopping from best ({best_metric:.10f}) ===")
    rng = np.random.RandomState(999)

    no_improve = 0
    for hop in range(100):
        x2, y2, r2 = best_x.copy(), best_y.copy(), best_r.copy()

        # Perturbation
        ptype = rng.randint(0, 6)
        if ptype == 0:
            # Displace 1-3 circles
            nc = rng.randint(1, 4)
            for _ in range(nc):
                i = rng.randint(N)
                x2[i] += rng.normal(0, 0.05)
                y2[i] += rng.normal(0, 0.05)
                r2[i] *= rng.uniform(0.8, 1.2)
        elif ptype == 1:
            # Swap 2
            i, j = rng.choice(N, 2, replace=False)
            x2[i], x2[j] = x2[j], x2[i]
            y2[i], y2[j] = y2[j], y2[i]
        elif ptype == 2:
            # Global noise
            scale = rng.uniform(0.02, 0.1)
            x2 += rng.normal(0, scale, N)
            y2 += rng.normal(0, scale, N)
        elif ptype == 3:
            # Rotate subset
            k = rng.randint(3, N//2)
            idxs = rng.choice(N, k, replace=False)
            angle = rng.uniform(-0.3, 0.3)
            cx, cy = np.mean(x2[idxs]), np.mean(y2[idxs])
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            for idx in idxs:
                dx, dy = x2[idx]-cx, y2[idx]-cy
                x2[idx] = cx + cos_a*dx - sin_a*dy
                y2[idx] = cy + sin_a*dx + cos_a*dy
        elif ptype == 4:
            # Shrink one large, grow neighbors
            i = rng.choice(np.argsort(r2)[-8:])
            r2[i] *= 0.7
            # Find neighbors
            dists = np.sqrt((x2-x2[i])**2 + (y2-y2[i])**2)
            neighbors = np.argsort(dists)[1:4]
            for j in neighbors:
                r2[j] *= 1.1
        elif ptype == 5:
            # Move smallest to random position
            i = np.argmin(r2)
            x2[i] = rng.uniform(0.1, 0.9)
            y2[i] = rng.uniform(0.1, 0.9)
            r2[i] = rng.uniform(0.05, 0.12)

        r2 = np.maximum(r2, 0.01)
        x2 = np.clip(x2, r2+0.001, 1-r2-0.001)
        y2 = np.clip(y2, r2+0.001, 1-r2-0.001)

        # Penalty then SLSQP
        x2, y2, r2 = optimize_penalty(x2, y2, r2, maxiter=1500)
        x2, y2, r2, metric, success = optimize_slsqp(x2, y2, r2, maxiter=5000)

        if success and is_feasible(x2, y2, r2) and metric > best_metric:
            print(f"  Hop {hop}: IMPROVED {best_metric:.10f} -> {metric:.10f} (type={ptype})")
            best_metric = metric
            best_x, best_y, best_r = x2.copy(), y2.copy(), r2.copy()
            save_solution(best_x, best_y, best_r,
                        os.path.join(WORKDIR, 'solution_n26.json'))
            no_improve = 0
        else:
            no_improve += 1

        if hop % 20 == 0:
            elapsed = time.time() - t0
            print(f"  Hop {hop}/100, {elapsed:.0f}s, best={best_metric:.10f}, no_improve={no_improve}")

    # Final save
    save_solution(best_x, best_y, best_r, os.path.join(WORKDIR, 'solution_n26.json'))

    elapsed = time.time() - t0
    print(f"\n=== FINAL ===")
    print(f"Parent:  {parent_metric:.10f}")
    print(f"Best:    {best_metric:.10f}")
    print(f"Delta:   {best_metric - parent_metric:.2e}")
    print(f"Time:    {elapsed:.0f}s")

    return best_metric

if __name__ == '__main__':
    main()
