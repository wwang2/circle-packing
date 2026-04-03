"""
Mobius-001 v2: Focused search with working optimizer.

Key fixes:
1. Use greedy constructive initialization (guaranteed feasible start)
2. Smooth penalty schedule with more stages
3. SLSQP polish on every promising solution
4. Mobius deformations as topology-changing moves
"""

import json
import math
import numpy as np
from scipy.optimize import minimize
from pathlib import Path
import time
import sys

SEED = 42
N = 26
WORKTREE = Path("/Users/wujiewang/code/circle-packing/.worktrees/mobius-001")
OUTPUT_DIR = WORKTREE / "orbits/mobius-001"

def load_solution(path):
    with open(path) as f:
        data = json.load(f)
    return np.array(data["circles"])

def save_solution(circles, path):
    data = {"circles": [[float(c[0]), float(c[1]), float(c[2])] for c in circles]}
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def sum_radii(c): return float(np.sum(c[:, 2]))

def validate(circles, tol=1e-10):
    n = len(circles)
    mv = 0.0
    for i in range(n):
        x, y, r = circles[i]
        if r <= 0: return False, abs(r)
        mv = max(mv, r-x, x+r-1, r-y, y+r-1)
    for i in range(n):
        for j in range(i+1, n):
            dx = circles[i,0]-circles[j,0]; dy = circles[i,1]-circles[j,1]
            d = math.sqrt(dx*dx+dy*dy)
            mv = max(mv, circles[i,2]+circles[j,2]-d)
    return mv <= tol, mv


def greedy_init(n, rng, radii=None):
    """Greedy constructive: place circles one by one, largest first."""
    if radii is None:
        radii = rng.uniform(0.04, 0.14, n)
    radii = np.sort(radii)[::-1]  # largest first

    circles = np.zeros((n, 3))
    placed = 0

    for idx in range(n):
        r = radii[idx]
        best_score = -1
        best_pos = None

        for _ in range(200):
            x = rng.uniform(r + 0.001, 1 - r - 0.001)
            y = rng.uniform(r + 0.001, 1 - r - 0.001)

            # Check overlap with placed circles
            ok = True
            min_gap = float('inf')
            for j in range(placed):
                dx = x - circles[j, 0]; dy = y - circles[j, 1]
                gap = math.sqrt(dx*dx + dy*dy) - r - circles[j, 2]
                if gap < -1e-6:
                    ok = False; break
                min_gap = min(min_gap, gap)

            if ok:
                # Score: prefer positions that leave room but are snug
                wall_gap = min(x - r, 1 - x - r, y - r, 1 - y - r)
                score = min_gap if min_gap < float('inf') else wall_gap
                if best_pos is None or rng.random() < 0.3:  # some randomness
                    best_pos = (x, y)
                    best_score = score

        if best_pos is None:
            # Force placement
            best_pos = (rng.uniform(r, 1-r), rng.uniform(r, 1-r))

        circles[placed] = [best_pos[0], best_pos[1], r]
        placed += 1

    return circles


def slsqp_optimize(circles, maxiter=8000):
    """Direct SLSQP optimization with vectorized constraints."""
    n = len(circles)
    x0 = circles.flatten()
    nn = 3 * n

    def objective(x):
        return -np.sum(x[2::3])

    def grad_obj(x):
        g = np.zeros(nn)
        g[2::3] = -1.0
        return g

    def all_con(x):
        xs = x[0::3]; ys = x[1::3]; rs = x[2::3]
        vals = []
        vals.extend(xs - rs)
        vals.extend(1.0 - xs - rs)
        vals.extend(ys - rs)
        vals.extend(1.0 - ys - rs)
        vals.extend(rs - 1e-6)
        for i in range(n):
            dx = xs[i] - xs[i+1:]
            dy = ys[i] - ys[i+1:]
            vals.extend(dx*dx + dy*dy - (rs[i] + rs[i+1:])**2)
        return np.array(vals)

    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * n
    result = minimize(objective, x0, method='SLSQP', jac=grad_obj,
                     bounds=bounds, constraints=[{'type': 'ineq', 'fun': all_con}],
                     options={'maxiter': maxiter, 'ftol': 1e-15})
    out = result.x.reshape(n, 3)
    valid, viol = validate(out)
    return out, -result.fun if valid else 0.0, valid


def penalty_optimize(circles):
    """Smooth penalty optimization."""
    n = len(circles)
    x = circles.flatten().copy()
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-5, 0.5)] * n

    stages = [0.1, 1, 5, 20, 100, 500, 2000, 10000, 50000, 200000, 1e6]

    for pw in stages:
        def obj_grad(x, pw=pw):
            xs = x[0::3]; ys = x[1::3]; rs = x[2::3]
            obj = -np.sum(rs)
            grad = np.zeros_like(x)
            grad[2::3] = -1.0

            vl = np.maximum(0, rs-xs); vr = np.maximum(0, xs+rs-1)
            vb = np.maximum(0, rs-ys); vt = np.maximum(0, ys+rs-1)
            obj += pw * np.sum(vl**2 + vr**2 + vb**2 + vt**2)
            grad[0::3] += pw * (-2*vl + 2*vr)
            grad[1::3] += pw * (-2*vb + 2*vt)
            grad[2::3] += pw * (2*vl + 2*vr + 2*vb + 2*vt)

            for i in range(n):
                js = np.arange(i+1, n)
                if len(js) == 0: continue
                dx = xs[i]-xs[js]; dy = ys[i]-ys[js]
                dsq = dx*dx+dy*dy
                md = rs[i]+rs[js]; mdsq = md**2
                active = mdsq > dsq
                if not np.any(active): continue
                a_js = js[active]
                factor = mdsq[active] - dsq[active]
                obj += pw * np.sum(factor**2)
                f2 = 2*factor
                a_dx = dx[active]; a_dy = dy[active]; a_md = md[active]
                grad[3*i] += pw * np.sum(f2*(-2*a_dx))
                grad[3*i+1] += pw * np.sum(f2*(-2*a_dy))
                for k, j in enumerate(a_js):
                    grad[3*j] += pw * f2[k] * 2 * a_dx[k]
                    grad[3*j+1] += pw * f2[k] * 2 * a_dy[k]
                    grad[3*i+2] += pw * f2[k] * 2 * a_md[k]
                    grad[3*j+2] += pw * f2[k] * 2 * a_md[k]
            return obj, grad

        result = minimize(obj_grad, x, jac=True, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 2000, 'ftol': 1e-15})
        x = result.x.copy()

    out = x.reshape(n, 3)
    valid, viol = validate(out, tol=1e-8)
    return out, sum_radii(out) if valid else 0.0, valid


def full_optimize(init_circles):
    """Full pipeline: penalty -> SLSQP."""
    pen_c, pen_m, pen_v = penalty_optimize(init_circles)
    if not pen_v:
        # Try SLSQP directly from init
        try:
            s_c, s_m, s_v = slsqp_optimize(init_circles, maxiter=5000)
            if s_v:
                return s_c, s_m, s_v
        except:
            pass
        return pen_c, pen_m, pen_v

    # Polish with SLSQP
    try:
        s_c, s_m, s_v = slsqp_optimize(pen_c, maxiter=8000)
        if s_v and s_m >= pen_m:
            return s_c, s_m, s_v
    except:
        pass
    return pen_c, pen_m, pen_v


def apply_mobius_to_circle(center, radius, a, b, c, d):
    det = a*d - b*c
    cz_d = c*center + d
    denom = abs(cz_d)**2 - abs(c)**2 * radius**2
    if abs(denom) < 1e-14: return None, None
    new_center = ((a*center+b)*np.conj(cz_d) - a*np.conj(c)*radius**2) / denom
    new_radius = abs(det) * radius / abs(denom)
    return new_center, new_radius


def mobius_deform(circles, cluster, rng, strength):
    """Apply Mobius deformation to a subset of circles."""
    cx = np.mean(circles[cluster, 0])
    cy = np.mean(circles[cluster, 1])

    a = complex(1+rng.normal(0,strength*0.1), rng.normal(0,strength*0.1))
    b = complex(rng.normal(0,strength*0.05), rng.normal(0,strength*0.05))
    c = complex(rng.normal(0,strength*0.02), rng.normal(0,strength*0.02))
    d = complex(1+rng.normal(0,strength*0.1), rng.normal(0,strength*0.1))
    det = a*d-b*c
    if abs(det) < 1e-10: return None
    s = det**0.5; a/=s; b/=s; c/=s; d/=s

    new = circles.copy()
    for idx in cluster:
        z = complex(circles[idx,0]-cx, circles[idx,1]-cy)
        nz, nr = apply_mobius_to_circle(z, circles[idx,2], a, b, c, d)
        if nz is None or nr is None or nr <= 0: return None
        nx, ny = nz.real+cx, nz.imag+cy
        nr = max(nr, 1e-4)
        nx = max(nr, min(1-nr, nx))
        ny = max(nr, min(1-nr, ny))
        new[idx] = [nx, ny, nr]
    return new


def find_adj(circles, tol=1e-4):
    """Adjacency from contacts."""
    n = len(circles)
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i+1, n):
            dx = circles[i,0]-circles[j,0]; dy = circles[i,1]-circles[j,1]
            gap = math.sqrt(dx*dx+dy*dy) - circles[i,2] - circles[j,2]
            if abs(gap) < tol:
                adj[i].append(j); adj[j].append(i)
    return adj


def main():
    base = load_solution(WORKTREE / "orbits/topo-001/solution_n26.json")
    base_metric = sum_radii(base)
    print(f"Base: {base_metric:.10f}", flush=True)

    rng = np.random.RandomState(SEED)
    best = base_metric
    best_circles = base.copy()
    all_metrics = []

    # ============ Phase 1: Greedy constructive + SLSQP (fast, reliable) ============
    print("\n=== Phase 1: Greedy constructive inits + SLSQP ===", flush=True)
    t0 = time.time()

    size_dists = [
        lambda rng: rng.uniform(0.06, 0.12, N),
        lambda rng: np.concatenate([rng.uniform(0.10, 0.15, 5), rng.uniform(0.05, 0.09, N-5)]),
        lambda rng: np.concatenate([rng.uniform(0.12, 0.18, 3), rng.uniform(0.05, 0.10, N-3)]),
        lambda rng: np.concatenate([[0.15], rng.uniform(0.06, 0.11, N-1)]),
        lambda rng: np.linspace(0.13, 0.05, N) + rng.uniform(-0.01, 0.01, N),
        lambda rng: np.full(N, 0.085) + rng.uniform(-0.01, 0.01, N),
        lambda rng: np.concatenate([rng.uniform(0.11, 0.14, 8), rng.uniform(0.06, 0.09, N-8)]),
        lambda rng: np.concatenate([rng.uniform(0.10, 0.13, 10), rng.uniform(0.05, 0.08, N-10)]),
    ]

    for trial in range(80):
        dist_fn = size_dists[trial % len(size_dists)]
        radii = np.maximum(dist_fn(rng), 0.02)

        init = greedy_init(N, rng, radii)
        try:
            opt, metric, valid = slsqp_optimize(init, maxiter=5000)
            if valid and metric > 0:
                all_metrics.append(metric)
                if metric > best + 1e-12:
                    print(f"  Trial {trial}: {metric:.10f} NEW BEST", flush=True)
                    best = metric
                    best_circles = opt.copy()
            if (trial+1) % 20 == 0:
                print(f"  Trial {trial+1}/80: best={best:.10f}, valid={len(all_metrics)}/{trial+1}",
                      flush=True)
        except Exception as e:
            pass

    print(f"Phase 1: {time.time()-t0:.1f}s, best={best:.10f}, "
          f"valid={len(all_metrics)}/80", flush=True)

    # ============ Phase 2: Penalty + SLSQP with diverse inits ============
    print("\n=== Phase 2: Penalty + SLSQP diverse starts ===", flush=True)
    t0 = time.time()

    for trial in range(60):
        radii = size_dists[trial % len(size_dists)](rng)
        radii = np.maximum(radii, 0.02)
        init = greedy_init(N, rng, radii)

        try:
            opt, metric, valid = full_optimize(init)
            if valid and metric > 0:
                all_metrics.append(metric)
                if metric > best + 1e-12:
                    print(f"  Trial {trial}: {metric:.10f} NEW BEST", flush=True)
                    best = metric
                    best_circles = opt.copy()
            if (trial+1) % 20 == 0:
                print(f"  Trial {trial+1}/60: best={best:.10f}", flush=True)
        except:
            pass

    print(f"Phase 2: {time.time()-t0:.1f}s, best={best:.10f}", flush=True)

    # ============ Phase 3: Mobius deformations on best ============
    print("\n=== Phase 3: Mobius cluster deformations ===", flush=True)
    t0 = time.time()

    adj = find_adj(best_circles)

    for trial in range(200):
        center = rng.randint(N)
        neighbors = adj.get(center, [])
        if len(neighbors) < 1:
            # Use spatial neighbors
            dists = np.sum((best_circles[:, :2] - best_circles[center, :2])**2, axis=1)
            dists[center] = float('inf')
            neighbors = list(np.argsort(dists)[:5])

        k = min(rng.randint(2, 6), len(neighbors))
        cluster = [center] + list(rng.choice(neighbors, k, replace=False))

        strength = rng.choice([0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0])
        deformed = mobius_deform(best_circles, cluster, rng, strength)
        if deformed is None: continue

        try:
            opt, metric, valid = slsqp_optimize(deformed, maxiter=5000)
            if valid and metric > best + 1e-12:
                print(f"  Mobius {trial}: {best:.10f} -> {metric:.10f} "
                      f"(str={strength})", flush=True)
                best = metric
                best_circles = opt.copy()
                adj = find_adj(best_circles)
        except:
            pass

        if (trial+1) % 50 == 0:
            print(f"  Mobius {trial+1}/200: best={best:.10f}", flush=True)

    print(f"Phase 3: {time.time()-t0:.1f}s, best={best:.10f}", flush=True)

    # ============ Phase 4: Large Mobius (half the packing) ============
    print("\n=== Phase 4: Large Mobius deformations ===", flush=True)
    t0 = time.time()

    for trial in range(100):
        # Select half the circles
        k = rng.randint(8, 18)
        cluster = list(rng.choice(N, k, replace=False))
        strength = rng.choice([0.05, 0.1, 0.2, 0.5, 1.0])

        deformed = mobius_deform(best_circles, cluster, rng, strength)
        if deformed is None: continue

        try:
            opt, metric, valid = slsqp_optimize(deformed, maxiter=5000)
            if valid and metric > best + 1e-12:
                print(f"  Large Mobius {trial}: {best:.10f} -> {metric:.10f}", flush=True)
                best = metric
                best_circles = opt.copy()
        except:
            pass

        if (trial+1) % 50 == 0:
            print(f"  Large Mobius {trial+1}/100: best={best:.10f}", flush=True)

    print(f"Phase 4: {time.time()-t0:.1f}s, best={best:.10f}", flush=True)

    # ============ Phase 5: Inversive distance perturbation ============
    print("\n=== Phase 5: Inversive distance perturbation ===", flush=True)
    t0 = time.time()

    for trial in range(100):
        new = best_circles.copy()

        # Shuffle radii while keeping rough positions
        if rng.random() < 0.3:
            # Partial radius shuffle
            k = rng.randint(3, 10)
            indices = rng.choice(N, k, replace=False)
            shuffled_radii = new[indices, 2].copy()
            rng.shuffle(shuffled_radii)
            new[indices, 2] = shuffled_radii
        elif rng.random() < 0.5:
            # Scale radii non-uniformly
            scales = 1.0 + rng.normal(0, 0.05, N)
            new[:, 2] *= np.maximum(scales, 0.5)
        else:
            # Perturb positions
            new[:, 0] += rng.normal(0, 0.02, N)
            new[:, 1] += rng.normal(0, 0.02, N)

        # Clamp
        for i in range(N):
            new[i, 2] = max(new[i, 2], 0.02)
            new[i, 0] = max(new[i, 2], min(1-new[i, 2], new[i, 0]))
            new[i, 1] = max(new[i, 2], min(1-new[i, 2], new[i, 1]))

        try:
            opt, metric, valid = slsqp_optimize(new, maxiter=5000)
            if valid and metric > best + 1e-12:
                print(f"  Inversive {trial}: {best:.10f} -> {metric:.10f}", flush=True)
                best = metric
                best_circles = opt.copy()
        except:
            pass

        if (trial+1) % 50 == 0:
            print(f"  Inversive {trial+1}/100: best={best:.10f}", flush=True)

    print(f"Phase 5: {time.time()-t0:.1f}s, best={best:.10f}", flush=True)

    # ============ Final polish ============
    print("\n=== Final SLSQP polish ===", flush=True)
    try:
        polished, pm, pv = slsqp_optimize(best_circles, maxiter=20000)
        if pv and pm >= best:
            best = pm
            best_circles = polished
    except:
        pass

    valid, viol = validate(best_circles)
    print(f"\nFINAL: {best:.10f} (valid={valid}, viol={viol:.2e})", flush=True)
    print(f"Improvement over base: {best - base_metric:.2e}", flush=True)

    save_solution(best_circles, OUTPUT_DIR / "solution_n26.json")

    # Save all metrics
    with open(OUTPUT_DIR / "search_metrics.json", 'w') as f:
        json.dump({
            'all_metrics': sorted(all_metrics, reverse=True)[:100],
            'best': best,
            'base': base_metric,
            'n_valid': len(all_metrics)
        }, f, indent=2)

    print(f"\nSaved. Valid solutions: {len(all_metrics)}", flush=True)


if __name__ == "__main__":
    main()
