"""
Aggressive topology search using:
1. Massive random multi-start (1000+ trials, penalty only for speed)
2. Conformal disk mapping
3. Systematic size distribution variation
4. Genetic/evolutionary mixing of top solutions
"""

import json
import math
import numpy as np
from scipy.optimize import minimize
from scipy.special import ellipk, ellipkinc
from pathlib import Path
import time

SEED = 42
N = 26
WORKTREE = Path("/Users/wujiewang/code/circle-packing/.worktrees/mobius-001")
OUTPUT_DIR = WORKTREE / "orbits/mobius-001"


def save_solution(circles, path):
    data = {"circles": [[float(c[0]), float(c[1]), float(c[2])] for c in circles]}
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def load_solution(path):
    with open(path) as f:
        data = json.load(f)
    return np.array(data["circles"])

def sum_radii(c): return np.sum(c[:, 2])

def validate(circles, tol=1e-10):
    n = len(circles)
    max_viol = 0.0
    for i in range(n):
        x, y, r = circles[i]
        if r <= 0: return False, abs(r)
        max_viol = max(max_viol, r - x, x + r - 1.0, r - y, y + r - 1.0)
    for i in range(n):
        for j in range(i+1, n):
            dx = circles[i,0] - circles[j,0]
            dy = circles[i,1] - circles[j,1]
            dist = math.sqrt(dx*dx + dy*dy)
            overlap = (circles[i,2] + circles[j,2]) - dist
            max_viol = max(max_viol, overlap)
    return max_viol <= tol, max_viol


# ============ FAST PENALTY OPTIMIZER ============

def penalty_obj_and_grad(x, n, pw):
    """Combined objective + gradient for speed."""
    xs = x[0::3]; ys = x[1::3]; rs = x[2::3]

    obj = -np.sum(rs)
    grad = np.zeros_like(x)
    grad[2::3] = -1.0

    # Containment
    vl = np.maximum(0, rs - xs); vr = np.maximum(0, xs + rs - 1.0)
    vb = np.maximum(0, rs - ys); vt = np.maximum(0, ys + rs - 1.0)

    obj += pw * np.sum(vl**2 + vr**2 + vb**2 + vt**2)
    grad[0::3] += pw * (-2*vl + 2*vr)
    grad[1::3] += pw * (-2*vb + 2*vt)
    grad[2::3] += pw * (2*vl + 2*vr + 2*vb + 2*vt)

    # Non-overlap
    for i in range(n):
        js = np.arange(i+1, n)
        if len(js) == 0: continue
        dx = xs[i] - xs[js]; dy = ys[i] - ys[js]
        dist_sq = dx*dx + dy*dy
        md = rs[i] + rs[js]
        md_sq = md**2
        active = md_sq > dist_sq
        if not np.any(active): continue

        a_js = js[active]
        factor = md_sq[active] - dist_sq[active]
        obj += pw * np.sum(factor**2)

        f2 = 2 * factor
        a_dx = dx[active]; a_dy = dy[active]; a_md = md[active]

        grad[3*i] += pw * np.sum(f2 * (-2*a_dx))
        grad[3*i+1] += pw * np.sum(f2 * (-2*a_dy))
        for k, j in enumerate(a_js):
            grad[3*j] += pw * f2[k] * 2 * a_dx[k]
            grad[3*j+1] += pw * f2[k] * 2 * a_dy[k]
            grad[3*i+2] += pw * f2[k] * 2 * a_md[k]
            grad[3*j+2] += pw * f2[k] * 2 * a_md[k]

    return obj, grad


def fast_optimize(circles, stages=None):
    """Fast penalty-only optimization."""
    if stages is None:
        stages = [10, 1000, 100000, 1e7]
    n = len(circles)
    x = circles.flatten().copy()
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-5, 0.5)] * n

    for pw in stages:
        result = minimize(
            lambda x: penalty_obj_and_grad(x, n, pw),
            x, jac=True, method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 800, 'ftol': 1e-14}
        )
        x = result.x.copy()

    out = x.reshape(n, 3)
    valid, viol = validate(out, tol=1e-8)
    return out, sum_radii(out) if valid else 0.0, valid


def slsqp_polish(circles, maxiter=5000):
    """SLSQP polish for valid solutions."""
    n = len(circles)
    x0 = circles.flatten()
    nn = 3 * n

    def objective(x): return -np.sum(x[2::3])
    def grad_obj(x):
        g = np.zeros(nn); g[2::3] = -1.0; return g

    def all_con(x):
        xs=x[0::3]; ys=x[1::3]; rs=x[2::3]
        vals = list(xs-rs) + list(1-xs-rs) + list(ys-rs) + list(1-ys-rs) + list(rs-1e-6)
        for i in range(n):
            dx=xs[i]-xs[i+1:]; dy=ys[i]-ys[i+1:]
            vals.extend(dx*dx+dy*dy-(rs[i]+rs[i+1:])**2)
        return np.array(vals)

    bounds = [(0.0,1.0),(0.0,1.0),(1e-6,0.5)] * n
    result = minimize(objective, x0, method='SLSQP', jac=grad_obj,
                     bounds=bounds, constraints=[{'type':'ineq','fun':all_con}],
                     options={'maxiter':maxiter,'ftol':1e-15})
    out = result.x.reshape(n, 3)
    valid, _ = validate(out)
    return out, -result.fun if valid else 0.0, valid


# ============ DIVERSE INITIALIZATION GENERATORS ============

def gen_random(n, rng):
    """Purely random placement."""
    circles = []
    for _ in range(n):
        r = rng.uniform(0.03, 0.15)
        circles.append([rng.uniform(r, 1-r), rng.uniform(r, 1-r), r])
    return np.array(circles)

def gen_poisson_disk(n, rng, min_dist=0.08):
    """Poisson disk sampling for positions."""
    points = []
    for _ in range(n * 100):
        if len(points) >= n: break
        x, y = rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95)
        ok = True
        for px, py in points:
            if (x-px)**2 + (y-py)**2 < min_dist**2:
                ok = False; break
        if ok: points.append((x, y))

    while len(points) < n:
        points.append((rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)))

    circles = []
    for x, y in points[:n]:
        circles.append([x, y, rng.uniform(0.04, 0.12)])
    return np.array(circles)

def gen_size_distribution(n, rng, dist_type='uniform'):
    """Generate with specific size distribution then place."""
    if dist_type == 'uniform':
        radii = rng.uniform(0.05, 0.12, n)
    elif dist_type == 'bimodal':
        n_big = rng.randint(3, 8)
        radii = np.concatenate([rng.uniform(0.10, 0.15, n_big),
                                rng.uniform(0.04, 0.08, n - n_big)])
    elif dist_type == 'power_law':
        radii = 0.04 + 0.11 * rng.power(0.5, n)
    elif dist_type == 'one_big':
        radii = np.full(n, 0.07)
        radii[0] = rng.uniform(0.15, 0.22)
    elif dist_type == 'two_big':
        radii = np.full(n, 0.06)
        radii[0] = rng.uniform(0.12, 0.18)
        radii[1] = rng.uniform(0.12, 0.18)
    elif dist_type == 'decreasing':
        radii = np.linspace(0.14, 0.04, n) + rng.uniform(-0.01, 0.01, n)
        radii = np.maximum(radii, 0.02)
    elif dist_type == 'equal':
        radii = np.full(n, 0.08 + rng.uniform(-0.005, 0.005))
    else:
        radii = rng.uniform(0.03, 0.14, n)

    # Place circles greedily
    circles = np.zeros((n, 3))
    order = np.argsort(-radii)  # Place big circles first

    for idx in order:
        r = radii[idx]
        best_pos = None
        best_min_dist = -1

        for _ in range(50):
            x = rng.uniform(r + 0.01, 1 - r - 0.01)
            y = rng.uniform(r + 0.01, 1 - r - 0.01)

            # Check min distance to placed circles
            min_d = float('inf')
            ok = True
            for j in range(n):
                if j == idx or circles[j, 2] == 0: continue
                dx = x - circles[j, 0]; dy = y - circles[j, 1]
                d = math.sqrt(dx*dx + dy*dy) - r - circles[j, 2]
                if d < -0.001: ok = False; break
                min_d = min(min_d, d)

            if ok and min_d > best_min_dist:
                best_min_dist = min_d
                best_pos = (x, y)

        if best_pos is None:
            best_pos = (rng.uniform(r, 1-r), rng.uniform(r, 1-r))

        circles[idx] = [best_pos[0], best_pos[1], r]

    return circles


def gen_ring(n, rng, config):
    """Configurable ring initialization."""
    layers = config  # list of (count, radius_from_center, circle_radius)
    circles = []
    for count, dist, r in layers:
        for i in range(count):
            theta = 2 * math.pi * i / count + rng.uniform(-0.15, 0.15)
            x = 0.5 + dist * math.cos(theta)
            y = 0.5 + dist * math.sin(theta)
            circles.append([x, y, r + rng.uniform(-0.01, 0.01)])
    # Pad or truncate
    while len(circles) < n:
        circles.append([rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9), 0.05])
    circles = np.array(circles[:n])
    for i in range(n):
        circles[i, 2] = max(circles[i, 2], 0.02)
        circles[i, 0] = max(circles[i, 2], min(1-circles[i, 2], circles[i, 0]))
        circles[i, 1] = max(circles[i, 2], min(1-circles[i, 2], circles[i, 1]))
    return circles


def gen_conformal_disk(n, rng):
    """
    Generate packing in unit disk, then map to square via conformal map.
    The conformal map z -> w from disk to square stretches radii non-uniformly.
    """
    # Pack circles in unit disk first
    circles_disk = []
    for i in range(n):
        for _ in range(100):
            # Random point in disk
            theta = rng.uniform(0, 2*math.pi)
            rho = math.sqrt(rng.uniform(0, 1)) * 0.85  # stay away from boundary
            x = rho * math.cos(theta)
            y = rho * math.sin(theta)
            r = rng.uniform(0.03, 0.12) * (1 - 0.3 * rho)  # smaller near boundary

            ok = True
            # Check boundary
            if rho + r > 0.95: ok = False
            # Check other circles
            for cx, cy, cr in circles_disk:
                dx, dy = x-cx, y-cy
                if math.sqrt(dx*dx + dy*dy) < r + cr + 0.001:
                    ok = False; break
            if ok:
                circles_disk.append((x, y, r))
                break
        else:
            circles_disk.append((rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), 0.04))

    # Map disk to square: simple stretching approximation
    # Use the Schwarz-Christoffel idea: map disk to square
    # Approximate: radial stretching + angular mapping to corners
    result = np.zeros((n, 3))
    for i, (x, y, r) in enumerate(circles_disk):
        # Normalize to [-1, 1] disk
        # Map to square: polar coords -> square coords
        rho = math.sqrt(x*x + y*y)
        theta = math.atan2(y, x)

        if rho < 1e-10:
            sx, sy = 0.5, 0.5
        else:
            # Simple conformal-ish mapping: stretch along diagonals
            # This maps circle to roughly square shape
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            max_cs = max(abs(cos_t), abs(sin_t))
            stretch = 1.0 / max_cs if max_cs > 0.1 else 1.0
            sx = 0.5 + 0.45 * rho * stretch * cos_t / math.sqrt(stretch)
            sy = 0.5 + 0.45 * rho * stretch * sin_t / math.sqrt(stretch)

        # Scale radius by local stretching
        scale_factor = 0.5  # approximate
        sr = r * scale_factor

        result[i] = [max(sr, min(1-sr, sx)), max(sr, min(1-sr, sy)), max(sr, 0.02)]

    return result


# ============ EVOLUTIONARY MIXING ============

def crossover(parent1, parent2, rng):
    """Create offspring by mixing two solutions."""
    n = len(parent1)
    child = np.zeros((n, 3))

    # Strategy: take positions from parent1, radii from parent2 (or vice versa)
    if rng.random() < 0.5:
        # Spatial crossover: split by x-coordinate
        threshold = rng.uniform(0.3, 0.7)
        for i in range(n):
            if parent1[i, 0] < threshold:
                child[i] = parent1[i]
            else:
                # Find closest circle from parent2 in the right half
                right_mask = parent2[:, 0] >= threshold
                if np.any(right_mask):
                    dists = np.sum((parent2[right_mask, :2] - parent1[i, :2])**2, axis=1)
                    best_j = np.where(right_mask)[0][np.argmin(dists)]
                    child[i] = parent2[best_j]
                else:
                    child[i] = parent1[i]
    else:
        # Random subset crossover
        mask = rng.random(n) < 0.5
        child[mask] = parent1[mask]
        child[~mask] = parent2[~mask]

    return child


# ============ MAIN ============

def main():
    base = load_solution(WORKTREE / "orbits/topo-001/solution_n26.json")
    base_metric = sum_radii(base)
    print(f"Base: {base_metric:.10f}")

    rng = np.random.RandomState(SEED)
    best = base_metric
    best_circles = base.copy()
    all_metrics = []
    top_solutions = [(base_metric, base.copy())]  # Keep top K solutions

    # ============ Phase 1: Massive random multi-start ============
    print("\n=== Phase 1: Massive random multi-start (300 trials) ===")
    t0 = time.time()

    dist_types = ['uniform', 'bimodal', 'power_law', 'one_big', 'two_big', 'decreasing', 'equal', 'random']
    ring_configs = [
        [(1, 0, 0.13), (8, 0.22, 0.10), (12, 0.38, 0.08), (4, 0.42, 0.09), (1, 0.42, 0.07)],
        [(2, 0.08, 0.12), (7, 0.22, 0.10), (10, 0.37, 0.08), (7, 0.44, 0.06)],
        [(3, 0.08, 0.11), (9, 0.25, 0.09), (14, 0.40, 0.07)],
        [(1, 0, 0.15), (6, 0.18, 0.11), (12, 0.35, 0.08), (7, 0.44, 0.06)],
        [(4, 0.10, 0.10), (8, 0.25, 0.10), (14, 0.40, 0.07)],
        [(1, 0, 0.12), (5, 0.15, 0.10), (10, 0.30, 0.09), (10, 0.43, 0.06)],
        [(2, 0.06, 0.13), (8, 0.22, 0.10), (10, 0.37, 0.08), (6, 0.44, 0.07)],
        [(1, 0, 0.14), (7, 0.20, 0.10), (11, 0.36, 0.08), (7, 0.44, 0.06)],
    ]

    for trial in range(300):
        # Diverse initialization strategy
        strategy = trial % 20
        try:
            if strategy < 4:
                init = gen_random(N, rng)
            elif strategy < 8:
                init = gen_poisson_disk(N, rng, min_dist=rng.uniform(0.06, 0.14))
            elif strategy < 16:
                dt = dist_types[(strategy - 8) % len(dist_types)]
                init = gen_size_distribution(N, rng, dt)
            elif strategy < 18:
                rc = ring_configs[trial % len(ring_configs)]
                init = gen_ring(N, rng, rc)
            elif strategy < 19:
                init = gen_conformal_disk(N, rng)
            else:
                # Evolutionary: crossover of top solutions
                if len(top_solutions) >= 2:
                    idx1, idx2 = rng.choice(len(top_solutions), 2, replace=False)
                    init = crossover(top_solutions[idx1][1], top_solutions[idx2][1], rng)
                else:
                    init = gen_random(N, rng)

            opt, metric, valid = fast_optimize(init)

            if valid and metric > 0:
                all_metrics.append(metric)

                # Keep top 20 solutions
                if len(top_solutions) < 20 or metric > top_solutions[-1][0]:
                    top_solutions.append((metric, opt.copy()))
                    top_solutions.sort(key=lambda x: -x[0])
                    top_solutions = top_solutions[:20]

                if metric > best + 1e-12:
                    print(f"  Trial {trial}: {metric:.10f} NEW BEST (strategy {strategy})")
                    best = metric
                    best_circles = opt.copy()
        except:
            pass

        if (trial+1) % 100 == 0:
            n_valid = len(all_metrics)
            top3 = sorted(all_metrics, reverse=True)[:3] if all_metrics else []
            print(f"  Trial {trial+1}/300: best={best:.10f}, valid={n_valid}, "
                  f"top3={[f'{m:.6f}' for m in top3]}")

    print(f"Phase 1: {time.time()-t0:.1f}s, best={best:.10f}")

    # ============ Phase 2: SLSQP polish on top solutions ============
    print("\n=== Phase 2: SLSQP polish on top 20 ===")
    t0 = time.time()

    for i, (m, c) in enumerate(top_solutions):
        try:
            polished, pm, pv = slsqp_polish(c, maxiter=10000)
            if pv and pm > best + 1e-12:
                print(f"  Polished #{i}: {m:.10f} -> {pm:.10f} NEW BEST")
                best = pm
                best_circles = polished.copy()
            elif pv and pm > m:
                top_solutions[i] = (pm, polished.copy())
        except:
            pass

    print(f"Phase 2: {time.time()-t0:.1f}s, best={best:.10f}")

    # ============ Phase 3: Evolutionary mixing of top solutions ============
    print("\n=== Phase 3: Evolutionary mixing (200 offspring) ===")
    t0 = time.time()

    for trial in range(200):
        if len(top_solutions) < 2: break

        idx1, idx2 = rng.choice(min(10, len(top_solutions)), 2, replace=False)
        child = crossover(top_solutions[idx1][1], top_solutions[idx2][1], rng)

        # Add mutation
        noise = rng.normal(0, 0.02, child.shape)
        noise[:, 2] *= 0.3
        child += noise
        for i in range(N):
            child[i, 2] = max(child[i, 2], 0.02)
            child[i, 0] = max(child[i, 2], min(1-child[i, 2], child[i, 0]))
            child[i, 1] = max(child[i, 2], min(1-child[i, 2], child[i, 1]))

        try:
            opt, metric, valid = fast_optimize(child)
            if valid and metric > best + 1e-12:
                print(f"  Evo {trial}: {metric:.10f} NEW BEST")
                best = metric
                best_circles = opt.copy()

            if valid and metric > 0:
                if len(top_solutions) < 20 or metric > top_solutions[-1][0]:
                    top_solutions.append((metric, opt.copy()))
                    top_solutions.sort(key=lambda x: -x[0])
                    top_solutions = top_solutions[:20]
        except:
            pass

        if (trial+1) % 50 == 0:
            print(f"  Evo {trial+1}/200: best={best:.10f}")

    print(f"Phase 3: {time.time()-t0:.1f}s, best={best:.10f}")

    # ============ Phase 4: Targeted Schwarz-Christoffel ============
    print("\n=== Phase 4: Conformal disk->square packings (100 trials) ===")
    t0 = time.time()

    for trial in range(100):
        try:
            init = gen_conformal_disk(N, rng)
            opt, metric, valid = fast_optimize(init)
            if valid and metric > best + 1e-12:
                print(f"  Conformal {trial}: {metric:.10f} NEW BEST")
                best = metric
                best_circles = opt.copy()
        except:
            pass

        if (trial+1) % 50 == 0:
            print(f"  Conformal {trial+1}/100: best={best:.10f}")

    print(f"Phase 4: {time.time()-t0:.1f}s, best={best:.10f}")

    # ============ Final SLSQP polish ============
    print("\n=== Final polish ===")
    try:
        polished, pm, pv = slsqp_polish(best_circles, maxiter=20000)
        if pv and pm > best:
            best = pm
            best_circles = polished
    except:
        pass

    valid, viol = validate(best_circles)
    print(f"\nFINAL: {best:.10f} (valid={valid}, viol={viol:.2e})")
    print(f"Improvement: {best - base_metric:.2e}")

    save_solution(best_circles, OUTPUT_DIR / "solution_n26.json")

    # Save metrics
    with open(OUTPUT_DIR / "search_metrics.json", 'w') as f:
        json.dump({
            'all_metrics': sorted(all_metrics, reverse=True)[:100],
            'best': best,
            'base': base_metric,
            'n_valid': len(all_metrics),
            'top_solutions': [m for m, _ in top_solutions[:10]]
        }, f, indent=2)

    print(f"Saved. Total valid solutions found: {len(all_metrics)}")
    return best, all_metrics


if __name__ == "__main__":
    best, metrics = main()
