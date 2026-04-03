"""
Mobius-001: Fast circle packing optimizer.

Uses penalty-based L-BFGS-B for speed, then SLSQP polish.
Explores via Mobius deformations, topology search, and multi-start.
"""

import json
import math
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from pathlib import Path
import time
import itertools

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

def sum_radii(circles):
    return np.sum(circles[:, 2])

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

def penalty_objective(x, n, penalty_weight):
    """
    Vectorized penalty objective: -sum(r) + penalty * violations.
    x is flat array [x0,y0,r0, x1,y1,r1, ...]
    """
    xs = x[0::3]
    ys = x[1::3]
    rs = x[2::3]

    obj = -np.sum(rs)
    penalty = 0.0

    # Containment: r <= x, x+r <= 1, r <= y, y+r <= 1
    viol_left = np.maximum(0, rs - xs)
    viol_right = np.maximum(0, xs + rs - 1.0)
    viol_bottom = np.maximum(0, rs - ys)
    viol_top = np.maximum(0, ys + rs - 1.0)
    viol_rmin = np.maximum(0, 1e-5 - rs)

    penalty += np.sum(viol_left**2 + viol_right**2 + viol_bottom**2 + viol_top**2 + viol_rmin**2)

    # Non-overlap: vectorized
    for i in range(n):
        dx = xs[i] - xs[i+1:]
        dy = ys[i] - ys[i+1:]
        dist_sq = dx*dx + dy*dy
        min_dist = rs[i] + rs[i+1:]
        min_dist_sq = min_dist**2
        overlap_sq = np.maximum(0, min_dist_sq - dist_sq)
        penalty += np.sum(overlap_sq)

    return obj + penalty_weight * penalty


def penalty_gradient(x, n, penalty_weight):
    """Analytical gradient of penalty objective."""
    grad = np.zeros_like(x)
    xs = x[0::3]
    ys = x[1::3]
    rs = x[2::3]

    # Gradient of -sum(r)
    grad[2::3] = -1.0

    # Containment penalties
    viol_left = np.maximum(0, rs - xs)
    viol_right = np.maximum(0, xs + rs - 1.0)
    viol_bottom = np.maximum(0, rs - ys)
    viol_top = np.maximum(0, ys + rs - 1.0)

    # d/dx of (r-x)^2 when active: 2*(r-x)*(-1) for x, 2*(r-x)*(1) for r
    grad[0::3] += penalty_weight * (-2 * viol_left + 2 * viol_right)
    grad[1::3] += penalty_weight * (-2 * viol_bottom + 2 * viol_top)
    grad[2::3] += penalty_weight * (2 * viol_left + 2 * viol_right + 2 * viol_bottom + 2 * viol_top)

    viol_rmin = np.maximum(0, 1e-5 - rs)
    grad[2::3] += penalty_weight * (-2 * viol_rmin)

    # Non-overlap penalties
    for i in range(n):
        js = np.arange(i+1, n)
        if len(js) == 0:
            continue
        dx = xs[i] - xs[js]
        dy = ys[i] - ys[js]
        dist_sq = dx*dx + dy*dy
        min_dist = rs[i] + rs[js]
        min_dist_sq = min_dist**2
        active = min_dist_sq > dist_sq  # overlap

        if not np.any(active):
            continue

        overlap_factor = 2 * (min_dist_sq[active] - dist_sq[active])
        act_js = js[active]
        act_dx = dx[active]
        act_dy = dy[active]
        act_min_dist = min_dist[active]

        # d/d(x_i): 2 * (min_dist^2 - dist^2) * d/d(x_i) [ min_dist^2 - dist^2 ]
        # = 2 * factor * (-2*dx)  for dist^2 part
        # min_dist^2 = (ri+rj)^2, d/d(x_i) = 0
        grad[3*i] += penalty_weight * np.sum(2 * (min_dist_sq[active] - dist_sq[active]) * (-2 * act_dx))
        grad[3*i+1] += penalty_weight * np.sum(2 * (min_dist_sq[active] - dist_sq[active]) * (-2 * act_dy))

        for k_idx, j in enumerate(act_js):
            factor = 2 * (min_dist_sq[active][k_idx] - dist_sq[active][k_idx])
            grad[3*j] += penalty_weight * factor * (2 * act_dx[k_idx])
            grad[3*j+1] += penalty_weight * factor * (2 * act_dy[k_idx])
            # d/d(r_i) and d/d(r_j) from min_dist^2 = (ri+rj)^2
            grad[3*i+2] += penalty_weight * factor * 2 * act_min_dist[k_idx]
            grad[3*j+2] += penalty_weight * factor * 2 * act_min_dist[k_idx]

    return grad


def optimize_penalty(circles, max_stages=8, maxiter=2000):
    """Progressive penalty optimization with L-BFGS-B."""
    n = len(circles)
    x = circles.flatten().copy()
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-5, 0.5)] * n

    best_metric = 0.0
    best_x = x.copy()

    for stage, pw in enumerate([1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1e6, 1e7]):
        if stage >= max_stages:
            break
        result = minimize(
            penalty_objective, x, args=(n, pw),
            jac=lambda x, n=n, pw=pw: penalty_gradient(x, n, pw),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-15}
        )
        x = result.x.copy()

        circ = x.reshape(n, 3)
        valid, max_viol = validate(circ, tol=1e-8)
        metric = sum_radii(circ)

        if valid and metric > best_metric:
            best_metric = metric
            best_x = x.copy()

    return best_x.reshape(n, 3), best_metric


def optimize_slsqp_fast(circles, maxiter=3000):
    """SLSQP with vectorized constraint evaluation."""
    n = len(circles)
    x0 = circles.flatten()

    def objective(x):
        return -np.sum(x[2::3])

    def grad_objective(x):
        g = np.zeros_like(x)
        g[2::3] = -1.0
        return g

    # Compute all constraint values at once
    def all_constraints(x):
        xs = x[0::3]; ys = x[1::3]; rs = x[2::3]
        vals = []
        # Containment (4n)
        vals.extend(xs - rs)  # x - r >= 0
        vals.extend(1.0 - xs - rs)  # 1 - x - r >= 0
        vals.extend(ys - rs)
        vals.extend(1.0 - ys - rs)
        # r > 0 (n)
        vals.extend(rs - 1e-6)
        # Non-overlap (n*(n-1)/2)
        for i in range(n):
            dx = xs[i] - xs[i+1:]
            dy = ys[i] - ys[i+1:]
            vals.extend(dx*dx + dy*dy - (rs[i] + rs[i+1:])**2)
        return np.array(vals)

    constraints = [{'type': 'ineq', 'fun': all_constraints}]
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * n

    result = minimize(objective, x0, method='SLSQP', jac=grad_objective,
                     bounds=bounds, constraints=constraints,
                     options={'maxiter': maxiter, 'ftol': 1e-15, 'disp': False})

    out = result.x.reshape(n, 3)
    return out, -result.fun


def full_optimize(circles, use_slsqp_polish=True):
    """Penalty -> SLSQP pipeline."""
    # Stage 1: Penalty
    pen_circ, pen_metric = optimize_penalty(circles)

    valid, viol = validate(pen_circ, tol=1e-8)
    if not valid or not use_slsqp_polish:
        return pen_circ, pen_metric

    # Stage 2: SLSQP polish
    try:
        slsqp_circ, slsqp_metric = optimize_slsqp_fast(pen_circ, maxiter=5000)
        valid2, viol2 = validate(slsqp_circ)
        if valid2 and slsqp_metric > pen_metric:
            return slsqp_circ, slsqp_metric
    except:
        pass

    return pen_circ, pen_metric


# ============ MOBIUS DEFORMATIONS ============

def apply_mobius_to_circle(center, radius, a, b, c, d):
    """Apply Mobius f(z)=(az+b)/(cz+d) to a circle."""
    det = a * d - b * c
    cz_d = c * center + d
    cz_d_sq = abs(cz_d)**2
    c_sq = abs(c)**2
    denom = cz_d_sq - c_sq * radius**2
    if abs(denom) < 1e-14:
        return None, None
    new_center = ((a * center + b) * np.conj(cz_d) - a * np.conj(c) * radius**2) / denom
    new_radius = abs(det) * radius / abs(denom)
    return new_center, new_radius


def find_contacts(circles, tol=1e-4):
    """Contact graph as adjacency dict."""
    n = len(circles)
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i+1, n):
            dx = circles[i,0] - circles[j,0]
            dy = circles[i,1] - circles[j,1]
            dist = math.sqrt(dx*dx + dy*dy)
            gap = dist - (circles[i,2] + circles[j,2])
            if abs(gap) < tol:
                adj[i].append(j)
                adj[j].append(i)
    return adj


def mobius_deform_cluster(circles, cluster_indices, rng, strength=0.1):
    """Apply Mobius transformation to a cluster of circles."""
    cx = np.mean(circles[cluster_indices, 0])
    cy = np.mean(circles[cluster_indices, 1])

    a = complex(1.0 + rng.normal(0, strength*0.1), rng.normal(0, strength*0.1))
    b = complex(rng.normal(0, strength*0.05), rng.normal(0, strength*0.05))
    c = complex(rng.normal(0, strength*0.02), rng.normal(0, strength*0.02))
    d = complex(1.0 + rng.normal(0, strength*0.1), rng.normal(0, strength*0.1))

    det = a*d - b*c
    if abs(det) < 1e-10:
        return None
    scale = det**0.5
    a, b, c, d = a/scale, b/scale, c/scale, d/scale

    new_circles = circles.copy()
    for idx in cluster_indices:
        center = complex(circles[idx, 0] - cx, circles[idx, 1] - cy)
        radius = circles[idx, 2]
        new_c, new_r = apply_mobius_to_circle(center, radius, a, b, c, d)
        if new_c is None or new_r is None or new_r <= 0:
            return None
        nx = new_c.real + cx
        ny = new_c.imag + cy
        nr = max(new_r, 1e-4)
        nx = max(nr, min(1-nr, nx))
        ny = max(nr, min(1-nr, ny))
        new_circles[idx] = [nx, ny, nr]
    return new_circles


# ============ INITIALIZATIONS ============

def ring_init_v(n, rng, variant):
    """Ring-based initialization with many variants."""
    circles = []
    if variant == 0:
        # 1+8+12+4+1
        circles.append([0.5, 0.5, 0.13])
        for i in range(8):
            t = 2*math.pi*i/8 + rng.uniform(-0.1, 0.1)
            circles.append([0.5+0.22*math.cos(t), 0.5+0.22*math.sin(t), 0.10+rng.uniform(-0.01,0.01)])
        for i in range(12):
            t = 2*math.pi*i/12 + rng.uniform(-0.05, 0.05)
            circles.append([0.5+0.38*math.cos(t), 0.5+0.38*math.sin(t), 0.08+rng.uniform(-0.01,0.01)])
        for i in range(4):
            t = math.pi/4 + math.pi/2*i
            circles.append([0.5+0.42*math.cos(t), 0.5+0.42*math.sin(t), 0.09])
        circles.append([0.5, 0.08, 0.07])
    elif variant == 1:
        # 2+7+10+7
        circles.append([0.35, 0.5, 0.12]); circles.append([0.65, 0.5, 0.12])
        for i in range(7):
            t = 2*math.pi*i/7 + rng.uniform(-0.15, 0.15)
            circles.append([0.5+0.2*math.cos(t), 0.5+0.2*math.sin(t), 0.10])
        for i in range(10):
            t = 2*math.pi*i/10 + rng.uniform(-0.1, 0.1)
            circles.append([0.5+0.37*math.cos(t), 0.5+0.37*math.sin(t), 0.08])
        for i in range(7):
            t = 2*math.pi*i/7 + math.pi/7 + rng.uniform(-0.1, 0.1)
            circles.append([0.5+0.44*math.cos(t), 0.5+0.44*math.sin(t), 0.06])
    elif variant == 2:
        # Hex grid
        positions = []
        for row in range(6):
            y = 0.1 + row * 0.16
            offset = 0.08 if row % 2 else 0.0
            for col in range(5):
                x = 0.1 + offset + col * 0.2
                if x > 0.92 or y > 0.92: continue
                positions.append([x + rng.uniform(-0.02, 0.02), y + rng.uniform(-0.02, 0.02)])
        rng.shuffle(positions)
        for p in positions[:n]:
            circles.append([p[0], p[1], 0.07 + rng.uniform(-0.02, 0.02)])
    elif variant == 3:
        # 4+8+14 quad center
        for dx, dy in [(-0.08,-0.08),(0.08,-0.08),(-0.08,0.08),(0.08,0.08)]:
            circles.append([0.5+dx, 0.5+dy, 0.10])
        for i in range(8):
            t = 2*math.pi*i/8 + math.pi/8
            circles.append([0.5+0.25*math.cos(t), 0.5+0.25*math.sin(t), 0.10])
        for i in range(14):
            t = 2*math.pi*i/14
            circles.append([0.5+0.40*math.cos(t), 0.5+0.40*math.sin(t), 0.07])
    elif variant == 4:
        # 3+9+14 triangle center
        for i in range(3):
            t = 2*math.pi*i/3 + math.pi/6
            circles.append([0.5+0.08*math.cos(t), 0.5+0.08*math.sin(t), 0.11])
        for i in range(9):
            t = 2*math.pi*i/9 + rng.uniform(-0.1, 0.1)
            circles.append([0.5+0.25*math.cos(t), 0.5+0.25*math.sin(t), 0.09])
        for i in range(14):
            t = 2*math.pi*i/14
            circles.append([0.5+0.40*math.cos(t), 0.5+0.40*math.sin(t), 0.07])
    elif variant == 5:
        # Grid 5x5 + 1
        idx = 0
        for row in range(5):
            for col in range(5):
                x = 0.1 + col * 0.2 + rng.uniform(-0.02, 0.02)
                y = 0.1 + row * 0.2 + rng.uniform(-0.02, 0.02)
                circles.append([x, y, 0.08 + rng.uniform(-0.02, 0.02)])
                idx += 1
        circles.append([0.5, 0.5, 0.05])
    elif variant == 6:
        # Asymmetric: big on left, small on right
        for i in range(8):
            t = 2*math.pi*i/8
            circles.append([0.3+0.15*math.cos(t), 0.5+0.15*math.sin(t), 0.12])
        for i in range(12):
            t = 2*math.pi*i/12
            circles.append([0.7+0.2*math.cos(t), 0.5+0.2*math.sin(t), 0.07])
        for i in range(6):
            t = 2*math.pi*i/6
            circles.append([0.5+0.4*math.cos(t), 0.5+0.4*math.sin(t), 0.06])
    elif variant == 7:
        # Spiral
        for i in range(n):
            t = 0.5 + i * 0.8
            r_dist = 0.05 + 0.35 * i / n
            x = 0.5 + r_dist * math.cos(t)
            y = 0.5 + r_dist * math.sin(t)
            r = 0.12 - 0.06 * i / n + rng.uniform(-0.01, 0.01)
            circles.append([x, y, max(r, 0.03)])
    else:
        # Random
        for i in range(n):
            r = rng.uniform(0.03, 0.12)
            circles.append([rng.uniform(r, 1-r), rng.uniform(r, 1-r), r])

    circles = np.array(circles[:n])
    for i in range(len(circles)):
        circles[i, 2] = max(circles[i, 2], 0.02)
        circles[i, 0] = max(circles[i, 2], min(1 - circles[i, 2], circles[i, 0]))
        circles[i, 1] = max(circles[i, 2], min(1 - circles[i, 2], circles[i, 1]))
    return circles


# ============ MAIN SEARCH ============

def main():
    base_path = WORKTREE / "orbits/topo-001/solution_n26.json"
    base = load_solution(base_path)
    base_metric = sum_radii(base)
    print(f"Base solution: {base_metric:.10f}")

    rng = np.random.RandomState(SEED)
    overall_best = base_metric
    overall_best_circles = base.copy()
    all_metrics = []

    # ============ Phase 1: Multi-start with diverse topologies ============
    print("\n=== Phase 1: Multi-start diverse topologies ===")
    t0 = time.time()

    for trial in range(60):
        variant = trial % 9
        init = ring_init_v(N, rng, variant)

        try:
            opt, metric = full_optimize(init, use_slsqp_polish=True)
            valid, viol = validate(opt)
            if valid:
                all_metrics.append(metric)
                if metric > overall_best + 1e-12:
                    print(f"  Trial {trial} (v{variant}): {metric:.10f} NEW BEST")
                    overall_best = metric
                    overall_best_circles = opt.copy()
                elif trial < 10 or (trial+1) % 10 == 0:
                    print(f"  Trial {trial} (v{variant}): {metric:.10f}")
        except Exception as e:
            pass

    print(f"Phase 1: {time.time()-t0:.1f}s, best={overall_best:.10f}, "
          f"tried={len(all_metrics)}, unique basins={len(set(round(m, 4) for m in all_metrics))}")

    # ============ Phase 2: Mobius deformations on best ============
    print("\n=== Phase 2: Mobius cluster deformations ===")
    t0 = time.time()

    adj = find_contacts(overall_best_circles)

    for trial in range(100):
        # Pick a random center circle and its neighbors
        center = rng.randint(N)
        neighbors = adj.get(center, [])
        if len(neighbors) < 2:
            continue

        # Random subset of neighbors
        k = min(rng.randint(2, 5), len(neighbors))
        chosen = list(rng.choice(neighbors, k, replace=False))
        cluster = [center] + chosen

        strength = rng.choice([0.01, 0.03, 0.1, 0.2, 0.5, 1.0])
        deformed = mobius_deform_cluster(overall_best_circles, cluster, rng, strength)
        if deformed is None:
            continue

        try:
            opt, metric = full_optimize(deformed)
            valid, viol = validate(opt)
            if valid and metric > overall_best + 1e-12:
                print(f"  Mobius {trial}: {overall_best:.10f} -> {metric:.10f} "
                      f"(cluster size={len(cluster)}, str={strength:.2f})")
                overall_best = metric
                overall_best_circles = opt.copy()
                adj = find_contacts(overall_best_circles)
        except:
            pass

        if (trial+1) % 25 == 0:
            print(f"  Mobius {trial+1}/100: best={overall_best:.10f}")

    print(f"Phase 2: {time.time()-t0:.1f}s, best={overall_best:.10f}")

    # ============ Phase 3: Global Mobius on whole packing ============
    print("\n=== Phase 3: Global Mobius transformations ===")
    t0 = time.time()

    for trial in range(50):
        # Apply a gentle Mobius transform to ALL circles
        all_indices = list(range(N))
        strength = rng.choice([0.005, 0.01, 0.02, 0.05])
        deformed = mobius_deform_cluster(overall_best_circles, all_indices, rng, strength)
        if deformed is None:
            continue

        try:
            opt, metric = full_optimize(deformed)
            valid, viol = validate(opt)
            if valid and metric > overall_best + 1e-12:
                print(f"  Global Mobius {trial}: {overall_best:.10f} -> {metric:.10f}")
                overall_best = metric
                overall_best_circles = opt.copy()
        except:
            pass

        if (trial+1) % 25 == 0:
            print(f"  Global {trial+1}/50: best={overall_best:.10f}")

    print(f"Phase 3: {time.time()-t0:.1f}s, best={overall_best:.10f}")

    # ============ Phase 4: Permutation search ============
    print("\n=== Phase 4: Circle permutation + re-optimize ===")
    t0 = time.time()

    for trial in range(40):
        perm = overall_best_circles.copy()
        # Swap 2-4 circle positions (but keep radii)
        n_swaps = rng.randint(1, 4)
        for _ in range(n_swaps):
            i, j = rng.choice(N, 2, replace=False)
            # Swap positions, keep radii
            perm[i, 0], perm[j, 0] = perm[j, 0], perm[i, 0]
            perm[i, 1], perm[j, 1] = perm[j, 1], perm[i, 1]

        try:
            opt, metric = full_optimize(perm)
            valid, viol = validate(opt)
            if valid and metric > overall_best + 1e-12:
                print(f"  Perm {trial}: {overall_best:.10f} -> {metric:.10f}")
                overall_best = metric
                overall_best_circles = opt.copy()
        except:
            pass

        if (trial+1) % 20 == 0:
            print(f"  Perm {trial+1}/40: best={overall_best:.10f}")

    print(f"Phase 4: {time.time()-t0:.1f}s, best={overall_best:.10f}")

    # ============ Phase 5: Remove-and-reinsert ============
    print("\n=== Phase 5: Remove-and-reinsert ===")
    t0 = time.time()

    for trial in range(N):
        # Remove circle i, optimize n-1 circles, then re-add
        reduced = np.delete(overall_best_circles, trial, axis=0)

        # Optimize the n-1 circles (they can grow)
        try:
            opt_reduced, _ = full_optimize(reduced)

            # Find best spot to re-insert
            best_insert_metric = 0
            best_insert = None

            for attempt in range(20):
                r_new = rng.uniform(0.02, 0.10)
                x_new = rng.uniform(r_new, 1-r_new)
                y_new = rng.uniform(r_new, 1-r_new)

                full = np.vstack([opt_reduced, [[x_new, y_new, r_new]]])
                opt_full, metric = full_optimize(full)
                valid, viol = validate(opt_full)

                if valid and metric > best_insert_metric:
                    best_insert_metric = metric
                    best_insert = opt_full.copy()

            if best_insert is not None and best_insert_metric > overall_best + 1e-12:
                print(f"  Remove-reinsert {trial}: {overall_best:.10f} -> {best_insert_metric:.10f}")
                overall_best = best_insert_metric
                overall_best_circles = best_insert.copy()
        except:
            pass

        if (trial+1) % 10 == 0:
            print(f"  Remove-reinsert {trial+1}/{N}: best={overall_best:.10f}")

    print(f"Phase 5: {time.time()-t0:.1f}s, best={overall_best:.10f}")

    # ============ Final summary ============
    print("\n" + "="*60)
    print(f"FINAL BEST: {overall_best:.10f}")
    print(f"Improvement over base: {overall_best - base_metric:.2e}")
    valid, viol = validate(overall_best_circles)
    print(f"Valid: {valid}, Max violation: {viol:.2e}")
    print("="*60)

    save_solution(overall_best_circles, OUTPUT_DIR / "solution_n26.json")
    print(f"Saved to {OUTPUT_DIR / 'solution_n26.json'}")

    # Save metrics for plotting
    with open(OUTPUT_DIR / "search_metrics.json", 'w') as f:
        json.dump({'all_metrics': all_metrics, 'best': overall_best,
                   'base': base_metric}, f)


if __name__ == "__main__":
    main()
