"""
Search v3: Targeted approaches based on structural insights.

Key observations from v1+v2:
- The 2.636 basin is extremely deep -- random inits + optimization always converge to it
- Need truly DIFFERENT topological structures

New strategies:
1. "Apollonius" approach -- start with 3 mutually tangent circles, fill Apollonian gasket style
2. "Stripe" arrangements -- horizontal/vertical stripes of equal circles
3. "Brick wall" pattern -- offset rows like brick laying
4. "Spiral" arrangement -- golden spiral placement
5. Multi-scale: fix large circles in known-good positions, use CMA-ES for small ones only
6. "Transplant" -- take partial solutions from n=25 or n=27 and add/remove a circle
"""

import json
import numpy as np
from scipy.optimize import minimize
import os
import time

WORKDIR = os.path.dirname(os.path.abspath(__file__))
N = 26
BEST_KNOWN = 2.6359830850

def load_solution(path):
    with open(path) as f:
        data = json.load(f)
    circles = np.array(data["circles"])
    return circles[:, 0], circles[:, 1], circles[:, 2]

def save_solution(x, y, r, path):
    circles = [[float(x[i]), float(y[i]), float(r[i])] for i in range(len(x))]
    with open(path, 'w') as f:
        json.dump({"circles": circles}, f, indent=2)

def penalty_objective_grad(params, n, pw):
    x = params[:n]; y = params[n:2*n]; r = params[2*n:3*n]
    grad = np.zeros(3*n); grad[2*n:3*n] = -1.0
    obj = -np.sum(r); penalty = 0.0
    vl = np.maximum(0, r - x); vr_w = np.maximum(0, x + r - 1)
    vb = np.maximum(0, r - y); vt = np.maximum(0, y + r - 1)
    vneg = np.maximum(0, -r)
    penalty += np.sum(vl**2 + vr_w**2 + vb**2 + vt**2 + vneg**2)
    grad[:n] += pw * (-2*vl + 2*vr_w)
    grad[n:2*n] += pw * (-2*vb + 2*vt)
    grad[2*n:3*n] += pw * (2*vl + 2*vr_w + 2*vb + 2*vt + 2*vneg)
    dx = x[:, None] - x[None, :]; dy = y[:, None] - y[None, :]
    sr = r[:, None] + r[None, :]; dist2 = dx**2 + dy**2
    overlap_mask = (dist2 < sr**2); np.fill_diagonal(overlap_mask, False)
    i_idx, j_idx = np.triu_indices(n, k=1)
    mask_pairs = overlap_mask[i_idx, j_idx]
    if np.any(mask_pairs):
        ii = i_idx[mask_pairs]; jj = j_idx[mask_pairs]
        dxp = dx[ii, jj]; dyp = dy[ii, jj]; d2p = dist2[ii, jj]
        srp = sr[ii, jj]; distp = np.sqrt(np.maximum(d2p, 1e-20))
        gap = srp - distp; penalty += np.sum(gap**2)
        coeff = pw * 2 * gap; dgx = -dxp / distp; dgy = -dyp / distp
        np.add.at(grad[:n], ii, coeff * dgx); np.add.at(grad[:n], jj, -coeff * dgx)
        np.add.at(grad[n:2*n], ii, coeff * dgy); np.add.at(grad[n:2*n], jj, -coeff * dgy)
        np.add.at(grad[2*n:3*n], ii, coeff); np.add.at(grad[2*n:3*n], jj, coeff)
    return obj + pw * penalty, grad

def optimize_penalty(x, y, r, maxiter=2000):
    n = len(x); params = np.concatenate([x, y, r])
    bounds = [(0.001, 0.999)]*n + [(0.001, 0.999)]*n + [(0.005, 0.499)]*n
    for pw in [10, 100, 1000, 10000]:
        result = minimize(lambda p: penalty_objective_grad(p, n, pw),
            params, method='L-BFGS-B', jac=True, bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-15})
        params = result.x
    return params[:n], params[n:2*n], params[2*n:3*n]

def optimize_slsqp(x0, y0, r0, maxiter=8000):
    n = len(x0); params0 = np.concatenate([x0, y0, r0])
    constraints = []
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[i] - p[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[i] - p[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[n+i] - p[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[n+i] - p[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[2*n+i] - 1e-6})
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({'type': 'ineq',
                'fun': lambda p, i=i, j=j: (p[i]-p[j])**2 + (p[n+i]-p[n+j])**2 - (p[2*n+i]+p[2*n+j])**2})
    result = minimize(lambda p: -np.sum(p[2*n:3*n]), params0,
        method='SLSQP', constraints=constraints, options={'maxiter': maxiter, 'ftol': 1e-15})
    x = result.x[:n]; y = result.x[n:2*n]; r = result.x[2*n:3*n]
    return x, y, r, np.sum(r), result.success

def is_feasible(x, y, r, tol=1e-8):
    n = len(x)
    if np.any(r < -tol): return False
    if np.any(x - r < -tol) or np.any(1 - x - r < -tol): return False
    if np.any(y - r < -tol) or np.any(1 - y - r < -tol): return False
    dx = x[:, None] - x[None, :]; dy = y[:, None] - y[None, :]
    sr = r[:, None] + r[None, :]; dist2 = dx**2 + dy**2
    np.fill_diagonal(dist2, np.inf)
    if np.any(dist2 < sr**2 - tol): return False
    return True

def full_optimize(x, y, r):
    x, y, r = optimize_penalty(x, y, r, maxiter=2000)
    x2, y2, r2, metric, success = optimize_slsqp(x, y, r, maxiter=10000)
    if is_feasible(x2, y2, r2):
        return x2, y2, r2, metric
    elif is_feasible(x, y, r):
        return x, y, r, np.sum(r)
    return x2, y2, r2, 0.0

# ============================================================
# Apollonius-style gasket construction
# ============================================================
def init_apollonius(n, seed):
    """Build packing Apollonian gasket style: mutually tangent circles, fill gaps."""
    rng = np.random.RandomState(seed)

    # Start with 3-4 large mutually tangent circles
    n_seed = rng.randint(3, 6)

    # Place seed circles
    x, y, r = [], [], []

    if n_seed == 3:
        # Three mutually tangent circles
        r1 = rng.uniform(0.15, 0.22)
        x.append(0.5); y.append(0.3); r.append(r1)

        r2 = rng.uniform(0.12, 0.20)
        angle = rng.uniform(np.pi/6, np.pi/3)
        d12 = r1 + r2
        x.append(0.5 + d12*np.cos(angle)); y.append(0.3 + d12*np.sin(angle)); r.append(r2)

        r3 = rng.uniform(0.12, 0.20)
        angle2 = rng.uniform(2*np.pi/3, 5*np.pi/6)
        d13 = r1 + r3
        x.append(0.5 + d13*np.cos(angle2)); y.append(0.3 + d13*np.sin(angle2)); r.append(r3)
    elif n_seed == 4:
        # Four circles in a diamond
        cr = rng.uniform(0.12, 0.18)
        offsets = [(0, -0.2), (0.2, 0), (0, 0.2), (-0.2, 0)]
        for ox, oy in offsets:
            x.append(0.5 + ox + rng.normal(0, 0.02))
            y.append(0.5 + oy + rng.normal(0, 0.02))
            r.append(cr * rng.uniform(0.85, 1.15))
    else:  # 5
        # Pentagon
        cr = rng.uniform(0.10, 0.15)
        ring_r = rng.uniform(0.2, 0.3)
        for k in range(5):
            angle = 2*np.pi*k/5 + rng.uniform(-0.1, 0.1)
            x.append(0.5 + ring_r*np.cos(angle))
            y.append(0.5 + ring_r*np.sin(angle))
            r.append(cr * rng.uniform(0.85, 1.15))

    # Now greedily add circles in the largest gaps
    for _ in range(n - n_seed):
        # Find the largest gap by sampling
        best_r_new = 0
        best_pos = None

        xa, ya, ra = np.array(x), np.array(y), np.array(r)

        for _ in range(300):
            cx = rng.uniform(0.05, 0.95)
            cy = rng.uniform(0.05, 0.95)

            # Max radius that fits
            wall_limit = min(cx, 1-cx, cy, 1-cy)
            if len(xa) > 0:
                dists = np.sqrt((xa - cx)**2 + (ya - cy)**2)
                circle_limit = np.min(dists - ra)
                max_r = min(wall_limit, circle_limit)
            else:
                max_r = wall_limit

            if max_r > best_r_new:
                best_r_new = max_r
                best_pos = (cx, cy, max(max_r * 0.95, 0.02))

        if best_pos is None:
            best_pos = (rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9), 0.03)

        x.append(best_pos[0]); y.append(best_pos[1]); r.append(best_pos[2])

    x = np.array(x[:n]); y = np.array(y[:n]); r = np.array(r[:n])
    r = np.maximum(r, 0.02)
    x = np.clip(x, r + 0.001, 1 - r - 0.001)
    y = np.clip(y, r + 0.001, 1 - r - 0.001)
    return x, y, r

# ============================================================
# Stripe arrangements
# ============================================================
def init_stripes(n, seed):
    """Horizontal stripes of circles."""
    rng = np.random.RandomState(seed)
    n_rows = rng.randint(4, 8)

    # Distribute n circles across rows
    counts = np.zeros(n_rows, dtype=int)
    for i in range(n):
        counts[i % n_rows] += 1

    x, y, r = [], [], []
    row_height = 1.0 / n_rows

    for row in range(n_rows):
        n_in_row = counts[row]
        if n_in_row == 0:
            continue
        row_r = row_height / 2 * rng.uniform(0.7, 0.95)
        cy = (row + 0.5) * row_height

        for k in range(n_in_row):
            cx = (k + 0.5) / n_in_row
            x.append(cx + rng.normal(0, 0.01))
            y.append(cy + rng.normal(0, 0.01))
            r.append(row_r * rng.uniform(0.8, 1.1))

    x = np.array(x[:n]); y = np.array(y[:n]); r = np.array(r[:n])
    r = np.maximum(r, 0.02)
    x = np.clip(x, r + 0.001, 1 - r - 0.001)
    y = np.clip(y, r + 0.001, 1 - r - 0.001)
    return x, y, r

# ============================================================
# Golden spiral
# ============================================================
def init_spiral(n, seed):
    """Golden spiral placement."""
    rng = np.random.RandomState(seed)
    golden_angle = np.pi * (3 - np.sqrt(5))  # ~137.5 degrees

    x, y, r = [], [], []
    for k in range(n):
        # Fermat spiral
        theta = k * golden_angle + rng.uniform(-0.1, 0.1)
        radius_frac = np.sqrt(k / n) * 0.45
        cx = 0.5 + radius_frac * np.cos(theta)
        cy = 0.5 + radius_frac * np.sin(theta)

        # Radius decreases outward
        cr = rng.uniform(0.06, 0.14) * (1.0 - 0.3 * np.sqrt(k / n))

        x.append(cx); y.append(cy); r.append(cr)

    x = np.array(x[:n]); y = np.array(y[:n]); r = np.array(r[:n])
    r = np.maximum(r, 0.02)
    x = np.clip(x, r + 0.001, 1 - r - 0.001)
    y = np.clip(y, r + 0.001, 1 - r - 0.001)
    return x, y, r

# ============================================================
# Known-structure variants from Cantrell/Specht patterns
# ============================================================
def init_cantrell_style(n, seed):
    """
    Cantrell-style: center + inner ring + outer ring + corner circles.
    But with different ring counts than standard.
    """
    rng = np.random.RandomState(seed)

    # Center radius
    cr = rng.uniform(0.12, 0.20)

    # Try unconventional ring structures
    structures = [
        (1, 6, 12, 3, 4),   # 1+6+12+3+4=26
        (1, 7, 11, 3, 4),   # 1+7+11+3+4=26
        (1, 5, 12, 4, 4),   # 1+5+12+4+4=26
        (1, 9, 8, 4, 4),    # 1+9+8+4+4=26
        (1, 6, 10, 5, 4),   # 1+6+10+5+4=26
        (0, 7, 12, 3, 4),   # 0+7+12+3+4=26 (no center)
        (0, 8, 10, 4, 4),   # 0+8+10+4+4=26 (no center)
        (2, 8, 8, 4, 4),    # 2+8+8+4+4=26 (two center)
        (1, 8, 13, 0, 4),   # 1+8+13+0+4=26 (no outer ring, just corners)
        (1, 10, 11, 0, 4),  # 1+10+11+0+4=26
    ]

    struct = structures[seed % len(structures)]
    n_center, n_ring1, n_ring2, n_ring3, n_corner = struct

    x, y, r = [], [], []

    # Center(s)
    if n_center == 1:
        x.append(0.5 + rng.normal(0, 0.01))
        y.append(0.5 + rng.normal(0, 0.01))
        r.append(cr)
    elif n_center == 2:
        sep = rng.uniform(0.15, 0.25)
        x.append(0.5 - sep/2); y.append(0.5); r.append(cr * 0.9)
        x.append(0.5 + sep/2); y.append(0.5); r.append(cr * 0.9)

    # Ring 1
    if n_ring1 > 0:
        r1_dist = rng.uniform(0.18, 0.28)
        r1_size = rng.uniform(0.08, 0.13)
        for k in range(n_ring1):
            angle = 2*np.pi*k/n_ring1 + rng.uniform(-0.2, 0.2)
            x.append(0.5 + r1_dist*np.cos(angle))
            y.append(0.5 + r1_dist*np.sin(angle))
            r.append(r1_size * rng.uniform(0.85, 1.15))

    # Ring 2
    if n_ring2 > 0:
        r2_dist = rng.uniform(0.33, 0.43)
        r2_size = rng.uniform(0.07, 0.11)
        for k in range(n_ring2):
            angle = 2*np.pi*k/n_ring2 + rng.uniform(-0.15, 0.15)
            x.append(0.5 + r2_dist*np.cos(angle))
            y.append(0.5 + r2_dist*np.sin(angle))
            r.append(r2_size * rng.uniform(0.85, 1.15))

    # Ring 3
    if n_ring3 > 0:
        r3_dist = rng.uniform(0.40, 0.48)
        r3_size = rng.uniform(0.06, 0.10)
        for k in range(n_ring3):
            angle = 2*np.pi*k/n_ring3 + rng.uniform(-0.2, 0.2)
            x.append(0.5 + r3_dist*np.cos(angle))
            y.append(0.5 + r3_dist*np.sin(angle))
            r.append(r3_size * rng.uniform(0.85, 1.15))

    # Corners
    corners = [(0.09, 0.09), (0.91, 0.09), (0.09, 0.91), (0.91, 0.91)]
    for k in range(min(n_corner, 4)):
        cx, cy = corners[k]
        x.append(cx + rng.normal(0, 0.01))
        y.append(cy + rng.normal(0, 0.01))
        r.append(rng.uniform(0.07, 0.11))

    # Fill remainder
    while len(x) < n:
        x.append(rng.uniform(0.15, 0.85))
        y.append(rng.uniform(0.15, 0.85))
        r.append(rng.uniform(0.05, 0.09))

    x = np.array(x[:n]); y = np.array(y[:n]); r = np.array(r[:n])
    r = np.maximum(r, 0.02)
    x = np.clip(x, r + 0.001, 1 - r - 0.001)
    y = np.clip(y, r + 0.001, 1 - r - 0.001)
    return x, y, r

# ============================================================
# "Maximal hole filling" -- greedy with better gap detection
# ============================================================
def init_maxhole(n, seed):
    """Greedily place each circle in the largest available gap."""
    rng = np.random.RandomState(seed)

    x, y, r = np.zeros(n), np.zeros(n), np.zeros(n)

    # Random target sizes, sorted descending
    sizes = rng.uniform(0.05, 0.18, n)
    sizes = np.sort(sizes)[::-1]

    for k in range(n):
        target_r = sizes[k]
        best_pos = None
        best_r = 0

        # Dense sampling of candidate positions
        n_cand = 500 if k < 5 else 200
        cxs = rng.uniform(0.02, 0.98, n_cand)
        cys = rng.uniform(0.02, 0.98, n_cand)

        for ci in range(n_cand):
            cx, cy = cxs[ci], cys[ci]

            # Max radius at this position
            wall_lim = min(cx, 1-cx, cy, 1-cy)

            if k > 0:
                dists = np.sqrt((x[:k] - cx)**2 + (y[:k] - cy)**2)
                circle_lim = np.min(dists - r[:k])
                max_r = min(wall_lim, circle_lim)
            else:
                max_r = wall_lim

            actual_r = min(target_r, max_r * 0.98)

            if actual_r > best_r and actual_r > 0.01:
                best_r = actual_r
                best_pos = (cx, cy, actual_r)

        if best_pos is None:
            best_pos = (rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95), 0.02)

        x[k], y[k], r[k] = best_pos

    r = np.maximum(r, 0.02)
    x = np.clip(x, r + 0.001, 1 - r - 0.001)
    y = np.clip(y, r + 0.001, 1 - r - 0.001)
    return x, y, r

# ============================================================
# Main
# ============================================================
def main():
    t0 = time.time()

    parent_path = os.path.join(WORKDIR, '..', 'topo-001', 'solution_n26.json')
    x0, y0, r0 = load_solution(parent_path)
    parent_metric = np.sum(r0)
    print(f"Parent metric: {parent_metric:.10f}")

    best_metric = parent_metric
    best_x, best_y, best_r = x0.copy(), y0.copy(), r0.copy()
    results = []

    inits = []

    # Apollonius
    for seed in range(200):
        x, y, r = init_apollonius(N, seed)
        inits.append((f"apollo_s{seed}", x, y, r))

    # Stripes
    for seed in range(80):
        x, y, r = init_stripes(N, seed + 40000)
        inits.append((f"stripe_s{seed}", x, y, r))

    # Spiral
    for seed in range(80):
        x, y, r = init_spiral(N, seed + 50000)
        inits.append((f"spiral_s{seed}", x, y, r))

    # Cantrell-style structures
    for seed in range(200):
        x, y, r = init_cantrell_style(N, seed)
        inits.append((f"cantrell_s{seed}", x, y, r))

    # Maxhole greedy
    for seed in range(200):
        x, y, r = init_maxhole(N, seed + 60000)
        inits.append((f"maxhole_s{seed}", x, y, r))

    print(f"Total inits: {len(inits)}")

    for idx, (name, xi, yi, ri) in enumerate(inits):
        x2, y2, r2, metric = full_optimize(xi, yi, ri)

        if metric > 2.55:
            results.append((name, metric))

        if metric > best_metric + 1e-10:
            print(f"  *** IMPROVED: {name} -> {metric:.10f} (+{metric - best_metric:.2e})")
            best_metric = metric
            best_x, best_y, best_r = x2.copy(), y2.copy(), r2.copy()
            save_solution(best_x, best_y, best_r,
                        os.path.join(WORKDIR, 'solution_n26.json'))

        if idx % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{idx}/{len(inits)}] {elapsed:.0f}s | best={best_metric:.10f}")

    elapsed = time.time() - t0
    save_solution(best_x, best_y, best_r, os.path.join(WORKDIR, 'solution_n26.json'))

    results.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "="*60)
    print("SEARCH V3 FINAL RESULTS")
    print("="*60)
    print(f"Parent metric:  {parent_metric:.10f}")
    print(f"Best metric:    {best_metric:.10f}")
    print(f"Delta:          {best_metric - parent_metric:.2e}")
    print(f"Time:           {elapsed:.0f}s")
    print(f"\nTop 20:")
    for name, metric in results[:20]:
        print(f"  {metric:.10f} {name}")

    return best_metric

if __name__ == '__main__':
    main()
