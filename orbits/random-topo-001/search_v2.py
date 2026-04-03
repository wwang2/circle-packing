"""
Search v2: Simulated annealing with large-step topology mutations.
Key insight: the standard basin is too deep for local perturbation.
We need to make RADICAL changes -- completely different arrangements.

Strategy:
1. SA that makes very large moves (swap half the circles, etc.)
2. Lattice-based inits (hex, square, mixed) with different packing fractions
3. "Build from corners" -- fill corners first, then edges, then interior
4. Completely random with heavy optimization pressure
"""

import json
import numpy as np
from scipy.optimize import minimize, differential_evolution
import os
import time
import math

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
    """Vectorized penalty objective with gradient."""
    x = params[:n]
    y = params[n:2*n]
    r = params[2*n:3*n]

    grad = np.zeros(3*n)
    grad[2*n:3*n] = -1.0
    obj = -np.sum(r)
    penalty = 0.0

    vl = np.maximum(0, r - x)
    vr_w = np.maximum(0, x + r - 1)
    vb = np.maximum(0, r - y)
    vt = np.maximum(0, y + r - 1)
    vneg = np.maximum(0, -r)

    penalty += np.sum(vl**2 + vr_w**2 + vb**2 + vt**2 + vneg**2)
    grad[:n] += pw * (-2*vl + 2*vr_w)
    grad[n:2*n] += pw * (-2*vb + 2*vt)
    grad[2*n:3*n] += pw * (2*vl + 2*vr_w + 2*vb + 2*vt + 2*vneg)

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    sr = r[:, None] + r[None, :]
    dist2 = dx**2 + dy**2
    overlap_mask = (dist2 < sr**2)
    np.fill_diagonal(overlap_mask, False)
    i_idx, j_idx = np.triu_indices(n, k=1)
    mask_pairs = overlap_mask[i_idx, j_idx]

    if np.any(mask_pairs):
        ii = i_idx[mask_pairs]
        jj = j_idx[mask_pairs]
        dxp = dx[ii, jj]
        dyp = dy[ii, jj]
        d2p = dist2[ii, jj]
        srp = sr[ii, jj]
        distp = np.sqrt(np.maximum(d2p, 1e-20))
        gap = srp - distp
        penalty += np.sum(gap**2)
        coeff = pw * 2 * gap
        dgx = -dxp / distp
        dgy = -dyp / distp
        np.add.at(grad[:n], ii, coeff * dgx)
        np.add.at(grad[:n], jj, -coeff * dgx)
        np.add.at(grad[n:2*n], ii, coeff * dgy)
        np.add.at(grad[n:2*n], jj, -coeff * dgy)
        np.add.at(grad[2*n:3*n], ii, coeff)
        np.add.at(grad[2*n:3*n], jj, coeff)

    return obj + pw * penalty, grad

def optimize_penalty(x, y, r, maxiter=2000):
    n = len(x)
    params = np.concatenate([x, y, r])
    bounds = [(0.001, 0.999)]*n + [(0.001, 0.999)]*n + [(0.005, 0.499)]*n
    for pw in [10, 100, 1000, 10000]:
        result = minimize(
            lambda p: penalty_objective_grad(p, n, pw),
            params, method='L-BFGS-B', jac=True, bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-15}
        )
        params = result.x
    return params[:n], params[n:2*n], params[2*n:3*n]

def optimize_slsqp(x0, y0, r0, maxiter=8000):
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
        lambda p: -np.sum(p[2*n:3*n]), params0,
        method='SLSQP', constraints=constraints,
        options={'maxiter': maxiter, 'ftol': 1e-15, 'disp': False}
    )
    x = result.x[:n]; y = result.x[n:2*n]; r = result.x[2*n:3*n]
    return x, y, r, np.sum(r), result.success

def is_feasible(x, y, r, tol=1e-8):
    n = len(x)
    if np.any(r < -tol): return False
    if np.any(x - r < -tol) or np.any(1 - x - r < -tol): return False
    if np.any(y - r < -tol) or np.any(1 - y - r < -tol): return False
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    sr = r[:, None] + r[None, :]
    dist2 = dx**2 + dy**2
    np.fill_diagonal(dist2, np.inf)
    if np.any(dist2 < sr**2 - tol): return False
    return True

def full_optimize(x, y, r):
    """Penalty + SLSQP pipeline."""
    x, y, r = optimize_penalty(x, y, r, maxiter=1500)
    x2, y2, r2, metric, success = optimize_slsqp(x, y, r, maxiter=8000)
    if is_feasible(x2, y2, r2):
        return x2, y2, r2, metric
    elif is_feasible(x, y, r):
        return x, y, r, np.sum(r)
    return x2, y2, r2, 0.0

# ============================================================
# Strategy 1: Build from corners outward
# ============================================================

def init_corners_first(n, seed):
    """Place 4 corner circles first, then edge midpoints, then fill interior."""
    rng = np.random.RandomState(seed)
    x, y, r = [], [], []

    # 4 corners - varied sizes
    corner_r = rng.uniform(0.08, 0.14, 4)
    for (cx, cy), cr in zip([(0, 0), (1, 0), (0, 1), (1, 1)], corner_r):
        x.append(cr if cx == 0 else 1 - cr)
        y.append(cr if cy == 0 else 1 - cr)
        r.append(cr)

    # 4 edge midpoints
    n_edge_mid = rng.randint(2, 5)
    edge_r = rng.uniform(0.08, 0.14, n_edge_mid)
    edge_positions = [(0.5, 0), (0.5, 1), (0, 0.5), (1, 0.5)]
    rng.shuffle(edge_positions)
    for k in range(n_edge_mid):
        ex, ey = edge_positions[k % 4]
        er = edge_r[k]
        if ey == 0: ey = er
        elif ey == 1: ey = 1 - er
        if ex == 0: ex = er
        elif ex == 1: ex = 1 - er
        x.append(ex + rng.normal(0, 0.02))
        y.append(ey + rng.normal(0, 0.02))
        r.append(er)

    # Fill remaining along edges and interior
    placed = len(x)
    remaining = n - placed

    # Some more edge circles
    n_more_edge = rng.randint(0, min(remaining, 8))
    for _ in range(n_more_edge):
        side = rng.randint(4)
        er = rng.uniform(0.05, 0.12)
        pos = rng.uniform(er + 0.01, 1 - er - 0.01)
        if side == 0: x.append(pos); y.append(er); r.append(er)
        elif side == 1: x.append(pos); y.append(1-er); r.append(er)
        elif side == 2: x.append(er); y.append(pos); r.append(er)
        else: x.append(1-er); y.append(pos); r.append(er)

    # Interior
    while len(x) < n:
        ir = rng.uniform(0.06, 0.15)
        x.append(rng.uniform(0.2, 0.8))
        y.append(rng.uniform(0.2, 0.8))
        r.append(ir)

    x = np.array(x[:n]); y = np.array(y[:n]); r = np.array(r[:n])
    r = np.maximum(r, 0.02)
    x = np.clip(x, r + 0.001, 1 - r - 0.001)
    y = np.clip(y, r + 0.001, 1 - r - 0.001)
    return x, y, r

# ============================================================
# Strategy 2: Hexagonal close-packed with defects
# ============================================================

def init_hex_defect(n, seed):
    """Hex packing with random defects (missing circles, extra circles)."""
    rng = np.random.RandomState(seed)

    # Try different hex sizes
    r_base = rng.uniform(0.07, 0.12)
    positions = []
    row = 0
    yy = r_base
    while yy < 1 - r_base:
        xx = r_base + (row % 2) * r_base
        while xx < 1 - r_base:
            positions.append((xx, yy, r_base * rng.uniform(0.8, 1.2)))
            xx += 2 * r_base * rng.uniform(0.95, 1.05)
        yy += r_base * np.sqrt(3) * rng.uniform(0.95, 1.05)
        row += 1

    rng.shuffle(positions)

    if len(positions) > n:
        positions = positions[:n]

    # Add extra circles in gaps if needed
    while len(positions) < n:
        cx = rng.uniform(0.08, 0.92)
        cy = rng.uniform(0.08, 0.92)
        cr = rng.uniform(0.04, 0.09)
        positions.append((cx, cy, cr))

    x = np.array([p[0] for p in positions])
    y = np.array([p[1] for p in positions])
    r = np.array([p[2] for p in positions])
    r = np.maximum(r, 0.02)
    x = np.clip(x, r + 0.001, 1 - r - 0.001)
    y = np.clip(y, r + 0.001, 1 - r - 0.001)
    return x, y, r

# ============================================================
# Strategy 3: Quasicrystal / Penrose-like arrangement
# ============================================================

def init_quasicrystal(n, seed):
    """Quasicrystal-like arrangement with 5-fold or 7-fold symmetry."""
    rng = np.random.RandomState(seed)
    fold = rng.choice([5, 7, 8])

    x, y, r = [], [], []

    # Center
    cr = rng.uniform(0.10, 0.18)
    x.append(0.5); y.append(0.5); r.append(cr)

    # Rings with quasi-periodic radii
    n_rings = rng.randint(2, 5)
    placed = 1
    for ring in range(n_rings):
        ring_r = 0.15 + ring * rng.uniform(0.10, 0.15)
        n_on_ring = fold + ring * rng.randint(0, 3)
        circle_r = rng.uniform(0.06, 0.12)

        for k in range(n_on_ring):
            if placed >= n:
                break
            angle = 2 * np.pi * k / n_on_ring + rng.uniform(-0.15, 0.15)
            # Add golden ratio perturbation
            angle += ring * 2.399963  # ~137.5 degrees (golden angle)
            cx = 0.5 + ring_r * np.cos(angle)
            cy = 0.5 + ring_r * np.sin(angle)
            x.append(cx); y.append(cy)
            r.append(circle_r * rng.uniform(0.8, 1.2))
            placed += 1

    # Fill remaining
    while len(x) < n:
        x.append(rng.uniform(0.1, 0.9))
        y.append(rng.uniform(0.1, 0.9))
        r.append(rng.uniform(0.05, 0.10))

    x = np.array(x[:n]); y = np.array(y[:n]); r = np.array(r[:n])
    r = np.maximum(r, 0.02)
    x = np.clip(x, r + 0.001, 1 - r - 0.001)
    y = np.clip(y, r + 0.001, 1 - r - 0.001)
    return x, y, r

# ============================================================
# Strategy 4: Size-optimized -- fix positions, optimize sizes
# ============================================================

def init_fixed_topology_varied_sizes(n, seed, x0, y0):
    """Use positions from known solution but randomize sizes."""
    rng = np.random.RandomState(seed)

    # Randomly permute which circle gets which position
    perm = rng.permutation(n)
    x = x0[perm].copy()
    y = y0[perm].copy()

    # Generate completely new size distribution
    style = rng.randint(4)
    if style == 0:
        # Fewer large, more small
        r = np.zeros(n)
        n_large = rng.randint(3, 7)
        r[:n_large] = rng.uniform(0.12, 0.18, n_large)
        r[n_large:] = rng.uniform(0.05, 0.09, n - n_large)
    elif style == 1:
        # More uniform
        r = rng.uniform(0.08, 0.12, n)
    elif style == 2:
        # Power law
        r = 0.05 + 0.15 * rng.power(2, n)
    else:
        # Bimodal
        r = np.where(rng.random(n) > 0.5,
                      rng.uniform(0.10, 0.15, n),
                      rng.uniform(0.05, 0.08, n))

    r = np.maximum(r, 0.02)
    x = np.clip(x, r + 0.001, 1 - r - 0.001)
    y = np.clip(y, r + 0.001, 1 - r - 0.001)
    return x, y, r

# ============================================================
# Strategy 5: Simulated annealing with radical moves
# ============================================================

def simulated_annealing(x0, y0, r0, seed, n_steps=500, T_start=0.05, T_end=0.001):
    """SA that makes large topology-changing moves."""
    rng = np.random.RandomState(seed)
    n = len(x0)

    # Start from optimized solution
    x, y, r = x0.copy(), y0.copy(), r0.copy()
    current_metric = np.sum(r) if is_feasible(x, y, r) else 0.0

    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    best_metric = current_metric

    for step in range(n_steps):
        T = T_start * (T_end / T_start) ** (step / max(n_steps - 1, 1))

        # Make a radical move
        xn, yn, rn = x.copy(), y.copy(), r.copy()
        move = rng.randint(8)

        if move == 0:
            # Scramble half the circles
            k = n // 2
            idxs = rng.choice(n, k, replace=False)
            for idx in idxs:
                xn[idx] = rng.uniform(0.05, 0.95)
                yn[idx] = rng.uniform(0.05, 0.95)
                rn[idx] = rng.uniform(0.04, 0.14)
        elif move == 1:
            # Grow one, shrink all neighbors
            i = rng.randint(n)
            growth = rng.uniform(1.1, 1.5)
            rn[i] *= growth
            dists = np.sqrt((x - x[i])**2 + (y - y[i])**2)
            neighbors = np.argsort(dists)[1:5]
            rn[neighbors] *= rng.uniform(0.6, 0.9, len(neighbors))
        elif move == 2:
            # Merge two small circles into one larger
            sizes = np.argsort(r)
            i, j = sizes[0], sizes[1]
            xn[i] = (x[i] + x[j]) / 2
            yn[i] = (y[i] + y[j]) / 2
            rn[i] = r[i] + r[j]  # Total area preserved approximately
            # Move the other to a gap
            xn[j] = rng.uniform(0.1, 0.9)
            yn[j] = rng.uniform(0.1, 0.9)
            rn[j] = rng.uniform(0.03, 0.07)
        elif move == 3:
            # Split one large into two smaller
            i = np.argmax(r)
            angle = rng.uniform(0, 2*np.pi)
            shift = r[i] * 0.5
            xn[i] = x[i] + shift * np.cos(angle)
            yn[i] = y[i] + shift * np.sin(angle)
            rn[i] = r[i] * 0.7
            # Use a random small circle for the split
            j = np.argmin(r)
            xn[j] = x[i] - shift * np.cos(angle)
            yn[j] = y[i] - shift * np.sin(angle)
            rn[j] = r[i] * 0.5
        elif move == 4:
            # Rotate entire config
            angle = rng.uniform(0, 2*np.pi)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            cx, cy = 0.5, 0.5
            dx, dy = x - cx, y - cy
            xn = cx + cos_a*dx - sin_a*dy
            yn = cy + sin_a*dx + cos_a*dy
            rn = r.copy()
        elif move == 5:
            # Mirror + perturb
            if rng.random() > 0.5:
                xn = 1.0 - x
            else:
                yn = 1.0 - y
            # Perturb a few
            k = rng.randint(2, 6)
            idxs = rng.choice(n, k, replace=False)
            xn[idxs] += rng.normal(0, 0.05, k)
            yn[idxs] += rng.normal(0, 0.05, k)
            rn[idxs] *= rng.uniform(0.8, 1.2, k)
        elif move == 6:
            # Random permutation of positions (keep sizes fixed)
            perm = rng.permutation(n)
            xn = x[perm]
            yn = y[perm]
            rn = r.copy()  # Keep original sizes at original positions
        elif move == 7:
            # Completely random restart
            xn = rng.uniform(0.08, 0.92, n)
            yn = rng.uniform(0.08, 0.92, n)
            rn = rng.uniform(0.04, 0.14, n)

        rn = np.maximum(rn, 0.02)
        xn = np.clip(xn, rn + 0.001, 1 - rn - 0.001)
        yn = np.clip(yn, rn + 0.001, 1 - rn - 0.001)

        # Quick penalty optimization
        xn, yn, rn = optimize_penalty(xn, yn, rn, maxiter=800)

        if is_feasible(xn, yn, rn, tol=1e-5):
            new_metric = np.sum(rn)
        else:
            new_metric = np.sum(rn) - 0.1

        # Acceptance
        delta = new_metric - current_metric
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
            x, y, r = xn, yn, rn
            current_metric = new_metric

            if new_metric > best_metric:
                best_x, best_y, best_r = xn.copy(), yn.copy(), rn.copy()
                best_metric = new_metric

    return best_x, best_y, best_r, best_metric

# ============================================================
# Strategy 6: Differential evolution on the full parameter space
# ============================================================

def de_objective(params):
    """Objective for differential evolution."""
    n = N
    x = params[:n]
    y = params[n:2*n]
    r = params[2*n:3*n]

    penalty = 0.0
    # Containment
    penalty += np.sum(np.maximum(0, r - x)**2)
    penalty += np.sum(np.maximum(0, x + r - 1)**2)
    penalty += np.sum(np.maximum(0, r - y)**2)
    penalty += np.sum(np.maximum(0, y + r - 1)**2)

    # Non-overlap
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    sr = r[:, None] + r[None, :]
    dist2 = dx**2 + dy**2
    np.fill_diagonal(dist2, np.inf)
    violations = np.maximum(0, sr**2 - dist2)
    penalty += np.sum(violations[np.triu_indices(n, k=1)])

    return -np.sum(r) + 1000 * penalty


def main():
    t0 = time.time()

    parent_path = os.path.join(WORKDIR, '..', 'topo-001', 'solution_n26.json')
    x0, y0, r0 = load_solution(parent_path)
    parent_metric = np.sum(r0)
    print(f"Parent metric: {parent_metric:.10f}")

    best_metric = parent_metric
    best_x, best_y, best_r = x0.copy(), y0.copy(), r0.copy()
    results = []

    # ============================================================
    # Phase 1: Diverse structured initializations
    # ============================================================
    print("\n=== Phase 1: Diverse structured inits ===")

    inits = []

    # Corners first
    for seed in range(100):
        x, y, r = init_corners_first(N, seed)
        inits.append((f"corners_s{seed}", x, y, r))

    # Hex defect
    for seed in range(100):
        x, y, r = init_hex_defect(N, seed + 10000)
        inits.append((f"hexdef_s{seed}", x, y, r))

    # Quasicrystal
    for seed in range(100):
        x, y, r = init_quasicrystal(N, seed + 20000)
        inits.append((f"quasi_s{seed}", x, y, r))

    # Fixed topology, varied sizes
    for seed in range(100):
        x, y, r = init_fixed_topology_varied_sizes(N, seed + 30000, x0, y0)
        inits.append((f"fixpos_s{seed}", x, y, r))

    print(f"  {len(inits)} initializations")

    for idx, (name, xi, yi, ri) in enumerate(inits):
        x2, y2, r2, metric = full_optimize(xi, yi, ri)

        if metric > 2.60:
            results.append((name, metric))

        if metric > best_metric + 1e-10:
            print(f"  *** IMPROVED: {name} -> {metric:.10f}")
            best_metric = metric
            best_x, best_y, best_r = x2.copy(), y2.copy(), r2.copy()
            save_solution(best_x, best_y, best_r,
                        os.path.join(WORKDIR, 'solution_n26.json'))

        if idx % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{idx}/{len(inits)}] {elapsed:.0f}s | best={best_metric:.10f}")

    # ============================================================
    # Phase 2: Simulated annealing runs
    # ============================================================
    print("\n=== Phase 2: Simulated annealing ===")

    for sa_seed in range(20):
        # Start from parent
        bx, by, br, bm = simulated_annealing(
            x0, y0, r0, seed=sa_seed*100,
            n_steps=300, T_start=0.1, T_end=0.0005
        )

        # Polish the SA result
        if bm > 2.50:
            bx2, by2, br2, metric = full_optimize(bx, by, br)
            results.append((f"sa_s{sa_seed}", metric))

            if metric > best_metric + 1e-10:
                print(f"  *** SA IMPROVED: sa_s{sa_seed} -> {metric:.10f}")
                best_metric = metric
                best_x, best_y, best_r = bx2.copy(), by2.copy(), br2.copy()
                save_solution(best_x, best_y, best_r,
                            os.path.join(WORKDIR, 'solution_n26.json'))

        elapsed = time.time() - t0
        print(f"  SA {sa_seed}/20 | {elapsed:.0f}s | best_sa={bm:.10f} | polished={metric if bm > 2.50 else 0:.10f}")

    # ============================================================
    # Phase 3: Differential evolution (short)
    # ============================================================
    print("\n=== Phase 3: Differential evolution ===")

    bounds = [(0.01, 0.99)]*N + [(0.01, 0.99)]*N + [(0.02, 0.3)]*N

    for de_seed in range(3):
        print(f"  DE run {de_seed}...")
        result = differential_evolution(
            de_objective, bounds,
            seed=de_seed*42,
            maxiter=200,
            popsize=30,
            tol=1e-12,
            mutation=(0.5, 1.5),
            recombination=0.9,
            polish=False,
        )

        x = result.x[:N]; y = result.x[N:2*N]; r = result.x[2*N:3*N]

        # Polish
        x2, y2, r2, metric = full_optimize(x, y, r)
        results.append((f"de_s{de_seed}", metric))

        if metric > best_metric + 1e-10:
            print(f"  *** DE IMPROVED: de_s{de_seed} -> {metric:.10f}")
            best_metric = metric
            best_x, best_y, best_r = x2.copy(), y2.copy(), r2.copy()
            save_solution(best_x, best_y, best_r,
                        os.path.join(WORKDIR, 'solution_n26.json'))

        elapsed = time.time() - t0
        print(f"  DE {de_seed}: raw={-result.fun:.8f} -> polished={metric:.10f} | {elapsed:.0f}s")

    # ============================================================
    # Summary
    # ============================================================
    elapsed = time.time() - t0

    save_solution(best_x, best_y, best_r, os.path.join(WORKDIR, 'solution_n26.json'))

    results.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "="*60)
    print("SEARCH V2 FINAL RESULTS")
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
