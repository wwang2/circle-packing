"""
Massive random topology search for n=26 circle packing.
Strategy: Generate diverse initial configurations, optimize each, track distinct topologies.
Uses vectorized numpy + multiprocessing for throughput.
"""

import json
import numpy as np
from scipy.optimize import minimize
import os
import time
import sys
from multiprocessing import Pool, cpu_count
import hashlib

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

# ============================================================
# Vectorized SLSQP optimizer with analytical Jacobian
# ============================================================

def make_constraints_fast(n):
    """Build constraints list with analytical Jacobians for SLSQP."""
    constraints = []

    # Containment: x_i - r_i >= 0
    for i in range(n):
        def fun_xl(p, i=i, n=n):
            return p[i] - p[2*n+i]
        def jac_xl(p, i=i, n=n):
            g = np.zeros(3*n)
            g[i] = 1.0
            g[2*n+i] = -1.0
            return g
        constraints.append({'type': 'ineq', 'fun': fun_xl, 'jac': jac_xl})

    # Containment: 1 - x_i - r_i >= 0
    for i in range(n):
        def fun_xr(p, i=i, n=n):
            return 1.0 - p[i] - p[2*n+i]
        def jac_xr(p, i=i, n=n):
            g = np.zeros(3*n)
            g[i] = -1.0
            g[2*n+i] = -1.0
            return g
        constraints.append({'type': 'ineq', 'fun': fun_xr, 'jac': jac_xr})

    # Containment: y_i - r_i >= 0
    for i in range(n):
        def fun_yl(p, i=i, n=n):
            return p[n+i] - p[2*n+i]
        def jac_yl(p, i=i, n=n):
            g = np.zeros(3*n)
            g[n+i] = 1.0
            g[2*n+i] = -1.0
            return g
        constraints.append({'type': 'ineq', 'fun': fun_yl, 'jac': jac_yl})

    # Containment: 1 - y_i - r_i >= 0
    for i in range(n):
        def fun_yr(p, i=i, n=n):
            return 1.0 - p[n+i] - p[2*n+i]
        def jac_yr(p, i=i, n=n):
            g = np.zeros(3*n)
            g[n+i] = -1.0
            g[2*n+i] = -1.0
            return g
        constraints.append({'type': 'ineq', 'fun': fun_yr, 'jac': jac_yr})

    # Positive radius: r_i >= 1e-6
    for i in range(n):
        def fun_rp(p, i=i, n=n):
            return p[2*n+i] - 1e-6
        def jac_rp(p, i=i, n=n):
            g = np.zeros(3*n)
            g[2*n+i] = 1.0
            return g
        constraints.append({'type': 'ineq', 'fun': fun_rp, 'jac': jac_rp})

    # Non-overlap: dist_ij^2 - (r_i+r_j)^2 >= 0
    for i in range(n):
        for j in range(i+1, n):
            def fun_no(p, i=i, j=j, n=n):
                return ((p[i]-p[j])**2 + (p[n+i]-p[n+j])**2 -
                        (p[2*n+i]+p[2*n+j])**2)
            def jac_no(p, i=i, j=j, n=n):
                g = np.zeros(3*n)
                dx = p[i] - p[j]
                dy = p[n+i] - p[n+j]
                sr = p[2*n+i] + p[2*n+j]
                g[i] = 2*dx
                g[j] = -2*dx
                g[n+i] = 2*dy
                g[n+j] = -2*dy
                g[2*n+i] = -2*sr
                g[2*n+j] = -2*sr
                return g
            constraints.append({'type': 'ineq', 'fun': fun_no, 'jac': jac_no})

    return constraints

# Pre-build constraints once
CONSTRAINTS = make_constraints_fast(N)

def obj_fun(p, n=N):
    return -np.sum(p[2*n:3*n])

def obj_jac(p, n=N):
    g = np.zeros(3*n)
    g[2*n:3*n] = -1.0
    return g

def optimize_slsqp(x0, y0, r0, maxiter=5000):
    """SLSQP with analytical Jacobians."""
    n = len(x0)
    params0 = np.concatenate([x0, y0, r0])

    result = minimize(
        obj_fun, params0,
        jac=obj_jac,
        method='SLSQP',
        constraints=CONSTRAINTS,
        options={'maxiter': maxiter, 'ftol': 1e-15, 'disp': False}
    )

    x = result.x[:n]
    y = result.x[n:2*n]
    r = result.x[2*n:3*n]
    return x, y, r, np.sum(r), result.success

def penalty_objective_grad(params, n, pw):
    """Vectorized penalty objective with gradient."""
    x = params[:n]
    y = params[n:2*n]
    r = params[2*n:3*n]

    grad = np.zeros(3*n)
    grad[2*n:3*n] = -1.0

    obj = -np.sum(r)
    penalty = 0.0

    # Containment penalties (vectorized)
    vl = np.maximum(0, r - x)
    vr = np.maximum(0, x + r - 1)
    vb = np.maximum(0, r - y)
    vt = np.maximum(0, y + r - 1)
    vneg = np.maximum(0, -r)

    penalty += np.sum(vl**2 + vr**2 + vb**2 + vt**2 + vneg**2)

    grad[:n] += pw * (-2*vl + 2*vr)
    grad[n:2*n] += pw * (-2*vb + 2*vt)
    grad[2*n:3*n] += pw * (2*vl + 2*vr + 2*vb + 2*vt + 2*vneg)

    # Non-overlap (vectorized using broadcasting)
    dx = x[:, None] - x[None, :]  # n x n
    dy = y[:, None] - y[None, :]
    sr = r[:, None] + r[None, :]
    dist2 = dx**2 + dy**2
    overlap_mask = (dist2 < sr**2)
    np.fill_diagonal(overlap_mask, False)

    # Only upper triangle
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
    """Progressive penalty optimization with L-BFGS-B."""
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

    x = params[:n]; y = params[n:2*n]; r = params[2*n:3*n]
    return x, y, r

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

def get_contact_hash(x, y, r, tol=1e-4):
    """Get a topology hash from the contact graph."""
    n = len(x)
    contacts = []
    # Circle-circle contacts
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            gap = dist - (r[i] + r[j])
            if abs(gap) < tol:
                contacts.append(('cc', i, j))
    # Wall contacts
    sorted_idx = np.argsort(-r)  # Sort by radius descending for canonical labeling
    rank = np.empty(n, dtype=int)
    rank[sorted_idx] = np.arange(n)

    wall_contacts = []
    for i in range(n):
        if abs(x[i] - r[i]) < tol: wall_contacts.append(('wl', rank[i]))
        if abs(1 - x[i] - r[i]) < tol: wall_contacts.append(('wr', rank[i]))
        if abs(y[i] - r[i]) < tol: wall_contacts.append(('wb', rank[i]))
        if abs(1 - y[i] - r[i]) < tol: wall_contacts.append(('wt', rank[i]))

    cc_sorted = sorted([(rank[i], rank[j]) if rank[i] < rank[j] else (rank[j], rank[i])
                         for _, i, j in contacts])
    all_contacts = sorted(wall_contacts) + cc_sorted
    return hashlib.md5(str(all_contacts).encode()).hexdigest()[:12]

# ============================================================
# Initialization generators
# ============================================================

def init_greedy_constructive(n, seed, n_large=None, r_large_range=None):
    """Place circles greedily: largest first, each in best gap."""
    rng = np.random.RandomState(seed)

    if n_large is None:
        n_large = rng.randint(2, 8)
    if r_large_range is None:
        r_large_range = (0.10, 0.20)

    # Generate target radii
    radii = np.zeros(n)
    radii[:n_large] = rng.uniform(r_large_range[0], r_large_range[1], n_large)
    radii[n_large:] = rng.uniform(0.04, 0.10, n - n_large)
    radii = np.sort(radii)[::-1]  # Place largest first

    x = np.zeros(n)
    y = np.zeros(n)
    r = np.zeros(n)

    for k in range(n):
        rk = radii[k]
        best_pos = None
        best_score = -np.inf

        # Try many random positions
        n_tries = 200 if k < 5 else 100
        for _ in range(n_tries):
            cx = rng.uniform(rk + 0.001, 1 - rk - 0.001)
            cy = rng.uniform(rk + 0.001, 1 - rk - 0.001)

            # Check overlap with placed circles
            if k > 0:
                dists = np.sqrt((x[:k] - cx)**2 + (y[:k] - cy)**2)
                min_gap = np.min(dists - r[:k])
                if min_gap < rk:
                    # Shrink radius to fit
                    rk_adj = min(rk, min_gap * 0.95)
                    if rk_adj < 0.02:
                        continue
                else:
                    rk_adj = rk
            else:
                rk_adj = rk
                min_gap = 1.0

            # Score: prefer positions that use space well
            wall_dist = min(cx - rk_adj, 1 - cx - rk_adj, cy - rk_adj, 1 - cy - rk_adj)
            score = rk_adj - 0.1 * max(0, wall_dist - 0.01)

            if score > best_score:
                best_score = score
                best_pos = (cx, cy, rk_adj)

        if best_pos is None:
            # Fallback: place randomly with small radius
            cx = rng.uniform(0.05, 0.95)
            cy = rng.uniform(0.05, 0.95)
            best_pos = (cx, cy, 0.03)

        x[k], y[k], r[k] = best_pos

    return x, y, r

def init_grid_jittered(n, seed, rows=None, cols=None):
    """Place circles on a jittered grid."""
    rng = np.random.RandomState(seed)
    if rows is None:
        rows = rng.choice([4, 5, 6])
    if cols is None:
        cols = rng.choice([5, 6, 7])

    positions = []
    for i in range(rows):
        for j in range(cols):
            cx = (j + 0.5) / cols + rng.normal(0, 0.02)
            cy = (i + 0.5) / rows + rng.normal(0, 0.02)
            positions.append((cx, cy))

    rng.shuffle(positions)
    positions = positions[:n]
    while len(positions) < n:
        positions.append((rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)))

    x = np.array([p[0] for p in positions])
    y = np.array([p[1] for p in positions])

    # Varied radii
    r = rng.uniform(0.06, 0.12, n)
    r = np.maximum(r, 0.02)
    x = np.clip(x, r + 0.001, 1 - r - 0.001)
    y = np.clip(y, r + 0.001, 1 - r - 0.001)
    return x, y, r

def init_poisson_disk(n, seed, min_dist=None):
    """Poisson disk sampling with varied radii."""
    rng = np.random.RandomState(seed)
    if min_dist is None:
        min_dist = rng.uniform(0.10, 0.18)

    points = []
    for _ in range(n * 50):
        cx = rng.uniform(0.05, 0.95)
        cy = rng.uniform(0.05, 0.95)
        if len(points) > 0:
            dists = np.sqrt(np.sum((np.array(points) - [cx, cy])**2, axis=1))
            if np.min(dists) < min_dist:
                continue
        points.append([cx, cy])
        if len(points) >= n:
            break

    while len(points) < n:
        points.append([rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95)])

    x = np.array([p[0] for p in points[:n]])
    y = np.array([p[1] for p in points[:n]])
    r = rng.uniform(0.05, 0.12, n)
    r = np.maximum(r, 0.02)
    x = np.clip(x, r + 0.001, 1 - r - 0.001)
    y = np.clip(y, r + 0.001, 1 - r - 0.001)
    return x, y, r

def init_cluster(n, seed, n_clusters=None):
    """Cluster-based: place clusters of circles around random centers."""
    rng = np.random.RandomState(seed)
    if n_clusters is None:
        n_clusters = rng.randint(2, 6)

    # Cluster centers
    cluster_cx = rng.uniform(0.2, 0.8, n_clusters)
    cluster_cy = rng.uniform(0.2, 0.8, n_clusters)
    cluster_r = rng.uniform(0.12, 0.25, n_clusters)

    # Assign circles to clusters
    assignment = rng.randint(0, n_clusters, n)

    x = np.zeros(n)
    y = np.zeros(n)
    r = np.zeros(n)

    for k in range(n):
        cl = assignment[k]
        angle = rng.uniform(0, 2*np.pi)
        dist = rng.uniform(0, cluster_r[cl])
        x[k] = cluster_cx[cl] + dist * np.cos(angle)
        y[k] = cluster_cy[cl] + dist * np.sin(angle)
        r[k] = rng.uniform(0.04, 0.12)

    r = np.maximum(r, 0.02)
    x = np.clip(x, r + 0.001, 1 - r - 0.001)
    y = np.clip(y, r + 0.001, 1 - r - 0.001)
    return x, y, r

def init_ring_variant(n, seed, n_inner=None, n_outer=None, center_r=None):
    """Ring with configurable parameters."""
    rng = np.random.RandomState(seed)
    if n_inner is None:
        n_inner = rng.randint(5, 10)
    if center_r is None:
        center_r = rng.uniform(0.10, 0.18)

    x, y, r = [0.5 + rng.normal(0, 0.01)], [0.5 + rng.normal(0, 0.01)], [center_r]

    # Inner ring
    r_ring1 = rng.uniform(0.18, 0.28)
    for i in range(n_inner):
        angle = 2*np.pi*i/n_inner + rng.uniform(-0.3, 0.3)
        x.append(0.5 + r_ring1*np.cos(angle))
        y.append(0.5 + r_ring1*np.sin(angle))
        r.append(rng.uniform(0.08, 0.13))

    # Fill remainder
    placed = 1 + n_inner
    remaining = n - placed
    if remaining > 0:
        # Outer positions
        r_ring2 = rng.uniform(0.35, 0.45)
        if n_outer is None:
            n_outer = min(remaining, rng.randint(max(1, remaining-4), remaining+1))
        n_outer = min(n_outer, remaining)

        for i in range(n_outer):
            angle = 2*np.pi*i/max(n_outer, 1) + rng.uniform(-0.2, 0.2)
            x.append(0.5 + r_ring2*np.cos(angle))
            y.append(0.5 + r_ring2*np.sin(angle))
            r.append(rng.uniform(0.07, 0.11))

        # Corners/edges for the rest
        corners = [(0.09, 0.09), (0.91, 0.09), (0.09, 0.91), (0.91, 0.91),
                    (0.5, 0.09), (0.5, 0.91), (0.09, 0.5), (0.91, 0.5)]
        rng.shuffle(corners)
        for cx, cy in corners:
            if len(x) >= n:
                break
            x.append(cx + rng.normal(0, 0.02))
            y.append(cy + rng.normal(0, 0.02))
            r.append(rng.uniform(0.06, 0.10))

    while len(x) < n:
        x.append(rng.uniform(0.15, 0.85))
        y.append(rng.uniform(0.15, 0.85))
        r.append(0.05)

    x = np.array(x[:n]); y = np.array(y[:n]); r = np.array(r[:n])
    r = np.maximum(r, 0.02)
    x = np.clip(x, r + 0.001, 1 - r - 0.001)
    y = np.clip(y, r + 0.001, 1 - r - 0.001)
    return x, y, r

def init_asymmetric(n, seed):
    """Deliberately asymmetric layout -- break the near-symmetry of known optimal."""
    rng = np.random.RandomState(seed)

    x, y, r = [], [], []

    # One big circle off-center
    big_r = rng.uniform(0.14, 0.22)
    big_x = rng.uniform(0.25, 0.75)
    big_y = rng.uniform(0.25, 0.75)
    x.append(big_x); y.append(big_y); r.append(big_r)

    # Fill rest with varied sizes
    for _ in range(n - 1):
        ri = rng.uniform(0.04, 0.14)
        xi = rng.uniform(ri + 0.001, 1 - ri - 0.001)
        yi = rng.uniform(ri + 0.001, 1 - ri - 0.001)
        x.append(xi); y.append(yi); r.append(ri)

    x = np.array(x[:n]); y = np.array(y[:n]); r = np.array(r[:n])
    r = np.maximum(r, 0.02)
    x = np.clip(x, r + 0.001, 1 - r - 0.001)
    y = np.clip(y, r + 0.001, 1 - r - 0.001)
    return x, y, r

def init_two_big(n, seed):
    """Two large circles with smaller ones packed around."""
    rng = np.random.RandomState(seed)

    r1 = rng.uniform(0.15, 0.22)
    r2 = rng.uniform(0.15, 0.22)

    # Place two big circles
    x1 = rng.uniform(0.25, 0.45)
    y1 = rng.uniform(0.3, 0.7)
    x2 = rng.uniform(0.55, 0.75)
    y2 = rng.uniform(0.3, 0.7)

    x = [x1, x2]
    y = [y1, y2]
    r = [r1, r2]

    # Fill rest
    for _ in range(n - 2):
        ri = rng.uniform(0.04, 0.12)
        xi = rng.uniform(ri + 0.001, 1 - ri - 0.001)
        yi = rng.uniform(ri + 0.001, 1 - ri - 0.001)
        x.append(xi); y.append(yi); r.append(ri)

    x = np.array(x[:n]); y = np.array(y[:n]); r = np.array(r[:n])
    r = np.maximum(r, 0.02)
    x = np.clip(x, r + 0.001, 1 - r - 0.001)
    y = np.clip(y, r + 0.001, 1 - r - 0.001)
    return x, y, r

def init_edge_heavy(n, seed):
    """Many circles along edges, few in middle."""
    rng = np.random.RandomState(seed)

    n_edge = rng.randint(n//2, 3*n//4)
    n_interior = n - n_edge

    x, y, r = [], [], []

    # Edge circles
    for _ in range(n_edge):
        side = rng.randint(4)
        ri = rng.uniform(0.05, 0.12)
        if side == 0:  # bottom
            x.append(rng.uniform(ri, 1-ri)); y.append(ri); r.append(ri)
        elif side == 1:  # top
            x.append(rng.uniform(ri, 1-ri)); y.append(1-ri); r.append(ri)
        elif side == 2:  # left
            x.append(ri); y.append(rng.uniform(ri, 1-ri)); r.append(ri)
        else:  # right
            x.append(1-ri); y.append(rng.uniform(ri, 1-ri)); r.append(ri)

    # Interior circles (larger)
    for _ in range(n_interior):
        ri = rng.uniform(0.08, 0.16)
        x.append(rng.uniform(0.2, 0.8))
        y.append(rng.uniform(0.2, 0.8))
        r.append(ri)

    x = np.array(x[:n]); y = np.array(y[:n]); r = np.array(r[:n])
    r = np.maximum(r, 0.02)
    x = np.clip(x, r + 0.001, 1 - r - 0.001)
    y = np.clip(y, r + 0.001, 1 - r - 0.001)
    return x, y, r


# ============================================================
# Perturbation of known best
# ============================================================

def perturb_explode_region(x0, y0, r0, seed):
    """Take a cluster of adjacent circles and randomize them."""
    rng = np.random.RandomState(seed)
    n = len(x0)
    x, y, r = x0.copy(), y0.copy(), r0.copy()

    # Pick a center circle
    center = rng.randint(n)
    dists = np.sqrt((x - x[center])**2 + (y - y[center])**2)
    k = rng.randint(3, min(8, n))
    neighbors = np.argsort(dists)[:k]

    # Randomize these circles
    cx, cy = np.mean(x[neighbors]), np.mean(y[neighbors])
    spread = rng.uniform(0.1, 0.3)
    for idx in neighbors:
        x[idx] = cx + rng.normal(0, spread)
        y[idx] = cy + rng.normal(0, spread)
        r[idx] = rng.uniform(0.04, 0.14)

    r = np.maximum(r, 0.02)
    x = np.clip(x, r + 0.001, 1 - r - 0.001)
    y = np.clip(y, r + 0.001, 1 - r - 0.001)
    return x, y, r

def perturb_rotate_cluster(x0, y0, r0, seed):
    """Rotate a subset of circles around their centroid."""
    rng = np.random.RandomState(seed)
    n = len(x0)
    x, y, r = x0.copy(), y0.copy(), r0.copy()

    k = rng.randint(3, n//2)
    idxs = rng.choice(n, k, replace=False)
    angle = rng.uniform(0.1, np.pi)

    cx, cy = np.mean(x[idxs]), np.mean(y[idxs])
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    for idx in idxs:
        dx, dy = x[idx] - cx, y[idx] - cy
        x[idx] = cx + cos_a*dx - sin_a*dy
        y[idx] = cy + sin_a*dx + cos_a*dy

    x = np.clip(x, r + 0.001, 1 - r - 0.001)
    y = np.clip(y, r + 0.001, 1 - r - 0.001)
    return x, y, r

def perturb_mirror_region(x0, y0, r0, seed):
    """Mirror a subset across an axis."""
    rng = np.random.RandomState(seed)
    n = len(x0)
    x, y, r = x0.copy(), y0.copy(), r0.copy()

    k = rng.randint(3, n//2)
    idxs = rng.choice(n, k, replace=False)

    axis = rng.choice(['x', 'y', 'diag'])
    if axis == 'x':
        y[idxs] = 1.0 - y[idxs]
    elif axis == 'y':
        x[idxs] = 1.0 - x[idxs]
    else:
        x[idxs], y[idxs] = y[idxs].copy(), x[idxs].copy()

    x = np.clip(x, r + 0.001, 1 - r - 0.001)
    y = np.clip(y, r + 0.001, 1 - r - 0.001)
    return x, y, r

def perturb_redistribute_sizes(x0, y0, r0, seed):
    """Change the size distribution: make some larger, others smaller."""
    rng = np.random.RandomState(seed)
    n = len(x0)
    x, y, r = x0.copy(), y0.copy(), r0.copy()

    # Pick K circles to enlarge, shrink the rest
    k_enlarge = rng.randint(2, 8)
    idxs = rng.choice(n, k_enlarge, replace=False)
    factor = rng.uniform(1.1, 1.5)

    r[idxs] *= factor
    others = np.setdiff1d(np.arange(n), idxs)
    r[others] *= rng.uniform(0.7, 0.95)

    r = np.maximum(r, 0.02)
    x = np.clip(x, r + 0.001, 1 - r - 0.001)
    y = np.clip(y, r + 0.001, 1 - r - 0.001)
    return x, y, r

def perturb_swap_and_resize(x0, y0, r0, seed):
    """Swap positions of some circles and resize them."""
    rng = np.random.RandomState(seed)
    n = len(x0)
    x, y, r = x0.copy(), y0.copy(), r0.copy()

    n_swaps = rng.randint(2, 6)
    for _ in range(n_swaps):
        i, j = rng.choice(n, 2, replace=False)
        x[i], x[j] = x[j], x[i]
        y[i], y[j] = y[j], y[i]
        # Don't swap radii -- this changes the topology

    # Also perturb some radii
    k = rng.randint(3, 10)
    idxs = rng.choice(n, k, replace=False)
    r[idxs] *= rng.uniform(0.8, 1.3, k)

    r = np.maximum(r, 0.02)
    x = np.clip(x, r + 0.001, 1 - r - 0.001)
    y = np.clip(y, r + 0.001, 1 - r - 0.001)
    return x, y, r

def perturb_remove_readd(x0, y0, r0, seed):
    """Remove N circles, re-optimize remaining, then re-add."""
    rng = np.random.RandomState(seed)
    n = len(x0)
    x, y, r = x0.copy(), y0.copy(), r0.copy()

    # Remove 2-5 circles (replace with random positions/sizes)
    n_remove = rng.randint(2, 6)
    remove_idxs = rng.choice(n, n_remove, replace=False)

    for idx in remove_idxs:
        x[idx] = rng.uniform(0.1, 0.9)
        y[idx] = rng.uniform(0.1, 0.9)
        r[idx] = rng.uniform(0.04, 0.10)

    r = np.maximum(r, 0.02)
    x = np.clip(x, r + 0.001, 1 - r - 0.001)
    y = np.clip(y, r + 0.001, 1 - r - 0.001)
    return x, y, r

# ============================================================
# Worker function for multiprocessing
# ============================================================

def optimize_one(args):
    """Optimize a single initialization. Returns (metric, x, y, r, name)."""
    name, x, y, r = args

    try:
        # Phase 1: penalty optimization
        x, y, r = optimize_penalty(x, y, r, maxiter=1500)

        # Check if promising
        if not is_feasible(x, y, r, tol=1e-3):
            metric_est = np.sum(r) - 0.1  # penalize infeasible
        else:
            metric_est = np.sum(r)

        if metric_est < 2.40:
            return (metric_est, None, None, None, name, False)

        # Phase 2: SLSQP polish
        x2, y2, r2, metric, success = optimize_slsqp(x, y, r, maxiter=8000)

        if success and is_feasible(x2, y2, r2):
            return (metric, x2, y2, r2, name, True)
        else:
            return (np.sum(r), x, y, r, name, False)
    except Exception as e:
        return (0.0, None, None, None, name, False)


def main():
    t0 = time.time()

    # Load parent solution
    parent_path = os.path.join(WORKDIR, '..', 'topo-001', 'solution_n26.json')
    x0, y0, r0 = load_solution(parent_path)
    parent_metric = np.sum(r0)
    print(f"Parent metric: {parent_metric:.10f}")
    print(f"Best known: {BEST_KNOWN:.10f}")

    best_metric = parent_metric
    best_x, best_y, best_r = x0.copy(), y0.copy(), r0.copy()
    topologies_seen = set()
    results_log = []

    n_workers = max(1, cpu_count() - 1)
    print(f"Using {n_workers} workers")

    # ============================================================
    # PHASE A: Random from scratch (massive diversity)
    # ============================================================
    print("\n" + "="*60)
    print("PHASE A: Random topology generation from scratch")
    print("="*60)

    batch_tasks = []

    # Greedy constructive with varied size distributions
    for n_large in range(1, 9):
        for seed in range(30):
            x, y, r = init_greedy_constructive(N, seed + n_large*1000, n_large=n_large)
            batch_tasks.append((f"greedy_L{n_large}_s{seed}", x, y, r))

    # Grid jittered
    for rows in [4, 5, 6]:
        for cols in [5, 6, 7]:
            for seed in range(10):
                x, y, r = init_grid_jittered(N, seed + rows*100 + cols*10)
                batch_tasks.append((f"grid_{rows}x{cols}_s{seed}", x, y, r))

    # Poisson disk
    for seed in range(80):
        x, y, r = init_poisson_disk(N, seed + 50000)
        batch_tasks.append((f"poisson_s{seed}", x, y, r))

    # Cluster-based
    for n_cl in range(2, 7):
        for seed in range(20):
            x, y, r = init_cluster(N, seed + n_cl * 2000)
            batch_tasks.append((f"cluster_c{n_cl}_s{seed}", x, y, r))

    # Ring variants
    for n_inner in range(5, 12):
        for seed in range(20):
            x, y, r = init_ring_variant(N, seed + n_inner * 3000, n_inner=n_inner)
            batch_tasks.append((f"ring_i{n_inner}_s{seed}", x, y, r))

    # Asymmetric
    for seed in range(60):
        x, y, r = init_asymmetric(N, seed + 60000)
        batch_tasks.append((f"asym_s{seed}", x, y, r))

    # Two big
    for seed in range(60):
        x, y, r = init_two_big(N, seed + 70000)
        batch_tasks.append((f"twobig_s{seed}", x, y, r))

    # Edge heavy
    for seed in range(40):
        x, y, r = init_edge_heavy(N, seed + 80000)
        batch_tasks.append((f"edge_s{seed}", x, y, r))

    print(f"Phase A: {len(batch_tasks)} initializations")

    # Process in chunks to show progress
    chunk_size = 50
    n_improved = 0
    n_valid = 0

    for chunk_start in range(0, len(batch_tasks), chunk_size):
        chunk = batch_tasks[chunk_start:chunk_start+chunk_size]
        # Use sequential (multiprocessing has overhead with constraints closure)
        for task in chunk:
            name, xi, yi, ri = task
            result = optimize_one(task)
            metric, xr, yr, rr, rname, valid = result

            if valid and xr is not None:
                n_valid += 1
                topo_hash = get_contact_hash(xr, yr, rr)
                topologies_seen.add(topo_hash)

                if metric > best_metric + 1e-10:
                    n_improved += 1
                    print(f"  *** IMPROVED: {rname} -> {metric:.10f} (+{metric-best_metric:.2e}) topo={topo_hash}")
                    best_metric = metric
                    best_x, best_y, best_r = xr.copy(), yr.copy(), rr.copy()
                    save_solution(best_x, best_y, best_r,
                                os.path.join(WORKDIR, 'solution_n26.json'))

                results_log.append((rname, metric, topo_hash))

        elapsed = time.time() - t0
        done = min(chunk_start + chunk_size, len(batch_tasks))
        print(f"  [{done}/{len(batch_tasks)}] {elapsed:.0f}s | valid={n_valid} | topos={len(topologies_seen)} | best={best_metric:.10f}")

    # ============================================================
    # PHASE B: Perturbation of known best
    # ============================================================
    print("\n" + "="*60)
    print("PHASE B: Multi-contact perturbation of known best")
    print("="*60)

    perturb_fns = [
        perturb_explode_region,
        perturb_rotate_cluster,
        perturb_mirror_region,
        perturb_redistribute_sizes,
        perturb_swap_and_resize,
        perturb_remove_readd,
    ]

    n_perturb = 600
    n_perturb_improved = 0
    perturb_no_improve = 0

    for pidx in range(n_perturb):
        seed = pidx + 100000
        fn = perturb_fns[pidx % len(perturb_fns)]
        xi, yi, ri = fn(x0, y0, r0, seed)

        result = optimize_one((f"perturb_{fn.__name__}_s{seed}", xi, yi, ri))
        metric, xr, yr, rr, rname, valid = result

        if valid and xr is not None:
            topo_hash = get_contact_hash(xr, yr, rr)
            topologies_seen.add(topo_hash)

            if metric > best_metric + 1e-10:
                n_perturb_improved += 1
                perturb_no_improve = 0
                print(f"  *** IMPROVED: {rname} -> {metric:.10f} (+{metric-best_metric:.2e}) topo={topo_hash}")
                best_metric = metric
                best_x, best_y, best_r = xr.copy(), yr.copy(), rr.copy()
                save_solution(best_x, best_y, best_r,
                            os.path.join(WORKDIR, 'solution_n26.json'))
            else:
                perturb_no_improve += 1

            results_log.append((rname, metric, topo_hash))

        if pidx % 200 == 0:
            elapsed = time.time() - t0
            print(f"  [{pidx}/{n_perturb}] {elapsed:.0f}s | topos={len(topologies_seen)} | best={best_metric:.10f}")

    # ============================================================
    # Summary
    # ============================================================
    elapsed = time.time() - t0

    # Save final solution
    save_solution(best_x, best_y, best_r, os.path.join(WORKDIR, 'solution_n26.json'))

    # Sort results by metric
    results_log.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Parent metric:  {parent_metric:.10f}")
    print(f"Best metric:    {best_metric:.10f}")
    print(f"Delta:          {best_metric - parent_metric:.2e}")
    print(f"Best known:     {BEST_KNOWN:.10f}")
    print(f"vs best known:  {best_metric - BEST_KNOWN:.2e}")
    print(f"Unique topologies found: {len(topologies_seen)}")
    print(f"Total time: {elapsed:.0f}s")

    print(f"\nTop 20 results:")
    for name, metric, topo in results_log[:20]:
        print(f"  {metric:.10f} [{topo}] {name}")

    # Save results summary
    with open(os.path.join(WORKDIR, 'results_summary.json'), 'w') as f:
        json.dump({
            'best_metric': best_metric,
            'parent_metric': parent_metric,
            'n_topologies': len(topologies_seen),
            'total_time': elapsed,
            'top_results': [(n, m, t) for n, m, t in results_log[:50]]
        }, f, indent=2)

    return best_metric

if __name__ == '__main__':
    main()
