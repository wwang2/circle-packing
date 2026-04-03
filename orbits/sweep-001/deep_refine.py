"""
Deep refinement for circle packing solutions.
sweep-002: close 2-4% gaps to SOTA for n=24,25,27,29,31.

Strategy per n:
  1. Load existing warm-start solution
  2. Multi-start from diverse inits (ring, hex, grid, random, greedy)
     with progressive penalty L-BFGS-B + SLSQP polish
  3. Basin-hopping from best with 2000+ hops
  4. Save improved solution
"""

import numpy as np
from scipy.optimize import minimize
import json
import os
import time
import sys
import itertools


# ── Penalty objective with analytical gradient ──────────────────────────

def penalty_obj_grad(vec, n, mu):
    """Penalty-based objective: -sum(r) + mu * sum(violations^2)."""
    xs = vec[:n]; ys = vec[n:2*n]; rs = vec[2*n:]
    obj = -np.sum(rs)
    gx = np.zeros(n); gy = np.zeros(n); gr = -np.ones(n)

    # Containment
    v = np.maximum(0, rs - xs); obj += mu*np.sum(v**2); gx -= 2*mu*v; gr += 2*mu*v
    v = np.maximum(0, xs+rs-1); obj += mu*np.sum(v**2); gx += 2*mu*v; gr += 2*mu*v
    v = np.maximum(0, rs - ys); obj += mu*np.sum(v**2); gy -= 2*mu*v; gr += 2*mu*v
    v = np.maximum(0, ys+rs-1); obj += mu*np.sum(v**2); gy += 2*mu*v; gr += 2*mu*v

    # Non-overlap
    dx = xs[:,None]-xs[None,:]; dy = ys[:,None]-ys[None,:]
    dist = np.sqrt(np.maximum(dx**2+dy**2, 1e-30))
    rsum = rs[:,None]+rs[None,:]
    mask = np.triu(np.ones((n,n),dtype=bool),k=1)
    olap = np.maximum(0, rsum-dist)*mask
    obj += mu*np.sum(olap**2)

    inv_d = np.where(dist>1e-15, 1.0/dist, 0.0)
    of = 2*mu*olap*inv_d
    gx += np.sum(of*dx,axis=1)-np.sum(of*dx,axis=0)
    gy += np.sum(of*dy,axis=1)-np.sum(of*dy,axis=0)
    or_ = 2*mu*olap
    gr -= np.sum(or_,axis=1)+np.sum(or_,axis=0)

    # Positive radius
    nv = np.maximum(0,-rs); obj += 100*mu*np.sum(nv**2); gr -= 200*mu*nv
    return obj, np.concatenate([gx,gy,gr])


def progressive_lbfgsb(xs, ys, rs, n, max_iter=500, mus=None):
    """Progressive penalty L-BFGS-B optimization."""
    if mus is None:
        mus = [10, 100, 1000, 10000, 100000]
    vec = np.concatenate([xs,ys,rs])
    bds = [(1e-4,1-1e-4)]*n + [(1e-4,1-1e-4)]*n + [(1e-6,0.5)]*n
    for mu in mus:
        r = minimize(lambda v: penalty_obj_grad(v,n,mu), vec, jac=True,
                     method='L-BFGS-B', bounds=bds,
                     options={'maxiter':max_iter,'ftol':1e-15,'gtol':1e-12})
        vec = r.x
    return vec[:n], vec[n:2*n], vec[2*n:]


# ── SLSQP polish with exact constraints ────────────────────────────────

def slsqp_polish(xs, ys, rs, n, max_iter=1000):
    """Polish with SLSQP using exact inequality constraints."""
    vec0 = np.concatenate([xs, ys, rs])

    def objective(vec):
        return -np.sum(vec[2*n:])

    constraints = []
    # Containment: x_i - r_i >= 0
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[i] - v[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[n+i] - v[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[n+i] - v[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*n+i] - 1e-10})

    # Non-overlap: dist(i,j) - r_i - r_j >= 0
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j: np.sqrt((v[i]-v[j])**2 + (v[n+i]-v[n+j])**2) - v[2*n+i] - v[2*n+j]
            })

    bds = [(1e-5, 1-1e-5)]*n + [(1e-5, 1-1e-5)]*n + [(1e-8, 0.5)]*n

    try:
        r = minimize(objective, vec0, method='SLSQP', bounds=bds,
                     constraints=constraints,
                     options={'maxiter': max_iter, 'ftol': 1e-15, 'disp': False})
        xo, yo, ro = r.x[:n], r.x[n:2*n], r.x[2*n:]
        # Verify feasibility
        v, m, _ = validate(xo, yo, ro, n)
        if v and m > np.sum(rs) - 1e-12:
            return xo, yo, ro
    except:
        pass
    return xs, ys, rs


# ── Repair and grow ────────────────────────────────────────────────────

def repair_and_grow(xs, ys, rs, n):
    """Repair violations and grow radii to fill gaps."""
    # Clip to containment
    for i in range(n):
        rs[i] = min(rs[i], xs[i], 1-xs[i], ys[i], 1-ys[i])
    # Resolve overlaps
    for _ in range(500):
        ok = True
        for i in range(n):
            for j in range(i+1, n):
                d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                o = rs[i] + rs[j] - d
                if o > 1e-13:
                    t = rs[i] + rs[j]
                    if t > 0:
                        s = o + 1e-14
                        rs[i] -= s*rs[i]/t
                        rs[j] -= s*rs[j]/t
                    ok = False
        if ok:
            break
    rs = np.maximum(rs, 1e-15)
    # Grow radii
    for _ in range(20):
        ch = False
        for i in range(n):
            mr = min(xs[i], 1-xs[i], ys[i], 1-ys[i])
            for j in range(n):
                if j != i:
                    d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                    mr = min(mr, d - rs[j])
            if mr > rs[i] + 1e-15:
                rs[i] = mr
                ch = True
        if not ch:
            break
    return rs


# ── Local search with grid probing ─────────────────────────────────────

def local_search(xs, ys, rs, n, step=0.01, gp=9, iters=5):
    """Grid-based local search for each circle."""
    for it in range(iters):
        imp = False
        order = np.random.permutation(n)
        offs = np.linspace(-step, step, gp)
        for i in order:
            bx, by, bg = xs[i], ys[i], 0.0
            old_r = rs[i]
            for dx in offs:
                for dy in offs:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = xs[i]+dx, ys[i]+dy
                    if nx < 1e-3 or nx > 1-1e-3 or ny < 1e-3 or ny > 1-1e-3:
                        continue
                    mr = min(nx, 1-nx, ny, 1-ny)
                    for j in range(n):
                        if j == i:
                            continue
                        d = np.sqrt((nx-xs[j])**2 + (ny-ys[j])**2)
                        mr = min(mr, d - rs[j])
                    if mr <= 0:
                        continue
                    g = mr - old_r
                    if g > bg + 1e-15:
                        bx, by, bg = nx, ny, g
                        br = mr
            if bg > 1e-15:
                xs[i], ys[i], rs[i] = bx, by, br
                imp = True
        if not imp:
            break
        step *= 0.55
    return xs, ys, rs


# ── Gradient descent on positions for sum-of-radii ─────────────────────

def gradient_polish(xs, ys, rs, n, steps=200, lr=0.0005):
    """Move circles along gradient of max-feasible-radius."""
    for _ in range(steps):
        moved = False
        for i in range(n):
            # Current max radius
            mr = min(xs[i], 1-xs[i], ys[i], 1-ys[i])
            binding_wall = None
            if mr == xs[i]: binding_wall = 'left'
            elif mr == 1-xs[i]: binding_wall = 'right'
            elif mr == ys[i]: binding_wall = 'bottom'
            elif mr == 1-ys[i]: binding_wall = 'top'

            binding_j = -1
            for j in range(n):
                if j == i:
                    continue
                d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                gap = d - rs[j]
                if gap < mr:
                    mr = gap
                    binding_j = j

            if binding_j >= 0:
                # Move away from binding neighbor
                d = np.sqrt((xs[i]-xs[binding_j])**2 + (ys[i]-ys[binding_j])**2)
                if d > 1e-15:
                    gx = (xs[i]-xs[binding_j])/d
                    gy = (ys[i]-ys[binding_j])/d
                    nx = xs[i] + lr*gx
                    ny = ys[i] + lr*gy
                    nx = np.clip(nx, 1e-4, 1-1e-4)
                    ny = np.clip(ny, 1e-4, 1-1e-4)
                    # Check if this improves
                    nmr = min(nx, 1-nx, ny, 1-ny)
                    for j2 in range(n):
                        if j2 == i:
                            continue
                        dd = np.sqrt((nx-xs[j2])**2 + (ny-ys[j2])**2)
                        nmr = min(nmr, dd - rs[j2])
                    if nmr > mr + 1e-15:
                        xs[i], ys[i] = nx, ny
                        rs[i] = nmr
                        moved = True
        if not moved:
            break
        lr *= 0.995
    return xs, ys, rs


# ── Initialization strategies ──────────────────────────────────────────

def hex_init(n, noise=0.0):
    side = int(np.ceil(np.sqrt(n*2/np.sqrt(3)))) + 1
    pts = []
    for row in range(side+3):
        for col in range(side+3):
            x = (col + 0.5*(row%2) + 0.5)/(side+2)
            y = (row*np.sqrt(3)/2 + 0.5)/(side+2)
            if 0.02 < x < 0.98 and 0.02 < y < 0.98:
                pts.append((x, y))
    pts = np.array(pts)
    if len(pts) >= n:
        sel = [len(pts)//2]
        for _ in range(n-1):
            md = np.min([np.sum((pts-pts[s])**2, axis=1) for s in sel], axis=0)
            md[sel] = -1
            sel.append(np.argmax(md))
        pts = pts[sel]
    else:
        pts = np.vstack([pts, np.random.uniform(0.05, 0.95, (n-len(pts), 2))])
    xs, ys = pts[:n, 0].copy(), pts[:n, 1].copy()
    if noise > 0:
        xs += np.random.randn(n)*noise
        ys += np.random.randn(n)*noise
        xs = np.clip(xs, 0.02, 0.98)
        ys = np.clip(ys, 0.02, 0.98)
    return xs, ys


def grid_init(n, noise=0.0):
    side = int(np.ceil(np.sqrt(n)))
    pts = [((i+0.5)/side, (j+0.5)/side) for i in range(side) for j in range(side)]
    pts = np.array(pts[:n])
    xs, ys = pts[:, 0].copy(), pts[:, 1].copy()
    if noise > 0:
        xs += np.random.randn(n)*noise
        ys += np.random.randn(n)*noise
        xs = np.clip(xs, 0.02, 0.98)
        ys = np.clip(ys, 0.02, 0.98)
    return xs, ys


def ring_init(n):
    """Place circles in concentric rings."""
    pts = []
    # Center circle
    pts.append((0.5, 0.5))
    remaining = n - 1
    ring = 1
    while remaining > 0:
        r_ring = ring * 0.15
        count = min(remaining, int(2 * np.pi * r_ring / 0.12) + 6*ring)
        count = max(count, 1)
        for k in range(count):
            angle = 2 * np.pi * k / count + ring * 0.3
            x = 0.5 + r_ring * np.cos(angle)
            y = 0.5 + r_ring * np.sin(angle)
            x = np.clip(x, 0.03, 0.97)
            y = np.clip(y, 0.03, 0.97)
            pts.append((x, y))
            remaining -= 1
            if remaining == 0:
                break
        ring += 1
    pts = np.array(pts[:n])
    return pts[:, 0].copy(), pts[:, 1].copy()


def greedy_init(n):
    """Greedy constructive: place circles one at a time maximizing radius."""
    xs = np.zeros(n)
    ys = np.zeros(n)
    rs = np.zeros(n)

    # First circle at center
    xs[0], ys[0] = 0.5, 0.5
    rs[0] = 0.5

    for i in range(1, n):
        best_r = 0
        best_x, best_y = 0.5, 0.5
        # Try many candidates
        for _ in range(500):
            cx = np.random.uniform(0.02, 0.98)
            cy = np.random.uniform(0.02, 0.98)
            mr = min(cx, 1-cx, cy, 1-cy)
            for j in range(i):
                d = np.sqrt((cx-xs[j])**2 + (cy-ys[j])**2)
                mr = min(mr, d - rs[j])
            if mr > best_r:
                best_r = mr
                best_x, best_y = cx, cy
        xs[i], ys[i] = best_x, best_y
        rs[i] = max(best_r, 1e-10)

    return xs, ys, rs


def random_init(n):
    m = 0.4/np.sqrt(n)
    return np.random.uniform(m, 1-m, n), np.random.uniform(m, 1-m, n)


def sunflower_init(n):
    """Sunflower/Fibonacci spiral arrangement."""
    golden = (1 + np.sqrt(5)) / 2
    pts = []
    for i in range(n):
        r = 0.45 * np.sqrt((i + 0.5) / n)
        theta = 2 * np.pi * i / golden**2
        x = 0.5 + r * np.cos(theta)
        y = 0.5 + r * np.sin(theta)
        pts.append((np.clip(x, 0.03, 0.97), np.clip(y, 0.03, 0.97)))
    pts = np.array(pts)
    return pts[:, 0].copy(), pts[:, 1].copy()


# ── Validation ─────────────────────────────────────────────────────────

def validate(xs, ys, rs, n, tol=1e-10):
    mv = 0.0
    for i in range(n):
        mv = max(mv, rs[i]-xs[i], xs[i]+rs[i]-1, rs[i]-ys[i], ys[i]+rs[i]-1)
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
            mv = max(mv, rs[i]+rs[j]-d)
    return mv <= tol, np.sum(rs), mv


# ── Full pipeline for one init ─────────────────────────────────────────

def optimize_from_init(xs, ys, n, do_slsqp=True):
    """Run full optimization pipeline from given initial positions."""
    rs = np.full(n, 0.35/np.sqrt(n))

    # Progressive penalty L-BFGS-B
    xs, ys, rs = progressive_lbfgsb(xs, ys, rs, n, max_iter=500,
                                      mus=[10, 100, 1000, 10000, 100000])
    rs = repair_and_grow(xs, ys, rs, n)

    # Local search
    xs, ys, rs = local_search(xs, ys, rs, n, step=0.02, gp=9, iters=5)
    rs = repair_and_grow(xs, ys, rs, n)

    # Gradient polish
    xs, ys, rs = gradient_polish(xs, ys, rs, n, steps=100, lr=0.001)
    rs = repair_and_grow(xs, ys, rs, n)

    # SLSQP polish
    if do_slsqp:
        xs, ys, rs = slsqp_polish(xs, ys, rs, n, max_iter=2000)
        rs = repair_and_grow(xs, ys, rs, n)

    v, m, _ = validate(xs, ys, rs, n)
    if v:
        return xs, ys, rs, m
    return None


# ── Basin hopping ──────────────────────────────────────────────────────

def basin_hop(xs, ys, rs, n, time_budget=120, verbose=True):
    """Basin hopping from a warm start."""
    bxs, bys, brs = xs.copy(), ys.copy(), rs.copy()
    bm = np.sum(rs)
    cxs, cys, crs = xs.copy(), ys.copy(), rs.copy()
    cm = bm

    t0 = time.time()
    stale = 0
    hop = 0
    temp = 0.005  # SA temperature

    while time.time() - t0 < time_budget:
        txs, tys = cxs.copy(), cys.copy()

        # Adaptive perturbation based on staleness
        if stale > 80:
            # Major restart from best
            txs, tys = bxs.copy(), bys.copy()
            txs += np.random.randn(n)*0.08
            tys += np.random.randn(n)*0.08
            stale = 0
        elif stale > 40:
            # Large perturbation
            k = np.random.randint(n//3, 2*n//3+1)
            idx = np.random.choice(n, k, replace=False)
            txs[idx] += np.random.randn(k)*0.05
            tys[idx] += np.random.randn(k)*0.05
        else:
            st = np.random.randint(10)
            if st == 0:
                # Small random perturbation
                k = np.random.randint(1, max(2, n//5))
                idx = np.random.choice(n, k, replace=False)
                s2 = 0.005 + 0.025*np.random.rand()
                txs[idx] += np.random.randn(k)*s2
                tys[idx] += np.random.randn(k)*s2
            elif st == 1:
                # Swap two circles
                i, j = np.random.choice(n, 2, replace=False)
                txs[i], txs[j] = txs[j], txs[i]
                tys[i], tys[j] = tys[j], tys[i]
            elif st == 2:
                # Global small shift
                s2 = 0.003 + 0.012*np.random.rand()
                txs += np.random.randn(n)*s2
                tys += np.random.randn(n)*s2
            elif st == 3:
                # Move smallest circle
                w = np.argmin(crs)
                txs[w] = np.random.uniform(0.05, 0.95)
                tys[w] = np.random.uniform(0.05, 0.95)
            elif st == 4:
                # Move 2-4 smallest circles
                k = min(4, n)
                w2 = np.argsort(crs)[:k]
                for w in w2:
                    txs[w] = np.random.uniform(0.05, 0.95)
                    tys[w] = np.random.uniform(0.05, 0.95)
            elif st == 5:
                # Rotate a cluster
                a = np.random.uniform(-0.4, 0.4)
                k = np.random.randint(2, max(3, n//3))
                idx = np.random.choice(n, k, replace=False)
                cx2, cy2 = np.mean(txs[idx]), np.mean(tys[idx])
                ca, sa = np.cos(a), np.sin(a)
                for ii in idx:
                    ddx, ddy = txs[ii]-cx2, tys[ii]-cy2
                    txs[ii] = cx2+ca*ddx-sa*ddy
                    tys[ii] = cy2+sa*ddx+ca*ddy
            elif st == 6:
                # Mirror subset
                k = np.random.randint(1, max(2, n//4))
                idx = np.random.choice(n, k, replace=False)
                if np.random.rand() < 0.5:
                    txs[idx] = 1.0 - txs[idx]
                else:
                    tys[idx] = 1.0 - tys[idx]
            elif st == 7:
                # Move circle toward centroid of neighbors' gaps
                i = np.random.randint(n)
                gaps = []
                for j in range(n):
                    if j == i:
                        continue
                    d = np.sqrt((cxs[i]-cxs[j])**2 + (cys[i]-cys[j])**2)
                    gaps.append((d - crs[j], j))
                gaps.sort(reverse=True)
                if len(gaps) >= 2:
                    j1, j2 = gaps[0][1], gaps[1][1]
                    # Move toward midpoint of two farthest neighbors
                    mx = (cxs[j1] + cxs[j2]) / 2
                    my = (cys[j1] + cys[j2]) / 2
                    txs[i] = 0.7*txs[i] + 0.3*mx
                    tys[i] = 0.7*tys[i] + 0.3*my
            elif st == 8:
                # Push two closest circles apart
                min_gap = float('inf')
                mi, mj = 0, 1
                for i in range(n):
                    for j in range(i+1, n):
                        d = np.sqrt((cxs[i]-cxs[j])**2 + (cys[i]-cys[j])**2)
                        gap = d - crs[i] - crs[j]
                        if gap < min_gap:
                            min_gap = gap
                            mi, mj = i, j
                d = np.sqrt((txs[mi]-txs[mj])**2 + (tys[mi]-tys[mj])**2)
                if d > 1e-10:
                    push = 0.01 + 0.02*np.random.rand()
                    dx = (txs[mi]-txs[mj])/d
                    dy = (tys[mi]-tys[mj])/d
                    txs[mi] += push*dx; tys[mi] += push*dy
                    txs[mj] -= push*dx; tys[mj] -= push*dy
            else:
                # Scale from center
                sc = 0.98 + 0.04*np.random.rand()
                txs = 0.5 + sc*(txs - 0.5)
                tys = 0.5 + sc*(tys - 0.5)

        txs = np.clip(txs, 0.02, 0.98)
        tys = np.clip(tys, 0.02, 0.98)

        # Light optimization
        trs = np.full(n, 0.35/np.sqrt(n))
        trs = repair_and_grow(txs, tys, trs, n)
        txs, tys, trs = local_search(txs, tys, trs, n, step=0.02, gp=7, iters=3)
        trs = repair_and_grow(txs, tys, trs, n)

        # Occasional heavier optimization
        if hop % 50 == 0 and hop > 0:
            txs, tys, trs = progressive_lbfgsb(txs, tys, trs, n, max_iter=200,
                                                 mus=[100, 1000, 10000, 100000])
            trs = repair_and_grow(txs, tys, trs, n)
            txs, tys, trs = local_search(txs, tys, trs, n, step=0.01, gp=9, iters=3)
            trs = repair_and_grow(txs, tys, trs, n)

        v, met, _ = validate(txs, tys, trs, n)
        if v:
            if met > bm:
                bxs, bys, brs = txs.copy(), tys.copy(), trs.copy()
                bm = met
                cxs, cys, crs = txs.copy(), tys.copy(), trs.copy()
                cm = met
                stale = 0
                if verbose:
                    print(f"    BH {hop:5d}: {met:.10f} NEW BEST ({time.time()-t0:.0f}s)")
            else:
                d = met - cm
                if d > 0 or np.random.rand() < np.exp(d / temp):
                    cxs, cys, crs = txs.copy(), tys.copy(), trs.copy()
                    cm = met
                stale += 1
        else:
            stale += 1

        hop += 1
        # Cool temperature slowly
        if hop % 100 == 0:
            temp *= 0.95

    if verbose:
        print(f"  Basin hop: {hop} hops in {time.time()-t0:.0f}s, best={bm:.10f}")
    return bxs, bys, brs, bm


# ── Main solver ────────────────────────────────────────────────────────

def load_solution(path):
    with open(path) as f:
        data = json.load(f)
    circles = data['circles']
    n = len(circles)
    xs = np.array([c[0] for c in circles])
    ys = np.array([c[1] for c in circles])
    rs = np.array([c[2] for c in circles])
    return xs, ys, rs, n


def save_solution(xs, ys, rs, n, filepath):
    circles = [[float(xs[i]), float(ys[i]), float(rs[i])] for i in range(n)]
    with open(filepath, 'w') as f:
        json.dump({"circles": circles, "n": n, "metric": float(np.sum(rs))}, f, indent=2)
    print(f"Saved {filepath} (metric={np.sum(rs):.10f})")


def deep_refine(n_val, time_budget=300, verbose=True):
    """Deep refinement for a single n value."""
    out_dir = os.path.dirname(os.path.abspath(__file__))
    sol_path = os.path.join(out_dir, f"solution_n{n_val}.json")

    if verbose:
        print(f"\n{'='*70}")
        print(f"  DEEP REFINE n={n_val} (budget={time_budget}s)")
        print(f"{'='*70}")

    t0 = time.time()

    # Load warm start
    xs0, ys0, rs0, n = load_solution(sol_path)
    v0, m0, _ = validate(xs0, ys0, rs0, n)
    if verbose:
        print(f"  Warm start: metric={m0:.10f} valid={v0}")

    bxs, bys, brs = xs0.copy(), ys0.copy(), rs0.copy()
    bm = m0 if v0 else 0

    # Phase 1: Polish warm start with SLSQP (10% budget)
    if verbose:
        print(f"\nPhase 1: SLSQP polish of warm start")
    txs, tys, trs = slsqp_polish(xs0.copy(), ys0.copy(), rs0.copy(), n, max_iter=3000)
    trs = repair_and_grow(txs, tys, trs, n)
    v, m, _ = validate(txs, tys, trs, n)
    if v and m > bm:
        bm = m; bxs, bys, brs = txs.copy(), tys.copy(), trs.copy()
        if verbose: print(f"  SLSQP improved: {m:.10f}")

    # Also try L-BFGS-B on warm start with heavy penalty schedule
    txs, tys, trs = progressive_lbfgsb(xs0.copy(), ys0.copy(), rs0.copy(), n,
                                         max_iter=800,
                                         mus=[10, 50, 200, 1000, 5000, 20000, 100000, 500000])
    trs = repair_and_grow(txs, tys, trs, n)
    txs, tys, trs = local_search(txs, tys, trs, n, step=0.01, gp=11, iters=5)
    trs = repair_and_grow(txs, tys, trs, n)
    txs, tys, trs = gradient_polish(txs, tys, trs, n, steps=300, lr=0.001)
    trs = repair_and_grow(txs, tys, trs, n)
    v, m, _ = validate(txs, tys, trs, n)
    if v and m > bm:
        bm = m; bxs, bys, brs = txs.copy(), tys.copy(), trs.copy()
        if verbose: print(f"  L-BFGS-B re-polish: {m:.10f}")

    if verbose:
        print(f"  After phase 1: best={bm:.10f} ({time.time()-t0:.0f}s)")

    # Phase 2: Multi-start (30% budget)
    if verbose:
        print(f"\nPhase 2: Multi-start diverse initializations")

    ms_budget = time_budget * 0.30
    ms_t0 = time.time()
    s = 0
    inits = ['hex', 'grid', 'ring', 'random', 'sunflower', 'greedy']

    while time.time() - ms_t0 < ms_budget and time.time() - t0 < time_budget * 0.40:
        try:
            init_type = inits[s % len(inits)]
            noise = 0.002 * (s // len(inits))

            if init_type == 'hex':
                xs, ys = hex_init(n, noise)
            elif init_type == 'grid':
                xs, ys = grid_init(n, noise)
            elif init_type == 'ring':
                xs, ys = ring_init(n)
                if noise > 0:
                    xs += np.random.randn(n)*noise
                    ys += np.random.randn(n)*noise
                    xs = np.clip(xs, 0.02, 0.98)
                    ys = np.clip(ys, 0.02, 0.98)
            elif init_type == 'sunflower':
                xs, ys = sunflower_init(n)
                if noise > 0:
                    xs += np.random.randn(n)*noise
                    ys += np.random.randn(n)*noise
                    xs = np.clip(xs, 0.02, 0.98)
                    ys = np.clip(ys, 0.02, 0.98)
            elif init_type == 'greedy':
                xs, ys, _ = greedy_init(n)
            else:
                xs, ys = random_init(n)

            result = optimize_from_init(xs, ys, n, do_slsqp=(s % 3 == 0))
            if result is not None:
                txs, tys, trs, m = result
                if m > bm:
                    bm = m; bxs, bys, brs = txs.copy(), tys.copy(), trs.copy()
                    if verbose:
                        print(f"  Start {s:4d} ({init_type:10s}): {m:.10f} NEW BEST")
        except Exception as e:
            pass
        s += 1

    if verbose:
        print(f"  Phase 2: {s} starts, best={bm:.10f} ({time.time()-t0:.0f}s)")

    # Phase 3: Basin hopping from best (55% budget)
    remaining = max(30, time_budget - (time.time() - t0) - 20)
    if verbose:
        print(f"\nPhase 3: Basin hopping ({remaining:.0f}s budget)")

    bxs, bys, brs, bm = basin_hop(bxs, bys, brs, n, time_budget=remaining, verbose=verbose)

    # Phase 4: Final fine polish
    if verbose:
        print(f"\nPhase 4: Final polish")

    # Fine local search
    txs, tys, trs = bxs.copy(), bys.copy(), brs.copy()
    for step in [0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]:
        txs, tys, trs = local_search(txs, tys, trs, n, step=step, gp=11, iters=3)
        trs = repair_and_grow(txs, tys, trs, n)

    # Gradient polish
    txs, tys, trs = gradient_polish(txs, tys, trs, n, steps=500, lr=0.0005)
    trs = repair_and_grow(txs, tys, trs, n)

    # Final SLSQP
    txs, tys, trs = slsqp_polish(txs, tys, trs, n, max_iter=5000)
    trs = repair_and_grow(txs, tys, trs, n)

    v, m, _ = validate(txs, tys, trs, n)
    if v and m > bm:
        bm = m; bxs, bys, brs = txs.copy(), tys.copy(), trs.copy()
        if verbose: print(f"  Final polish: {m:.10f}")

    elapsed = time.time() - t0
    if verbose:
        print(f"\n  FINAL n={n_val}: {bm:.10f} ({elapsed:.1f}s)")

    # Save if improved
    v_orig, m_orig, _ = validate(xs0, ys0, rs0, n)
    if bm > m_orig:
        save_solution(bxs, bys, brs, n, sol_path)
        if verbose:
            print(f"  IMPROVED: {m_orig:.10f} -> {bm:.10f} (+{bm-m_orig:.10f})")
    else:
        if verbose:
            print(f"  No improvement: {m_orig:.10f} (best attempt: {bm:.10f})")

    return bm


if __name__ == "__main__":
    targets = {24: 2.530, 25: 2.587, 27: 2.685, 29: 2.790, 31: 2.889}

    if len(sys.argv) > 1:
        n_values = [int(sys.argv[1])]
        budget = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    else:
        # Priority order: largest gaps first
        n_values = [31, 29, 27, 25, 24]
        budget = 300

    results = {}
    for nv in n_values:
        np.random.seed(12345 + nv)
        m = deep_refine(nv, time_budget=budget, verbose=True)
        results[nv] = m

    print("\n" + "="*70)
    print("DEEP REFINE SUMMARY")
    print("="*70)
    for nv in n_values:
        sota = targets.get(nv, 0)
        m = results.get(nv, 0)
        gap = (m/sota - 1)*100 if sota else 0
        print(f"  n={nv}: {m:.10f}  SOTA={sota:.3f}  gap={gap:+.1f}%")
