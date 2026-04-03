"""
V6: Adapted from OpenEvolve's best_program.py that achieved 2.63598.
Key: 7 specialized initialization patterns + 3-stage optimization
(L-BFGS-B positions -> SLSQP radii -> SLSQP joint).
"""

import numpy as np
from scipy.optimize import minimize
import json
import sys
import time
import math
from pathlib import Path


# ============================================================
# OpenEvolve-style initialization patterns
# ============================================================

def pattern_specialized_26():
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    corner_radius = 0.118
    centers[0] = [corner_radius, corner_radius]
    centers[1] = [1 - corner_radius, corner_radius]
    centers[2] = [corner_radius, 1 - corner_radius]
    centers[3] = [1 - corner_radius, 1 - corner_radius]
    radii[0:4] = corner_radius
    edge_radius = 0.103
    centers[4] = [0.5, edge_radius]
    centers[5] = [0.5, 1 - edge_radius]
    centers[6] = [edge_radius, 0.5]
    centers[7] = [1 - edge_radius, 0.5]
    radii[4:8] = edge_radius
    centers[8] = [0.5, 0.5]
    radii[8] = 0.124
    inner_radius = 0.098
    for i in range(8):
        angle = 2 * np.pi * i / 8
        dist = radii[8] + inner_radius
        centers[9 + i] = [0.5 + dist * np.cos(angle), 0.5 + dist * np.sin(angle)]
        radii[9 + i] = inner_radius
    outer_radius = 0.083
    for i in range(2):
        angle = 2 * np.pi * i / 2 + np.pi/2
        dist = radii[8] + 2 * inner_radius + outer_radius + 0.01
        centers[17 + i] = [0.5 + dist * np.cos(angle), 0.5 + dist * np.sin(angle)]
        radii[17 + i] = outer_radius
    outer_radius2 = 0.071
    for i in range(6):
        angle = 2 * np.pi * i / 6 + np.pi/6
        dist = 0.25
        centers[19 + i] = [0.15 + dist * np.cos(angle), 0.15 + dist * np.sin(angle)]
        radii[19 + i] = outer_radius2
    outer_radius3 = 0.076
    for i in range(min(3, n - 25)):
        angle = 2 * np.pi * i / 3 + np.pi/3
        dist = 0.25
        centers[25 - 2 + i] = [0.85 + dist * np.cos(angle), 0.85 + dist * np.sin(angle)]
        radii[25 - 2 + i] = outer_radius3
    return centers, radii


def pattern_hybrid_26():
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    centers[0] = [0.5, 0.5]; radii[0] = 0.128
    inner_radius = 0.103
    for i in range(6):
        angle = 2 * np.pi * i / 6
        dist = radii[0] + inner_radius
        centers[i+1] = [0.5 + dist*np.cos(angle), 0.5 + dist*np.sin(angle)]
        radii[i+1] = inner_radius
    middle_radius = 0.093
    for i in range(8):
        angle = 2*np.pi*i/8 + np.pi/8
        dist = radii[0] + 2*inner_radius + middle_radius
        centers[i+7] = [0.5 + dist*np.cos(angle), 0.5 + dist*np.sin(angle)]
        radii[i+7] = middle_radius
    corner_radius = 0.113
    centers[15] = [corner_radius, corner_radius]
    centers[16] = [1-corner_radius, corner_radius]
    centers[17] = [corner_radius, 1-corner_radius]
    centers[18] = [1-corner_radius, 1-corner_radius]
    radii[15:19] = corner_radius
    edge_radius = 0.088
    centers[19] = [0.5, edge_radius]
    centers[20] = [0.5, 1-edge_radius]
    centers[21] = [edge_radius, 0.5]
    centers[22] = [1-edge_radius, 0.5]
    radii[19:23] = edge_radius
    small_radius = 0.073
    centers[23] = [0.25, 0.25]
    centers[24] = [0.75, 0.25]
    centers[25] = [0.25, 0.75]
    radii[23:26] = small_radius
    return centers, radii


def pattern_ring_26():
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    centers[0] = [0.5, 0.5]; radii[0] = 0.133
    ring1_radius = 0.098
    for i in range(8):
        angle = 2*np.pi*i/8
        dist = radii[0] + ring1_radius
        centers[i+1] = [0.5+dist*np.cos(angle), 0.5+dist*np.sin(angle)]
        radii[i+1] = ring1_radius
    ring2_radius = 0.088
    for i in range(12):
        angle = 2*np.pi*i/12 + np.pi/12
        dist = radii[0] + 2*ring1_radius + ring2_radius
        centers[i+9] = [0.5+dist*np.cos(angle), 0.5+dist*np.sin(angle)]
        radii[i+9] = ring2_radius
    corner_radius = 0.093
    centers[21] = [corner_radius, corner_radius]
    centers[22] = [1-corner_radius, corner_radius]
    centers[23] = [corner_radius, 1-corner_radius]
    centers[24] = [1-corner_radius, 1-corner_radius]
    radii[21:25] = corner_radius
    centers[25] = [0.5, 0.15]; radii[25] = 0.083
    return centers, radii


def pattern_greedy_26():
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    init_pos = [(0.123, 0.123), (0.877, 0.123), (0.123, 0.877), (0.877, 0.877), (0.5, 0.5)]
    init_r = [0.123, 0.123, 0.123, 0.123, 0.143]
    for i in range(5):
        centers[i] = init_pos[i]; radii[i] = init_r[i]
    edge_pos = [(0.5,0.1),(0.5,0.9),(0.1,0.5),(0.9,0.5),
                (0.3,0.1),(0.7,0.1),(0.3,0.9),(0.7,0.9),
                (0.1,0.3),(0.1,0.7),(0.9,0.3),(0.9,0.7)]
    edge_r = [0.103]*4 + [0.093]*8
    for i in range(12):
        centers[i+5] = edge_pos[i]; radii[i+5] = edge_r[i]
    inner_pos = [(0.3,0.3),(0.5,0.3),(0.7,0.3),(0.3,0.5),(0.7,0.5),
                 (0.3,0.7),(0.5,0.7),(0.7,0.7),(0.4,0.4)]
    inner_r = [0.093]*8 + [0.083]
    for i in range(min(9, n-17)):
        centers[i+17] = inner_pos[i]; radii[i+17] = inner_r[i]
    return centers, radii


def pattern_optimized_grid_26():
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    x_coords = np.linspace(0.1, 0.9, 6)
    y_coords = np.linspace(0.1, 0.9, 5)
    count = 0
    for i in range(5):
        for j in range(6):
            if count < n:
                centers[count] = [x_coords[j], y_coords[i]]
                radii[count] = 0.083
                count += 1
    for i in range(n):
        if (centers[i,0] < 0.2 and centers[i,1] < 0.2) or \
           (centers[i,0] > 0.8 and centers[i,1] < 0.2) or \
           (centers[i,0] < 0.2 and centers[i,1] > 0.8) or \
           (centers[i,0] > 0.8 and centers[i,1] > 0.8):
            radii[i] = 0.088
        elif abs(centers[i,0]-0.5) < 0.01 and abs(centers[i,1]-0.5) < 0.01:
            radii[i] = 0.103
        else:
            radii[i] = 0.078
    return centers, radii


def pattern_corner_optimized_26():
    """Corner-optimized: 4 big corners + structured interior."""
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    # 4 large corner circles
    cr = 0.135
    centers[0] = [cr, cr]; centers[1] = [1-cr, cr]
    centers[2] = [cr, 1-cr]; centers[3] = [1-cr, 1-cr]
    radii[:4] = cr
    # 4 edge midpoints
    er = 0.105
    centers[4] = [0.5, er]; centers[5] = [0.5, 1-er]
    centers[6] = [er, 0.5]; centers[7] = [1-er, 0.5]
    radii[4:8] = er
    # 4 between corner and edge
    br = 0.090
    centers[8] = [0.30, er]; centers[9] = [0.70, er]
    centers[10] = [0.30, 1-er]; centers[11] = [0.70, 1-er]
    radii[8:12] = br
    # Side pairs
    sr = 0.085
    centers[12] = [er, 0.30]; centers[13] = [er, 0.70]
    centers[14] = [1-er, 0.30]; centers[15] = [1-er, 0.70]
    radii[12:16] = sr
    # Inner ring
    ir = 0.080
    inner_pos = [(0.30, 0.30), (0.50, 0.30), (0.70, 0.30),
                 (0.30, 0.50), (0.70, 0.50),
                 (0.30, 0.70), (0.50, 0.70), (0.70, 0.70),
                 (0.50, 0.50), (0.40, 0.40)]
    for i in range(min(10, n-16)):
        centers[16+i] = inner_pos[i]
        radii[16+i] = ir
    return centers, radii


def pattern_billiard_26():
    """Billiard-table inspired."""
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    # Corner circles
    cr = 0.113
    corners = [(cr,cr), (1-cr,cr), (cr,1-cr), (1-cr,1-cr)]
    for i, (x, y) in enumerate(corners):
        centers[i] = [x, y]; radii[i] = cr
    # Edge circles (2 per edge)
    er = 0.098
    count = 4
    for x in [0.35, 0.65]:
        centers[count] = [x, er]; radii[count] = er; count += 1
        centers[count] = [x, 1-er]; radii[count] = er; count += 1
    for y in [0.35, 0.65]:
        centers[count] = [er, y]; radii[count] = er; count += 1
        centers[count] = [1-er, y]; radii[count] = er; count += 1
    # Center + inner
    centers[count] = [0.5, 0.5]; radii[count] = 0.118; count += 1
    # Fill remaining with triangular pattern
    ir = 0.078
    inner = [(0.5, 0.28), (0.5, 0.72), (0.28, 0.5), (0.72, 0.5),
             (0.35, 0.35), (0.65, 0.35), (0.35, 0.65), (0.65, 0.65),
             (0.5, 0.5)]
    for i in range(min(len(inner), n - count)):
        centers[count] = inner[i]; radii[count] = ir; count += 1
    return centers, radii


# ============================================================
# Additional diverse initializations (my own)
# ============================================================

def poisson_disk_init(n, seed=42, min_dist_factor=0.8):
    rng = np.random.RandomState(seed)
    min_dist = min_dist_factor / math.sqrt(n)
    r_est = min_dist / 2.5
    margin = max(r_est, 0.02)
    positions = []
    for _ in range(80000):
        if len(positions) >= n: break
        x = rng.uniform(margin, 1-margin)
        y = rng.uniform(margin, 1-margin)
        if all((x-px)**2+(y-py)**2 >= min_dist**2 for px, py in positions):
            positions.append((x, y))
    while len(positions) < n:
        positions.append((rng.uniform(margin, 1-margin), rng.uniform(margin, 1-margin)))
    centers = np.array(positions[:n])
    radii = np.full(n, r_est)
    return centers, radii


def random_varied_init(n, seed=0):
    """Random positions with varied radii."""
    rng = np.random.RandomState(seed)
    centers = np.column_stack([rng.uniform(0.05, 0.95, n), rng.uniform(0.05, 0.95, n)])
    radii = rng.uniform(0.04, 0.12, n)
    return centers, radii


# ============================================================
# 3-stage OpenEvolve-style optimization
# ============================================================

def calculate_penalty(centers, radii):
    n = len(centers)
    penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            overlap = radii[i] + radii[j] - dist
            if overlap > 0:
                penalty += 300 * overlap**2 * (radii[i] + radii[j])
    for i in range(n):
        if centers[i,0] - radii[i] < 0:
            penalty += 300 * (radii[i] - centers[i,0])**2
        if centers[i,0] + radii[i] > 1:
            penalty += 300 * (centers[i,0] + radii[i] - 1)**2
        if centers[i,1] - radii[i] < 0:
            penalty += 300 * (radii[i] - centers[i,1])**2
        if centers[i,1] + radii[i] > 1:
            penalty += 300 * (centers[i,1] + radii[i] - 1)**2
    return penalty


def optimize_3stage(centers, radii, pos_iter=150, rad_iter=500, joint_iter=800):
    """3-stage optimization matching OpenEvolve's approach."""
    n = len(centers)

    # Stage 1: Position optimization with L-BFGS-B
    def obj_pos(x):
        c = x.reshape((n, 2))
        return calculate_penalty(c, radii)

    res = minimize(obj_pos, centers.flatten(), method='L-BFGS-B',
                   bounds=[(0, 1)] * (2*n),
                   options={'maxiter': pos_iter, 'ftol': 1e-6})
    centers = res.x.reshape((n, 2))

    # Stage 2: Radius optimization with SLSQP
    def obj_rad(r):
        return -np.sum(r)

    def cons_rad(r):
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                constraints.append(dist - r[i] - r[j])
        for i in range(n):
            constraints.append(centers[i, 0] - r[i])
            constraints.append(1 - centers[i, 0] - r[i])
            constraints.append(centers[i, 1] - r[i])
            constraints.append(1 - centers[i, 1] - r[i])
        return np.array(constraints)

    res = minimize(obj_rad, radii, method='SLSQP',
                   constraints={'type': 'ineq', 'fun': cons_rad},
                   bounds=[(0.01, 0.25)] * n,
                   options={'maxiter': rad_iter, 'ftol': 1e-8})
    radii = res.x

    # Stage 3: Joint optimization with SLSQP
    def obj_joint(x):
        return -np.sum(x[2*n:])

    def cons_joint(x):
        c = x[:2*n].reshape((n, 2))
        r = x[2*n:]
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(c[i] - c[j])
                constraints.append(dist - r[i] - r[j])
        for i in range(n):
            constraints.append(c[i, 0] - r[i])
            constraints.append(1 - c[i, 0] - r[i])
            constraints.append(c[i, 1] - r[i])
            constraints.append(1 - c[i, 1] - r[i])
        return np.array(constraints)

    x0 = np.concatenate([centers.flatten(), radii])
    bounds = [(0.01, 0.99)] * (2*n) + [(0.01, 0.25)] * n

    res = minimize(obj_joint, x0, method='SLSQP',
                   constraints={'type': 'ineq', 'fun': cons_joint},
                   bounds=bounds,
                   options={'maxiter': joint_iter, 'ftol': 1e-10})

    final_centers = res.x[:2*n].reshape((n, 2))
    final_radii = res.x[2*n:]
    return final_centers, final_radii


def optimize_3stage_aggressive(centers, radii):
    """More aggressive version with more iterations and tighter tolerance."""
    n = len(centers)

    # Stage 1: Position with progressive penalty
    for penalty_mult in [100, 500, 2000]:
        def obj_pos(x, pm=penalty_mult):
            c = x.reshape((n, 2))
            pen = 0.0
            for i in range(n):
                for j in range(i+1, n):
                    dist = np.linalg.norm(c[i] - c[j])
                    ol = radii[i] + radii[j] - dist
                    if ol > 0: pen += pm * ol**2
                if c[i,0]-radii[i] < 0: pen += pm * (radii[i]-c[i,0])**2
                if c[i,0]+radii[i] > 1: pen += pm * (c[i,0]+radii[i]-1)**2
                if c[i,1]-radii[i] < 0: pen += pm * (radii[i]-c[i,1])**2
                if c[i,1]+radii[i] > 1: pen += pm * (c[i,1]+radii[i]-1)**2
            return pen

        res = minimize(obj_pos, centers.flatten(), method='L-BFGS-B',
                       bounds=[(0.01, 0.99)] * (2*n),
                       options={'maxiter': 300, 'ftol': 1e-10})
        centers = res.x.reshape((n, 2))

    # Stage 2: Radius with SLSQP
    def cons_rad(r):
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                constraints.append(np.linalg.norm(centers[i]-centers[j]) - r[i] - r[j])
            constraints.extend([centers[i,0]-r[i], 1-centers[i,0]-r[i],
                              centers[i,1]-r[i], 1-centers[i,1]-r[i]])
        return np.array(constraints)

    res = minimize(lambda r: -np.sum(r), radii, method='SLSQP',
                   constraints={'type': 'ineq', 'fun': cons_rad},
                   bounds=[(0.005, 0.3)] * n,
                   options={'maxiter': 1000, 'ftol': 1e-12})
    radii = res.x

    # Stage 3: Joint with SLSQP (multiple rounds)
    def cons_joint(x):
        c = x[:2*n].reshape((n, 2)); r = x[2*n:]
        constraints = []
        for i in range(n):
            for j in range(i+1, n):
                constraints.append(np.linalg.norm(c[i]-c[j]) - r[i] - r[j])
            constraints.extend([c[i,0]-r[i], 1-c[i,0]-r[i], c[i,1]-r[i], 1-c[i,1]-r[i]])
        return np.array(constraints)

    x0 = np.concatenate([centers.flatten(), radii])
    bounds = [(0.005, 0.995)] * (2*n) + [(0.005, 0.3)] * n

    for max_it in [2000, 5000, 10000]:
        res = minimize(lambda x: -np.sum(x[2*n:]), x0, method='SLSQP',
                       constraints={'type': 'ineq', 'fun': cons_joint},
                       bounds=bounds,
                       options={'maxiter': max_it, 'ftol': 1e-15})
        x0 = res.x

    final_centers = res.x[:2*n].reshape((n, 2))
    final_radii = res.x[2*n:]
    return final_centers, final_radii


def check_valid(centers, radii, tol=1e-10):
    n = len(centers)
    for i in range(n):
        r = radii[i]
        if r <= 0 or r-centers[i,0] > tol or centers[i,0]+r-1 > tol or \
           r-centers[i,1] > tol or centers[i,1]+r-1 > tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(centers[i]-centers[j])
            if radii[i]+radii[j]-dist > tol:
                return False
    return True


def repair(centers, radii, tol=1e-10):
    """Shrink radii to satisfy constraints."""
    n = len(radii)
    for _ in range(200):
        changed = False
        for i in range(n):
            r = radii[i]
            r_new = min(r, centers[i,0]-tol, 1-centers[i,0]-tol,
                       centers[i,1]-tol, 1-centers[i,1]-tol)
            if r_new < r:
                radii[i] = max(r_new, 1e-8); changed = True
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(centers[i]-centers[j])
                if radii[i]+radii[j] > dist-tol and radii[i]+radii[j] > 0:
                    scale = max((dist-2*tol)/(radii[i]+radii[j]), 0.01)
                    if scale < 1:
                        radii[i] *= scale; radii[j] *= scale; changed = True
        if not changed:
            break
    return radii


def main():
    n = 26
    output = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / "solution_n26_v6.json")

    # All OpenEvolve patterns
    patterns = [
        ("specialized", pattern_specialized_26),
        ("hybrid", pattern_hybrid_26),
        ("ring", pattern_ring_26),
        ("greedy", pattern_greedy_26),
        ("grid", pattern_optimized_grid_26),
        ("corner_opt", pattern_corner_optimized_26),
        ("billiard", pattern_billiard_26),
    ]

    # Add Poisson disk and random variants
    for s in range(30):
        patterns.append((f"poisson_s{s}", lambda s=s: poisson_disk_init(n, seed=s)))
    for mdf in [0.6, 0.7, 0.8, 0.9, 1.0]:
        for s in range(3):
            patterns.append((f"poisson_md{mdf}_s{s}", lambda s=s, mdf=mdf: poisson_disk_init(n, seed=100+s, min_dist_factor=mdf)))
    for s in range(20):
        patterns.append((f"random_s{s}", lambda s=s: random_varied_init(n, seed=s)))

    # Also create perturbed versions of the OpenEvolve patterns
    base_patterns = [pattern_specialized_26, pattern_hybrid_26, pattern_ring_26,
                     pattern_greedy_26, pattern_corner_optimized_26, pattern_billiard_26]
    for bp_idx, bp in enumerate(base_patterns):
        for s in range(10):
            def make_perturbed(bp=bp, s=s):
                rng = np.random.RandomState(s + bp_idx*100)
                c, r = bp()
                c += rng.normal(0, 0.02, c.shape)
                c = np.clip(c, 0.02, 0.98)
                return c, r
            patterns.append((f"pert_{bp_idx}_s{s}", make_perturbed))

    print(f"Total patterns: {len(patterns)}", flush=True)

    best_metric = 0.0
    best_centers = None
    best_radii = None
    candidates = []  # (metric, centers, radii) for aggressive polish

    t_start = time.time()

    # Phase 1: Standard 3-stage optimization
    print("Phase 1: Standard 3-stage optimization", flush=True)
    for idx, (name, pattern_fn) in enumerate(patterns):
        t0 = time.time()
        try:
            centers, radii = pattern_fn()
            centers, radii = optimize_3stage(centers, radii)
            radii = repair(centers, radii)
            metric = np.sum(radii)
            valid = check_valid(centers, radii)
            dt = time.time() - t0

            if valid and metric > 0.5:
                candidates.append((metric, centers.copy(), radii.copy()))

            if valid and metric > best_metric:
                best_metric = metric
                best_centers = centers.copy()
                best_radii = radii.copy()
                print(f"  [{idx+1}/{len(patterns)}] {name}: {metric:.6f} ** BEST ** [{dt:.1f}s]", flush=True)
            elif idx % 20 == 0:
                print(f"  [{idx+1}/{len(patterns)}] {name}: {metric:.6f} {'ok' if valid else 'INV'} [{dt:.1f}s]", flush=True)
        except Exception as e:
            if idx % 20 == 0:
                print(f"  [{idx+1}/{len(patterns)}] {name}: ERR {e}", flush=True)

    # Phase 2: Aggressive optimization on top candidates
    candidates.sort(key=lambda t: -t[0])
    num_agg = min(15, len(candidates))
    print(f"\nPhase 2: Aggressive polish on top {num_agg}", flush=True)

    for rank in range(num_agg):
        mb, c, r = candidates[rank]
        t0 = time.time()
        try:
            c2, r2 = optimize_3stage_aggressive(c.copy(), r.copy())
            r2 = repair(c2, r2)
            metric = np.sum(r2)
            valid = check_valid(c2, r2)
            dt = time.time() - t0
            if valid and metric > best_metric:
                best_metric = metric
                best_centers = c2.copy()
                best_radii = r2.copy()
                print(f"  #{rank+1}: {mb:.6f} -> {metric:.6f} ** BEST ** [{dt:.1f}s]", flush=True)
            elif rank < 5:
                print(f"  #{rank+1}: {mb:.6f} -> {metric:.6f} [{dt:.1f}s]", flush=True)
        except Exception as e:
            if rank < 5:
                print(f"  #{rank+1}: ERR {e}", flush=True)

    # Load previous best and compare
    prev_path = Path(__file__).parent / "solution_n26.json"
    if prev_path.exists():
        with open(prev_path) as f:
            data = json.load(f)
        prev_circles = data["circles"]
        prev_metric = sum(c[2] for c in prev_circles)
        if prev_metric > best_metric:
            print(f"\nPrevious best {prev_metric:.10f} still better than {best_metric:.10f}", flush=True)
            # Try aggressive polish on previous best too
            prev_c = np.array([[c[0], c[1]] for c in prev_circles])
            prev_r = np.array([c[2] for c in prev_circles])
            try:
                c2, r2 = optimize_3stage_aggressive(prev_c, prev_r)
                r2 = repair(c2, r2)
                m2 = np.sum(r2)
                if check_valid(c2, r2) and m2 > best_metric:
                    best_metric = m2
                    best_centers = c2
                    best_radii = r2
                    print(f"  Previous best polished: {m2:.10f}", flush=True)
            except:
                pass
            if prev_metric > best_metric:
                best_metric = prev_metric
                best_centers = prev_c
                best_radii = prev_r

    # Phase 3: Basin-hopping
    if best_centers is not None:
        print(f"\nPhase 3: Basin-hopping from {best_metric:.10f}", flush=True)
        rng = np.random.RandomState(42)
        no_imp = 0
        for att in range(60):
            if no_imp >= 20: break
            scale = rng.choice([0.005, 0.01, 0.02, 0.03, 0.05])
            c_pert = best_centers.copy() + rng.normal(0, scale, best_centers.shape)
            c_pert = np.clip(c_pert, 0.01, 0.99)
            r_pert = best_radii.copy()
            if rng.random() < 0.3:
                r_pert *= rng.uniform(0.8, 1.2, len(r_pert))
                r_pert = np.clip(r_pert, 0.005, 0.25)
            try:
                c2, r2 = optimize_3stage_aggressive(c_pert, r_pert)
                r2 = repair(c2, r2)
                metric = np.sum(r2)
                if check_valid(c2, r2) and metric > best_metric + 1e-10:
                    best_metric = metric
                    best_centers = c2
                    best_radii = r2
                    no_imp = 0
                    print(f"  #{att+1} sc={scale:.3f}: {metric:.10f} ** IMPROVED **", flush=True)
                else:
                    no_imp += 1
            except:
                no_imp += 1

    total = time.time() - t_start
    print(f"\nFinal: {best_metric:.10f} in {total:.1f}s", flush=True)

    if best_centers is not None:
        circles = [[best_centers[i,0], best_centers[i,1], best_radii[i]] for i in range(n)]
        with open(output, 'w') as f:
            json.dump({"circles": circles}, f, indent=2)
        print(f"Saved to {output}", flush=True)


if __name__ == "__main__":
    main()
