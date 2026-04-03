"""
Mobius-001: Circle packing optimizer using Mobius deformations and inversive geometry.

Strategy:
1. Load best known solution (topo-001)
2. Apply local Mobius transformations to clusters of circles
3. Re-optimize with SLSQP to restore feasibility
4. Also try fresh multi-start with topology-diverse initializations
"""

import json
import math
import numpy as np
from scipy.optimize import minimize
from pathlib import Path
import itertools
import sys
import time

SEED = 42
N = 26
WORKTREE = Path("/Users/wujiewang/code/circle-packing/.worktrees/mobius-001")
SOLUTION_PATH = WORKTREE / "orbits/topo-001/solution_n26.json"
OUTPUT_DIR = WORKTREE / "orbits/mobius-001"

def load_solution(path):
    with open(path) as f:
        data = json.load(f)
    circles = data["circles"]
    return np.array(circles)  # shape (n, 3): x, y, r

def save_solution(circles, path):
    data = {"circles": [[float(c[0]), float(c[1]), float(c[2])] for c in circles]}
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def sum_radii(circles):
    return np.sum(circles[:, 2])

def validate(circles, tol=1e-10):
    """Check all constraints. Returns (valid, max_violation)."""
    n = len(circles)
    max_viol = 0.0
    for i in range(n):
        x, y, r = circles[i]
        if r <= 0:
            return False, abs(r)
        max_viol = max(max_viol, r - x, x + r - 1.0, r - y, y + r - 1.0)
    for i in range(n):
        for j in range(i+1, n):
            dx = circles[i,0] - circles[j,0]
            dy = circles[i,1] - circles[j,1]
            dist = math.sqrt(dx*dx + dy*dy)
            overlap = (circles[i,2] + circles[j,2]) - dist
            max_viol = max(max_viol, overlap)
    return max_viol <= tol, max_viol

def circles_to_vec(circles):
    return circles.flatten()

def vec_to_circles(vec, n=N):
    return vec.reshape(n, 3)

def optimize_slsqp(circles, maxiter=5000):
    """Optimize sum of radii with SLSQP."""
    n = len(circles)
    x0 = circles_to_vec(circles)

    def objective(x):
        return -np.sum(x[2::3])  # minimize negative sum of radii

    def grad_objective(x):
        g = np.zeros_like(x)
        g[2::3] = -1.0
        return g

    constraints = []

    # Containment: x_i - r_i >= 0, 1 - x_i - r_i >= 0, y_i - r_i >= 0, 1 - y_i - r_i >= 0
    for i in range(n):
        ix, iy, ir = 3*i, 3*i+1, 3*i+2
        constraints.append({'type': 'ineq', 'fun': lambda x, ix=ix, ir=ir: x[ix] - x[ir]})
        constraints.append({'type': 'ineq', 'fun': lambda x, ix=ix, ir=ir: 1.0 - x[ix] - x[ir]})
        constraints.append({'type': 'ineq', 'fun': lambda x, iy=iy, ir=ir: x[iy] - x[ir]})
        constraints.append({'type': 'ineq', 'fun': lambda x, iy=iy, ir=ir: 1.0 - x[iy] - x[ir]})
        constraints.append({'type': 'ineq', 'fun': lambda x, ir=ir: x[ir] - 1e-6})  # r > 0

    # Non-overlap: dist(i,j)^2 - (r_i + r_j)^2 >= 0
    for i in range(n):
        for j in range(i+1, n):
            ix, iy, ir = 3*i, 3*i+1, 3*i+2
            jx, jy, jr = 3*j, 3*j+1, 3*j+2
            def sep_con(x, ix=ix, iy=iy, ir=ir, jx=jx, jy=jy, jr=jr):
                dx = x[ix] - x[jx]
                dy = x[iy] - x[jy]
                return dx*dx + dy*dy - (x[ir] + x[jr])**2
            constraints.append({'type': 'ineq', 'fun': sep_con})

    bounds = []
    for i in range(n):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)])

    result = minimize(objective, x0, method='SLSQP', jac=grad_objective,
                     bounds=bounds, constraints=constraints,
                     options={'maxiter': maxiter, 'ftol': 1e-15, 'disp': False})

    return vec_to_circles(result.x, n), -result.fun

def optimize_slsqp_analytical(circles, maxiter=10000):
    """Optimize with analytical Jacobians for constraints."""
    n = len(circles)
    x0 = circles_to_vec(circles)

    def objective(x):
        return -np.sum(x[2::3])

    def grad_objective(x):
        g = np.zeros_like(x)
        g[2::3] = -1.0
        return g

    constraints = []

    # Containment constraints with Jacobians
    for i in range(n):
        ix, iy, ir = 3*i, 3*i+1, 3*i+2

        def make_cont(idx_pos, idx_r, sign_pos, offset):
            def fun(x, ip=idx_pos, ir_=idx_r, sp=sign_pos, off=offset):
                return sp * x[ip] - x[ir_] + off
            def jac(x, ip=idx_pos, ir_=idx_r, sp=sign_pos, nn=3*n):
                g = np.zeros(nn)
                g[ip] = sp
                g[ir_] = -1.0
                return g
            return fun, jac

        f, j = make_cont(ix, ir, 1.0, 0.0)  # x - r >= 0
        constraints.append({'type': 'ineq', 'fun': f, 'jac': j})
        f, j = make_cont(ix, ir, -1.0, 1.0)  # 1 - x - r >= 0
        constraints.append({'type': 'ineq', 'fun': f, 'jac': j})
        f, j = make_cont(iy, ir, 1.0, 0.0)  # y - r >= 0
        constraints.append({'type': 'ineq', 'fun': f, 'jac': j})
        f, j = make_cont(iy, ir, -1.0, 1.0)  # 1 - y - r >= 0
        constraints.append({'type': 'ineq', 'fun': f, 'jac': j})

        # r > 0
        def make_rpos(idx_r):
            def fun(x, ir_=idx_r): return x[ir_] - 1e-6
            def jac(x, ir_=idx_r, nn=3*n):
                g = np.zeros(nn)
                g[ir_] = 1.0
                return g
            return fun, jac
        f, j = make_rpos(ir)
        constraints.append({'type': 'ineq', 'fun': f, 'jac': j})

    # Non-overlap constraints with Jacobians
    for i in range(n):
        for j in range(i+1, n):
            ix, iy, ir = 3*i, 3*i+1, 3*i+2
            jx, jy, jr = 3*j, 3*j+1, 3*j+2

            def make_sep(ix_, iy_, ir_, jx_, jy_, jr_):
                def fun(x, a=ix_, b=iy_, c=ir_, d=jx_, e=jy_, f=jr_):
                    dx = x[a] - x[d]
                    dy = x[b] - x[e]
                    return dx*dx + dy*dy - (x[c] + x[f])**2
                def jac(x, a=ix_, b=iy_, c=ir_, d=jx_, e=jy_, f=jr_, nn=3*n):
                    g = np.zeros(nn)
                    dx = x[a] - x[d]
                    dy = x[b] - x[e]
                    sr = x[c] + x[f]
                    g[a] = 2*dx
                    g[d] = -2*dx
                    g[b] = 2*dy
                    g[e] = -2*dy
                    g[c] = -2*sr
                    g[f] = -2*sr
                    return g
                return fun, jac

            f, j = make_sep(ix, iy, ir, jx, jy, jr)
            constraints.append({'type': 'ineq', 'fun': f, 'jac': j})

    bounds = []
    for i in range(n):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)])

    result = minimize(objective, x0, method='SLSQP', jac=grad_objective,
                     bounds=bounds, constraints=constraints,
                     options={'maxiter': maxiter, 'ftol': 1e-15, 'disp': False})

    return vec_to_circles(result.x, n), -result.fun


# ============ MOBIUS TRANSFORMATIONS ============

def mobius_transform_complex(z, a, b, c, d):
    """Apply Mobius transformation f(z) = (az + b) / (cz + d) in the complex plane."""
    return (a * z + b) / (c * z + d)

def circle_to_complex(x, y, r):
    """Convert circle (x, y, r) to complex center + radius."""
    return complex(x, y), r

def apply_mobius_to_circle(center, radius, a, b, c, d):
    """
    Apply Mobius transformation to a circle.
    A Mobius transformation maps circles to circles.

    For a circle with center z0 and radius rho:
    The image under f(z) = (az+b)/(cz+d) is another circle.

    Using the formula for Mobius action on circles:
    If w0 = f(z0), and the determinant is det = ad - bc, then:
    - new_center = f(z0) - det * conj(c) * rho^2 / (|cz0+d|^2 - |c|^2 * rho^2)  (approx)

    Actually, the clean formula uses the matrix representation.
    For the generalized circle (center z0, radius rho), under Mobius f:

    new_center = (a*z0 + b)*conj(c*z0 + d) - a*conj(c)*rho^2) / (|c*z0+d|^2 - |c|^2*rho^2)
    new_radius = |det| * rho / ||c*z0+d|^2 - |c|^2*rho^2|
    """
    det = a * d - b * c
    cz_d = c * center + d
    cz_d_sq = abs(cz_d)**2
    c_sq = abs(c)**2
    denom = cz_d_sq - c_sq * radius**2

    if abs(denom) < 1e-14:
        return None, None  # Circle maps to a line

    new_center = ((a * center + b) * np.conj(cz_d) - a * np.conj(c) * radius**2) / denom
    new_radius = abs(det) * radius / abs(denom)

    return new_center, new_radius


def find_contact_graph(circles, tol=1e-4):
    """Find which circles are in contact (or nearly so)."""
    n = len(circles)
    contacts = []
    for i in range(n):
        for j in range(i+1, n):
            dx = circles[i,0] - circles[j,0]
            dy = circles[i,1] - circles[j,1]
            dist = math.sqrt(dx*dx + dy*dy)
            gap = dist - (circles[i,2] + circles[j,2])
            if abs(gap) < tol:
                contacts.append((i, j, gap))
    return contacts

def find_wall_contacts(circles, tol=1e-4):
    """Find which circles touch walls."""
    wall_contacts = []
    for i, (x, y, r) in enumerate(circles):
        if abs(x - r) < tol: wall_contacts.append((i, 'left'))
        if abs(1 - x - r) < tol: wall_contacts.append((i, 'right'))
        if abs(y - r) < tol: wall_contacts.append((i, 'bottom'))
        if abs(1 - y - r) < tol: wall_contacts.append((i, 'top'))
    return wall_contacts

def find_clusters(circles, contacts, max_size=6):
    """Find clusters of adjacent circles from contact graph."""
    from collections import defaultdict
    adj = defaultdict(set)
    for i, j, _ in contacts:
        adj[i].add(j)
        adj[j].add(i)

    clusters = []
    n = len(circles)
    # For each circle, find its neighborhood
    for center_idx in range(n):
        neighbors = list(adj[center_idx])
        if len(neighbors) < 2:
            continue
        # Cluster = center + some neighbors
        for size in range(3, min(max_size+1, len(neighbors)+2)):
            for combo in itertools.combinations(neighbors, min(size-1, len(neighbors))):
                cluster = [center_idx] + list(combo)
                if len(cluster) >= 3:
                    clusters.append(cluster)

    return clusters


def mobius_deform_cluster(circles, cluster_indices, rng, strength=0.1):
    """
    Apply a random Mobius transformation to a cluster of circles.
    Then re-embed into the unit square.
    """
    cluster = circles[cluster_indices].copy()

    # Compute cluster centroid
    cx = np.mean(cluster[:, 0])
    cy = np.mean(cluster[:, 1])

    # Generate random Mobius parameters near identity
    # f(z) = (az+b)/(cz+d) near identity means a~1, d~1, b~0, c~0
    a = complex(1.0 + rng.normal(0, strength * 0.1), rng.normal(0, strength * 0.1))
    b = complex(rng.normal(0, strength * 0.05), rng.normal(0, strength * 0.05))
    c = complex(rng.normal(0, strength * 0.02), rng.normal(0, strength * 0.02))
    d = complex(1.0 + rng.normal(0, strength * 0.1), rng.normal(0, strength * 0.1))

    # Normalize so det = 1
    det = a * d - b * c
    if abs(det) < 1e-10:
        return None
    scale = det**0.5
    a, b, c, d = a/scale, b/scale, c/scale, d/scale

    new_circles = circles.copy()

    for idx, ci in zip(cluster_indices, range(len(cluster_indices))):
        center = complex(circles[idx, 0] - cx, circles[idx, 1] - cy)
        radius = circles[idx, 2]

        new_center, new_radius = apply_mobius_to_circle(center, radius, a, b, c, d)
        if new_center is None or new_radius is None or new_radius <= 0:
            return None

        new_x = new_center.real + cx
        new_y = new_center.imag + cy

        # Clamp to unit square
        new_r = max(new_radius, 1e-4)
        new_x = max(new_r, min(1 - new_r, new_x))
        new_y = max(new_r, min(1 - new_r, new_y))

        new_circles[idx] = [new_x, new_y, new_r]

    return new_circles


def inversive_distance(c1, c2):
    """Compute inversive distance between two circles."""
    dx = c1[0] - c2[0]
    dy = c1[1] - c2[1]
    dist_sq = dx*dx + dy*dy
    return (dist_sq - c1[2]**2 - c2[2]**2) / (2 * c1[2] * c2[2])


# ============ DIVERSE INITIALIZATIONS ============

def ring_init(n, rng, variant=0):
    """Ring-based initialization with variations."""
    circles = []

    if variant == 0:
        # Standard 1+8+12+4+1 ring
        circles.append([0.5, 0.5, 0.13])
        for i in range(8):
            theta = 2 * math.pi * i / 8 + rng.uniform(-0.1, 0.1)
            r = 0.10 + rng.uniform(-0.01, 0.01)
            circles.append([0.5 + 0.22 * math.cos(theta), 0.5 + 0.22 * math.sin(theta), r])
        for i in range(12):
            theta = 2 * math.pi * i / 12 + rng.uniform(-0.05, 0.05)
            r = 0.08 + rng.uniform(-0.01, 0.01)
            circles.append([0.5 + 0.38 * math.cos(theta), 0.5 + 0.38 * math.sin(theta), r])
        for i in range(4):
            theta = math.pi/4 + math.pi/2 * i
            r = 0.09
            circles.append([0.5 + 0.42 * math.cos(theta), 0.5 + 0.42 * math.sin(theta), r])
        circles.append([0.5, 0.08, 0.07])
    elif variant == 1:
        # 2+7+10+7 ring - different topology
        for i in range(2):
            circles.append([0.35 + 0.3*i, 0.5, 0.12])
        for i in range(7):
            theta = 2 * math.pi * i / 7
            r = 0.10
            circles.append([0.5 + 0.2 * math.cos(theta), 0.5 + 0.2 * math.sin(theta), r])
        for i in range(10):
            theta = 2 * math.pi * i / 10
            r = 0.08
            circles.append([0.5 + 0.37 * math.cos(theta), 0.5 + 0.37 * math.sin(theta), r])
        for i in range(7):
            theta = 2 * math.pi * i / 7 + math.pi/7
            r = 0.06
            circles.append([0.5 + 0.44 * math.cos(theta), 0.5 + 0.44 * math.sin(theta), r])
    elif variant == 2:
        # 4+8+14 - quad center
        offsets = [(-0.08, -0.08), (0.08, -0.08), (-0.08, 0.08), (0.08, 0.08)]
        for dx, dy in offsets:
            circles.append([0.5+dx, 0.5+dy, 0.10])
        for i in range(8):
            theta = 2 * math.pi * i / 8 + math.pi/8
            r = 0.10
            circles.append([0.5 + 0.25 * math.cos(theta), 0.5 + 0.25 * math.sin(theta), r])
        for i in range(14):
            theta = 2 * math.pi * i / 14
            r = 0.07
            circles.append([0.5 + 0.40 * math.cos(theta), 0.5 + 0.40 * math.sin(theta), r])
    elif variant == 3:
        # Hex grid inspired
        rows = [4, 5, 5, 5, 4, 3]
        y_pos = [0.12, 0.28, 0.44, 0.60, 0.76, 0.90]
        r_base = 0.08
        idx = 0
        for row_idx, (count, y) in enumerate(zip(rows, y_pos)):
            x_start = 0.5 - (count - 1) * 0.11
            for j in range(count):
                if idx >= n:
                    break
                circles.append([x_start + j * 0.22 + rng.uniform(-0.02, 0.02),
                               y + rng.uniform(-0.02, 0.02),
                               r_base + rng.uniform(-0.02, 0.02)])
                idx += 1
    else:
        # Random
        for i in range(n):
            r = rng.uniform(0.03, 0.12)
            x = rng.uniform(r, 1-r)
            y = rng.uniform(r, 1-r)
            circles.append([x, y, r])

    circles = np.array(circles[:n])
    # Clamp
    for i in range(len(circles)):
        circles[i, 2] = max(circles[i, 2], 0.02)
        circles[i, 0] = max(circles[i, 2], min(1 - circles[i, 2], circles[i, 0]))
        circles[i, 1] = max(circles[i, 2], min(1 - circles[i, 2], circles[i, 1]))
    return circles


def perturb_solution(circles, rng, strength=0.05):
    """Randomly perturb a solution."""
    new = circles.copy()
    n = len(circles)
    for i in range(n):
        new[i, 0] += rng.normal(0, strength * 0.1)
        new[i, 1] += rng.normal(0, strength * 0.1)
        new[i, 2] *= (1 + rng.normal(0, strength * 0.05))
        new[i, 2] = max(new[i, 2], 0.01)
        new[i, 0] = max(new[i, 2], min(1 - new[i, 2], new[i, 0]))
        new[i, 1] = max(new[i, 2], min(1 - new[i, 2], new[i, 1]))
    return new


# ============ MAIN SEARCH ============

def mobius_deformation_search(base_circles, n_trials=200, seed=SEED):
    """Search by applying Mobius deformations to the best known solution."""
    rng = np.random.RandomState(seed)

    contacts = find_contact_graph(base_circles)
    clusters = find_clusters(base_circles, contacts, max_size=5)

    print(f"Found {len(contacts)} contacts and {len(clusters)} clusters")

    best_metric = sum_radii(base_circles)
    best_circles = base_circles.copy()

    results = []

    for trial in range(n_trials):
        # Pick a random cluster
        if len(clusters) == 0:
            break
        cluster = clusters[rng.randint(len(clusters))]

        # Try different deformation strengths
        strength = rng.choice([0.01, 0.05, 0.1, 0.2, 0.5])

        deformed = mobius_deform_cluster(best_circles, cluster, rng, strength)
        if deformed is None:
            continue

        # Optimize
        try:
            opt_circles, metric = optimize_slsqp(deformed, maxiter=3000)
            valid, max_viol = validate(opt_circles)

            if valid and metric > best_metric + 1e-12:
                print(f"  Trial {trial}: IMPROVED {best_metric:.10f} -> {metric:.10f} "
                      f"(cluster={cluster}, strength={strength:.2f})")
                best_metric = metric
                best_circles = opt_circles.copy()

            results.append({
                'trial': trial,
                'metric': metric,
                'valid': valid,
                'max_viol': max_viol,
                'cluster_size': len(cluster),
                'strength': strength
            })
        except Exception as e:
            pass

        if (trial + 1) % 20 == 0:
            print(f"  Mobius trial {trial+1}/{n_trials}: best={best_metric:.10f}")

    return best_circles, best_metric, results


def multi_start_search(n_starts=50, seed=SEED):
    """Multi-start optimization with diverse initializations."""
    rng = np.random.RandomState(seed)

    best_metric = 0.0
    best_circles = None
    results = []

    for start in range(n_starts):
        variant = start % 5
        init = ring_init(N, rng, variant=variant)

        try:
            opt_circles, metric = optimize_slsqp(init, maxiter=5000)
            valid, max_viol = validate(opt_circles)

            if valid and metric > best_metric:
                print(f"  Start {start}: metric={metric:.10f} (variant={variant}) NEW BEST")
                best_metric = metric
                best_circles = opt_circles.copy()

            results.append({
                'start': start,
                'metric': metric,
                'valid': valid,
                'variant': variant
            })
        except Exception as e:
            pass

        if (start + 1) % 10 == 0:
            print(f"  Multi-start {start+1}/{n_starts}: best={best_metric:.10f}")

    return best_circles, best_metric, results


def combined_search(base_circles, seed=SEED):
    """
    Combined strategy:
    1. Mobius deformations on best known solution
    2. Multi-start from scratch
    3. Basin hopping with Mobius perturbations
    """
    print("=" * 60)
    print("MOBIUS-001: Combined Inversive Geometry Search")
    print("=" * 60)

    base_metric = sum_radii(base_circles)
    print(f"\nBase solution metric: {base_metric:.10f}")

    overall_best = base_metric
    overall_best_circles = base_circles.copy()
    all_results = {}

    # Phase 1: Mobius deformations
    print("\n--- Phase 1: Mobius Deformations ---")
    t0 = time.time()
    mob_circles, mob_metric, mob_results = mobius_deformation_search(
        base_circles, n_trials=150, seed=seed)
    t1 = time.time()
    print(f"Phase 1 done in {t1-t0:.1f}s. Best: {mob_metric:.10f}")
    all_results['mobius'] = mob_results

    if mob_metric > overall_best + 1e-12:
        overall_best = mob_metric
        overall_best_circles = mob_circles.copy()

    # Phase 2: Multi-start
    print("\n--- Phase 2: Multi-Start Search ---")
    t0 = time.time()
    ms_circles, ms_metric, ms_results = multi_start_search(
        n_starts=40, seed=seed + 1000)
    t1 = time.time()
    print(f"Phase 2 done in {t1-t0:.1f}s. Best: {ms_metric:.10f}")
    all_results['multistart'] = ms_results

    if ms_metric > overall_best + 1e-12:
        overall_best = ms_metric
        overall_best_circles = ms_circles.copy()

    # Phase 3: Basin hopping with Mobius perturbations from best found
    print("\n--- Phase 3: Basin Hopping ---")
    rng = np.random.RandomState(seed + 2000)
    t0 = time.time()
    hop_results = []

    for hop in range(100):
        # Perturb with a mix of Euclidean and Mobius moves
        if rng.random() < 0.5:
            perturbed = perturb_solution(overall_best_circles, rng,
                                         strength=rng.choice([0.02, 0.05, 0.1, 0.2]))
        else:
            contacts = find_contact_graph(overall_best_circles)
            clusters = find_clusters(overall_best_circles, contacts, max_size=5)
            if clusters:
                cluster = clusters[rng.randint(len(clusters))]
                perturbed = mobius_deform_cluster(overall_best_circles, cluster, rng,
                                                   strength=rng.choice([0.05, 0.1, 0.2, 0.5]))
                if perturbed is None:
                    continue
            else:
                perturbed = perturb_solution(overall_best_circles, rng, strength=0.1)

        try:
            opt_circles, metric = optimize_slsqp_analytical(perturbed, maxiter=5000)
            valid, max_viol = validate(opt_circles)

            if valid and metric > overall_best + 1e-12:
                print(f"  Hop {hop}: IMPROVED {overall_best:.10f} -> {metric:.10f}")
                overall_best = metric
                overall_best_circles = opt_circles.copy()

            hop_results.append({'hop': hop, 'metric': metric, 'valid': valid})
        except:
            pass

        if (hop + 1) % 20 == 0:
            print(f"  Hop {hop+1}/100: best={overall_best:.10f}")

    t1 = time.time()
    print(f"Phase 3 done in {t1-t0:.1f}s. Best: {overall_best:.10f}")
    all_results['basin_hop'] = hop_results

    # Phase 4: Final polish with analytical Jacobians
    print("\n--- Phase 4: Final Polish ---")
    polished, polish_metric = optimize_slsqp_analytical(overall_best_circles, maxiter=20000)
    valid, max_viol = validate(polished)
    if valid and polish_metric > overall_best:
        overall_best = polish_metric
        overall_best_circles = polished.copy()
    print(f"Final metric: {overall_best:.10f} (valid={valid})")

    print("\n" + "=" * 60)
    print(f"FINAL BEST: {overall_best:.10f}")
    print(f"Improvement over base: {overall_best - base_metric:.2e}")
    print("=" * 60)

    return overall_best_circles, overall_best, all_results


if __name__ == "__main__":
    # Load best known solution
    base = load_solution(SOLUTION_PATH)
    print(f"Loaded {len(base)} circles, sum(r) = {sum_radii(base):.10f}")

    # Run combined search
    best_circles, best_metric, results = combined_search(base, seed=SEED)

    # Save result
    output_path = OUTPUT_DIR / "solution_n26.json"
    save_solution(best_circles, output_path)
    print(f"\nSaved to {output_path}")

    # Validate with evaluator
    valid, max_viol = validate(best_circles)
    print(f"Valid: {valid}, Max violation: {max_viol:.2e}")
