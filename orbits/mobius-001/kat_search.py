"""
KAT (Koebe-Andreev-Thurston) topology search.

Strategy: Generate random maximal planar graphs on 26+4 vertices (26 circles + 4 wall segments),
then use iterative algorithms to realize them as circle packings.
Score by sum of radii.

The key insight: different contact graph topologies may lead to different local optima.
The known best has a specific topology (58 circle-circle contacts + 20 wall contacts).
We want to explore OTHER topologies systematically.
"""

import json
import math
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
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


def generate_random_planar_graph(n, rng, method='delaunay'):
    """Generate a random planar graph on n vertices via Delaunay triangulation."""
    if method == 'delaunay':
        # Random points in [0,1]^2
        points = rng.uniform(0.05, 0.95, size=(n, 2))
        tri = Delaunay(points)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    e = (min(simplex[i], simplex[j]), max(simplex[i], simplex[j]))
                    edges.add(e)
        return points, list(edges)
    elif method == 'ring':
        # Points arranged in concentric rings
        points = []
        # Center
        points.append([0.5, 0.5])
        # Inner ring
        n_inner = min(8, n-1)
        for i in range(n_inner):
            theta = 2*math.pi*i/n_inner + rng.uniform(-0.1, 0.1)
            r = 0.2 + rng.uniform(-0.03, 0.03)
            points.append([0.5 + r*math.cos(theta), 0.5 + r*math.sin(theta)])
        # Outer ring
        remaining = n - 1 - n_inner
        for i in range(remaining):
            theta = 2*math.pi*i/remaining + rng.uniform(-0.1, 0.1)
            r = 0.38 + rng.uniform(-0.03, 0.03)
            points.append([0.5 + r*math.cos(theta), 0.5 + r*math.sin(theta)])
        points = np.array(points[:n])
        tri = Delaunay(points)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    e = (min(simplex[i], simplex[j]), max(simplex[i], simplex[j]))
                    edges.add(e)
        return points, list(edges)
    elif method == 'perturbed_hex':
        # Hex grid with perturbation
        points = []
        row = 0
        while len(points) < n:
            y = 0.1 + row * 0.17
            if y > 0.92: break
            offset = 0.09 if row % 2 else 0.0
            x = 0.1 + offset
            while x < 0.92 and len(points) < n:
                points.append([x + rng.uniform(-0.04, 0.04),
                              y + rng.uniform(-0.04, 0.04)])
                x += 0.18
            row += 1
        while len(points) < n:
            points.append([rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)])
        points = np.array(points[:n])
        tri = Delaunay(points)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    e = (min(simplex[i], simplex[j]), max(simplex[i], simplex[j]))
                    edges.add(e)
        return points, list(edges)


def topology_to_init(points, edges, n, rng):
    """
    Convert a planar graph topology to an initial circle packing.
    Use the graph structure to assign radii based on vertex degree.
    """
    from collections import Counter
    degree = Counter()
    for i, j in edges:
        degree[i] += 1
        degree[j] += 1

    circles = np.zeros((n, 3))
    # Scale radii inversely with degree (high-degree vertices = smaller circles)
    max_deg = max(degree.values()) if degree else 6
    for i in range(n):
        d = degree.get(i, 3)
        # Radius proportional to 1/sqrt(degree), scaled to fit
        circles[i, 0] = points[i, 0]
        circles[i, 1] = points[i, 1]
        circles[i, 2] = 0.04 + 0.08 * (1.0 - d / (max_deg + 1))

    # Scale radii down to avoid overlaps
    for _ in range(10):
        any_overlap = False
        for i, j in edges:
            dx = circles[i, 0] - circles[j, 0]
            dy = circles[i, 1] - circles[j, 1]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < circles[i, 2] + circles[j, 2]:
                scale = 0.95 * dist / (circles[i, 2] + circles[j, 2])
                circles[i, 2] *= scale
                circles[j, 2] *= scale
                any_overlap = True
        if not any_overlap:
            break

    # Ensure containment
    for i in range(n):
        r = circles[i, 2]
        circles[i, 0] = max(r, min(1-r, circles[i, 0]))
        circles[i, 1] = max(r, min(1-r, circles[i, 1]))

    return circles


def optimize_slsqp(circles, maxiter=5000):
    """Optimize with SLSQP and analytical Jacobians."""
    n = len(circles)
    x0 = circles.flatten()

    def objective(x):
        return -np.sum(x[2::3])

    def grad_objective(x):
        g = np.zeros_like(x)
        g[2::3] = -1.0
        return g

    constraints = []
    nn = 3 * n

    for i in range(n):
        ix, iy, ir = 3*i, 3*i+1, 3*i+2

        for idx_pos, sign, offset in [(ix, 1.0, 0.0), (ix, -1.0, 1.0),
                                       (iy, 1.0, 0.0), (iy, -1.0, 1.0)]:
            def make_cont(ip, ir_, sp, off):
                def fun(x, ip=ip, ir_=ir_, sp=sp, off=off):
                    return sp * x[ip] - x[ir_] + off
                def jac(x, ip=ip, ir_=ir_, sp=sp):
                    g = np.zeros(nn)
                    g[ip] = sp
                    g[ir_] = -1.0
                    return g
                return fun, jac
            f, j = make_cont(idx_pos, ir, sign, offset)
            constraints.append({'type': 'ineq', 'fun': f, 'jac': j})

        def make_rpos(ir_):
            def fun(x, ir_=ir_): return x[ir_] - 1e-6
            def jac(x, ir_=ir_):
                g = np.zeros(nn)
                g[ir_] = 1.0
                return g
            return fun, jac
        f, j = make_rpos(ir)
        constraints.append({'type': 'ineq', 'fun': f, 'jac': j})

    for i in range(n):
        for j in range(i+1, n):
            ix, iy, ir = 3*i, 3*i+1, 3*i+2
            jx, jy, jr = 3*j, 3*j+1, 3*j+2

            def make_sep(a, b, c, d, e, f_):
                def fun(x, a=a, b=b, c=c, d=d, e=e, f=f_):
                    dx = x[a] - x[d]
                    dy = x[b] - x[e]
                    return dx*dx + dy*dy - (x[c] + x[f])**2
                def jac(x, a=a, b=b, c=c, d=d, e=e, f=f_):
                    g = np.zeros(nn)
                    dx = x[a] - x[d]
                    dy = x[b] - x[e]
                    sr = x[c] + x[f]
                    g[a] = 2*dx; g[d] = -2*dx
                    g[b] = 2*dy; g[e] = -2*dy
                    g[c] = -2*sr; g[f] = -2*sr
                    return g
                return fun, jac
            f, j = make_sep(ix, iy, ir, jx, jy, jr)
            constraints.append({'type': 'ineq', 'fun': f, 'jac': j})

    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * n

    result = minimize(objective, x0, method='SLSQP', jac=grad_objective,
                     bounds=bounds, constraints=constraints,
                     options={'maxiter': maxiter, 'ftol': 1e-15, 'disp': False})

    out = result.x.reshape(n, 3)
    return out, -result.fun


def kat_topology_search(n_topologies=100, seed=SEED):
    """Search over different topologies via random planar graph generation."""
    rng = np.random.RandomState(seed)

    best_metric = 0.0
    best_circles = None
    results = []

    methods = ['delaunay', 'ring', 'perturbed_hex']

    print(f"Searching {n_topologies} random topologies...")

    for t in range(n_topologies):
        method = methods[t % len(methods)]

        try:
            points, edges = generate_random_planar_graph(N, rng, method=method)
            init = topology_to_init(points, edges, N, rng)

            opt_circles, metric = optimize_slsqp(init, maxiter=5000)
            valid, max_viol = validate(opt_circles)

            if valid and metric > best_metric:
                print(f"  Topo {t}: metric={metric:.10f} ({method}) NEW BEST")
                best_metric = metric
                best_circles = opt_circles.copy()

            results.append({
                'topology': t,
                'method': method,
                'metric': metric if valid else 0.0,
                'valid': valid,
                'n_edges': len(edges)
            })
        except Exception as e:
            results.append({'topology': t, 'method': method, 'metric': 0.0,
                           'valid': False, 'error': str(e)})

        if (t+1) % 20 == 0:
            print(f"  Topology {t+1}/{n_topologies}: best={best_metric:.10f}")

    return best_circles, best_metric, results


def inversive_perturbation_search(base_circles, n_trials=200, seed=SEED):
    """
    Perturb solution in inversive distance space.

    Instead of perturbing (x,y,r) directly, we:
    1. Compute pairwise inversive distances
    2. Perturb these inversive distances
    3. Reconstruct a packing from the perturbed distances
    4. Optimize to feasibility
    """
    rng = np.random.RandomState(seed)
    n = len(base_circles)

    best_metric = sum_radii(base_circles)
    best_circles = base_circles.copy()

    print(f"Inversive perturbation search ({n_trials} trials)...")

    for trial in range(n_trials):
        # Strategy: swap radii between pairs while keeping tangency
        new_circles = base_circles.copy()

        n_swaps = rng.randint(1, 5)
        for _ in range(n_swaps):
            i, j = rng.choice(n, 2, replace=False)

            # Mix radii: r_i' = alpha*r_i + (1-alpha)*r_j, etc.
            alpha = rng.uniform(0.3, 0.7)
            ri_new = alpha * new_circles[i, 2] + (1 - alpha) * new_circles[j, 2]
            rj_new = (1 - alpha) * new_circles[i, 2] + alpha * new_circles[j, 2]

            # Adjust positions to maintain approximate tangency
            dx = new_circles[j, 0] - new_circles[i, 0]
            dy = new_circles[j, 1] - new_circles[i, 1]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < 1e-10:
                continue

            old_sum = new_circles[i, 2] + new_circles[j, 2]
            new_sum = ri_new + rj_new

            # Scale positions to match new radii
            if old_sum > 1e-10:
                scale = new_sum / old_sum
                mid_x = (new_circles[i, 0] + new_circles[j, 0]) / 2
                mid_y = (new_circles[i, 1] + new_circles[j, 1]) / 2

                new_circles[i, 0] = mid_x - (mid_x - new_circles[i, 0]) * scale
                new_circles[i, 1] = mid_y - (mid_y - new_circles[i, 1]) * scale
                new_circles[j, 0] = mid_x - (mid_x - new_circles[j, 0]) * scale
                new_circles[j, 1] = mid_y - (mid_y - new_circles[j, 1]) * scale

            new_circles[i, 2] = ri_new
            new_circles[j, 2] = rj_new

        # Add small random perturbation
        noise = rng.normal(0, 0.01, size=new_circles.shape)
        noise[:, 2] *= 0.3  # Less noise on radii
        new_circles += noise

        # Clamp
        for i in range(n):
            new_circles[i, 2] = max(new_circles[i, 2], 0.01)
            new_circles[i, 0] = max(new_circles[i, 2], min(1 - new_circles[i, 2], new_circles[i, 0]))
            new_circles[i, 1] = max(new_circles[i, 2], min(1 - new_circles[i, 2], new_circles[i, 1]))

        try:
            opt, metric = optimize_slsqp(new_circles, maxiter=5000)
            valid, max_viol = validate(opt)

            if valid and metric > best_metric + 1e-12:
                print(f"  Inversive trial {trial}: IMPROVED {best_metric:.10f} -> {metric:.10f}")
                best_metric = metric
                best_circles = opt.copy()
        except:
            pass

        if (trial+1) % 50 == 0:
            print(f"  Inversive trial {trial+1}/{n_trials}: best={best_metric:.10f}")

    return best_circles, best_metric


if __name__ == "__main__":
    # Load base solution
    base_path = WORKTREE / "orbits/topo-001/solution_n26.json"
    with open(base_path) as f:
        data = json.load(f)
    base = np.array(data["circles"])
    print(f"Base metric: {sum_radii(base):.10f}")

    print("\n" + "="*60)
    print("Phase A: KAT Topology Search")
    print("="*60)
    t0 = time.time()
    kat_circles, kat_metric, kat_results = kat_topology_search(n_topologies=80, seed=SEED)
    print(f"KAT search: {time.time()-t0:.1f}s, best={kat_metric:.10f}")

    print("\n" + "="*60)
    print("Phase B: Inversive Perturbation")
    print("="*60)
    t0 = time.time()
    inv_circles, inv_metric = inversive_perturbation_search(base, n_trials=150, seed=SEED+100)
    print(f"Inversive search: {time.time()-t0:.1f}s, best={inv_metric:.10f}")

    # Pick overall best
    overall_best = base.copy()
    overall_metric = sum_radii(base)

    if kat_circles is not None and kat_metric > overall_metric:
        overall_best = kat_circles
        overall_metric = kat_metric
    if inv_circles is not None and inv_metric > overall_metric:
        overall_best = inv_circles
        overall_metric = inv_metric

    print(f"\nOverall best: {overall_metric:.10f}")

    # Save
    save_solution(overall_best, OUTPUT_DIR / "solution_n26_kat.json")
    print(f"Saved to {OUTPUT_DIR / 'solution_n26_kat.json'}")
