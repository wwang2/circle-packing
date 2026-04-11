#!/usr/bin/env python3
"""
Edge-flip topology search: Starting from the known optimal contact graph,
systematically flip edges to generate neighboring topologies and test each.

The idea: the optimal packing has a specific contact graph (triangulation).
By flipping edges in this triangulation, we generate all neighboring topologies.
Some of these might have higher sum-of-radii optima.

Also tries: removing contacts (making the graph non-maximal) and re-optimizing,
which can change the topology fundamentally.
"""

import json
import math
import numpy as np
from scipy.optimize import minimize
from pathlib import Path
import time
import itertools

WORKTREE = Path("/Users/wujiewang/code/circle-packing/.worktrees/mobius-001")
OUTPUT_DIR = WORKTREE / "orbits/mobius-001"
N = 26

def fp(*args, **kwargs):
    print(*args, **kwargs, flush=True)

def load_solution(path):
    with open(path) as f:
        return np.array(json.load(f)["circles"])

def save_solution(circles, path):
    data = {"circles": [[float(c[0]), float(c[1]), float(c[2])] for c in circles]}
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def sum_radii(c):
    return float(np.sum(c[:, 2]))

def max_violation(circles):
    n = len(circles)
    mv = 0.0
    for i in range(n):
        x, y, r = circles[i]
        mv = max(mv, r - x, x + r - 1, r - y, y + r - 1)
    for i in range(n):
        for j in range(i + 1, n):
            dx = circles[i, 0] - circles[j, 0]
            dy = circles[i, 1] - circles[j, 1]
            d = math.sqrt(dx * dx + dy * dy)
            mv = max(mv, circles[i, 2] + circles[j, 2] - d)
    return mv

def extract_contact_graph(circles, tol=1e-5):
    """Extract the contact graph from a circle packing."""
    n = len(circles)
    edges = set()
    wall = {i: set() for i in range(n)}

    for i in range(n):
        x, y, r = circles[i]
        if abs(x - r) < tol:
            wall[i].add('L')
        if abs(x + r - 1) < tol:
            wall[i].add('R')
        if abs(y - r) < tol:
            wall[i].add('B')
        if abs(y + r - 1) < tol:
            wall[i].add('T')

    for i in range(n):
        for j in range(i+1, n):
            dx = circles[i, 0] - circles[j, 0]
            dy = circles[i, 1] - circles[j, 1]
            d = math.sqrt(dx*dx + dy*dy)
            gap = d - (circles[i, 2] + circles[j, 2])
            if abs(gap) < tol:
                edges.add((i, j))

    return edges, wall

def optimize_with_target_contacts(init_circles, target_contacts, maxiter=5000):
    """
    Optimize packing, encouraging specific circle-circle tangencies.
    Uses penalty terms to push pairs toward tangency.
    """
    n = len(init_circles)
    x0 = init_circles.flatten()

    contact_set = set(target_contacts)

    def objective(x):
        xs = x[0::3]; ys = x[1::3]; rs = x[2::3]
        obj = -np.sum(rs)

        # Add attraction penalty for desired contacts
        for (i, j) in contact_set:
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            d = math.sqrt(dx*dx + dy*dy)
            target = rs[i] + rs[j]
            gap = d - target
            if gap > 0:
                obj += 0.1 * gap ** 2  # Attract

        return obj

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
            rsum = rs[i] + rs[i+1:]
            dist_sq = dx*dx + dy*dy
            vals.extend(dist_sq - rsum**2)
        return np.array(vals)

    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * n
    result = minimize(objective, x0, method='SLSQP',
                      bounds=bounds, constraints=[{'type': 'ineq', 'fun': all_con}],
                      options={'maxiter': maxiter, 'ftol': 1e-15})
    out = result.x.reshape(n, 3)
    return out, -np.sum(out[:, 2])

def optimize_standard(init_circles, maxiter=5000):
    n = len(init_circles)
    x0 = init_circles.flatten()

    def objective(x):
        return -np.sum(x[2::3])

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
            rsum = rs[i] + rs[i+1:]
            dist_sq = dx*dx + dy*dy
            vals.extend(dist_sq - rsum**2)
        return np.array(vals)

    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * n
    result = minimize(objective, x0, method='SLSQP',
                      bounds=bounds, constraints=[{'type': 'ineq', 'fun': all_con}],
                      options={'maxiter': maxiter, 'ftol': 1e-15})
    out = result.x.reshape(n, 3)
    return out, -result.fun


def separate_and_rejoin(circles, indices_to_move, dx, dy):
    """Move a subset of circles and re-optimize."""
    c = circles.copy()
    for i in indices_to_move:
        c[i, 0] += dx
        c[i, 1] += dy
        c[i, 0] = np.clip(c[i, 0], c[i, 2] + 0.001, 1 - c[i, 2] - 0.001)
        c[i, 1] = np.clip(c[i, 1], c[i, 2] + 0.001, 1 - c[i, 2] - 0.001)
    return c


def main():
    start_time = time.time()

    base = load_solution(OUTPUT_DIR / "solution_n26.json")
    best_metric = sum_radii(base)
    best_circles = base.copy()
    fp(f"Starting metric: {best_metric:.10f}")

    # Extract contact graph
    edges, wall = extract_contact_graph(base)
    edge_list = sorted(edges)
    fp(f"Contact graph: {len(edge_list)} edges, wall contacts: {sum(len(v) for v in wall.values())}")

    # Build adjacency
    adj = {i: set() for i in range(N)}
    for (i, j) in edge_list:
        adj[i].add(j)
        adj[j].add(i)

    fp(f"Degree sequence: {sorted([len(adj[i]) for i in range(N)], reverse=True)}")

    # Strategy 1: Single edge removal + re-optimize
    fp(f"\n=== Strategy 1: Remove single edge ({len(edge_list)} trials) ===")
    rng = np.random.RandomState(42)
    s1_best = 0

    for idx, (i, j) in enumerate(edge_list):
        # Break the contact by slightly separating the circles
        c = base.copy()
        dx = c[i, 0] - c[j, 0]
        dy = c[i, 1] - c[j, 1]
        d = math.sqrt(dx*dx + dy*dy)
        if d > 1e-10:
            # Push them slightly apart
            sep = 0.005
            c[i, 0] += sep * dx / d
            c[i, 1] += sep * dy / d
            c[j, 0] -= sep * dx / d
            c[j, 1] -= sep * dy / d
            c[:, 0] = np.clip(c[:, 0], c[:, 2] + 0.001, 1 - c[:, 2] - 0.001)
            c[:, 1] = np.clip(c[:, 1], c[:, 2] + 0.001, 1 - c[:, 2] - 0.001)

        try:
            opt, metric = optimize_standard(c, maxiter=3000)
            viol = max_violation(opt)
            if viol <= 1e-10:
                s1_best = max(s1_best, metric)
                if metric > best_metric + 1e-12:
                    best_metric = metric
                    best_circles = opt.copy()
                    fp(f"  IMPROVED! Remove ({i},{j}): {metric:.10f}")
        except:
            pass

    fp(f"  Single edge removal best: {s1_best:.10f}")

    # Strategy 2: Remove edge pairs
    fp(f"\n=== Strategy 2: Remove edge pairs (500 random trials) ===")
    s2_best = 0

    for trial in range(500):
        # Pick 2-3 random edges to remove
        n_remove = rng.randint(2, 4)
        to_remove = rng.choice(len(edge_list), n_remove, replace=False)

        c = base.copy()
        for idx in to_remove:
            i, j = edge_list[idx]
            dx = c[i, 0] - c[j, 0]
            dy = c[i, 1] - c[j, 1]
            d = math.sqrt(dx*dx + dy*dy)
            if d > 1e-10:
                sep = 0.003
                c[i, 0] += sep * dx / d
                c[i, 1] += sep * dy / d
                c[j, 0] -= sep * dx / d
                c[j, 1] -= sep * dy / d

        c[:, 0] = np.clip(c[:, 0], c[:, 2] + 0.001, 1 - c[:, 2] - 0.001)
        c[:, 1] = np.clip(c[:, 1], c[:, 2] + 0.001, 1 - c[:, 2] - 0.001)

        try:
            opt, metric = optimize_standard(c, maxiter=3000)
            viol = max_violation(opt)
            if viol <= 1e-10:
                s2_best = max(s2_best, metric)
                if metric > best_metric + 1e-12:
                    best_metric = metric
                    best_circles = opt.copy()
                    fp(f"  IMPROVED! trial {trial}: {metric:.10f}")
        except:
            pass

        if trial % 100 == 99:
            fp(f"    ... {trial+1}/500, best={s2_best:.10f}")

    fp(f"  Edge pair removal best: {s2_best:.10f}")

    # Strategy 3: Cluster displacement - move a connected component
    fp(f"\n=== Strategy 3: Cluster displacement (200 trials) ===")
    s3_best = 0

    for trial in range(200):
        # Pick a random circle and its neighbors
        center = rng.randint(N)
        depth = rng.randint(1, 3)
        cluster = {center}
        frontier = {center}
        for _ in range(depth):
            new_frontier = set()
            for node in frontier:
                new_frontier.update(adj[node])
            frontier = new_frontier - cluster
            cluster.update(frontier)

        cluster = list(cluster)
        if len(cluster) < 2 or len(cluster) > N - 2:
            continue

        # Random displacement
        dx = rng.normal(0, 0.05)
        dy = rng.normal(0, 0.05)

        c = separate_and_rejoin(base, cluster, dx, dy)

        try:
            opt, metric = optimize_standard(c, maxiter=3000)
            viol = max_violation(opt)
            if viol <= 1e-10:
                s3_best = max(s3_best, metric)
                if metric > best_metric + 1e-12:
                    best_metric = metric
                    best_circles = opt.copy()
                    fp(f"  IMPROVED! trial {trial}: {metric:.10f}")
        except:
            pass

        if trial % 50 == 49:
            fp(f"    ... {trial+1}/200, best={s3_best:.10f}")

    fp(f"  Cluster displacement best: {s3_best:.10f}")

    # Strategy 4: Circle swap + optimize
    fp(f"\n=== Strategy 4: Circle swap (300 trials) ===")
    s4_best = 0

    for trial in range(300):
        c = base.copy()
        # Swap 1-3 pairs of circles (positions, not radii)
        n_swaps = rng.randint(1, 4)
        for _ in range(n_swaps):
            i, j = rng.choice(N, 2, replace=False)
            # Swap positions but keep radii
            c[i, 0], c[j, 0] = c[j, 0], c[i, 0]
            c[i, 1], c[j, 1] = c[j, 1], c[i, 1]

        # Ensure feasibility
        c[:, 0] = np.clip(c[:, 0], c[:, 2] + 0.001, 1 - c[:, 2] - 0.001)
        c[:, 1] = np.clip(c[:, 1], c[:, 2] + 0.001, 1 - c[:, 2] - 0.001)

        try:
            opt, metric = optimize_standard(c, maxiter=3000)
            viol = max_violation(opt)
            if viol <= 1e-10:
                s4_best = max(s4_best, metric)
                if metric > best_metric + 1e-12:
                    best_metric = metric
                    best_circles = opt.copy()
                    fp(f"  IMPROVED! trial {trial}: {metric:.10f}")
        except:
            pass

        if trial % 100 == 99:
            fp(f"    ... {trial+1}/300, best={s4_best:.10f}")

    fp(f"  Circle swap best: {s4_best:.10f}")

    # Strategy 5: Insert/remove circle (change n locally, then re-add)
    fp(f"\n=== Strategy 5: Remove + re-add circle (N trials) ===")
    s5_best = 0

    for target in range(N):
        c = base.copy()
        # Remove the smallest few circles and try re-inserting them
        radii_order = np.argsort(c[:, 2])

        # Remove target circle
        removed = c[target].copy()
        c_reduced = np.delete(c, target, axis=0)

        # Try 10 random positions for the removed circle
        for _ in range(10):
            c_new = np.zeros((N, 3))
            c_new[:target] = c_reduced[:target]
            c_new[target+1:] = c_reduced[target:]

            # Random new position
            r = removed[2] * (0.5 + rng.random())
            x = r + rng.random() * (1 - 2*r)
            y = r + rng.random() * (1 - 2*r)
            c_new[target] = [x, y, r]

            try:
                opt, metric = optimize_standard(c_new, maxiter=3000)
                viol = max_violation(opt)
                if viol <= 1e-10:
                    s5_best = max(s5_best, metric)
                    if metric > best_metric + 1e-12:
                        best_metric = metric
                        best_circles = opt.copy()
                        fp(f"  IMPROVED! remove+readd circle {target}: {metric:.10f}")
            except:
                pass

    fp(f"  Remove + re-add best: {s5_best:.10f}")

    elapsed = time.time() - start_time
    fp(f"\n=== SUMMARY ===")
    fp(f"Time: {elapsed:.1f}s")
    fp(f"Best metric: {best_metric:.10f}")
    fp(f"Valid: {max_violation(best_circles) <= 1e-10}")

    current = load_solution(OUTPUT_DIR / "solution_n26.json")
    if best_metric > sum_radii(current) + 1e-14:
        save_solution(best_circles, OUTPUT_DIR / "solution_n26.json")
        fp("Saved new best!")
    else:
        fp(f"No improvement over current ({sum_radii(current):.10f})")


if __name__ == "__main__":
    main()
