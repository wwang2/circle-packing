#!/usr/bin/env python3
"""
Inversive distance gradient descent approach.

Instead of optimizing (x,y,r) directly, parameterize by radii only and
reconstruct positions from the contact graph. This decouples topology from geometry.

Also tries: simulated annealing in radius space with contact graph reconstruction.
"""

import json
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution, dual_annealing
from scipy.spatial import Delaunay
from pathlib import Path
import time

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

def optimize_slsqp(init_circles, maxiter=5000):
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


def radius_only_optimization(base_circles, n_trials=200):
    """
    Fix the topology (contact graph), optimize only radii.
    Then reconstruct positions via force-directed placement.
    """
    n = len(base_circles)

    # Extract contact graph from base solution
    contacts = []
    wall_contacts = {i: [] for i in range(n)}  # 'left', 'right', 'bottom', 'top'

    for i in range(n):
        x, y, r = base_circles[i]
        if abs(x - r) < 1e-6:
            wall_contacts[i].append('left')
        if abs(x + r - 1) < 1e-6:
            wall_contacts[i].append('right')
        if abs(y - r) < 1e-6:
            wall_contacts[i].append('bottom')
        if abs(y + r - 1) < 1e-6:
            wall_contacts[i].append('top')

    for i in range(n):
        for j in range(i+1, n):
            dx = base_circles[i, 0] - base_circles[j, 0]
            dy = base_circles[i, 1] - base_circles[j, 1]
            d = math.sqrt(dx*dx + dy*dy)
            gap = d - (base_circles[i, 2] + base_circles[j, 2])
            if abs(gap) < 1e-6:
                contacts.append((i, j))

    fp(f"  Contact graph: {len(contacts)} circle-circle, wall contacts extracted")

    best_metric = 0
    best_circles = None
    rng = np.random.RandomState(42)

    for trial in range(n_trials):
        # Perturb radii
        radii = base_circles[:, 2].copy()
        perturbation = rng.normal(0, 0.01 * (1 + trial / 50), n)
        radii = radii * (1 + perturbation)
        radii = np.clip(radii, 0.005, 0.45)

        # Reconstruct positions using contact constraints via force-directed
        circles = np.zeros((n, 3))
        circles[:, 2] = radii
        circles[:, 0] = base_circles[:, 0].copy()
        circles[:, 1] = base_circles[:, 1].copy()

        # Force-directed relaxation
        for iteration in range(200):
            forces = np.zeros((n, 2))

            # Contact forces: push to exact tangency
            for (i, j) in contacts:
                dx = circles[i, 0] - circles[j, 0]
                dy = circles[i, 1] - circles[j, 1]
                d = math.sqrt(dx*dx + dy*dy)
                target = circles[i, 2] + circles[j, 2]
                if d < 1e-10:
                    continue
                error = d - target
                fx = error * dx / d * 0.3
                fy = error * dy / d * 0.3
                forces[i] -= np.array([fx, fy])
                forces[j] += np.array([fx, fy])

            # Wall forces
            for i in range(n):
                r = circles[i, 2]
                if 'left' in wall_contacts[i]:
                    forces[i, 0] += (r - circles[i, 0]) * 0.5
                if 'right' in wall_contacts[i]:
                    forces[i, 0] += (1 - r - circles[i, 0]) * 0.5
                if 'bottom' in wall_contacts[i]:
                    forces[i, 1] += (r - circles[i, 1]) * 0.5
                if 'top' in wall_contacts[i]:
                    forces[i, 1] += (1 - r - circles[i, 1]) * 0.5

            circles[:, 0] += forces[:, 0]
            circles[:, 1] += forces[:, 1]

            # Clip
            circles[:, 0] = np.clip(circles[:, 0], circles[:, 2] + 0.001, 1 - circles[:, 2] - 0.001)
            circles[:, 1] = np.clip(circles[:, 1], circles[:, 2] + 0.001, 1 - circles[:, 2] - 0.001)

        # Polish with SLSQP
        try:
            opt, metric = optimize_slsqp(circles, maxiter=3000)
            viol = max_violation(opt)
            if viol <= 1e-10 and metric > best_metric:
                best_metric = metric
                best_circles = opt.copy()
                if trial % 50 == 0:
                    fp(f"    trial {trial}: metric={metric:.10f}")
        except:
            pass

    return best_circles, best_metric


def dual_annealing_search(base_circles, maxiter=500):
    """
    Use scipy dual_annealing on the full (x,y,r) space.
    This is a global optimizer that combines simulated annealing with local search.
    """
    n = len(base_circles)

    def objective(x):
        xs = x[0::3]; ys = x[1::3]; rs = x[2::3]

        # Penalty for constraint violations
        penalty = 0.0
        for i in range(n):
            penalty += max(0, rs[i] - xs[i]) ** 2 * 1e6
            penalty += max(0, xs[i] + rs[i] - 1) ** 2 * 1e6
            penalty += max(0, rs[i] - ys[i]) ** 2 * 1e6
            penalty += max(0, ys[i] + rs[i] - 1) ** 2 * 1e6

        for i in range(n):
            for j in range(i+1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist_sq = dx*dx + dy*dy
                rsum = rs[i] + rs[j]
                overlap = rsum*rsum - dist_sq
                if overlap > 0:
                    penalty += overlap * 1e6

        return -np.sum(rs) + penalty

    bounds = [(0.0, 1.0), (0.0, 1.0), (0.005, 0.45)] * n
    x0 = base_circles.flatten()

    fp("  Running dual annealing (this may take a while)...")
    result = dual_annealing(objective, bounds, x0=x0, maxiter=maxiter,
                           seed=42, initial_temp=5230.0,
                           visit=2.62, accept=-5.0)

    out = result.x.reshape(n, 3)
    return out, -result.fun


def differential_evolution_search(base_circles, maxiter=300):
    """Use differential evolution as a global optimizer."""
    n = len(base_circles)

    def objective(x):
        xs = x[0::3]; ys = x[1::3]; rs = x[2::3]

        penalty = 0.0
        for i in range(n):
            penalty += max(0, rs[i] - xs[i]) ** 2 * 1e8
            penalty += max(0, xs[i] + rs[i] - 1) ** 2 * 1e8
            penalty += max(0, rs[i] - ys[i]) ** 2 * 1e8
            penalty += max(0, ys[i] + rs[i] - 1) ** 2 * 1e8

        for i in range(n):
            for j in range(i+1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist_sq = dx*dx + dy*dy
                rsum = rs[i] + rs[j]
                overlap = rsum*rsum - dist_sq
                if overlap > 0:
                    penalty += overlap * 1e8

        return -np.sum(rs) + penalty

    bounds = [(0.0, 1.0), (0.0, 1.0), (0.005, 0.45)] * n

    # Use the known solution to seed initial population
    x0 = base_circles.flatten()

    fp("  Running differential evolution...")
    result = differential_evolution(objective, bounds, x0=x0, maxiter=maxiter,
                                    seed=42, popsize=15, mutation=(0.5, 1.5),
                                    recombination=0.9, tol=1e-14,
                                    strategy='best1bin', polish=False)

    out = result.x.reshape(n, 3)
    return out


def symmetry_breaking_search(base_circles, n_trials=500):
    """
    Try breaking the approximate symmetries in the current best solution.
    Look for near-symmetric pairs and break them asymmetrically.
    """
    n = len(base_circles)
    rng = np.random.RandomState(31415)

    best_metric = sum_radii(base_circles)
    best_circles = base_circles.copy()

    # Find near-symmetric pairs (similar radii)
    radii = base_circles[:, 2]
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            if abs(radii[i] - radii[j]) / max(radii[i], radii[j]) < 0.05:
                pairs.append((i, j))

    fp(f"  Found {len(pairs)} near-symmetric pairs")

    for trial in range(n_trials):
        c = base_circles.copy()

        # Pick a random pair and break symmetry
        if pairs and rng.random() < 0.7:
            i, j = pairs[rng.randint(len(pairs))]
            # Make one bigger, one smaller
            delta = rng.uniform(0.005, 0.03)
            c[i, 2] += delta
            c[j, 2] -= delta * 0.8

        # Also add random perturbation
        n_perturb = rng.randint(1, min(8, n))
        indices = rng.choice(n, n_perturb, replace=False)
        for idx in indices:
            c[idx, 0] += rng.normal(0, 0.02)
            c[idx, 1] += rng.normal(0, 0.02)
            c[idx, 2] *= (1 + rng.normal(0, 0.05))

        c[:, 2] = np.clip(c[:, 2], 0.005, 0.45)
        c[:, 0] = np.clip(c[:, 0], c[:, 2] + 0.001, 1 - c[:, 2] - 0.001)
        c[:, 1] = np.clip(c[:, 1], c[:, 2] + 0.001, 1 - c[:, 2] - 0.001)

        try:
            opt, metric = optimize_slsqp(c, maxiter=3000)
            viol = max_violation(opt)
            if viol <= 1e-10 and metric > best_metric + 1e-12:
                best_metric = metric
                best_circles = opt.copy()
                fp(f"    trial {trial}: NEW BEST metric={metric:.10f}")
        except:
            pass

        if trial % 100 == 99:
            fp(f"    ... {trial+1}/{n_trials}")

    return best_circles, best_metric


def main():
    start_time = time.time()

    base = load_solution(OUTPUT_DIR / "solution_n26.json")
    best_metric = sum_radii(base)
    best_circles = base.copy()
    fp(f"Starting metric: {best_metric:.10f}")

    # Strategy 1: Radius-only optimization with topology preservation
    fp("\n=== Strategy 1: Radius-only optimization (200 trials) ===")
    c1, m1 = radius_only_optimization(base, n_trials=200)
    if c1 is not None and m1 > best_metric + 1e-12:
        best_metric = m1
        best_circles = c1.copy()
        fp(f"  Radius-only improved: {m1:.10f}")
    else:
        fp(f"  Radius-only best: {m1:.10f} (no improvement)")

    # Strategy 2: Symmetry breaking
    fp("\n=== Strategy 2: Symmetry breaking (500 trials) ===")
    c2, m2 = symmetry_breaking_search(base, n_trials=500)
    if m2 > best_metric + 1e-12:
        best_metric = m2
        best_circles = c2.copy()
        fp(f"  Symmetry breaking improved: {m2:.10f}")
    else:
        fp(f"  Symmetry breaking best: {m2:.10f} (no improvement)")

    # Strategy 3: Dual annealing (global)
    fp("\n=== Strategy 3: Dual annealing ===")
    try:
        c3 = dual_annealing_search(base, maxiter=200)
        # Polish
        c3_opt, m3 = optimize_slsqp(c3[0] if isinstance(c3, tuple) else c3, maxiter=5000)
        viol = max_violation(c3_opt)
        if viol <= 1e-10 and m3 > best_metric + 1e-12:
            best_metric = m3
            best_circles = c3_opt.copy()
            fp(f"  Dual annealing improved: {m3:.10f}")
        else:
            fp(f"  Dual annealing best: {m3:.10f}, viol={viol:.2e}")
    except Exception as e:
        fp(f"  Dual annealing failed: {e}")

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
