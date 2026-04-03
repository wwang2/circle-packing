"""
Topology Jump Search for Circle Packing n=26.

Instead of random starts, we start from the KNOWN BEST solution and make
carefully designed perturbations that FORCE a different contact topology.

Strategy:
1. Extract the contact graph of the known solution
2. For each subset of 3-8 contacts, BREAK them by moving circles apart
3. Re-optimize with SLSQP - the optimizer will find a NEW basin
4. If the new basin has higher metric, we've found improvement

The key insight: at the known optimum, all 78 constraints are active.
If we break K contacts, we create K degrees of freedom. The optimizer
uses these DOFs to potentially find a different topology.
"""

import json
import numpy as np
from scipy.optimize import minimize
import os
import sys
import time
import random
from itertools import combinations

WORKDIR = os.path.dirname(os.path.abspath(__file__))
N = 26
SEED = 42


def load_solution(path):
    with open(path) as f:
        data = json.load(f)
    circles = np.array(data["circles"])
    return circles[:, 0], circles[:, 1], circles[:, 2]


def save_solution(x, y, r, path):
    circles = [[float(x[i]), float(y[i]), float(r[i])] for i in range(len(x))]
    with open(path, 'w') as f:
        json.dump({"circles": circles}, f, indent=2)


def is_feasible(x, y, r, tol=1e-10):
    n = len(x)
    for i in range(n):
        if r[i] <= 0: return False
        if x[i] - r[i] < -tol or 1 - x[i] - r[i] < -tol: return False
        if y[i] - r[i] < -tol or 1 - y[i] - r[i] < -tol: return False
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if dist < r[i] + r[j] - tol: return False
    return True


def extract_contacts(x, y, r, tol=1e-6):
    """Extract circle-circle and wall contacts."""
    n = len(x)
    cc = []  # circle-circle: (i, j, gap)
    wc = []  # wall: (i, wall_type, gap)

    for i in range(n):
        gaps = [
            (x[i] - r[i], 'L'),
            (1 - x[i] - r[i], 'R'),
            (y[i] - r[i], 'B'),
            (1 - y[i] - r[i], 'T'),
        ]
        for gap, wtype in gaps:
            if abs(gap) < tol:
                wc.append((i, wtype, gap))

    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            gap = dist - (r[i] + r[j])
            if abs(gap) < tol:
                cc.append((i, j, gap))

    return cc, wc


def optimize_packing(x0, y0, r0, max_iter=1000):
    """Optimize circle packing with SLSQP."""
    n = len(x0)

    def pack(x, y, r):
        return np.concatenate([x, y, r])

    def unpack(v):
        return v[:n], v[n:2*n], v[2*n:3*n]

    def objective(v):
        return -np.sum(v[2*n:3*n])

    def obj_jac(v):
        g = np.zeros(3*n)
        g[2*n:3*n] = -1.0
        return g

    constraints = []
    # Containment
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[i] - v[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[n+i] - v[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[n+i] - v[2*n+i]})

    # Non-overlap
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j: (
                    (v[i]-v[j])**2 + (v[n+i]-v[n+j])**2 - (v[2*n+i]+v[2*n+j])**2
                )
            })

    bounds = [(0.001, 0.999)]*n + [(0.001, 0.999)]*n + [(0.001, 0.5)]*n
    v0 = pack(x0, y0, r0)

    result = minimize(
        objective, v0, method='SLSQP',
        jac=obj_jac,
        constraints=constraints, bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-15, 'disp': False}
    )

    x, y, r = unpack(result.x)
    metric = np.sum(r)
    feasible = is_feasible(x, y, r, tol=1e-8)

    return x, y, r, metric, feasible


def break_contacts_and_optimize(x, y, r, contacts_to_break, push_strength=0.01):
    """
    Break specific contacts by pushing circles apart, then re-optimize.

    contacts_to_break: list of (i, j) circle pairs to separate
    """
    n = len(x)
    x2, y2, r2 = x.copy(), y.copy(), r.copy()

    for i, j in contacts_to_break:
        # Push circles i and j apart
        dx = x2[i] - x2[j]
        dy = y2[i] - y2[j]
        dist = np.sqrt(dx**2 + dy**2)
        if dist > 0:
            # Shrink both radii slightly to create gap
            r2[i] *= (1 - push_strength)
            r2[j] *= (1 - push_strength)

    # Clamp
    r2 = np.clip(r2, 0.005, 0.49)
    x2 = np.clip(x2, r2 + 0.001, 1 - r2 - 0.001)
    y2 = np.clip(y2, r2 + 0.001, 1 - r2 - 0.001)

    return optimize_packing(x2, y2, r2)


def move_circle_and_optimize(x, y, r, circle_idx, new_x, new_y, new_r=None):
    """Move a specific circle to a new position, then re-optimize."""
    x2, y2, r2 = x.copy(), y.copy(), r.copy()
    x2[circle_idx] = new_x
    y2[circle_idx] = new_y
    if new_r is not None:
        r2[circle_idx] = new_r
    else:
        r2[circle_idx] = min(r2[circle_idx], new_x, 1-new_x, new_y, 1-new_y) * 0.5

    # Resolve overlaps by shrinking
    for _ in range(10):
        overlaps = False
        for j in range(len(x2)):
            if j == circle_idx:
                continue
            dist = np.sqrt((x2[circle_idx]-x2[j])**2 + (y2[circle_idx]-y2[j])**2)
            if dist < r2[circle_idx] + r2[j]:
                r2[circle_idx] *= 0.9
                overlaps = True
        if not overlaps:
            break

    r2 = np.clip(r2, 0.005, 0.49)
    x2 = np.clip(x2, r2 + 0.001, 1 - r2 - 0.001)
    y2 = np.clip(y2, r2 + 0.001, 1 - r2 - 0.001)

    return optimize_packing(x2, y2, r2)


def topology_fingerprint(x, y, r, tol=1e-6):
    """Create a canonical fingerprint of the contact topology."""
    cc, wc = extract_contacts(x, y, r, tol)
    # Sort by indices for canonical form
    cc_set = frozenset((min(i,j), max(i,j)) for i, j, _ in cc)
    wc_set = frozenset((i, w) for i, w, _ in wc)
    return (cc_set, wc_set)


def main():
    t0 = time.time()
    rng = np.random.RandomState(SEED)

    # Load known best
    known_path = os.path.join(WORKDIR, '..', 'topo-001', 'solution_n26.json')
    xk, yk, rk = load_solution(known_path)
    known_metric = np.sum(rk)

    cc, wc = extract_contacts(xk, yk, rk)
    print(f"Known best: metric={known_metric:.10f}")
    print(f"Contacts: {len(cc)} circle-circle, {len(wc)} wall")

    # Show the contact graph
    print("\nCircle-circle contacts:")
    for i, j, gap in sorted(cc):
        print(f"  ({i:2d}, {j:2d})  gap={gap:.2e}")

    print("\nWall contacts:")
    for i, wall, gap in sorted(wc):
        print(f"  circle {i:2d} -> {wall}  gap={gap:.2e}")

    known_fp = topology_fingerprint(xk, yk, rk)

    best_metric = known_metric
    best_sol = (xk.copy(), yk.copy(), rk.copy())
    seen_topologies = {known_fp: known_metric}
    n_same_topo = 0
    n_new_topo = 0

    # ===== Strategy 1: Break random subsets of contacts =====
    print("\n" + "="*60)
    print("STRATEGY 1: Break random contact subsets")
    print("="*60)

    for attempt in range(200):
        # Break 2-6 random contacts
        n_break = rng.randint(2, 7)
        to_break = [cc[k][:2] for k in rng.choice(len(cc), min(n_break, len(cc)), replace=False)]

        # Try different push strengths
        for strength in [0.005, 0.01, 0.02, 0.05]:
            x2, y2, r2, metric, feasible = break_contacts_and_optimize(
                xk, yk, rk, to_break, push_strength=strength
            )

            if feasible and metric > 2.5:
                fp = topology_fingerprint(x2, y2, r2)
                if fp not in seen_topologies:
                    seen_topologies[fp] = metric
                    n_new_topo += 1
                    if metric > best_metric:
                        best_metric = metric
                        best_sol = (x2.copy(), y2.copy(), r2.copy())
                        print(f"  [{attempt+1}] NEW BEST: {metric:.10f} "
                              f"(broke {n_break} contacts, strength={strength})")
                    elif metric > known_metric - 0.01:
                        cc2, wc2 = extract_contacts(x2, y2, r2)
                        print(f"  [{attempt+1}] New topology: {metric:.10f} "
                              f"(cc={len(cc2)}, wc={len(wc2)})")
                else:
                    n_same_topo += 1

                break  # Found a solution, move to next attempt

        if (attempt + 1) % 50 == 0:
            print(f"  Progress: {attempt+1}/200, new topologies: {n_new_topo}, "
                  f"same: {n_same_topo}, best: {best_metric:.10f}")

    # ===== Strategy 2: Move individual circles to new positions =====
    print("\n" + "="*60)
    print("STRATEGY 2: Relocate circles")
    print("="*60)

    # Sort circles by radius (smallest first - easiest to move)
    sorted_by_r = np.argsort(rk)

    for ci_idx in range(min(15, N)):
        circle = sorted_by_r[ci_idx]

        # Try placing this circle in various gaps
        for trial in range(20):
            # Random new position
            new_x = rng.uniform(0.05, 0.95)
            new_y = rng.uniform(0.05, 0.95)

            x2, y2, r2, metric, feasible = move_circle_and_optimize(
                xk, yk, rk, circle, new_x, new_y
            )

            if feasible and metric > 2.5:
                fp = topology_fingerprint(x2, y2, r2)
                if fp not in seen_topologies:
                    seen_topologies[fp] = metric
                    n_new_topo += 1
                    if metric > best_metric:
                        best_metric = metric
                        best_sol = (x2.copy(), y2.copy(), r2.copy())
                        print(f"  [circle {circle}, trial {trial}] NEW BEST: {metric:.10f}")

        if (ci_idx + 1) % 5 == 0:
            print(f"  Tried {ci_idx+1} circles, best: {best_metric:.10f}")

    # ===== Strategy 3: Swap circle positions =====
    print("\n" + "="*60)
    print("STRATEGY 3: Swap circle positions")
    print("="*60)

    for attempt in range(100):
        # Swap 2-4 circles
        n_swap = rng.randint(2, 5)
        indices = rng.choice(N, n_swap, replace=False)

        x2, y2, r2 = xk.copy(), yk.copy(), rk.copy()

        # Cycle the positions: circle[indices[0]] goes to position of indices[1], etc.
        positions_x = x2[indices].copy()
        positions_y = y2[indices].copy()
        radii = r2[indices].copy()

        perm = np.roll(np.arange(n_swap), 1)
        for k in range(n_swap):
            x2[indices[k]] = positions_x[perm[k]]
            y2[indices[k]] = positions_y[perm[k]]
            # Keep original radii at positions (don't move with circle)

        # Small perturbation
        x2 += rng.uniform(-0.005, 0.005, N)
        y2 += rng.uniform(-0.005, 0.005, N)
        r2 = np.clip(r2, 0.005, 0.49)
        x2 = np.clip(x2, r2 + 0.001, 1 - r2 - 0.001)
        y2 = np.clip(y2, r2 + 0.001, 1 - r2 - 0.001)

        x2, y2, r2, metric, feasible = optimize_packing(x2, y2, r2)

        if feasible and metric > 2.5:
            fp = topology_fingerprint(x2, y2, r2)
            if fp not in seen_topologies:
                seen_topologies[fp] = metric
                n_new_topo += 1
                if metric > best_metric:
                    best_metric = metric
                    best_sol = (x2.copy(), y2.copy(), r2.copy())
                    print(f"  [attempt {attempt+1}] NEW BEST: {metric:.10f} "
                          f"(swapped {indices})")
                elif metric > known_metric - 0.01:
                    print(f"  [attempt {attempt+1}] New topology: {metric:.10f}")

        if (attempt + 1) % 25 == 0:
            print(f"  Progress: {attempt+1}/100, best: {best_metric:.10f}")

    # ===== Strategy 4: Large perturbation basin hopping =====
    print("\n" + "="*60)
    print("STRATEGY 4: Large perturbation basin hopping")
    print("="*60)

    for attempt in range(100):
        # Large perturbation to escape basin
        strength = rng.uniform(0.03, 0.15)

        x2 = xk.copy() + rng.uniform(-strength, strength, N)
        y2 = yk.copy() + rng.uniform(-strength, strength, N)
        r2 = rk.copy() * rng.uniform(1 - strength*3, 1 + strength*3, N)

        r2 = np.clip(r2, 0.005, 0.49)
        x2 = np.clip(x2, r2 + 0.001, 1 - r2 - 0.001)
        y2 = np.clip(y2, r2 + 0.001, 1 - r2 - 0.001)

        x2, y2, r2, metric, feasible = optimize_packing(x2, y2, r2)

        if feasible and metric > 2.5:
            fp = topology_fingerprint(x2, y2, r2)
            if fp not in seen_topologies:
                seen_topologies[fp] = metric
                n_new_topo += 1
                if metric > best_metric:
                    best_metric = metric
                    best_sol = (x2.copy(), y2.copy(), r2.copy())
                    print(f"  [attempt {attempt+1}] NEW BEST: {metric:.10f} "
                          f"(strength={strength:.3f})")

        if (attempt + 1) % 25 == 0:
            print(f"  Progress: {attempt+1}/100, best: {best_metric:.10f}")

    # ===== Summary =====
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print(f"Total unique topologies found: {len(seen_topologies)}")
    print(f"Best metric: {best_metric:.10f}")
    print(f"Known best:  {known_metric:.10f}")
    print(f"Improvement: {best_metric - known_metric:.2e}")

    # Top 10 topologies
    sorted_topos = sorted(seen_topologies.items(), key=lambda x: -x[1])
    print(f"\nTop 10 topologies:")
    for i, (fp, metric) in enumerate(sorted_topos[:10]):
        cc_set, wc_set = fp
        print(f"  {i+1}. metric={metric:.10f} (cc={len(cc_set)}, wc={len(wc_set)})")

    # Save best solution
    sol_path = os.path.join(WORKDIR, 'solution_n26.json')
    save_solution(*best_sol, sol_path)

    elapsed = time.time() - t0
    print(f"\nTime: {elapsed:.1f}s")

    return best_metric


if __name__ == '__main__':
    main()
