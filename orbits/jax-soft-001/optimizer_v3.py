"""
V3: Constructive + Destructive Search with JAX refinement

Fundamentally different approach:
1. Constructive: Build packing by placing circles one at a time in largest gaps
2. Try ALL possible orderings of circle sizes
3. After construction, refine with JAX soft-body + SLSQP
4. Destructive: From known solution, remove circles and rebuild differently

Key insight: The ordering in which circles are placed determines the topology.
By varying the construction order and size assignment, we explore topologies
that gradient methods cannot reach.
"""

import json
import math
import os
import sys
import time
from pathlib import Path
from itertools import permutations

import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Voronoi

jax.config.update("jax_enable_x64", True)

N = 26


def make_energy_fn(n):
    @jax.jit
    def energy(params, beta, alpha):
        x = params[:n]
        y = params[n:2*n]
        r = params[2*n:]
        E_radii = -alpha * jnp.sum(r)
        wall_left = jnp.maximum(0.0, r - x)
        wall_right = jnp.maximum(0.0, x + r - 1.0)
        wall_bottom = jnp.maximum(0.0, r - y)
        wall_top = jnp.maximum(0.0, y + r - 1.0)
        E_wall = beta * jnp.sum(wall_left**2 + wall_right**2 + wall_bottom**2 + wall_top**2)
        E_pos = beta * jnp.sum(jnp.maximum(0.0, -r)**2)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = jnp.sqrt(dx**2 + dy**2 + 1e-30)
        sum_r = r[:, None] + r[None, :]
        overlap = jnp.maximum(0.0, sum_r - dist)
        mask = jnp.triu(jnp.ones((n, n), dtype=jnp.float64), k=1)
        E_overlap = beta * jnp.sum((overlap * mask)**2)
        return E_radii + E_wall + E_overlap + E_pos
    return energy


energy_26 = make_energy_fn(26)
vg_26 = jax.jit(jax.value_and_grad(energy_26))


def max_radius_at_point(px, py, circles):
    """Maximum radius for a circle at (px, py) given existing circles."""
    max_r = min(px, 1.0 - px, py, 1.0 - py)  # wall constraint
    for (cx, cy, cr) in circles:
        dist = math.sqrt((px - cx)**2 + (py - cy)**2)
        max_r = min(max_r, dist - cr)
    return max(max_r, 0.0)


def find_best_placement(circles, grid_res=50):
    """Find the position that allows the largest new circle."""
    best_r = 0.0
    best_pos = (0.5, 0.5)

    for gx in np.linspace(0.02, 0.98, grid_res):
        for gy in np.linspace(0.02, 0.98, grid_res):
            r = max_radius_at_point(gx, gy, circles)
            if r > best_r:
                best_r = r
                best_pos = (gx, gy)

    # Refine with scipy
    from scipy.optimize import minimize_scalar, minimize as scipy_min

    def neg_radius(p):
        return -max_radius_at_point(p[0], p[1], circles)

    result = scipy_min(neg_radius, best_pos, method='Nelder-Mead',
                       options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 1000})
    if result.success:
        px, py = result.x
        r = max_radius_at_point(px, py, circles)
        if r > best_r:
            best_r = r
            best_pos = (px, py)

    return best_pos, best_r


def construct_packing_greedy(n=N, grid_res=60):
    """Greedy constructive: place circles one at a time in the largest gap."""
    circles = []
    for i in range(n):
        pos, r = find_best_placement(circles, grid_res=grid_res)
        if r < 0.001:
            # Fallback: place tiny circle
            r = 0.005
        circles.append((pos[0], pos[1], r))
    return circles


def construct_packing_sized(target_radii, grid_res=50):
    """Place circles in order of decreasing target radius.

    For each circle, find the position that maximizes its radius
    (but cap at target_radius to leave room for others).
    """
    circles = []
    # Sort by decreasing radius
    order = np.argsort(target_radii)[::-1]

    for idx in order:
        target_r = target_radii[idx]
        pos, max_r = find_best_placement(circles, grid_res=grid_res)

        # Use min of target and available
        r = min(target_r, max_r * 0.95)  # leave 5% margin
        r = max(r, 0.005)
        circles.append((pos[0], pos[1], r))

    # Reorder to match original indices
    reordered = [None] * len(target_radii)
    for i, idx in enumerate(order):
        reordered[idx] = circles[i]

    return reordered


def circles_to_params(circles):
    x = np.array([c[0] for c in circles])
    y = np.array([c[1] for c in circles])
    r = np.array([c[2] for c in circles])
    return np.concatenate([x, y, r])


def optimize_adam(params, total_steps=3000, lr_init=5e-3, lr_final=1e-6):
    """Quick JAX optimization."""
    params = jnp.array(params, dtype=jnp.float64)
    n = len(params) // 3

    schedule = optax.cosine_decay_schedule(lr_init, total_steps, alpha=lr_final/lr_init)
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(params)

    best_metric = -1e10
    best_params = params

    for step in range(total_steps):
        t = step / total_steps
        # Slow annealing
        if t < 0.6:
            beta = 10.0 ** (t * 2.0)
        else:
            beta = 10.0 ** (1.2 + (t - 0.6) * 17.0)

        val, grad = vg_26(params, beta, 1.0)
        grad_norm = jnp.sqrt(jnp.sum(grad**2))
        grad = jnp.where(grad_norm > 10.0, grad * 10.0 / grad_norm, grad)

        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)

        if step % 500 == 0 or step == total_steps - 1:
            r = params[2*N:]
            x = params[:N]
            y = params[N:2*N]
            metric = float(jnp.sum(r))

            wall_viol = float(jnp.max(jnp.array([
                jnp.max(r - x), jnp.max(x + r - 1.0),
                jnp.max(r - y), jnp.max(y + r - 1.0)
            ])))

            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = jnp.sqrt(dx**2 + dy**2 + 1e-30)
            sum_r_mat = r[:, None] + r[None, :]
            mask_mat = jnp.triu(jnp.ones((N, N), dtype=jnp.float64), k=1)
            overlap = (sum_r_mat - dist) * mask_mat
            max_overlap = float(jnp.max(overlap))
            max_viol = max(wall_viol, max_overlap)

            if max_viol < 1e-6 and metric > best_metric:
                best_metric = metric
                best_params = params.copy()

    return best_params, best_metric


def polish_slsqp(params_np, ftol=1e-15, maxiter=20000):
    params_np = np.array(params_np, dtype=np.float64)
    n = N

    def objective(p):
        return -np.sum(p[2*n:])

    def obj_jac(p):
        g = np.zeros_like(p)
        g[2*n:] = -1.0
        return g

    constraints = []
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[i] - p[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1.0 - p[i] - p[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[n+i] - p[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1.0 - p[n+i] - p[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[2*n+i] - 1e-10})

    for i in range(n):
        for j in range(i+1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: math.sqrt(
                    (p[i]-p[j])**2 + (p[n+i]-p[n+j])**2
                ) - p[2*n+i] - p[2*n+j]
            })

    result = minimize(
        objective, params_np, jac=obj_jac,
        method='SLSQP', constraints=constraints,
        options={'maxiter': maxiter, 'ftol': ftol, 'disp': False}
    )
    return result.x, -result.fun


def check_feasibility(params, tol=1e-10):
    x, y, r = params[:N], params[N:2*N], params[2*N:]
    max_viol = 0.0
    for i in range(N):
        max_viol = max(max_viol, r[i] - x[i], x[i] + r[i] - 1.0, r[i] - y[i], y[i] + r[i] - 1.0)
    for i in range(N):
        for j in range(i+1, N):
            dist = math.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            max_viol = max(max_viol, r[i] + r[j] - dist)
    return max_viol


def load_solution(path):
    with open(path) as f:
        data = json.load(f)
    circles = data["circles"] if "circles" in data else data
    x = np.array([c[0] for c in circles])
    y = np.array([c[1] for c in circles])
    r = np.array([c[2] for c in circles])
    return np.concatenate([x, y, r])


def save_solution(params, path):
    circles = [[float(params[i]), float(params[N+i]), float(params[2*N+i])] for i in range(N)]
    with open(path, 'w') as f:
        json.dump({"circles": circles}, f, indent=2)


def main():
    print("=" * 60)
    print("V3: Constructive + Destructive Topology Search")
    print("=" * 60)

    workdir = Path(__file__).parent
    best_known_path = workdir.parent / "topo-001" / "solution_n26.json"
    base_params = load_solution(str(best_known_path))
    base_metric = float(np.sum(base_params[2*N:]))
    print(f"Base metric: {base_metric:.10f}")

    # Get the radius distribution from best known
    known_radii = base_params[2*N:].copy()
    known_radii_sorted = np.sort(known_radii)[::-1]

    overall_best = 0.0
    overall_best_params = None
    all_results = []

    # ---- Strategy 1: Pure greedy construction ----
    print("\n--- Strategy 1: Greedy construction ---")
    for grid_res in [40, 60, 80, 100]:
        circles = construct_packing_greedy(N, grid_res=grid_res)
        params = circles_to_params(circles)
        metric_raw = float(np.sum(params[2*N:]))
        print(f"  Greedy (grid={grid_res}): raw metric = {metric_raw:.10f}")

        # Refine with JAX
        best_params, best_metric = optimize_adam(params, total_steps=4000)
        if best_metric > 2.0:
            polished, pol_metric = polish_slsqp(np.array(best_params))
            viol = check_feasibility(polished)
            if viol < 1e-10 and pol_metric > best_metric:
                best_metric = pol_metric
                best_params = polished

        tag = f"greedy_grid{grid_res}"
        all_results.append((tag, best_metric))
        if best_metric > overall_best:
            overall_best = best_metric
            overall_best_params = best_params
            print(f"    -> Refined: {best_metric:.10f} NEW BEST")
        else:
            print(f"    -> Refined: {best_metric:.10f}")

    # ---- Strategy 2: Sized construction with shuffled radii ----
    print("\n--- Strategy 2: Sized construction ---")
    for trial in range(30):
        rng = np.random.RandomState(30000 + trial)

        # Shuffle the known radius distribution
        radii = known_radii.copy()
        rng.shuffle(radii)

        # Also try some noise
        radii = radii * (1.0 + rng.normal(0, 0.1, N))
        radii = np.clip(radii, 0.02, 0.2)

        circles = construct_packing_sized(radii, grid_res=50)
        params = circles_to_params(circles)

        # Refine
        best_params, best_metric = optimize_adam(params, total_steps=3000)
        if best_metric > 2.0:
            polished, pol_metric = polish_slsqp(np.array(best_params))
            viol = check_feasibility(polished)
            if viol < 1e-10 and pol_metric > best_metric:
                best_metric = pol_metric
                best_params = polished

        tag = f"sized_{trial}"
        all_results.append((tag, best_metric))
        if best_metric > overall_best:
            overall_best = best_metric
            overall_best_params = best_params
            print(f"  [{tag}] {best_metric:.10f} NEW BEST")
        elif trial % 10 == 0:
            print(f"  [{tag}] {best_metric:.10f}")

    # ---- Strategy 3: Destructive rebuild ----
    print("\n--- Strategy 3: Destructive rebuild ---")
    for trial in range(30):
        rng = np.random.RandomState(40000 + trial)

        # Start from known solution
        x, y, r = base_params[:N].copy(), base_params[N:2*N].copy(), base_params[2*N:].copy()
        circles_list = [(x[i], y[i], r[i]) for i in range(N)]

        # Remove 3-6 random circles
        n_remove = rng.randint(3, 7)
        remove_idx = rng.choice(N, size=n_remove, replace=False)
        remaining = [circles_list[i] for i in range(N) if i not in remove_idx]
        removed_radii = [r[i] for i in remove_idx]

        # Reinsert removed circles greedily
        for rad in sorted(removed_radii, reverse=True):
            pos, max_r = find_best_placement(remaining, grid_res=50)
            new_r = min(rad * 1.2, max_r * 0.9)  # try slightly larger
            new_r = max(new_r, 0.005)
            remaining.append((pos[0], pos[1], new_r))

        params = circles_to_params(remaining)

        # Refine
        best_params, best_metric = optimize_adam(params, total_steps=4000)
        if best_metric > 2.0:
            polished, pol_metric = polish_slsqp(np.array(best_params))
            viol = check_feasibility(polished)
            if viol < 1e-10 and pol_metric > best_metric:
                best_metric = pol_metric
                best_params = polished

        tag = f"destruct_{trial}_rm{n_remove}"
        all_results.append((tag, best_metric))
        if best_metric > overall_best:
            overall_best = best_metric
            overall_best_params = best_params
            print(f"  [{tag}] {best_metric:.10f} NEW BEST")
        elif trial % 10 == 0:
            print(f"  [{tag}] {best_metric:.10f}")

    # ---- Strategy 4: Symmetric constructions ----
    print("\n--- Strategy 4: Symmetric templates ---")
    templates = []

    # Template A: 4-fold symmetric (6 + 4 corners + 1 center + 15 more)
    def make_4fold(rng):
        circles = []
        # 4 corner circles
        cr = rng.uniform(0.06, 0.12)
        for cx, cy in [(cr, cr), (1-cr, cr), (cr, 1-cr), (1-cr, 1-cr)]:
            circles.append((cx, cy, cr))
        # Center
        circles.append((0.5, 0.5, rng.uniform(0.1, 0.15)))
        # 4 edge-center circles
        er = rng.uniform(0.08, 0.13)
        for cx, cy in [(0.5, er), (0.5, 1-er), (er, 0.5), (1-er, 0.5)]:
            circles.append((cx, cy, er))
        # 8 circles at 45-degree positions
        ring_r = rng.uniform(0.22, 0.35)
        for i in range(8):
            angle = math.pi/4 + 2*math.pi*i/8
            cx = 0.5 + ring_r * math.cos(angle)
            cy = 0.5 + ring_r * math.sin(angle)
            circles.append((cx, cy, rng.uniform(0.04, 0.09)))
        # Fill remaining with random
        while len(circles) < N:
            cx = rng.uniform(0.1, 0.9)
            cy = rng.uniform(0.1, 0.9)
            r = max_radius_at_point(cx, cy, circles)
            r = max(r * 0.7, 0.01)
            circles.append((cx, cy, r))
        return circles[:N]

    # Template B: Hexagonal close-packed
    def make_hex_pack(rng, r_base=None):
        if r_base is None:
            r_base = rng.uniform(0.07, 0.11)
        circles = []
        dx = 2 * r_base
        dy = r_base * math.sqrt(3)
        for row in range(20):
            for col in range(20):
                cx = r_base + col * dx + (r_base if row % 2 else 0)
                cy = r_base + row * dy
                if r_base < cx < 1-r_base and r_base < cy < 1-r_base:
                    circles.append((cx, cy, r_base * (1.0 + rng.normal(0, 0.05))))
        rng.shuffle(circles)
        # Pad if fewer than N
        while len(circles) < N:
            cx = rng.uniform(0.1, 0.9)
            cy = rng.uniform(0.1, 0.9)
            circles.append((cx, cy, 0.03))
        return circles[:N]

    # Template C: Concentric rings with different counts
    def make_rings(rng, ring_sizes=None):
        if ring_sizes is None:
            # Random ring configuration summing to N
            ring_sizes = []
            remaining = N
            ring_r = 0.0
            while remaining > 0:
                if ring_r < 0.01:
                    # Center
                    ring_sizes.append(1)
                    remaining -= 1
                else:
                    count = min(rng.randint(3, 10), remaining)
                    ring_sizes.append(count)
                    remaining -= count
                ring_r += rng.uniform(0.12, 0.22)

        circles = []
        ring_r = 0.0
        for count in ring_sizes:
            if count == 1 and ring_r < 0.01:
                circles.append((0.5, 0.5, rng.uniform(0.1, 0.15)))
            else:
                for i in range(count):
                    angle = 2 * math.pi * i / count + rng.uniform(0, 0.3)
                    cx = 0.5 + ring_r * math.cos(angle)
                    cy = 0.5 + ring_r * math.sin(angle)
                    cx = np.clip(cx, 0.05, 0.95)
                    cy = np.clip(cy, 0.05, 0.95)
                    circles.append((cx, cy, rng.uniform(0.04, 0.1)))
            ring_r += rng.uniform(0.12, 0.22)

        # Pad if fewer than N
        while len(circles) < N:
            cx = rng.uniform(0.1, 0.9)
            cy = rng.uniform(0.1, 0.9)
            circles.append((cx, cy, 0.03))
        return circles[:N]

    for trial in range(30):
        rng = np.random.RandomState(50000 + trial)

        if trial % 3 == 0:
            circles = make_4fold(rng)
            tname = "4fold"
        elif trial % 3 == 1:
            circles = make_hex_pack(rng)
            tname = "hex"
        else:
            # Different ring configurations
            configs = [
                [1, 6, 10, 9],
                [1, 7, 12, 6],
                [1, 5, 8, 12],
                [1, 8, 10, 7],
                [1, 6, 8, 6, 5],
                [1, 9, 12, 4],
                [2, 7, 11, 6],
                [1, 8, 12, 5],
                [1, 6, 12, 7],
                [1, 7, 10, 8],
            ]
            config = configs[trial % len(configs)]
            circles = make_rings(rng, config)
            tname = f"ring{'_'.join(map(str, config))}"

        params = circles_to_params(circles)
        # Add noise
        params += rng.normal(0, 0.01, len(params))
        params[2*N:] = np.maximum(params[2*N:], 0.01)

        best_params, best_metric = optimize_adam(params, total_steps=4000,
                                                  lr_init=1e-2, lr_final=1e-6)
        if best_metric > 2.0:
            polished, pol_metric = polish_slsqp(np.array(best_params))
            viol = check_feasibility(polished)
            if viol < 1e-10 and pol_metric > best_metric:
                best_metric = pol_metric
                best_params = polished

        tag = f"template_{tname}_{trial}"
        all_results.append((tag, best_metric))
        if best_metric > overall_best:
            overall_best = best_metric
            overall_best_params = best_params
            print(f"  [{tag}] {best_metric:.10f} NEW BEST")
        elif trial % 10 == 0:
            print(f"  [{tag}] {best_metric:.10f}")

    # ---- Report ----
    print("\n" + "=" * 60)
    print(f"OVERALL BEST: {overall_best:.10f}")
    print(f"Known best:   {base_metric:.10f}")
    print(f"Improvement:  {overall_best - base_metric:.2e}")
    print("=" * 60)

    if overall_best_params is not None:
        viol = check_feasibility(np.array(overall_best_params))
        if viol < 1e-10:
            save_solution(np.array(overall_best_params), str(workdir / "solution_n26_v3.json"))
            if overall_best > base_metric:
                save_solution(np.array(overall_best_params), str(workdir / "solution_n26.json"))
                print("NEW RECORD! Updated solution_n26.json")

    results_sorted = sorted(all_results, key=lambda x: -x[1])
    with open(str(workdir / "results_v3.json"), 'w') as f:
        json.dump(results_sorted[:50], f, indent=2)

    print("\nTop 10:")
    for tag, m in results_sorted[:10]:
        print(f"  {tag}: {m:.10f}")


if __name__ == "__main__":
    main()
