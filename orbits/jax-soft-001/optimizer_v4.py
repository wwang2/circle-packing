"""
V4: Parallel Tempering + Curriculum Optimization

Focused version: runs strategies sequentially with progress output.
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax
import numpy as np
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)

WORKDIR = Path(__file__).parent
N_TARGET = 26

# Force flush on all prints
import functools
print = functools.partial(print, flush=True)


# ============================================================
# Energy function
# ============================================================

_energy_cache = {}
_vg_cache = {}

def get_energy_and_vg(n):
    if n not in _energy_cache:
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
        _energy_cache[n] = energy
        _vg_cache[n] = jax.jit(jax.value_and_grad(energy))
    return _energy_cache[n], _vg_cache[n]


# ============================================================
# Utilities
# ============================================================

def check_feasibility(params, n, tol=1e-10):
    x, y, r = params[:n], params[n:2*n], params[2*n:]
    max_viol = 0.0
    for i in range(n):
        max_viol = max(max_viol, r[i] - x[i], x[i] + r[i] - 1.0,
                       r[i] - y[i], y[i] + r[i] - 1.0)
    for i in range(n):
        for j in range(i+1, n):
            dist = math.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            max_viol = max(max_viol, r[i] + r[j] - dist)
    return max_viol


def polish_slsqp(params_np, n, ftol=1e-15, maxiter=20000):
    params_np = np.array(params_np, dtype=np.float64)
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
    result = minimize(objective, params_np, jac=obj_jac, method='SLSQP',
                      constraints=constraints,
                      options={'maxiter': maxiter, 'ftol': ftol, 'disp': False})
    return result.x, -result.fun


def load_solution(path):
    with open(path) as f:
        data = json.load(f)
    circles = data["circles"] if "circles" in data else data
    x = np.array([c[0] for c in circles])
    y = np.array([c[1] for c in circles])
    r = np.array([c[2] for c in circles])
    return np.concatenate([x, y, r])


def save_solution(params, path, n):
    circles = [[float(params[i]), float(params[n+i]), float(params[2*n+i])] for i in range(n)]
    with open(path, 'w') as f:
        json.dump({"circles": circles}, f, indent=2)


def max_radius_at_point(px, py, circles):
    max_r = min(px, 1.0 - px, py, 1.0 - py)
    for (cx, cy, cr) in circles:
        dist = math.sqrt((px - cx)**2 + (py - cy)**2)
        max_r = min(max_r, dist - cr)
    return max(max_r, 0.0)


def find_best_placement(circles, grid_res=60):
    best_r = 0.0
    best_pos = (0.5, 0.5)
    for gx in np.linspace(0.02, 0.98, grid_res):
        for gy in np.linspace(0.02, 0.98, grid_res):
            r = max_radius_at_point(gx, gy, circles)
            if r > best_r:
                best_r = r
                best_pos = (gx, gy)
    from scipy.optimize import minimize as scipy_min
    def neg_radius(p):
        return -max_radius_at_point(p[0], p[1], circles)
    result = scipy_min(neg_radius, best_pos, method='Nelder-Mead',
                       options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 1000})
    if result.success:
        r = max_radius_at_point(result.x[0], result.x[1], circles)
        if r > best_r:
            best_r = r
            best_pos = (result.x[0], result.x[1])
    return best_pos, best_r


# ============================================================
# Core optimizer: Adam with annealing
# ============================================================

def optimize_adam(params, n, total_steps=3000, lr_init=5e-3, lr_final=1e-6,
                  beta_schedule='slow', alpha=1.0):
    params = jnp.array(params, dtype=jnp.float64)
    _, vg_fn = get_energy_and_vg(n)

    schedule = optax.cosine_decay_schedule(lr_init, total_steps, alpha=lr_final/lr_init)
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(params)

    best_metric = -1e10
    best_params = params

    for step in range(total_steps):
        t = step / total_steps
        if beta_schedule == 'slow':
            if t < 0.6:
                beta = 10.0 ** (t * 2.0)
            else:
                beta = 10.0 ** (1.2 + (t - 0.6) * 17.0)
        elif beta_schedule == 'ultra_slow':
            if t < 0.8:
                beta = 10.0 ** (t * 1.25)
            else:
                beta = 10.0 ** (1.0 + (t - 0.8) * 35.0)
        elif beta_schedule == 'cyclic':
            base = t * 8.0
            cycle = 2.0 * math.sin(2 * math.pi * t * 3)
            beta = 10.0 ** max(0.0, base + cycle)
        else:
            beta = 10.0 ** (t * 8.0)

        val, grad = vg_fn(params, beta, alpha)
        grad_norm = jnp.sqrt(jnp.sum(grad**2))
        grad = jnp.where(grad_norm > 10.0, grad * 10.0 / grad_norm, grad)
        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)

        if step % 200 == 0 or step == total_steps - 1:
            r = params[2*n:]
            x = params[:n]
            y = params[n:2*n]
            metric = float(jnp.sum(r))
            wall_viol = float(jnp.max(jnp.array([
                jnp.max(r - x), jnp.max(x + r - 1.0),
                jnp.max(r - y), jnp.max(y + r - 1.0)
            ])))
            dx_m = x[:, None] - x[None, :]
            dy_m = y[:, None] - y[None, :]
            dist = jnp.sqrt(dx_m**2 + dy_m**2 + 1e-30)
            sum_r_mat = r[:, None] + r[None, :]
            mask_mat = jnp.triu(jnp.ones((n, n), dtype=jnp.float64), k=1)
            overlap = (sum_r_mat - dist) * mask_mat
            max_overlap = float(jnp.max(overlap))
            max_viol = max(wall_viol, max_overlap)
            if max_viol < 1e-6 and metric > best_metric:
                best_metric = metric
                best_params = params.copy()

    return np.array(best_params), best_metric


# ============================================================
# Strategy 1: Parallel Tempering
# ============================================================

def parallel_tempering(base_params, n=26, n_replicas=8, n_sweeps=40,
                       steps_per_sweep=300, seed=42):
    print(f"  [PT] {n_replicas} replicas, {n_sweeps} sweeps, {steps_per_sweep} steps/sweep")
    rng = np.random.RandomState(seed)
    energy_fn, vg_fn = get_energy_and_vg(n)

    # Beta levels: geometric from soft to hard
    beta_levels = np.logspace(-1, 8, n_replicas)

    # Initialize replicas
    replicas = []
    for i in range(n_replicas):
        noise_scale = 0.12 * (1.0 - i / n_replicas)
        noise = rng.normal(0, noise_scale, len(base_params))
        p = base_params.copy() + noise
        p[2*n:] = np.maximum(p[2*n:], 0.01)
        replicas.append(p)

    best_metric = 0.0
    best_params = None
    n_swaps_accepted = 0
    n_swaps_total = 0

    for sweep in range(n_sweeps):
        t0 = time.time()

        # Optimize each replica at its beta
        for i in range(n_replicas):
            params = jnp.array(replicas[i], dtype=jnp.float64)
            beta = beta_levels[i]
            lr = 1e-2 if beta < 100 else 5e-3 if beta < 1e5 else 1e-3
            schedule = optax.cosine_decay_schedule(lr, steps_per_sweep, alpha=0.01)
            optimizer = optax.adam(schedule)
            opt_state = optimizer.init(params)

            for step in range(steps_per_sweep):
                val, grad = vg_fn(params, beta, 1.0)
                grad_norm = jnp.sqrt(jnp.sum(grad**2))
                grad = jnp.where(grad_norm > 10.0, grad * 10.0 / grad_norm, grad)
                updates, opt_state = optimizer.update(grad, opt_state)
                params = optax.apply_updates(params, updates)

            replicas[i] = np.array(params)

        # Attempt swaps between adjacent replicas
        for i in range(n_replicas - 1):
            n_swaps_total += 1
            p_i = jnp.array(replicas[i], dtype=jnp.float64)
            p_j = jnp.array(replicas[i+1], dtype=jnp.float64)
            beta_i, beta_j = beta_levels[i], beta_levels[i+1]

            E_i_i = float(energy_fn(p_i, beta_i, 1.0))
            E_j_j = float(energy_fn(p_j, beta_j, 1.0))
            E_i_j = float(energy_fn(p_i, beta_j, 1.0))
            E_j_i = float(energy_fn(p_j, beta_i, 1.0))

            delta = (E_i_j + E_j_i) - (E_i_i + E_j_j)
            if delta < 0 or rng.random() < math.exp(-min(delta, 500)):
                replicas[i], replicas[i+1] = replicas[i+1].copy(), replicas[i].copy()
                n_swaps_accepted += 1

        # Radical moves on soft replicas
        for i in range(min(3, n_replicas // 2)):
            if rng.random() < 0.4:
                p = replicas[i].copy()
                move = rng.randint(5)
                if move == 0:
                    # Teleport 2-4 circles
                    idx = rng.choice(n, size=rng.randint(2, 5), replace=False)
                    for k in idx:
                        p[k] = rng.uniform(0.05, 0.95)
                        p[n+k] = rng.uniform(0.05, 0.95)
                elif move == 1:
                    # Swap positions of 3-5 pairs
                    for _ in range(rng.randint(3, 6)):
                        a, b = rng.choice(n, 2, replace=False)
                        p[a], p[b] = p[b], p[a]
                        p[n+a], p[n+b] = p[n+b], p[n+a]
                elif move == 2:
                    # Shuffle radii
                    radii = p[2*n:].copy()
                    rng.shuffle(radii)
                    p[2*n:] = radii
                elif move == 3:
                    # Reflect subset
                    idx = rng.choice(n, size=n//2, replace=False)
                    if rng.random() < 0.5:
                        p[idx] = 1.0 - p[idx]
                    else:
                        p[n+idx] = 1.0 - p[n+idx]
                elif move == 4:
                    # Redistribute radii
                    total = np.sum(p[2*n:])
                    new_r = rng.dirichlet(np.ones(n) * 0.5) * total
                    p[2*n:] = np.clip(new_r, 0.01, 0.25)
                p[2*n:] = np.maximum(p[2*n:], 0.01)
                replicas[i] = p

        # Check hard replica
        p_hard = replicas[-1]
        metric = float(np.sum(p_hard[2*n:]))
        viol = check_feasibility(p_hard, n)

        if viol < 1e-6 and metric > best_metric:
            best_metric = metric
            best_params = p_hard.copy()

        elapsed = time.time() - t0
        if sweep % 5 == 0:
            swap_rate = n_swaps_accepted / max(1, n_swaps_total)
            print(f"    sweep {sweep}/{n_sweeps}: metric={metric:.8f} viol={viol:.2e} "
                  f"swaps={swap_rate:.2f} ({elapsed:.1f}s)")

    # Polish all replicas
    print(f"  [PT] Polishing replicas...")
    for i in range(n_replicas):
        p = replicas[i]
        raw_m = float(np.sum(p[2*n:]))
        if raw_m > 2.5:
            pol_p, pol_m = optimize_adam(p, n, total_steps=2000, beta_schedule='slow')
            if pol_m > 2.0:
                pol_p2, pol_m2 = polish_slsqp(pol_p, n)
                viol = check_feasibility(pol_p2, n)
                if viol < 1e-10 and pol_m2 > best_metric:
                    best_metric = pol_m2
                    best_params = pol_p2
                    print(f"    Replica {i}: {pol_m2:.10f} NEW BEST")

    return best_params, best_metric


# ============================================================
# Strategy 2: Curriculum growth
# ============================================================

def curriculum_grow(start_n=20, target_n=26, seed=42):
    print(f"  [CUR] Growing from n={start_n} to n={target_n}")
    rng = np.random.RandomState(seed)
    n = start_n

    # Initialize
    circles = []
    circles.append((0.5, 0.5, 0.15))
    for i in range(min(7, n - 1)):
        angle = 2 * math.pi * i / 7
        circles.append((0.5 + 0.28 * math.cos(angle),
                        0.5 + 0.28 * math.sin(angle), 0.11))
    for i in range(max(0, n - 8)):
        angle = 2 * math.pi * i / max(1, n - 8) + math.pi/12
        circles.append((0.5 + 0.42 * math.cos(angle),
                        0.5 + 0.42 * math.sin(angle), 0.07))
    while len(circles) < n:
        circles.append((rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9), 0.04))
    circles = circles[:n]

    x = np.array([c[0] for c in circles])
    y = np.array([c[1] for c in circles])
    r = np.array([c[2] for c in circles])
    params = np.concatenate([x, y, r])

    # Optimize at start_n
    params, metric = optimize_adam(params, n, total_steps=3000, beta_schedule='slow')
    if metric > 0:
        params, metric = polish_slsqp(params, n)
    print(f"    n={n}: metric={metric:.10f}")

    while n < target_n:
        n_new = n + 1
        circles_list = [(float(params[i]), float(params[n+i]), float(params[2*n+i]))
                        for i in range(n)]

        best_m = 0.0
        best_p = None

        # Try 3 insertion strategies
        for strategy in range(3):
            if strategy == 0:
                pos, gap_r = find_best_placement(circles_list, grid_res=70)
                new_r = gap_r * 0.85
            elif strategy == 1:
                # Random position
                pos = (rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9))
                new_r = max_radius_at_point(pos[0], pos[1], circles_list) * 0.7
            else:
                # Near smallest existing circle
                radii = np.array([c[2] for c in circles_list])
                smallest_idx = np.argmin(radii)
                cx, cy = circles_list[smallest_idx][0], circles_list[smallest_idx][1]
                pos = (cx + rng.normal(0, 0.1), cy + rng.normal(0, 0.1))
                pos = (np.clip(pos[0], 0.05, 0.95), np.clip(pos[1], 0.05, 0.95))
                new_r = max_radius_at_point(pos[0], pos[1], circles_list) * 0.7

            new_r = max(new_r, 0.005)
            x_new = np.append(params[:n], pos[0])
            y_new = np.append(params[n:2*n], pos[1])
            r_new = np.append(params[2*n:], new_r)
            params_new = np.concatenate([x_new, y_new, r_new])

            opt_p, opt_m = optimize_adam(params_new, n_new, total_steps=3000,
                                         lr_init=1e-2, beta_schedule='slow')
            if opt_m > 2.0:
                pol_p, pol_m = polish_slsqp(opt_p, n_new)
                viol = check_feasibility(pol_p, n_new)
                if viol < 1e-10 and pol_m > best_m:
                    best_m = pol_m
                    best_p = pol_p

        if best_p is not None:
            params = best_p
            metric = best_m
        else:
            pos, gap_r = find_best_placement(circles_list, grid_res=100)
            x_new = np.append(params[:n], pos[0])
            y_new = np.append(params[n:2*n], pos[1])
            r_new = np.append(params[2*n:], max(gap_r * 0.5, 0.005))
            params = np.concatenate([x_new, y_new, r_new])
            params, metric = optimize_adam(params, n_new, total_steps=5000,
                                           beta_schedule='ultra_slow')
            if metric > 0:
                params, metric = polish_slsqp(params, n_new)

        n = n_new
        print(f"    n={n}: metric={metric:.10f}")

    return params, metric


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("V4: Parallel Tempering + Curriculum")
    print("=" * 60)

    best_known_path = WORKDIR / "solution_n26.json"
    if not best_known_path.exists():
        best_known_path = WORKDIR.parent / "topo-001" / "solution_n26.json"

    base_params = load_solution(str(best_known_path))
    base_metric = float(np.sum(base_params[2*N_TARGET:]))
    print(f"Base metric: {base_metric:.10f}")

    overall_best_metric = base_metric
    overall_best_params = base_params.copy()
    all_results = []

    # ---- Parallel Tempering (3 trials) ----
    print("\n--- PARALLEL TEMPERING ---")
    for trial in range(3):
        seed = 100000 + trial * 777
        rng = np.random.RandomState(seed)

        if trial == 0:
            init = base_params.copy()
        elif trial == 1:
            init = base_params.copy()
            init[:N_TARGET] = 1.0 - init[:N_TARGET]
        else:
            init = base_params + rng.normal(0, 0.08, len(base_params))
            init[2*N_TARGET:] = np.maximum(init[2*N_TARGET:], 0.01)

        pt_p, pt_m = parallel_tempering(init, n=N_TARGET, n_replicas=6,
                                         n_sweeps=30, steps_per_sweep=250, seed=seed)
        tag = f"PT_{trial}"
        if pt_p is not None:
            all_results.append((tag, pt_m))
            if pt_m > overall_best_metric:
                overall_best_metric = pt_m
                overall_best_params = pt_p
                print(f"  [{tag}] NEW BEST: {pt_m:.10f}")
            else:
                print(f"  [{tag}] best={pt_m:.10f}")
        else:
            all_results.append((tag, 0.0))
            print(f"  [{tag}] failed")

    # ---- Curriculum (4 start sizes x 2 trials) ----
    print("\n--- CURRICULUM GROWTH ---")
    for start_n in [20, 22, 23, 24]:
        for trial in range(2):
            seed = 200000 + start_n * 100 + trial
            cur_p, cur_m = curriculum_grow(start_n=start_n, target_n=N_TARGET, seed=seed)

            tag = f"CUR_n{start_n}_t{trial}"
            if cur_p is not None:
                viol = check_feasibility(cur_p, N_TARGET)
                if viol < 1e-10:
                    all_results.append((tag, cur_m))
                    if cur_m > overall_best_metric:
                        overall_best_metric = cur_m
                        overall_best_params = cur_p
                        print(f"  [{tag}] NEW BEST: {cur_m:.10f}")
                    else:
                        print(f"  [{tag}] metric={cur_m:.10f}")
                else:
                    all_results.append((tag, 0.0))
                    print(f"  [{tag}] infeasible (viol={viol:.2e})")
            else:
                all_results.append((tag, 0.0))
                print(f"  [{tag}] failed")

    # ---- Report ----
    print("\n" + "=" * 60)
    print(f"OVERALL BEST: {overall_best_metric:.10f}")
    print(f"Base metric:  {base_metric:.10f}")
    print(f"Improvement:  {overall_best_metric - base_metric:.2e}")
    print("=" * 60)

    if overall_best_params is not None:
        viol = check_feasibility(overall_best_params, N_TARGET)
        if viol < 1e-10:
            save_solution(overall_best_params, str(WORKDIR / "solution_n26_v4.json"), N_TARGET)
            print(f"Saved solution_n26_v4.json (viol={viol:.2e})")
            if overall_best_metric > base_metric:
                save_solution(overall_best_params, str(WORKDIR / "solution_n26.json"), N_TARGET)
                print("NEW RECORD! Updated solution_n26.json")

    results_sorted = sorted(all_results, key=lambda x: -x[1])
    with open(str(WORKDIR / "results_v4.json"), 'w') as f:
        json.dump(results_sorted, f, indent=2)

    print("\nAll results:")
    for tag, m in results_sorted:
        print(f"  {tag}: {m:.10f}")

    return overall_best_metric, overall_best_params


if __name__ == "__main__":
    main()
