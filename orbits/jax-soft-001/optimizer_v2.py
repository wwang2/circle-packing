"""
V2 JAX Soft-Body Optimizer: Aggressive Topology Exploration

Key changes from V1:
1. Remove-and-reinsert: temporarily remove 2-5 circles, re-optimize, reinsert
2. Size redistribution: shuffle radii assignments
3. Reflection/rotation transforms of subsets
4. Much longer soft phases (low beta)
5. Simulated annealing acceptance criterion for topology jumps
"""

import json
import math
import os
import sys
import time
from pathlib import Path
from functools import partial
from itertools import combinations

import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax
import numpy as np
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)

N = 26


# ============================================================
# Energy (same as V1 but parameterized for variable N)
# ============================================================

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


def optimize_adam(params, energy_fn, vg_fn, n, total_steps=3000,
                  lr_init=5e-3, lr_final=1e-6, beta_schedule='standard', alpha=1.0):
    """Optimize with Adam and annealing."""
    params = jnp.array(params, dtype=jnp.float64)

    schedule = optax.cosine_decay_schedule(lr_init, total_steps, alpha=lr_final/lr_init)
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(params)

    best_metric = -1e10
    best_params = params

    for step in range(total_steps):
        t = step / total_steps

        if beta_schedule == 'standard':
            beta = 10.0 ** (t * 8.0)
        elif beta_schedule == 'slow':
            if t < 0.6:
                beta = 10.0 ** (t * 2.0)
            else:
                beta = 10.0 ** (1.2 + (t - 0.6) * 17.0)
        elif beta_schedule == 'cyclic':
            base = t * 8.0
            cycle = 2.0 * math.sin(2 * math.pi * t * 3)
            beta = 10.0 ** max(0.0, base + cycle)
        elif beta_schedule == 'ultra_slow':
            # Stay soft for 80% of steps
            if t < 0.8:
                beta = 10.0 ** (t * 1.25)  # 1 to 10
            else:
                beta = 10.0 ** (1.0 + (t - 0.8) * 35.0)  # 10 to 10^8
        elif beta_schedule == 'pulse':
            # Periodic pulses of hardening
            beta_base = 10.0 ** (t * 6.0)
            pulse = 0.5 * (1.0 + math.sin(2 * math.pi * t * 8))
            beta = beta_base * (0.01 + pulse)
        else:
            beta = 10.0 ** (t * 8.0)

        val, grad = vg_fn(params, beta, alpha)

        grad_norm = jnp.sqrt(jnp.sum(grad**2))
        grad = jnp.where(grad_norm > 10.0, grad * 10.0 / grad_norm, grad)

        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)

        if step % 200 == 0 or step == total_steps - 1:
            r = params[2*n:]
            metric = float(jnp.sum(r))
            x = params[:n]
            y = params[n:2*n]

            wall_viol = float(jnp.max(jnp.array([
                jnp.max(r - x), jnp.max(x + r - 1.0),
                jnp.max(r - y), jnp.max(y + r - 1.0)
            ])))

            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = jnp.sqrt(dx**2 + dy**2 + 1e-30)
            sum_r_mat = r[:, None] + r[None, :]
            mask_mat = jnp.triu(jnp.ones((n, n), dtype=jnp.float64), k=1)
            overlap = (sum_r_mat - dist) * mask_mat
            max_overlap = float(jnp.max(overlap))

            max_viol = max(wall_viol, max_overlap)

            if max_viol < 1e-6 and metric > best_metric:
                best_metric = metric
                best_params = params.copy()

    return best_params, best_metric


def polish_slsqp(params_np, n=N, ftol=1e-15, maxiter=20000):
    """Polish with SLSQP."""
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

    result = minimize(
        objective, params_np, jac=obj_jac,
        method='SLSQP', constraints=constraints,
        options={'maxiter': maxiter, 'ftol': ftol, 'disp': False}
    )

    return result.x, -result.fun


def check_feasibility(params, n=N, tol=1e-10):
    x = params[:n]
    y = params[n:2*n]
    r = params[2*n:]
    max_viol = 0.0
    for i in range(n):
        max_viol = max(max_viol, r[i] - x[i], x[i] + r[i] - 1.0, r[i] - y[i], y[i] + r[i] - 1.0)
    for i in range(n):
        for j in range(i+1, n):
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


def save_solution(params, path, n=N):
    circles = [[float(params[i]), float(params[n+i]), float(params[2*n+i])] for i in range(n)]
    with open(path, 'w') as f:
        json.dump({"circles": circles}, f, indent=2)


# ============================================================
# Topology perturbation strategies
# ============================================================

def perturb_swap_positions(params, rng, n_swaps=2):
    """Swap positions of random circle pairs (keep radii in place)."""
    p = params.copy()
    x, y, r = p[:N], p[N:2*N], p[2*N:]

    for _ in range(n_swaps):
        i, j = rng.choice(N, size=2, replace=False)
        x[i], x[j] = x[j], x[i]
        y[i], y[j] = y[j], y[i]
        # Radii stay -- this changes the assignment of sizes to positions

    return np.concatenate([x, y, r])


def perturb_shuffle_radii(params, rng):
    """Shuffle which positions get which radii."""
    p = params.copy()
    r = p[2*N:].copy()
    rng.shuffle(r)
    p[2*N:] = r
    return p


def perturb_reflect_subset(params, rng, n_reflect=None):
    """Reflect a subset of circles across a random axis."""
    p = params.copy()
    x, y = p[:N].copy(), p[N:2*N].copy()

    if n_reflect is None:
        n_reflect = rng.randint(3, N//2 + 1)

    indices = rng.choice(N, size=n_reflect, replace=False)

    # Random axis through center
    axis_type = rng.randint(4)
    if axis_type == 0:  # horizontal
        y[indices] = 1.0 - y[indices]
    elif axis_type == 1:  # vertical
        x[indices] = 1.0 - x[indices]
    elif axis_type == 2:  # diagonal
        x[indices], y[indices] = y[indices].copy(), x[indices].copy()
    else:  # anti-diagonal
        x[indices], y[indices] = 1.0 - y[indices].copy(), 1.0 - x[indices].copy()

    p[:N] = x
    p[N:2*N] = y
    return p


def perturb_rotate_subset(params, rng, n_rotate=None):
    """Rotate a subset of circles around the center."""
    p = params.copy()
    x, y = p[:N].copy(), p[N:2*N].copy()

    if n_rotate is None:
        n_rotate = rng.randint(3, N//2 + 1)

    indices = rng.choice(N, size=n_rotate, replace=False)
    angle = rng.uniform(0.1, 2 * math.pi)

    cx, cy = 0.5, 0.5
    for i in indices:
        dx, dy = x[i] - cx, y[i] - cy
        x[i] = cx + dx * math.cos(angle) - dy * math.sin(angle)
        y[i] = cy + dx * math.sin(angle) + dy * math.cos(angle)

    # Clamp to unit square (with margin)
    r = p[2*N:]
    x = np.clip(x, r + 0.001, 1.0 - r - 0.001)
    y = np.clip(y, r + 0.001, 1.0 - r - 0.001)

    p[:N] = x
    p[N:2*N] = y
    return p


def perturb_remove_reinsert(params, rng, n_remove=3):
    """Remove n circles, let others expand, then reinsert in gaps."""
    p = params.copy()
    x, y, r = p[:N].copy(), p[N:2*N].copy(), p[2*N:].copy()

    # Remove smallest circles
    remove_idx = np.argsort(r)[:n_remove]
    keep_idx = np.setdiff1d(np.arange(N), remove_idx)

    # Save removed circle radii for later
    removed_r = r[remove_idx].copy()

    # Move removed circles to random positions with tiny radii
    for idx in remove_idx:
        x[idx] = rng.uniform(0.05, 0.95)
        y[idx] = rng.uniform(0.05, 0.95)
        r[idx] = 0.01  # very small

    return np.concatenate([x, y, r])


def perturb_explosion(params, rng, strength=0.15):
    """Push all circles outward from center (or inward)."""
    p = params.copy()
    x, y, r = p[:N].copy(), p[N:2*N].copy(), p[2*N:].copy()

    cx, cy = 0.5, 0.5
    for i in range(N):
        dx, dy = x[i] - cx, y[i] - cy
        dist = math.sqrt(dx**2 + dy**2) + 1e-10
        factor = 1.0 + rng.uniform(-strength, strength)
        x[i] = cx + dx * factor
        y[i] = cy + dy * factor

    x = np.clip(x, r + 0.001, 1.0 - r - 0.001)
    y = np.clip(y, r + 0.001, 1.0 - r - 0.001)

    p[:N] = x
    p[N:2*N] = y
    return p


def perturb_size_classes(params, rng):
    """Redistribute radii into different size classes.

    Instead of the optimal size distribution, try:
    - More equal sizes
    - More extreme sizes (few big, many small)
    - Different number of large circles
    """
    p = params.copy()
    r = p[2*N:].copy()
    total_r = np.sum(r)

    strategy = rng.randint(4)

    if strategy == 0:
        # More equal: all radii = total/N * (1 + small noise)
        r = np.ones(N) * (total_r / N) * (1.0 + rng.normal(0, 0.1, N))
    elif strategy == 1:
        # Power law: few big, many small
        r = rng.power(2.0, N)
        r = r / np.sum(r) * total_r
    elif strategy == 2:
        # Bimodal: half big, half small
        r[:N//2] = total_r * 0.6 / (N//2) * (1.0 + rng.normal(0, 0.05, N//2))
        r[N//2:] = total_r * 0.4 / (N - N//2) * (1.0 + rng.normal(0, 0.05, N - N//2))
        rng.shuffle(r)
    else:
        # Exponential distribution
        r = rng.exponential(0.1, N)
        r = r / np.sum(r) * total_r

    r = np.clip(r, 0.01, 0.25)
    p[2*N:] = r
    return p


def find_largest_gap(params, n=N):
    """Find the position with the largest gap between existing circles."""
    x, y, r = params[:n], params[n:2*n], params[2*n:]

    best_gap = 0
    best_pos = (0.5, 0.5)

    # Sample grid points
    for gx in np.linspace(0.05, 0.95, 30):
        for gy in np.linspace(0.05, 0.95, 30):
            # Min distance to any circle boundary
            min_gap = min(gx, 1.0 - gx, gy, 1.0 - gy)  # wall distance
            for i in range(n):
                dist = math.sqrt((gx - x[i])**2 + (gy - y[i])**2)
                gap = dist - r[i]
                min_gap = min(min_gap, gap)

            if min_gap > best_gap:
                best_gap = min_gap
                best_pos = (gx, gy)

    return best_pos, best_gap


# ============================================================
# Main V2 search
# ============================================================

def main():
    print("=" * 60)
    print("V2 JAX Optimizer: Aggressive Topology Exploration")
    print("=" * 60)

    workdir = Path(__file__).parent

    # Load best known solution
    best_known_path = workdir.parent / "topo-001" / "solution_n26.json"
    if not best_known_path.exists():
        best_known_path = workdir / "solution_n26.json"

    base_params = load_solution(str(best_known_path))
    base_metric = float(np.sum(base_params[2*N:]))
    print(f"Base solution metric: {base_metric:.10f}")

    overall_best_metric = 0.0
    overall_best_params = None
    all_results = []

    perturbation_fns = [
        ("swap2", lambda p, rng: perturb_swap_positions(p, rng, 2)),
        ("swap4", lambda p, rng: perturb_swap_positions(p, rng, 4)),
        ("swap6", lambda p, rng: perturb_swap_positions(p, rng, 6)),
        ("shuffle", perturb_shuffle_radii),
        ("reflect", perturb_reflect_subset),
        ("rotate", perturb_rotate_subset),
        ("remove2", lambda p, rng: perturb_remove_reinsert(p, rng, 2)),
        ("remove3", lambda p, rng: perturb_remove_reinsert(p, rng, 3)),
        ("remove5", lambda p, rng: perturb_remove_reinsert(p, rng, 5)),
        ("explode", perturb_explosion),
        ("size_cls", perturb_size_classes),
    ]

    schedules = ['standard', 'slow', 'cyclic', 'ultra_slow', 'pulse']

    trial_count = 0

    for perturb_name, perturb_fn in perturbation_fns:
        for sched in schedules:
            for seed_offset in range(5):
                seed = 10000 + trial_count * 7
                rng = np.random.RandomState(seed)
                key = jrandom.PRNGKey(seed)

                # Apply perturbation
                params_init = perturb_fn(base_params.copy(), rng)

                # Add some noise
                noise = rng.normal(0, 0.005, 3*N)
                params_init = params_init + noise
                params_init[2*N:] = np.maximum(params_init[2*N:], 0.01)

                # Optimize with JAX
                best_params, best_metric = optimize_adam(
                    params_init, energy_26, vg_26, N,
                    total_steps=4000, lr_init=1e-2, lr_final=1e-6,
                    beta_schedule=sched, alpha=1.0
                )

                # Polish
                if best_metric > 2.0:
                    polished, pol_metric = polish_slsqp(np.array(best_params))
                    viol = check_feasibility(polished)
                    if viol < 1e-10 and pol_metric > best_metric:
                        best_params = polished
                        best_metric = pol_metric

                tag = f"{perturb_name}_{sched}_{seed_offset}"
                all_results.append((tag, best_metric))
                trial_count += 1

                if best_metric > overall_best_metric:
                    overall_best_metric = best_metric
                    overall_best_params = best_params
                    print(f"  [{tag}] NEW BEST: {best_metric:.10f} (trial {trial_count})")
                elif trial_count % 25 == 0:
                    print(f"  [{tag}] metric={best_metric:.10f} (trial {trial_count}/{len(perturbation_fns)*len(schedules)*5})")

    # ---- Phase 2: Iterated perturbation from our best ----
    print("\n--- Phase 2: Iterated perturbation chains ---")
    if overall_best_params is not None:
        current = np.array(overall_best_params)
    else:
        current = base_params.copy()

    for chain in range(10):
        rng = np.random.RandomState(20000 + chain)

        # Apply 2-3 perturbations in sequence
        p = current.copy()
        n_perturbations = rng.randint(2, 4)
        applied = []
        for _ in range(n_perturbations):
            idx = rng.randint(len(perturbation_fns))
            name, fn = perturbation_fns[idx]
            p = fn(p, rng)
            applied.append(name)

        p[2*N:] = np.maximum(p[2*N:], 0.01)

        # Long optimization with ultra_slow schedule
        best_params, best_metric = optimize_adam(
            p, energy_26, vg_26, N,
            total_steps=6000, lr_init=1e-2, lr_final=1e-6,
            beta_schedule='ultra_slow', alpha=1.0
        )

        if best_metric > 2.0:
            polished, pol_metric = polish_slsqp(np.array(best_params))
            viol = check_feasibility(polished)
            if viol < 1e-10 and pol_metric > best_metric:
                best_params = polished
                best_metric = pol_metric

        tag = f"chain_{chain}_{'->'.join(applied)}"
        all_results.append((tag, best_metric))

        if best_metric > overall_best_metric:
            overall_best_metric = best_metric
            overall_best_params = best_params
            current = np.array(best_params)
            print(f"  [{tag}] NEW BEST: {best_metric:.10f}")
        else:
            print(f"  [{tag}] metric={best_metric:.10f}")

    # ---- Save ----
    print("\n" + "=" * 60)
    print(f"OVERALL BEST METRIC: {overall_best_metric:.10f}")
    print(f"Current best known:   {base_metric:.10f}")
    print(f"Improvement:          {overall_best_metric - base_metric:.2e}")
    print("=" * 60)

    if overall_best_params is not None:
        # Check feasibility one more time
        viol = check_feasibility(np.array(overall_best_params))
        if viol < 1e-10:
            save_solution(np.array(overall_best_params), str(workdir / "solution_n26_v2.json"))
            print(f"Saved valid solution to solution_n26_v2.json")

            # If better than current best, also update main solution
            if overall_best_metric > base_metric:
                save_solution(np.array(overall_best_params), str(workdir / "solution_n26.json"))
                print(f"NEW RECORD! Updated solution_n26.json")
        else:
            print(f"Best solution has violations: {viol:.2e}")

    # Save results
    results_sorted = sorted(all_results, key=lambda x: -x[1])
    with open(str(workdir / "results_v2.json"), 'w') as f:
        json.dump(results_sorted[:50], f, indent=2)

    print("\nTop 10:")
    for tag, m in results_sorted[:10]:
        print(f"  {tag}: {m:.10f}")

    return overall_best_metric


if __name__ == "__main__":
    main()
