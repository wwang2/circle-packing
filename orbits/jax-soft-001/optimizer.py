"""
JAX-based soft-body circle packing optimizer with constraint annealing.

Key idea: Use soft (penalty-based) constraints that allow overlap initially,
then gradually harden them. This lets gradients flow through topology changes
that hard-constraint optimizers cannot explore.

The energy function is differentiable everywhere, enabling smooth optimization
across topological transitions.
"""

import json
import math
import os
import sys
import time
from pathlib import Path
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax
import numpy as np
from scipy.optimize import minimize

# Ensure we use 64-bit precision
jax.config.update("jax_enable_x64", True)

N = 26  # number of circles

# ============================================================
# Energy function (fully differentiable)
# ============================================================

@jax.jit
def energy(params, beta, alpha):
    """Compute total energy = -alpha*sum(r) + beta*penalties.

    params: shape (3*N,) = [x0..x25, y0..y25, r0..r25]
    beta: penalty strength (annealed)
    alpha: radius reward strength
    """
    x = params[:N]
    y = params[N:2*N]
    r = params[2*N:]

    # Attractive: maximize sum of radii
    E_radii = -alpha * jnp.sum(r)

    # Wall containment penalties (soft)
    # r <= x, x <= 1-r, r <= y, y <= 1-r
    wall_left = jnp.maximum(0.0, r - x)
    wall_right = jnp.maximum(0.0, x + r - 1.0)
    wall_bottom = jnp.maximum(0.0, r - y)
    wall_top = jnp.maximum(0.0, y + r - 1.0)
    E_wall = beta * jnp.sum(wall_left**2 + wall_right**2 + wall_bottom**2 + wall_top**2)

    # Positive radius penalty
    E_pos = beta * jnp.sum(jnp.maximum(0.0, -r)**2)

    # Non-overlap penalty (vectorized over all pairs)
    # dist_ij >= r_i + r_j
    dx = x[:, None] - x[None, :]  # (N, N)
    dy = y[:, None] - y[None, :]  # (N, N)
    dist = jnp.sqrt(dx**2 + dy**2 + 1e-30)  # small eps to avoid grad issues at 0

    sum_r = r[:, None] + r[None, :]  # (N, N)
    overlap = jnp.maximum(0.0, sum_r - dist)  # positive when overlapping

    # Only upper triangle (avoid double counting and self)
    mask = jnp.triu(jnp.ones((N, N), dtype=jnp.float64), k=1)
    E_overlap = beta * jnp.sum((overlap * mask)**2)

    return E_radii + E_wall + E_overlap + E_pos


# Gradient of energy
grad_energy = jax.jit(jax.grad(energy))

# Also compile value_and_grad for efficiency
value_and_grad_energy = jax.jit(jax.value_and_grad(energy))


# ============================================================
# Initialization strategies
# ============================================================

def init_ring(key, n=N):
    """Concentric ring initialization (known to work well for n=26).
    1 center + 8 inner + 12 middle + 4 corners + 1 extra
    """
    circles = []
    # Center circle
    circles.append((0.5, 0.5, 0.13))

    # Inner ring: 8 circles
    for i in range(8):
        angle = 2 * math.pi * i / 8
        r_ring = 0.25
        cx = 0.5 + r_ring * math.cos(angle)
        cy = 0.5 + r_ring * math.sin(angle)
        circles.append((cx, cy, 0.10))

    # Middle ring: 12 circles
    for i in range(12):
        angle = 2 * math.pi * i / 12 + math.pi / 12
        r_ring = 0.40
        cx = 0.5 + r_ring * math.cos(angle)
        cy = 0.5 + r_ring * math.sin(angle)
        circles.append((cx, cy, 0.08))

    # Corners: 4 circles
    for cx, cy in [(0.08, 0.08), (0.92, 0.08), (0.08, 0.92), (0.92, 0.92)]:
        circles.append((cx, cy, 0.08))

    # Extra circle near center
    circles.append((0.5, 0.08, 0.07))

    circles = circles[:n]
    x = np.array([c[0] for c in circles])
    y = np.array([c[1] for c in circles])
    r = np.array([c[2] for c in circles])
    return np.concatenate([x, y, r])


def init_random(key, n=N):
    """Random initialization with small radii."""
    k1, k2, k3 = jrandom.split(key, 3)
    x = jrandom.uniform(k1, (n,), minval=0.05, maxval=0.95)
    y = jrandom.uniform(k2, (n,), minval=0.05, maxval=0.95)
    r = jrandom.uniform(k3, (n,), minval=0.02, maxval=0.06)
    return jnp.concatenate([x, y, r])


def init_hex(key, n=N):
    """Hexagonal grid initialization."""
    positions = []
    spacing = 0.18
    for row in range(10):
        for col in range(10):
            cx = 0.1 + col * spacing + (0.09 if row % 2 else 0.0)
            cy = 0.1 + row * spacing * 0.866
            if 0.05 < cx < 0.95 and 0.05 < cy < 0.95:
                positions.append((cx, cy))

    # Pick n positions (with some randomness)
    key_np = int(jrandom.randint(key, (), 0, 10000))
    rng = np.random.RandomState(key_np)
    indices = rng.choice(len(positions), size=min(n, len(positions)), replace=False)
    selected = [positions[i] for i in sorted(indices)]

    # Pad if needed
    while len(selected) < n:
        selected.append((rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)))

    x = np.array([p[0] for p in selected[:n]])
    y = np.array([p[1] for p in selected[:n]])
    r = np.ones(n) * 0.04
    return np.concatenate([x, y, r])


def init_from_solution(path, noise_scale=0.0, key=None):
    """Load a solution from JSON and optionally add noise."""
    with open(path) as f:
        data = json.load(f)
    circles = data["circles"] if "circles" in data else data
    x = np.array([c[0] for c in circles])
    y = np.array([c[1] for c in circles])
    r = np.array([c[2] for c in circles])
    params = np.concatenate([x, y, r])

    if noise_scale > 0 and key is not None:
        noise = jrandom.normal(key, params.shape) * noise_scale
        params = params + np.array(noise)
        # Clamp radii to be positive
        params = np.array(params)
        params[2*N:] = np.maximum(params[2*N:], 0.01)

    return params


# ============================================================
# Annealing schedules
# ============================================================

def schedule_standard(step, total_steps):
    """Standard exponential annealing: beta goes from 1 to 1e8."""
    t = step / total_steps
    log_beta = 0.0 + t * 8.0  # 10^0 to 10^8
    return 10.0 ** log_beta

def schedule_slow(step, total_steps):
    """Slow annealing with longer soft phase."""
    t = step / total_steps
    if t < 0.5:
        log_beta = t * 4.0  # 10^0 to 10^2 in first half
    else:
        log_beta = 2.0 + (t - 0.5) * 12.0  # 10^2 to 10^8 in second half
    return 10.0 ** log_beta

def schedule_cyclic(step, total_steps):
    """Cyclic annealing: periodically soften constraints."""
    t = step / total_steps
    base = t * 8.0  # overall trend
    cycle = 1.5 * math.sin(2 * math.pi * t * 5)  # 5 cycles
    log_beta = max(0.0, base + cycle)
    return 10.0 ** log_beta

def schedule_fast(step, total_steps):
    """Fast annealing: quickly harden."""
    t = step / total_steps
    log_beta = t * 10.0  # 10^0 to 10^10
    return 10.0 ** min(log_beta, 10.0)


# ============================================================
# Main optimization loop
# ============================================================

def optimize_one(params_init, schedule_fn, total_steps=4000,
                 lr_init=1e-2, lr_final=1e-5, alpha=1.0, seed=0):
    """Run one annealing trajectory.

    Returns: (best_params, best_metric, history)
    """
    params = jnp.array(params_init, dtype=jnp.float64)

    # Cosine decay learning rate
    schedule = optax.cosine_decay_schedule(lr_init, total_steps, alpha=lr_final/lr_init)
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(params)

    best_metric = -1e10
    best_params = params
    history = []

    for step in range(total_steps):
        beta = schedule_fn(step, total_steps)

        val, grad = value_and_grad_energy(params, beta, alpha)

        # Gradient clipping
        grad_norm = jnp.sqrt(jnp.sum(grad**2))
        grad = jnp.where(grad_norm > 10.0, grad * 10.0 / grad_norm, grad)

        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)

        # Evaluate feasibility periodically
        if step % 100 == 0 or step == total_steps - 1:
            r = params[2*N:]
            metric = float(jnp.sum(r))

            # Check feasibility
            x = params[:N]
            y = params[N:2*N]

            # Wall violations
            wall_viol = float(jnp.max(jnp.array([
                jnp.max(r - x), jnp.max(x + r - 1.0),
                jnp.max(r - y), jnp.max(y + r - 1.0)
            ])))

            # Overlap violations
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = jnp.sqrt(dx**2 + dy**2 + 1e-30)
            sum_r_mat = r[:, None] + r[None, :]
            mask = jnp.triu(jnp.ones((N, N), dtype=jnp.float64), k=1)
            overlap = (sum_r_mat - dist) * mask
            max_overlap = float(jnp.max(overlap))

            max_viol = max(wall_viol, max_overlap)
            feasible = max_viol < 1e-6

            history.append({
                'step': step, 'beta': beta, 'energy': float(val),
                'metric': metric, 'max_viol': max_viol, 'feasible': feasible
            })

            if feasible and metric > best_metric:
                best_metric = metric
                best_params = params.copy()

    return best_params, best_metric, history


def polish_scipy(params, tol=1e-12):
    """Final polish with scipy SLSQP to get exact constraint satisfaction."""
    params_np = np.array(params, dtype=np.float64)

    def objective(p):
        return -np.sum(p[2*N:])

    def obj_grad(p):
        g = np.zeros_like(p)
        g[2*N:] = -1.0
        return g

    constraints = []

    # Wall containment: x_i - r_i >= 0, 1 - x_i - r_i >= 0, etc.
    for i in range(N):
        # x - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: p[i] - p[2*N+i],
        })
        # 1 - x - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: 1.0 - p[i] - p[2*N+i],
        })
        # y - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: p[N+i] - p[2*N+i],
        })
        # 1 - y - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: 1.0 - p[N+i] - p[2*N+i],
        })
        # r > 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda p, i=i: p[2*N+i] - 1e-8,
        })

    # Non-overlap: dist(i,j) - r_i - r_j >= 0
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: math.sqrt(
                    (p[i]-p[j])**2 + (p[N+i]-p[N+j])**2
                ) - p[2*N+i] - p[2*N+j],
            })

    result = minimize(
        objective, params_np, jac=obj_grad,
        method='SLSQP', constraints=constraints,
        options={'maxiter': 10000, 'ftol': tol, 'disp': False}
    )

    return result.x, -result.fun


def params_to_circles(params):
    """Convert flat params to list of [x, y, r]."""
    x = params[:N]
    y = params[N:2*N]
    r = params[2*N:]
    return [[float(x[i]), float(y[i]), float(r[i])] for i in range(N)]


def save_solution(params, path):
    """Save solution to JSON."""
    circles = params_to_circles(params)
    with open(path, 'w') as f:
        json.dump({"circles": circles}, f, indent=2)


# ============================================================
# Main search
# ============================================================

def main():
    print("=" * 60)
    print("JAX Soft-Body Circle Packing Optimizer")
    print("=" * 60)

    workdir = Path(__file__).parent
    best_known_path = workdir.parent / "topo-001" / "solution_n26.json"

    overall_best_metric = 0.0
    overall_best_params = None
    all_results = []

    schedules = {
        'standard': schedule_standard,
        'slow': schedule_slow,
        'cyclic': schedule_cyclic,
        'fast': schedule_fast,
    }

    # ---- Phase A: Warm starts from best known solution ----
    print("\n--- Phase A: Warm starts from best known ---")
    if best_known_path.exists():
        for sched_name, sched_fn in schedules.items():
            for noise_idx in range(10):
                noise_scale = 0.001 * (noise_idx + 1)  # 0.001 to 0.01
                key = jrandom.PRNGKey(42 + noise_idx * 100)
                params_init = init_from_solution(str(best_known_path),
                                                  noise_scale=noise_scale, key=key)

                best_params, best_metric, history = optimize_one(
                    params_init, sched_fn, total_steps=3000,
                    lr_init=5e-3, lr_final=1e-6, alpha=1.0,
                    seed=42 + noise_idx
                )

                # Polish with SLSQP
                if best_metric > 0:
                    polished, pol_metric = polish_scipy(best_params)
                    if pol_metric > best_metric:
                        best_params = polished
                        best_metric = pol_metric

                tag = f"warm_{sched_name}_noise{noise_scale:.3f}"
                all_results.append((tag, best_metric))

                if best_metric > overall_best_metric:
                    overall_best_metric = best_metric
                    overall_best_params = best_params
                    print(f"  [{tag}] NEW BEST: {best_metric:.10f}")
                else:
                    print(f"  [{tag}] metric={best_metric:.10f}")

    # ---- Phase B: Random initializations ----
    print("\n--- Phase B: Random initializations ---")
    for trial in range(50):
        key = jrandom.PRNGKey(1000 + trial)

        # Alternate between init strategies
        if trial % 3 == 0:
            params_init = init_ring(key)
            init_type = "ring"
        elif trial % 3 == 1:
            params_init = init_random(key)
            init_type = "random"
        else:
            params_init = init_hex(key)
            init_type = "hex"

        # Pick schedule
        sched_name = list(schedules.keys())[trial % len(schedules)]
        sched_fn = schedules[sched_name]

        best_params, best_metric, history = optimize_one(
            params_init, sched_fn, total_steps=4000,
            lr_init=1e-2, lr_final=1e-5, alpha=1.0,
            seed=1000 + trial
        )

        # Polish
        if best_metric > 0:
            polished, pol_metric = polish_scipy(best_params)
            if pol_metric > best_metric:
                best_params = polished
                best_metric = pol_metric

        tag = f"cold_{init_type}_{sched_name}_{trial}"
        all_results.append((tag, best_metric))

        if best_metric > overall_best_metric:
            overall_best_metric = best_metric
            overall_best_params = best_params
            print(f"  [{tag}] NEW BEST: {best_metric:.10f}")
        elif trial % 10 == 0:
            print(f"  [{tag}] metric={best_metric:.10f}")

    # ---- Phase C: Large perturbation from best found ----
    print("\n--- Phase C: Large perturbations ---")
    if overall_best_params is not None:
        for trial in range(30):
            key = jrandom.PRNGKey(5000 + trial)
            noise = jrandom.normal(key, (3*N,)) * 0.05
            params_init = np.array(overall_best_params) + np.array(noise)
            params_init[2*N:] = np.maximum(params_init[2*N:], 0.01)

            sched_name = list(schedules.keys())[trial % len(schedules)]
            sched_fn = schedules[sched_name]

            best_params, best_metric, history = optimize_one(
                params_init, sched_fn, total_steps=3000,
                lr_init=3e-3, lr_final=1e-6, alpha=1.0,
                seed=5000 + trial
            )

            if best_metric > 0:
                polished, pol_metric = polish_scipy(best_params)
                if pol_metric > best_metric:
                    best_params = polished
                    best_metric = pol_metric

            tag = f"perturb_{sched_name}_{trial}"
            all_results.append((tag, best_metric))

            if best_metric > overall_best_metric:
                overall_best_metric = best_metric
                overall_best_params = best_params
                print(f"  [{tag}] NEW BEST: {best_metric:.10f}")
            elif trial % 10 == 0:
                print(f"  [{tag}] metric={best_metric:.10f}")

    # ---- Phase D: Cyclic re-annealing (topology escape) ----
    print("\n--- Phase D: Cyclic re-annealing ---")
    if overall_best_params is not None:
        for trial in range(20):
            key = jrandom.PRNGKey(8000 + trial)
            noise = jrandom.normal(key, (3*N,)) * 0.02
            params_init = np.array(overall_best_params) + np.array(noise)
            params_init[2*N:] = np.maximum(params_init[2*N:], 0.01)

            # Use cyclic schedule with more steps
            best_params, best_metric, history = optimize_one(
                params_init, schedule_cyclic, total_steps=6000,
                lr_init=5e-3, lr_final=1e-6, alpha=1.0,
                seed=8000 + trial
            )

            if best_metric > 0:
                polished, pol_metric = polish_scipy(best_params)
                if pol_metric > best_metric:
                    best_params = polished
                    best_metric = pol_metric

            tag = f"cyclic_{trial}"
            all_results.append((tag, best_metric))

            if best_metric > overall_best_metric:
                overall_best_metric = best_metric
                overall_best_params = best_params
                print(f"  [{tag}] NEW BEST: {best_metric:.10f}")
            elif trial % 10 == 0:
                print(f"  [{tag}] metric={best_metric:.10f}")

    # ---- Save results ----
    print("\n" + "=" * 60)
    print(f"OVERALL BEST METRIC: {overall_best_metric:.10f}")
    print("=" * 60)

    if overall_best_params is not None:
        save_solution(overall_best_params, str(workdir / "solution_n26.json"))
        print(f"Saved to {workdir / 'solution_n26.json'}")

    # Save all results for analysis
    results_summary = sorted(all_results, key=lambda x: -x[1])
    with open(str(workdir / "results_summary.json"), 'w') as f:
        json.dump(results_summary[:50], f, indent=2)

    # Print top 10
    print("\nTop 10 results:")
    for tag, metric in results_summary[:10]:
        print(f"  {tag}: {metric:.10f}")

    return overall_best_metric, overall_best_params


if __name__ == "__main__":
    main()
