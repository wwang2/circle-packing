"""
V6: Basin-Hopping with Topology-Disrupting Moves

Key ideas:
1. Basin-hopping outer loop: perturb -> optimize -> accept/reject (Metropolis)
2. Topology-disrupting perturbations:
   - Teleport circles to random positions
   - Swap circle positions (changes contact graph)
   - Redistribute radii (Dirichlet samples)
   - Mirror/rotate subsets of circles
   - "Squeeze": temporarily shrink all radii, repack from scratch
3. Alternative objectives at low beta to find different topologies:
   - Coverage (pi*sum(r^2)) favors many small circles
   - Entropy (-sum(r*log(r))) favors uniform sizes
4. Multiple independent chains with different random seeds
"""

import json
import math
import os
import sys
import time
import functools
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax
import numpy as np
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)

WORKDIR = Path(__file__).parent
N = 26

print = functools.partial(print, flush=True)


# ============================================================
# Energy functions
# ============================================================

@jax.jit
def energy_sum_radii(params, beta, alpha):
    """Standard: maximize sum of radii."""
    n = N
    x, y, r = params[:n], params[n:2*n], params[2*n:]
    E_obj = -alpha * jnp.sum(r)

    wall_l = jnp.maximum(0.0, r - x)
    wall_r = jnp.maximum(0.0, x + r - 1.0)
    wall_b = jnp.maximum(0.0, r - y)
    wall_t = jnp.maximum(0.0, y + r - 1.0)
    E_wall = beta * jnp.sum(wall_l**2 + wall_r**2 + wall_b**2 + wall_t**2)

    E_pos = beta * jnp.sum(jnp.maximum(0.0, -r)**2)

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = jnp.sqrt(dx**2 + dy**2 + 1e-30)
    sum_r = r[:, None] + r[None, :]
    overlap = jnp.maximum(0.0, sum_r - dist)
    mask = jnp.triu(jnp.ones((n, n), dtype=jnp.float64), k=1)
    E_overlap = beta * jnp.sum((overlap * mask)**2)

    return E_obj + E_wall + E_overlap + E_pos


@jax.jit
def energy_coverage(params, beta, alpha):
    """Alternative: maximize coverage = sum(r^2)."""
    n = N
    x, y, r = params[:n], params[n:2*n], params[2*n:]
    E_obj = -alpha * jnp.sum(r**2)

    wall_l = jnp.maximum(0.0, r - x)
    wall_r = jnp.maximum(0.0, x + r - 1.0)
    wall_b = jnp.maximum(0.0, r - y)
    wall_t = jnp.maximum(0.0, y + r - 1.0)
    E_wall = beta * jnp.sum(wall_l**2 + wall_r**2 + wall_b**2 + wall_t**2)

    E_pos = beta * jnp.sum(jnp.maximum(0.0, -r)**2)

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = jnp.sqrt(dx**2 + dy**2 + 1e-30)
    sum_r = r[:, None] + r[None, :]
    overlap = jnp.maximum(0.0, sum_r - dist)
    mask = jnp.triu(jnp.ones((n, n), dtype=jnp.float64), k=1)
    E_overlap = beta * jnp.sum((overlap * mask)**2)

    return E_obj + E_wall + E_overlap + E_pos


@jax.jit
def energy_entropy(params, beta, alpha):
    """Alternative: maximize entropy = -sum(r*log(r))."""
    n = N
    x, y, r = params[:n], params[n:2*n], params[2*n:]
    r_safe = jnp.maximum(r, 1e-10)
    E_obj = alpha * jnp.sum(r_safe * jnp.log(r_safe))  # minimize r*log(r) = maximize entropy

    wall_l = jnp.maximum(0.0, r - x)
    wall_r = jnp.maximum(0.0, x + r - 1.0)
    wall_b = jnp.maximum(0.0, r - y)
    wall_t = jnp.maximum(0.0, y + r - 1.0)
    E_wall = beta * jnp.sum(wall_l**2 + wall_r**2 + wall_b**2 + wall_t**2)

    E_pos = beta * jnp.sum(jnp.maximum(0.0, -r)**2)

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = jnp.sqrt(dx**2 + dy**2 + 1e-30)
    sum_r = r[:, None] + r[None, :]
    overlap = jnp.maximum(0.0, sum_r - dist)
    mask = jnp.triu(jnp.ones((n, n), dtype=jnp.float64), k=1)
    E_overlap = beta * jnp.sum((overlap * mask)**2)

    return E_obj + E_wall + E_overlap + E_pos


# Pre-compile value_and_grad for each energy
_vg_sum = jax.jit(jax.value_and_grad(energy_sum_radii))
_vg_cov = jax.jit(jax.value_and_grad(energy_coverage))
_vg_ent = jax.jit(jax.value_and_grad(energy_entropy))


# ============================================================
# Utilities
# ============================================================

def check_feasibility(params, tol=1e-10):
    x, y, r = params[:N], params[N:2*N], params[2*N:]
    max_viol = 0.0
    for i in range(N):
        max_viol = max(max_viol, r[i] - x[i], x[i] + r[i] - 1.0,
                       r[i] - y[i], y[i] + r[i] - 1.0)
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


def get_contact_graph(params, tol=1e-4):
    """Return set of contact pairs (i,j) where circles are nearly touching."""
    x, y, r = params[:N], params[N:2*N], params[2*N:]
    contacts = set()
    for i in range(N):
        for j in range(i+1, N):
            dist = math.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            gap = dist - r[i] - r[j]
            if abs(gap) < tol:
                contacts.add((i, j))
        # Wall contacts
        if abs(x[i] - r[i]) < tol:
            contacts.add((i, 'L'))
        if abs(1.0 - x[i] - r[i]) < tol:
            contacts.add((i, 'R'))
        if abs(y[i] - r[i]) < tol:
            contacts.add((i, 'B'))
        if abs(1.0 - y[i] - r[i]) < tol:
            contacts.add((i, 'T'))
    return frozenset(contacts)


def contact_graph_signature(params, tol=1e-4):
    """Compute a topology signature: sorted degree sequence."""
    x, y, r = params[:N], params[N:2*N], params[2*N:]
    degrees = [0] * N
    for i in range(N):
        for j in range(i+1, N):
            dist = math.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            gap = dist - r[i] - r[j]
            if abs(gap) < tol:
                degrees[i] += 1
                degrees[j] += 1
        if abs(x[i] - r[i]) < tol: degrees[i] += 1
        if abs(1.0 - x[i] - r[i]) < tol: degrees[i] += 1
        if abs(y[i] - r[i]) < tol: degrees[i] += 1
        if abs(1.0 - y[i] - r[i]) < tol: degrees[i] += 1
    return tuple(sorted(degrees, reverse=True))


# ============================================================
# Core optimizer: annealing with switchable objective
# ============================================================

def optimize_anneal(params, total_steps=4000, lr_init=5e-3, lr_final=1e-6,
                    objective='sum_radii', schedule='slow', alpha=1.0):
    """Run annealing optimization with given objective function."""
    params = jnp.array(params, dtype=jnp.float64)

    if objective == 'coverage':
        vg_fn = _vg_cov
    elif objective == 'entropy':
        vg_fn = _vg_ent
    else:
        vg_fn = _vg_sum

    sched = optax.cosine_decay_schedule(lr_init, total_steps, alpha=lr_final/lr_init)
    optimizer = optax.adam(sched)
    opt_state = optimizer.init(params)

    best_metric = -1e10
    best_params = params

    for step in range(total_steps):
        t = step / total_steps

        if schedule == 'slow':
            if t < 0.6:
                beta = 10.0 ** (t * 2.0)
            else:
                beta = 10.0 ** (1.2 + (t - 0.6) * 17.0)
        elif schedule == 'ultra_slow':
            if t < 0.8:
                beta = 10.0 ** (t * 1.5)
            else:
                beta = 10.0 ** (1.2 + (t - 0.8) * 34.0)
        elif schedule == 'fast':
            beta = 10.0 ** (t * 8.0)
        elif schedule == 'soft_then_hard':
            # Stay very soft for 70% then slam hard
            if t < 0.7:
                beta = 10.0 ** (t * 0.5)
            else:
                beta = 10.0 ** (0.35 + (t - 0.7) * 25.5)
        elif schedule == 'cyclic':
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

        if step % 500 == 0 or step == total_steps - 1:
            r = params[2*N:]
            metric = float(jnp.sum(r))
            if metric > best_metric:
                best_metric = metric
                best_params = params

    return np.array(best_params), best_metric


def polish_slsqp(params_np, ftol=1e-15, maxiter=20000):
    """Polish with SLSQP to get exact feasibility."""
    params_np = np.array(params_np, dtype=np.float64)

    def objective(p):
        return -np.sum(p[2*N:])

    def obj_jac(p):
        g = np.zeros_like(p)
        g[2*N:] = -1.0
        return g

    constraints = []
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[i] - p[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1.0 - p[i] - p[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[N+i] - p[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1.0 - p[N+i] - p[2*N+i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[2*N+i] - 1e-10})

    for i in range(N):
        for j in range(i+1, N):
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: math.sqrt(
                    (p[i]-p[j])**2 + (p[N+i]-p[N+j])**2
                ) - p[2*N+i] - p[2*N+j]
            })

    result = minimize(objective, params_np, jac=obj_jac, method='SLSQP',
                      constraints=constraints,
                      options={'maxiter': maxiter, 'ftol': ftol, 'disp': False})
    return result.x, -result.fun


# ============================================================
# Topology-disrupting perturbations
# ============================================================

def perturb_teleport(params, rng, n_circles=None):
    """Teleport 2-6 circles to random positions."""
    p = params.copy()
    if n_circles is None:
        n_circles = rng.randint(2, 7)
    idx = rng.choice(N, size=min(n_circles, N), replace=False)
    for k in idx:
        p[k] = rng.uniform(0.05, 0.95)
        p[N+k] = rng.uniform(0.05, 0.95)
        # Keep radius but maybe adjust
        p[2*N+k] *= rng.uniform(0.5, 1.5)
        p[2*N+k] = np.clip(p[2*N+k], 0.01, 0.2)
    return p


def perturb_swap_positions(params, rng, n_swaps=None):
    """Swap positions of several pairs of circles."""
    p = params.copy()
    if n_swaps is None:
        n_swaps = rng.randint(2, 8)
    for _ in range(n_swaps):
        a, b = rng.choice(N, 2, replace=False)
        p[a], p[b] = p[b], p[a]
        p[N+a], p[N+b] = p[N+b], p[N+a]
    return p


def perturb_redistribute_radii(params, rng, concentration=0.3):
    """Sample new radii from Dirichlet, keeping total roughly the same."""
    p = params.copy()
    total = np.sum(p[2*N:])
    alphas = np.ones(N) * concentration
    new_r = rng.dirichlet(alphas) * total
    p[2*N:] = np.clip(new_r, 0.005, 0.3)
    return p


def perturb_mirror_subset(params, rng):
    """Mirror a subset of circles across x=0.5 or y=0.5."""
    p = params.copy()
    n_select = rng.randint(N//4, 3*N//4)
    idx = rng.choice(N, size=n_select, replace=False)
    if rng.random() < 0.5:
        p[idx] = 1.0 - p[idx]  # mirror x
    else:
        p[N+idx] = 1.0 - p[N+idx]  # mirror y
    return p


def perturb_rotate_cluster(params, rng):
    """Rotate a cluster of nearby circles by 90 or 180 degrees around their centroid."""
    p = params.copy()
    # Pick a random circle and find its nearest neighbors
    pivot = rng.randint(N)
    px, py = p[pivot], p[N+pivot]

    dists = np.sqrt((p[:N] - px)**2 + (p[N:2*N] - py)**2)
    cluster_size = rng.randint(3, min(10, N))
    idx = np.argsort(dists)[:cluster_size]

    # Rotate around centroid
    cx = np.mean(p[idx])
    cy = np.mean(p[N+idx])

    angle = rng.choice([math.pi/2, math.pi, 3*math.pi/2])
    cos_a, sin_a = math.cos(angle), math.sin(angle)

    for k in idx:
        dx = p[k] - cx
        dy = p[N+k] - cy
        p[k] = cx + cos_a * dx - sin_a * dy
        p[N+k] = cy + sin_a * dx + cos_a * dy

    # Clip to valid range
    for k in idx:
        p[k] = np.clip(p[k], 0.02, 0.98)
        p[N+k] = np.clip(p[N+k], 0.02, 0.98)

    return p


def perturb_squeeze_repack(params, rng, factor=0.6):
    """Shrink all radii, add random perturbation, let optimizer re-expand."""
    p = params.copy()
    p[2*N:] *= factor
    # Add position noise
    p[:N] += rng.normal(0, 0.05, N)
    p[N:2*N] += rng.normal(0, 0.05, N)
    p[:N] = np.clip(p[:N], 0.02, 0.98)
    p[N:2*N] = np.clip(p[N:2*N], 0.02, 0.98)
    p[2*N:] = np.maximum(p[2*N:], 0.005)
    return p


def perturb_random_init(rng, strategy='hex'):
    """Generate a completely random initialization."""
    if strategy == 'hex':
        # Hexagonal grid + noise
        positions = []
        r_est = 1.0 / (2 * math.sqrt(N))
        rows = int(math.ceil(math.sqrt(N)))
        cols = int(math.ceil(N / rows))
        for i in range(rows):
            for j in range(cols):
                if len(positions) >= N:
                    break
                x = (j + 0.5 * (i % 2)) / cols * 0.8 + 0.1
                y = i / rows * 0.8 + 0.1
                positions.append((x + rng.normal(0, 0.03), y + rng.normal(0, 0.03)))
        positions = positions[:N]
        x = np.clip([p[0] for p in positions], 0.05, 0.95)
        y = np.clip([p[1] for p in positions], 0.05, 0.95)
        r = np.full(N, r_est) * rng.uniform(0.5, 1.5, N)
        r = np.clip(r, 0.01, 0.2)
    elif strategy == 'random':
        x = rng.uniform(0.05, 0.95, N)
        y = rng.uniform(0.05, 0.95, N)
        r = rng.uniform(0.03, 0.15, N)
    elif strategy == 'concentric':
        # Concentric rings
        x, y, r = [], [], []
        # Center circle
        x.append(0.5)
        y.append(0.5)
        r.append(0.14 + rng.normal(0, 0.02))
        # Inner ring
        n_inner = 7
        for i in range(n_inner):
            angle = 2 * math.pi * i / n_inner + rng.normal(0, 0.1)
            x.append(0.5 + 0.27 * math.cos(angle))
            y.append(0.5 + 0.27 * math.sin(angle))
            r.append(0.10 + rng.normal(0, 0.015))
        # Outer ring
        n_outer = N - 1 - n_inner
        for i in range(n_outer):
            angle = 2 * math.pi * i / n_outer + rng.normal(0, 0.1)
            x.append(0.5 + 0.42 * math.cos(angle))
            y.append(0.5 + 0.42 * math.sin(angle))
            r.append(0.07 + rng.normal(0, 0.01))
        x = np.clip(x[:N], 0.05, 0.95)
        y = np.clip(y[:N], 0.05, 0.95)
        r = np.clip(r[:N], 0.01, 0.2)
    else:
        # Grid
        side = int(math.ceil(math.sqrt(N)))
        positions = []
        for i in range(side):
            for j in range(side):
                if len(positions) >= N:
                    break
                positions.append(((i + 0.5) / side, (j + 0.5) / side))
        positions = positions[:N]
        x = np.array([p[0] for p in positions]) + rng.normal(0, 0.02, N)
        y = np.array([p[1] for p in positions]) + rng.normal(0, 0.02, N)
        x = np.clip(x, 0.05, 0.95)
        y = np.clip(y, 0.05, 0.95)
        r = np.full(N, 0.08) * rng.uniform(0.6, 1.4, N)

    return np.concatenate([np.array(x), np.array(y), np.array(r)])


ALL_PERTURBATIONS = [
    ('teleport_2', lambda p, rng: perturb_teleport(p, rng, 2)),
    ('teleport_4', lambda p, rng: perturb_teleport(p, rng, 4)),
    ('teleport_6', lambda p, rng: perturb_teleport(p, rng, 6)),
    ('swap_3', lambda p, rng: perturb_swap_positions(p, rng, 3)),
    ('swap_6', lambda p, rng: perturb_swap_positions(p, rng, 6)),
    ('swap_10', lambda p, rng: perturb_swap_positions(p, rng, 10)),
    ('redistribute_soft', lambda p, rng: perturb_redistribute_radii(p, rng, 0.5)),
    ('redistribute_hard', lambda p, rng: perturb_redistribute_radii(p, rng, 0.1)),
    ('mirror', perturb_mirror_subset),
    ('rotate_cluster', perturb_rotate_cluster),
    ('squeeze_60', lambda p, rng: perturb_squeeze_repack(p, rng, 0.6)),
    ('squeeze_40', lambda p, rng: perturb_squeeze_repack(p, rng, 0.4)),
    ('squeeze_20', lambda p, rng: perturb_squeeze_repack(p, rng, 0.2)),
]


# ============================================================
# Basin-Hopping with Topology Tracking
# ============================================================

def basin_hop(base_params, n_hops=50, seed=42, tag="BH"):
    """Basin-hopping with topology-disrupting perturbations."""
    rng = np.random.RandomState(seed)

    current_params = base_params.copy()
    current_metric = float(np.sum(current_params[2*N:]))

    best_params = current_params.copy()
    best_metric = current_metric

    seen_topologies = set()
    topo_sig = contact_graph_signature(current_params)
    seen_topologies.add(topo_sig)

    temperature = 0.05  # Metropolis temperature for accepting worse solutions

    print(f"  [{tag}] Starting basin-hop from metric={current_metric:.10f}")
    print(f"  [{tag}] Initial topology: {topo_sig[:5]}...")

    for hop in range(n_hops):
        t0 = time.time()

        # Pick perturbation
        pert_name, pert_fn = ALL_PERTURBATIONS[rng.randint(len(ALL_PERTURBATIONS))]

        # Also randomly decide objective for this hop
        obj_roll = rng.random()
        if obj_roll < 0.5:
            objective = 'sum_radii'
        elif obj_roll < 0.75:
            objective = 'coverage'
        else:
            objective = 'entropy'

        # Pick schedule
        schedule = rng.choice(['slow', 'ultra_slow', 'soft_then_hard', 'cyclic', 'fast'])

        # Perturb
        perturbed = pert_fn(current_params, rng)

        # Optimize with chosen objective
        steps = rng.choice([3000, 4000, 5000])
        lr = rng.choice([1e-2, 5e-3, 2e-3])

        opt_params, opt_raw = optimize_anneal(
            perturbed, total_steps=steps, lr_init=lr,
            objective=objective, schedule=schedule
        )

        # Always finish with sum_radii objective for final polish
        if objective != 'sum_radii':
            opt_params, opt_raw = optimize_anneal(
                opt_params, total_steps=2000, lr_init=2e-3,
                objective='sum_radii', schedule='slow'
            )

        # SLSQP polish
        polished, metric = polish_slsqp(opt_params)
        viol = check_feasibility(polished)

        elapsed = time.time() - t0

        if viol < 1e-10:
            topo = contact_graph_signature(polished)
            is_new_topo = topo not in seen_topologies
            seen_topologies.add(topo)

            # Metropolis acceptance
            delta = metric - current_metric
            if delta > 0 or rng.random() < math.exp(delta / temperature):
                current_params = polished.copy()
                current_metric = metric
                accept = "ACCEPT"
            else:
                accept = "reject"

            if metric > best_metric:
                best_metric = metric
                best_params = polished.copy()
                accept = "NEW BEST"

            topo_str = "NEW_TOPO" if is_new_topo else "known"
            if hop % 3 == 0 or accept == "NEW BEST":
                print(f"    hop {hop}: {pert_name}/{objective}/{schedule} -> "
                      f"m={metric:.10f} [{accept}] topo={topo_str} ({elapsed:.1f}s)")
        else:
            if hop % 5 == 0:
                print(f"    hop {hop}: {pert_name}/{objective}/{schedule} -> "
                      f"infeasible viol={viol:.2e} ({elapsed:.1f}s)")

    print(f"  [{tag}] Best: {best_metric:.10f}, {len(seen_topologies)} topologies seen")
    return best_params, best_metric


def cold_start_search(n_starts=20, seed=42, tag="COLD"):
    """Run many cold starts with diverse initializations."""
    rng = np.random.RandomState(seed)

    best_params = None
    best_metric = 0.0
    seen_topologies = set()

    strategies = ['hex', 'random', 'concentric', 'grid']
    schedules = ['slow', 'ultra_slow', 'soft_then_hard', 'cyclic']
    objectives = ['sum_radii', 'sum_radii', 'sum_radii', 'coverage', 'entropy']

    print(f"  [{tag}] Running {n_starts} cold starts")

    for i in range(n_starts):
        t0 = time.time()
        strategy = strategies[i % len(strategies)]
        schedule = schedules[i % len(schedules)]
        objective = objectives[i % len(objectives)]

        init = perturb_random_init(rng, strategy)

        # First optimize with chosen objective
        opt_p, opt_m = optimize_anneal(
            init, total_steps=5000, lr_init=1e-2,
            objective=objective, schedule=schedule
        )

        # Then with sum_radii
        if objective != 'sum_radii':
            opt_p, opt_m = optimize_anneal(
                opt_p, total_steps=3000, lr_init=3e-3,
                objective='sum_radii', schedule='slow'
            )

        # Polish
        polished, metric = polish_slsqp(opt_p)
        viol = check_feasibility(polished)
        elapsed = time.time() - t0

        if viol < 1e-10 and metric > 2.0:
            topo = contact_graph_signature(polished)
            is_new = topo not in seen_topologies
            seen_topologies.add(topo)

            if metric > best_metric:
                best_metric = metric
                best_params = polished.copy()
                print(f"    start {i}: {strategy}/{objective}/{schedule} -> "
                      f"m={metric:.10f} NEW_BEST topo={'NEW' if is_new else 'known'} ({elapsed:.1f}s)")
            elif is_new and i % 3 == 0:
                print(f"    start {i}: {strategy}/{objective}/{schedule} -> "
                      f"m={metric:.10f} new_topo ({elapsed:.1f}s)")
        elif i % 5 == 0:
            raw_m = float(np.sum(opt_p[2*N:]))
            print(f"    start {i}: {strategy}/{objective}/{schedule} -> "
                  f"raw_m={raw_m:.6f} viol={viol:.2e} ({elapsed:.1f}s)")

    print(f"  [{tag}] Best: {best_metric:.10f}, {len(seen_topologies)} topologies")
    return best_params, best_metric


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("V6: Basin-Hopping with Topology-Disrupting Moves")
    print("=" * 60)

    sol_path = WORKDIR / "solution_n26.json"
    base_params = load_solution(str(sol_path))
    base_metric = float(np.sum(base_params[2*N:]))
    base_topo = contact_graph_signature(base_params)
    print(f"Base metric: {base_metric:.10f}")
    print(f"Base topology: {base_topo}")

    overall_best_metric = base_metric
    overall_best_params = base_params.copy()
    all_results = []

    # ---- Phase 1: Basin hopping from known solution (4 chains) ----
    print("\n--- PHASE 1: BASIN HOPPING ---")
    for chain in range(4):
        seed = 300000 + chain * 1337
        bh_p, bh_m = basin_hop(base_params, n_hops=40, seed=seed, tag=f"BH_{chain}")
        all_results.append((f"BH_{chain}", bh_m))
        if bh_p is not None and bh_m > overall_best_metric:
            overall_best_metric = bh_m
            overall_best_params = bh_p
            print(f"  NEW OVERALL BEST from BH_{chain}: {bh_m:.10f}")

    # ---- Phase 2: Cold starts with diverse objectives ----
    print("\n--- PHASE 2: COLD STARTS ---")
    for trial in range(3):
        seed = 400000 + trial * 2023
        cs_p, cs_m = cold_start_search(n_starts=15, seed=seed, tag=f"COLD_{trial}")
        all_results.append((f"COLD_{trial}", cs_m))
        if cs_p is not None and cs_m > overall_best_metric:
            overall_best_metric = cs_m
            overall_best_params = cs_p
            print(f"  NEW OVERALL BEST from COLD_{trial}: {cs_m:.10f}")

    # ---- Phase 3: Basin-hop from best cold start results ----
    print("\n--- PHASE 3: BASIN HOP FROM BEST FOUND ---")
    if overall_best_metric > base_metric:
        for chain in range(2):
            seed = 500000 + chain * 999
            bh_p, bh_m = basin_hop(overall_best_params, n_hops=30, seed=seed,
                                    tag=f"BH2_{chain}")
            all_results.append((f"BH2_{chain}", bh_m))
            if bh_p is not None and bh_m > overall_best_metric:
                overall_best_metric = bh_m
                overall_best_params = bh_p
                print(f"  NEW OVERALL BEST from BH2_{chain}: {bh_m:.10f}")

    # ---- Phase 4: Basin-hop from mirror of best known ----
    print("\n--- PHASE 4: MIRROR + BASIN HOP ---")
    mirror_params = base_params.copy()
    mirror_params[:N] = 1.0 - mirror_params[:N]  # Mirror x
    for chain in range(2):
        seed = 600000 + chain * 503
        bh_p, bh_m = basin_hop(mirror_params, n_hops=30, seed=seed,
                                tag=f"MIRROR_{chain}")
        all_results.append((f"MIRROR_{chain}", bh_m))
        if bh_p is not None and bh_m > overall_best_metric:
            overall_best_metric = bh_m
            overall_best_params = bh_p
            print(f"  NEW OVERALL BEST from MIRROR_{chain}: {bh_m:.10f}")

    # ---- Report ----
    print("\n" + "=" * 60)
    print(f"OVERALL BEST: {overall_best_metric:.10f}")
    print(f"Base metric:  {base_metric:.10f}")
    improvement = overall_best_metric - base_metric
    print(f"Improvement:  {improvement:.2e}")
    print("=" * 60)

    if overall_best_params is not None:
        viol = check_feasibility(overall_best_params)
        if viol < 1e-10:
            save_solution(overall_best_params, str(WORKDIR / "solution_n26_v6.json"))
            print(f"Saved solution_n26_v6.json (viol={viol:.2e})")
            if overall_best_metric >= base_metric:
                save_solution(overall_best_params, str(WORKDIR / "solution_n26.json"))
                print("Updated solution_n26.json")

    results_sorted = sorted(all_results, key=lambda x: -x[1])
    with open(str(WORKDIR / "results_v6.json"), 'w') as f:
        json.dump(results_sorted, f, indent=2)

    print("\nAll results:")
    for tag, m in results_sorted[:10]:
        print(f"  {tag}: {m:.10f}")

    return overall_best_metric, overall_best_params


if __name__ == "__main__":
    main()
