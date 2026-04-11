"""
V5: Basin Hopping with SLSQP polish + Massive random restarts

Key insight from V1-V4: gradient-based approaches all converge to the same
2.636 basin. Need to explore the landscape at a coarser level.

Approach:
1. Basin Hopping: perturb -> SLSQP polish -> Metropolis accept/reject
   This treats each basin as a single point and hops between them.
2. Diverse perturbations designed to change the contact graph topology
3. Many random restarts with different structural templates
4. Focus on n=26 with completely different circle size distributions
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, dual_annealing

import functools
print = functools.partial(print, flush=True)

WORKDIR = Path(__file__).parent
N = 26
SEED = 42


def check_feasibility(params, n=N, tol=1e-10):
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


def polish_slsqp(params_np, n=N, ftol=1e-15, maxiter=15000):
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


def save_solution(params, path, n=N):
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
    return best_pos, best_r


def get_contact_graph(params, n=N, tol=1e-6):
    """Return the contact graph as a set of (i,j) pairs."""
    x, y, r = params[:n], params[n:2*n], params[2*n:]
    contacts = set()
    for i in range(n):
        # Wall contacts
        if abs(x[i] - r[i]) < tol:
            contacts.add(('wall_left', i))
        if abs(x[i] + r[i] - 1.0) < tol:
            contacts.add(('wall_right', i))
        if abs(y[i] - r[i]) < tol:
            contacts.add(('wall_bottom', i))
        if abs(y[i] + r[i] - 1.0) < tol:
            contacts.add(('wall_top', i))
        # Circle contacts
        for j in range(i+1, n):
            dist = math.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if abs(dist - r[i] - r[j]) < tol:
                contacts.add((i, j))
    return contacts


def topology_signature(params, n=N, tol=1e-6):
    """Create a topology signature based on sorted radius values and contact counts."""
    x, y, r = params[:n], params[n:2*n], params[2*n:]

    # Count contacts per circle
    contact_counts = [0] * n
    for i in range(n):
        if abs(x[i] - r[i]) < tol: contact_counts[i] += 1
        if abs(x[i] + r[i] - 1.0) < tol: contact_counts[i] += 1
        if abs(y[i] - r[i]) < tol: contact_counts[i] += 1
        if abs(y[i] + r[i] - 1.0) < tol: contact_counts[i] += 1
        for j in range(i+1, n):
            dist = math.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if abs(dist - r[i] - r[j]) < tol:
                contact_counts[i] += 1
                contact_counts[j] += 1

    # Signature: sorted (radius, contact_count) pairs
    pairs = sorted(zip(np.round(r, 4), contact_counts), reverse=True)
    return tuple(pairs)


# ============================================================
# Perturbation strategies for basin hopping
# ============================================================

def perturb_teleport(params, rng, n_circles=3):
    """Teleport random circles to new positions, recompute radii."""
    p = params.copy()
    idx = rng.choice(N, size=n_circles, replace=False)
    for k in idx:
        p[k] = rng.uniform(0.05, 0.95)
        p[N+k] = rng.uniform(0.05, 0.95)
        # Set radius to fit
        circles = [(p[i], p[N+i], p[2*N+i]) for i in range(N) if i != k]
        p[2*N+k] = max_radius_at_point(p[k], p[N+k], circles) * 0.8
        p[2*N+k] = max(p[2*N+k], 0.005)
    return p


def perturb_swap_sizes(params, rng, n_swaps=3):
    """Swap sizes between distant circles."""
    p = params.copy()
    for _ in range(n_swaps):
        i, j = rng.choice(N, size=2, replace=False)
        p[2*N+i], p[2*N+j] = p[2*N+j], p[2*N+i]
    return p


def perturb_cluster_move(params, rng):
    """Move a cluster of nearby circles together."""
    p = params.copy()
    # Pick a random circle as cluster center
    center = rng.randint(N)
    cx, cy = p[center], p[N+center]

    # Find circles within distance 0.3
    cluster = []
    for i in range(N):
        dist = math.sqrt((p[i] - cx)**2 + (p[N+i] - cy)**2)
        if dist < 0.3:
            cluster.append(i)

    if len(cluster) < 2:
        return p

    # Move the whole cluster
    dx = rng.normal(0, 0.15)
    dy = rng.normal(0, 0.15)
    for i in cluster:
        p[i] = np.clip(p[i] + dx, p[2*N+i] + 0.001, 1.0 - p[2*N+i] - 0.001)
        p[N+i] = np.clip(p[N+i] + dy, p[2*N+i] + 0.001, 1.0 - p[2*N+i] - 0.001)

    return p


def perturb_redistribute(params, rng):
    """Completely redistribute sizes with a different distribution."""
    p = params.copy()
    total = np.sum(p[2*N:])

    strategy = rng.randint(5)
    if strategy == 0:
        # Uniform
        r = np.ones(N) / N * total * (1 + rng.normal(0, 0.1, N))
    elif strategy == 1:
        # 3 large + rest small
        r = np.ones(N) * 0.05
        big = rng.choice(N, size=3, replace=False)
        r[big] = 0.15
        r = r / np.sum(r) * total
    elif strategy == 2:
        # 5 large + rest small
        r = np.ones(N) * 0.04
        big = rng.choice(N, size=5, replace=False)
        r[big] = 0.14
        r = r / np.sum(r) * total
    elif strategy == 3:
        # Dirichlet with concentration
        r = rng.dirichlet(np.ones(N) * rng.uniform(0.3, 2.0)) * total
    else:
        # Two size classes
        r = np.where(rng.random(N) < 0.4, 0.12, 0.07)
        r = r / np.sum(r) * total

    r = np.clip(r, 0.01, 0.25)
    p[2*N:] = r
    return p


def perturb_mirror(params, rng):
    """Mirror a subset of circles."""
    p = params.copy()
    n_mirror = rng.randint(N//4, 3*N//4)
    idx = rng.choice(N, size=n_mirror, replace=False)

    mirror_type = rng.randint(6)
    if mirror_type == 0:
        p[idx] = 1.0 - p[idx]  # flip x
    elif mirror_type == 1:
        p[N+idx] = 1.0 - p[N+idx]  # flip y
    elif mirror_type == 2:
        # Swap x,y for subset
        tmp = p[idx].copy()
        p[idx] = p[N+idx]
        p[N+idx] = tmp
    elif mirror_type == 3:
        # 90-degree rotation of subset around center
        cx, cy = 0.5, 0.5
        for i in idx:
            dx, dy = p[i] - cx, p[N+i] - cy
            p[i] = cx - dy
            p[N+i] = cy + dx
    elif mirror_type == 4:
        # 180-degree rotation of subset
        for i in idx:
            p[i] = 1.0 - p[i]
            p[N+i] = 1.0 - p[N+i]
    else:
        # Random rotation of subset
        angle = rng.uniform(0, 2*math.pi)
        cx, cy = 0.5, 0.5
        for i in idx:
            dx, dy = p[i] - cx, p[N+i] - cy
            p[i] = cx + dx*math.cos(angle) - dy*math.sin(angle)
            p[N+i] = cy + dx*math.sin(angle) + dy*math.cos(angle)

    # Clamp positions
    for i in range(N):
        ri = max(p[2*N+i], 0.01)
        p[i] = np.clip(p[i], ri + 0.001, 1.0 - ri - 0.001)
        p[N+i] = np.clip(p[N+i], ri + 0.001, 1.0 - ri - 0.001)

    return p


def perturb_remove_rebuild(params, rng, n_remove=4):
    """Remove circles and rebuild them greedily."""
    p = params.copy()
    remove_idx = rng.choice(N, size=n_remove, replace=False)

    # Keep circles
    remaining = [(p[i], p[N+i], p[2*N+i]) for i in range(N) if i not in remove_idx]

    # Rebuild removed circles in gaps
    for idx in remove_idx:
        pos, gap_r = find_best_placement(remaining, grid_res=40)
        new_r = max(gap_r * 0.85, 0.005)
        remaining.append((pos[0], pos[1], new_r))
        p[idx] = pos[0]
        p[N+idx] = pos[1]
        p[2*N+idx] = new_r

    return p


# ============================================================
# Greedy constructive from scratch with different templates
# ============================================================

def construct_from_template(template_name, rng):
    """Build n=26 packing from a structural template."""
    circles = []

    if template_name == "big_center":
        # 1 big center + ring of medium + outer small
        circles.append((0.5, 0.5, 0.18))
        for i in range(6):
            a = 2*math.pi*i/6
            circles.append((0.5+0.3*math.cos(a), 0.5+0.3*math.sin(a), 0.11))
        # Fill rest greedily

    elif template_name == "four_big":
        # 4 big circles near center + smaller ones around
        offsets = [(0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7)]
        for ox, oy in offsets:
            circles.append((ox, oy, 0.14))

    elif template_name == "diagonal":
        # Circles along diagonal with varying sizes
        for i in range(6):
            t = (i + 0.5) / 6
            circles.append((t, t, 0.12 - abs(t-0.5)*0.08))
        for i in range(6):
            t = (i + 0.5) / 6
            circles.append((t, 1-t, 0.10 - abs(t-0.5)*0.06))

    elif template_name == "grid_3x3":
        # 3x3 grid of medium circles + smaller in gaps
        for i in range(3):
            for j in range(3):
                cx = 0.2 + i * 0.3
                cy = 0.2 + j * 0.3
                circles.append((cx, cy, 0.12))

    elif template_name == "hex_tight":
        # Hexagonal close packing
        r0 = 0.09
        dy = r0 * math.sqrt(3)
        for row in range(6):
            for col in range(6):
                cx = r0 + col * 2 * r0 + (r0 if row % 2 else 0)
                cy = r0 + row * dy
                if r0 < cx < 1-r0 and r0 < cy < 1-r0:
                    circles.append((cx, cy, r0 * (1 + rng.normal(0, 0.05))))

    elif template_name == "concentric_3":
        # 3 concentric rings
        circles.append((0.5, 0.5, 0.13))
        for i in range(5):
            a = 2*math.pi*i/5
            circles.append((0.5+0.22*math.cos(a), 0.5+0.22*math.sin(a), 0.10))
        for i in range(10):
            a = 2*math.pi*i/10 + math.pi/10
            circles.append((0.5+0.38*math.cos(a), 0.5+0.38*math.sin(a), 0.08))

    elif template_name == "two_clusters":
        # Two clusters of circles
        for i in range(7):
            a = 2*math.pi*i/7
            r_ring = 0.15
            circles.append((0.3+r_ring*math.cos(a), 0.5+r_ring*math.sin(a), 0.09))
        for i in range(7):
            a = 2*math.pi*i/7
            r_ring = 0.15
            circles.append((0.7+r_ring*math.cos(a), 0.5+r_ring*math.sin(a), 0.09))

    elif template_name == "edge_heavy":
        # Many circles along edges, few in center
        # Top and bottom edges
        for i in range(5):
            t = (i + 0.5) / 5
            circles.append((t, 0.08, 0.08))
            circles.append((t, 0.92, 0.08))
        # Left and right edges
        for i in range(3):
            t = (i + 1) / 4
            circles.append((0.08, t, 0.08))
            circles.append((0.92, t, 0.08))
        # Center
        circles.append((0.5, 0.5, 0.15))
        circles.append((0.35, 0.35, 0.10))
        circles.append((0.65, 0.65, 0.10))

    elif template_name == "spiral":
        # Spiral arrangement
        for i in range(N):
            t = i / N
            angle = t * 4 * math.pi
            r_spiral = 0.05 + t * 0.35
            cx = 0.5 + r_spiral * math.cos(angle)
            cy = 0.5 + r_spiral * math.sin(angle)
            cx = np.clip(cx, 0.05, 0.95)
            cy = np.clip(cy, 0.05, 0.95)
            circles.append((cx, cy, 0.08 - t * 0.03))

    elif template_name == "random_power":
        # Random positions with power-law radii
        radii = rng.power(3.0, N) * 0.15 + 0.02
        for i in range(N):
            cx = rng.uniform(0.1, 0.9)
            cy = rng.uniform(0.1, 0.9)
            circles.append((cx, cy, radii[i]))

    # Pad to N if needed
    while len(circles) < N:
        cx = rng.uniform(0.1, 0.9)
        cy = rng.uniform(0.1, 0.9)
        # Find max radius at this position
        r = max_radius_at_point(cx, cy, circles)
        circles.append((cx, cy, max(r * 0.7, 0.01)))

    circles = circles[:N]
    x = np.array([c[0] for c in circles])
    y = np.array([c[1] for c in circles])
    r = np.array([c[2] for c in circles])
    return np.concatenate([x, y, r])


# ============================================================
# Basin Hopping
# ============================================================

def basin_hopping(base_params, n_hops=50, temp=0.01, seed=42):
    """
    Basin hopping: perturb -> polish -> accept/reject.
    Each hop fully optimizes with SLSQP, so we compare local optima.
    """
    print(f"  [BH] Basin hopping: {n_hops} hops, temp={temp}")
    rng = np.random.RandomState(seed)

    current = base_params.copy()
    current_metric = float(np.sum(current[2*N:]))
    viol = check_feasibility(current)
    if viol > 1e-10:
        current, current_metric = polish_slsqp(current)

    best_metric = current_metric
    best_params = current.copy()

    perturbations = [
        ("teleport2", lambda p, rng: perturb_teleport(p, rng, 2)),
        ("teleport4", lambda p, rng: perturb_teleport(p, rng, 4)),
        ("teleport6", lambda p, rng: perturb_teleport(p, rng, 6)),
        ("swap2", lambda p, rng: perturb_swap_sizes(p, rng, 2)),
        ("swap4", lambda p, rng: perturb_swap_sizes(p, rng, 4)),
        ("cluster", perturb_cluster_move),
        ("redist", perturb_redistribute),
        ("mirror", perturb_mirror),
        ("rebuild3", lambda p, rng: perturb_remove_rebuild(p, rng, 3)),
        ("rebuild5", lambda p, rng: perturb_remove_rebuild(p, rng, 5)),
        ("rebuild8", lambda p, rng: perturb_remove_rebuild(p, rng, 8)),
    ]

    n_accepted = 0
    topologies_seen = set()

    for hop in range(n_hops):
        # Pick random perturbation
        pname, pfn = perturbations[rng.randint(len(perturbations))]

        # Perturb
        candidate = pfn(current, rng)
        candidate[2*N:] = np.maximum(candidate[2*N:], 0.005)

        # Polish with SLSQP
        try:
            polished, pol_metric = polish_slsqp(candidate, ftol=1e-14)
            viol = check_feasibility(polished)
        except Exception:
            continue

        if viol > 1e-10:
            # Infeasible after polish, skip
            if hop % 10 == 0:
                print(f"    hop {hop}: {pname} -> infeasible (viol={viol:.2e})")
            continue

        # Check topology
        topo = topology_signature(polished)
        is_new_topo = topo not in topologies_seen
        topologies_seen.add(topo)

        # Metropolis acceptance
        delta = pol_metric - current_metric
        if delta > 0 or rng.random() < math.exp(delta / temp):
            current = polished
            current_metric = pol_metric
            n_accepted += 1

            if pol_metric > best_metric:
                best_metric = pol_metric
                best_params = polished.copy()
                print(f"    hop {hop}: {pname} -> {pol_metric:.10f} NEW BEST"
                      f" {'(new topo!)' if is_new_topo else ''}")
            elif hop % 10 == 0 or is_new_topo:
                print(f"    hop {hop}: {pname} -> {pol_metric:.10f}"
                      f" accepted {'(new topo!)' if is_new_topo else ''}")
        elif hop % 10 == 0:
            print(f"    hop {hop}: {pname} -> {pol_metric:.10f} rejected")

    print(f"  [BH] Accept rate: {n_accepted}/{n_hops}, "
          f"unique topologies: {len(topologies_seen)}")

    return best_params, best_metric


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("V5: Basin Hopping + Template Restarts")
    print("=" * 60)

    best_known_path = WORKDIR / "solution_n26.json"
    if not best_known_path.exists():
        best_known_path = WORKDIR.parent / "topo-001" / "solution_n26.json"

    base_params = load_solution(str(best_known_path))
    base_metric = float(np.sum(base_params[2*N:]))
    print(f"Base metric: {base_metric:.10f}")

    overall_best_metric = base_metric
    overall_best_params = base_params.copy()
    all_results = []

    # ---- Part 1: Basin Hopping from known best ----
    print("\n--- BASIN HOPPING (from known best) ---")
    for trial in range(3):
        seed = 500000 + trial * 1337
        temp = [0.005, 0.01, 0.02][trial]

        bh_p, bh_m = basin_hopping(base_params, n_hops=40, temp=temp, seed=seed)
        tag = f"BH_t{trial}_T{temp}"
        all_results.append((tag, bh_m))
        if bh_m > overall_best_metric:
            overall_best_metric = bh_m
            overall_best_params = bh_p
            print(f"  [{tag}] NEW BEST: {bh_m:.10f}")
        else:
            print(f"  [{tag}] best={bh_m:.10f}")

    # ---- Part 2: Template restarts ----
    print("\n--- TEMPLATE RESTARTS ---")
    templates = ["big_center", "four_big", "diagonal", "grid_3x3", "hex_tight",
                 "concentric_3", "two_clusters", "edge_heavy", "spiral", "random_power"]

    for tname in templates:
        for trial in range(3):
            seed = 600000 + hash(tname) % 10000 + trial
            rng = np.random.RandomState(seed)

            params = construct_from_template(tname, rng)

            # Polish directly with SLSQP
            try:
                polished, pol_metric = polish_slsqp(params, ftol=1e-14)
                viol = check_feasibility(polished)

                if viol < 1e-10:
                    tag = f"TPL_{tname}_t{trial}"
                    all_results.append((tag, pol_metric))
                    if pol_metric > overall_best_metric:
                        overall_best_metric = pol_metric
                        overall_best_params = polished
                        print(f"  [{tag}] {pol_metric:.10f} NEW BEST")
                    elif trial == 0:
                        print(f"  [{tag}] {pol_metric:.10f}")
                else:
                    if trial == 0:
                        print(f"  [TPL_{tname}_t{trial}] infeasible (viol={viol:.2e})")
            except Exception as e:
                if trial == 0:
                    print(f"  [TPL_{tname}_t{trial}] error: {e}")

    # ---- Part 3: Basin hopping from best templates ----
    # Find the best result that isn't from the known basin
    print("\n--- BASIN HOPPING (from best template) ---")
    results_sorted = sorted(all_results, key=lambda x: -x[1])

    # Try BH from the top 3 non-base-metric results
    bh_seeds_tried = 0
    for tag, metric in results_sorted:
        if metric < base_metric - 0.01 and metric > 2.5 and "TPL" in tag:
            # This is a different basin - try BH from here
            # We need to reconstruct the params... use the template
            tname = tag.split("_")[1]
            trial_idx = int(tag.split("_t")[-1])
            seed_tpl = 600000 + hash(tname) % 10000 + trial_idx
            rng = np.random.RandomState(seed_tpl)
            params = construct_from_template(tname, rng)
            polished, _ = polish_slsqp(params)

            bh_p, bh_m = basin_hopping(polished, n_hops=30, temp=0.01,
                                        seed=700000 + bh_seeds_tried)
            tag2 = f"BH_from_{tname}"
            all_results.append((tag2, bh_m))
            if bh_m > overall_best_metric:
                overall_best_metric = bh_m
                overall_best_params = bh_p
                print(f"  [{tag2}] NEW BEST: {bh_m:.10f}")
            else:
                print(f"  [{tag2}] best={bh_m:.10f}")

            bh_seeds_tried += 1
            if bh_seeds_tried >= 3:
                break

    # ---- Report ----
    print("\n" + "=" * 60)
    print(f"OVERALL BEST: {overall_best_metric:.10f}")
    print(f"Base metric:  {base_metric:.10f}")
    print(f"Improvement:  {overall_best_metric - base_metric:.2e}")
    print("=" * 60)

    if overall_best_params is not None:
        viol = check_feasibility(overall_best_params)
        if viol < 1e-10:
            save_solution(overall_best_params, str(WORKDIR / "solution_n26_v5.json"))
            print(f"Saved solution_n26_v5.json (viol={viol:.2e})")
            if overall_best_metric > base_metric:
                save_solution(overall_best_params, str(WORKDIR / "solution_n26.json"))
                print("NEW RECORD!")

    results_sorted = sorted(all_results, key=lambda x: -x[1])
    with open(str(WORKDIR / "results_v5.json"), 'w') as f:
        json.dump(results_sorted, f, indent=2)

    print("\nTop 15:")
    for tag, m in results_sorted[:15]:
        print(f"  {tag}: {m:.10f}")


if __name__ == "__main__":
    main()
