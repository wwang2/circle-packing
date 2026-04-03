"""CMA-ES v4: Topology perturbation + bipop restarts.

New strategies:
1. Swap circles (permute positions) and re-optimize
2. Remove smallest circle, re-optimize remaining 25, then re-insert
3. Bipop-CMA-ES with automatic restarts
4. Multi-start with structured initializations (hex grid, concentric, etc.)
"""

import json
import math
import sys
import os
import numpy as np
import cma
from scipy.optimize import minimize, linprog

WORKDIR = os.path.dirname(os.path.abspath(__file__))

def log(msg):
    print(msg, flush=True)

def load_solution(path):
    with open(path) as f:
        data = json.load(f)
    circles = data.get("circles", data)
    return np.array(circles)

def save_solution(circles, path):
    data = {"circles": [[float(x), float(y), float(r)] for x, y, r in circles]}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def compute_violation(circles):
    n = len(circles)
    total_viol = 0.0
    max_viol = 0.0
    for i in range(n):
        x, y, r = circles[i]
        for v in [r - x, x + r - 1.0, r - y, y + r - 1.0, -r]:
            if v > 0:
                total_viol += v
                max_viol = max(max_viol, v)
    for i in range(n):
        xi, yi, ri = circles[i]
        for j in range(i + 1, n):
            xj, yj, rj = circles[j]
            dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)
            overlap = (ri + rj) - dist
            if overlap > 0:
                total_viol += overlap
                max_viol = max(max_viol, overlap)
    return total_viol, max_viol

def optimal_radii_lp(positions):
    """LP to find optimal radii given positions."""
    n = len(positions)
    c = -np.ones(n)
    A_rows, b_rows = [], []
    for i in range(n):
        for j in range(i + 1, n):
            dist = math.sqrt((positions[i][0] - positions[j][0])**2 +
                           (positions[i][1] - positions[j][1])**2)
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_rows.append(row)
            b_rows.append(dist)
    for i in range(n):
        xi, yi = positions[i]
        row = np.zeros(n)
        row[i] = 1.0
        for bound in [xi, 1.0 - xi, yi, 1.0 - yi]:
            A_rows.append(row.copy())
            b_rows.append(bound)
    result = linprog(c, A_ub=np.array(A_rows), b_ub=np.array(b_rows),
                     bounds=[(0, None) for _ in range(n)], method='highs')
    if result.success:
        return result.x, -result.fun
    return np.zeros(n), 0.0

def slsqp_polish(circles, maxiter=5000):
    n = len(circles)
    x0 = circles.flatten()
    def neg_sum_radii(x):
        return -np.sum(x.reshape(n, 3)[:, 2])
    constraints = []
    for i in range(n):
        ri = 3 * i + 2; xi = 3 * i; yi = 3 * i + 1
        constraints.append({"type": "ineq", "fun": lambda x, idx=ri: x[idx] - 1e-12})
        constraints.append({"type": "ineq", "fun": lambda x, ix=xi, ir=ri: x[ix] - x[ir]})
        constraints.append({"type": "ineq", "fun": lambda x, ix=xi, ir=ri: 1.0 - x[ix] - x[ir]})
        constraints.append({"type": "ineq", "fun": lambda x, iy=yi, ir=ri: x[iy] - x[ir]})
        constraints.append({"type": "ineq", "fun": lambda x, iy=yi, ir=ri: 1.0 - x[iy] - x[ir]})
    for i in range(n):
        for j in range(i + 1, n):
            def overlap_con(x, ii=i, jj=j):
                xi, yi, ri = x[3*ii], x[3*ii+1], x[3*ii+2]
                xj, yj, rj = x[3*jj], x[3*jj+1], x[3*jj+2]
                return math.sqrt((xi-xj)**2 + (yi-yj)**2) - ri - rj
            constraints.append({"type": "ineq", "fun": overlap_con})
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
    res = minimize(neg_sum_radii, x0, method="SLSQP", constraints=constraints,
                   bounds=bounds, options={"maxiter": maxiter, "ftol": 1e-15, "disp": False})
    result_circles = res.x.reshape(n, 3)
    viol, max_v = compute_violation(result_circles)
    sr = np.sum(result_circles[:, 2])
    return result_circles, sr, viol, max_v

def generate_hex_grid(n, jitter=0.0, rng=None):
    """Generate hex grid positions for n circles."""
    if rng is None:
        rng = np.random.RandomState(42)
    # Estimate grid dimensions
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    positions = []
    for r in range(rows + 2):
        for c in range(cols + 2):
            x = (c + 0.5 * (r % 2)) / (cols + 1)
            y = r / (rows + 1)
            if 0.02 < x < 0.98 and 0.02 < y < 0.98:
                positions.append([x, y])
    positions = np.array(positions[:n])
    if len(positions) < n:
        extra = rng.uniform(0.05, 0.95, size=(n - len(positions), 2))
        positions = np.vstack([positions, extra])
    if jitter > 0:
        positions += rng.normal(0, jitter, positions.shape)
        positions = np.clip(positions, 0.02, 0.98)
    return positions[:n]

def generate_concentric(n, rng=None):
    """Concentric ring arrangement."""
    if rng is None:
        rng = np.random.RandomState(42)
    positions = [[0.5, 0.5]]  # center
    remaining = n - 1
    ring = 1
    while remaining > 0:
        radius = 0.15 * ring
        count = min(remaining, max(4, int(6 * ring)))
        for i in range(count):
            angle = 2 * np.pi * i / count + rng.normal(0, 0.05)
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            positions.append([np.clip(x, 0.02, 0.98), np.clip(y, 0.02, 0.98)])
            remaining -= 1
            if remaining <= 0:
                break
        ring += 1
    return np.array(positions[:n])

def swap_and_polish(circles, n_swaps=50, seed=77777):
    """Swap pairs of circles and re-polish."""
    rng = np.random.RandomState(seed)
    best_sr = np.sum(circles[:, 2])
    best_circles = circles.copy()
    n = len(circles)

    for attempt in range(n_swaps):
        swapped = best_circles.copy()
        # Swap 2-4 circles
        n_swap = rng.randint(2, min(5, n))
        indices = rng.choice(n, n_swap, replace=False)
        # Circular permutation of positions
        positions = swapped[indices, :2].copy()
        np.random.shuffle(positions)
        swapped[indices, :2] = positions
        # Recompute optimal radii
        radii, sr = optimal_radii_lp(swapped[:, :2])
        swapped[:, 2] = radii

        pol_c, pol_sr, pol_v, pol_mv = slsqp_polish(swapped, maxiter=3000)
        if pol_mv < 1e-10 and pol_sr > best_sr:
            log(f"  Swap {attempt}: IMPROVED {best_sr:.10f} -> {pol_sr:.10f}")
            best_sr = pol_sr
            best_circles = pol_c.copy()
        elif attempt % 10 == 0:
            log(f"  Swap {attempt}: sr={pol_sr:.10f}, max_v={pol_mv:.2e} (best={best_sr:.10f})")

    return best_circles, best_sr

def remove_reinsert(circles, seed=66666):
    """Remove smallest circles one at a time, re-optimize, then reinsert."""
    rng = np.random.RandomState(seed)
    best_sr = np.sum(circles[:, 2])
    best_circles = circles.copy()
    n = len(circles)

    # Sort by radius (smallest first)
    sorted_idx = np.argsort(circles[:, 2])

    for k in range(min(5, n)):
        idx = sorted_idx[k]
        log(f"  Removing circle {idx} (r={circles[idx, 2]:.6f})")

        # Remove circle, optimize remaining
        remaining = np.delete(circles, idx, axis=0)
        pol_c, pol_sr, pol_v, pol_mv = slsqp_polish(remaining, maxiter=3000)
        log(f"    {n-1} circles: sr={pol_sr:.10f}")

        # Find best gap to insert new circle
        best_insert = None
        best_insert_sr = 0

        for trial in range(20):
            # Try random position
            x = rng.uniform(0.02, 0.98)
            y = rng.uniform(0.02, 0.98)
            # Find max radius at this position
            max_r = min(x, 1 - x, y, 1 - y)
            for j in range(n - 1):
                dist = math.sqrt((x - pol_c[j, 0])**2 + (y - pol_c[j, 1])**2)
                max_r = min(max_r, dist - pol_c[j, 2])
            if max_r > 0.005:
                new_circles = np.vstack([pol_c, [[x, y, max_r * 0.9]]])
                p_c, p_sr, p_v, p_mv = slsqp_polish(new_circles, maxiter=3000)
                if p_mv < 1e-10 and p_sr > best_insert_sr:
                    best_insert_sr = p_sr
                    best_insert = p_c.copy()

        if best_insert is not None and best_insert_sr > best_sr:
            log(f"    Reinserted: {best_sr:.10f} -> {best_insert_sr:.10f}")
            best_sr = best_insert_sr
            best_circles = best_insert
        else:
            log(f"    No improvement from reinsertion (best_insert_sr={best_insert_sr:.10f})")

    return best_circles, best_sr

def multi_start_structured(n, n_starts=10, seed=11111):
    """Multi-start from structured initializations."""
    rng = np.random.RandomState(seed)
    best_sr = 0.0
    best_circles = None

    for trial in range(n_starts):
        # Pick initialization type
        init_type = trial % 4
        if init_type == 0:
            positions = generate_hex_grid(n, jitter=0.02 * (trial + 1), rng=rng)
        elif init_type == 1:
            positions = generate_concentric(n, rng=rng)
        elif init_type == 2:
            positions = rng.uniform(0.05, 0.95, size=(n, 2))
        else:
            # Diagonal symmetric
            half = n // 2
            pos1 = rng.uniform(0.05, 0.95, size=(half, 2))
            pos2 = 1.0 - pos1  # mirror
            if n % 2 == 1:
                positions = np.vstack([pos1, pos2, [[0.5, 0.5]]])
            else:
                positions = np.vstack([pos1, pos2])
            positions = positions[:n]

        radii, lp_sr = optimal_radii_lp(positions)
        circles = np.column_stack([positions, radii])

        pol_c, pol_sr, pol_v, pol_mv = slsqp_polish(circles, maxiter=5000)
        if pol_mv < 1e-10 and pol_sr > best_sr:
            best_sr = pol_sr
            best_circles = pol_c.copy()
            log(f"  Start {trial} ({['hex','concentric','random','symmetric'][init_type]}): "
                f"sr={pol_sr:.10f} ** NEW BEST **")
        elif trial % 3 == 0:
            log(f"  Start {trial} ({['hex','concentric','random','symmetric'][init_type]}): "
                f"sr={pol_sr:.10f}, max_v={pol_mv:.2e}")

    return best_circles, best_sr

def main():
    parent_path = os.path.join(WORKDIR, "..", "sa-001", "solution_n26.json")
    init_circles = load_solution(parent_path)
    n = len(init_circles)
    init_sr = np.sum(init_circles[:, 2])

    cur_path = os.path.join(WORKDIR, "solution_n26.json")
    if os.path.exists(cur_path):
        cur_circles = load_solution(cur_path)
        cur_sr = np.sum(cur_circles[:, 2])
        viol, mv = compute_violation(cur_circles)
        if mv < 1e-10 and cur_sr >= init_sr:
            init_circles = cur_circles
            init_sr = cur_sr
    log(f"Starting from: n={n}, sum_radii={init_sr:.10f}")

    best_circles = init_circles.copy()
    best_sr = init_sr
    no_improve_phases = 0

    # === Phase 1: Swap topology ===
    log("\n=== Phase 1: Swap topology ===")
    sw_c, sw_sr = swap_and_polish(best_circles, n_swaps=40, seed=77777)
    if sw_sr > best_sr:
        best_sr = sw_sr
        best_circles = sw_c
        save_solution(best_circles, cur_path)
        no_improve_phases = 0
    else:
        no_improve_phases += 1
    log(f"  Phase 1 result: {sw_sr:.10f} (best={best_sr:.10f})")

    # === Phase 2: Remove-reinsert ===
    log("\n=== Phase 2: Remove-reinsert ===")
    rr_c, rr_sr = remove_reinsert(best_circles, seed=66666)
    if rr_sr > best_sr:
        best_sr = rr_sr
        best_circles = rr_c
        save_solution(best_circles, cur_path)
        no_improve_phases = 0
    else:
        no_improve_phases += 1
    log(f"  Phase 2 result: {rr_sr:.10f} (best={best_sr:.10f})")

    # === Phase 3: Multi-start structured ===
    log("\n=== Phase 3: Multi-start structured ===")
    ms_c, ms_sr = multi_start_structured(n, n_starts=15, seed=11111)
    if ms_c is not None and ms_sr > best_sr:
        best_sr = ms_sr
        best_circles = ms_c
        save_solution(best_circles, cur_path)
        no_improve_phases = 0
    else:
        no_improve_phases += 1
    log(f"  Phase 3 result: {ms_sr:.10f} (best={best_sr:.10f})")

    # === Phase 4: Aggressive perturbation with multiple polish rounds ===
    log("\n=== Phase 4: Multi-round perturbation ===")
    rng = np.random.RandomState(99999)
    for attempt in range(30):
        perturbed = best_circles.copy()
        # Choose perturbation strategy
        strategy = attempt % 3
        if strategy == 0:
            # Small perturbation to all
            perturbed[:, :2] += rng.normal(0, 0.005, size=(n, 2))
            perturbed[:, 2] *= (1 + rng.normal(0, 0.002, size=n))
        elif strategy == 1:
            # Large perturbation to few
            k = rng.randint(1, 4)
            idx = rng.choice(n, k, replace=False)
            perturbed[idx, :2] += rng.normal(0, 0.05, size=(k, 2))
        else:
            # Swap + perturb
            i, j = rng.choice(n, 2, replace=False)
            perturbed[i, :2], perturbed[j, :2] = perturbed[j, :2].copy(), perturbed[i, :2].copy()
            perturbed[:, :2] += rng.normal(0, 0.003, size=(n, 2))

        perturbed[:, 2] = np.clip(perturbed[:, 2], 0.005, 0.495)
        perturbed[:, 0] = np.clip(perturbed[:, 0], perturbed[:, 2], 1 - perturbed[:, 2])
        perturbed[:, 1] = np.clip(perturbed[:, 1], perturbed[:, 2], 1 - perturbed[:, 2])

        # Double polish
        pol_c, pol_sr, pol_v, pol_mv = slsqp_polish(perturbed, maxiter=3000)
        if pol_mv < 1e-10 and pol_sr > best_sr * 0.999:
            pol_c2, pol_sr2, pol_v2, pol_mv2 = slsqp_polish(pol_c, maxiter=5000)
            if pol_mv2 < 1e-10 and pol_sr2 > best_sr:
                log(f"  Perturb {attempt}: IMPROVED {best_sr:.10f} -> {pol_sr2:.10f}")
                best_sr = pol_sr2
                best_circles = pol_c2
                save_solution(best_circles, cur_path)
        if attempt % 10 == 0:
            log(f"  Perturb {attempt}: best={best_sr:.10f}")

    # === Final ===
    log(f"\n{'='*60}")
    log(f"FINAL: sum_radii = {best_sr:.10f}")
    log(f"Parent:           {np.sum(load_solution(parent_path)[:, 2]):.10f}")
    log(f"Delta:            {best_sr - np.sum(load_solution(parent_path)[:, 2]):+.12f}")
    viol, max_v = compute_violation(best_circles)
    log(f"Validation: total_viol={viol:.2e}, max_viol={max_v:.2e}")
    save_solution(best_circles, cur_path)
    log(f"Saved: {cur_path}")

if __name__ == "__main__":
    main()
