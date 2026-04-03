"""
v9: Discrete topology enumeration.

The KEY idea: instead of continuous optimization, enumerate SPECIFIC contact
graph topologies and solve the KKT system for each.

For a rigid packing with N=26:
- 3*N = 78 variables (x, y, r for each circle)
- Need exactly 78 active constraints for zero DOF
- Active constraints: circle-circle tangencies + wall contacts

The known topology has:
- 58 circle-circle contacts
- 20 wall contacts
- Total: 78

Strategy: Take the known topology, make SPECIFIC changes (swap contacts),
and solve the resulting system of 78 nonlinear equations.

If the system has a solution with all radii > 0 and all dual variables > 0,
we have found a valid local optimum.
"""

import json
import numpy as np
from scipy.optimize import fsolve
import os
import time
from itertools import combinations

WORKDIR = os.path.dirname(os.path.abspath(__file__))
N = 26


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


def extract_topology(x, y, r, tol=1e-6):
    """Extract active constraints from solution."""
    n = len(x)
    active = []
    for i in range(n):
        if abs(x[i] - r[i]) < tol: active.append(('L', i))
        if abs(1 - x[i] - r[i]) < tol: active.append(('R', i))
        if abs(y[i] - r[i]) < tol: active.append(('B', i))
        if abs(1 - y[i] - r[i]) < tol: active.append(('T', i))
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if abs(dist - r[i] - r[j]) < tol:
                active.append(('C', i, j))
    return active


def solve_topology(active_constraints, x0, y0, r0):
    """
    Given a set of active constraints (topology), solve the KKT system.

    The KKT system for a rigid packing:
    - Stationarity: grad_obj = sum(lambda_k * grad_g_k) for all k
    - Primal feasibility: g_k(x) = 0 for all active k
    - lambda_k > 0 for all k (checked after solving)
    """
    n = len(x0)
    n_active = len(active_constraints)
    n_vars = 3 * n
    n_total = n_vars + n_active

    if n_active != n_vars:
        # System is under/over-determined
        return None, 0, False

    def kkt_eqns(vars):
        x = vars[:n]
        y = vars[n:2*n]
        r = vars[2*n:3*n]
        lam = vars[3*n:]

        equations = np.zeros(n_total)
        grad_L = np.zeros(n_vars)
        grad_L[2*n:3*n] = -1.0

        for k, constraint in enumerate(active_constraints):
            ctype = constraint[0]
            if ctype == 'L':
                i = constraint[1]
                grad_L[i] -= lam[k]; grad_L[2*n+i] += lam[k]
            elif ctype == 'R':
                i = constraint[1]
                grad_L[i] += lam[k]; grad_L[2*n+i] += lam[k]
            elif ctype == 'B':
                i = constraint[1]
                grad_L[n+i] -= lam[k]; grad_L[2*n+i] += lam[k]
            elif ctype == 'T':
                i = constraint[1]
                grad_L[n+i] += lam[k]; grad_L[2*n+i] += lam[k]
            elif ctype == 'C':
                i, j = constraint[1], constraint[2]
                dx = x[i]-x[j]; dy = y[i]-y[j]; sr = r[i]+r[j]
                grad_L[i] -= lam[k]*2*dx; grad_L[j] += lam[k]*2*dx
                grad_L[n+i] -= lam[k]*2*dy; grad_L[n+j] += lam[k]*2*dy
                grad_L[2*n+i] += lam[k]*2*sr; grad_L[2*n+j] += lam[k]*2*sr

        equations[:n_vars] = grad_L

        for k, constraint in enumerate(active_constraints):
            ctype = constraint[0]
            if ctype == 'L':
                i = constraint[1]; equations[n_vars+k] = x[i] - r[i]
            elif ctype == 'R':
                i = constraint[1]; equations[n_vars+k] = 1 - x[i] - r[i]
            elif ctype == 'B':
                i = constraint[1]; equations[n_vars+k] = y[i] - r[i]
            elif ctype == 'T':
                i = constraint[1]; equations[n_vars+k] = 1 - y[i] - r[i]
            elif ctype == 'C':
                i, j = constraint[1], constraint[2]
                equations[n_vars+k] = (x[i]-x[j])**2 + (y[i]-y[j])**2 - (r[i]+r[j])**2

        return equations

    # Initial dual variables
    A = np.zeros((n_vars, n_active))
    for k, constraint in enumerate(active_constraints):
        ctype = constraint[0]
        if ctype == 'L':
            i = constraint[1]; A[i,k] = 1; A[2*n+i,k] = -1
        elif ctype == 'R':
            i = constraint[1]; A[i,k] = -1; A[2*n+i,k] = -1
        elif ctype == 'B':
            i = constraint[1]; A[n+i,k] = 1; A[2*n+i,k] = -1
        elif ctype == 'T':
            i = constraint[1]; A[n+i,k] = -1; A[2*n+i,k] = -1
        elif ctype == 'C':
            i, j = constraint[1], constraint[2]
            dx = x0[i]-x0[j]; dy = y0[i]-y0[j]; sr = r0[i]+r0[j]
            A[i,k] = 2*dx; A[j,k] = -2*dx
            A[n+i,k] = 2*dy; A[n+j,k] = -2*dy
            A[2*n+i,k] = -2*sr; A[2*n+j,k] = -2*sr

    grad_obj = np.zeros(n_vars)
    grad_obj[2*n:3*n] = -1.0

    try:
        lam0, _, _, _ = np.linalg.lstsq(A, grad_obj, rcond=None)
    except:
        lam0 = np.ones(n_active) * 0.1

    vars0 = np.concatenate([x0, y0, r0, lam0])

    try:
        result = fsolve(kkt_eqns, vars0, full_output=True, maxfev=5000)
        vars_sol = result[0]
        info = result[1]
        ier = result[2]

        x_sol = vars_sol[:n]
        y_sol = vars_sol[n:2*n]
        r_sol = vars_sol[2*n:3*n]
        lam_sol = vars_sol[3*n:]

        residual = np.linalg.norm(kkt_eqns(vars_sol))

        if residual > 1e-8:
            return None, 0, False

        # Check all radii positive
        if np.any(r_sol <= 0):
            return None, 0, False

        # Check all dual variables positive (or non-negative)
        if np.any(lam_sol < -1e-6):
            return None, 0, False

        metric = np.sum(r_sol)
        feasible = is_feasible(x_sol, y_sol, r_sol, tol=1e-8)

        return (x_sol, y_sol, r_sol), metric, feasible

    except Exception as e:
        return None, 0, False


def get_non_contacts(x, y, r, active):
    """Get pairs that are NOT in contact (potential new contacts)."""
    n = len(x)
    contact_set = set()
    for c in active:
        if c[0] == 'C':
            contact_set.add((c[1], c[2]))

    non_contacts = []
    for i in range(n):
        for j in range(i+1, n):
            if (i, j) not in contact_set:
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                gap = dist - r[i] - r[j]
                non_contacts.append((i, j, gap))

    return sorted(non_contacts, key=lambda x: x[2])


def get_wall_non_contacts(x, y, r, active):
    """Get wall contacts that are NOT active (potential new wall contacts)."""
    n = len(x)
    wall_set = set()
    for c in active:
        if c[0] in ('L', 'R', 'B', 'T'):
            wall_set.add((c[0], c[1]))

    non_walls = []
    for i in range(n):
        for wtype, gap_fn in [('L', lambda i: x[i]-r[i]),
                               ('R', lambda i: 1-x[i]-r[i]),
                               ('B', lambda i: y[i]-r[i]),
                               ('T', lambda i: 1-y[i]-r[i])]:
            if (wtype, i) not in wall_set:
                gap = gap_fn(i)
                non_walls.append((wtype, i, gap))

    return sorted(non_walls, key=lambda x: x[2])


def main():
    t0 = time.time()
    rng = np.random.RandomState(42)

    known_path = os.path.join(WORKDIR, '..', 'topo-001', 'solution_n26.json')
    xk, yk, rk = load_solution(known_path)
    known_metric = np.sum(rk)
    print(f"Known best: {known_metric:.15f}")

    active = extract_topology(xk, yk, rk)
    cc = [c for c in active if c[0] == 'C']
    wc = [c for c in active if c[0] in ('L', 'R', 'B', 'T')]
    print(f"Active: {len(cc)} circle-circle + {len(wc)} wall = {len(active)} total")

    # Non-contacts sorted by gap (closest first)
    non_cc = get_non_contacts(xk, yk, rk, active)
    non_wc = get_wall_non_contacts(xk, yk, rk, active)
    print(f"Non-contacts: {len(non_cc)} circle-circle, {len(non_wc)} wall")
    print(f"Closest non-contacts:")
    for i, j, gap in non_cc[:10]:
        print(f"  ({i:2d}, {j:2d}): gap={gap:.6f}")
    print(f"Closest wall non-contacts:")
    for wtype, i, gap in non_wc[:10]:
        print(f"  circle {i:2d} -> {wtype}: gap={gap:.6f}")

    best_metric = known_metric
    best_sol = (xk.copy(), yk.copy(), rk.copy())
    n_solved = 0
    n_tried = 0
    n_better = 0

    # ===== Strategy: Swap 1 contact =====
    # Remove one contact, add another, solve the new system
    print(f"\n{'='*60}")
    print("Swap 1 contact: remove + add")
    print(f"{'='*60}")

    # Try removing each circle-circle contact and adding the closest non-contact
    for remove_idx in range(len(cc)):
        for add_idx in range(min(5, len(non_cc))):
            n_tried += 1
            new_active = [c for idx, c in enumerate(active) if not (c[0] == 'C' and idx == len(wc) + remove_idx)]
            # Wait - indices are mixed. Let me fix this.
            break
        break

    # Build clean lists
    cc_list = [(c[1], c[2]) for c in cc]
    wc_list = [(c[0], c[1]) for c in wc]

    print(f"\nSwap 1 circle-circle contact:")
    for rm_idx in range(len(cc_list)):
        rm_pair = cc_list[rm_idx]
        remaining_cc = [p for idx, p in enumerate(cc_list) if idx != rm_idx]

        # Try adding 5 closest non-contacts
        for add_idx in range(min(5, len(non_cc))):
            add_pair = (non_cc[add_idx][0], non_cc[add_idx][1])
            if add_pair in remaining_cc:
                continue

            # Build new topology
            new_cc = remaining_cc + [add_pair]
            new_active = []
            for wtype, i in wc_list:
                new_active.append((wtype, i))
            for i, j in new_cc:
                new_active.append(('C', i, j))

            if len(new_active) != 78:
                continue

            n_tried += 1
            sol, metric, feasible = solve_topology(new_active, xk, yk, rk)

            if sol is not None:
                n_solved += 1
                if feasible and metric > best_metric:
                    n_better += 1
                    best_metric = metric
                    best_sol = sol
                    print(f"  *** NEW BEST: {metric:.15f} ***")
                    print(f"      Removed ({rm_pair[0]},{rm_pair[1]}), Added ({add_pair[0]},{add_pair[1]})")

        if (rm_idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{rm_idx+1}/{len(cc_list)}] tried={n_tried}, solved={n_solved}, "
                  f"better={n_better}, time={elapsed:.0f}s")

    # ===== Strategy: Swap 2 contacts simultaneously =====
    print(f"\n{'='*60}")
    print("Swap 2 contacts simultaneously")
    print(f"{'='*60}")

    # Random sampling of 2-contact swaps
    for attempt in range(500):
        # Remove 2 random contacts
        rm_indices = rng.choice(len(cc_list), 2, replace=False)
        remaining_cc = [p for idx, p in enumerate(cc_list) if idx not in rm_indices]

        # Add 2 random non-contacts (from closest 20)
        add_pool = [(nc[0], nc[1]) for nc in non_cc[:20] if (nc[0], nc[1]) not in remaining_cc]
        if len(add_pool) < 2:
            continue
        add_indices = rng.choice(len(add_pool), 2, replace=False)
        add_pairs = [add_pool[i] for i in add_indices]

        new_cc = remaining_cc + add_pairs
        new_active = []
        for wtype, i in wc_list:
            new_active.append((wtype, i))
        for i, j in new_cc:
            new_active.append(('C', i, j))

        if len(new_active) != 78:
            continue

        n_tried += 1
        sol, metric, feasible = solve_topology(new_active, xk, yk, rk)

        if sol is not None:
            n_solved += 1
            if feasible and metric > best_metric:
                n_better += 1
                best_metric = metric
                best_sol = sol
                print(f"  *** NEW BEST: {metric:.15f} (attempt {attempt+1}) ***")

        if (attempt + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{attempt+1}/500] tried={n_tried}, solved={n_solved}, "
                  f"better={n_better}, time={elapsed:.0f}s")

    # ===== Strategy: Swap 3 contacts =====
    print(f"\n{'='*60}")
    print("Swap 3 contacts simultaneously")
    print(f"{'='*60}")

    for attempt in range(500):
        rm_indices = rng.choice(len(cc_list), 3, replace=False)
        remaining_cc = [p for idx, p in enumerate(cc_list) if idx not in rm_indices]

        add_pool = [(nc[0], nc[1]) for nc in non_cc[:30] if (nc[0], nc[1]) not in remaining_cc]
        if len(add_pool) < 3:
            continue
        add_indices = rng.choice(len(add_pool), 3, replace=False)
        add_pairs = [add_pool[i] for i in add_indices]

        new_cc = remaining_cc + add_pairs
        new_active = []
        for wtype, i in wc_list:
            new_active.append((wtype, i))
        for i, j in new_cc:
            new_active.append(('C', i, j))

        if len(new_active) != 78:
            continue

        n_tried += 1
        sol, metric, feasible = solve_topology(new_active, xk, yk, rk)

        if sol is not None:
            n_solved += 1
            if feasible and metric > best_metric:
                n_better += 1
                best_metric = metric
                best_sol = sol
                print(f"  *** NEW BEST: {metric:.15f} (attempt {attempt+1}) ***")

        if (attempt + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{attempt+1}/500] tried={n_tried}, solved={n_solved}, "
                  f"better={n_better}, time={elapsed:.0f}s")

    # ===== Strategy: Also swap wall contacts =====
    print(f"\n{'='*60}")
    print("Swap wall + circle contacts")
    print(f"{'='*60}")

    for attempt in range(300):
        # Remove 1 wall + 1 circle contact, add 1 wall + 1 circle
        rm_cc_idx = rng.randint(len(cc_list))
        rm_wc_idx = rng.randint(len(wc_list))

        remaining_cc = [p for idx, p in enumerate(cc_list) if idx != rm_cc_idx]
        remaining_wc = [p for idx, p in enumerate(wc_list) if idx != rm_wc_idx]

        # Add closest non-contact circle pair
        add_cc_pool = [(nc[0], nc[1]) for nc in non_cc[:15] if (nc[0], nc[1]) not in remaining_cc]
        add_wc_pool = [(nc[0], nc[1]) for nc in non_wc[:15] if (nc[0], nc[1]) not in remaining_wc]

        if not add_cc_pool or not add_wc_pool:
            continue

        add_cc = add_cc_pool[rng.randint(len(add_cc_pool))]
        add_wc = add_wc_pool[rng.randint(len(add_wc_pool))]

        new_active = []
        for wtype, i in remaining_wc:
            new_active.append((wtype, i))
        new_active.append((add_wc[0], add_wc[1]))
        for i, j in remaining_cc:
            new_active.append(('C', i, j))
        new_active.append(('C', add_cc[0], add_cc[1]))

        if len(new_active) != 78:
            continue

        n_tried += 1
        sol, metric, feasible = solve_topology(new_active, xk, yk, rk)

        if sol is not None:
            n_solved += 1
            if feasible and metric > best_metric:
                n_better += 1
                best_metric = metric
                best_sol = sol
                print(f"  *** NEW BEST: {metric:.15f} (attempt {attempt+1}) ***")

        if (attempt + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{attempt+1}/300] tried={n_tried}, solved={n_solved}, "
                  f"better={n_better}, time={elapsed:.0f}s")

    # ===== Summary =====
    sol_path = os.path.join(WORKDIR, 'solution_n26.json')
    save_solution(*best_sol, sol_path)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"FINAL: {best_metric:.15f} (known={known_metric:.15f})")
    print(f"Total tried: {n_tried}, solved: {n_solved}, better: {n_better}")
    print(f"Time: {elapsed:.0f}s")


if __name__ == '__main__':
    main()
