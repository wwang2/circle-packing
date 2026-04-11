"""
Aggressive n=32 optimizer.
Strategy:
1. Analyze contact graph of current best to understand topology
2. Try symmetry-breaking perturbations
3. Large-scale rearrangement: shuffle subsets of circles
4. Billiard-like dynamics simulation
5. Greedy from scratch with optimized placement
"""

import json
import math
import numpy as np
from scipy.optimize import minimize
from pathlib import Path
import time
import sys
import itertools

N = 32
BEST_FILE = Path(__file__).parent / "solution_n32.json"
LOG_FILE = Path(__file__).parent / "log.md"

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

def neg_sum_radii(x):
    return -np.sum(x[2::3])

def neg_sum_radii_jac(x):
    j = np.zeros(len(x))
    j[2::3] = -1.0
    return j

def all_ineq(x):
    c = x.reshape(-1, 3)
    n = len(c)
    xs, ys, rs = c[:,0], c[:,1], c[:,2]
    cont = np.concatenate([xs - rs, 1.0 - xs - rs, ys - rs, 1.0 - ys - rs, rs - 1e-6])
    overlap = np.empty(n*(n-1)//2)
    k = 0
    for i in range(n):
        for j in range(i+1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            overlap[k] = math.sqrt(dx*dx + dy*dy) - rs[i] - rs[j]
            k += 1
    return np.concatenate([cont, overlap])

def is_valid(x, tol=1e-10):
    return np.all(all_ineq(x) >= -tol)

def get_metric(x):
    return np.sum(x[2::3])

def make_bounds():
    bounds = []
    for _ in range(N):
        bounds.append((1e-4, 1.0 - 1e-4))
        bounds.append((1e-4, 1.0 - 1e-4))
        bounds.append((1e-6, 0.5))
    return bounds

def slsqp_opt(x0, maxiter=5000):
    constraints = [{"type": "ineq", "fun": all_ineq}]
    bounds = make_bounds()
    result = minimize(
        neg_sum_radii, x0, method='SLSQP', jac=neg_sum_radii_jac,
        bounds=bounds, constraints=constraints,
        options={'maxiter': maxiter, 'ftol': 1e-16, 'disp': False}
    )
    return result.x, -result.fun

def analyze_contacts(x, tol=1e-4):
    """Find near-contacts in the packing."""
    c = x.reshape(-1, 3)
    n = len(c)
    contacts = []
    wall_contacts = []

    for i in range(n):
        xi, yi, ri = c[i]
        if abs(xi - ri) < tol: wall_contacts.append((i, 'left'))
        if abs(1 - xi - ri) < tol: wall_contacts.append((i, 'right'))
        if abs(yi - ri) < tol: wall_contacts.append((i, 'bottom'))
        if abs(1 - yi - ri) < tol: wall_contacts.append((i, 'top'))

    for i in range(n):
        for j in range(i+1, n):
            dx = c[i,0]-c[j,0]; dy = c[i,1]-c[j,1]
            dist = math.sqrt(dx*dx + dy*dy)
            gap = dist - c[i,2] - c[j,2]
            if abs(gap) < tol:
                contacts.append((i, j, gap))

    return contacts, wall_contacts

def greedy_packing(n, rng, strategy='largest_gap'):
    """Build packing greedily by placing circles one at a time in the largest gap."""
    circles = []

    # Place first circle at center
    r0 = 0.15
    circles.append([0.5, 0.5, r0])

    for k in range(1, n):
        best_r = 0
        best_pos = None

        # Sample many candidate positions
        n_samples = 10000
        for _ in range(n_samples):
            cx = rng.uniform(0.01, 0.99)
            cy = rng.uniform(0.01, 0.99)
            max_r = min(cx, 1-cx, cy, 1-cy)
            for (ox, oy, orr) in circles:
                d = math.sqrt((cx-ox)**2 + (cy-oy)**2)
                max_r = min(max_r, d - orr)
            if max_r > best_r:
                best_r = max_r
                best_pos = [cx, cy, max_r]

        if best_pos and best_pos[2] > 0.001:
            circles.append(best_pos)
        else:
            r = 0.002
            circles.append([rng.uniform(r+0.01, 1-r-0.01), rng.uniform(r+0.01, 1-r-0.01), r])

    return np.array(circles).flatten()

def billiard_dynamics(x, rng, steps=200, dt=0.001):
    """Simulate billiard-like dynamics where circles bounce and grow."""
    c = x.reshape(-1, 3).copy()
    n = len(c)

    # Random velocities
    vx = rng.normal(0, 0.1, n)
    vy = rng.normal(0, 0.1, n)

    for step in range(steps):
        # Move circles
        c[:, 0] += vx * dt
        c[:, 1] += vy * dt

        # Wall bounces
        for i in range(n):
            if c[i,0] - c[i,2] < 0:
                c[i,0] = c[i,2] + 1e-4
                vx[i] = abs(vx[i])
            if c[i,0] + c[i,2] > 1:
                c[i,0] = 1 - c[i,2] - 1e-4
                vx[i] = -abs(vx[i])
            if c[i,1] - c[i,2] < 0:
                c[i,1] = c[i,2] + 1e-4
                vy[i] = abs(vy[i])
            if c[i,1] + c[i,2] > 1:
                c[i,1] = 1 - c[i,2] - 1e-4
                vy[i] = -abs(vy[i])

        # Circle-circle collisions (elastic)
        for i in range(n):
            for j in range(i+1, n):
                dx = c[i,0]-c[j,0]; dy = c[i,1]-c[j,1]
                dist = math.sqrt(dx*dx + dy*dy)
                min_dist = c[i,2] + c[j,2]
                if dist < min_dist and dist > 1e-10:
                    # Push apart
                    overlap = min_dist - dist
                    nx_dir = dx/dist; ny_dir = dy/dist
                    c[i,0] += nx_dir * overlap * 0.5
                    c[i,1] += ny_dir * overlap * 0.5
                    c[j,0] -= nx_dir * overlap * 0.5
                    c[j,1] -= ny_dir * overlap * 0.5
                    # Elastic collision
                    dvx = vx[i]-vx[j]; dvy = vy[i]-vy[j]
                    dot = dvx*nx_dir + dvy*ny_dir
                    vx[i] -= dot*nx_dir; vy[i] -= dot*ny_dir
                    vx[j] += dot*nx_dir; vy[j] += dot*ny_dir

        # Grow radii every 10 steps
        if step % 10 == 0:
            for i in range(n):
                max_r = min(c[i,0], 1-c[i,0], c[i,1], 1-c[i,1])
                for j in range(n):
                    if j == i: continue
                    dx = c[i,0]-c[j,0]; dy = c[i,1]-c[j,1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    max_r = min(max_r, dist - c[j,2])
                grow = (max_r - c[i,2]) * 0.1
                if grow > 0:
                    c[i,2] += grow

        # Damping
        vx *= 0.99
        vy *= 0.99

    return c.flatten()

def shuffle_subset(x, rng, num_shuffle=8):
    """Remove a subset of circles, rearrange the rest, reinsert."""
    c = x.reshape(-1, 3).copy()
    n = len(c)

    # Pick a spatial cluster to shuffle
    pivot = rng.integers(n)
    dists = np.sqrt((c[:,0]-c[pivot,0])**2 + (c[:,1]-c[pivot,1])**2)
    shuffle_idx = np.argsort(dists)[:num_shuffle]

    # Save positions of circles to shuffle
    shuffled = c[shuffle_idx].copy()

    # Try random rearrangements of these circles' positions
    best_arrangement = None
    best_total = -1

    for _ in range(100):
        perm = rng.permutation(num_shuffle)
        trial = c.copy()
        # Swap positions but keep radii
        for k, idx in enumerate(shuffle_idx):
            src = shuffle_idx[perm[k]]
            trial[idx, 0] = shuffled[perm[k], 0]
            trial[idx, 1] = shuffled[perm[k], 1]
            # Adjust radius to fit
            max_r = min(trial[idx,0], 1-trial[idx,0], trial[idx,1], 1-trial[idx,1])
            for j in range(n):
                if j == idx: continue
                dx = trial[idx,0]-trial[j,0]; dy = trial[idx,1]-trial[j,1]
                dist = math.sqrt(dx*dx + dy*dy)
                max_r = min(max_r, dist - trial[j,2])
            trial[idx, 2] = max(0.001, max_r)

        total = np.sum(trial[:, 2])
        if total > best_total:
            best_total = total
            best_arrangement = trial.copy()

    return best_arrangement.flatten() if best_arrangement is not None else x

def mirror_transform(x, rng):
    """Apply a random symmetry transformation (mirror, rotate 90, etc.)."""
    c = x.reshape(-1, 3).copy()
    choice = rng.integers(6)
    if choice == 0:  # Mirror x
        c[:, 0] = 1 - c[:, 0]
    elif choice == 1:  # Mirror y
        c[:, 1] = 1 - c[:, 1]
    elif choice == 2:  # Rotate 90
        c[:, 0], c[:, 1] = c[:, 1].copy(), (1 - c[:, 0]).copy()
    elif choice == 3:  # Rotate 180
        c[:, 0] = 1 - c[:, 0]; c[:, 1] = 1 - c[:, 1]
    elif choice == 4:  # Mirror diagonal
        c[:, 0], c[:, 1] = c[:, 1].copy(), c[:, 0].copy()
    elif choice == 5:  # Mirror anti-diagonal
        c[:, 0], c[:, 1] = (1 - c[:, 1]).copy(), (1 - c[:, 0]).copy()
    return c.flatten()

def run():
    best_circles = load_solution(BEST_FILE)
    best_x = best_circles.flatten()
    best_metric = get_metric(best_x)
    log(f"Starting metric: {best_metric:.10f}")

    # Analyze current solution
    contacts, wall_contacts = analyze_contacts(best_x)
    log(f"Contact graph: {len(contacts)} circle-circle contacts, {len(wall_contacts)} wall contacts")
    c = best_x.reshape(-1, 3)
    log(f"Radii: min={c[:,2].min():.6f}, max={c[:,2].max():.6f}, mean={c[:,2].mean():.6f}")
    log(f"Sorted radii: {sorted(c[:,2], reverse=True)[:10]}")

    improvements = []
    start_time = time.time()

    def try_update(x, source):
        nonlocal best_x, best_metric
        if is_valid(x):
            m = get_metric(x)
            if m > best_metric + 1e-12:
                imp = m - best_metric
                best_metric = m
                best_x = x.copy()
                save_solution(x.reshape(-1, 3), BEST_FILE)
                improvements.append((source, m))
                log(f"  *** NEW BEST: {m:.10f} (+{imp:.2e}) [{source}]")
                return True
        return False

    # Strategy 1: Greedy packings from scratch + SLSQP polish
    log("\n=== Strategy 1: Greedy packings (100 trials) ===")
    for trial in range(100):
        if trial % 20 == 0:
            log(f"  Greedy trial {trial}/100, best={best_metric:.10f}")
        rng = np.random.default_rng(trial * 59 + 7000)
        x0 = greedy_packing(N, rng)
        x_opt, m = slsqp_opt(x0, maxiter=5000)
        try_update(x_opt, f"greedy-{trial}")

    log(f"\nAfter Strategy 1: {best_metric:.10f}")

    # Strategy 2: Billiard dynamics + SLSQP
    log("\n=== Strategy 2: Billiard dynamics (50 trials) ===")
    for trial in range(50):
        if trial % 10 == 0:
            log(f"  Billiard trial {trial}/50")
        rng = np.random.default_rng(trial * 41 + 8000)

        # Start from perturbed best
        x0 = best_x.copy()
        c0 = x0.reshape(-1, 3)
        c0[:, 0] += rng.normal(0, 0.02, N)
        c0[:, 1] += rng.normal(0, 0.02, N)
        for i in range(N):
            c0[i,0] = np.clip(c0[i,0], c0[i,2]+1e-4, 1-c0[i,2]-1e-4)
            c0[i,1] = np.clip(c0[i,1], c0[i,2]+1e-4, 1-c0[i,2]-1e-4)
        x0 = c0.flatten()

        x_dyn = billiard_dynamics(x0, rng, steps=300, dt=0.002)
        x_opt, m = slsqp_opt(x_dyn, maxiter=5000)
        try_update(x_opt, f"billiard-{trial}")

    log(f"\nAfter Strategy 2: {best_metric:.10f}")

    # Strategy 3: Shuffle subsets
    log("\n=== Strategy 3: Shuffle subsets (100 trials) ===")
    for trial in range(100):
        if trial % 20 == 0:
            log(f"  Shuffle trial {trial}/100")
        rng = np.random.default_rng(trial * 23 + 9000)
        num_shuffle = rng.integers(3, 10)
        x0 = shuffle_subset(best_x, rng, num_shuffle=num_shuffle)
        x_opt, m = slsqp_opt(x0, maxiter=5000)
        try_update(x_opt, f"shuffle-{trial}-n{num_shuffle}")

    log(f"\nAfter Strategy 3: {best_metric:.10f}")

    # Strategy 4: Mirror + optimize
    log("\n=== Strategy 4: Symmetry transforms ===")
    for trial in range(30):
        rng = np.random.default_rng(trial * 71 + 10000)
        x0 = mirror_transform(best_x, rng)
        x_opt, m = slsqp_opt(x0, maxiter=5000)
        try_update(x_opt, f"mirror-{trial}")

    log(f"\nAfter Strategy 4: {best_metric:.10f}")

    # Strategy 5: Multi-scale perturbation
    log("\n=== Strategy 5: Multi-scale perturbation (500 trials) ===")
    for trial in range(500):
        if trial % 100 == 0:
            log(f"  Multi-scale trial {trial}/500, best={best_metric:.10f}")
        rng = np.random.default_rng(trial * 13 + 11000)

        c = best_x.reshape(-1, 3).copy()
        n = N

        # Pick a random scale and number of circles to perturb
        num_perturb = rng.integers(1, n+1)
        scale = rng.exponential(0.02)
        idx = rng.choice(n, size=num_perturb, replace=False)

        c[idx, 0] += rng.normal(0, scale, num_perturb)
        c[idx, 1] += rng.normal(0, scale, num_perturb)

        # Also try perturbing radii
        if rng.random() < 0.3:
            c[idx, 2] *= (1 + rng.normal(0, 0.05, num_perturb))
            c[idx, 2] = np.maximum(c[idx, 2], 0.005)

        # Clamp
        for i in range(n):
            r = max(0.001, c[i,2])
            c[i,2] = r
            c[i,0] = np.clip(c[i,0], r+1e-4, 1-r-1e-4)
            c[i,1] = np.clip(c[i,1], r+1e-4, 1-r-1e-4)

        x_opt, m = slsqp_opt(c.flatten(), maxiter=3000)
        try_update(x_opt, f"multiscale-{trial}")

    log(f"\nAfter Strategy 5: {best_metric:.10f}")

    elapsed = time.time() - start_time
    log(f"\n{'='*60}")
    log(f"FINAL BEST: {best_metric:.10f}")
    log(f"Improvements: {len(improvements)}")
    log(f"Time: {elapsed:.1f}s")
    for src, m in improvements:
        log(f"  {src}: {m:.10f}")

    with open(LOG_FILE, "a") as f:
        f.write(f"\n## Optimize4 run {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"- Final metric: {best_metric:.10f}\n")
        f.write(f"- Time: {elapsed:.1f}s\n")
        f.write(f"- Improvements: {len(improvements)}\n")
        for src, m in improvements:
            f.write(f"  - {src}: {m:.10f}\n")

    return best_metric

if __name__ == "__main__":
    run()
