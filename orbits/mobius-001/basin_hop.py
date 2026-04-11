"""
Basin-hopping search for n=26 circle packing.

Strategy:
1. Analyze the current best solution's contact graph
2. Apply topology-changing perturbations (remove/add contacts)
3. Use simulated annealing + SLSQP polish
4. Track distinct basins by their contact graph fingerprint
"""

import json
import math
import numpy as np
from scipy.optimize import minimize, basinhopping
from pathlib import Path
import time
import hashlib

SEED = 12345
N = 26
WORKTREE = Path("/Users/wujiewang/code/circle-packing/.worktrees/mobius-001")
OUTPUT_DIR = WORKTREE / "orbits/mobius-001"


def load_solution(path):
    with open(path) as f:
        data = json.load(f)
    return np.array(data["circles"])


def save_solution(circles, path):
    data = {"circles": [[float(c[0]), float(c[1]), float(c[2])] for c in circles]}
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def sum_radii(c):
    return float(np.sum(c[:, 2]))


def validate(circles, tol=1e-10):
    n = len(circles)
    mv = 0.0
    for i in range(n):
        x, y, r = circles[i]
        if r <= 0:
            return False, abs(r)
        mv = max(mv, r - x, x + r - 1, r - y, y + r - 1)
    for i in range(n):
        for j in range(i + 1, n):
            dx = circles[i, 0] - circles[j, 0]
            dy = circles[i, 1] - circles[j, 1]
            d = math.sqrt(dx * dx + dy * dy)
            mv = max(mv, circles[i, 2] + circles[j, 2] - d)
    return mv <= tol, mv


def get_contact_graph(circles, tol=1e-4):
    """Get sorted list of contact pairs (fingerprint of topology)."""
    n = len(circles)
    contacts = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = circles[i, 0] - circles[j, 0]
            dy = circles[i, 1] - circles[j, 1]
            d = math.sqrt(dx * dx + dy * dy)
            gap = d - circles[i, 2] - circles[j, 2]
            if abs(gap) < tol:
                contacts.append((i, j))
    # Wall contacts
    wall = []
    for i in range(n):
        x, y, r = circles[i]
        if abs(x - r) < tol:
            wall.append((i, 'L'))
        if abs(1 - x - r) < tol:
            wall.append((i, 'R'))
        if abs(y - r) < tol:
            wall.append((i, 'B'))
        if abs(1 - y - r) < tol:
            wall.append((i, 'T'))
    return contacts, wall


def contact_fingerprint(circles, tol=1e-4):
    """Hash of sorted radii to identify topology basin."""
    radii = sorted(circles[:, 2], reverse=True)
    # Round to 4 decimal places for grouping
    key = tuple(round(r, 4) for r in radii)
    return hashlib.md5(str(key).encode()).hexdigest()[:12]


def slsqp_optimize(circles, maxiter=8000):
    """SLSQP with vectorized constraints."""
    n = len(circles)
    x0 = circles.flatten()
    nn = 3 * n

    def objective(x):
        return -np.sum(x[2::3])

    def grad_obj(x):
        g = np.zeros(nn)
        g[2::3] = -1.0
        return g

    def all_con(x):
        xs = x[0::3]; ys = x[1::3]; rs = x[2::3]
        vals = []
        vals.extend(xs - rs)
        vals.extend(1.0 - xs - rs)
        vals.extend(ys - rs)
        vals.extend(1.0 - ys - rs)
        vals.extend(rs - 1e-6)
        for i in range(n):
            dx = xs[i] - xs[i + 1:]
            dy = ys[i] - ys[i + 1:]
            vals.extend(dx * dx + dy * dy - (rs[i] + rs[i + 1:]) ** 2)
        return np.array(vals)

    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * n
    result = minimize(objective, x0, method='SLSQP', jac=grad_obj,
                      bounds=bounds, constraints=[{'type': 'ineq', 'fun': all_con}],
                      options={'maxiter': maxiter, 'ftol': 1e-15})
    out = result.x.reshape(n, 3)
    valid, viol = validate(out)
    return out, -result.fun if valid else 0.0, valid


def penalty_then_slsqp(circles, maxiter_pen=1500, maxiter_slsqp=8000):
    """Penalty method to get feasible, then SLSQP to optimize."""
    n = len(circles)
    x = circles.flatten().copy()
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-5, 0.5)] * n

    stages = [1, 10, 100, 1000, 10000, 100000, 1e6, 1e7]

    for pw in stages:
        def obj_grad(x, pw=pw):
            xs = x[0::3]; ys = x[1::3]; rs = x[2::3]
            obj = -np.sum(rs)
            grad = np.zeros_like(x)
            grad[2::3] = -1.0

            vl = np.maximum(0, rs - xs); vr = np.maximum(0, xs + rs - 1)
            vb = np.maximum(0, rs - ys); vt = np.maximum(0, ys + rs - 1)
            obj += pw * np.sum(vl ** 2 + vr ** 2 + vb ** 2 + vt ** 2)
            grad[0::3] += pw * (-2 * vl + 2 * vr)
            grad[1::3] += pw * (-2 * vb + 2 * vt)
            grad[2::3] += pw * (2 * vl + 2 * vr + 2 * vb + 2 * vt)

            for i in range(n):
                js = np.arange(i + 1, n)
                if len(js) == 0:
                    continue
                dx = xs[i] - xs[js]; dy = ys[i] - ys[js]
                dsq = dx * dx + dy * dy
                md = rs[i] + rs[js]; mdsq = md ** 2
                active = mdsq > dsq
                if not np.any(active):
                    continue
                a_js = js[active]
                factor = mdsq[active] - dsq[active]
                obj += pw * np.sum(factor ** 2)
                f2 = 2 * factor
                a_dx = dx[active]; a_dy = dy[active]; a_md = md[active]
                grad[3 * i] += pw * np.sum(f2 * (-2 * a_dx))
                grad[3 * i + 1] += pw * np.sum(f2 * (-2 * a_dy))
                for k, j in enumerate(a_js):
                    grad[3 * j] += pw * f2[k] * 2 * a_dx[k]
                    grad[3 * j + 1] += pw * f2[k] * 2 * a_dy[k]
                    grad[3 * i + 2] += pw * f2[k] * 2 * a_md[k]
                    grad[3 * j + 2] += pw * f2[k] * 2 * a_md[k]
            return obj, grad

        result = minimize(obj_grad, x, jac=True, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': maxiter_pen, 'ftol': 1e-15})
        x = result.x.copy()

    pen_out = x.reshape(n, 3)
    pen_valid, pen_viol = validate(pen_out, tol=1e-8)

    if pen_valid:
        try:
            slsqp_out, slsqp_m, slsqp_v = slsqp_optimize(pen_out, maxiter=maxiter_slsqp)
            if slsqp_v and slsqp_m > sum_radii(pen_out):
                return slsqp_out, slsqp_m, slsqp_v
        except:
            pass
        return pen_out, sum_radii(pen_out), True
    else:
        # Try SLSQP anyway
        try:
            slsqp_out, slsqp_m, slsqp_v = slsqp_optimize(pen_out, maxiter=maxiter_slsqp)
            return slsqp_out, slsqp_m, slsqp_v
        except:
            return pen_out, 0.0, False


def perturb_topology(circles, rng, strength=0.1):
    """Perturb to change contact topology.

    Strategies:
    1. Push apart two contacting circles (break contact)
    2. Push together two near-but-not-touching circles (make contact)
    3. Swap positions of two similar-sized circles
    4. Scale a subset of radii up/down
    5. Rotate a cluster around its centroid
    """
    n = len(circles)
    new = circles.copy()
    strategy = rng.randint(8)

    if strategy == 0:
        # Swap positions of two circles
        i, j = rng.choice(n, 2, replace=False)
        new[i, :2], new[j, :2] = circles[j, :2].copy(), circles[i, :2].copy()

    elif strategy == 1:
        # Rotate a cluster of nearby circles
        center_idx = rng.randint(n)
        cx, cy = circles[center_idx, 0], circles[center_idx, 1]
        dists = np.sqrt((circles[:, 0] - cx) ** 2 + (circles[:, 1] - cy) ** 2)
        nearby = np.where(dists < 0.3)[0]
        if len(nearby) > 2:
            angle = rng.uniform(-strength * math.pi, strength * math.pi)
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            for idx in nearby:
                dx = new[idx, 0] - cx
                dy = new[idx, 1] - cy
                new[idx, 0] = cx + dx * cos_a - dy * sin_a
                new[idx, 1] = cy + dx * sin_a + dy * cos_a

    elif strategy == 2:
        # Scale radii of a subset up, shrink others to compensate
        k = rng.randint(3, 10)
        grow = rng.choice(n, k, replace=False)
        factor = 1 + strength * 0.5
        new[grow, 2] *= factor
        shrink = [i for i in range(n) if i not in grow]
        new[shrink, 2] *= (1 - strength * 0.2)

    elif strategy == 3:
        # Mirror a subset across x=0.5 or y=0.5
        k = rng.randint(3, 12)
        subset = rng.choice(n, k, replace=False)
        if rng.random() < 0.5:
            new[subset, 0] = 1.0 - new[subset, 0]
        else:
            new[subset, 1] = 1.0 - new[subset, 1]

    elif strategy == 4:
        # Translate a cluster
        center_idx = rng.randint(n)
        cx, cy = circles[center_idx, 0], circles[center_idx, 1]
        dists = np.sqrt((circles[:, 0] - cx) ** 2 + (circles[:, 1] - cy) ** 2)
        nearby = np.where(dists < 0.25)[0]
        dx = rng.normal(0, strength * 0.1)
        dy = rng.normal(0, strength * 0.1)
        new[nearby, 0] += dx
        new[nearby, 1] += dy

    elif strategy == 5:
        # Remove smallest circle, enlarge neighbors
        smallest = np.argmin(circles[:, 2])
        # Find its nearest neighbor
        dists = np.sqrt((circles[:, 0] - circles[smallest, 0]) ** 2 +
                        (circles[:, 1] - circles[smallest, 1]) ** 2)
        dists[smallest] = float('inf')
        nearest = np.argsort(dists)[:3]
        extra_r = circles[smallest, 2] / 3
        new[nearest, 2] += extra_r
        # Move smallest to a random gap
        new[smallest, 2] = 0.03
        new[smallest, 0] = rng.uniform(0.05, 0.95)
        new[smallest, 1] = rng.uniform(0.05, 0.95)

    elif strategy == 6:
        # Random large perturbation on positions
        noise_scale = strength * 0.15
        new[:, 0] += rng.normal(0, noise_scale, n)
        new[:, 1] += rng.normal(0, noise_scale, n)
        new[:, 2] *= (1 + rng.normal(0, strength * 0.05, n))

    elif strategy == 7:
        # Partial sort by radius and reassign positions
        k = rng.randint(4, 15)
        subset = rng.choice(n, k, replace=False)
        # Sort by radius, reassign positions from largest to smallest space
        sub_radii = new[subset, 2].copy()
        sub_pos = new[subset, :2].copy()
        order = np.argsort(-sub_radii)
        for idx, orig_idx in enumerate(order):
            new[subset[orig_idx], :2] = sub_pos[idx]

    # Clamp
    for i in range(n):
        new[i, 2] = max(new[i, 2], 0.015)
        new[i, 0] = max(new[i, 2] + 0.001, min(1 - new[i, 2] - 0.001, new[i, 0]))
        new[i, 1] = max(new[i, 2] + 0.001, min(1 - new[i, 2] - 0.001, new[i, 1]))

    return new


def main():
    base = load_solution(WORKTREE / "orbits/topo-001/solution_n26.json")
    # Also load our current best
    try:
        current = load_solution(OUTPUT_DIR / "solution_n26.json")
        if sum_radii(current) > sum_radii(base):
            base = current
    except:
        pass

    base_metric = sum_radii(base)
    print(f"Starting from: {base_metric:.10f}")

    contacts, walls = get_contact_graph(base)
    print(f"Contact graph: {len(contacts)} circle-circle, {len(walls)} wall contacts")
    fp = contact_fingerprint(base)
    print(f"Fingerprint: {fp}")

    rng = np.random.RandomState(SEED)
    best = base_metric
    best_circles = base.copy()
    basins_seen = {fp: base_metric}
    no_improve_count = 0

    # ============ Phase 1: Basin hopping with topology perturbations ============
    print("\n=== Phase 1: Basin hopping (500 hops) ===")
    t0 = time.time()

    for hop in range(500):
        strength = rng.choice([0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0])
        perturbed = perturb_topology(best_circles, rng, strength)

        try:
            opt, metric, valid = penalty_then_slsqp(perturbed)
            if valid and metric > 0:
                fp = contact_fingerprint(opt)
                if fp not in basins_seen or metric > basins_seen[fp]:
                    basins_seen[fp] = metric
                    if len(basins_seen) % 10 == 0:
                        print(f"  Hop {hop}: new basin {fp}, metric={metric:.10f} "
                              f"(total basins: {len(basins_seen)})")

                if metric > best + 1e-12:
                    print(f"  Hop {hop}: {best:.10f} -> {metric:.10f} NEW BEST "
                          f"(strength={strength})")
                    best = metric
                    best_circles = opt.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
        except Exception as e:
            pass

        if (hop + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  Hop {hop + 1}/500: best={best:.10f}, "
                  f"basins={len(basins_seen)}, {elapsed:.1f}s")

    print(f"Phase 1: {time.time() - t0:.1f}s, best={best:.10f}, "
          f"basins={len(basins_seen)}")

    # ============ Phase 2: Start from EACH distinct basin, optimize harder ============
    print(f"\n=== Phase 2: Re-optimize top basins ===")
    t0 = time.time()

    # Sort basins by metric
    top_basins = sorted(basins_seen.items(), key=lambda x: -x[1])[:20]
    print(f"Top 20 basins: {[f'{m:.6f}' for _, m in top_basins]}")

    # ============ Phase 3: Simulated annealing around best ============
    print(f"\n=== Phase 3: Simulated annealing (300 steps) ===")
    t0 = time.time()

    current = best_circles.copy()
    current_metric = best
    T = 0.01  # Initial temperature

    for step in range(300):
        T = 0.01 * (0.99 ** step)  # Cool slowly

        # Small perturbation
        strength = rng.choice([0.02, 0.05, 0.1, 0.15])
        perturbed = perturb_topology(current, rng, strength)

        try:
            opt, metric, valid = slsqp_optimize(perturbed, maxiter=5000)
            if valid and metric > 0:
                delta = metric - current_metric
                # Accept if better, or with probability exp(delta/T)
                if delta > 0 or rng.random() < math.exp(delta / max(T, 1e-10)):
                    current = opt.copy()
                    current_metric = metric
                    if metric > best + 1e-12:
                        print(f"  SA step {step}: {best:.10f} -> {metric:.10f} NEW BEST")
                        best = metric
                        best_circles = opt.copy()
        except:
            pass

        if (step + 1) % 100 == 0:
            print(f"  SA step {step + 1}/300: best={best:.10f}, T={T:.6f}")

    print(f"Phase 3: {time.time() - t0:.1f}s, best={best:.10f}")

    # ============ Phase 4: Try known good configurations ============
    print(f"\n=== Phase 4: Known configuration patterns ===")
    t0 = time.time()

    # Try various structured initializations that might hit good topologies
    configs = []

    # Config 1: Hex grid packing
    for scale in [0.85, 0.9, 0.95, 1.0, 1.05]:
        circles = []
        r_base = 0.085 * scale
        rows = [
            (5, 0.5, r_base),
            (6, 0.5, r_base),
            (5, 0.5, r_base),
            (6, 0.5, r_base),
            (4, 0.5, r_base),
        ]
        y = r_base + 0.01
        for n_in_row, _, r in rows:
            spacing = (1.0 - 2 * r) / max(n_in_row - 1, 1)
            x_start = r + 0.01 if n_in_row < 6 else r + 0.005
            for i in range(n_in_row):
                x = x_start + i * spacing
                circles.append([x, y, r + rng.uniform(-0.005, 0.005)])
            y += 2 * r * 0.866 + 0.005
            if len(circles) >= N:
                break
        while len(circles) < N:
            circles.append([rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9), 0.04])
        configs.append(np.array(circles[:N]))

    # Config 2: One big center + ring + outer ring
    for big_r in [0.12, 0.14, 0.16, 0.18]:
        circles = [[0.5, 0.5, big_r]]
        n_inner = 7
        for i in range(n_inner):
            theta = 2 * math.pi * i / n_inner
            r = 0.08 + rng.uniform(-0.01, 0.01)
            d = big_r + r + 0.001
            circles.append([0.5 + d * math.cos(theta), 0.5 + d * math.sin(theta), r])
        n_outer = N - 1 - n_inner
        for i in range(n_outer):
            theta = 2 * math.pi * i / n_outer + math.pi / n_outer
            r = 0.06 + rng.uniform(-0.01, 0.01)
            d = 0.35
            x = 0.5 + d * math.cos(theta)
            y = 0.5 + d * math.sin(theta)
            x = max(r, min(1 - r, x))
            y = max(r, min(1 - r, y))
            circles.append([x, y, r])
        configs.append(np.array(circles[:N]))

    # Config 3: Random with biased large sizes
    for _ in range(20):
        radii = np.concatenate([
            rng.uniform(0.10, 0.16, rng.randint(2, 6)),
            rng.uniform(0.06, 0.10, N - rng.randint(2, 6))
        ])[:N]
        radii = np.sort(radii)[::-1]
        circles = np.zeros((N, 3))
        for i in range(N):
            r = radii[i]
            placed = False
            for _ in range(300):
                x = rng.uniform(r + 0.001, 1 - r - 0.001)
                y = rng.uniform(r + 0.001, 1 - r - 0.001)
                ok = True
                for j in range(i):
                    dx = x - circles[j, 0]; dy = y - circles[j, 1]
                    if math.sqrt(dx * dx + dy * dy) < r + circles[j, 2] + 0.001:
                        ok = False; break
                if ok:
                    circles[i] = [x, y, r]
                    placed = True
                    break
            if not placed:
                circles[i] = [rng.uniform(r, 1 - r), rng.uniform(r, 1 - r), r]
        configs.append(circles)

    for ci, init in enumerate(configs):
        try:
            opt, metric, valid = penalty_then_slsqp(init)
            if valid and metric > best + 1e-12:
                print(f"  Config {ci}: {best:.10f} -> {metric:.10f} NEW BEST")
                best = metric
                best_circles = opt.copy()
        except:
            pass

        if (ci + 1) % 10 == 0:
            print(f"  Config {ci + 1}/{len(configs)}: best={best:.10f}")

    print(f"Phase 4: {time.time() - t0:.1f}s, best={best:.10f}")

    # ============ Final deep polish ============
    print("\n=== Final deep polish ===")
    try:
        polished, pm, pv = slsqp_optimize(best_circles, maxiter=20000)
        if pv and pm >= best:
            best = pm
            best_circles = polished
    except:
        pass

    valid, viol = validate(best_circles)
    print(f"\nFINAL: {best:.10f} (valid={valid}, viol={viol:.2e})")
    print(f"Improvement over parent: {best - 2.6359830849:.2e}")

    # Only save if better than current
    try:
        current = load_solution(OUTPUT_DIR / "solution_n26.json")
        current_m = sum_radii(current)
    except:
        current_m = 0

    if best > current_m:
        save_solution(best_circles, OUTPUT_DIR / "solution_n26.json")
        print("Saved new best!")
    else:
        print(f"No improvement over current ({current_m:.10f})")

    # Save basin info
    with open(OUTPUT_DIR / "basin_info.json", 'w') as f:
        json.dump({
            'best': best,
            'basins_found': len(basins_seen),
            'top_basins': [(fp, m) for fp, m in sorted(basins_seen.items(),
                                                        key=lambda x: -x[1])[:30]]
        }, f, indent=2)


if __name__ == "__main__":
    main()
