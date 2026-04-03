"""
Topology enumeration approach for n=26.

Key insight from topo-001: the current best has 78 active constraints
(58 circle-circle + 20 wall contacts), and is fully rigid (0 DOF).

Strategy: Instead of random inits, SYSTEMATICALLY try different numbers
of wall contacts and different circle-circle contact patterns.

The current best uses a specific ring pattern. Let me try:
1. Solutions with FEWER wall contacts (more DOF -> can potentially grow radii)
2. Solutions with different symmetry (D4, C4, D2, etc.)
3. Known good layouts from the literature at nearby n values, adapted
4. The "Tactical Maniac" arrangement if we can reconstruct it
"""

import json
import math
import numpy as np
from scipy.optimize import minimize
from pathlib import Path
import time
import itertools

SEED = 54321
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


def penalty_then_slsqp(circles, maxiter_slsqp=10000):
    """Penalty method then SLSQP polish."""
    n = len(circles)
    x = circles.flatten().copy()
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-5, 0.5)] * n

    stages = [0.1, 1, 10, 100, 1000, 10000, 100000, 1e6, 1e7, 1e8]
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
                if len(js) == 0: continue
                dx = xs[i] - xs[js]; dy = ys[i] - ys[js]
                dsq = dx * dx + dy * dy
                md = rs[i] + rs[js]; mdsq = md ** 2
                active = mdsq > dsq
                if not np.any(active): continue
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
                          options={'maxiter': 2000, 'ftol': 1e-15})
        x = result.x.copy()

    pen_out = x.reshape(n, 3)
    pen_valid, _ = validate(pen_out, tol=1e-8)

    # SLSQP polish
    if pen_valid:
        try:
            nn = 3 * n
            def objective(x): return -np.sum(x[2::3])
            def grad_obj(x):
                g = np.zeros(nn); g[2::3] = -1.0; return g
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
            slsqp_bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * n
            result = minimize(objective, pen_out.flatten(), method='SLSQP', jac=grad_obj,
                              bounds=slsqp_bounds, constraints=[{'type': 'ineq', 'fun': all_con}],
                              options={'maxiter': maxiter_slsqp, 'ftol': 1e-15})
            slsqp_out = result.x.reshape(n, 3)
            valid, _ = validate(slsqp_out)
            if valid:
                return slsqp_out, -result.fun, True
        except:
            pass
        return pen_out, sum_radii(pen_out), True

    return pen_out, 0.0, False


def make_symmetric_init(pattern, rng):
    """Create initialization with specific symmetry pattern."""
    circles = []

    if pattern == 'D4_ring':
        # D4 symmetry: 1 center + 4-fold symmetric rings
        # 1 + 4 + 8 + 8 + 4 + 1 = 26
        circles.append([0.5, 0.5, 0.13])
        # Inner ring of 4
        for i in range(4):
            t = math.pi / 4 + i * math.pi / 2
            circles.append([0.5 + 0.18 * math.cos(t), 0.5 + 0.18 * math.sin(t), 0.10])
        # Middle ring of 8
        for i in range(8):
            t = i * math.pi / 4
            circles.append([0.5 + 0.32 * math.cos(t), 0.5 + 0.32 * math.sin(t), 0.085])
        # Outer ring of 8
        for i in range(8):
            t = math.pi / 8 + i * math.pi / 4
            r_pos = 0.42
            x = max(0.06, min(0.94, 0.5 + r_pos * math.cos(t)))
            y = max(0.06, min(0.94, 0.5 + r_pos * math.sin(t)))
            circles.append([x, y, 0.065])
        # Corner fillers: 4
        for cx, cy in [(0.06, 0.06), (0.94, 0.06), (0.06, 0.94), (0.94, 0.94)]:
            circles.append([cx, cy, 0.055])
        # One more
        circles.append([0.5, 0.06, 0.055])

    elif pattern == 'C4_asym':
        # C4 symmetry: 2 + 6*4 = 26
        circles.append([0.5, 0.5, 0.11])
        circles.append([0.5, 0.5 + 0.14, 0.08])  # break D4 by offset
        # 4-fold copies of a 6-circle motif
        motif = [(0.25, 0.12, 0.09), (0.12, 0.25, 0.09),
                 (0.08, 0.08, 0.07), (0.35, 0.08, 0.065),
                 (0.08, 0.35, 0.065), (0.20, 0.20, 0.07)]
        for rot in range(4):
            for mx, my, mr in motif:
                if rot == 0:
                    circles.append([mx, my, mr])
                elif rot == 1:
                    circles.append([1 - my, mx, mr])
                elif rot == 2:
                    circles.append([1 - mx, 1 - my, mr])
                else:
                    circles.append([my, 1 - mx, mr])

    elif pattern == 'D2_rect':
        # D2 symmetry: mirror about both axes
        # 2 on center line + 12 pairs = 26
        circles.append([0.5, 0.5, 0.12])
        circles.append([0.5, 0.15, 0.09])
        # Symmetric pairs (x, y) -> (1-x, y), (x, 1-y), (1-x, 1-y)
        half_motif = [
            (0.15, 0.5, 0.09), (0.15, 0.22, 0.08), (0.15, 0.78, 0.08),
            (0.35, 0.2, 0.075), (0.35, 0.8, 0.075), (0.35, 0.5, 0.08),
        ]
        for mx, my, mr in half_motif:
            circles.append([mx, my, mr])
            circles.append([1 - mx, my, mr])

    elif pattern == 'hex_tight':
        # Tight hex packing with specific row counts
        # Rows: 5, 5, 5, 5, 5, 1 = 26 with offset
        r_base = 0.088
        y = r_base + 0.005
        dy = r_base * math.sqrt(3) + 0.003
        for row in range(5):
            n_in_row = 5
            offset = (r_base + 0.005) if row % 2 == 1 else 0
            dx = (1.0 - 2 * r_base) / (n_in_row - 1)
            for i in range(n_in_row):
                x = r_base + 0.005 + i * dx + offset * 0.5
                circles.append([min(1 - r_base, max(r_base, x)), y,
                                r_base + rng.uniform(-0.003, 0.003)])
            y += dy
        # One extra
        circles.append([0.5, min(1 - 0.04, y), 0.04])

    elif pattern == 'hex_54444':
        # 5-4-4-4-4-5 rows = 26
        r_base = 0.09
        row_counts = [5, 4, 4, 4, 4, 5]
        y = r_base + 0.01
        dy = r_base * 1.8
        for ri, count in enumerate(row_counts):
            dx = (1.0 - 2 * r_base) / max(count - 1, 1)
            offset = (1.0 - (count - 1) * dx - 2 * r_base) / 2
            for i in range(count):
                x = r_base + offset + i * dx
                circles.append([x, y, r_base + rng.uniform(-0.005, 0.005)])
            y += dy

    elif pattern == 'hex_5544':
        # 5-5-4-4-4-4 rows = 26
        r_base = 0.088
        row_counts = [5, 5, 4, 4, 4, 4]
        y = r_base + 0.01
        dy = r_base * 1.85
        for ri, count in enumerate(row_counts):
            dx = (1.0 - 2 * r_base) / max(count - 1, 1)
            offset = (1.0 - (count - 1) * dx - 2 * r_base) / 2
            for i in range(count):
                x = r_base + offset + i * dx
                circles.append([x, y, r_base + rng.uniform(-0.005, 0.005)])
            y += dy

    elif pattern == 'corner_fill':
        # Large corner circles + gap-filling strategy
        # 4 corner circles + 4 edge-center + 1 center + 13 fill
        corner_r = 0.16
        for cx, cy in [(corner_r, corner_r), (1 - corner_r, corner_r),
                        (corner_r, 1 - corner_r), (1 - corner_r, 1 - corner_r)]:
            circles.append([cx, cy, corner_r])
        # Edge centers
        edge_r = 0.10
        for cx, cy in [(0.5, edge_r), (0.5, 1 - edge_r),
                        (edge_r, 0.5), (1 - edge_r, 0.5)]:
            circles.append([cx, cy, edge_r])
        # Center
        circles.append([0.5, 0.5, 0.09])
        # Fill remaining 13
        for _ in range(13):
            x = rng.uniform(0.05, 0.95)
            y = rng.uniform(0.05, 0.95)
            circles.append([x, y, 0.05 + rng.uniform(0, 0.03)])

    elif pattern == 'golden_spiral':
        # Fermat spiral placement
        phi = (1 + math.sqrt(5)) / 2
        for i in range(N):
            theta = 2 * math.pi * i / phi
            r_pos = 0.42 * math.sqrt((i + 0.5) / N)
            x = 0.5 + r_pos * math.cos(theta)
            y = 0.5 + r_pos * math.sin(theta)
            # Radius decreases with distance from center
            cr = 0.12 - 0.06 * r_pos / 0.42
            x = max(cr, min(1 - cr, x))
            y = max(cr, min(1 - cr, y))
            circles.append([x, y, cr + rng.uniform(-0.005, 0.005)])

    elif pattern == 'two_big':
        # Two large circles + 24 smaller ones filling gaps
        big_r = rng.uniform(0.15, 0.20)
        circles.append([big_r + 0.02, 0.5, big_r])
        circles.append([1 - big_r - 0.02, 0.5, big_r])
        for i in range(24):
            theta = 2 * math.pi * i / 24
            r_pos = 0.3
            x = 0.5 + r_pos * math.cos(theta)
            y = 0.5 + r_pos * math.sin(theta)
            cr = rng.uniform(0.04, 0.08)
            x = max(cr, min(1 - cr, x))
            y = max(cr, min(1 - cr, y))
            circles.append([x, y, cr])

    elif pattern == 'three_big':
        # Three large circles in triangle
        big_r = rng.uniform(0.12, 0.16)
        cx, cy = 0.5, 0.5
        for i in range(3):
            t = 2 * math.pi * i / 3 - math.pi / 2
            d = 0.22
            circles.append([cx + d * math.cos(t), cy + d * math.sin(t), big_r])
        for i in range(23):
            x = rng.uniform(0.06, 0.94)
            y = rng.uniform(0.06, 0.94)
            circles.append([x, y, rng.uniform(0.04, 0.08)])

    # Ensure we have exactly N circles
    while len(circles) < N:
        circles.append([rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9),
                        rng.uniform(0.03, 0.06)])
    circles = circles[:N]

    result = np.array(circles)
    # Clamp
    for i in range(N):
        result[i, 2] = max(result[i, 2], 0.015)
        result[i, 0] = max(result[i, 2], min(1 - result[i, 2], result[i, 0]))
        result[i, 1] = max(result[i, 2], min(1 - result[i, 2], result[i, 1]))

    # Add noise
    result[:, :2] += rng.normal(0, 0.01, (N, 2))
    result[:, 2] *= (1 + rng.normal(0, 0.02, N))
    for i in range(N):
        result[i, 2] = max(result[i, 2], 0.015)
        result[i, 0] = max(result[i, 2], min(1 - result[i, 2], result[i, 0]))
        result[i, 1] = max(result[i, 2], min(1 - result[i, 2], result[i, 1]))

    return result


def main():
    # Load best known
    base = load_solution(WORKTREE / "orbits/topo-001/solution_n26.json")
    try:
        current = load_solution(OUTPUT_DIR / "solution_n26.json")
        if sum_radii(current) > sum_radii(base):
            base = current
    except:
        pass
    base_metric = sum_radii(base)
    print(f"Starting from: {base_metric:.10f}")

    rng = np.random.RandomState(SEED)
    best = base_metric
    best_circles = base.copy()

    patterns = ['D4_ring', 'C4_asym', 'D2_rect', 'hex_tight', 'hex_54444',
                'hex_5544', 'corner_fill', 'golden_spiral', 'two_big', 'three_big']

    print(f"\n=== Topology Enumeration: {len(patterns)} patterns x 20 seeds ===")
    t0 = time.time()

    results_by_pattern = {}

    for pi, pattern in enumerate(patterns):
        pattern_best = 0
        for seed_offset in range(20):
            rng2 = np.random.RandomState(SEED + pi * 100 + seed_offset)
            try:
                init = make_symmetric_init(pattern, rng2)
                opt, metric, valid = penalty_then_slsqp(init)
                if valid and metric > 0:
                    pattern_best = max(pattern_best, metric)
                    if metric > best + 1e-12:
                        print(f"  {pattern} seed={seed_offset}: "
                              f"{best:.10f} -> {metric:.10f} NEW BEST")
                        best = metric
                        best_circles = opt.copy()
            except:
                pass

        results_by_pattern[pattern] = pattern_best
        elapsed = time.time() - t0
        print(f"  Pattern {pi+1}/{len(patterns)} '{pattern}': "
              f"best={pattern_best:.6f}, elapsed={elapsed:.1f}s")

    print(f"\nPhase 1 done: {time.time()-t0:.1f}s, best={best:.10f}")
    print("\nResults by pattern:")
    for p, m in sorted(results_by_pattern.items(), key=lambda x: -x[1]):
        print(f"  {p}: {m:.10f}")

    # ============ Phase 2: Variations on the best-performing patterns ============
    top_patterns = sorted(results_by_pattern.items(), key=lambda x: -x[1])[:3]
    print(f"\n=== Phase 2: Deep search on top patterns ===")

    for pattern, _ in top_patterns:
        print(f"\n  Deep search: {pattern}")
        for seed_offset in range(50):
            rng2 = np.random.RandomState(SEED + 10000 + seed_offset)
            try:
                init = make_symmetric_init(pattern, rng2)
                # Larger perturbation to explore more
                init[:, :2] += rng2.normal(0, 0.03, (N, 2))
                init[:, 2] *= (1 + rng2.normal(0, 0.05, N))
                for i in range(N):
                    init[i, 2] = max(init[i, 2], 0.015)
                    init[i, 0] = max(init[i, 2], min(1 - init[i, 2], init[i, 0]))
                    init[i, 1] = max(init[i, 2], min(1 - init[i, 2], init[i, 1]))

                opt, metric, valid = penalty_then_slsqp(init)
                if valid and metric > best + 1e-12:
                    print(f"    seed={seed_offset}: {best:.10f} -> {metric:.10f} NEW BEST")
                    best = metric
                    best_circles = opt.copy()
            except:
                pass

        print(f"  Done {pattern}: best={best:.10f}")

    # ============ Phase 3: Reconstruct "Tactical Maniac" arrangement ============
    # The known results show: 2.63593 from a novel arrangement
    # This is LESS than our 2.63598, so likely the same topology with less polish
    # Skip this - focus on finding genuinely new topologies

    # ============ Phase 4: Try n=27 minus one ============
    print(f"\n=== Phase 4: Remove-one-circle from n=27 candidates ===")
    # Generate an n=27 packing, remove the smallest circle
    for trial in range(30):
        rng2 = np.random.RandomState(SEED + 20000 + trial)
        # Make an n=27 initialization
        init27 = np.zeros((27, 3))
        radii = rng2.uniform(0.05, 0.12, 27)
        radii = np.sort(radii)[::-1]
        for i in range(27):
            r = radii[i]
            placed = False
            for _ in range(200):
                x = rng2.uniform(r + 0.001, 1 - r - 0.001)
                y = rng2.uniform(r + 0.001, 1 - r - 0.001)
                ok = True
                for j in range(i):
                    dx = x - init27[j, 0]; dy = y - init27[j, 1]
                    if math.sqrt(dx * dx + dy * dy) < r + init27[j, 2] + 0.001:
                        ok = False; break
                if ok:
                    init27[i] = [x, y, r]; placed = True; break
            if not placed:
                init27[i] = [rng2.uniform(r, 1 - r), rng2.uniform(r, 1 - r), r]

        # Optimize n=27
        try:
            opt27, m27, v27 = penalty_then_slsqp(init27)
            if v27:
                # Remove each circle, re-optimize
                for drop in range(27):
                    c26 = np.delete(opt27, drop, axis=0)
                    opt26, m26, v26 = penalty_then_slsqp(c26)
                    if v26 and m26 > best + 1e-12:
                        print(f"    Trial {trial}, drop {drop}: {best:.10f} -> {m26:.10f} NEW BEST")
                        best = m26
                        best_circles = opt26.copy()
        except:
            pass

        if (trial + 1) % 10 == 0:
            print(f"  Trial {trial+1}/30: best={best:.10f}")

    # Final
    valid, viol = validate(best_circles)
    print(f"\nFINAL: {best:.10f} (valid={valid}, viol={viol:.2e})")

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


if __name__ == "__main__":
    main()
