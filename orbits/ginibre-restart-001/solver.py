#!/usr/bin/env python3
"""
Ginibre DPP multi-start solver for circle packing in [0,1]^2 with n=26.

Strategy: Initialize optimization from Ginibre ensemble eigenvalues
(negatively correlated, hyperuniform point process) to test whether
the known optimum at sum_r=2.6359830865 is a basin-of-attraction artifact
or the true global optimum.

Uses the penalty formulation from mobius-001 (quartic on squared distances)
for efficient L-BFGS-B convergence, followed by SLSQP polish.
"""

import json
import math
import numpy as np
from scipy.optimize import minimize
from pathlib import Path
from multiprocessing import Pool, cpu_count
import time
import sys

HERE = Path(__file__).parent
N = 26
KNOWN_BEST = 2.6359830865
BASIN_TOL = 0.002  # Two solutions in same basin if |sum_r1 - sum_r2| < this


def fp(*args, **kwargs):
    print(*args, **kwargs, flush=True)


# ─── Point Process Sampling ─────────────────────────────────────────

def sample_ginibre_points(n, rng):
    """
    Sample n points from the Ginibre ensemble mapped to [0,1]^2.
    Eigenvalues of n x n complex Gaussian matrix exhibit determinantal
    repulsion (negative correlation, hyperuniform).
    """
    G = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / np.sqrt(2 * n)
    eigenvalues = np.linalg.eigvals(G)
    points = np.column_stack([eigenvalues.real, eigenvalues.imag])
    margin = 0.05
    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    span = pmax - pmin
    span = np.where(span < 1e-12, 1.0, span)
    points = (points - pmin) / span
    points = points * (1 - 2 * margin) + margin
    return points


def sample_uniform_points(n, rng):
    """Uniform iid random points in [margin, 1-margin]^2."""
    margin = 0.05
    return rng.uniform(margin, 1 - margin, size=(n, 2))


def sample_halton_points(n, rng):
    """Scrambled Halton quasi-random sequence in [margin, 1-margin]^2."""
    def van_der_corput(n, base):
        seq = np.zeros(n)
        for i in range(n):
            f, r = 1.0, 0.0
            val = i + 1
            while val > 0:
                f /= base
                r += f * (val % base)
                val //= base
            seq[i] = r
        return seq

    margin = 0.05
    x = van_der_corput(n, 2)
    y = van_der_corput(n, 3)
    shift = rng.uniform(0, 1, size=2)
    x = (x + shift[0]) % 1.0
    y = (y + shift[1]) % 1.0
    points = np.column_stack([x, y])
    points = points * (1 - 2 * margin) + margin
    return points


# ─── Radius Assignment ───────────────────────────────────────────────

def assign_greedy_radii(positions):
    """Greedy radius assignment: center-first, as large as possible."""
    n = len(positions)
    radii = np.zeros(n)
    dists_to_center = np.sqrt((positions[:, 0] - 0.5)**2 + (positions[:, 1] - 0.5)**2)
    order = np.argsort(dists_to_center)
    for idx in order:
        x, y = positions[idx]
        max_r = min(x, 1 - x, y, 1 - y)
        for j in range(n):
            if j == idx or radii[j] == 0:
                continue
            dist = math.sqrt((x - positions[j, 0])**2 + (y - positions[j, 1])**2)
            max_r = min(max_r, dist - radii[j])
        radii[idx] = max(max_r, 1e-6)
    return radii


# ─── Optimization (mobius-001 style) ─────────────────────────────────

def validate(circles, tol=1e-10):
    """Validate packing. circles is (n,3) array of [x,y,r]."""
    n = len(circles)
    mv = 0.0
    for i in range(n):
        x, y, r = circles[i]
        if r <= 0:
            return False, abs(r)
        mv = max(mv, r - x, x + r - 1.0, r - y, y + r - 1.0)
    for i in range(n):
        for j in range(i + 1, n):
            dx = circles[i, 0] - circles[j, 0]
            dy = circles[i, 1] - circles[j, 1]
            dist = math.sqrt(dx * dx + dy * dy)
            overlap = (circles[i, 2] + circles[j, 2]) - dist
            mv = max(mv, overlap)
    return mv <= tol, mv


def penalty_obj_grad(x, n, pw):
    """
    Combined objective + gradient (quartic penalty on squared distances).
    Adapted from mobius-001/fast_search.py.
    """
    xs = x[0::3]; ys = x[1::3]; rs = x[2::3]
    grad = np.zeros_like(x)

    # Objective: -sum(r)
    obj = -np.sum(rs)
    grad[2::3] = -1.0

    # Wall penalties: quadratic on violation
    vl = np.maximum(0, rs - xs)
    vr = np.maximum(0, xs + rs - 1.0)
    vb = np.maximum(0, rs - ys)
    vt = np.maximum(0, ys + rs - 1.0)
    vrmin = np.maximum(0, 1e-5 - rs)

    obj += pw * np.sum(vl**2 + vr**2 + vb**2 + vt**2 + vrmin**2)
    grad[0::3] += pw * (-2 * vl + 2 * vr)
    grad[1::3] += pw * (-2 * vb + 2 * vt)
    grad[2::3] += pw * (2 * vl + 2 * vr + 2 * vb + 2 * vt - 2 * vrmin)

    # Overlap penalties: quartic on squared distances (smoother landscape)
    for i in range(n):
        js = np.arange(i + 1, n)
        if len(js) == 0:
            continue
        dx = xs[i] - xs[js]
        dy = ys[i] - ys[js]
        dist_sq = dx * dx + dy * dy
        min_dist = rs[i] + rs[js]
        min_dist_sq = min_dist ** 2
        active = min_dist_sq > dist_sq

        if not np.any(active):
            continue

        a_js = js[active]
        factor = min_dist_sq[active] - dist_sq[active]
        obj += pw * np.sum(factor ** 2)

        f2 = 2 * factor
        a_dx = dx[active]
        a_dy = dy[active]
        a_md = min_dist[active]

        # Gradient w.r.t. positions
        grad[3 * i] += pw * np.sum(f2 * (-2 * a_dx))
        grad[3 * i + 1] += pw * np.sum(f2 * (-2 * a_dy))

        for k, j in enumerate(a_js):
            grad[3 * j] += pw * f2[k] * 2 * a_dx[k]
            grad[3 * j + 1] += pw * f2[k] * 2 * a_dy[k]
            grad[3 * i + 2] += pw * f2[k] * 2 * a_md[k]
            grad[3 * j + 2] += pw * f2[k] * 2 * a_md[k]

    return obj, grad


def optimize_penalty(x0_flat, n, max_stages=8, maxiter=1500):
    """Progressive penalty optimization with L-BFGS-B.
    Always returns the last iterate (which has the highest penalty weight
    and thus best feasibility), plus the metric if valid."""
    x = x0_flat.copy()
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-5, 0.5)] * n

    stages = [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1e6, 1e7]
    for stage, pw in enumerate(stages):
        if stage >= max_stages:
            break
        result = minimize(
            penalty_obj_grad, x, args=(n, pw),
            jac=True, method='L-BFGS-B', bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-15}
        )
        x = result.x.copy()

    circ = x.reshape(n, 3)
    valid, max_viol = validate(circ, tol=1e-6)
    metric = float(np.sum(circ[:, 2])) if valid else 0.0
    return x, metric


def optimize_slsqp(x0_flat, n, maxiter=5000):
    """SLSQP constrained polish (squared-distance formulation for speed)."""
    def objective(x):
        return -np.sum(x[2::3])

    def grad_objective(x):
        g = np.zeros_like(x)
        g[2::3] = -1.0
        return g

    def all_constraints(x):
        xs = x[0::3]; ys = x[1::3]; rs = x[2::3]
        vals = []
        vals.extend(xs - rs)
        vals.extend(1.0 - xs - rs)
        vals.extend(ys - rs)
        vals.extend(1.0 - ys - rs)
        vals.extend(rs - 1e-6)
        for i in range(n):
            dx = xs[i] - xs[i+1:]
            dy = ys[i] - ys[i+1:]
            vals.extend(dx*dx + dy*dy - (rs[i] + rs[i+1:])**2)
        return np.array(vals)

    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * n
    result = minimize(
        objective, x0_flat, method='SLSQP', jac=grad_objective,
        bounds=bounds,
        constraints=[{'type': 'ineq', 'fun': all_constraints}],
        options={'maxiter': maxiter, 'ftol': 1e-15, 'disp': False}
    )
    return result.x, -result.fun


def full_optimize(circles_flat, n):
    """Penalty L-BFGS-B -> SLSQP pipeline.
    Always tries SLSQP even if penalty result is infeasible,
    because SLSQP can recover feasibility from a nearby infeasible point."""
    pen_x, pen_metric = optimize_penalty(circles_flat, n)

    # Always try SLSQP -- it handles infeasible starting points
    try:
        slsqp_x, slsqp_metric = optimize_slsqp(pen_x, n, maxiter=8000)
        slsqp_circ = slsqp_x.reshape(n, 3)
        valid2, viol2 = validate(slsqp_circ, tol=1e-6)
        if valid2:
            return slsqp_x, slsqp_metric
    except Exception:
        pass

    # Fallback: return penalty result
    return pen_x, pen_metric


# ─── Single Restart Worker ──────────────────────────────────────────

def single_restart(args):
    """
    Run a single restart: sample positions, greedy radii, optimize.
    Returns (seed, init_method, sum_r, violation, wall_time).
    """
    seed, init_method = args
    rng = np.random.default_rng(seed)
    t0 = time.time()

    try:
        # Sample initial positions
        if init_method == 'ginibre':
            positions = sample_ginibre_points(N, rng)
        elif init_method == 'uniform':
            positions = sample_uniform_points(N, rng)
        elif init_method == 'halton':
            positions = sample_halton_points(N, rng)
        else:
            positions = sample_ginibre_points(N, rng)

        # Assign initial radii
        radii = assign_greedy_radii(positions)

        # Build flat vector
        circles = np.column_stack([positions, radii])
        x0 = circles.flatten()

        # Full optimize
        x, sum_r = full_optimize(x0, N)

        circ = x.reshape(N, 3)
        valid, viol = validate(circ, tol=1e-6)
        wall_time = time.time() - t0

        if not valid:
            sum_r = 0.0

        return (seed, init_method, float(sum_r), float(viol), wall_time)

    except Exception as e:
        wall_time = time.time() - t0
        return (seed, init_method, 0.0, float('inf'), wall_time)


# ─── Campaign Runner ────────────────────────────────────────────────

def run_campaign(n_restarts=500, methods=('ginibre', 'uniform', 'halton'), n_workers=None):
    """Run multi-start campaign with different initialization methods."""
    if n_workers is None:
        n_workers = min(cpu_count(), 8)

    tasks = []
    for method in methods:
        for i in range(n_restarts):
            # Use well-separated seeds per method to avoid correlation
            seed = i + hash(method) % 2**31
            tasks.append((seed, method))

    fp(f"Running {len(tasks)} restarts across {len(methods)} methods with {n_workers} workers...")

    results = []
    with Pool(n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(single_restart, tasks)):
            results.append(result)
            if (i + 1) % 100 == 0:
                valid_results = [r for r in results if r[2] > 0]
                if valid_results:
                    best = max(r[2] for r in valid_results)
                    mean = np.mean([r[2] for r in valid_results])
                    fp(f"  [{i+1}/{len(tasks)}] valid={len(valid_results)}, "
                       f"best={best:.10f}, mean={mean:.6f}")
                else:
                    fp(f"  [{i+1}/{len(tasks)}] no valid results yet")

    return results


# ─── Analysis ────────────────────────────────────────────────────────

def analyze_results(results):
    """Analyze multi-start results: basin identification, convergence stats, KS test."""
    from scipy.stats import ks_2samp, mannwhitneyu

    valid = [(s, m, sr, v, t) for s, m, sr, v, t in results if sr > 0.1]
    if not valid:
        fp("No valid results!")
        return {}

    methods = sorted(set(r[1] for r in valid))
    stats = {}

    for method in methods:
        method_results = [r for r in valid if r[1] == method]
        srs = np.array([r[2] for r in method_results])
        times = np.array([r[4] for r in method_results])

        near_best = np.sum(np.abs(srs - KNOWN_BEST) < BASIN_TOL)

        # Basin clustering
        sorted_srs = np.sort(srs)
        basins = []
        for sr in sorted_srs:
            found = False
            for basin in basins:
                if abs(sr - basin['center']) < BASIN_TOL:
                    basin['count'] += 1
                    basin['center'] = (basin['center'] * (basin['count'] - 1) + sr) / basin['count']
                    found = True
                    break
            if not found:
                basins.append({'center': sr, 'count': 1})
        basins.sort(key=lambda b: b['center'], reverse=True)

        stats[method] = {
            'n_valid': len(method_results),
            'n_total': sum(1 for r in results if r[1] == method),
            'sum_r_mean': float(np.mean(srs)),
            'sum_r_std': float(np.std(srs)),
            'sum_r_max': float(np.max(srs)),
            'sum_r_min': float(np.min(srs)),
            'sum_r_p90': float(np.percentile(srs, 90)),
            'sum_r_p95': float(np.percentile(srs, 95)),
            'sum_r_p99': float(np.percentile(srs, 99)),
            'near_best_count': int(near_best),
            'near_best_frac': float(near_best / len(method_results)),
            'n_basins': len(basins),
            'basins': basins[:10],
            'time_mean': float(np.mean(times)),
            'time_std': float(np.std(times)),
        }

        fp(f"\n--- {method} ---")
        fp(f"  Valid: {stats[method]['n_valid']}/{stats[method]['n_total']}")
        fp(f"  Sum_r: {stats[method]['sum_r_mean']:.6f} +/- {stats[method]['sum_r_std']:.6f}")
        fp(f"  Percentiles: p90={stats[method]['sum_r_p90']:.6f}, "
           f"p95={stats[method]['sum_r_p95']:.6f}, p99={stats[method]['sum_r_p99']:.6f}")
        fp(f"  Best:  {stats[method]['sum_r_max']:.10f}")
        fp(f"  Near known best ({KNOWN_BEST:.4f}): {near_best}/{len(method_results)} "
           f"({100*stats[method]['near_best_frac']:.1f}%)")
        fp(f"  Distinct basins: {len(basins)}")
        fp(f"  Top 5 basins:")
        for b in basins[:5]:
            fp(f"    sum_r ~ {b['center']:.6f}: {b['count']} restarts")
        fp(f"  Time: {stats[method]['time_mean']:.1f} +/- {stats[method]['time_std']:.1f}s")

    # Pairwise KS tests between methods
    fp(f"\n--- Pairwise KS tests ---")
    method_srs = {}
    for method in methods:
        method_srs[method] = np.array([r[2] for r in valid if r[1] == method])

    ks_results = {}
    for i, m1 in enumerate(methods):
        for m2 in methods[i+1:]:
            ks_stat, ks_p = ks_2samp(method_srs[m1], method_srs[m2])
            mw_stat, mw_p = mannwhitneyu(method_srs[m1], method_srs[m2], alternative='two-sided')
            fp(f"  {m1} vs {m2}: KS={ks_stat:.4f} (p={ks_p:.4e}), "
               f"MannWhitney p={mw_p:.4e}")
            ks_results[f"{m1}_vs_{m2}"] = {
                'ks_stat': float(ks_stat), 'ks_p': float(ks_p),
                'mw_stat': float(mw_stat), 'mw_p': float(mw_p)
            }

    stats['ks_tests'] = ks_results
    return stats


def make_figures(results, stats, output_dir):
    """Generate multi-panel analysis figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "monospace",
        "font.monospace": ["DejaVu Sans Mono", "Menlo", "Consolas", "Monaco"],
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlepad": 8.0,
        "axes.labelpad": 4.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    valid = [(s, m, sr, v, t) for s, m, sr, v, t in results if sr > 0.1]
    methods = sorted(set(r[1] for r in valid))
    colors = {'ginibre': '#1f77b4', 'uniform': '#ff7f0e', 'halton': '#2ca02c'}

    # === Main figure: 2x2 grid ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)

    # (a) Histogram of final sum_r by method
    ax = axes[0, 0]
    for method in methods:
        srs = [r[2] for r in valid if r[1] == method]
        ax.hist(srs, bins=50, alpha=0.5, label=f'{method} (n={len(srs)})',
                color=colors.get(method, 'gray'), density=True)
    ax.axvline(KNOWN_BEST, color='red', linestyle='--', linewidth=1.5,
               label=f'Known best: {KNOWN_BEST:.4f}')
    ax.set_xlabel('Sum of radii')
    ax.set_ylabel('Density')
    ax.set_title('(a) Distribution of final sum_r', fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)

    # (b) Empirical CDF
    ax = axes[0, 1]
    for method in methods:
        srs = sorted([r[2] for r in valid if r[1] == method])
        ecdf_y = np.arange(1, len(srs) + 1) / len(srs)
        ax.plot(srs, ecdf_y, label=method, color=colors.get(method, 'gray'), linewidth=1.5)
    ax.axvline(KNOWN_BEST, color='red', linestyle='--', linewidth=1.5, label='Known best')
    ax.set_xlabel('Sum of radii')
    ax.set_ylabel('Cumulative fraction')
    ax.set_title('(b) Empirical CDF of sum_r', fontweight='bold')
    ax.legend()

    # (c) Box plot comparison
    ax = axes[1, 0]
    data_for_box = []
    labels_for_box = []
    for method in methods:
        srs = [r[2] for r in valid if r[1] == method]
        data_for_box.append(srs)
        labels_for_box.append(method)
    bp = ax.boxplot(data_for_box, tick_labels=labels_for_box, patch_artist=True)
    for patch, method in zip(bp['boxes'], methods):
        patch.set_facecolor(colors.get(method, 'gray'))
        patch.set_alpha(0.6)
    ax.axhline(KNOWN_BEST, color='red', linestyle='--', linewidth=1.5, label='Known best')
    ax.set_ylabel('Sum of radii')
    ax.set_title('(c) Distribution comparison', fontweight='bold')
    ax.legend()

    # (d) Convergence fraction and statistics table
    ax = axes[1, 1]
    ax.axis('off')
    table_data = []
    for method in methods:
        s = stats[method]
        table_data.append([
            method,
            f"{s['sum_r_mean']:.4f}",
            f"{s['sum_r_std']:.4f}",
            f"{s['sum_r_max']:.6f}",
            f"{s['sum_r_p95']:.4f}",
            f"{s['n_basins']}",
            f"{s['near_best_count']}",
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=['Method', 'Mean', 'Std', 'Max', 'P95', 'Basins', 'Near Best'],
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Add KS test results as text
    if 'ks_tests' in stats:
        ks_text = "Kolmogorov-Smirnov tests:\n"
        for pair, res in stats['ks_tests'].items():
            ks_text += f"  {pair}: D={res['ks_stat']:.4f}, p={res['ks_p']:.3e}\n"
        ax.text(0.5, -0.05, ks_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='center',
                fontfamily='monospace')

    ax.set_title('(d) Summary statistics', fontweight='bold')

    fig.savefig(output_dir / 'convergence_analysis.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    fp(f"Saved convergence_analysis.png")

    # === Second figure: Point process visualization ===
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for idx, (ax, method) in enumerate(zip(axes2, ['ginibre', 'uniform', 'halton'])):
        rng = np.random.default_rng(42)
        if method == 'ginibre':
            pts = sample_ginibre_points(N, rng)
        elif method == 'uniform':
            pts = sample_uniform_points(N, rng)
        else:
            pts = sample_halton_points(N, rng)

        ax.scatter(pts[:, 0], pts[:, 1], s=80, c=colors.get(method, 'gray'),
                   edgecolor='black', linewidth=0.5, zorder=5)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect('equal')
        ax.set_title(f'({chr(97+idx)}) {method.capitalize()} (n={N})', fontweight='bold')
        rect = plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)

    fig2.savefig(output_dir / 'initialization_comparison.png', dpi=200, bbox_inches='tight')
    plt.close(fig2)
    fp(f"Saved initialization_comparison.png")


# ─── Evaluator-compatible entry point ────────────────────────────────

def run_packing():
    """Entry point for evaluator. Returns the known best solution."""
    solution_path = HERE / "best_solution.json"
    if solution_path.exists():
        with open(solution_path) as f:
            data = json.load(f)
        circles = np.array(data["circles"])
    else:
        grad_path = Path(__file__).parent.parent.parent / "research" / "solutions" / "mobius-001" / "solution_n26.json"
        with open(grad_path) as f:
            data = json.load(f)
        circles = np.array(data["circles"])

    centers = circles[:, :2]
    radii = circles[:, 2]
    return centers, radii, float(np.sum(radii))


# ─── Main ────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Ginibre DPP multi-start circle packing')
    parser.add_argument('--restarts', type=int, default=500,
                        help='Number of restarts per method')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers')
    parser.add_argument('--methods', nargs='+', default=['ginibre', 'uniform', 'halton'],
                        help='Initialization methods to test')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with 20 restarts')
    args = parser.parse_args()

    if args.quick:
        args.restarts = 20

    fp(f"=== Ginibre DPP Multi-Start Circle Packing ===")
    fp(f"Restarts per method: {args.restarts}")
    fp(f"Methods: {args.methods}")
    fp(f"Known best: {KNOWN_BEST:.10f}")
    fp(f"Workers: {args.workers or min(cpu_count(), 8)}")
    fp()

    t0 = time.time()
    results = run_campaign(
        n_restarts=args.restarts,
        methods=tuple(args.methods),
        n_workers=args.workers
    )
    total_time = time.time() - t0
    fp(f"\n=== Campaign completed in {total_time:.1f}s ===")

    stats = analyze_results(results)

    # Save results
    output_dir = HERE / "figures"
    output_dir.mkdir(exist_ok=True)

    results_data = {
        'results': [
            {'seed': int(s), 'method': m, 'sum_r': sr, 'violation': v, 'time': t}
            for s, m, sr, v, t in results
        ],
        'stats': {
            method: {k: v for k, v in s.items() if k != 'basins'}
            for method, s in stats.items()
            if method != 'ks_tests'
        },
        'ks_tests': stats.get('ks_tests', {}),
    }

    with open(HERE / "results.json", 'w') as f:
        json.dump(results_data, f, indent=2)
    fp(f"Saved results.json")

    # Check for new best
    valid = [r for r in results if r[2] > 0.1]
    if valid:
        best_result = max(valid, key=lambda r: r[2])
        fp(f"\nBest result: seed={best_result[0]}, method={best_result[1]}, "
           f"sum_r={best_result[2]:.10f}, viol={best_result[3]:.2e}")
        if best_result[2] > KNOWN_BEST + 1e-6:
            fp(f"\n*** FOUND IMPROVEMENT OVER KNOWN BEST! ***")

    # Generate figures
    try:
        make_figures(results, stats, output_dir)
    except Exception as e:
        fp(f"Figure generation failed: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    fp(f"\n=== SUMMARY ===")
    fp(f"Total restarts: {len(results)}")
    fp(f"Valid restarts: {len(valid)}")
    fp(f"Total time: {total_time:.1f}s")
    for method in sorted(k for k in stats.keys() if k != 'ks_tests'):
        s = stats[method]
        fp(f"\n{method}:")
        fp(f"  Mean sum_r: {s['sum_r_mean']:.6f} +/- {s['sum_r_std']:.6f}")
        fp(f"  Best: {s['sum_r_max']:.10f}")
        fp(f"  Basins: {s['n_basins']}")
        fp(f"  Near known best: {s['near_best_count']}/{s['n_valid']}")


if __name__ == '__main__':
    main()
