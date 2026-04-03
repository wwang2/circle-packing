"""
Corrected upper bounds for circle packing sum-of-radii.

BUG FOUND: The Oler SOCP bound in best_bound.py was INVALID because it
dropped the linear term 4r from the Oler-Groemer inequality.

CORRECT Oler-Groemer inequality for n equal circles of radius r in unit square:
  n * 2*sqrt(3) * r^2 <= 1 + 4*r + pi*r^2

This can be rewritten as:
  (2*sqrt(3)*n - pi) * r^2 - 4*r - 1 <= 0

For n=1: (2*sqrt(3) - pi)*r^2 - 4*r - 1 <= 0
  = (3.464 - 3.142)*r^2 - 4r - 1 = 0.322*r^2 - 4r - 1 <= 0
  At r=0.5: 0.322*0.25 - 2 - 1 = 0.081 - 3 = -2.919 <= 0. YES, valid.
  So Oler doesn't constrain n=1 at all (it's always satisfied).

For MIXED radii, the correct Oler-Groemer generalization:
  sum_i 2*sqrt(3)*r_i^2 <= 1 + 4*r_max + pi*r_max^2

Note: 1 + 4*r_max + pi*r_max^2 > 1 for r_max > 0.
So the Oler bound for mixed radii is WEAKER than FT!

Wait, that means Oler NEVER helps as an upper bound?!

Let me re-derive. The Fejes Toth bound says:
  Packing density <= pi/(2*sqrt(3)) (hex density)
  => sum(pi*r_i^2) / Area(K) <= pi/(2*sqrt(3))
  => sum(r_i^2) <= Area(K) / (2*sqrt(3))
  => sum(r_i^2) <= 1 / (2*sqrt(3))  for unit square

The Oler bound uses the "expanded area" (Minkowski sum of K with disk of radius r_max):
  => sum(2*sqrt(3)*r_i^2) <= Area(K_{r_max}) = 1 + 4*r_max + pi*r_max^2

Since Area(K_{r_max}) > Area(K) = 1, this is ALWAYS WEAKER than FT.
Oler's theorem is meant for COUNTING circles (given fixed radius r),
where the boundary correction HELPS by allowing more circles.
For MAXIMIZING sum of radii (our problem), it makes the bound WORSE.

So the CORRECT best bound is always FT, not Oler!

The earlier "Oler SOCP" results were WRONG because:
1. They dropped the 4r linear term (making the bound artificially tight)
2. This produced bounds BELOW the true optimum for small n (e.g., n=1: 0.481 < 0.500)

CORRECTED bounds:
- For n >= 5: FT = sqrt(n/(2*sqrt(3))) is the best available bound
- For n=1: 0.500 (containment)
- For n=2: 0.5858 (pair bound)
- For n=3: 0.8787 (SOCP with pair cuts)
- For n=4: 1.000 (top-4 sum)

HOWEVER: there IS a valid use of Oler for getting tighter bounds.
The "inner parallel body" version of Oler states:
  n * 2*sqrt(3)*r^2 <= A_inner + perimeter_inner * ???

Actually, the standard result is:
For n non-overlapping disks of radius r whose centers lie in a convex body K_r
(K shrunk by r, the set of valid centers):
  n <= Area(K_r) / (2*sqrt(3)*r^2)

K_r for unit square: centers in [r, 1-r]^2, area = (1-2r)^2.
  n <= (1-2r)^2 / (2*sqrt(3)*r^2)

For equal radii: n*2*sqrt(3)*r^2 <= (1-2r)^2

This is TIGHTER than FT for r close to 0.5:
  FT: n*2*sqrt(3)*r^2 <= 1
  Inner: n*2*sqrt(3)*r^2 <= (1-2r)^2

Since (1-2r)^2 < 1 for 0 < r < 0.5, the inner body bound is TIGHTER!

For mixed radii:
  Each circle i's center is in [r_i, 1-r_i]^2, but the "area" argument
  uses a COMMON shrinking radius. This doesn't directly generalize.

  However, we can use the SMALLEST radius r_min = r_n as the shrinking:
  All centers are in [r_min, 1-r_min]^2 (since r_i >= r_min => center domain includes this).
  Area = (1-2*r_min)^2.

  But each circle's Voronoi area is >= 2*sqrt(3)*r_i^2, and the total Voronoi area
  within [r_min, 1-r_min]^2 is at most (1-2*r_min)^2.

  WAIT: the Voronoi cells can extend OUTSIDE [r_min, 1-r_min]^2.
  The bound should be: sum(Voronoi within center domain) <= area(center domain).
  But Voronoi cells extend beyond the center domain, so this isn't valid.

Actually, the standard FT argument for disks in a convex body K is:
  The Voronoi cells (within K, assigning each point in K to nearest center)
  partition K. Each Voronoi cell has area >= 2*sqrt(3)*r_i^2.
  So sum(2*sqrt(3)*r_i^2) <= Area(K).

For K = [0,1]^2: Area(K) = 1. This gives FT.

For K' = [r_min, 1-r_min]^2 (center domain): this is NOT the right K.
The circles exist in [0,1]^2, not in the center domain.
The Voronoi partition is of [0,1]^2, not of the center domain.

So the correct bound is sum(2*sqrt(3)*r_i^2) <= 1 (FT).

The inner parallel body bound is for COUNTING equal circles:
  Centers in [r, 1-r]^2, each needs Voronoi area >= 2*sqrt(3)*r^2.
  The Voronoi cells partition [r, 1-r]^2 (approximately -- they extend to
  the boundary of the plane, but we restrict to points closest to a center
  that's in [r, 1-r]^2). Actually, non-center points in [0,1]^2 but outside
  [r, 1-r]^2 are NOT claimed by any center's Voronoi cell... unless the
  Voronoi diagram extends.

This is subtle. Let me just compute correct bounds.

Actually, I realize the FT bound per circle is a result about ANY packing
in ANY convex body. The result says: in a convex body K, if n non-overlapping
disks of radii r_1,...,r_n are contained in K, then
  sum(pi*r_i^2) <= delta_hex * Area(K)
where delta_hex = pi/(2*sqrt(3)).

This gives sum(r_i^2) <= 1/(2*sqrt(3)).

This is the BEST known bound of this type (area-based) for ARBITRARY convex bodies.
It's tight as n -> infinity (hexagonal packing achieves density delta_hex).

For a SPECIFIC convex body like the unit square, can we do better?
YES: the Groemer-Oler correction says:
  sum(pi*r_i^2) <= delta_hex * Area(K) + f(perimeter, r_max)
where f is a boundary correction that INCREASES the RHS.
This is meant for counting: it says MORE circles fit than the area bound suggests.
For sum-of-radii maximization, this WEAKENS the bound.

There IS a dual direction: for circles NEAR the boundary, the packing is
LESS efficient than hex. A circle near a straight boundary has a semi-hexagonal
Voronoi cell with MORE area than the interior hexagonal cell.

Claim: for a disk of radius r whose center is at distance d from a straight
boundary (d >= r), the Voronoi cell has area >= 2*sqrt(3)*r^2 + f(d,r)
where f > 0 when d is small relative to the hex cell radius.

If this is true: boundary circles WASTE more Voronoi area, giving:
  sum(2*sqrt(3)*r_i^2) + sum(f_i) <= 1
  sum(2*sqrt(3)*r_i^2) <= 1 - sum(f_i)

This would be TIGHTER than FT! The boundary waste f_i > 0 reduces the budget.

But I don't have a rigorous formula for f_i. This would require a careful
geometric argument about Voronoi cells near straight boundaries.

For now, FT = 2.7396 is our best bound for n=26.

Let me compute the CORRECT Oler equal-radius bound as a VALID upper bound.
For equal radii r: n*2*sqrt(3)*r^2 <= 1 + 4*r + pi*r^2 (Groemer/Oler)
  r satisfies quadratic: (2*sqrt(3)*n - pi)*r^2 - 4*r - 1 <= 0
  Sum = n*r.

  Solving: r <= [4 + sqrt(16 + 4*(2*sqrt(3)*n-pi))] / (2*(2*sqrt(3)*n-pi))
  = [2 + sqrt(4 + 2*sqrt(3)*n - pi)] / (2*sqrt(3)*n - pi)

For n=26: 2*sqrt(3)*26 - pi = 90.066 - 3.142 = 86.924
  r <= [2 + sqrt(4 + 86.924)] / 86.924 = [2 + sqrt(90.924)] / 86.924
  = [2 + 9.536] / 86.924 = 11.536 / 86.924 = 0.1328
  Sum = 26 * 0.1328 = 3.452

This is HIGHER than FT (2.740). So Oler equal-radius gives a WEAKER bound. Confirmed.

The equal-radius Oler can never beat FT for sum-of-radii.

So FT remains our best bound. Period.
"""

import numpy as np
import cvxpy as cp
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def fejes_toth_bound(n):
    return np.sqrt(n / (2*np.sqrt(3)))


def socp_bound(n, verbose=False):
    """Best SOCP bound with geometric cuts."""
    s = 2 * np.sqrt(3)
    pair_limit = 2 - np.sqrt(2)

    r = cp.Variable(n)
    constraints = [r >= 0, r <= 0.5]

    for i in range(n-1):
        constraints += [r[i] >= r[i+1]]

    # FT area
    constraints += [s * cp.sum_squares(r) <= 1]

    # Pair bound
    for i in range(min(n, 10)):
        for j in range(i+1, min(n, 10)):
            constraints += [r[i] + r[j] <= pair_limit]

    # Top-4
    if n >= 4:
        constraints += [r[0] + r[1] + r[2] + r[3] <= 1.0]

    # Individual radius bounds from FT
    for k in range(1, min(n+1, 30)):
        max_r_k = 1.0 / np.sqrt(s * k)
        if max_r_k < 0.5:
            constraints += [r[k-1] <= max_r_k]

    # Top-k sum bounds from FT
    for k in range(5, n):
        constraints += [cp.sum(r[:k]) <= fejes_toth_bound(k)]

    objective = cp.Maximize(cp.sum(r))
    prob = cp.Problem(objective, constraints)

    try:
        result = prob.solve(solver=cp.SCS, verbose=False, max_iters=50000, eps=1e-9)
        if verbose and r.value is not None:
            print(f"  SOCP (n={n}): {result:.6f}")
        return result
    except:
        return None


def compute_all_bounds():
    """Compute corrected bounds for all n."""
    known_best = {
        1: 0.5000, 2: 0.5858, 3: 0.7645, 4: 1.0000, 5: 1.0854,
        6: 1.1670, 7: 1.2885, 8: 1.3775, 9: 1.4809, 10: 1.5911,
        15: 2.0365, 20: 2.3010, 26: 2.6360, 30: 2.8425, 32: 2.9390,
    }

    n_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 26, 30, 32]
    results = {}

    print("CORRECTED Upper Bounds for Circle Packing Sum of Radii")
    print("=" * 95)
    print(f"{'n':>3} | {'Area':>8} | {'FT':>8} | {'SOCP':>8} | {'BEST UB':>8} | "
          f"{'Known LB':>8} | {'Gap':>8} | {'Gap%':>6} | Method")
    print("-" * 95)

    for n in n_values:
        area = np.sqrt(n / np.pi)
        ft = fejes_toth_bound(n)
        socp = socp_bound(n, verbose=False)

        bounds = {'area': area, 'fejes_toth': ft}
        if socp is not None and socp > 0:
            bounds['socp'] = socp

        best = min(v for v in bounds.values() if v is not None and v > 0)

        # Determine method
        if socp and abs(socp - best) < 1e-4:
            if abs(socp - ft) < 1e-4:
                method = "FT"
            else:
                method = "SOCP"
        elif abs(ft - best) < 1e-4:
            method = "FT"
        else:
            method = "Area"

        known = known_best.get(n, None)
        gap = best - known if known else None
        gap_pct = 100 * gap / known if gap is not None else None

        # Special methods for exact results
        if known and abs(best - known) < 0.001:
            method += " [EXACT]"

        # Validity check
        valid = True
        if known and best < known - 0.001:
            valid = False

        known_str = f"{known:8.4f}" if known else "    --  "
        gap_str = f"{gap:8.4f}" if gap is not None else "    --  "
        gap_pct_str = f"{gap_pct:5.1f}%" if gap_pct is not None else "  --  "
        socp_str = f"{socp:8.4f}" if socp else "    --  "

        print(f"{n:3d} | {area:8.4f} | {ft:8.4f} | {socp_str} | {best:8.4f} | "
              f"{known_str} | {gap_str} | {gap_pct_str} | {method}"
              f"{' **INVALID**' if not valid else ''}")

        results[str(n)] = {
            'area': float(area),
            'fejes_toth': float(ft),
            'socp': float(socp) if socp else None,
            'best_upper_bound': float(best),
            'known_lower_bound': float(known) if known else None,
            'gap': float(gap) if gap is not None else None,
            'gap_pct': float(gap_pct) if gap_pct is not None else None,
            'method': method,
        }

    return results


def make_figure(results, output_dir):
    """Create a figure showing bounds vs known values."""
    plt.rcParams.update({
        "font.family": "monospace",
        "font.monospace": ["DejaVu Sans Mono", "Menlo", "Consolas", "Monaco"],
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    n_vals = sorted([int(k) for k in results.keys()])
    ubs = [results[str(n)]['best_upper_bound'] for n in n_vals]
    fts = [results[str(n)]['fejes_toth'] for n in n_vals]
    areas = [results[str(n)]['area'] for n in n_vals]
    knowns = [results[str(n)]['known_lower_bound'] for n in n_vals]

    # Left: absolute values
    ax1.plot(n_vals, areas, '--', color='#7fcdbb', linewidth=1.5, label='Area bound', alpha=0.7)
    ax1.plot(n_vals, fts, '-', color='#2c7fb8', linewidth=2, label='Fejes Toth bound')
    ax1.plot(n_vals, ubs, '-o', color='#253494', linewidth=2, markersize=5, label='Best upper bound')
    ax1.scatter(n_vals, knowns, c='#41ae76', s=60, zorder=5, label='Known best (lower bound)')

    # Highlight exact results
    for i_n, n in enumerate([1, 2, 4]):
        idx = n_vals.index(n)
        ax1.scatter([n], [knowns[idx]], c='#f0f921', s=120, zorder=6, edgecolors='black',
                   linewidth=1.5, label='Exact (UB=LB)' if i_n == 0 else '')

    ax1.set_xlabel('Number of circles (n)', fontweight='bold')
    ax1.set_ylabel('Sum of radii', fontweight='bold')
    ax1.set_title('Upper Bounds vs Known Solutions', fontweight='bold')
    ax1.legend(loc='upper left')

    # Right: gap percentage
    gaps = []
    n_with_gaps = []
    for n in n_vals:
        g = results[str(n)].get('gap_pct')
        if g is not None:
            gaps.append(g)
            n_with_gaps.append(n)

    # Use GnBu-inspired colors
    colors = ['#41ae76' if g < 1 else '#2c7fb8' if g < 5 else '#253494' for g in gaps]
    ax2.bar(range(len(n_with_gaps)), gaps, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_xticks(range(len(n_with_gaps)))
    ax2.set_xticklabels([str(n) for n in n_with_gaps])
    ax2.set_xlabel('Number of circles (n)', fontweight='bold')
    ax2.set_ylabel('Gap (%)', fontweight='bold')
    ax2.set_title('Gap: (Upper Bound - Known) / Known', fontweight='bold')
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.axhline(y=5, color='gray', linewidth=1, linestyle='--', alpha=0.5)

    # Add value labels on bars
    for i, (n, g) in enumerate(zip(n_with_gaps, gaps)):
        label = f"{g:.1f}%"
        ax2.text(i, g + 0.3, label, ha='center', va='bottom', fontsize=9)
    fig_path = output_dir / "figures" / "upper_bounds_summary.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nFigure saved to {fig_path}")
    return fig_path


def main():
    results = compute_all_bounds()

    output_dir = Path(__file__).parent
    output_path = output_dir / "corrected_bounds.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY (CORRECTED)")
    print("=" * 70)
    print("Exact bounds: n=1 (0.500), n=2 (0.586), n=4 (1.000)")
    print("Best bound method for n>=5: Fejes Toth (hex packing density)")
    print()
    print("CRITICAL FINDING: The Oler SOCP bounds previously reported were")
    print("INVALID. The Oler-Groemer inequality gives a WEAKER bound than FT")
    print("for sum-of-radii optimization (boundary correction helps packing")
    print("density, not bounding it).")
    print()
    print("For n=26:")
    print(f"  Best upper bound: {results['26']['best_upper_bound']:.4f} (Fejes Toth)")
    print(f"  Known solution:   {results['26']['known_lower_bound']:.4f}")
    print(f"  Gap:              {results['26']['gap']:.4f} ({results['26']['gap_pct']:.1f}%)")
    print()
    print("To improve beyond FT, one would need:")
    print("  1. Higher-order Lasserre SDP (order >= 2, very expensive)")
    print("  2. Problem-specific valid inequalities for many small circles")
    print("  3. Branch-and-bound with SDP/LP relaxations per node")

    # Make figure
    fig_path = make_figure(results, output_dir)

    return results


if __name__ == "__main__":
    main()
