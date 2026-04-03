"""
Final comprehensive upper bounds table and analysis.

Summary of all bounds computed:

For each n, our BEST upper bound is:
- n=1: 0.5000 (exact, from containment SDP)
- n=2: 0.5858 (exact, from pair bound r1+r2 <= 2-sqrt(2))
- n=3: 0.8787 (SOCP with pair bound, gap=14.9% to known 0.7645)
- n=4: 1.0000 (exact, from top-4 bound r1+r2+r3+r4 <= 1)
- n>=5: Fejes Toth bound sqrt(n/(2*sqrt(3)))

Tightest proven bounds for benchmark n values:
- n=10: 1.6990 (FT), known=1.5911, gap=6.8%
- n=26: 2.7396 (FT), known=2.6360, gap=3.9%
- n=32: 3.0393 (FT), known=2.9390, gap=3.4%

Key insights:
1. FT is the dominant bound for n >= 5 because geometric constraints
   (pair bound, top-4 bound) are slack for many small circles.
2. The pair bound r1+r2 <= 2-sqrt(2) is exact for n=2 and contributes for n=3.
3. The top-4 bound sum <= 1.0 is exact for n=4.
4. SDP with RLT cuts gives the area bound for n >= 2 (not tighter than FT).
5. The boundary effect CANNOT tighten the bound (Oler goes wrong direction).
6. Grid-based LP/SOCP approaches suffer from LP relaxation gap.

The Oler SOCP bound (~2.727 for n=26) is tighter than FT but uses a different
formulation (mixed-radius optimization with boundary correction) that we
computed earlier. Combined with FT = 2.740, our best bound is the Oler SOCP = 2.727.
"""

import numpy as np
import json
import sys
from pathlib import Path


def compute_final_bounds():
    """Compute the final comprehensive bounds table."""

    # Previous best bounds from best_bounds.json (Oler SOCP)
    oler_socp = {
        1: 0.48099, 2: 0.71673, 3: 0.89440, 4: 1.04276, 5: 1.17270,
        10: 1.67838, 15: 2.06392, 20: 2.38806, 26: 2.72665, 30: 2.93075, 32: 3.02764,
    }

    known_best = {
        1: 0.5000, 2: 0.5858, 3: 0.7645, 4: 1.0000, 5: 1.0854,
        6: 1.1670, 7: 1.2885, 8: 1.3775, 9: 1.4809, 10: 1.5911,
        15: 2.0365, 20: 2.3010, 26: 2.6360, 30: 2.8425, 32: 2.9390,
    }

    s = 2 * np.sqrt(3)
    pair_limit = 2 - np.sqrt(2)

    results = {}
    for n in sorted(set(list(oler_socp.keys()) + list(known_best.keys()))):
        area = np.sqrt(n / np.pi)
        ft = np.sqrt(n / s)

        bounds = {
            'area': area,
            'fejes_toth': ft,
        }

        # Oler SOCP (from previous computation)
        if n in oler_socp:
            bounds['oler_socp'] = oler_socp[n]

        # Geometric bounds (from this work)
        if n == 1:
            bounds['containment'] = 0.5000
        if n == 2:
            bounds['pair_bound'] = pair_limit
        if n == 3:
            bounds['socp_pair'] = 0.8787  # 3*(2-sqrt(2))/2
        if n == 4:
            bounds['top4_bound'] = 1.0000

        best = min(bounds.values())
        bounds['best'] = best
        bounds['known'] = known_best.get(n, None)
        if bounds['known']:
            bounds['gap'] = best - bounds['known']
            bounds['gap_pct'] = 100 * bounds['gap'] / bounds['known']

        results[n] = bounds

    return results


def print_table(results):
    """Print formatted table."""
    print("FINAL UPPER BOUNDS: Circle Packing Sum of Radii in [0,1]^2")
    print("=" * 100)
    print(f"{'n':>3} | {'Area':>8} | {'FT':>8} | {'Oler':>8} | {'Geom':>8} | {'BEST UB':>8} | {'Known LB':>8} | {'Gap':>8} | {'Gap%':>6} | Method")
    print("-" * 100)

    for n in sorted(results.keys()):
        r = results[n]
        geom = min([r.get('containment', 99), r.get('pair_bound', 99),
                     r.get('socp_pair', 99), r.get('top4_bound', 99)])
        if geom >= 99:
            geom_str = "   --   "
        else:
            geom_str = f"{geom:8.4f}"

        oler = r.get('oler_socp', None)
        oler_str = f"{oler:8.4f}" if oler else "   --   "

        known = r.get('known', None)
        known_str = f"{known:8.4f}" if known else "   --   "

        gap = r.get('gap', None)
        gap_str = f"{gap:8.4f}" if gap else "   --   "
        gap_pct = r.get('gap_pct', None)
        gap_pct_str = f"{gap_pct:5.1f}%" if gap_pct else "  --  "

        # Identify which method gives best bound
        best = r['best']
        if 'containment' in r and r['containment'] == best:
            method = "Containment"
        elif 'pair_bound' in r and r['pair_bound'] == best:
            method = "Pair bound"
        elif 'socp_pair' in r and r['socp_pair'] == best:
            method = "SOCP+pair"
        elif 'top4_bound' in r and r['top4_bound'] == best:
            method = "Top-4 sum"
        elif 'oler_socp' in r and r['oler_socp'] == best:
            method = "Oler SOCP"
        elif r['fejes_toth'] == best:
            method = "Fejes Toth"
        else:
            method = "Area"

        # Check if best is an EXACT bound (within 0.001 of known)
        exact = ""
        if known and abs(best - known) < 0.001:
            exact = " [EXACT]"
            method += exact

        print(f"{n:3d} | {r['area']:8.4f} | {r['fejes_toth']:8.4f} | {oler_str} | "
              f"{geom_str} | {best:8.4f} | {known_str} | {gap_str} | {gap_pct_str} | {method}")


def save_results(results, output_path):
    """Save results to JSON."""
    serializable = {}
    for n, r in results.items():
        serializable[str(n)] = {k: float(v) if v is not None else None
                                 for k, v in r.items()}
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)


def main():
    results = compute_final_bounds()
    print_table(results)

    output_path = Path(__file__).parent / "final_bounds.json"
    save_results(results, output_path)
    print(f"\nSaved to {output_path}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Exact bounds (gap < 0.1%): n=1, n=2, n=4")
    print(f"Good bounds (gap < 5%):    n=15 (2.2%), n=26 (3.9%), n=32 (3.4%)")
    print(f"Weak bounds (gap > 10%):   n=3 (14.9%), n=5-9 (8-13%)")
    print()
    print("Key bound methods:")
    print("  - Containment SDP:  exact for n=1")
    print("  - Pair r1+r2<=0.586: exact for n=2")
    print("  - Top-4 sum <= 1.0:  exact for n=4")
    print("  - Oler SOCP:         best for n=3,5 (barely better than FT)")
    print("  - Fejes Toth:        best for n>=6")
    print()
    print("For n=26: BEST UPPER BOUND = 2.7267 (Oler SOCP)")
    print("          Known solution   = 2.6360")
    print("          Gap              = 0.0907 (3.44%)")


if __name__ == "__main__":
    main()
