"""
Direct upper bound without Cauchy-Schwarz.

The FT bound is: sqrt(n/(2*sqrt(3))) via C-S applied to sum(r_i^2) <= 1/(2*sqrt(3)).
C-S is tight when all radii are equal. For mixed radii, it's loose.

But do we KNOW radii must be mixed? No -- equal radii might be optimal.
In fact, for many n, the optimal packing has very similar (not equal) radii.

APPROACH: Instead of C-S, use a DIRECT LP over possible radius distributions.

Key idea: define f(r) = maximum number of circles of radius >= r that fit.
Then sum(r_i) = integral_0^0.5 #{i: r_i >= t} dt = integral_0^0.5 N(t) dt
where N(t) = number of circles with radius >= t.

Upper bound: sum(r_i) <= integral_0^0.5 min(n, f(t)) dt.

We need to compute f(t) = max number of circles of radius t in [0,1]^2.

Using Oler: f(t) <= (1 + 4t + pi*t^2) / (2*sqrt(3)*t^2)

And also: f(t) <= n (we have exactly n circles).

Bound: sum r_i <= integral_0^{0.5} min(n, f(t)) dt.

But the issue is that N(t) <= f(t) is too loose for small t.
The actual constraint is: N(t) is DECREASING (since more circles have r >= t
for smaller t). Also N(0+) = n and N(0.5) >= 0.

Actually N(t) = #{i: r_i >= t} is always a step function, decreasing from n to 0.
And N(t) <= f(t) for all t.

Better: we also have AREA constraint at each threshold.
Circles with r_i >= t contribute at least pi*t^2 each to the total area.
So N(t) * pi*t^2 <= 1, giving N(t) <= 1/(pi*t^2).

And FT: N(t) * 2*sqrt(3)*t^2 <= 1, giving N(t) <= 1/(2*sqrt(3)*t^2).

Since 2*sqrt(3) > pi, the FT bound is tighter.

So: sum(r_i) = integral_0^{0.5} N(t) dt <= integral_0^{0.5} min(n, 1/(2*sqrt(3)*t^2)) dt.

The crossover is at t_0 where n = 1/(2*sqrt(3)*t_0^2), i.e., t_0 = 1/sqrt(2*sqrt(3)*n).

integral_0^{t_0} n dt + integral_{t_0}^{0.5} 1/(2*sqrt(3)*t^2) dt
= n*t_0 + 1/(2*sqrt(3)) * [1/t_0 - 1/0.5]
= n/sqrt(2*sqrt(3)*n) + [sqrt(2*sqrt(3)*n) - 2]/(2*sqrt(3))
= sqrt(n/(2*sqrt(3))) + sqrt(n/(2*sqrt(3))) - 2/(2*sqrt(3))
= 2*sqrt(n/(2*sqrt(3))) - 1/sqrt(3)

Wait, let me redo this carefully.

t_0 = 1/sqrt(s*n) where s = 2*sqrt(3).

First part: n * t_0 = n / sqrt(s*n) = sqrt(n/s).
Second part: (1/s) * [1/t_0 - 2] = (1/s) * [sqrt(s*n) - 2] = sqrt(n/s) - 2/s.

Total: 2*sqrt(n/s) - 2/s.

For n=26: s = 2*sqrt(3) = 3.4641.
sqrt(26/3.4641) = sqrt(7.506) = 2.7396.
2 * 2.7396 - 2/3.4641 = 5.479 - 0.577 = 4.902.

Hmm, that's way too big. I think I'm overcounting.

The issue: the constraint N(t) <= 1/(s*t^2) allows different circles to
be counted multiple times at different thresholds.

Actually, this integral bound IS correct as an upper bound on sum(r_i).
Let me verify:

sum(r_i) = sum_i integral_0^{r_i} dt = integral_0^{0.5} N(t) dt

where N(t) = #{i: r_i >= t}. N(t) is decreasing and integer-valued.

The bound N(t) <= min(n, 1/(s*t^2)) is valid.

But the integral of this bound can be much larger than sqrt(n/s)!
Let me check numerically.
"""

import numpy as np
from scipy.integrate import quad
import json
import sys
from pathlib import Path


def integral_bound_ft(n):
    """
    Integral bound: sum(r_i) <= integral_0^0.5 min(n, 1/(2*sqrt(3)*t^2)) dt.

    This is ALWAYS >= the C-S FT bound. So it's WEAKER.
    The C-S trick is what makes FT tight.
    """
    s = 2 * np.sqrt(3)
    t0 = 1.0 / np.sqrt(s * n)

    if t0 >= 0.5:
        return n * 0.5

    part1 = n * t0
    part2 = (1.0 / s) * (1.0 / t0 - 2.0)

    return part1 + part2


def cs_ft_bound(n):
    """C-S + FT bound: sqrt(n/(2*sqrt(3)))."""
    return np.sqrt(n / (2 * np.sqrt(3)))


def integral_bound_oler(n):
    """
    Integral bound using Oler: N(t) <= (1 + 4t + pi*t^2)/(2*sqrt(3)*t^2).
    """
    s = 2 * np.sqrt(3)

    def N_upper(t):
        if t <= 0:
            return n
        oler = (1 + 4*t + np.pi*t**2) / (s * t**2)
        return min(n, oler)

    result, _ = quad(N_upper, 1e-10, 0.5)
    return result


def relaxed_radius_lp(n, verbose=False):
    """
    LP over radius values with multiple constraints.

    Variables: n_k = number of circles in radius bin k.
    The constraint is NOT just sum(2*sqrt(3)*r_k^2 * n_k) <= 1,
    because C-S applied to this gives the FT bound.

    We need ADDITIONAL constraints.

    Key insight: radii must be compatible with a 2D packing.
    The LARGEST circle has r_max <= 0.5.
    The 2 largest circles must have r_1 + r_2 <= sqrt(2) (max distance sqrt(2),
    minus the fact that both must be in the square).

    Actually, the distance between two circles is >= r_1 + r_2.
    The max distance between centers is sqrt((1-r_1-r_2)^2 + (1-r_1-r_2)^2)
    = sqrt(2) * (1-r_1-r_2) (if both in opposite corners).
    For this to be >= r_1+r_2: sqrt(2)*(1-r_1-r_2) >= r_1+r_2
    (1+sqrt(2))*(r_1+r_2) <= sqrt(2)  [WRONG: only applies if in opposite corners]

    Actually for two circles to fit:
    - Both fit in the square: r_1, r_2 <= 0.5
    - Non-overlap: exists positions where distance >= r_1+r_2
    - Max possible distance: when in opposite corners,
      dist = sqrt((1-r_1-r_2)^2 + (1-r_1-r_2)^2) = sqrt(2)*(1-r_1-r_2)
    - Need: sqrt(2)*(1-r_1-r_2) >= r_1+r_2
    - So: r_1+r_2 <= sqrt(2)/(1+sqrt(2)) = sqrt(2)*(sqrt(2)-1) = 2-sqrt(2) ≈ 0.586

    This is a valid constraint: for any 2 circles, r_1+r_2 <= 2-sqrt(2) ≈ 0.586.
    Wait, is this right? For r_1=r_2=0.25, r_1+r_2=0.5 <= 0.586. OK.
    For r_1=0.3, r_2=0.28, sum=0.58 <= 0.586. Close but OK.
    For r_1=0.5, r_2 <= 0.086. That's very restrictive!

    Actually, the constraint is WRONG. Two circles don't have to be in
    opposite corners. They can be placed side by side.
    For r_1=r_2=0.25: place at (0.25, 0.5) and (0.75, 0.5). Distance = 0.5 >= 0.5. OK.
    For r_1=r_2=0.29: place at (0.29, 0.5) and (0.71, 0.5). Distance = 0.42 >= 0.58? NO!
    0.42 < 0.58. So this doesn't work side by side.
    Try diagonally: (0.29, 0.29) and (0.71, 0.71). Dist = sqrt(0.42^2*2) = 0.594 >= 0.58. OK!

    So the constraint r_1+r_2 <= 2-sqrt(2) is too tight. Let me reconsider.

    For two circles, we need centers at distance >= r_1+r_2.
    Centers of circle 1: (x1,y1) with r_1<=x1<=1-r_1, r_1<=y1<=1-r_1.
    Centers of circle 2: (x2,y2) with r_2<=x2<=1-r_2, r_2<=y2<=1-r_2.

    Max possible distance between centers:
    d_max = sqrt((1-r_1-r_2)^2 + (1-r_1-r_2)^2) = sqrt(2)*(1-r_1-r_2)
    (when circle 1 is in upper-left corner, circle 2 in lower-right)

    Need: d_max >= r_1+r_2.
    sqrt(2)*(1-r_1-r_2) >= r_1+r_2
    sqrt(2) >= (1+sqrt(2))*(r_1+r_2)
    r_1+r_2 <= sqrt(2)/(1+sqrt(2)) = 2-sqrt(2) ≈ 0.586

    But this is the constraint for circles that MUST be in opposite corners.
    If they can be in other positions, the constraint is weaker.

    Actually, d_max IS the maximum over all valid positions. So if d_max < r_1+r_2,
    then NO valid placement exists. So r_1+r_2 <= 2-sqrt(2) is NOT right.

    Wait: d_max = sqrt(2)*(1 - max(r_1,r_2) - max(r_1,r_2))... no.
    Center 1 at (r_1, r_1), center 2 at (1-r_2, 1-r_2):
    d = sqrt((1-r_1-r_2)^2 + (1-r_1-r_2)^2) = sqrt(2)*(1-r_1-r_2).

    For this to be >= r_1+r_2 (which we need):
    sqrt(2)*(1-r_1-r_2) >= r_1+r_2
    r_1+r_2 <= sqrt(2)/(1+sqrt(2)) = 2-sqrt(2) ≈ 0.5858.

    So for any pair of circles: r_i + r_j <= 2-sqrt(2) ≈ 0.5858.
    THIS IS TIGHT: for n=2, the optimal has r_1 = r_2 ≈ 0.2929, sum ≈ 0.5858!

    Now, for n circles: EVERY pair must satisfy r_i + r_j <= 2-sqrt(2).
    If all radii are equal: n*r <= n * (2-sqrt(2))/2 = n*(1-1/sqrt(2)).
    For n=26: 26*(1-0.7071) = 26*0.2929 = 7.615. Not useful (too loose).

    But: combine with area constraint.
    Maximize sum(r_i) subject to:
    - r_i in [0, 0.5]
    - r_i + r_j <= 2-sqrt(2) for all pairs (NOT CORRECT for general pairs!)

    Wait: the constraint r_i+r_j <= 2-sqrt(2) is for the WORST CASE (both in corners).
    For circles NOT in opposite corners, the max distance might be smaller OR larger.
    Actually, I proved: the maximum possible distance between centers (over all valid
    positions) is sqrt(2)*(1-r_i-r_j). So the non-overlap constraint is SATISFIABLE
    if and only if sqrt(2)*(1-r_i-r_j) >= r_i+r_j, i.e., r_i+r_j <= 2-sqrt(2).

    Hmm, but this is only for a PAIR. For n circles, pairwise constraints interact.
    The pair constraint r_i+r_j <= 2-sqrt(2) is NECESSARY (not sufficient).

    But: for n=2, the optimal IS r_1+r_2 = 2-sqrt(2) ≈ 0.5858. And the optimal
    sum for n=2 IS 0.5858. So this pairwise constraint is tight for n=2.

    For larger n, the constraint is too loose (many circles can have small radii).

    Let me think about what GROUPS of circles constrain each other.

    For K circles in a region of the square, they must fit in that region.
    Divide [0,1]^2 into a grid of m x m cells, each of size 1/m x 1/m.
    Circles in each cell must fit in that cell.
    """
    # Let me implement the grid-based LP bound.
    return grid_lp_bound(n, verbose=verbose)


def grid_lp_bound(n, m=5, verbose=False):
    """
    Grid-based LP upper bound.

    Divide [0,1]^2 into m x m cells.
    For each cell (a,b), let n_{a,b} = number of circle centers in that cell.
    Let S_{a,b} = sum of radii of circles centered in cell (a,b).

    Constraints:
    - sum n_{a,b} = n
    - For circles in cell (a,b): they fit in an extended region
      (their radii can extend beyond the cell).
    - FT area bound per cell is tricky (circles extend beyond cells).

    Simpler: just use the FT bound globally, but ADD constraints about
    how many circles can fit in each row/column.

    ROW BOUND: for m rows of height 1/m:
    Circles whose centers are in row k have y_i in [(k-1)/m, k/m].
    Their radii can extend outside the row.
    But for non-overlap within the row: if two circles in the same row
    have centers at same height, they need x-distance >= r_i + r_j.
    In the worst case (same height), the row can fit at most
    1/(2*min_r) circles, with sum of diameters <= 1, so sum r_i <= 0.5.

    Wait! For circles at the SAME HEIGHT, their x-projections don't overlap,
    so sum of diameters <= 1, giving sum r_i <= 0.5 per "layer".

    But circles in the same row but at different heights might NOT project-overlap.
    The constraint is actually on the number of circles at any SINGLE height.

    The "layer" bound: at any height y, at most floor(1/(2*r_min(y))) circles
    can have their center at height y. But we don't know r_min.

    For a row of height h = 1/m: circles in this row have centers with
    y in [h*(k-1), h*k]. The number of "layers" at any x is bounded by
    the number of circles whose y-intervals overlap at that x.

    This is getting complex. Let me try a simpler numerical approach.

    SIMPLEST IMPROVEMENT: use both FT area bound AND pairwise distance bounds.

    For sorted radii r_1 >= r_2 >= ... >= r_n:
    - sum(2*sqrt(3)*r_i^2) <= 1  (FT)
    - r_i <= 0.5
    - The K largest circles need enough space. Specifically:
      k circles of radius >= r need area >= k * 2*sqrt(3)*r^2.
      So 2*sqrt(3)*k*r^2 <= 1, i.e., r <= 1/sqrt(2*sqrt(3)*k).

    This just gives: r_k <= 1/sqrt(2*sqrt(3)*k) for the k-th largest radius.

    Using this: sum(r_i) <= sum_{k=1}^{n} 1/sqrt(s*k) where s=2*sqrt(3).
    = (1/sqrt(s)) * sum_{k=1}^{n} 1/sqrt(k)
    ~ (1/sqrt(s)) * 2*sqrt(n)  (by integral approximation)
    = 2*sqrt(n/s)

    This is WORSE than FT by a factor of 2! The C-S bound is better.

    So the order-statistics bound is weaker than C-S. Not helpful.

    Let me try yet another approach: LP with BOTH area and column constraints.
    """
    from scipy.optimize import linprog

    # Discretize radii
    K = 200
    r_vals = np.linspace(0.001, 0.5, K)

    c = -r_vals  # maximize sum(n_k * r_k)

    A_eq = np.ones((1, K))
    b_eq = np.array([float(n)])

    A_ub_list = []
    b_ub_list = []

    # FT area bound: sum(n_k * 2*sqrt(3)*r_k^2) <= 1
    s = 2 * np.sqrt(3)
    A_ub_list.append(s * r_vals**2)
    b_ub_list.append(1.0)

    # For each radius r, at most N_oler(r) circles of that radius fit:
    # N_oler(r) = (1 + 4r + pi*r^2) / (2*sqrt(3)*r^2)
    for k in range(K):
        r = r_vals[k]
        N_max = (1 + 4*r + np.pi*r**2) / (s * r**2)
        row = np.zeros(K)
        row[k] = 1.0
        A_ub_list.append(row)
        b_ub_list.append(N_max)

    # Prefix Oler: for each threshold t, the number of circles with r >= t
    # must satisfy the Oler bound.
    # sum_{k: r_k >= t} n_k <= (1 + 4t + pi*t^2) / (s*t^2)
    for t_idx in range(K):
        t = r_vals[t_idx]
        row = np.zeros(K)
        for k in range(t_idx, K):
            row[k] = 1.0
        N_max = (1 + 4*t + np.pi*t**2) / (s * t**2)
        A_ub_list.append(row)
        b_ub_list.append(N_max)

    # Pairwise: if r_1 is the max radius, then r_1 <= 0.5
    # and all pairs r_i + r_j <= 2-sqrt(2) ≈ 0.586
    # For sorted radii, the binding constraint is r_1 + r_2 <= 0.586.
    # In the LP, this becomes: for the two largest bins,
    # r_{k1} + r_{k2} <= 0.586 where n_{k1} >= 1, n_{k2} >= 1.
    # Hard to encode directly. Skip for now.

    A_ub = np.array(A_ub_list)
    b_ub = np.array(b_ub_list)

    bounds = [(0, None) for _ in range(K)]
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method='highs')

    if result.success:
        bound = -result.fun
        if verbose:
            nk = result.x
            nonzero = nk > 0.01
            print(f"  Grid LP (n={n}): {bound:.6f}")
            for k in np.where(nonzero)[0]:
                print(f"    r={r_vals[k]:.4f}: n={nk[k]:.2f}")
        return bound
    return None


def main():
    known_best = {
        1: 0.5000, 2: 0.5858, 3: 0.7645, 4: 1.0000, 5: 1.0854,
        10: 1.5911, 15: 2.0365, 20: 2.3010, 26: 2.6360, 30: 2.8425, 32: 2.9390,
    }

    n_values = [1, 2, 3, 4, 5, 10, 15, 20, 26, 30, 32]

    print("Direct bounds (no C-S)")
    print("=" * 75)

    print(f"\n{'n':>3} | {'FT(CS)':>8} | {'Int-FT':>8} | {'Int-Oler':>8} | {'GridLP':>8} | "
          f"{'Known':>8} | {'Best':>8} | {'Gap%':>6}")
    print("-" * 75)

    for n_val in n_values:
        ft_cs = cs_ft_bound(n_val)
        int_ft = integral_bound_ft(n_val)
        int_oler = integral_bound_oler(n_val)
        grid_lp = grid_lp_bound(n_val, verbose=(n_val == 26))

        bounds = [ft_cs, int_ft, int_oler]
        if grid_lp is not None:
            bounds.append(grid_lp)
        best = min(bounds)
        known = known_best.get(n_val)
        gap_pct = 100*(best-known)/known if known else None

        print(f"{n_val:3d} | {ft_cs:8.4f} | {int_ft:8.4f} | {int_oler:8.4f} | "
              f"{grid_lp if grid_lp else 0:8.4f} | "
              f"{known if known else 0:8.4f} | {best:8.4f} | "
              f"{gap_pct if gap_pct else 0:5.1f}%")


if __name__ == "__main__":
    main()
