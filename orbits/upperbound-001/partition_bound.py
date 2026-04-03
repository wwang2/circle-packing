"""
Partition-based upper bound.

Divide [0,1]^2 into regions and bound the contribution from each region.

Key insight: circles near corners/edges have restricted placement.
A circle of radius r near a corner at (r,r) can't have too large r
because it wastes corner space.

APPROACH:
- Divide the square into: 4 corner regions, 4 edge regions, 1 interior region
- For each region, bound the contribution to sum(r_i)
- The bound per region accounts for boundary effects

APPROACH 2: Column/Row slicing with LP.

Divide [0,1]^2 into vertical strips of width w = 1/m.
For strip j (x in [(j-1)*w, j*w]):
  - Circles centered in strip j contribute to sum(r_i)
  - These circles project onto the x-axis with intervals of width 2*r_i
  - The circles in strip j have x_i in [(j-1)*w, j*w]

For non-overlapping circles at the same y-coordinate within a strip:
  Sum of diameters that fit in width w: sum(2*r_i) <= w for circles at same y.
  But circles can be at different y, so more can fit.

Within strip j of width w and height 1:
  This is a packing sub-problem in a w x 1 rectangle.
  Apply FT to this rectangle:
  sum(r_i^2) <= area/(2*sqrt(3)) = w/(2*sqrt(3)) for circles in this strip.

  But wait: circles can extend OUTSIDE the strip. A circle centered in strip j
  with radius r extends from x_i-r to x_i+r, which may be outside the strip.
  So the area "used" in the strip is LESS than pi*r^2.

  This makes the per-strip FT bound loose.

APPROACH 3: LP with per-row AND per-column AND global constraints.

For m rows of height 1/m and m columns of width 1/m:
Each circle is assigned to one row (by y-center) and one column (by x-center).

In row k: the circles centered there have sum(r_i^2) <= ???
In column j: similarly.

Cross constraint: circles in row k AND column j are in a 1/m x 1/m cell.
At most a few circles fit in each cell.

Let me implement this properly.
"""

import numpy as np
from scipy.optimize import linprog
import json
import sys
from pathlib import Path


def partition_lp_bound(n, m=6, verbose=False):
    """
    LP bound with m x m grid partition.

    For each grid cell (i,j) with i,j in 0..m-1:
    - Cell region: [i/m, (i+1)/m] x [j/m, (j+1)/m], size (1/m)^2
    - n_{ij} = number of circles centered in cell (i,j)
    - S_{ij} = sum of radii of circles in cell (i,j)

    Constraints:
    1. sum_{ij} n_{ij} = n
    2. FT per cell: for circles in cell (i,j), centered in a (1/m)x(1/m) box,
       the FT density gives: sum(r_k^2) for k in cell <= A_cell/(2*sqrt(3))
       But circles extend beyond the cell, so we use the OLER area for the cell:
       A_eff(cell) = (1/m)^2 + 4*(1/m)*R + pi*R^2 where R is max radius in cell.
       This is complicated.

    SIMPLER: just use FT globally + per-cell count constraints.

    Per cell: the maximum number of circles in a (1/m)x(1/m) cell is bounded
    by Oler: N_max(r, cell) = [(1/m)^2 + 4*(1/m)*r + pi*r^2] / (2*sqrt(3)*r^2).

    For a general radius, the binding per-cell constraint is:
    n_{ij} <= max_r N_max(r, cell) which goes to infinity as r -> 0.
    So individual cell counts aren't useful.

    But: the AREA of circles in each cell IS bounded.
    Each circle of radius r contributes area pi*r^2.
    Circles centered in cell (i,j) with radius r extend into neighboring cells.
    The area of circle k INSIDE cell (i,j) is at most pi*r_k^2 (all of it)
    and at least something that depends on how close the center is to the boundary.

    For a CONSERVATIVE bound: all of each circle's area is charged to its center cell.
    Then: sum_{k in cell} pi*r_k^2 <= (1/m)^2 ... NO! This is too restrictive.
    Circles can extend beyond their cell.

    Let me try a different angle.

    APPROACH: Think of this as a bin packing problem.

    Each circle of radius r "needs" a certain amount of space.
    By FT, it needs at least 2*sqrt(3)*r^2 of Voronoi cell area.
    Total: sum(2*sqrt(3)*r_i^2) <= 1.

    Additionally: in any row of height h = 1/m, the circles intersecting that row
    contribute their projections. At any horizontal slice, the total width of
    circles <= 1 (since they don't overlap).

    The total "bandwidth" of circles in a row:
    For circle i with center in row k at height y_i:
    It intersects the row at heights [max((k-1)/m, y_i-r_i), min(k/m, y_i+r_i)].
    The width of intersection at height y is 2*sqrt(r_i^2 - (y-y_i)^2).

    The AREA of circle i in row k is:
    integral over the overlap heights of 2*sqrt(r_i^2 - (y-y_i)^2) dy

    The total area of ALL circles in row k <= (1/m) * 1 = 1/m.
    (Since at each height in the row, total circle width <= 1.)

    So: for each row k: sum of (area of circle i in row k) <= 1/m.
    Summing over all rows: sum_k [sum of areas in row k] = sum_i pi*r_i^2.
    So: sum_i pi*r_i^2 <= m * (1/m) = 1. Just the area bound.

    Not helpful.

    TIGHTER: at each height y, the total circle width is <= 1.
    AND: the FT bound says the circles DON'T pack efficiently.
    Combining: sum pi*r_i^2 <= 1 * pi/(2*sqrt(3)) = pi/(2*sqrt(3)).
    Wait, that's saying the packing density <= pi/(2*sqrt(3)) at EACH height.
    At height y: total circle width <= 1, and the actual circle area at that
    height is total_width * dy. The density at that height is the ratio of
    circle width to available width (1), bounded by pi/(2*sqrt(3)) ≈ 0.907.

    So: at each y, total circle width <= pi/(2*sqrt(3)).
    This gives: integral_0^1 [total width at y] dy <= pi/(2*sqrt(3)) * 1.
    But integral = sum(pi*r_i^2), so sum(pi*r_i^2) <= pi/(2*sqrt(3)).
    i.e., sum(r_i^2) <= 1/(2*sqrt(3)). This IS the FT bound!

    So the "per-height density bound" IS the FT bound. No improvement from slicing.

    OK, I'm going in circles (pun intended). Let me try the ONLY approach
    that can potentially beat FT: a computational SDP with ACTUAL non-overlap
    constraints for specific small n.
    """
    # Fall back to FT
    return np.sqrt(n / (2 * np.sqrt(3)))


def enhanced_socp_bound(n, verbose=False):
    """
    Enhanced SOCP bound combining multiple constraints.

    Variables: r_1 >= r_2 >= ... >= r_n (sorted radii)

    Constraints:
    1. FT: sum(2*sqrt(3)*r_i^2) <= 1
    2. r_i <= 0.5
    3. r_i >= 0
    4. Pairwise sum constraint:
       For any two circles that could be the two largest:
       The space for two circles of radii r_1, r_2 requires:
       distance >= r_1+r_2, and both fit in square.
       Max distance = sqrt(2)*(1-r_1-r_2) (when in opposite corners).
       Need: sqrt(2)*(1-r_1-r_2) >= r_1+r_2.
       So: r_1+r_2 <= sqrt(2)/(1+sqrt(2)) = 2-sqrt(2) ≈ 0.5858.

    5. Triple constraint: for 3 largest circles, they must fit.
       The three circles need pairwise distances >= r_i+r_j.
       Place them at 3 corners: need the smallest pairwise distance >= r_i+r_j.
       Two circles in adjacent corners at distance 1-r_i-r_j.
       Need: 1-r_i-r_j >= r_i+r_j, so r_i+r_j <= 0.5 for adjacent corners.
       Two in diagonal corners: sqrt(2)*(1-r_i-r_j) >= r_i+r_j (same as above).
       For 3 circles: at least 2 pairs are in adjacent corners.
       So for the 2 largest: r_1+r_2 <= 0.5 (if adjacent)...

       Wait, corner placement: (r,r), (1-r,r), (r,1-r), (1-r,1-r).
       Adjacent pairs: distance = 1-2r (for equal r).
       Diagonal pairs: distance = sqrt(2)*(1-2r).
       For 3 circles at 3 corners:
       - 2 adjacent pairs (distance 1-r_i-r_j each) and 1 diagonal pair.
       - Need 1-r_i-r_j >= r_i+r_j for adjacent: r_i+r_j <= 0.5.
       - Actually: corners (r1,r1), (1-r2,r2), (r3,1-r3).
         d12 = 1-r1-r2, need >= r1+r2: r1+r2 <= 0.5.
         d13 = sqrt((r3-r1)^2+(1-r1-r3)^2)...too complicated.

       For EQUAL radii r at 3 corners of the square:
       Adjacent distance = 1-2r, need >= 2r: r <= 0.25.
       With r=0.25: 3*0.25 = 0.75. But FT allows sqrt(3/(2*sqrt(3))) = 0.9306.
       So the geometric 3-corner constraint (r <= 0.25) is TIGHTER for 3 circles
       than FT... if we knew they were at corners.

       But 3 circles don't have to be at corners! Place along a line:
       (r, 0.5), (3r, 0.5), (5r, 0.5): need 5r+r <= 1, 6r <= 1, r <= 1/6 ≈ 0.167.
       Sum = 0.5. Worse.

       Place in triangle: centers at (0.5, r), (0.5-d, 1-r), (0.5+d, 1-r).
       This gets complex. Let me just implement the SOCP with known constraints.

    6. Constraint from Oler with boundary correction (for the full set):
       sum(2*sqrt(3)*r_i^2) <= 1 + 4*r_1 + pi*r_1^2
       (where r_1 = r_max, from the Oler-Groemer inequality).

       Since 1+4r_1+pi*r_1^2 > 1 for r_1 > 0, this is WEAKER than FT.
       Not useful.

    So: the constraints are:
    (a) FT area
    (b) r_i <= 0.5
    (c) r_1 + r_2 <= 2 - sqrt(2)   [pair constraint]
    (d) Any pair of "adjacent-corner" circles: r_i + r_j <= 0.5
        But we don't know which pairs are adjacent. Skip.
    """
    try:
        import cvxpy as cp
    except ImportError:
        return None

    r = cp.Variable(n)
    s = 2 * np.sqrt(3)

    constraints = []
    constraints += [r >= 1e-10]
    constraints += [r <= 0.5]

    # Ordering
    for i in range(n-1):
        constraints += [r[i] >= r[i+1]]

    # FT area
    constraints += [s * cp.sum_squares(r) <= 1]

    # Pairwise: r_1 + r_2 <= 2 - sqrt(2)
    if n >= 2:
        constraints += [r[0] + r[1] <= 2 - np.sqrt(2)]

    # For the 4 largest: they can be placed at 4 corners.
    # Corners: distance between adjacent = 1-r_i-r_j, diagonal = sqrt(2)*(1-r_i-r_j).
    # 4 circles at 4 corners: 4 adjacent pairs, 2 diagonal pairs.
    # Adjacent: 1-r_i-r_j >= r_i+r_j => r_i+r_j <= 0.5.
    # For the 4 largest, if placed at corners, each adjacent pair needs sum <= 0.5.
    # The 4 corners form a square, each has 2 adjacent neighbors.
    # For 4 circles: r_1+r_2 <= 0.5, r_1+r_3 <= 0.5, r_2+r_4 <= 0.5, r_3+r_4 <= 0.5
    # (assuming some ordering of corners).
    # But we don't know which are adjacent!
    # The WEAKEST constraint: the two largest adjacent pair.
    # If the 4 largest are at corners, at least 4 adjacent pairs.
    # The largest radius r_1 is adjacent to 2 others, both must satisfy r_1+r_j <= 0.5.
    # So r_j <= 0.5 - r_1 for the 2 neighbors of r_1.
    # The 4th circle is diagonal to r_1 and adjacent to the other 2.

    # Actually, for 4 circles at 4 corners is NOT the only option.
    # The 4 largest circles can be anywhere. The pair constraint
    # r_1+r_2 <= 2-sqrt(2) is the valid one for ANY pair.

    # For 3 circles: can we derive r_1+r_2+r_3 <= something?
    # Place 3 circles: their centers form a triangle.
    # Each pairwise distance >= r_i+r_j.
    # All centers in [r_i, 1-r_i]^2 (containment).
    # By triangle inequality on the plane:
    #   d12 + d13 >= d23
    # Since d_ij >= r_i+r_j: (r1+r2)+(r1+r3) >= d23 >= r2+r3
    # => 2r1 >= 0, trivially true.
    # Not useful.

    # 3-circle area: the 3 circles take at least 2*sqrt(3)*(r_1^2+r_2^2+r_3^2) area.
    # Already in FT.

    # Hmm. The pairwise distance constraint r_1+r_2 <= 2-sqrt(2) is our only
    # improvement over FT.

    # What about: for n >= 5, at most 4 circles can have r > 0.25?
    # Because 5 circles of r=0.25 would need... let's check.
    # 5 circles of r=0.25: centers in [0.25, 0.75]^2.
    # Available area: 0.5 x 0.5 = 0.25.
    # Voronoi area per circle: 0.05. Need >= 2*sqrt(3)*0.0625 = 0.2165.
    # 0.05 < 0.2165. So 5 circles of r=0.25 DON'T fit by FT!
    # FT says: n_large * 2*sqrt(3)*r^2 <= 1 => n_large <= 1/(2*sqrt(3)*0.0625) = 4.62
    # So at most 4 circles of r >= 0.25. But this is already in FT.

    # What FT DOESN'T capture: the constraint that 4 circles of r=0.25
    # use up specific POSITIONS (the 4 corners), leaving less room for others.

    # The TOPOLOGICAL constraint: 4 large corner circles partition the
    # remaining space into a central region with area ~ 1 - 4*pi*0.25^2 = 0.215.
    # Actually area is 1 - 4*(pi*0.0625) = 1 - 0.785 = 0.215.
    # In this area, the remaining 22 circles must fit.
    # By FT: sum(r_i^2) for remaining <= 0.215/(2*sqrt(3)*pi) ... no, FT gives
    # sum(2*sqrt(3)*r_i^2) <= 0.215 ... no, the remaining area is what's LEFT.
    # The Voronoi cells of the remaining circles must fit in area <= 1 - corner_voronoi.
    # Corner Voronoi cells: each >= 2*sqrt(3)*0.0625 = 0.2165.
    # 4 corners use 4*0.2165 = 0.866 of Voronoi area.
    # Remaining: 1 - 0.866 = 0.134 for 22 circles.
    # 22 circles in 0.134: sum(r_i^2) <= 0.134/(2*sqrt(3)) = 0.0387.
    # sum(r_i) <= sqrt(22*0.0387) = sqrt(0.851) = 0.923.
    # Total: 4*0.25 + 0.923 = 1.923. But best known is 2.636!
    # So this partition is terrible. The actual solution doesn't use 4 big corners.

    # The challenge is finding the RIGHT partition of space.
    # This requires knowing the topology of the solution.

    objective = cp.Maximize(cp.sum(r))
    prob = cp.Problem(objective, constraints)

    result = prob.solve(solver=cp.SCS, verbose=False, max_iters=20000)

    if verbose and r.value is not None:
        rv = r.value
        print(f"  Enhanced SOCP (n={n}): {result:.6f}")
        print(f"  Top radii: {rv[:min(5,n)]}")
        print(f"  r1+r2 = {rv[0]+rv[1] if n>=2 else 'N/A':.6f} (limit {2-np.sqrt(2):.6f})")

    return result


def main():
    known_best = {
        1: 0.5000, 2: 0.5858, 3: 0.7645, 4: 1.0000, 5: 1.0854,
        10: 1.5911, 15: 2.0365, 20: 2.3010, 26: 2.6360, 30: 2.8425, 32: 2.9390,
    }

    n_values = [1, 2, 3, 4, 5, 10, 15, 20, 26, 30, 32]

    print("Enhanced SOCP bound with pairwise constraint")
    print("=" * 70)
    print(f"{'n':>3} | {'FT':>8} | {'SOCP':>8} | {'Known':>8} | {'Gap':>8} | {'Gap%':>6}")
    print("-" * 70)

    for n_val in n_values:
        ft = np.sqrt(n_val / (2*np.sqrt(3)))
        socp = enhanced_socp_bound(n_val, verbose=(n_val in [2, 26]))

        best = min(ft, socp) if socp else ft
        known = known_best.get(n_val)
        gap = best - known if known else None
        gap_pct = 100*gap/known if gap else None

        valid = best >= known - 1e-4 if known else True

        print(f"{n_val:3d} | {ft:8.4f} | {socp if socp else 0:8.4f} | "
              f"{known if known else 0:8.4f} | "
              f"{gap if gap else 0:8.4f} | {gap_pct if gap_pct else 0:5.1f}% "
              f"{'OK' if valid else 'FAIL!'}")


if __name__ == "__main__":
    main()
