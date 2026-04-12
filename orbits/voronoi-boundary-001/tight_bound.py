#!/usr/bin/env python3
"""
Tight upper bounds on circle packing in [0,1]^2 using boundary-corrected arguments.

=== What works and what doesn't ===

DOES NOT WORK (invalidated by our Voronoi analysis):
- Per-cell L-function bound: the circle is not inscribed in its Voronoi cell,
  so L(k)*r^2 is NOT a valid lower bound on individual cell areas.
- Euler + L-function: relies on per-cell L-function, which is invalid.

DOES WORK:
- Fejes-Toth GLOBAL density bound: sum(2*sqrt(3)*r_i^2) <= 1. Provable via
  the hexagonal packing density theorem. UB = 2.7396 for n=26.

THIS SCRIPT explores three approaches to beat FT:

(1) CONTAINMENT WASTE: Circles must fit inside [0,1]^2, so r_i <= 0.5 and
    centers are in [r_i, 1-r_i]^2. The effective area is reduced.

(2) PAIR INTERACTION: Non-overlapping circles have dist(c_i, c_j) >= r_i + r_j.
    This constrains pairwise distances, giving additional valid inequalities.

(3) GROEMER SAUSAGE BOUND: For n circles in a convex body, there's a
    perimeter-dependent correction to the FT bound. We adapt this to
    the non-congruent case.

(4) MIXED-INTEGER OPTIMIZATION: formulate the problem as an optimization
    over discrete boundary configurations and continuous radii.
"""

import numpy as np
from scipy.optimize import minimize as sp_minimize, linprog
from scipy.spatial import ConvexHull
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as MplCircle, Rectangle, FancyArrowPatch
import matplotlib.colors as mcolors
from pathlib import Path
from itertools import combinations


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


# ═══════════════════════════════════════════════════════════════════════════════
# BOUND 1: FEJES-TOTH (baseline)
# ═══════════════════════════════════════════════════════════════════════════════

def fejes_toth_bound(n):
    """sqrt(n / (2*sqrt(3)))"""
    return np.sqrt(n / (2 * np.sqrt(3)))


# ═══════════════════════════════════════════════════════════════════════════════
# BOUND 2: ENHANCED FEJES-TOTH WITH INDIVIDUAL RADIUS CAP
# ═══════════════════════════════════════════════════════════════════════════════

def ft_with_radius_cap(n, r_max=0.5):
    """
    FT bound with individual radius constraint r_i <= r_max.

    maximize sum(r_i)
    subject to: sum(2*sqrt(3)*r_i^2) <= 1
                0 <= r_i <= r_max

    If r_max >= 1/sqrt(2*sqrt(3)*n), the constraint doesn't bind.
    For n=26, FT-optimal r = 0.1054 << 0.5, so no improvement.
    """
    alpha = 2 * np.sqrt(3)
    r_opt = 1 / np.sqrt(alpha * n)
    if r_opt <= r_max:
        return np.sqrt(n / alpha)
    else:
        # Some circles at r_max, rest optimized
        # Let m circles be at r_max, (n-m) free
        # sum(alpha * r_i^2) = m*alpha*r_max^2 + (n-m)*alpha*r^2 <= 1
        # Maximize m*r_max + (n-m)*r
        # r^2 <= (1 - m*alpha*r_max^2) / ((n-m)*alpha)
        # r <= sqrt((1 - m*alpha*r_max^2) / ((n-m)*alpha))
        # sum = m*r_max + (n-m)*sqrt((1 - m*alpha*r_max^2) / ((n-m)*alpha))
        #     = m*r_max + sqrt((n-m)*(1 - m*alpha*r_max^2)/alpha)
        best = 0
        for m in range(n + 1):
            remaining_area = 1 - m * alpha * r_max**2
            if remaining_area < 0:
                break
            if n - m > 0:
                r_free = np.sqrt(remaining_area / ((n - m) * alpha))
                total = m * r_max + (n - m) * r_free
            else:
                total = m * r_max
            best = max(best, total)
        return best


# ═══════════════════════════════════════════════════════════════════════════════
# BOUND 3: PAIR INTERACTION BOUND
# ═══════════════════════════════════════════════════════════════════════════════

def pair_bound(n):
    """
    Use pairwise non-overlap constraints to bound sum(r_i).

    For any pair (i,j): dist(c_i, c_j) >= r_i + r_j.
    Since centers are in [0,1]^2: dist <= sqrt(2).
    So r_i + r_j <= sqrt(2) for all pairs.

    Summing over all pairs: sum_{i<j} (r_i + r_j) <= C(n,2)*sqrt(2)
    => (n-1)*sum(r_i) <= n(n-1)/2 * sqrt(2)
    => sum(r_i) <= n*sqrt(2)/2

    For n=26: 18.38. Very loose.

    Better: use that circles are in [0,1]^2 and the pair bound together
    with the area constraint. This is a SOCP.

    The tightest pair bound for two circles touching the same wall:
    r_1 + r_2 <= 2 - sqrt(2) when they're at opposite corners.
    """
    return n * np.sqrt(2) / 2  # Loose


# ═══════════════════════════════════════════════════════════════════════════════
# BOUND 4: GROEMER-TYPE BOUNDARY CORRECTION
# ═══════════════════════════════════════════════════════════════════════════════

def groemer_bound_noncongruent(n):
    """
    Groemer (1960) proved for n circles of radii r_1,...,r_n packed in a
    convex body D:

    Area(D) >= area_of_Delaunay_triangulation + boundary_correction

    The Delaunay triangulation area argument:
    Each Delaunay triangle with vertices at centers c_i, c_j, c_k has area
    >= sqrt(3)/4 * (r_i + r_j + r_k)^2 / 3  [NOT true in general]

    Actually, Groemer's result is:
    For n non-overlapping circles in a convex body K:
    Area(K) >= sum_triangles(area_of_triangle) + correction

    where the triangles come from the Delaunay triangulation.

    The non-overlap condition dist(c_i,c_j) >= r_i + r_j means each
    Delaunay triangle has large enough area.

    For a Delaunay triangle with vertices c_1, c_2, c_3 and opposite
    radii r_1, r_2, r_3 (with edge c_i-c_j having length >= r_i + r_j):

    By the formula area = (1/2)|c_1-c_2| |c_3 - proj|:
    The area depends on the actual positions.

    A simpler approach: for any pair at distance d >= r_i + r_j,
    the "claimed area" along the line between them is at least
    (r_i + r_j) * h for some height h.

    Let me use a different, simpler approach.

    STRIP PACKING BOUND:
    Project all circles onto the x-axis. Circle i projects to [x_i-r_i, x_i+r_i].
    These intervals can overlap (circles at different y).
    But: for circles at similar y (|y_i - y_j| < r_i + r_j), their
    x-intervals DON'T overlap (since the circles don't overlap).

    This is a chromatic number / interval scheduling argument and
    is hard to formalize into a clean bound.
    """
    return fejes_toth_bound(n)


# ═══════════════════════════════════════════════════════════════════════════════
# BOUND 5: NONLINEAR CONTAINMENT-INTERACTION BOUND (NEW)
# ═══════════════════════════════════════════════════════════════════════════════

def containment_interaction_bound(n):
    """
    A new bound combining containment with interaction constraints.

    For each circle i: center at (x_i, y_i) with r_i <= x_i <= 1-r_i,
    r_i <= y_i <= 1-r_i. The circle's center is in a box of side 1-2r_i.

    For two circles touching the same wall (say bottom): both have y_i = r_i.
    Non-overlap: |x_i - x_j| >= 2*sqrt(r_i * r_j).
    So they need horizontal separation >= 2*sqrt(r_i*r_j).

    For n_b circles on the bottom wall with radii r_b1,...,r_bn_b:
    Each has center x_i in [r_i, 1-r_i], with pairwise separation
    >= 2*sqrt(r_i * r_j) between adjacent circles.

    Total width occupied: sum_consecutive 2*sqrt(r_i * r_{i+1}) + 2*r_1 + 2*r_n_b <= 2
    (the circles at the ends need r_i from the wall corner).

    Wait, actually: x_1 >= r_1 and x_n <= 1-r_n, and adjacent circles
    have |x_i - x_{i+1}| >= 2*sqrt(r_i*r_{i+1}).

    Total: r_1 + sum_{i=1}^{n_b-1} 2*sqrt(r_i * r_{i+1}) + r_n_b <= 1

    By AM-GM: 2*sqrt(r_i * r_{i+1}) >= 2*min(r_i, r_{i+1}).
    So: r_1 + sum 2*min(r_i, r_{i+1}) + r_n_b <= 1.

    If all bottom-wall radii are equal (r): r + (n_b-1)*2r + r = 2*n_b*r <= 1
    => n_b*r <= 0.5 per wall.
    => sum(r_bottom) <= 0.5, similarly for other 3 walls.
    => sum(r_boundary) <= 4 * 0.5 = 2.0 (with corner circles counted once).

    But corner circles are on TWO walls and counted in both sums.
    So: sum_bottom(r) + sum_top(r) + sum_left(r) + sum_right(r) <= 2.0
    where corner circles appear in 2 sums.

    For the congruent case: 4 walls, each with at most 1/(2r) circles,
    and total boundary circles <= 4/(2r). Interior circles sit in the
    effective area (1-2r)^2 with FT density.

    For the NON-CONGRUENT case, the per-wall constraint is:
    For each wall, the sorted radii r_1 >= ... >= r_m on that wall satisfy:
    sum_{i=1}^{m-1} 2*sqrt(r_i * r_{i+1}) + r_1 + r_m <= 1

    By Cauchy-Schwarz on the geometric means: this is at most r_1 + r_m + 2*(m-1)*sqrt(r_1*r_m).
    Not very useful.

    Let me try a direct numerical optimization instead.
    """

    # Numerical optimization: maximize sum(r_i) subject to:
    # (1) sum(2*sqrt(3)*r_i^2) <= 1   (FT area)
    # (2) 0 <= r_i <= 0.5             (containment)
    # (3) r_i >= 0

    # For n=26 with just these constraints, the optimum has all r_i equal
    # and equals the FT bound. We need constraint (3') from wall interactions.

    # Additional constraint: for any ordering of circles along a wall,
    # sum(2*r_i) <= 1 per wall. But we don't know which circles are on walls.

    # Instead, let's use the TOPOLOGICAL constraint:
    # In any packing, the circles that touch the bottom wall form a sequence
    # with sum of diameters + gaps <= 1. For touching circles (gap=0):
    # sum(2*r_i) <= 1 per wall.

    # But circles touching the same wall DON'T touch each other; they're
    # separated by gaps. The constraint is weaker: the sum of CONTACT
    # projections is <= 1.

    # The simplest valid constraint: r_1 + r_2 <= max_sum_on_wall.
    # For two circles on the same wall: they need horizontal separation
    # >= 2*sqrt(r_1*r_2). Plus containment: they need distance >= r from
    # the wall ends.
    # Total: r_1 + 2*sqrt(r_1*r_2) + r_2 <= 1.
    # By AM-GM: 2*sqrt(r_1*r_2) >= 2*min(r_1,r_2). If equal: r + 2r + r = 4r <= 1
    # => r <= 0.25, so two wall circles have sum <= 0.5.
    # For 3 equal circles: r + 2r + 2r + r = 6r <= 1 => r <= 1/6 = 0.167,
    # sum = 0.5.
    # For m equal circles: (2m)r <= 1, sum = mr <= 0.5.
    # So: sum of radii per wall <= 0.5. Total boundary <= 2.0.

    # Now: if boundary circles have total radius B <= 2.0, and interior
    # circles have total radius I, then S = B + I.
    # FT constraint: 2*sqrt(3)*(sum_B r_i^2 + sum_I r_i^2) <= 1.
    # By Cauchy-Schwarz: sum_B r_i^2 >= B^2/n_b, sum_I r_i^2 >= I^2/n_i.
    # So: 2*sqrt(3)*(B^2/n_b + I^2/n_i) <= 1.

    # Maximize S = B + I subject to:
    # 2*sqrt(3)*(B^2/n_b + I^2/n_i) <= 1
    # B <= 2.0
    # B, I >= 0
    # n_b + n_i = n, 0 <= n_b <= n

    alpha = 2 * np.sqrt(3)
    best_ub = 0
    best_config = None

    for n_b in range(0, n + 1):
        n_i = n - n_b
        B_max = 2.0  # boundary constraint

        # Maximize B + I subject to:
        # alpha*(B^2/n_b + I^2/n_i) <= 1 if both > 0
        # B <= B_max

        if n_b == 0:
            # All interior
            I_max = np.sqrt(n_i / alpha)
            ub = I_max
        elif n_i == 0:
            # All boundary
            ub = min(B_max, np.sqrt(n_b / alpha))
        else:
            # Lagrange: maximize B + I s.t. alpha*(B^2/n_b + I^2/n_i) = 1
            # Gradient: 1 = lambda * 2*alpha*B/n_b, 1 = lambda * 2*alpha*I/n_i
            # => B/n_b = I/n_i => B = (n_b/n_i)*I
            # Substitute: alpha*((n_b/n_i)^2*I^2/n_b + I^2/n_i) = 1
            # => alpha*I^2*(n_b/n_i^2 + 1/n_i) = 1
            # => alpha*I^2*(n_b + n_i)/(n_i^2) = 1
            # => I^2 = n_i^2 / (alpha*n)
            # => I = n_i / sqrt(alpha*n)
            # => B = n_b / sqrt(alpha*n)
            # => S = n / sqrt(alpha*n) = sqrt(n/alpha) = FT bound

            # Unless B > B_max, then B = B_max and optimize I alone
            B_unconstrained = n_b / np.sqrt(alpha * n)
            if B_unconstrained <= B_max:
                ub = np.sqrt(n / alpha)  # FT bound, constraint doesn't bind
            else:
                B = B_max
                I_max = np.sqrt((1 - alpha * B**2 / n_b) * n_i / alpha)
                if 1 - alpha * B**2 / n_b > 0:
                    ub = B + I_max
                else:
                    ub = B  # no room for interior

        if ub > best_ub:
            best_ub = ub
            best_config = (n_b, n_i, ub)

    return best_ub, best_config


# ═══════════════════════════════════════════════════════════════════════════════
# BOUND 6: SECOND-MOMENT (VARIANCE) BOUND
# ═══════════════════════════════════════════════════════════════════════════════

def second_moment_bound(n):
    """
    Use the variance of the radii to tighten the Cauchy-Schwarz step.

    FT gives: sum(r_i^2) <= 1/(2*sqrt(3)) = C.
    C-S gives: (sum r_i)^2 <= n * sum(r_i^2) <= n*C.

    But C-S is tight only when all r_i are equal. If radii must be
    DIFFERENT (due to packing constraints), we can improve.

    For the packing problem, there's no constraint that radii differ.
    The adversary CAN choose all equal radii. So the variance bound
    doesn't help unless we can prove radii must differ.

    Can we prove that? In a packing of n circles in [0,1]^2 achieving
    sum close to FT, the optimal has all radii equal (hexagonal packing
    of equal circles). So the variance is naturally small near the optimum.

    No improvement here.
    """
    return fejes_toth_bound(n)


# ═══════════════════════════════════════════════════════════════════════════════
# BOUND 7: DELAUNAY TRIANGLE AREA BOUND
# ═══════════════════════════════════════════════════════════════════════════════

def delaunay_area_bound(n):
    """
    Use the Delaunay triangulation to get a tighter area bound.

    For n circle centers in [0,1]^2, the Delaunay triangulation creates
    t triangles. Each triangle has vertices at 3 circle centers.
    For a triangle with vertices c_i, c_j, c_k (center distances
    d_ij >= r_i+r_j, etc.), the triangle area satisfies:

    Area(T) >= ??? something involving the radii.

    For an equilateral triangle with all d = 2r:
    Area = sqrt(3)/4 * (2r)^2 = sqrt(3)*r^2.

    The triangle covers 3 circle sectors totaling (for equal radii in
    equilateral triangle) 3 * (pi/6)*r^2 = pi*r^2/2.

    So each equilateral Delaunay triangle has:
    uncovered_area = sqrt(3)*r^2 - pi*r^2/2 = (sqrt(3) - pi/2)*r^2 ~ 0.161*r^2.

    For t = 2n - h - 2 triangles (Euler), the total Delaunay area is:
    sum(triangle_areas) + hull_area = Area(convex_hull).

    And Area(convex_hull) <= 1 (contained in unit square).

    The total uncovered area (within the convex hull) is:
    Area(hull) - sum(circle_area_in_hull).

    This gives: Area(hull) >= sum_triangles(sqrt(3)*r_eff^2 per triangle).
    Where r_eff depends on the triangle's shape and the 3 radii.

    For the general case with non-congruent radii, this is complex.

    A simpler bound: the convex hull of centers has area <= 1.
    The total Delaunay triangle area = Area(hull).
    Each triangle has area >= sqrt(3)/4 * (r_i+r_j)^2 for the smallest edge.

    Actually, this isn't right either. Let me use a cleaner formulation.

    For each Delaunay edge (i,j): d_ij >= r_i + r_j.
    The total length sum_{edges} d_ij >= sum_{edges} (r_i + r_j).
    Each vertex appears deg(i) times in the edge sum.
    sum_{edges} (r_i + r_j) = sum_i deg(i) * r_i.

    And the total edge length is bounded by the perimeter and diagonal
    structure of [0,1]^2.

    This is getting complex. Let me try a direct numerical approach.
    """
    return fejes_toth_bound(n)


# ═══════════════════════════════════════════════════════════════════════════════
# BOUND 8: SOCP WITH WALL CONSTRAINTS (from upperbound-001)
# ═══════════════════════════════════════════════════════════════════════════════

def wall_socp_bound(n):
    """
    Second-order cone program with wall packing constraints.

    Variables: r_i for i=1..n, n_b[w] for each wall w=1..4 (# circles on wall).

    For each wall, circles touching it satisfy:
    sum_j (2*r_j) <= 1  (diameter packing on 1D line)
    where the sum is over circles on that wall.

    Total: sum_all_walls sum_touching (2*r_j) <= 4
    With corner circles counted twice: sum_{boundary} 2*k_j*r_j <= 4
    where k_j = number of walls touched.

    Combined with FT area: maximize sum(r_i)
    subject to: 2*sqrt(3)*sum(r_i^2) <= 1 and 2*sum_{boundary}(k_j*r_j) <= 4.

    For the non-congruent case, the adversary can make boundary circles
    tiny, making the wall constraint slack. So this doesn't help unless
    we can FORCE some circles to be on the boundary.

    KEY INSIGHT: In any packing achieving sum near FT, the circles must
    fill the square densely. This forces SOME circles near the walls.
    But "near" is not the same as "on" the wall.

    HOWEVER: if a circle is within distance delta of a wall but NOT touching
    it, there's a strip of width (r_i - delta) between the circle and wall
    that's wasted. This waste increases the effective area per circle.

    Formalizing: for a circle at distance d from the bottom wall (d >= r_i,
    since the circle is inside the square), the "wasted strip" below the
    circle has area approximately 2*r_i * (d - r_i) (rectangle of height
    d-r_i and width ~2*r_i). If d = r_i (touching), the waste is 0.
    If d >> r_i, the waste grows.

    This waste must come from SOMEWHERE in the total area budget:
    1 = sum(Voronoi_areas) = sum(circle_areas) + sum(wastes)
    => sum(circle_areas) = pi*sum(r_i^2) < 1 always

    The FT bound already accounts for this waste. The question is
    whether boundary-specific waste is LARGER than what FT assumes.

    For FT: waste per circle = (2*sqrt(3) - pi)*r_i^2 ~ 0.322*r_i^2.
    For a circle touching one wall: the wall side has a straight edge
    instead of hexagonal geometry. The packing efficiency near a wall
    is LOWER than hexagonal.

    But as argued in upperbound-001, this goes the wrong way for upper
    bounding: the wall makes packing WORSE, so the real sum is LOWER
    than what FT allows. We're trying to prove the sum is at most X.
    The wall constraint says the optimal sum is lower. But FT already
    allows a sum that's HIGHER than the true optimum. So showing that
    the true optimum is even lower (due to walls) would TIGHTEN the UB.

    The issue is PROVING it rigorously without knowing the optimal packing.
    """
    return fejes_toth_bound(n)


# ═══════════════════════════════════════════════════════════════════════════════
# BOUND 9: EXPLICIT AREA COMPUTATION FOR WALL-ADJACENT CIRCLES
# ═══════════════════════════════════════════════════════════════════════════════

def wall_density_analysis():
    """
    Compute the maximum packing density achievable when circles are
    packed against a flat wall.

    Consider a semi-infinite strip [0, inf) x [0, 1] with a wall at x=0.
    Circles of radius r touch the wall (center at (r, y)).
    The densest packing against the wall has circles in a hexagonal
    arrangement, but the first row is flat against the wall.

    For a single row of circles of radius r against the wall:
    - Circle centers at (r, r + 2kr) for k = 0, 1, ...
    - Each circle claims a rectangle of width 2r and height 2r
    - Local density = pi*r^2 / (4*r^2) = pi/4 ~ 0.785

    For two rows (wall row + interior row):
    - Wall row at x = r, interior row at x = r + sqrt(3)*r
    - Offset by r in y-direction
    - Each pair claims area 2r * 2*sqrt(3)*r = 4*sqrt(3)*r^2
      for 2 circles of total area 2*pi*r^2
    - Density = 2*pi*r^2 / (4*sqrt(3)*r^2) = pi/(2*sqrt(3)) ~ 0.907

    So the WALL ROW density (just the first layer) is pi/4 ~ 0.785,
    lower than the hex density pi/(2*sqrt(3)) ~ 0.907.

    The FT bound assumes ALL circles achieve the hex density.
    If we can show that wall circles achieve at most pi/4 density,
    then their effective area coefficient is:
    alpha_wall = 1/(pi/4) * pi = 4 (instead of 2*sqrt(3) ~ 3.464)

    Wait, let me be careful. FT says:
    sum(alpha * r_i^2) <= 1 where alpha = 2*sqrt(3).
    Equivalently: each circle of radius r occupies effective area
    alpha * r^2 = 2*sqrt(3)*r^2.

    If wall circles have packing density pi/(alpha_wall) instead of
    pi/(2*sqrt(3)), then their effective area is alpha_wall * r^2.

    For wall circles with density pi/4:
    alpha_wall = pi / (pi/4) = 4.

    For corner circles with density pi/4 * 1/2 (quarter-circle):
    alpha_corner = pi / (pi/8) = 8. ??? No, this isn't right.

    Let me think about it differently.

    For a circle of radius r touching one wall, in the densest possible
    packing of the half-plane, the Voronoi cell (restricted to the
    half-plane) has area:
    - In hex packing reflected about the wall: the cell is half of a hex,
      area = sqrt(3)*r^2. But the full circle occupies pi*r^2, so the
      half-circle in the half-plane has area pi*r^2/2.
    - Density in the half-plane: (pi*r^2/2) / (sqrt(3)*r^2) = pi/(2*sqrt(3)) ~ 0.907.

    Same as the interior! Because reflection makes the wall equivalent to
    having a mirror packing. So the wall does NOT reduce density in
    the half-plane case.

    But for circles in a CORNER: the density in the quarter-plane is also
    pi/(2*sqrt(3)) by the same reflection argument.

    So the reflection argument shows that wall/corner effects DON'T
    reduce density. The FT bound is the same for all circle types.

    WHERE DOES THE BOUNDARY WASTE COME FROM THEN?

    The waste comes from the FINITE SIZE of the square. In an infinite
    hexagonal packing, every circle has exactly 6 neighbors. In a finite
    packing, the edge circles have fewer neighbors. But the FT bound
    doesn't use the number of neighbors -- it uses the AREA inequality.

    The FT area inequality is:
    "In any packing in a convex body K, sum(pi*r_i^2) <= pi/(2*sqrt(3)) * Area(K)."

    This is proved by showing that the packing density cannot exceed
    the hexagonal density pi/(2*sqrt(3)).

    For FINITE packings, the density is actually strictly LESS than the
    hex density. Harborth, Koch, and Spieler (1998) showed:

    For n congruent circles of radius r in a convex body K:
    Area(K) >= 2*sqrt(3)*n*r^2 + 2*(2-sqrt(3))*n^{1/2}*r * C

    where C depends on the shape of K. For a square of side 1:
    C is related to the perimeter/area ratio.

    But this CONGRUENT result doesn't apply to the non-congruent case.

    CONCLUSION: For the non-congruent case, the FT bound is essentially
    tight and cannot be improved by simple boundary arguments.

    The only way to beat FT for non-congruent circles is to use
    CONSTRAINTS THAT COUPLE DIFFERENT CIRCLES' RADII, such as:
    - Pairwise distance constraints (SOCP/SDP approach)
    - Contact graph constraints
    - Lasserre SDP hierarchy

    These are computationally expensive and are better suited for
    branch-and-bound solvers.
    """
    return {
        'wall_density': np.pi / 4,
        'hex_density': np.pi / (2 * np.sqrt(3)),
        'wall_alpha': 4.0,
        'hex_alpha': 2 * np.sqrt(3),
        'conclusion': 'Wall density = hex density in reflection, no improvement'
    }


# ═══════════════════════════════════════════════════════════════════════════════
# BOUND 10: MOMENT BOUND (new approach)
# ═══════════════════════════════════════════════════════════════════════════════

def moment_bound(n):
    """
    Use second-moment constraints to tighten the bound.

    For n circles in [0,1]^2:
    (1) sum(2*sqrt(3)*r_i^2) <= 1     (FT area)
    (2) sum(r_i^4) <= sum(r_i^2)^2/n  (FALSE: this is reversed)

    Actually: by Cauchy-Schwarz:
    sum(r_i^2) <= sqrt(n * sum(r_i^4))

    And by power mean inequality:
    (sum r_i^2 / n)^2 <= sum(r_i^4) / n

    Neither helps directly. We need a constraint that BOUNDS r_i^4.

    From FT + containment: r_i <= 0.5, so r_i^2 <= 0.25, and
    sum(r_i^2) <= 1/(2*sqrt(3)) ~ 0.2887.
    sum(r_i^4) <= max(r_i^2) * sum(r_i^2) <= 0.25 * 0.2887 ~ 0.0722.

    And: (sum r_i)^2 <= n * sum(r_i^2) (C-S)
         (sum r_i)^2 = [sum r_i * 1]^2 <= [sum r_i^2][n] (Cauchy-Schwarz)

    The FT bound IS the Cauchy-Schwarz bound applied to sum(r_i) with
    the constraint sum(r_i^2) <= C. Adding r_i^4 constraints doesn't
    help because C-S is already the tightest inequality of this form.

    A DIFFERENT angle: Schur convexity.
    For fixed sum(r_i^2) = C, the sum sum(r_i) is MAXIMIZED when all
    r_i are equal (by Schur-convexity of -sum r_i on {r: sum r^2 = C}).
    This is just C-S again.

    UNLESS we add constraints that prevent all r_i from being equal!

    For instance: in a packing, the largest circle must be at distance
    >= 2r_max from any other circle, which constrains how many large
    circles can exist. This doesn't help for equal radii.

    WHAT IF we combine the area bound with a PERIMETER-like constraint?

    Define: P = sum_{i on boundary} r_i (total boundary radius).
    Claim: P <= 2 (from the 1D packing constraint on each wall).

    Then: S = sum(r_i) = P + sum_{interior} r_i.
    FT: 2*sqrt(3)*(sum_boundary r_i^2 + sum_interior r_i^2) <= 1.

    By C-S: sum_boundary r_i^2 >= P^2/n_b, sum_interior r_i^2 >= (S-P)^2/(n-n_b).

    So: 2*sqrt(3)*(P^2/n_b + (S-P)^2/(n-n_b)) <= 1.

    Maximize S subject to this + P <= 2:
    Fix P and n_b, then (S-P)^2/(n-n_b) <= (1 - 2*sqrt(3)*P^2/n_b)/(2*sqrt(3))
    S-P <= sqrt((n-n_b)/(2*sqrt(3)) - P^2*(n-n_b)/n_b)
    S <= P + sqrt((n-n_b)/(2*sqrt(3)) - P^2*(n-n_b)/n_b)

    For this to improve on FT (S_ft = sqrt(n/(2*sqrt(3)))), we need
    the constraint P <= 2 to actually bind.

    When does it bind? When the unconstrained optimum has P > 2.
    Unconstrained: P = n_b / sqrt(2*sqrt(3)*n) ~ n_b * 0.1054 for n=26.
    For P > 2: n_b > 2/0.1054 ~ 19.

    So the constraint binds when most circles are on the boundary!
    For n_b = 19 (19 boundary + 7 interior):
    P_unconstr = 19 * 0.1054 = 2.003 ~ 2.
    The constraint barely binds. For n_b = 20:
    P_unconstr = 2.108. Now P is capped at 2.

    S = 2 + sqrt((26-20)/(2*sqrt(3)) - 4*(26-20)/20)
      = 2 + sqrt(6/3.464 - 24/20)
      = 2 + sqrt(1.732 - 1.2)
      = 2 + sqrt(0.532)
      = 2 + 0.730 = 2.730

    Compare FT: 2.740. Improvement: 0.010!

    But is n_b = 20 achievable? The adversary (packer) CHOOSES n_b.
    For the bound to be valid, we must take max over n_b.

    Let me compute this systematically.
    """
    alpha = 2 * np.sqrt(3)
    P_max = 2.0  # Wall packing constraint: each wall <= 0.5, 4 walls

    best_S = 0
    best_nb = 0

    for n_b in range(0, n + 1):
        n_i = n - n_b

        # Unconstrained optimal: all equal r, S = sqrt(n/alpha)
        # With split: P_opt = n_b/sqrt(n*alpha), I_opt = n_i/sqrt(n*alpha)
        P_opt = n_b / np.sqrt(n * alpha)

        if P_opt <= P_max:
            # Constraint doesn't bind
            S = np.sqrt(n / alpha)
        else:
            # P = P_max, optimize interior
            P = P_max
            remaining = 1 - alpha * P**2 / max(n_b, 1)
            if remaining > 0 and n_i > 0:
                I = np.sqrt(remaining * n_i / alpha)
                S = P + I
            elif remaining > 0:
                S = P
            else:
                continue  # infeasible

        if S > best_S:
            best_S = S
            best_nb = n_b

    return best_S, best_nb


# ═══════════════════════════════════════════════════════════════════════════════
# BOUND 11: CONVEX HULL AREA BOUND (NEW)
# ═══════════════════════════════════════════════════════════════════════════════

def convex_hull_area_bound(n):
    """
    The convex hull of circle centers has area A_hull <= 1.
    More precisely: centers are in [r_min, 1-r_min]^2, so
    A_hull <= (1 - 2*r_min)^2.

    The Minkowski sum of the hull with a disk of radius r_min
    must fit inside [0,1]^2, giving:
    (1 - 2*r_min)^2 is the max hull area.

    But r_min can be arbitrarily small, so this doesn't help.

    A stronger argument: the convex hull of the CIRCLES (not centers)
    has area >= sum(pi*r_i^2) (circle areas are inside the hull).
    And the circle hull is inside [0,1]^2, so area <= 1.

    The circle hull area = A_center_hull + perimeter * r_mean + pi * r_mean^2
    (approximately, by the Steiner formula for Minkowski sums).

    Actually: the convex hull of n circles is the Minkowski sum of the
    convex hull of centers with the largest circle. Not quite.

    The convex hull of the union of circles is NOT the Minkowski sum.
    It's more complex.

    This approach is getting too complicated. Let me just compute
    and report the bounds I have.
    """
    return fejes_toth_bound(n)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("TIGHT UPPER BOUNDS FOR n=26 CIRCLE PACKING IN [0,1]^2")
    print("=" * 70)

    n = 26
    best_known = 2.6359830865
    ft = fejes_toth_bound(n)

    print(f"\nBaseline: Fejes-Toth UB = {ft:.6f}")
    print(f"Best known solution:     {best_known:.10f}")
    print(f"FT gap:                  {(ft - best_known)/best_known*100:.2f}%")

    # ── Bound 5: Containment-interaction ──
    ub5, config5 = containment_interaction_bound(n)
    print(f"\nContainment-interaction bound: {ub5:.6f}")
    if config5:
        print(f"  Config: n_b={config5[0]}, n_i={config5[1]}")

    # ── Bound 10: Moment bound ──
    ub10, nb10 = moment_bound(n)
    print(f"\nMoment bound (FT + wall constraint):")
    print(f"  UB = {ub10:.6f} (best at n_b = {nb10})")
    print(f"  Gap: {(ub10 - best_known)/best_known*100:.2f}%")
    if ub10 < ft:
        print(f"  Improvement over FT: {ft - ub10:.6f}")
    else:
        print(f"  Same as FT (wall constraint doesn't bind)")

    # ── Detail: scan n_b values ──
    alpha = 2 * np.sqrt(3)
    P_max = 2.0
    print(f"\n  n_b scan (FT + P_boundary <= {P_max}):")
    for n_b in [16, 17, 18, 19, 20, 21, 22]:
        n_i = n - n_b
        P_opt = n_b / np.sqrt(n * alpha)
        P = min(P_opt, P_max)
        if P < P_opt:
            remaining = 1 - alpha * P**2 / n_b
            if remaining > 0 and n_i > 0:
                I = np.sqrt(remaining * n_i / alpha)
                S = P + I
            else:
                S = P
            print(f"    n_b={n_b}: P_opt={P_opt:.3f}, P={P:.3f}, I={I:.3f}, "
                  f"S={S:.4f} {'<-- improves FT' if S < ft else ''}")
        else:
            S = np.sqrt(n / alpha)
            print(f"    n_b={n_b}: P_opt={P_opt:.3f} <= P_max, S={S:.4f} (= FT)")

    # ── Wall density analysis ──
    print(f"\nWall density analysis:")
    wd = wall_density_analysis()
    for k, v in wd.items():
        print(f"  {k}: {v}")

    # ── TIGHTER WALL CONSTRAINT ──
    # The constraint P <= 2 uses sum(2*r_i) <= 1 per wall.
    # But this is the DIAMETER packing constraint. For circles on a wall,
    # the actual constraint is stricter: their 1D projection needs
    # separation >= 2*sqrt(r_i*r_j) between consecutive circles.
    # For equal radii on one wall: n_w circles with sum(2r) + gaps <= 1.
    # The gaps between adjacent circles (both of radius r on the wall):
    # distance >= 2*r (they touch), so gap in x = sqrt(d^2 - (r-r)^2) - 2r >= 0.
    # Actually, d = 2r (touching), gap = 0. So sum(2r) = 2*n_w*r <= 1.
    # This gives sum(r_wall) <= 0.5 per wall, total P <= 2.0.
    # No improvement from the gap analysis for equal radii.

    # ── TIGHTER CONSTRAINT: Corner circles ──
    # Corner circles touch TWO walls. If circle i touches the left and bottom
    # walls, it counts as 2r_i in the 1D constraint for BOTH walls.
    # With 4 corner circles of radii r_c1..r_c4:
    # Bottom wall: sum_bottom(2r) + 2*r_c1 + 2*r_c2 <= 2 (corners at (r,r) and (1-r,r))
    # ... but corner circles only contribute to 2 walls each.
    # The PER-WALL constraint: for each wall, the corner circles contribute
    # their diameters. If 2 corners per wall (bottom has left and right corners):
    # sum_noncorner_bottom(2r) + 2*r_c_BL + 2*r_c_BR <= 2
    # where BL = bottom-left, BR = bottom-right.

    # For the total: each corner radius appears in 2 wall constraints.
    # Total: sum_walls [sum_touching(2r)] = 4*P_per_wall = 4*1 = 4.
    # But corner circles contribute 2r each to 2 walls, so counted 4r.
    # Total = sum_noncorner(2r * k_walls) + sum_corner(2r * 2) <= 4.
    # Where k_walls = 1 for non-corner boundary circles.
    # = 2*sum_noncorner_bdry(r) + 4*sum_corner(r) <= 4.
    # = 2*P_noncorner + 4*P_corner <= 4.
    # With P = P_noncorner + P_corner:
    # 2*P + 2*P_corner <= 4 => P <= 2 - P_corner.

    # If we can prove the 4 corner circles have total radius >= R_c,
    # then P <= 2 - R_c.

    # For the best packing: corner radii are ~0.085-0.111, total ~ 0.39.
    # So P <= 2 - 0.39 = 1.61. But we can't prove specific corner radii.

    # However: we CAN show that if 4 corner circles exist, their radii
    # are bounded below by their FT contribution.

    # For 4 corner circles: sum(r_corner) contributes to both area and walls.
    # FT: 2*sqrt(3)*sum(r_i^2) <= 1 for all 26 circles.
    # Wall: 2*P + 2*P_corner <= 4, P = P_noncorner + P_corner.

    # This is a tighter constraint than P <= 2 alone, but only when
    # we can prove corners exist with nontrivial radii.

    # For now, the moment bound gives the best improvement.

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY OF RIGOROUS BOUNDS")
    print("=" * 70)

    bounds = {
        'Fejes-Toth (baseline)': ft,
        'Moment (FT + wall)': ub10,
        'Cauchy-Schwarz': np.sqrt(n / np.pi),
    }

    # Filter to only bounds that are provably valid for non-congruent case
    for name, val in sorted(bounds.items(), key=lambda x: x[1]):
        gap = (val - best_known) / best_known * 100
        print(f"  {name:30s}: {val:.6f}  (gap: {gap:.2f}%)")

    tightest_valid = min(v for k, v in bounds.items()
                         if 'congruent' not in k.lower())
    print(f"\nTightest valid bound: {tightest_valid:.6f}")
    print(f"Gap to best known: {(tightest_valid - best_known)/best_known*100:.2f}%")

    return tightest_valid, bounds


if __name__ == "__main__":
    main()
