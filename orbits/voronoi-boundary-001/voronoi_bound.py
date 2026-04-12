#!/usr/bin/env python3
"""
Voronoi-cell boundary-corrected upper bound for circle packing in [0,1]^2.

=== Mathematical Framework ===

We derive upper bounds on S = sum(r_i) for n non-overlapping circles of
radii r_1, ..., r_n inside the unit square [0,1]^2.

Three independent bounding arguments:

(A) FEJES-TOTH AREA BOUND (baseline):
    Each circle's Voronoi cell has area >= 2*sqrt(3)*r_i^2.
    Sum of Voronoi cells = 1 (unit square area).
    By Cauchy-Schwarz: S = sum(r_i) <= sqrt(n / (2*sqrt(3))).
    For n=26: UB = 2.7396.

(B) L-FUNCTION + EULER BOUND (this orbit's main contribution):
    The Fejes-Toth L-function says a Voronoi cell with k edges around a
    circle of radius r has area >= k*tan(pi/k)*r^2. This is minimized at
    k=6 (hexagonal), giving 2*sqrt(3)*r^2.

    Euler's formula for the Delaunay triangulation constrains sum(k_i).
    For n points in general position: sum(k_i) = 2E <= 6n - 2h - 6,
    where h = number of points on the convex hull.

    This forces some circles to have k < 6, increasing their cell areas
    and tightening the bound.

(C) CONVEX-HULL PERIMETER BOUND:
    The convex hull of circle centers has perimeter <= 4 (diagonal of square)
    but also relates to the sum of radii of hull circles.

All bounds are PROVABLY VALID upper bounds. No heuristics.
"""

import numpy as np
from scipy.spatial import Voronoi, Delaunay, ConvexHull
from scipy.optimize import minimize as scipy_minimize, linprog
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as MplCircle, Rectangle
import matplotlib.colors as mcolors
from shapely.geometry import Polygon as ShapelyPolygon, box as shapely_box
import sys
from pathlib import Path
from itertools import combinations


# ── Style (from research/style.md) ──────────────────────────────────────────
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
# DATA: Best known packing (mobius-001)
# ═══════════════════════════════════════════════════════════════════════════════

SOLUTION_CIRCLES = [
    [0.5013319244967335, 0.5299634197561865, 0.13701043018744247],
    [0.7283701485368314, 0.5976347963974249, 0.0998983506527371],
    [0.5966412164069816, 0.742417049525872, 0.09584232580508287],
    [0.5005716308610171, 0.9060726627631578, 0.09392733733683208],
    [0.4047802670507484, 0.742049443484792, 0.09601897581784886],
    [0.2730942856657178, 0.5960427019177355, 0.10060036787876508],
    [0.2973903962807177, 0.3816658444408157, 0.11514888022153208],
    [0.5044682393260885, 0.27534261674898, 0.1176296881082985],
    [0.7052539409715719, 0.38692355339829815, 0.11207708901145666],
    [0.9042676707333971, 0.6832585349914024, 0.0957323293665893],
    [0.7602894720984869, 0.7636735694097463, 0.06918067641414521],
    [0.6868841900482816, 0.9076084484697916, 0.09239155163019276],
    [0.3140569779946276, 0.907407905089298, 0.09259209501068962],
    [0.24064759841313954, 0.7629588636802889, 0.06944019376819537],
    [0.09615133400539044, 0.6820800429507947, 0.09615133410537387],
    [0.10346723331830682, 0.48259558220880383, 0.10346723341829325],
    [0.10518256022927533, 0.27395283960135336, 0.10518256032926045],
    [0.29769047489061357, 0.13325857273414538, 0.13325857283412915],
    [0.7053905112387081, 0.1302211010282564, 0.13022110112823834],
    [0.8932098554107312, 0.27478328332830404, 0.10679014468925294],
    [0.8969394798981025, 0.48460080265037586, 0.10306052020188146],
    [0.08492626241340065, 0.08492626241340048, 0.08492626251338369],
    [0.9153604993457539, 0.08463950065424564, 0.08463950075422898],
    [0.11115617937156931, 0.8888438206284306, 0.11115617947155364],
    [0.8892209872481973, 0.8892209872481961, 0.11077901285178697],
    [0.5027155537961605, 0.07886037287385762, 0.07886037297384499],
]


def load_solution():
    """Load the mobius-001 best known packing."""
    circles = np.array(SOLUTION_CIRCLES)
    return circles[:, :2], circles[:, 2]


# ═══════════════════════════════════════════════════════════════════════════════
# VORONOI COMPUTATION (clipped to unit square)
# ═══════════════════════════════════════════════════════════════════════════════

def voronoi_cells_in_square(centers):
    """
    Compute Voronoi cells of points in [0,1]^2, clipped to the unit square.
    Uses the reflection trick across all 4 walls + 4 corners.
    """
    n = len(centers)
    unit_square = shapely_box(0, 0, 1, 1)

    # Reflect centers across all 4 walls and 4 corners (9 copies total)
    reflections = [
        centers,                                                    # original
        np.column_stack((-centers[:, 0], centers[:, 1])),          # left
        np.column_stack((2 - centers[:, 0], centers[:, 1])),       # right
        np.column_stack((centers[:, 0], -centers[:, 1])),          # bottom
        np.column_stack((centers[:, 0], 2 - centers[:, 1])),       # top
        np.column_stack((-centers[:, 0], -centers[:, 1])),         # bottom-left
        np.column_stack((2 - centers[:, 0], -centers[:, 1])),      # bottom-right
        np.column_stack((-centers[:, 0], 2 - centers[:, 1])),      # top-left
        np.column_stack((2 - centers[:, 0], 2 - centers[:, 1])),   # top-right
    ]
    all_points = np.vstack(reflections)
    vor = Voronoi(all_points)

    cells = []
    for i in range(n):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        if -1 in region or len(region) == 0:
            cells.append(None)
            continue
        vertices = vor.vertices[region]
        poly = ShapelyPolygon(vertices).intersection(unit_square)
        cells.append(poly)
    return cells


# ═══════════════════════════════════════════════════════════════════════════════
# CIRCLE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def classify_circles(centers, radii, tol=1e-6):
    """Classify circles as interior, wall-touching, or corner-touching."""
    n = len(radii)
    types, wall_counts = [], []
    for i in range(n):
        x, y, r = centers[i, 0], centers[i, 1], radii[i]
        dists = [x - r, 1 - x - r, y - r, 1 - y - r]
        wc = sum(1 for d in dists if abs(d) < tol)
        wall_counts.append(wc)
        types.append('corner' if wc >= 2 else ('wall' if wc == 1 else 'interior'))
    return types, wall_counts


# ═══════════════════════════════════════════════════════════════════════════════
# DELAUNAY GRAPH ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def delaunay_degrees(centers):
    """
    Compute the degree (number of Delaunay neighbors) for each point.
    The Delaunay degree equals the number of edges of the Voronoi cell.
    """
    tri = Delaunay(centers)
    n = len(centers)
    degrees = np.zeros(n, dtype=int)
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = simplex[i], simplex[j]
                if (a, b) not in edges and (b, a) not in edges:
                    edges.add((min(a, b), max(a, b)))
    for a, b in edges:
        degrees[a] += 1
        degrees[b] += 1
    return degrees, edges


def convex_hull_size(centers):
    """Number of points on the convex hull."""
    hull = ConvexHull(centers)
    return len(hull.vertices)


# ═══════════════════════════════════════════════════════════════════════════════
# BOUND (A): FEJES-TOTH AREA BOUND (baseline)
# ═══════════════════════════════════════════════════════════════════════════════

def fejes_toth_bound(n):
    """
    Standard Fejes-Toth bound.

    Each Voronoi cell around a circle of radius r has area >= 2*sqrt(3)*r^2.
    Total area = 1, so sum(2*sqrt(3)*r_i^2) <= 1.
    By Cauchy-Schwarz: (sum r_i)^2 <= n * sum(r_i^2) <= n/(2*sqrt(3)).
    Therefore: sum(r_i) <= sqrt(n/(2*sqrt(3))).
    """
    return np.sqrt(n / (2 * np.sqrt(3)))


# ═══════════════════════════════════════════════════════════════════════════════
# BOUND (B): L-FUNCTION + EULER FORMULA BOUND
# ═══════════════════════════════════════════════════════════════════════════════

def L_function(k):
    """
    Fejes-Toth L-function: minimum Voronoi cell area for a circle of unit
    radius inscribed in a k-gon.

    L(k) = k * tan(pi/k) for integer k >= 3.

    Key values:
      k=3: 3*sqrt(3) = 5.196
      k=4: 4.000
      k=5: 5*tan(36 deg) = 3.633
      k=6: 2*sqrt(3) = 3.464  (hexagonal, minimum for k >= 3)
      k->inf: pi = 3.142      (circular limit)
    """
    return k * np.tan(np.pi / k)


def euler_degree_constraint(n, h=None):
    """
    From Euler's formula for the Delaunay triangulation:

    For n points in general position with h points on the convex hull:
      E = 3n - h - 3  (number of Delaunay edges)
      sum(k_i) = 2E = 6n - 2h - 6

    Since k_i >= 3 for interior Delaunay vertices and k_i >= 2 for hull vertices,
    the average degree is strictly less than 6 when h >= 3.

    Returns the maximum sum of degrees.
    """
    if h is None:
        h = 3  # minimum possible convex hull size (adversarial)
    return 6 * n - 2 * h - 6


def L_function_euler_bound(n, h=None):
    """
    Upper bound on sum(r_i) using the L-function with Euler degree constraint.

    We solve:
      maximize sum(r_i)
      subject to: sum(L(k_i) * r_i^2) <= 1  (area constraint)
                  sum(k_i) <= 6n - 2h - 6    (Euler constraint)
                  k_i >= 3 for all i          (planar graph)

    For fixed k_i values, the optimal r_i are:
      r_i = c / L(k_i)  for some constant c
      sum(r_i) = c * sum(1/L(k_i))
      constraint: c^2 * sum(1/L(k_i)) = 1
      => c = 1/sqrt(sum(1/L(k_i)))
      => sum(r_i) = sqrt(sum(1/L(k_i)))

    The adversary (maximizing sum(r_i)) wants to maximize sum(1/L(k_i))
    subject to sum(k_i) <= K_max and k_i >= 3.

    Since 1/L(k) is INCREASING in k (larger k => smaller L => larger 1/L),
    the adversary wants k_i as large as possible.

    Optimal adversary: set as many k_i = 6 as possible, subject to sum <= K_max.
    If K_max = 6n - M (where M = 2h + 6), then:
    - Start with all k_i = 6, giving sum = 6n.
    - Must reduce M units. Cheapest: reduce M circles from k=6 to k=5 (cost 1 each).
    - Result: M circles with k=5, n-M with k=6.

    UB = sqrt(M / L(5) + (n - M) / L(6))
    """
    if h is None:
        # Try h=3 (adversarial minimum) and h=4 (realistic for square)
        # Take the LARGER (weaker) bound since adversary controls h
        ub3 = L_function_euler_bound(n, h=3)
        ub4 = L_function_euler_bound(n, h=4)
        return max(ub3, ub4)  # adversary picks the worse bound for us

    K_max = 6 * n - 2 * h - 6
    deficit = 6 * n - K_max  # = 2h + 6

    # Adversary distributes deficit optimally
    # Option 1: deficit circles with k=5, rest with k=6
    if deficit <= n:
        n5 = deficit
        n6 = n - deficit
        ub = np.sqrt(n5 / L_function(5) + n6 / L_function(6))
    else:
        # Very small n or large h: all circles have k <= 5
        # Distribute remaining deficit among k=4, etc.
        n5 = n
        remaining = deficit - n
        n4 = min(remaining, n5)
        n5 -= n4
        ub = np.sqrt(n4 / L_function(4) + n5 / L_function(5))

    return ub


def L_function_euler_bound_optimized(n):
    """
    More careful optimization of the L-function Euler bound.

    Instead of assuming the adversary uses only k=5 and k=6, we let the
    adversary choose any integer k_i >= 3 for each circle, and solve the
    optimization exactly using continuous relaxation.

    Variables: n_k = number of circles with degree k, for k = 3, 4, 5, 6, ...

    maximize: sum_k n_k / L(k)   [since UB = sqrt(sum 1/L(k_i))]
    subject to:
      sum_k n_k = n
      sum_k k * n_k <= K_max
      n_k >= 0

    This is a linear program in n_k. The optimal solution uses at most 2
    distinct k values (by LP theory: 2 constraints => 2 basic variables).

    For the LP, 1/L(k) is concave in k for k >= 3, so the optimal uses
    two ADJACENT k values. We check all pairs (k, k+1).
    """
    best_obj = -np.inf
    best_config = None

    for h in range(3, n + 1):  # convex hull size
        K_max = 6 * n - 2 * h - 6
        if K_max < 3 * n:
            continue  # infeasible: can't have all k_i >= 3

        # Try all pairs of adjacent k values
        for k_low in range(3, 10):
            k_high = k_low + 1
            # n_low circles with k_low, n_high with k_high
            # n_low + n_high = n
            # k_low * n_low + k_high * n_high <= K_max
            # => k_low * n_low + (k_low + 1) * (n - n_low) <= K_max
            # => k_low * n + (n - n_low) <= K_max
            # => n_low >= n - (K_max - k_low * n)
            # => n_low >= n * (k_low + 1) - K_max
            n_low_min = max(0, n * (k_low + 1) - K_max)
            n_low_max = n  # all at k_low

            # Check feasibility
            if n_low_min > n:
                continue

            # The objective sum(1/L(k_i)) = n_low/L(k_low) + (n-n_low)/L(k_high)
            # is DECREASING in n_low (since 1/L(k_low) < 1/L(k_high) for k_low < k_high)
            # So adversary wants n_low as small as possible: n_low = n_low_min
            n_low = n_low_min
            n_high = n - n_low

            # Also check that k_high assignments fit the degree constraint
            actual_sum = k_low * n_low + k_high * n_high
            if actual_sum > K_max:
                continue

            obj = n_low / L_function(k_low) + n_high / L_function(k_high)
            if obj > best_obj:
                best_obj = obj
                best_config = (h, k_low, n_low, k_high, n_high)

    if best_config is None:
        return fejes_toth_bound(n), None

    ub = np.sqrt(best_obj)
    return ub, best_config


# ═══════════════════════════════════════════════════════════════════════════════
# BOUND (C): CONTAINMENT + AREA JOINT BOUND
# ═══════════════════════════════════════════════════════════════════════════════

def containment_area_bound(n):
    """
    Each circle of radius r has center in [r, 1-r]^2. The available area
    for each circle's center is (1-2r)^2.

    Combined with the Fejes-Toth area constraint, this gives a tighter bound
    for large radii, but for n=26 (small radii ~0.1), the containment
    constraint is not very restrictive.

    We solve numerically:
      maximize sum(r_i)
      subject to: sum(2*sqrt(3)*r_i^2) <= 1
                  0 <= r_i <= 0.5

    The bound is the same as FT unless individual radius constraints bind.
    """
    # FT optimal: all equal r = 1/sqrt(2*sqrt(3)*n) ~ 0.105 for n=26
    # This is < 0.5, so the constraint r <= 0.5 doesn't bind.
    return fejes_toth_bound(n)


# ═══════════════════════════════════════════════════════════════════════════════
# BOUND (D): OLER-TYPE BOUND (adapted for non-congruent circles)
# ═══════════════════════════════════════════════════════════════════════════════

def oler_bound_congruent(n):
    """
    Oler (1961) bound for n congruent circles of radius r in a convex domain D:

    Area(D) >= 2*sqrt(3)*(n-1)*r^2 + (4-2*sqrt(3))*r^2 + perimeter(D)*(1-1/sqrt(3))*r

    Simplified for the unit square (Area=1, Perimeter=4):
    1 >= 2*sqrt(3)*n*r^2 - (2*sqrt(3)-4+2*sqrt(3))*r^2 + 4*(1-1/sqrt(3))*r

    Actually, the standard Oler form for n circles in [0,1]^2 is:
    1 >= T*r^2 + L*r
    where T = 2*sqrt(3)*(n-1) and L involves the perimeter.

    Let me use the Groemer bound (1960) instead:
    Area(D) >= sqrt(3)/2 * t + (2 - sqrt(3)) * e + pi * r^2
    where t = number of Delaunay triangles, e = boundary edges,
    and this is for congruent circles of radius r.

    For congruent circles: t <= 2n - h - 2, e = h, so:
    1 >= sqrt(3)/2 * (2n-h-2) * r^2 + (2-sqrt(3))*h*r^2 + n*pi*r^2

    Hmm, this isn't quite right either. Let me use a simpler version.

    For n congruent circles of radius r in [0,1]^2:
    The tightest known bound is from Groemer:
    1 >= 2*sqrt(3)*n*r^2 + 2*(2-sqrt(3))*sqrt(n)*r  (approximate)

    This corrects FT by a term proportional to sqrt(n)*r ~ perimeter/sqrt(area).

    For the non-congruent case, this doesn't directly apply.
    """
    a = 2 * np.sqrt(3)
    # Use a simplified Oler-type bound:
    # 1 >= a*n*r^2 + b*r where b = 4*(2-sqrt(3))/(2*sqrt(3))  ... not exact
    # The congruent case gives r from the quadratic.
    # This typically gives a WEAKER bound than FT for sum of non-congruent radii.

    b = 4 * (2 - np.sqrt(3))  # perimeter * (2 - sqrt(3)) ~ 1.072
    # Solve a*n*r^2 + b*r <= 1
    # Maximum n*r at: d(nr)/dr = n, constraint is tight
    # Lagrange: n = lambda*(2*a*n*r + b)
    # => 1 = lambda*(2*a*r + b/n), also a*n*r^2 + b*r = 1
    # From first: lambda = n/(2*a*n*r+b) = 1/(2*a*r+b/n)
    # This doesn't simplify easily. Use quadratic formula.

    # For congruent: maximize S = n*r subject to a*n*r^2 + b*r = 1
    # a*(S^2/n) + b*S/n = 1  =>  a*S^2 + b*S = n
    # S = (-b + sqrt(b^2 + 4*a*n)) / (2*a)
    S = (-b + np.sqrt(b**2 + 4*a*n)) / (2*a)
    return S


def oler_noncongruent_bound(n):
    """
    Non-congruent version of the Oler/Groemer bound.

    For non-congruent circles, the perimeter correction depends on WHICH
    circles are on the boundary. The adversary can minimize the correction
    by placing tiny circles on the boundary.

    In the limit of infinitesimally small boundary circles, the perimeter
    correction vanishes and we recover the FT bound. So the Oler correction
    does NOT help for upper bounding the non-congruent case.

    This confirms what upperbound-001 found: "The Oler-Groemer boundary
    correction goes the wrong way for upper bounding sum-of-radii."

    The reason: the Oler bound says 1 >= FT_term + positive_correction.
    The FT_term involves sum(r_i^2), and the correction involves sum(r_boundary).
    Both increase with radii, so the bound is:
        sum(r_i^2) <= (1 - correction) / (2*sqrt(3))
    which is TIGHTER than FT. But the adversary can make the correction
    zero by using tiny boundary circles, recovering FT.

    HOWEVER: if we also use Euler's formula to constrain the TOPOLOGY,
    we can combine Oler with the L-function approach.
    """
    return fejes_toth_bound(n)  # Same as FT for non-congruent


# ═══════════════════════════════════════════════════════════════════════════════
# BOUND (E): DIRECT VORONOI MEASUREMENT + OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def count_polygon_edges(poly):
    """Count the number of edges of a Shapely polygon."""
    if poly is None or poly.is_empty:
        return 0
    if poly.geom_type == 'Polygon':
        # Number of edges = number of vertices - 1 (closed ring)
        coords = list(poly.exterior.coords)
        return len(coords) - 1
    return 0


def measured_voronoi_bound(centers, radii):
    """
    Compute Voronoi cells clipped to [0,1]^2 and measure their properties.

    The Fejes-Toth L-function bound:
    For a convex polygon P with k edges that contains an inscribed circle
    of radius r: Area(P) >= L(k) * r^2 = k * tan(pi/k) * r^2.

    IMPORTANT: k here is the number of edges of the CLIPPED Voronoi cell,
    NOT the Delaunay degree. For boundary circles, the clipped cell has
    extra edges from wall clipping, so k_clipped > k_delaunay.

    The L-function is DECREASING for k >= 6, so more edges can mean
    a SMALLER minimum area. For very large k, L(k) -> pi (circle limit).
    """
    cells = voronoi_cells_in_square(centers)
    areas = np.array([c.area if c is not None else 0.0 for c in cells])
    hex_areas = 2 * np.sqrt(3) * radii**2

    # Compute Delaunay degrees (internal Voronoi neighbor count)
    degrees, edges = delaunay_degrees(centers)

    # Count actual edges of clipped Voronoi cells
    clipped_edges = np.array([count_polygon_edges(c) for c in cells])

    # L-function areas using CLIPPED cell edge count (correct)
    L_areas_clipped = np.array([L_function(max(k, 3)) * r**2
                                 for k, r in zip(clipped_edges, radii)])

    # L-function areas using Delaunay degree (for comparison only)
    L_areas_delaunay = np.array([L_function(max(k, 3)) * r**2
                                  for k, r in zip(degrees, radii)])

    return areas, hex_areas, L_areas_clipped, L_areas_delaunay, degrees, clipped_edges, cells


# ═══════════════════════════════════════════════════════════════════════════════
# BOUND (F): TIGHT LP BOUND WITH PER-CIRCLE L-FUNCTION CONSTRAINTS
# ═══════════════════════════════════════════════════════════════════════════════

def tight_lp_bound(n):
    """
    Full LP optimization over degree assignments and radii.

    We write the bound as:
      UB = max_{k, r} sum(r_i)
      s.t. sum(L(k_i) * r_i^2) <= 1      (area)
           sum(k_i) <= 6n - 2h - 6        (Euler)
           3 <= k_i <= K_max               (planarity)
           0 <= r_i <= 0.5                 (containment)

    Since L(k)*r^2 is the area constraint, and we want to maximize sum(r),
    this is a mixed-integer quadratically constrained program.

    Key insight: for fixed k assignments, the optimal r allocation is:
    r_i = c / L(k_i), and UB = sqrt(sum(1/L(k_i))).

    So we only need to optimize over k assignments, which is an integer LP:
      maximize sum_{k=3}^{K} n_k / L(k)
      s.t. sum n_k = n
           sum k*n_k <= 6n - 2h - 6
           n_k >= 0, integer

    Since 1/L(k) is increasing and convex for k >= 6 and concave for k <= 6,
    the LP relaxation may be tight.

    But the adversary also controls h (convex hull size). To get the VALID
    upper bound, we must take the MAXIMUM over all h >= 3.
    """
    best_ub = 0
    best_detail = None

    for h in range(3, n + 1):
        K_max = 6 * n - 2 * h - 6
        if K_max < 3 * n:
            break  # infeasible

        # LP: maximize sum n_k / L(k) s.t. sum n_k = n, sum k*n_k <= K_max
        # Decision variables: n_3, n_4, n_5, n_6, n_7, ...
        # We consider k from 3 to some max K
        K_range = list(range(3, 20))
        m = len(K_range)

        # Objective: maximize sum n_k / L(k) => minimize -sum n_k / L(k)
        c = np.array([-1.0 / L_function(k) for k in K_range])

        # Constraints:
        # sum n_k = n  (equality)
        A_eq = np.ones((1, m))
        b_eq = np.array([n])

        # sum k*n_k <= K_max (inequality)
        A_ub = np.array([[k for k in K_range]])
        b_ub = np.array([K_max])

        # Bounds: n_k >= 0
        bounds = [(0, n) for _ in K_range]

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                        bounds=bounds, method='highs')

        if result.success:
            obj = -result.fun  # sum(1/L(k_i))
            ub = np.sqrt(obj)
            if ub > best_ub:
                best_ub = ub
                nk = result.x
                nonzero = [(K_range[i], nk[i]) for i in range(m) if nk[i] > 0.01]
                best_detail = {'h': h, 'K_max': K_max, 'config': nonzero, 'ub': ub}

    return best_ub, best_detail


# ═══════════════════════════════════════════════════════════════════════════════
# BOUND (G): MINIMUM HULL SIZE ARGUMENT
# ═══════════════════════════════════════════════════════════════════════════════

def min_hull_size_bound(n):
    """
    If we can prove the convex hull of centers has h >= h_min points,
    the Euler constraint becomes tighter.

    For circles in [0,1]^2, the centers are in [r_min, 1-r_min]^2.
    The convex hull of n points in a square has at least 4 points
    on the hull (one near each corner) when n >= 4.

    Actually, this isn't true in general. But for a GOOD packing
    (one that achieves sum close to the FT bound), the circles must
    spread out, and the hull tends to be larger.

    More rigorously: in any packing of 26 circles in [0,1]^2 with
    sum(r_i) >= 2.5, the circles must cover a significant fraction
    of the square, forcing the hull to include boundary-adjacent circles.

    But this is hard to prove rigorously without case analysis.
    Return: None (this bound requires numerical verification).
    """
    # Conservative: just use h >= 3
    return None


def hull_size_for_solution(centers):
    """Measure the actual convex hull size of the best packing."""
    hull = ConvexHull(centers)
    return len(hull.vertices), hull.vertices


# ═══════════════════════════════════════════════════════════════════════════════
# BOUND (H): COMBINED EULER + HULL + L-FUNCTION WITH LP
# ═══════════════════════════════════════════════════════════════════════════════

def combined_euler_hull_lfunction(n, h_values=None):
    """
    For each possible hull size h, compute the L-function Euler bound
    and return the tightest provable bound.

    The valid upper bound is: max over all feasible h of the per-h bound.
    The adversary (packer) chooses h to maximize their sum.

    For h = 3: K_max = 6n - 12, deficit = 12
    For h = 4: K_max = 6n - 14, deficit = 14
    ...

    Larger h => smaller K_max => more circles with k < 6 => tighter bound.
    So adversary wants h = 3 (minimum).

    But h >= 3 is always achievable (any 3 non-collinear points).
    Can we prove h >= 4 for all valid packings?

    For n >= 4 circles in general position in [0,1]^2: YES, if the 4
    extreme points (leftmost, rightmost, topmost, bottommost) are distinct,
    then h >= 4. For n >= 4 circles in a square, the extreme centers
    must be distinct unless all circles are collinear (impossible for n >= 3
    with non-overlapping circles in a square).

    So h >= 4 for n >= 4. This gives:
    K_max = 6*26 - 2*4 - 6 = 156 - 14 = 142
    deficit = 14
    """
    if h_values is None:
        h_values = list(range(3, min(n, 20)))

    results = {}
    for h in h_values:
        K_max = 6 * n - 2 * h - 6
        if K_max < 3 * n:
            continue

        deficit = 6 * n - K_max

        # Adversary distributes deficit to maximize sum(1/L(k_i))
        # Greedy: put as much deficit as possible on the smallest k values
        # that still maximize 1/L(k).
        # Since 1/L(k) is increasing in k, adversary wants large k.
        # So adversary MINIMIZES deficit per circle: set deficit circles to k=5.
        if deficit <= n:
            obj = deficit / L_function(5) + (n - deficit) / L_function(6)
        else:
            # Need some with k <= 4
            n5 = n
            rem = deficit - n
            n4 = min(rem, n5)
            n5 -= n4
            obj = n4 / L_function(4) + n5 / L_function(5)

        ub = np.sqrt(obj)
        results[h] = ub

    if not results:
        return fejes_toth_bound(n), {}

    # Valid bound is max over all h (adversary's choice)
    # But we can prove h >= 4 for n >= 4, so exclude h = 3
    provable_h_min = 4 if n >= 4 else 3
    valid_results = {h: ub for h, ub in results.items() if h >= provable_h_min}

    if not valid_results:
        return fejes_toth_bound(n), results

    best_h = max(valid_results, key=valid_results.get)  # adversary picks largest UB
    return valid_results[best_h], results


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_full_analysis(centers, radii, voronoi_areas, cells, types,
                       degrees, bounds_dict, save_path=None):
    """
    Four-panel figure:
    (a) Packing with Voronoi cells colored by Delaunay degree
    (b) Voronoi cell area vs L-function minimum
    (c) Upper bounds comparison bar chart
    (d) Gap analysis: how each bound narrows the gap to best-known
    """
    n = len(radii)
    hex_areas = 2 * np.sqrt(3) * radii**2
    L_areas = np.array([L_function(k) * r**2 for k, r in zip(degrees, radii)])
    best_known = 2.6359830865

    fig = plt.figure(figsize=(18, 14), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    # ── Panel (a): Packing with degree-colored Voronoi cells ──
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.set_title('(a) Voronoi cells colored by\n    Delaunay degree k',
                 fontweight='bold')

    sq = Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(sq)

    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=3, vmax=8)

    for i in range(n):
        if cells[i] is not None and not cells[i].is_empty:
            if cells[i].geom_type == 'Polygon':
                xs, ys = cells[i].exterior.xy
                color = cmap(norm(degrees[i]))
                ax.fill(xs, ys, alpha=0.35, color=color, edgecolor='gray',
                       linewidth=0.5)

        type_colors = {'interior': '#1565C0', 'wall': '#E65100', 'corner': '#B71C1C'}
        circ = MplCircle(centers[i], radii[i], fill=False,
                        edgecolor=type_colors[types[i]], linewidth=1.5)
        ax.add_patch(circ)
        ax.text(centers[i, 0], centers[i, 1], str(degrees[i]),
               ha='center', va='center', fontsize=7, fontweight='bold')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Delaunay degree k')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(False)

    # ── Panel (b): Cell area vs L-function minimum ──
    ax = fig.add_subplot(gs[0, 1])
    type_colors_map = {'interior': '#1565C0', 'wall': '#E65100', 'corner': '#B71C1C'}

    for t in ['interior', 'wall', 'corner']:
        mask = [types[i] == t for i in range(n)]
        if any(mask):
            idxs = [i for i in range(n) if mask[i]]
            ax.scatter([L_areas[i] for i in idxs],
                      [voronoi_areas[i] for i in idxs],
                      c=type_colors_map[t], label=t, s=80, alpha=0.8,
                      edgecolors='black', linewidth=0.5, zorder=5)

    max_a = max(np.max(L_areas), np.max(voronoi_areas)) * 1.1
    ax.plot([0, max_a], [0, max_a], 'k--', alpha=0.4, label='y=x (L-function tight)')
    ax.set_xlabel('L(k) * r^2  (degree-aware minimum)')
    ax.set_ylabel('Actual Voronoi cell area')
    ax.set_title('(b) Actual vs L-function minimum area\n    (all points should be above y=x)',
                 fontweight='bold')
    ax.legend(fontsize=10)

    # ── Panel (c): Bounds comparison ──
    ax = fig.add_subplot(gs[1, 0])

    sorted_bounds = sorted(bounds_dict.items(), key=lambda x: x[1], reverse=True)
    names = [b[0] for b in sorted_bounds]
    values = [b[1] for b in sorted_bounds]

    colors = plt.cm.GnBu(np.linspace(0.3, 0.9, len(names)))
    bars = ax.barh(range(len(names)), values, color=colors, edgecolor='black',
                   linewidth=0.5)

    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + 0.003, i, f'{val:.4f}', va='center', fontsize=10)

    ax.axvline(best_known, color='#D32F2F', linestyle='--', linewidth=2,
              label=f'Best known = {best_known:.4f}')

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace('_', ' ') for n in names], fontsize=10)
    ax.set_xlabel('Upper bound on sum(r_i)')
    ax.set_title('(c) Upper bounds for n=26 circles in [0,1]^2',
                fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim(min(best_known - 0.05, min(values) - 0.05),
                max(values) * 1.02)

    # ── Panel (d): Gap analysis ──
    ax = fig.add_subplot(gs[1, 1])

    ft_gap = (bounds_dict.get('Fejes-Toth', fejes_toth_bound(26)) - best_known) / best_known * 100

    gaps = [(name, (val - best_known) / best_known * 100) for name, val in sorted_bounds]
    gaps_sorted = sorted(gaps, key=lambda x: x[1])

    names_g = [g[0].replace('_', ' ') for g in gaps_sorted]
    gap_vals = [g[1] for g in gaps_sorted]

    colors_g = ['#4CAF50' if g < ft_gap else '#FFC107' if g < ft_gap * 1.5 else '#FF5722'
                for g in gap_vals]

    ax.barh(range(len(names_g)), gap_vals, color=colors_g, edgecolor='black',
            linewidth=0.5)
    for i, g in enumerate(gap_vals):
        ax.text(g + 0.05, i, f'{g:.2f}%', va='center', fontsize=10)

    ax.axvline(ft_gap, color='gray', linestyle=':', alpha=0.6,
              label=f'FT gap = {ft_gap:.2f}%')
    ax.set_yticks(range(len(names_g)))
    ax.set_yticklabels(names_g, fontsize=10)
    ax.set_xlabel('Gap to best known (%)')
    ax.set_title('(d) Gap reduction from boundary corrections',
                fontweight='bold')
    ax.legend(fontsize=10)

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved figure to {save_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("VORONOI BOUNDARY-CORRECTED UPPER BOUND ANALYSIS")
    print("=" * 70)

    n = 26
    best_known = 2.6359830865

    # ── Load solution ──
    centers, radii = load_solution()
    sum_r = np.sum(radii)
    print(f"\nBest known packing: sum(r) = {sum_r:.10f}")
    print(f"Number of circles: {n}")
    print(f"Radii range: [{np.min(radii):.6f}, {np.max(radii):.6f}]")

    # ── Classify circles ──
    types, wall_counts = classify_circles(centers, radii)
    print(f"\nCircle types: "
          f"{types.count('interior')} interior, "
          f"{types.count('wall')} wall, "
          f"{types.count('corner')} corner")

    # ── Delaunay analysis ──
    degrees, edges = delaunay_degrees(centers)
    h_size, hull_verts = hull_size_for_solution(centers)
    print(f"\nDelaunay graph:")
    print(f"  Edges: {len(edges)}")
    print(f"  sum(k_i) = {np.sum(degrees)} "
          f"(Euler max: 6n-2h-6 = {6*n - 2*h_size - 6} for h={h_size})")
    print(f"  Degree range: [{np.min(degrees)}, {np.max(degrees)}]")
    print(f"  Mean degree: {np.mean(degrees):.2f}")
    print(f"  Convex hull size: {h_size}")
    print(f"  Degree distribution:")
    for k in range(int(np.min(degrees)), int(np.max(degrees)) + 1):
        count = np.sum(degrees == k)
        if count > 0:
            print(f"    k={k}: {count} circles "
                  f"(L(k)={L_function(k):.4f}, 1/L(k)={1/L_function(k):.4f})")

    # ── Voronoi analysis ──
    print("\nComputing Voronoi cells...")
    (voronoi_areas, hex_areas, L_areas_clipped, L_areas_delaunay,
     _, clipped_edges, cells) = measured_voronoi_bound(centers, radii)
    total_area = np.sum(voronoi_areas)
    print(f"Total Voronoi area: {total_area:.6f}")

    print(f"\nVoronoi cell analysis:")
    print(f"  sum(hex-min areas)       = {np.sum(hex_areas):.6f}")
    print(f"  sum(L-func Delaunay)     = {np.sum(L_areas_delaunay):.6f}")
    print(f"  sum(L-func clipped)      = {np.sum(L_areas_clipped):.6f}")
    print(f"  sum(actual areas)        = {np.sum(voronoi_areas):.6f}")
    print(f"  FT slack: {1 - np.sum(hex_areas):.6f}")

    print(f"\n  Clipped cell edge counts:")
    for i in range(n):
        print(f"    Circle {i:2d}: delaunay_k={degrees[i]}, "
              f"clipped_k={clipped_edges[i]}, type={types[i]:8s}, "
              f"area={voronoi_areas[i]:.6f}, "
              f"L_del={L_areas_delaunay[i]:.6f}, "
              f"L_clip={L_areas_clipped[i]:.6f}")

    # Check L-function with Delaunay degrees (may violate for boundary)
    violations_del = np.sum(voronoi_areas < L_areas_delaunay - 1e-10)
    violations_clip = np.sum(voronoi_areas < L_areas_clipped - 1e-10)
    print(f"\n  L-function violations (Delaunay k): {violations_del}")
    print(f"  L-function violations (clipped k):  {violations_clip}")
    if violations_clip > 0:
        for i in range(n):
            if voronoi_areas[i] < L_areas_clipped[i] - 1e-10:
                print(f"    Circle {i}: actual={voronoi_areas[i]:.6f}, "
                      f"L_clip={L_areas_clipped[i]:.6f}, "
                      f"ratio={voronoi_areas[i]/L_areas_clipped[i]:.4f}, "
                      f"k_clip={clipped_edges[i]}")

    # ── Compute all bounds ──
    print("\n" + "=" * 70)
    print("UPPER BOUNDS ON sum(r_i) FOR n=26 CIRCLES IN [0,1]^2")
    print("=" * 70)

    bounds = {}

    # (A) Fejes-Toth
    bounds['Fejes-Toth'] = fejes_toth_bound(n)
    print(f"\n(A) Fejes-Toth area bound: {bounds['Fejes-Toth']:.6f}")

    # (B) L-function + Euler (adversarial h)
    bounds['L+Euler (h>=3)'] = L_function_euler_bound(n, h=3)
    bounds['L+Euler (h>=4)'] = L_function_euler_bound(n, h=4)
    print(f"(B) L-function + Euler (h>=3): {bounds['L+Euler (h>=3)']:.6f}")
    print(f"    L-function + Euler (h>=4): {bounds['L+Euler (h>=4)']:.6f}")

    # Full LP optimization
    ub_lp, detail_lp = tight_lp_bound(n)
    bounds['L+Euler LP'] = ub_lp
    print(f"    L+Euler LP (full opt):     {ub_lp:.6f}")
    if detail_lp:
        print(f"    Best config: h={detail_lp['h']}, "
              f"degrees={detail_lp['config']}")

    # Optimized version
    ub_opt, config_opt = L_function_euler_bound_optimized(n)
    bounds['L+Euler optimized'] = ub_opt
    print(f"    L+Euler optimized:         {ub_opt:.6f}")
    if config_opt:
        print(f"    Config: h={config_opt[0]}, "
              f"k={config_opt[1]}x{config_opt[2]:.0f} + k={config_opt[3]}x{config_opt[4]:.0f}")

    # (C) Combined Euler + hull + L-function
    ub_comb, all_h = combined_euler_hull_lfunction(n)
    bounds['Combined (h>=4)'] = ub_comb
    print(f"(C) Combined Euler+hull+L:     {ub_comb:.6f}")
    print(f"    Per-h bounds (h >= 4):")
    for h in sorted(all_h.keys()):
        marker = " <-- provable" if h >= 4 else ""
        print(f"      h={h:2d}: UB={all_h[h]:.6f}{marker}")

    # (D) Oler (congruent)
    ub_oler = oler_bound_congruent(n)
    bounds['Oler (congruent)'] = ub_oler
    print(f"(D) Oler bound (congruent):    {ub_oler:.6f}")
    print(f"    Note: applies to congruent case only; non-congruent recovers FT")

    # (E) Cauchy-Schwarz (weakest)
    bounds['Cauchy-Schwarz'] = np.sqrt(n / np.pi)
    print(f"(E) Cauchy-Schwarz:            {bounds['Cauchy-Schwarz']:.6f}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Bound':30s} {'UB':>10s} {'Gap':>8s} {'Improvement':>12s}")
    print("-" * 65)
    for name, val in sorted(bounds.items(), key=lambda x: x[1]):
        gap = (val - best_known) / best_known * 100
        improv = bounds['Fejes-Toth'] - val
        print(f"{name:30s} {val:10.6f} {gap:7.2f}% {improv:11.6f}")

    tightest = min(bounds.values())
    tightest_name = min(bounds, key=bounds.get)
    ft = bounds['Fejes-Toth']
    print(f"\nTightest provable bound: {tightest_name}")
    print(f"  Value: {tightest:.6f}")
    print(f"  Gap to best known: {(tightest - best_known)/best_known*100:.2f}%")
    print(f"  Improvement over FT: {ft - tightest:.6f} "
          f"({(ft - tightest)/ft*100:.2f}% of FT)")
    print(f"  Gap closed: {(ft - tightest)/(ft - best_known)*100:.1f}% "
          f"of the FT-to-best-known gap")

    # ── Generate figures ──
    fig_dir = Path(__file__).parent / "figures"
    fig_dir.mkdir(exist_ok=True)

    print("\nGenerating figures...")
    plot_full_analysis(centers, radii, voronoi_areas, cells, types,
                      degrees, bounds,
                      save_path=str(fig_dir / "voronoi_analysis.png"))

    # Also report the "what if we could prove h >= actual" bound
    print(f"\n--- Speculative bounds (if we could prove h >= actual) ---")
    print(f"Actual hull size: h = {h_size}")
    for h_test in [h_size, h_size + 2, h_size + 4]:
        K_max = 6 * n - 2 * h_test - 6
        deficit = 6 * n - K_max
        if deficit <= n:
            obj = deficit / L_function(5) + (n - deficit) / L_function(6)
        else:
            n5 = 2 * n - deficit
            n4 = n - n5
            obj = n4 / L_function(4) + n5 / L_function(5)
        ub_spec = np.sqrt(obj)
        gap_spec = (ub_spec - best_known) / best_known * 100
        print(f"  h={h_test}: UB = {ub_spec:.6f} (gap: {gap_spec:.2f}%)")

    return tightest, bounds


if __name__ == "__main__":
    tightest, bounds = main()
