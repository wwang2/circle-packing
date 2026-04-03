"""
Attempt to beat the Fejes Toth bound for circle packing in a square.

Current best verified bound: sqrt(n/(2*sqrt(3))) = 2.7396 for n=26.
Known best solution: 2.6360.
Gap: 3.93%.

APPROACH: Boundary-aware density bound.

The Fejes Toth bound says each Voronoi cell has density <= pi/(2*sqrt(3)).
But for circles NEAR THE BOUNDARY of the square, the Voronoi cell is
clipped by the square boundary, creating "dead space" between the circle
and the wall.

For a circle of radius r with center at distance d from the nearest wall
(where d >= r for containment):
- If d = r (touching wall), the Voronoi cell includes a strip of width r
  between the circle and the wall. This strip has area ~ 2r * r = 2r^2
  (for a circle touching one wall).
- This dead space means the effective Voronoi cell area is LARGER than
  2*sqrt(3)*r^2, making the density LOWER.

The MINIMUM density (maximum Voronoi cell area) occurs for corner circles.

Can we quantify the total boundary waste?

LEMMA: For n circles in [0,1]^2, partition them into:
- Interior circles: center distance > R from all walls
- Boundary circles: center distance <= R from some wall

For interior circles, FT gives density <= pi/(2*sqrt(3)).
For boundary circles, the density is lower due to wall effects.

The total area used: sum(Voronoi_i) = 1 (partitions the square).
For boundary circles with extra waste, the effective constraint is:

sum_{interior} 2*sqrt(3)*r_i^2 + sum_{boundary} A_eff_i(r_i, d_i) <= 1

where A_eff_i >= 2*sqrt(3)*r_i^2 for boundary circles.

This is hard to compute without knowing the configuration.

ALTERNATIVE APPROACH: Integral bound.

The key observation: int_0^1 int_0^1 f(x,y) dx dy = 1 where f is the
indicator of [0,1]^2. We can write:

sum(r_i) = sum_i r_i
         = sum_i [r_i / A_i] * A_i    (where A_i = Voronoi cell area)

By Cauchy-Schwarz:
[sum r_i]^2 = [sum (r_i/A_i) * A_i]^2
            <= [sum (r_i/A_i)^2 * A_i] * [sum A_i]
            = [sum r_i^2/A_i] * 1

Now, r_i^2/A_i = r_i^2/A_i. FT says A_i >= 2*sqrt(3)*r_i^2.
So r_i^2/A_i <= 1/(2*sqrt(3)).
And sum r_i^2/A_i <= n/(2*sqrt(3)).
Giving [sum r_i]^2 <= n/(2*sqrt(3)). Same bound.

But: if some A_i are LARGER (boundary circles), then r_i^2/A_i is SMALLER,
and the bound is TIGHTER. The problem is we don't know which circles
are near the boundary.

ALTERNATIVE: WEIGHTED Cauchy-Schwarz.

Let w_i be weights. Then:
[sum r_i]^2 = [sum w_i * (r_i/w_i)]^2
            <= [sum w_i^2/A_i] * [sum r_i^2/w_i^2 * A_i]
            ... this doesn't simplify nicely.

ALTERNATIVE: Parametric bound.

For each possible Voronoi cell configuration, compute the bound.
Too complex.

APPROACH THAT WORKS: Quantified boundary correction.

The unit square has perimeter P = 4. The "boundary strip" of width epsilon
has area ~ 4*epsilon (for small epsilon).

Circles touching a wall: their Voronoi cells include the wall-circle gap.
The total "wall-adjacent" Voronoi area is at least the boundary strip area.

Specifically: the Voronoi cells of the k circles closest to each wall
must cover the boundary strip on that side.

At minimum, the total area of boundary Voronoi cells includes the
4 * (2r_boundary) strips = 8*r_boundary of dead space (for uniform
r_boundary). But we don't know r_boundary.

CLEANEST APPROACH: Use the Steiner-type inequality.

For a packing of n circles {B(x_i, r_i)} in a convex body K:
Define the "outer parallel body" of the union of circles by radius 0:
That's just the union of circles.

Area(union) = sum(pi*r_i^2) (since non-overlapping).

Now, the Minkowski content gives:
For ANY non-overlapping circles inside K:
  area(K) >= area(union) + "gap area"

The gap area is the area of K NOT covered by any circle.
This gap includes:
1. Interior gaps (between circles)
2. Boundary gaps (between circles and walls)

For hexagonal packing, interior gap fraction = 1 - pi/(2*sqrt(3)) ≈ 0.093.
Boundary gap is additional.

Total gap >= boundary gap.
Boundary gap: each wall of length 1 has circles touching it (or near it).
Between the wall and the nearest circle, there's a gap.

The minimum boundary gap for a circle of radius r at distance r from a wall
is the area between the wall and the circle in the Voronoi cell.
For a circle touching a wall at x=0: the gap between x=0 and the circle
(which extends from 0 to 2r in x) is: the area in the Voronoi cell to the
LEFT of the circle = integral from 0 to r of [chord_width - 0] dx...
Actually the gap is the area in the strip 0 < x < r that's outside the circle.

For a circle centered at (r, y_c) with radius r:
The strip 0 < x < r has width 1 and contains part of the circle.
Circle footprint in strip: the circle extends from x=0 to x=2r.
In the strip 0 < x < r, the circle's chord width at position x is
2*sqrt(r^2 - (x-r)^2) = 2*sqrt(2rx - x^2) = 2*sqrt(x)*sqrt(2r-x).

The CIRCLE area in the strip 0 < x < r is:
integral_0^r 2*sqrt(x*(2r-x)) dx = pi*r^2/2 (semicircle area).

The Voronoi cell of this circle in the strip includes the full strip width
from 0 to r (approximately). If the strip has width w (in y), the strip
area is r*w, and the circle area in it is ~pi*r^2/2.
The gap area is r*w - pi*r^2/2.

This is getting too complicated. Let me try a NUMERICAL approach:
solve an LP/SDP that directly models the boundary effects.

SIMPLEST TIGHTER BOUND: Dowker's theorem.

Dowker (1944): For any convex polygon with m sides inscribed in a circle
of radius r, its area is at most m*r^2*sin(2*pi/m)/2.

For the Voronoi cell of a circle in hexagonal packing, m=6:
Area = 6*r^2*sin(pi/3)/2 ... wait, this is different.

Actually, Fejes Toth's result uses: for a convex polygon containing a circle
of radius r, the minimum area is achieved by a regular hexagon, giving
area = 2*sqrt(3)*r^2.

For a Voronoi cell CLIPPED by a wall: the cell is no longer convex (or is
convex but different shape). The minimum area may be larger.

CONCRETE BOUND: Consider the perimeter method.

For any packing: sum(2*pi*r_i) = total circle perimeter.
The "free boundary" of the packing (boundary of union of circles inside K)
has length sum(2*pi*r_i) - contact_lengths.

Also: perimeter of K = 4 = sum of visible wall segments + sum of circle
arcs visible from outside... This gives a relationship between radii
and wall contacts.

For circles touching walls: each circle touching a wall at one point has
approximately 2r of wall contact (the chord). Wait, a circle touching a
flat wall at one point has zero contact length.

Actually: the boundary of the "packing region" (union of circles + wall gaps)
equals the boundary of K = 4. The part of the boundary that's wall = 4 -
(total circle arc visible from outside).

The wall-visible arc of a circle touching one wall is pi*r (semicircle).
For interior circles, wall-visible arc = 0.

So: 4 - sum_{boundary circles} visible_arc_i = remaining_wall.
And: remaining_wall >= 0.

This gives: sum visible_arc_i <= 4.

Not directly useful for bounding sum(r_i).

Let me just compute the FT bound with the Oler correction properly
using the CORRECT Oler formula, and verify it numerically.
"""

import numpy as np
from scipy.optimize import minimize_scalar, minimize
import json
import sys
from pathlib import Path


def fejes_toth_cs_bound(n):
    """
    Fejes Toth + Cauchy-Schwarz bound.
    sum r_i <= sqrt(n/(2*sqrt(3))).
    """
    return np.sqrt(n / (2 * np.sqrt(3)))


def oler_generalized_bound(n, verbose=False):
    """
    Generalized Oler bound for mixed radii.

    The correct Oler-Groemer inequality for mixed-radii packings in a convex
    body K states:

    For n non-overlapping circles of radii r_1,...,r_n contained in K:
    sum_i (2*sqrt(3) * r_i^2) <= area(K) + perimeter(K) * max(r_i) + pi * max(r_i)^2

    For K = [0,1]^2:
    sum_i (2*sqrt(3) * r_i^2) <= 1 + 4*r_max + pi*r_max^2

    Combined with C-S: (sum r_i)^2 <= n * sum(r_i^2)
                                    <= n * (1 + 4*r_max + pi*r_max^2) / (2*sqrt(3))

    We need to find the r_max that gives the tightest bound.
    r_max is in [0, 0.5] and r_max >= max(r_i).
    The bound holds for ANY r_max >= max(r_i).

    For the C-S bound to be tight, all r_i should be equal to r_max.
    Then: n * 2*sqrt(3)*r_max^2 <= 1 + 4*r_max + pi*r_max^2.
    And sum = n*r_max.

    This gives a VALID bound for equal radii. For mixed radii, the bound is:
    sum r_i <= sqrt(n * (1 + 4*r_max + pi*r_max^2) / (2*sqrt(3)))

    But we need r_max >= max actual radius. Since we don't know r_max,
    we OPTIMIZE over r_max in [0, 0.5]:

    Bound(r_max) = sqrt(n * (1 + 4*r_max + pi*r_max^2) / (2*sqrt(3)))

    This is INCREASING in r_max (since 1 + 4r + pi*r^2 is increasing for r > 0).
    So the tightest bound is at the SMALLEST r_max.

    But r_max >= some minimum value. What's the minimum possible r_max
    for n circles maximizing sum(r_i)?

    If r_max is small, then sum(r_i) <= n*r_max is small. So there's a tradeoff.

    The actual bound: for any packing, EITHER:
    (a) r_max <= R, in which case sum(r_i) <= n*R, OR
    (b) r_max > R, in which case sum(r_i) <= sqrt(n*(1+4R+pi*R^2)/(2*sqrt(3)))
        ... wait, this is wrong. If r_max > R, we need to use r_max, not R.

    Actually the correct thing: the bound is
    sum r_i <= sqrt(n * (1 + 4*r_max + pi*r_max^2) / (2*sqrt(3)))
    where r_max = max(r_i). This depends on the actual packing.

    To get a universal bound, we take the maximum over all possible r_max:
    UB = max_{r in [0, 0.5]} min(n*r, sqrt(n*(1+4r+pi*r^2)/(2*sqrt(3))))

    The first term (n*r) bounds sum when r_max = r.
    The second term comes from Oler + C-S.

    The minimum of the two: for small r, n*r < sqrt(...), so n*r controls.
    For large r, sqrt(...) < n*r, so sqrt controls.
    The crossover gives the tightest bound.

    Actually: for a given packing, both bounds hold simultaneously.
    sum r_i <= min(n*r_max, sqrt(n*(1+4*r_max+pi*r_max^2)/(2*sqrt(3))))

    The worst case for us is when this min is maximized.
    We compute: max over r in [0,0.5] of min(n*r, sqrt(...)).
    """
    s3_2 = 2 * np.sqrt(3)

    def bound_min(r):
        b1 = n * r  # trivial bound when r_max = r
        A_eff = 1 + 4*r + np.pi*r**2
        b2 = np.sqrt(n * A_eff / s3_2)  # Oler + C-S
        return min(b1, b2)

    # Find r that maximizes min(b1, b2)
    # At the intersection: n*r = sqrt(n*A_eff/s3_2)
    # n^2*r^2 = n*A_eff/s3_2
    # n*r^2 = A_eff/s3_2 = (1 + 4r + pi*r^2) / s3_2
    # s3_2 * n * r^2 = 1 + 4r + pi*r^2
    # (s3_2*n - pi)*r^2 - 4r - 1 = 0

    a_coeff = s3_2 * n - np.pi
    b_coeff = -4
    c_coeff = -1

    if a_coeff <= 0:
        r_cross = 0.5
    else:
        disc = b_coeff**2 - 4*a_coeff*c_coeff
        r_cross = (-b_coeff + np.sqrt(disc)) / (2*a_coeff)
        r_cross = min(r_cross, 0.5)

    best = bound_min(r_cross)

    # Also check boundary
    best = max(best, bound_min(0.5))

    if verbose:
        print(f"  Oler generalized (n={n}): crossover r={r_cross:.6f}, bound={best:.6f}")
        # Show the two curves
        for r_test in [0.01, 0.05, 0.1, r_cross, 0.2, 0.3, 0.5]:
            b1 = n * r_test
            A_eff = 1 + 4*r_test + np.pi*r_test**2
            b2 = np.sqrt(n * A_eff / s3_2)
            print(f"    r={r_test:.3f}: trivial={b1:.4f}, Oler+CS={b2:.4f}, min={min(b1,b2):.4f}")

    return best


def steiner_voronoi_bound(n, verbose=False):
    """
    Bound using the Steiner formula applied to Voronoi cells.

    For each circle i with Voronoi cell V_i (clipped to [0,1]^2):
    Let p_i = perimeter of V_i.
    Let a_i = area of V_i.

    By FT: a_i >= 2*sqrt(3)*r_i^2.
    Also: sum(a_i) = 1.

    By isoperimetric inequality for Voronoi cells:
    p_i >= perimeter of regular hexagon with area a_i = 6*sqrt(a_i/(6*tan(pi/6))) ...
    Actually this is just p >= 2*sqrt(pi*a) for convex shapes. Not useful directly.

    But: sum(p_i) >= perimeter(K) + 2 * sum of interior edge lengths.
    For cells sharing an edge, the edge appears twice in the sum.

    Actually, the total perimeter of all Voronoi cells = 2 * total_edge_length + boundary.
    Total boundary on walls = 4 (perimeter of square).

    So: sum(p_i) = 4 + 2 * (total internal edge length).

    This relates the Voronoi cell perimeters to 4 + 2L where L is internal edge length.
    Not directly useful for bounding sum(r_i).

    Let me try: using the relation between a Voronoi cell and its inscribed circle.

    For a circle of radius r in its Voronoi cell V:
    r is the inradius of V (largest inscribed circle).
    For a convex polygon with inradius r and perimeter p: area = r*p/2.
    (Since the polygon is a union of triangles from the center to each edge,
    each with height r and base = edge length.)

    Wait, this is only true if the circle is tangent to ALL edges.
    For a Voronoi cell, the inscribed circle (the actual circle) may not
    be tangent to all edges. It IS tangent to edges corresponding to
    touching neighbors, but not to all Voronoi edges.

    Actually, the inradius of V is AT LEAST r (the circle fits inside V).
    And for a convex polygon: area >= inradius * semi-perimeter.
    Wait: area = inradius * semi-perimeter ONLY if the inradius touches all edges.
    Otherwise: area >= inradius * semi-perimeter is not guaranteed.

    Actually for a convex body containing a circle of radius r:
    area >= r * semi-perimeter is FALSE in general.
    Consider a very elongated rectangle containing a small circle.

    Let me abandon this line and try the parametric Oler bound.
    """
    return oler_generalized_bound(n, verbose=verbose)


def main():
    known_best = {
        1: 0.5000, 2: 0.5858, 3: 0.7645, 4: 1.0000, 5: 1.0854,
        10: 1.5911, 15: 2.0365, 20: 2.3010, 26: 2.6360, 30: 2.8425, 32: 2.9390,
    }

    n_values = [1, 2, 3, 4, 5, 10, 15, 20, 26, 30, 32]

    print("Comparison of Upper Bounds")
    print("=" * 85)
    print(f"{'n':>3} | {'FT':>8} | {'Oler-gen':>8} | {'Best':>8} | {'Known':>8} | {'Gap':>8} | {'Gap%':>6}")
    print("-" * 85)

    results = {}
    for n_val in n_values:
        ft = fejes_toth_cs_bound(n_val)
        og = oler_generalized_bound(n_val, verbose=(n_val == 26))

        best = min(ft, og)
        known = known_best.get(n_val)
        gap = best - known if known else None
        gap_pct = 100 * gap / known if gap else None

        valid = best >= known - 1e-6 if known else True

        print(f"{n_val:3d} | {ft:8.4f} | {og:8.4f} | {best:8.4f} | "
              f"{known if known else 0:8.4f} | "
              f"{gap if gap else 0:8.4f} | {gap_pct if gap_pct else 0:5.1f}% "
              f"{'OK' if valid else 'INVALID!'}")

        results[n_val] = {'fejes_toth': ft, 'oler_gen': og, 'best': best}

    # Summary
    print(f"\nBest bound for n=26: {results[26]['best']:.6f}")
    print(f"Gap from known best: {results[26]['best'] - 2.636:.6f} ({100*(results[26]['best'] - 2.636)/2.636:.2f}%)")

    return results


if __name__ == "__main__":
    results = main()
