#!/usr/bin/env python3
"""
Verify whether the L-function bound holds for clipped Voronoi cells.

The Dowker-Fejes Toth theorem states:
  For a convex k-gon P that CONTAINS a circle of radius r:
  Area(P) >= k * tan(pi/k) * r^2

This applies to any convex polygon containing a disk, regardless of
whether the disk is inscribed (tangent to all sides) or just contained.

We verify this for each circle in the best packing by:
1. Computing the clipped Voronoi cell
2. Checking if the cell is convex
3. Counting the actual number of edges
4. Checking if the circle is fully contained in the cell
5. Computing the L-function bound and comparing to actual area
"""

import numpy as np
from scipy.spatial import Voronoi, Delaunay
from shapely.geometry import Polygon as ShapelyPolygon, box as shapely_box, Point
from shapely.validation import explain_validity
import sys


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


def voronoi_cells_in_square(centers):
    """Compute Voronoi cells clipped to unit square using reflection trick."""
    n = len(centers)
    unit_square = shapely_box(0, 0, 1, 1)

    reflections = [
        centers,
        np.column_stack((-centers[:, 0], centers[:, 1])),
        np.column_stack((2 - centers[:, 0], centers[:, 1])),
        np.column_stack((centers[:, 0], -centers[:, 1])),
        np.column_stack((centers[:, 0], 2 - centers[:, 1])),
        np.column_stack((-centers[:, 0], -centers[:, 1])),
        np.column_stack((2 - centers[:, 0], -centers[:, 1])),
        np.column_stack((-centers[:, 0], 2 - centers[:, 1])),
        np.column_stack((2 - centers[:, 0], 2 - centers[:, 1])),
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


def count_edges(poly):
    """Count edges of a polygon, being careful about collinear vertices."""
    if poly is None or poly.is_empty or poly.geom_type != 'Polygon':
        return 0

    coords = list(poly.exterior.coords)
    n = len(coords) - 1  # last point = first point

    if n < 3:
        return n

    # Remove collinear points to get the TRUE number of edges
    true_vertices = []
    for i in range(n):
        p_prev = np.array(coords[(i - 1) % n])
        p_curr = np.array(coords[i])
        p_next = np.array(coords[(i + 1) % n])

        # Check if p_curr is collinear with p_prev and p_next
        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        cross = abs(v1[0] * v2[1] - v1[1] * v2[0])

        if cross > 1e-10:  # Not collinear
            true_vertices.append(i)

    return len(true_vertices)


def L_function(k):
    """Fejes-Toth L-function: k * tan(pi/k)."""
    if k < 3:
        return float('inf')
    return k * np.tan(np.pi / k)


def main():
    circles = np.array(SOLUTION_CIRCLES)
    centers = circles[:, :2]
    radii = circles[:, 2]
    n = len(radii)

    print("Verifying L-function bound for clipped Voronoi cells")
    print("=" * 70)

    cells = voronoi_cells_in_square(centers)

    print(f"\n{'i':>3s} {'r':>8s} {'Area':>8s} {'k_raw':>5s} {'k_true':>6s} "
          f"{'L(k)*r2':>8s} {'ratio':>7s} {'convex':>6s} {'contains':>8s} {'type':>8s}")
    print("-" * 80)

    violations = 0
    for i in range(n):
        cell = cells[i]
        r = radii[i]
        cx, cy = centers[i]

        if cell is None or cell.is_empty:
            print(f"{i:3d}  -- empty cell --")
            continue

        area = cell.area

        # Raw edge count (from coords)
        coords = list(cell.exterior.coords)
        k_raw = len(coords) - 1

        # True edge count (removing collinear vertices)
        k_true = count_edges(cell)

        # L-function bound
        L_bound = L_function(k_true) * r**2

        # Check convexity
        is_convex = cell.convex_hull.area - cell.area < 1e-12

        # Check if circle is contained in cell
        circle_approx = Point(cx, cy).buffer(r, resolution=128)
        is_contained = cell.contains(circle_approx)

        # Also check by distance: min distance from center to each edge >= r
        dist_to_boundary = cell.exterior.distance(Point(cx, cy))
        # Actually, for containment, we need the distance from the center
        # to the boundary to be >= r. But Point.distance gives 0 if inside.
        # Let's check each edge.
        boundary_coords = list(cell.exterior.coords)
        min_edge_dist = float('inf')
        for j in range(len(boundary_coords) - 1):
            p1 = np.array(boundary_coords[j])
            p2 = np.array(boundary_coords[j + 1])
            # Distance from (cx, cy) to line segment p1-p2
            center = np.array([cx, cy])
            edge = p2 - p1
            edge_len = np.linalg.norm(edge)
            if edge_len < 1e-15:
                continue
            edge_unit = edge / edge_len
            t = np.dot(center - p1, edge_unit)
            t = max(0, min(edge_len, t))
            closest = p1 + t * edge_unit
            dist = np.linalg.norm(center - closest)
            min_edge_dist = min(min_edge_dist, dist)

        # Classify
        dists = [cx - r, 1 - cx - r, cy - r, 1 - cy - r]
        walls = sum(1 for d in dists if abs(d) < 1e-6)
        ctype = 'corner' if walls >= 2 else ('wall' if walls == 1 else 'interior')

        ratio = area / L_bound if L_bound > 0 else float('inf')
        violated = ratio < 1 - 1e-10

        marker = " *** VIOLATION" if violated else ""
        if violated:
            violations += 1

        print(f"{i:3d} {r:8.5f} {area:8.6f} {k_raw:5d} {k_true:6d} "
              f"{L_bound:8.6f} {ratio:7.4f} {'Y' if is_convex else 'N':>6s} "
              f"{'Y' if is_contained else 'N':>8s} {ctype:>8s}"
              f"{marker}")

        if violated:
            print(f"      min_edge_dist = {min_edge_dist:.8f}, r = {r:.8f}, "
                  f"dist/r = {min_edge_dist/r:.6f}")
            print(f"      L({k_true}) = {L_function(k_true):.6f}")

    print(f"\nTotal violations: {violations}/{n}")

    # The Dowker theorem: Among k-gons of area A containing a circle of radius r,
    # the minimum area is k*tan(pi/k)*r^2, achieved by the REGULAR k-gon.
    # THIS REQUIRES THE CIRCLE TO BE THE INCIRCLE (tangent to all sides).
    # For a circle merely CONTAINED (not inscribed), the bound is weaker.
    #
    # For a circle CONTAINED in a convex k-gon:
    # Area >= pi * r^2 (trivially, since polygon contains the circle).
    # A tighter bound: the polygon must contain the circle, so its
    # "inradius" (radius of largest inscribed circle) >= r.
    # The minimum area k-gon with inradius >= r is the regular k-gon with
    # inradius = r, which has area = k * tan(pi/k) * r^2.
    #
    # Wait -- that IS the L-function! So if the INRADIUS of the polygon >= r
    # (which is true because the circle of radius r is contained and the
    # inradius is the max inscribed circle radius), then:
    # Area >= k * tan(pi/k) * r^2.
    #
    # But the inradius of a polygon is the MAXIMUM inscribed circle radius,
    # which could be LARGER than the specific circle's radius.
    # If our circle of radius r is contained, is the inradius >= r?
    # YES: if a circle of radius r fits inside the polygon, the inradius >= r.
    #
    # So the L-function bound SHOULD hold for all cells.
    # The violations suggest either:
    # (a) The clipped Voronoi cell is NOT convex (clipping can create non-convexity)
    # (b) The circle is NOT fully contained in the clipped cell
    # (c) Edge counting is wrong

    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    print(f"\nIf the Voronoi cell is convex and contains the circle,")
    print(f"then Area >= L(k)*r^2 MUST hold (Dowker's theorem).")
    print(f"Violations indicate either non-convexity or non-containment.")

    # Check violation details
    for i in range(n):
        cell = cells[i]
        r = radii[i]
        k_true = count_edges(cell)
        area = cell.area
        L_bound = L_function(k_true) * r**2

        if area < L_bound - 1e-10:
            print(f"\nCircle {i} (r={r:.6f}):")
            print(f"  Voronoi cell area: {area:.8f}")
            print(f"  L({k_true})*r^2:       {L_bound:.8f}")
            print(f"  Deficit:           {L_bound - area:.8f}")
            print(f"  Is convex:         {cell.convex_hull.area - cell.area < 1e-12}")
            print(f"  Convex hull area:  {cell.convex_hull.area:.8f}")

            # Check if the cell actually contains the circle
            cx, cy = centers[i]
            # Sample points on the circle boundary
            theta = np.linspace(0, 2*np.pi, 360)
            circle_pts = np.column_stack([cx + r*np.cos(theta), cy + r*np.sin(theta)])
            outside = 0
            for pt in circle_pts:
                if not cell.contains(Point(pt)):
                    outside += 1
            print(f"  Circle boundary points outside cell: {outside}/360")

            if outside > 0:
                print(f"  => Circle NOT fully contained in cell!")
                print(f"     This means the Voronoi cell is too small.")
                print(f"     Likely cause: another circle's center is closer")
                print(f"     to parts of this circle's boundary.")


if __name__ == "__main__":
    main()
