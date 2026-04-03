"""
Boundary effect bound for circle packing in a unit square.

THE KEY INSIGHT that can beat FT:
In hexagonal packing, each circle's Voronoi cell has area exactly 2*sqrt(3)*r^2.
But in a SQUARE, circles near the boundary have Voronoi cells that extend
OUTSIDE the square. The portion INSIDE the square is LESS than 2*sqrt(3)*r^2.

Therefore: the "wasted" boundary area means the effective packing is less efficient.

Formal argument:
Let V_i be the Voronoi cell of circle i (in the plane, not just in the square).
By FT: |V_i| >= 2*sqrt(3)*r_i^2 for all i.
The Voronoi partition of the plane covers the ENTIRE plane.
Within the square [0,1]^2: sum(|V_i intersect [0,1]^2|) = 1.
Outside the square: sum(|V_i intersect outside|) = total - 1.

Now: |V_i intersect [0,1]^2| <= |V_i| (obviously).
And: sum(|V_i|) >= sum(2*sqrt(3)*r_i^2).

But we need: sum(|V_i intersect [0,1]^2|) = 1.
And: |V_i| = |V_i intersect inside| + |V_i intersect outside|.

For a circle whose center is at distance d from the nearest boundary:
The Voronoi cell extends roughly distance d outward from the center.
Wait, the Voronoi cell extends to the midpoints between neighboring circles.

Actually, let me think about this differently.

APPROACH: Bounding the "EXTERIOR Voronoi area".
For circle i with center at (x_i, y_i) and radius r_i:
The Voronoi cell V_i contains all points closer to (x_i, y_i) than to any other center.
The part of V_i OUTSIDE [0,1]^2 is the "waste" for circle i.

For a circle near the boundary: its Voronoi cell extends outside.
The amount of exterior area depends on:
(a) Distance of center from boundary
(b) Distances to neighboring circles (which determine Voronoi cell size)

A LOWER BOUND on exterior waste:
If circle i is at distance d_i from the nearest boundary (d_i >= r_i),
and the Voronoi cell has "radius" at least r_i (since no other circle center
can be within 2*r_i of circle i's center), then:

The Voronoi cell V_i contains a disk of radius r_i centered at (x_i, y_i).
No wait, it doesn't: the Voronoi cell is the set of points closer to (x_i,y_i)
than to any other center. The nearest other center is at distance >= r_i + r_j >= 2*r_min.
So V_i contains the disk of radius (r_i + r_j)/2 >= r_i (if r_j >= r_i).

Actually, V_i contains {z : |z - c_i| <= |z - c_j| for all j} which includes
{z : |z - c_i| <= d_min/2} where d_min = min_j |c_i - c_j|.
And d_min >= r_i + r_min (non-overlap), so V_i contains disk of radius (r_i + r_min)/2.

For the exterior waste:
If center is at distance d from the nearest edge, the Voronoi cell extends
at least (r_i + r_min)/2 from the center. The portion outside the square:
For a center at distance d from the left edge (x_i = d + epsilon),
the cell extends left to at least x_i - (r_i + r_min)/2 = d - (r_i + r_min)/2 + epsilon.
If d < (r_i + r_min)/2, the cell extends outside: waste >= some function of d.

This is getting complicated. Let me try a simpler, more practical approach.

PRACTICAL APPROACH: Compute the boundary effect numerically.

For n circles with centers (x_i, y_i) and radii r_i:
Define the EFFECTIVE available area:
  A_eff = 1 + sum_i waste_i
where waste_i is the exterior Voronoi area for circle i.

Then FT gives: sum(2*sqrt(3)*r_i^2) <= A_eff.
Since A_eff > 1, this is WEAKER than basic FT.

WAIT: I'm going backwards. The correct argument is:

sum(|V_i|) >= sum(2*sqrt(3)*r_i^2)  [FT per circle]
|V_i| = |V_i cap square| + |V_i cap exterior|
sum(|V_i cap square|) = 1  [partition of square]
So: 1 + sum(|V_i cap exterior|) >= sum(2*sqrt(3)*r_i^2)

This gives: sum(2*sqrt(3)*r_i^2) <= 1 + EXTERIOR_WASTE.
Exterior waste is POSITIVE, so this is WEAKER than FT.

Hmm, so the boundary effect makes the bound WEAKER, not tighter?!

Wait, no. Let me reconsider. The FT bound sum(2*sqrt(3)*r_i^2) <= 1 comes from:
"The Voronoi cells partition the square, so sum of Voronoi areas = 1."
But actually, the Voronoi cells DON'T partition just the square -
they partition the ENTIRE PLANE.

The correct statement is:
If we assign each point in [0,1]^2 to its nearest circle center,
we get regions R_i (restricted Voronoi cells) with sum(|R_i|) = 1.
Each |R_i| >= |V_i cap square| (the restriction to the square).

Now, is |R_i| >= 2*sqrt(3)*r_i^2? NOT NECESSARILY!
Because R_i might be truncated by the square boundary.
The cell might extend outside, and the truncation reduces R_i.

So the correct argument is:
For each i: |R_i| >= 2*sqrt(3)*r_i^2 ONLY IF R_i contains the full hexagonal cell.
For boundary circles: R_i is truncated, |R_i| < 2*sqrt(3)*r_i^2 is possible!

Wait, that means FT is WRONG for finite domains?!

No, FT still holds because:
The Voronoi cells in the PLANE have |V_i| >= 2*sqrt(3)*r_i^2.
Within the square, sum(|V_i cap square|) = 1.
But |V_i cap square| can be < |V_i|, so:
1 = sum(|V_i cap square|) <= sum(|V_i|) is not useful.

The correct FT bound for a convex body K is:
sum(|V_i cap K|) = |K|
Each |V_i cap K| >= ??? (NOT necessarily >= 2*sqrt(3)*r_i^2)

ACTUALLY: the standard Fejes Toth bound for disks in a convex body K says:
n * delta_hex * r^2 <= |K| (for EQUAL radii r, density delta_hex = pi/(2*sqrt(3)))
This comes from: the packing density of disks in any convex body <= delta_hex.
So: n * pi * r^2 <= delta_hex * |K| => n * pi * r^2 / |K| <= delta_hex.
Rearranging: n * 2*sqrt(3) * r^2 <= |K| for the Voronoi version.

For MIXED radii, the standard result is:
sum(pi * r_i^2) <= delta_hex * |K| = pi/(2*sqrt(3)) * |K|
i.e., sum(r_i^2) <= |K| / (2*sqrt(3)).

For |K| = 1 (unit square): sum(r_i^2) <= 1/(2*sqrt(3)).
This IS the FT bound.

But WAIT: is this tight for a finite square? The hexagonal density delta_hex
is the ASYMPTOTIC density for infinite packings. For a FINITE square,
the density is strictly LESS than delta_hex due to boundary effects!

So: sum(r_i^2) < 1/(2*sqrt(3)) strictly. The FT bound is NOT tight!

HOWEVER: the bound sum(r_i^2) <= 1/(2*sqrt(3)) is still VALID.
It's just not TIGHT. To get a TIGHTER bound, we need to account for
the boundary effect.

GROEMER'S THEOREM (1960):
For n disks of radius r in a convex body K with area A and perimeter L:
n * pi * r^2 <= A + L * r + pi * r^2
This is WEAKER than FT for large n (more area budget due to boundary correction).

But for SUM-OF-RADII, we want: sum(r_i^2) <= f(r_max, n, ...)
where f < 1/(2*sqrt(3)). This would give a TIGHTER bound.

APPROACH: Boundary density reduction.
The packing density in a square of side s is at most:
delta(n) = n*pi*r^2/s^2 <= delta_hex - c/sqrt(n)
for some constant c > 0, due to boundary effects.

If we can quantify this: sum(r_i^2) <= 1/(2*sqrt(3)) - epsilon(n)
then sum(r_i) <= sqrt(n * (1/(2*sqrt(3)) - epsilon(n))) by CS.

For n=26: if epsilon = 0.005, then:
sum(r_i^2) <= 0.2887 - 0.005 = 0.2837
sum(r_i) <= sqrt(26 * 0.2837) = sqrt(7.376) = 2.716
vs FT = 2.740. Improvement of 0.024!

But what is epsilon? For the EQUAL-RADIUS case:
n circles of radius r: FT gives n*r^2 <= 1/(2*sqrt(3)).
With boundary correction (Oler): n*r^2 <= (1+4r+pi*r^2)/(2*sqrt(3)).
This is WEAKER (larger RHS). Not helpful for upper bounds!

The issue: Oler's bound says we can fit MORE circles than FT predicts
(due to boundary waste being "reclaimed" by extending outside).
For UPPER BOUNDING, this makes the bound WEAKER.

For upper bounding, we want to say: "circles waste MORE space near boundaries,
so the effective packing density is LOWER."

This would mean: sum(r_i^2) <= 1/(2*sqrt(3)) - delta_waste.
But the standard theory (Oler, Groemer) goes the OTHER direction!

I think the resolution is: for ARBITRARY circle POSITIONS, the FT bound is tight.
The boundary effect only helps if we KNOW the circles are well-placed.
For our MAX sum problem, the solver will place circles optimally,
and the FT bound correctly accounts for the maximum.

So FT is the correct bound and cannot be improved by boundary arguments.
The only way to beat FT is to use PROBLEM-SPECIFIC constraints
(like the pair bound, top-4 bound, etc.) that don't reduce to FT.

Let me try yet another approach: bounding based on the PACKING GRAPH.

PACKING GRAPH APPROACH:
In the optimal packing, many circles are in CONTACT (tangent to each other
or to the boundary). The contact graph constrains the packing.

For a contact graph with m edges (contacts):
Each edge (i,j) means dist(c_i, c_j) = r_i + r_j.
For a PLANAR packing, the contact graph is planar: m <= 3n - 6.

For each circle, degree in contact graph <= 6 (kissing number in 2D is 6).
But boundary circles have degree <= 3 typically.

This limits the "support structure" of the packing.

For sum-of-radii, the contact graph structure constrains which radii are possible.
But this is hard to turn into a computable bound.

APPROACH: LP on radius distribution.
Instead of tracking individual radii, discretize the possible radius values
and count how many circles of each size fit.

Radius values: r_1 > r_2 > ... > r_m (m classes)
n_k = number of circles of radius class k.
sum(n_k) = n
sum(n_k * r_k) = objective (maximize)
FT: sum(n_k * 2*sqrt(3) * r_k^2) <= 1
Kissing: each circle of radius r_k can touch at most 6 others.
Packing in [0,1]^2: specific constraints per radius class.

For radius class k: max number of non-overlapping circles of radius r_k in [0,1]^2:
n_k <= (1+4*r_k+pi*r_k^2) / (2*sqrt(3)*r_k^2) [Oler for equal radii]

This is just the Oler bound per class. Combined with FT, not clear if it helps.

APPROACH: CONTINUOUS LP (infinite-dimensional).
Maximize integral r * dn(r) subject to:
integral 2*sqrt(3)*r^2 * dn(r) <= 1  [FT]
integral dn(r) <= n  [count]
dn(r) <= n_max(r)  [max circles of radius r]
dn(r) >= 0

For continuous n(r): density of circles with radius in [r, r+dr].
By calculus of variations: optimal has all circles of equal radius r* = 1/sqrt(2*sqrt(3)*n).
Sum = n*r* = sqrt(n/(2*sqrt(3))). This IS FT.

The count constraint integral dn(r) <= n is exactly tight at equal radius.
Additional constraints needed: e.g., large-radius circles preclude nearby circles.

I'll implement the numerical approach: for a GIVEN set of known upper bounds
on top-k sums, compute the best bound on the full sum.
"""

import numpy as np
import cvxpy as cp
import json
import sys
from pathlib import Path


def compute_top_k_bounds(max_k=32):
    """
    Compute the best known upper bounds on the sum of top-k radii.

    Uses:
    - k=1: r <= 0.5 (containment)
    - k=2: sum <= 2-sqrt(2) (diagonal pair)
    - k=4: sum <= 1.0 (corner placement)
    - Otherwise: FT bound sqrt(k/(2*sqrt(3)))

    We also try to IMPROVE bounds for specific k values.
    """
    s = 2 * np.sqrt(3)
    bounds = {}

    for k in range(1, max_k + 1):
        ft = np.sqrt(k / s)
        # Geometric bounds
        geo = float('inf')
        if k == 1:
            geo = 0.5
        elif k == 2:
            geo = 2 - np.sqrt(2)  # 0.5858
        elif k == 3:
            # From pair bound: 3 pairs, each sum <= 2-sqrt(2)
            # So 2*(r1+r2+r3) <= 3*(2-sqrt(2)). sum <= 1.5*(2-sqrt(2)) = 0.8787.
            # Can we do better?
            # Try: r1+r2 <= 2-sqrt(2), r1 <= 0.5, and FT: s*(r1^2+r2^2+r3^2) <= 1.
            # Optimize r1+r2+r3 subject to these.
            geo = _compute_k_bound(3)
        elif k == 4:
            geo = 1.0
        elif k == 5:
            geo = _compute_k_bound(5)
        elif k <= 10:
            geo = _compute_k_bound(k)

        bounds[k] = min(ft, geo)

    return bounds


def _compute_k_bound(k):
    """Compute tight upper bound on sum of top-k radii via SOCP."""
    s = 2 * np.sqrt(3)
    r = cp.Variable(k)

    constraints = [r >= 0, r <= 0.5]
    for i in range(k-1):
        constraints += [r[i] >= r[i+1]]

    # FT area
    constraints += [s * cp.sum_squares(r) <= 1]

    # Pair bound for all top pairs
    pair_limit = 2 - np.sqrt(2)
    for i in range(k):
        for j in range(i+1, k):
            constraints += [r[i] + r[j] <= pair_limit]

    # Top-4 bound if applicable
    if k >= 4:
        constraints += [r[0] + r[1] + r[2] + r[3] <= 1.0]

    # Individual FT bounds
    for i in range(k):
        max_ri = 1.0 / np.sqrt(s * (i+1))
        if max_ri < 0.5:
            constraints += [r[i] <= max_ri]

    objective = cp.Maximize(cp.sum(r))
    prob = cp.Problem(objective, constraints)

    try:
        result = prob.solve(solver=cp.SCS, verbose=False, max_iters=50000, eps=1e-9)
        return result
    except:
        return np.sqrt(k / s)


def optimal_split_bound(n, top_k_bounds, verbose=False):
    """
    Compute bound by splitting: sum = (top-k sum) + (bottom n-k sum).

    For each k in 1..n-1:
    - Top-k sum <= top_k_bounds[k]
    - Bottom n-k radii: each <= r_k, and FT on remaining area.
    - r_k <= 1/sqrt(s*(k))  [from FT: k circles >= r_k implies k*s*r_k^2 <= 1]
    Wait: k-th largest satisfies k*s*r_k^2 <= 1 (since there are k circles >= r_k).
    But the bottom n-k circles use area too.
    Actually: s * sum(r_i^2) <= 1 for ALL n circles.
    Top-k: s * sum(r_i^2, i<=k) = s * Q_top
    Bottom n-k: s * sum(r_i^2, i>k) = s * Q_bot
    s * (Q_top + Q_bot) <= 1.

    From the top-k bound: sum(r_i, i<=k) <= B_k = top_k_bounds[k].
    From CS: sum(r_i, i<=k)^2 <= k * Q_top => Q_top >= B_k^2 / k.
    Hmm, we want to MINIMIZE Q_top (to leave more room for Q_bot).
    Actually we want to maximize TOTAL sum, so:

    sum = top_sum + bot_sum
    For fixed top_sum = S (where S <= B_k):
    Q_top >= S^2 / k (by CS)
    Q_bot <= (1 - s*Q_top) / s <= (1 - s*S^2/k) / s = 1/s - S^2/k
    bot_sum <= sqrt((n-k) * Q_bot) = sqrt((n-k) * (1/s - S^2/k))

    Maximize: S + sqrt((n-k) * (1/s - S^2/k)) over S in [0, B_k].

    d/dS [S + sqrt((n-k)*(1/s - S^2/k))] = 1 + (n-k)*(-2S/k) / (2*sqrt((n-k)*(1/s-S^2/k)))
    = 1 - sqrt(n-k)*S / (k * sqrt(1/s - S^2/k))

    Set to 0: sqrt(n-k)*S / (k * sqrt(1/s - S^2/k)) = 1
    (n-k)*S^2 / (k^2 * (1/s - S^2/k)) = 1
    (n-k)*S^2 = k^2/s - k*S^2
    S^2 * (n-k+k) = k^2/s
    S^2 * n = k^2/s
    S = k / sqrt(n*s)

    Is this <= B_k?
    S = k/sqrt(n*s) vs B_k.

    If B_k = sqrt(k/s) (= FT for k): S = k/sqrt(n*s) = sqrt(k/s) * sqrt(k/n).
    For k < n: sqrt(k/n) < 1, so S < B_k. The minimum is in the interior.

    Total at interior optimum:
    k/sqrt(n*s) + sqrt((n-k)*(1/s - k/(n*s)))
    = k/sqrt(n*s) + sqrt((n-k)*(n-k)/(n*s))
    = k/sqrt(n*s) + (n-k)/sqrt(n*s)
    = n/sqrt(n*s)
    = sqrt(n/s) = FT bound.

    So the interior optimum gives exactly FT. The bound ONLY improves if
    B_k < k/sqrt(n*s), i.e., the top-k bound is BELOW the CS-optimal allocation.

    In that case, S is constrained to B_k, and:
    total = B_k + sqrt((n-k)*(1/s - B_k^2/k))

    This is potentially < FT if B_k < k/sqrt(n*s).

    For n=26, k=2: k/sqrt(n*s) = 2/sqrt(26*3.464) = 2/sqrt(90.06) = 2/9.49 = 0.211.
    B_2 = 0.5858. Since 0.5858 > 0.211, the constraint is NOT binding.

    For n=26, k=4: k/sqrt(n*s) = 4/9.49 = 0.421.
    B_4 = 1.0. Since 1.0 > 0.421, NOT binding.

    So for n=26, NONE of our top-k bounds are binding!
    The "optimal" allocation already satisfies all constraints.

    To get improvement: need B_k < k/sqrt(n*s) for some k.
    k/sqrt(n*s) = k/sqrt(26*3.464) = k/9.49.
    Need B_k < k/9.49.
    B_k for equal radii: B_k = k * r = k / sqrt(26*3.464/26) = k/sqrt(3.464) = k/1.861.
    Wait, for equal radii: each r = sqrt(1/(n*s)) = 1/sqrt(26*3.464) = 1/9.49 = 0.1054.
    B_k for top-k = k * 0.1054.
    Our bound B_k must be < k * 0.1054 to be binding.

    k=1: B_1 = 0.5 >> 0.1054. Not binding.
    k=2: B_2 = 0.5858 >> 0.2108. Not binding.
    ...
    k=4: B_4 = 1.0 >> 0.4216. Not binding.

    For n=26, the equal-radius allocation has r ≈ 0.105 per circle.
    Our geometric bounds only constrain radii > 0.25 or so.
    With 26 circles, individual radii are much smaller than 0.25,
    so geometric bounds are completely slack.

    CONCLUSION: For n >= 10 or so, our geometric constraints don't help.
    FT is the dominant bound. To improve, we'd need geometric constraints
    that apply to SMALL radii (r ~ 0.1), which is very challenging.
    """
    s = 2 * np.sqrt(3)

    best = np.sqrt(n / s)  # FT bound
    best_k = -1
    best_binding = False

    for k in range(1, n):
        if k not in top_k_bounds:
            continue
        B_k = top_k_bounds[k]
        S_opt = k / np.sqrt(n * s)  # unconstrained optimum

        if B_k < S_opt:
            # Constraint is binding!
            Q_top_min = B_k**2 / k
            Q_bot_max = 1/s - Q_top_min
            if Q_bot_max < 0:
                continue
            bot_max = np.sqrt((n-k) * Q_bot_max)
            total = B_k + bot_max
            if total < best:
                best = total
                best_k = k
                best_binding = True
        # If not binding, total = FT regardless of k.

    if verbose:
        if best_binding:
            print(f"  Split bound (n={n}): {best:.6f} at k={best_k} (binding!)")
        else:
            print(f"  Split bound (n={n}): {best:.6f} (FT, no constraint binding)")

    return best


def main():
    known_best = {
        1: 0.5000, 2: 0.5858, 3: 0.7645, 4: 1.0000, 5: 1.0854,
        6: 1.1670, 7: 1.2885, 8: 1.3775, 9: 1.4809, 10: 1.5911,
        15: 2.0365, 20: 2.3010, 26: 2.6360, 30: 2.8425, 32: 2.9390,
    }

    if len(sys.argv) > 1:
        n_values = [int(x) for x in sys.argv[1:]]
    else:
        n_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 26, 30, 32]

    print("Computing top-k bounds...")
    top_k_bounds = compute_top_k_bounds(max_k=max(n_values))

    print("\nTop-k upper bounds:")
    for k in sorted(top_k_bounds.keys()):
        ft_k = np.sqrt(k / (2*np.sqrt(3)))
        print(f"  k={k:2d}: UB={top_k_bounds[k]:.6f} (FT={ft_k:.6f}, ratio={top_k_bounds[k]/ft_k:.4f})")

    print("\n" + "=" * 90)
    print(f"{'n':>3} | {'FT':>8} | {'Split':>8} | {'BEST':>8} | {'Known':>8} | {'Gap':>8} | {'Gap%':>6}")
    print("-" * 90)

    all_results = {}
    for n_val in n_values:
        ft = np.sqrt(n_val / (2*np.sqrt(3)))
        split = optimal_split_bound(n_val, top_k_bounds, verbose=(n_val in [3, 10, 26]))

        best = min(ft, split)
        known = known_best.get(n_val, 0)
        gap = best - known if known else 0
        gap_pct = 100*gap/known if known else 0
        valid = best >= known - 1e-3 if known else True

        print(f"{n_val:3d} | {ft:8.4f} | {split:8.4f} | {best:8.4f} | "
              f"{known:8.4f} | {gap:8.4f} | {gap_pct:5.1f}% "
              f"{'OK' if valid else '**FAIL**'}")

        all_results[str(n_val)] = {
            'ft': float(ft), 'split': float(split),
            'best': float(best), 'known': float(known),
        }

    output_path = Path(__file__).parent / "boundary_effect_bounds.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
