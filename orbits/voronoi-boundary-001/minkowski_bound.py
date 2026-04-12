#!/usr/bin/env python3
"""
Upper bound via Minkowski sum argument for non-congruent circle packing.

=== The key idea ===

For n circles of radii r_1,...,r_n packed in [0,1]^2, consider the
"inner parallel body" approach:

Each circle of radius r_i has its center in the reduced square
S_i = [r_i, 1-r_i]^2, which has area (1 - 2*r_i)^2.

The centers c_1,...,c_n are DISTINCT points with pairwise distances
d(c_i, c_j) >= r_i + r_j.

We want: maximize S = sum(r_i)
subject to: centers in their respective S_i, pairwise constraints.

This is equivalent to a GENERALIZED DISK PACKING problem: pack disks of
radii (r_i + r_j)/2 centered at the midpoints, in appropriate domains.

=== Approach 1: Jensen's inequality on the containment area ===

The effective containment area for a circle of radius r is (1-2r)^2.
If r is small: (1-2r)^2 ~ 1 - 4r + 4r^2.
The "wasted perimeter area" is ~ 4r (to first order in r).

For all circles: sum of wasted areas ~ 4*sum(r_i) = 4*S.
Total effective area ~ n*1 - 4*S (crude; not quite right).

More precisely: the Minkowski inner body argument.

=== Approach 2: Groemer's containment inequality ===

THEOREM (Hadwiger, 1957; Groemer, 1960):
For n non-overlapping convex bodies K_1,...,K_n contained in a convex body K:

sum(Area(K_i)) <= Area(K) - sum_{i<j} A(K_i, K_j) * I(i,j)

where A(K_i, K_j) is a mixed area term and I(i,j) is a contact indicator.
This is NOT directly useful.

=== Approach 3: Inner parallel body ===

For a convex body K and radius r, the inner parallel body K_{-r} is the
set of points at distance >= r from the boundary of K:
K_{-r} = {x in K : dist(x, boundary(K)) >= r}.

For K = [0,1]^2: K_{-r} = [r, 1-r]^2, with Area(K_{-r}) = (1-2r)^2.

For a circle of radius r in K: center in K_{-r}.

Now: n circles with pairwise distance >= r_i + r_j have centers forming
a "hard-disk" system in their respective K_{-r_i}.

The FT bound applied to this system:
Consider the LARGEST inner body K_{-r_min} = [r_min, 1-r_min]^2.
ALL centers are in this body (since r_i >= r_min for all i implies
K_{-r_i} subset K_{-r_min}).

Hmm, that's backwards. K_{-r} SHRINKS as r grows. So bigger circles
have smaller containment regions.

Let me use a different decomposition.

=== Approach 4: Explicit two-group bound ===

Partition circles into "large" (r >= r*) and "small" (r < r*) for a
threshold r*.

Large circles: each has center in K_{-r*} = [r*, 1-r*]^2.
Their pairwise distance >= 2*r* (since both have r >= r*).
So large circle centers form a packing of disks of radius r* in K_{-r*}.

FT for large circles: sum(2*sqrt(3)*r*^2 * n_large) <= (1-2*r*)^2.
But this uses r* instead of actual r_i, which is wasteful.

Better: FT for large circles directly:
sum(2*sqrt(3)*r_i^2) <= Area(effective containment for large circles)
                     <= 1 (unit square area)

This is just the original FT bound again.

=== Approach 5: Dual LP with pairwise constraints ===

This IS what the parent orbit (upperbound-001) did with the LP relaxation.
The LP uses McCormick envelopes for x_i*x_j products and is loose.

=== Approach 6: Moment-based bound (functional) ===

For the circle packing problem, we can write a dual INFINITE-DIMENSIONAL
LP (Lasserre-type):

maximize S = sum(r_i)
subject to: for all (x_i, y_i, r_i) in [0,1]^2 x [0,0.5]:
            non-overlap and containment constraints.

The dual of this LP gives an upper bound via "certificates".
The FT bound IS such a certificate: the function f(x,y,r) = r has
integral sum(r_i), and the constraint certificate is the area function
g(x,y,r) = 2*sqrt(3)*r^2 with integral <= 1.

To beat FT, we need a better certificate function. Candidates:
g(x,y,r) = 2*sqrt(3)*r^2 + h(x,y)*r for some function h that
captures boundary effects.

For h(x,y) = epsilon * (distance to boundary)^{-1}: this penalizes
circles near the boundary. But we need g to be a VALID lower bound
on the Voronoi cell area, which is hard to prove for general h.

=== Approach 7: The Boroczky-Szabo approach ===

Boroczky and Szabo (2007) proved tighter bounds for finite packings
using the concept of "k-neighbor packings". Their result:

For n congruent circles in a convex body K of area A and perimeter P:
A >= 2*sqrt(3)*n*r^2 + (2 - sqrt(3))*P*r + more corrections.

For K = [0,1]^2: A=1, P=4.
1 >= 2*sqrt(3)*n*r^2 + (2-sqrt(3))*4*r

For NON-CONGRUENT circles, we can try to adapt this using a weighted
version:

1 >= 2*sqrt(3)*sum(r_i^2) + (2-sqrt(3)) * sum_{boundary}(2*r_i)

where the boundary sum is over circles touching the boundary.

But as noted in the parent orbit, the boundary correction term is
POSITIVE, which means it tightens the bound. However, for the
non-congruent case, the adversary can make boundary circles tiny,
making this correction negligible.

UNLESS: we can show that in any near-optimal packing, the boundary
circles must have non-trivial radii. This requires a structural
argument about how circles fill the square.

=== Approach 8: Perimeter penalty bound (new) ===

Consider the "expanded" problem: each circle of radius r "uses up"
an effective area of alpha*r^2 + beta*r, where beta > 0 is a
perimeter penalty.

If we can show: for ANY packing in [0,1]^2:
  sum(alpha*r_i^2 + beta*r_i) <= Area + beta*perimeter/something

then: alpha*sum(r_i^2) + beta*sum(r_i) <= C
=> sum(r_i) <= (C - alpha*sum(r_i^2)) / beta
=> sum(r_i) <= C/beta (loose) or combined with FT: ...

The issue is finding valid alpha, beta, C.

THEOREM (Oler, 1961): For n non-overlapping unit disks whose centers
have convex hull H:
Area(H) >= sqrt(3)*(n-1) + (rest of Euler/face counting)

For the FINITE packing in a square:
Area([0,1]^2) >= Area(hull of centers + r_max) >= Area(hull) + perim*r_max + pi*r_max^2

where hull of centers has area A_hull, perimeter P_hull.
Steiner formula: Area(hull + B(r)) = A_hull + P_hull*r + pi*r^2.

For the circles to fit: hull + B(r_max) subset [0,1]^2.
So: A_hull + P_hull*r_max + pi*r_max^2 <= 1.

And from FT applied to the hull: A_hull >= 2*sqrt(3)*sum(r_i^2) - corrections.

This is getting complicated. Let me just compute what I CAN prove rigorously
and write it up.
"""

import numpy as np
from scipy.optimize import minimize_scalar


def fejes_toth_bound(n):
    return np.sqrt(n / (2 * np.sqrt(3)))


def boroczky_szabo_congruent(n):
    """
    Boroczky-Szabo bound for n congruent circles in unit square.

    1 >= 2*sqrt(3)*n*r^2 + 4*(2-sqrt(3))*r

    Solve for max S = n*r:
    Let a = 2*sqrt(3), b = 4*(2-sqrt(3)).
    a*(S/n)^2*n + b*(S/n) <= 1
    a*S^2/n + b*S/n <= 1
    a*S^2 + b*S <= n
    S = (-b + sqrt(b^2 + 4*a*n)) / (2*a)
    """
    a = 2 * np.sqrt(3)
    b = 4 * (2 - np.sqrt(3))
    S = (-b + np.sqrt(b**2 + 4 * a * n)) / (2 * a)
    return S


def boroczky_noncongruent_attempt(n):
    """
    Attempt to extend Boroczky-Szabo to non-congruent circles.

    For non-congruent circles with radii r_1,...,r_n:
    1 >= 2*sqrt(3)*sum(r_i^2) + (2-sqrt(3)) * sum_{boundary}(2*r_j)

    The boundary sum counts the perimeter covered by boundary circles.
    Since we don't know which are boundary circles, the adversary
    chooses to minimize the boundary sum (to maximize radii).

    Adversary strategy: put no circles on the boundary.
    Then boundary correction = 0 and we get FT.

    BUT: if circles DON'T touch the boundary, each circle has
    center at distance > r_i from each wall. The "exposed" boundary
    (wall area not adjacent to any circle) is:
    Perimeter - sum_{boundary}(2*r_j) >= 4 - 0 = 4.

    This exposed boundary means there's a strip along each wall
    that's "wasted" -- it's not efficiently packed.

    Can we turn this into a bound? Yes, if we can show that the
    wasted strip area contributes to a tighter FT constraint.

    The wasted strip: for each wall, if no circle touches it,
    the strip [0,1] x [0, r_min] has area r_min that's not
    efficiently used. But the adversary can make r_min tiny.

    For all walls: total strip area = 4*r_min (minus corners).
    If r_min -> 0, waste -> 0.

    ALTERNATIVE: instead of strips, count the "exposed wall area".
    For each wall segment not adjacent to a circle, there's a
    rectangular region of width r_nearest * something.

    This is too vague. Let me try a clean optimization formulation.
    """
    # For each allocation of circles to types:
    # n_c corner circles, n_w wall circles, n_i interior circles
    # Corner: r_c + gap_c to corner is wasted
    # Wall: gap to wall is wasted
    # Interior: contained in [r, 1-r]^2

    # Without any structural constraint, FT is the best we can do
    return fejes_toth_bound(n)


def parametric_bound(n):
    """
    A VALID parametric bound: sweep over parameter r_min (minimum radius)
    and combine FT with containment.

    For any packing with all r_i >= r_min:
    - All centers in [r_min, 1-r_min]^2 (area = (1-2*r_min)^2)
    - FT applied to the inner square:
      sum(2*sqrt(3)*r_i^2) <= (1-2*r_min)^2

    But we don't know r_min a priori. We can ENUMERATE:
    For each possible r_min in [0, 0.5]:
    Case 1: all r_i >= r_min. Then sum(2*sqrt(3)*r_i^2) <= (1-2*r_min)^2.
    By C-S: (sum r_i)^2 <= n*sum(r_i^2) <= n*(1-2*r_min)^2 / (2*sqrt(3))
    => sum(r_i) <= (1-2*r_min)*sqrt(n/(2*sqrt(3)))

    Case 2: some r_i < r_min. Then those circles contribute little.

    For Case 1: UB(r_min) = (1-2*r_min)*sqrt(n/(2*sqrt(3)))
    This is DECREASING in r_min. Best at r_min = 0: FT bound.

    Wait, this is wrong. The containment area should be the FULL
    unit square, not the inner square. The circles are in [0,1]^2,
    and the FT bound uses the total area = 1 regardless of where
    the centers are.

    The issue: FT says the packing density in ANY convex body is at
    most pi/(2*sqrt(3)). The "body" is [0,1]^2 with area 1. The
    circles are contained in [0,1]^2. So sum(pi*r_i^2) <= pi/(2*sqrt(3)) * 1,
    giving sum(r_i^2) <= 1/(2*sqrt(3)). This doesn't depend on where
    the circles are -- it's a global bound on the total packed area.

    The centers being in [r_i, 1-r_i]^2 is already accounted for
    by the containment constraint. It doesn't reduce the available
    area for packing.

    So the FT bound really is: sum(r_i^2) <= 1/(2*sqrt(3)),
    and (sum r_i)^2 <= n/(2*sqrt(3)).

    I cannot improve on this without additional constraints that
    bind for all packings.
    """
    alpha = 2 * np.sqrt(3)

    # The FT bound IS the unconstrained optimum
    ft = np.sqrt(n / alpha)

    # BUT: there's one more constraint we haven't fully exploited.
    # For each circle: r_i <= 0.5.
    # And more importantly: for each PAIR:
    #   dist(c_i, c_j) >= r_i + r_j.
    # Since dist <= sqrt(2) (diagonal of unit square):
    #   r_i + r_j <= sqrt(2) for all i,j.

    # For the FT-optimal equal radii: r = sqrt(1/(n*alpha)) ~ 0.1054.
    # r + r = 0.2108 << sqrt(2) = 1.414.
    # So the pair constraint doesn't bind.

    # For n=2: r_1 + r_2 <= sqrt(2), and FT gives sum <= sqrt(2/alpha) = 0.760.
    # But pair bound gives sum <= sqrt(2) = 1.414. FT is tighter.

    # For LARGE radii near 0.5: pair bound gives r_i + r_j <= sqrt(2).
    # With just 2 circles at max size: sum = sqrt(2) ~ 1.414.
    # FT for n=2: sqrt(2/alpha) = 0.760. But the actual max for n=2 is
    # ~0.586 (two circles touching diagonally).
    # So for n=2, FT (0.760) is already quite loose.

    # The pair constraint r_i + r_j <= sqrt(2) is not useful here.
    # SOCP with all O(n^2) pair constraints would be tighter.

    return ft


def main():
    n = 26
    best_known = 2.6359830865

    print("=" * 70)
    print("RIGOROUS BOUNDS FOR n=26 NON-CONGRUENT CIRCLE PACKING")
    print("=" * 70)

    ft = fejes_toth_bound(n)
    bs = boroczky_szabo_congruent(n)

    print(f"\n1. Fejes-Toth (non-congruent):     {ft:.6f}  (gap: {(ft-best_known)/best_known*100:.2f}%)")
    print(f"2. Boroczky-Szabo (congruent only): {bs:.6f}  (NOT valid for non-congruent)")
    print(f"   Best known solution:             {best_known:.10f}")

    print(f"\n--- Analysis ---")
    print(f"The FT bound of {ft:.4f} is the tightest known PROVABLE upper bound")
    print(f"for the non-congruent circle packing problem in [0,1]^2.")
    print(f"")
    print(f"The boundary-corrected Boroczky-Szabo bound ({bs:.4f}) applies only")
    print(f"to congruent circles and gives sum < best_known, which is consistent")
    print(f"(the optimal non-congruent packing beats the congruent bound).")
    print(f"")
    print(f"To beat FT for non-congruent circles, we would need:")
    print(f"  (a) Higher-order SDP (Lasserre hierarchy), OR")
    print(f"  (b) Branch-and-bound with LP/SDP relaxations, OR")
    print(f"  (c) Problem-specific coupling inequalities between radii")
    print(f"")
    print(f"The gap {(ft-best_known)/best_known*100:.2f}% appears to be a fundamental")
    print(f"limitation of area-based arguments for non-congruent packings.")

    return ft


if __name__ == "__main__":
    main()
