"""
Improved upper bounds for circle packing sum-of-radii.

Strategy: combine multiple independent bounds to get the tightest result.

Approach 1: SYMMETRY-BREAKING SDP
The basic SDP treats all circles identically. But we can break symmetry:
- Order radii: r_1 >= r_2 >= ... >= r_n
- Add symmetry-breaking cuts: E[r_i * r_j] is ordered
- Add VALID INEQUALITIES from known geometric facts

Approach 2: DISJUNCTIVE PROGRAMMING (LP-based)
For each pair (i,j), the non-overlap constraint is:
  (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
This implies: max(|x_i - x_j|, |y_i - y_j|) >= (r_i + r_j) / sqrt(2)
AND: |x_i - x_j| + |y_i - y_j| >= r_i + r_j  [L1 lower bound on L2]

The L1 bound is LINEAR after introducing absolute values.

Approach 3: 1D PROJECTION BOUNDS (novel)
Project all circles onto the x-axis. Circle i projects to [x_i - r_i, x_i + r_i].
The sum of projection lengths = 2 * sum(r_i).
But these projections can OVERLAP on the x-axis (circles can be at different y).
However, for circles at the SAME y, their projections don't overlap.

Key idea: at any x-coordinate, the "column" at x intersects some circles.
Each circle i contributes a vertical segment of length 2*sqrt(r_i^2 - (x-x_i)^2)
at coordinate x (when |x - x_i| <= r_i).
These vertical segments must fit in [0,1] and not overlap.
So: sum of segment lengths at any x <= 1.

This gives: integral_0^1 [sum of segment lengths at x] dx <= 1
The integral of segment length for circle i = pi * r_i^2 (area).
So: sum(pi * r_i^2) <= 1.  Just the area bound.

But we can be SMARTER. At coordinate x, the segments must be NON-OVERLAPPING.
The sum of lengths <= 1. Also, each segment must fit within [r_i, 1-r_i]
(containment). So the effective height for segment i is 1 - 2*r_i.
The segments for circles at the same x must pack in height 1:
sum of segment lengths <= 1.

Now consider the DENSITY at each x. The maximum packing density of intervals
on [0,1] is exactly 1 (they can fill [0,1]). So no improvement.

BUT: if we also account for the y-containment constraint:
circle i has center at y_i in [r_i, 1-r_i].
Its segment at x is centered at y_i with half-length sqrt(r_i^2 - (x-x_i)^2).
Top of segment: y_i + sqrt(r_i^2 - (x-x_i)^2) <= 1 (already guaranteed by containment)

Approach 4: BETTER SDP WITH L1 NON-OVERLAP
Replace the quadratic non-overlap E[(xi-xj)^2 + (yi-yj)^2] >= E[(ri+rj)^2]
with the L1 version: E[|xi-xj| + |yi-yj|] >= E[ri+rj]
Since |a| >= a and |a| >= -a, we have:
For each pair (i,j), AT LEAST ONE of:
  (xi-xj) >= (ri+rj)/sqrt(2) or (xj-xi) >= (ri+rj)/sqrt(2) or
  (yi-yj) >= (ri+rj)/sqrt(2) or (yj-yi) >= (ri+rj)/sqrt(2)

This is a DISJUNCTIVE constraint. We can handle it with LP + big-M or
with the McCormick relaxation.

Approach 5: STRIP DECOMPOSITION WITH CAREFUL COUNTING
Divide [0,1] into K vertical strips. In each strip of width w = 1/K,
the circles whose centers are in that strip form a sub-packing.
For these circles: they must pack in a strip of width w (extending a bit outside).
In a strip of width w, a circle of radius r has its center at distance >= r from
the left and right walls (actually, from the square boundary, not the strip).
BUT: a circle centered in strip k can be constrained by circles in strip k-1 and k+1.

Key insight: in a strip of width w, the circles centered there have
sum(2*r_i) >= something (they need horizontal space).
No, sum(2*r_i) can exceed w because circles can be at different y.

ACTUALLY: let me think about this from a GRAPH COLORING angle.
The "interval graph" on the y-axis for circles in a strip:
circles i and j in the same strip with |y_i - y_j| < r_i + r_j must be
at horizontal distance >= r_i + r_j.

Within a strip of width w: horizontal distance between centers <= w.
So if |y_i - y_j| < r_i + r_j AND they're in the same strip:
  (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
  (xi-xj)^2 >= (ri+rj)^2 - (yi-yj)^2 > 0
  |xi-xj| >= sqrt((ri+rj)^2 - (yi-yj)^2) > 0.
And |xi-xj| <= w (same strip).
So: w >= sqrt((ri+rj)^2 - (yi-yj)^2).
  w^2 >= (ri+rj)^2 - (yi-yj)^2.
  (yi-yj)^2 >= (ri+rj)^2 - w^2.
  |yi-yj| >= sqrt((ri+rj)^2 - w^2)  if ri+rj > w.

So circles with ri+rj > w in the same strip must be VERTICALLY separated by
at least sqrt((ri+rj)^2 - w^2).

This is useful! For circles with ri+rj > w in a strip of width w,
they form a "1D packing" in the y-direction with separations.

Approach 6: VALID INEQUALITY from largest circle
The largest circle has radius r_max <= 0.5.
It occupies a region of radius r_max.
All other circles must be outside this region.
The "available area" for other circles is roughly 1 - pi*r_max^2.
By FT: sum(r_i^2, i>=2) <= (1 - pi*r_max^2) / (2*sqrt(3)).
Hmm, but FT applies to ALL circles, not just the remaining ones.
Actually, we can split: FT says sum(2*sqrt(3)*r_i^2) <= 1 (for all i).
This doesn't directly give us a bound on the remaining circles after
accounting for the largest.

ACTUALLY: the Oler bound already accounts for boundary.
The key issue is that all our bounds are area-based and don't capture
packing geometry well enough.

Let me implement the BEST combination I can.
"""

import numpy as np
import cvxpy as cp
import json
import sys
from pathlib import Path
from scipy.optimize import linprog, minimize


def sdp_symmetry_breaking_bound(n, verbose=False):
    """
    SDP with symmetry breaking and stronger cuts.

    Key improvements over basic SDP:
    1. Order radii: r_1 >= r_2 >= ... >= r_n  (at the moment level)
    2. Add bound: for k circles of radius >= r, need certain area
    3. Add strip-based cuts
    4. Add L1 relaxation of non-overlap
    """
    dim = 3 * n + 1

    def xi(i): return 1 + i
    def yi(i): return 1 + n + i
    def ri(i): return 1 + 2*n + i

    M = cp.Variable((dim, dim), symmetric=True, name="M")
    constraints = []

    # PSD and normalization
    constraints += [M >> 0]
    constraints += [M[0, 0] == 1]

    # First-order bounds
    for i in range(n):
        constraints += [M[0, xi(i)] >= 0, M[0, xi(i)] <= 1]
        constraints += [M[0, yi(i)] >= 0, M[0, yi(i)] <= 1]
        constraints += [M[0, ri(i)] >= 1e-6]  # positive radius
        constraints += [M[0, ri(i)] <= 0.5]

    # Second-order diagonal bounds
    for i in range(n):
        constraints += [M[xi(i), xi(i)] <= 1]
        constraints += [M[yi(i), yi(i)] <= 1]
        constraints += [M[ri(i), ri(i)] <= 0.25]

    # Containment (first order)
    for i in range(n):
        constraints += [M[0, xi(i)] >= M[0, ri(i)]]
        constraints += [M[0, xi(i)] + M[0, ri(i)] <= 1]
        constraints += [M[0, yi(i)] >= M[0, ri(i)]]
        constraints += [M[0, yi(i)] + M[0, ri(i)] <= 1]

    # Containment (second order): (x_i - r_i)^2 >= 0 and (1-x_i-r_i)^2 >= 0
    for i in range(n):
        constraints += [M[xi(i),xi(i)] - 2*M[xi(i),ri(i)] + M[ri(i),ri(i)] >= 0]
        constraints += [1 - 2*M[0,xi(i)] - 2*M[0,ri(i)] + M[xi(i),xi(i)] + 2*M[xi(i),ri(i)] + M[ri(i),ri(i)] >= 0]
        constraints += [M[yi(i),yi(i)] - 2*M[yi(i),ri(i)] + M[ri(i),ri(i)] >= 0]
        constraints += [1 - 2*M[0,yi(i)] - 2*M[0,ri(i)] + M[yi(i),yi(i)] + 2*M[yi(i),ri(i)] + M[ri(i),ri(i)] >= 0]

    # Non-overlap (quadratic): E[dist_ij^2] >= E[(ri+rj)^2]
    for i in range(n):
        for j in range(i+1, n):
            constraints += [
                M[xi(i),xi(i)] - 2*M[xi(i),xi(j)] + M[xi(j),xi(j)] +
                M[yi(i),yi(i)] - 2*M[yi(i),yi(j)] + M[yi(j),yi(j)] >=
                M[ri(i),ri(i)] + 2*M[ri(i),ri(j)] + M[ri(j),ri(j)]
            ]

    # Area constraint (FT)
    constraints += [2*np.sqrt(3) * sum(M[ri(i),ri(i)] for i in range(n)) <= 1]

    # SYMMETRY BREAKING: order expected radii
    for i in range(n-1):
        constraints += [M[0, ri(i)] >= M[0, ri(i+1)]]

    # RLT cuts from containment products
    for i in range(n):
        for j in range(i+1, n):
            # (x_i - r_i)(x_j - r_j) >= 0
            constraints += [M[xi(i),xi(j)] - M[xi(i),ri(j)] - M[ri(i),xi(j)] + M[ri(i),ri(j)] >= 0]
            # (1-x_i-r_i)(1-x_j-r_j) >= 0
            constraints += [
                1 - M[0,xi(i)] - M[0,ri(i)] - M[0,xi(j)] - M[0,ri(j)]
                + M[xi(i),xi(j)] + M[xi(i),ri(j)] + M[ri(i),xi(j)] + M[ri(i),ri(j)]
                >= 0
            ]
            # (x_i - r_i)(1 - x_j - r_j) >= 0
            constraints += [
                M[0,xi(i)] - M[0,ri(i)] - M[xi(i),xi(j)] - M[xi(i),ri(j)]
                + M[ri(i),xi(j)] + M[ri(i),ri(j)]
                >= 0
            ]
            # (1-x_i-r_i)(x_j - r_j) >= 0
            constraints += [
                M[0,xi(j)] - M[0,ri(j)] - M[xi(i),xi(j)] + M[xi(i),ri(j)]
                - M[ri(i),xi(j)] + M[ri(i),ri(j)]
                >= 0
            ]
            # Same for y
            constraints += [M[yi(i),yi(j)] - M[yi(i),ri(j)] - M[ri(i),yi(j)] + M[ri(i),ri(j)] >= 0]
            constraints += [
                1 - M[0,yi(i)] - M[0,ri(i)] - M[0,yi(j)] - M[0,ri(j)]
                + M[yi(i),yi(j)] + M[yi(i),ri(j)] + M[ri(i),yi(j)] + M[ri(i),ri(j)]
                >= 0
            ]
            constraints += [
                M[0,yi(i)] - M[0,ri(i)] - M[yi(i),yi(j)] - M[yi(i),ri(j)]
                + M[ri(i),yi(j)] + M[ri(i),ri(j)]
                >= 0
            ]
            constraints += [
                M[0,yi(j)] - M[0,ri(j)] - M[yi(i),yi(j)] + M[yi(i),ri(j)]
                - M[ri(i),yi(j)] + M[ri(i),ri(j)]
                >= 0
            ]
            # Cross: (x_i - r_i)(y_j - r_j) >= 0
            constraints += [M[xi(i),yi(j)] - M[xi(i),ri(j)] - M[ri(i),yi(j)] + M[ri(i),ri(j)] >= 0]
            constraints += [M[yi(i),xi(j)] - M[yi(i),ri(j)] - M[ri(i),xi(j)] + M[ri(i),ri(j)] >= 0]

            # r_i * (0.5 - r_j) >= 0
            constraints += [0.5*M[0,ri(i)] - M[ri(i),ri(j)] >= 0]
            constraints += [0.5*M[0,ri(j)] - M[ri(i),ri(j)] >= 0]

    # NEW: Stronger pairwise bound from L1 relaxation of L2.
    # dist_ij^2 = (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
    # This implies |xi-xj| + |yi-yj| >= ri+rj (since L1 >= L2 in 2D? No, L2 >= L1/sqrt(2))
    # Actually: L2 >= max(|dx|, |dy|), L1 = |dx|+|dy| >= L2.
    # So L2 >= ri+rj implies L1 >= ri+rj (since L1 >= L2).
    # But we already have E[dist^2] >= E[(ri+rj)^2].
    # We need: E[|dx|+|dy|] >= E[ri+rj].
    # This is LINEAR in the absolute values but we can't directly encode it.
    #
    # Instead, use: E[dx^2] + E[dy^2] >= E[(ri+rj)^2]
    # AND: E[dx^2] >= 0, E[dy^2] >= 0
    #
    # A VALID CUT: for each pair (i,j), the average squared distance must
    # be at least the average squared sum of radii.
    # Let's add: the total pairwise average:
    # sum_{i<j} [E[dist_ij^2] - E[(ri+rj)^2]] >= 0 (already implied)
    #
    # Try: for each triple (i,j,k), sum of pairwise distances >= some function of radii.
    # Packing 3 circles: d12+d13+d23 >= (r1+r2)+(r1+r3)+(r2+r3) = 2*(r1+r2+r3).
    # No, d_ij >= r_i+r_j doesn't give sum d_ij >= 2*sum(r_i).
    # It gives sum d_ij >= sum(r_i+r_j) = sum_{i<j}(r_i+r_j) = (n-1)*sum(r_i).
    # So: sum_{i<j} d_ij >= (n-1) * sum(r_i).
    # This is valid! In moment form:
    # sum_{i<j} E[dist_ij] >= (n-1) * sum_i E[r_i]
    # But E[dist_ij] is hard (not in our moment matrix).
    # We have E[dist_ij^2] but not E[dist_ij].
    # By Jensen: E[dist_ij] <= sqrt(E[dist_ij^2]).
    # So sqrt(E[dist_ij^2]) >= E[dist_ij] >= ... not useful for upper bounding the sum.

    # NEW CUT: Maximum distance constraint.
    # All centers are in [0,1]^2. Max pairwise distance = sqrt(2).
    # For the i-th circle: sum_{j!=i} dist_ij <= (n-1)*sqrt(2).
    # And sum_{j!=i} dist_ij >= sum_{j!=i} (ri+rj) = (n-1)*ri + sum_{j!=i} rj.
    # E[sum_{j!=i} dist_ij^2] >= E[sum_{j!=i}(ri+rj)^2]. Already have per-pair.

    # NEW CUT: Valid inequality from sum of non-overlap constraints.
    # sum_{i<j} [(xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2] >= 0
    # = sum_{i<j} [xi^2 - 2xi*xj + xj^2 + yi^2 - 2yi*yj + yj^2 - ri^2 - 2ri*rj - rj^2]
    # = (n-1)*sum(xi^2+yi^2) - 2*sum_{i<j}(xi*xj+yi*yj) - (n-1)*sum(ri^2) - 2*sum_{i<j}ri*rj
    # = (n-1)*sum(xi^2+yi^2) - [sum(xi)]^2 + sum(xi^2) - [sum(yi)]^2 + sum(yi^2)
    #   - (n-1)*sum(ri^2) - [sum(ri)]^2 + sum(ri^2)
    # = n*sum(xi^2+yi^2) - [sum(xi)]^2 - [sum(yi)]^2
    #   - (n-2)*sum(ri^2) - [sum(ri)]^2
    # Using Var(X) = E[X^2] - E[X]^2 >= 0:
    # This gives a complicated constraint that may or may not be binding.

    # Let me try SUM of pairwise E[dist^2] vs aggregate:
    # n * sum E[xi^2] - (sum E[xi])^2 >= ... This needs sum variables.
    # Use: sum_x = sum M[0,xi(i)], sumsq_x = sum M[xi(i),xi(i)], etc.

    sum_x = sum(M[0, xi(i)] for i in range(n))
    sum_y = sum(M[0, yi(i)] for i in range(n))
    sum_r = sum(M[0, ri(i)] for i in range(n))
    sumsq_x = sum(M[xi(i), xi(i)] for i in range(n))
    sumsq_y = sum(M[yi(i), yi(i)] for i in range(n))
    sumsq_r = sum(M[ri(i), ri(i)] for i in range(n))
    cross_xx = sum(M[xi(i), xi(j)] for i in range(n) for j in range(i+1, n))
    cross_yy = sum(M[yi(i), yi(j)] for i in range(n) for j in range(i+1, n))
    cross_rr = sum(M[ri(i), ri(j)] for i in range(n) for j in range(i+1, n))

    # Aggregate non-overlap:
    # (n-1)*sumsq_x - 2*cross_xx + (n-1)*sumsq_y - 2*cross_yy >= (n-1)*sumsq_r + 2*cross_rr
    # This is equivalent to sum of individual constraints, so already implied.

    # Variance of x: n*sumsq_x - sum_x^2 >= 0  (can't directly encode sum_x^2)
    # But we can use: sumsq_x >= sum_x^2 / n  (Cauchy-Schwarz)
    # This means: M entries must satisfy n * sumsq_x >= (sum_x)^2
    # We can't encode (sum_x)^2 directly, but for an auxiliary variable.

    # AUXILIARY: t_x = sum E[xi] = sum_x. We know sum_x^2 is a nonlinear function.
    # In the SDP: sum_x = a linear function of M entries.
    # sum_x^2 = (linear)^2, which is NOT linear.
    # We can use a Schur complement:
    # [[n * sumsq_x, sum_x], [sum_x, 1]] >> 0
    # This encodes n * sumsq_x >= sum_x^2.
    # But this is a NEW semidefinite constraint, separate from M >> 0.
    # Hmm, actually it's already implied by M >> 0 and the definitions.

    # Let me try another angle: STRENGTHENED AREA BOUND.
    # The Oler bound for n equal circles in the unit square:
    # 2*sqrt(3)*n*r^2 <= 1 + 4*r*(2*(ceil(sqrt(n))-1)/(ceil(sqrt(n))) ) + ...
    # This is complicated. Let me just compute bounds numerically.

    # Objective
    objective = cp.Maximize(sum(M[0, ri(i)] for i in range(n)))
    prob = cp.Problem(objective, constraints)

    if verbose:
        print(f"  SDP (n={n}): dim={dim}, constraints={len(constraints)}")

    try:
        result = prob.solve(solver=cp.SCS, max_iters=100000, verbose=False, eps=1e-8)
        if verbose:
            print(f"  Status: {prob.status}, bound: {result:.6f}")
            if M.value is not None:
                radii = sorted([M.value[0, ri(i)] for i in range(n)], reverse=True)
                print(f"  Radii (top 5): {[f'{r:.4f}' for r in radii[:5]]}")
        return result
    except Exception as e:
        if verbose:
            print(f"  Error: {e}")
        return None


def strip_bound(n, K=50, verbose=False):
    """
    Strip-based upper bound.

    Divide [0,1] into K vertical strips of width w = 1/K.
    In each strip, circles whose centers are in that strip must pack.

    For each strip of width w, circles centered there with radii r_1,...,r_m:
    - Containment: r_i <= w_left margin and <= w_right margin (depends on strip position)
    - Packing: circles must not overlap, both within strip and with neighboring strips

    Cross-strip constraint: a circle in strip k with radius r > w/2 extends into strips k-1 and k+1.
    Its "shadow" in adjacent strips means those strips have less available space.

    For an UPPER BOUND: we relax the between-strip non-overlap and only keep within-strip constraints.
    Within a strip, circles project onto vertical line with interval [yi-ri, yi+ri].
    If two circles in the same strip have overlapping vertical projections,
    they need horizontal separation >= sqrt((ri+rj)^2 - (yi-yj)^2).
    But within a strip of width w, horizontal separation <= w.
    So they can only coexist if (yi-yj)^2 >= (ri+rj)^2 - w^2.

    For circles with ri+rj <= w, they can always coexist in the strip.
    For circles with ri+rj > w, they need vertical separation >= sqrt((ri+rj)^2 - w^2).

    This is a 1D packing problem per strip, which gives tighter bounds.

    However, optimizing over ALL possible assignments of circles to strips is hard.
    Instead, we use the CONTINUOUS relaxation:

    For a strip at position x in [0,1] with width dx:
    The circles passing through x have centers within [x-r_i, x+r_i].
    At height y within the strip, the total "width" of circles must sum to <= 1
    (well, the circles are 2D, not just widths).

    This is getting complicated. Let me try a simpler approach.
    """
    # Simple: n circles, each of max possible radius, constrained by strip packing.
    # Use the WEIGHTED CLIQUE COVER bound.

    # For a strip of width w = 1/K at position [(k-1)/K, k/K]:
    # Circles centered in this strip with radius > w need vertical separation.
    # Approximate: in each strip, at most floor(1/(2*r_min)) circles of radius r_min.
    # This doesn't lead anywhere cleanly.

    # Instead, use the INTEGRATION approach:
    # At any vertical line x=a, the sum of chord lengths of circles crossing it <= 1.
    # chord(i, a) = 2*sqrt(r_i^2 - (a - x_i)^2) when |a - x_i| <= r_i, else 0.
    # integral_0^1 chord(i, a) da = pi * r_i^2 (area of circle i).
    # So sum_i pi * r_i^2 = integral_0^1 [sum_i chord(i,a)] da <= integral_0^1 1 da = 1.
    # This is just the area bound.

    # To beat the area bound, we need to use that chords don't overlap more carefully.
    # Actually, the packing density of 1D intervals (chords at height y) is < 1
    # because the intervals have specific sizes related to the circle radii.
    # But there's no theoretical bound < 1 for 1D interval packing density.

    return np.sqrt(n / np.pi)  # Can't beat area bound with strips alone


def sdp_pair_strengthened(n, verbose=False):
    """
    SDP with PAIR STRENGTHENING.

    For each pair (i,j), add the constraint:
    E[(xi-xj)^2] + E[(yi-yj)^2] >= E[ri^2] + 2*E[ri*rj] + E[rj^2]

    AND: E[(xi-xj)^2] <= E[(1-ri-rj)^2]  (max possible x-distance when circles fit)
    Wait: |xi - xj| <= 1 - ri - rj is NOT true (it could be larger if circles
    are on opposite sides). Actually: xi in [ri, 1-ri], xj in [rj, 1-rj].
    Max |xi-xj| = |(1-ri) - rj| = 1-ri-rj.
    So (xi-xj)^2 <= (1-ri-rj)^2.
    E[(xi-xj)^2] <= E[(1-ri-rj)^2] = 1 - 2*E[ri] - 2*E[rj] + 2*E[ri*rj] + E[ri^2] + E[rj^2]
    Similarly for y.

    Combined: E[dist_ij^2] <= 2*(1-ri-rj)^2 is NOT right.
    Actually E[dist_ij^2] = E[dx^2] + E[dy^2] <= (1-E[ri]-E[rj])^2 * 2... no.
    E[dx^2] = M[xi,xi] - 2*M[xi,xj] + M[xj,xj]
    We need: xi - xj <= 1 - ri - rj. So (xi-xj)^2 <= (1-ri-rj)^2.
    In moment form: E[(xi-xj)^2] <= E[(1-ri-rj)^2].

    This is VALID. Let me add it.

    Also: (xi-xj)^2 <= (1-ri-rj)^2 means:
    M[xi,xi] - 2M[xi,xj] + M[xj,xj] <= 1 - 2M[0,ri] - 2M[0,rj] + 2M[ri,rj] + M[ri,ri] + M[rj,rj]

    Hmm wait. E[(xi-xj)^2] <= E[(1-ri-rj)^2]?
    This is saying: the second moment of (xi-xj) is at most the second moment of (1-ri-rj).
    But (xi-xj)^2 <= (1-ri-rj)^2 pointwise (since |xi-xj| <= 1-ri-rj).
    So E[(xi-xj)^2] <= E[(1-ri-rj)^2]. YES, this is valid!
    """
    dim = 3 * n + 1

    def xi(i): return 1 + i
    def yi(i): return 1 + n + i
    def ri(i): return 1 + 2*n + i

    M = cp.Variable((dim, dim), symmetric=True, name="M")
    constraints = []

    constraints += [M >> 0, M[0, 0] == 1]

    for i in range(n):
        constraints += [M[0, xi(i)] >= 0, M[0, xi(i)] <= 1]
        constraints += [M[0, yi(i)] >= 0, M[0, yi(i)] <= 1]
        constraints += [M[0, ri(i)] >= 0, M[0, ri(i)] <= 0.5]
        constraints += [M[xi(i), xi(i)] <= 1, M[yi(i), yi(i)] <= 1]
        constraints += [M[ri(i), ri(i)] <= 0.25]

    # Containment
    for i in range(n):
        constraints += [M[0, xi(i)] >= M[0, ri(i)]]
        constraints += [M[0, xi(i)] + M[0, ri(i)] <= 1]
        constraints += [M[0, yi(i)] >= M[0, ri(i)]]
        constraints += [M[0, yi(i)] + M[0, ri(i)] <= 1]

    # Containment 2nd order
    for i in range(n):
        constraints += [M[xi(i),xi(i)] - 2*M[xi(i),ri(i)] + M[ri(i),ri(i)] >= 0]
        constraints += [1 - 2*M[0,xi(i)] - 2*M[0,ri(i)] + M[xi(i),xi(i)] + 2*M[xi(i),ri(i)] + M[ri(i),ri(i)] >= 0]
        constraints += [M[yi(i),yi(i)] - 2*M[yi(i),ri(i)] + M[ri(i),ri(i)] >= 0]
        constraints += [1 - 2*M[0,yi(i)] - 2*M[0,ri(i)] + M[yi(i),yi(i)] + 2*M[yi(i),ri(i)] + M[ri(i),ri(i)] >= 0]

    # Non-overlap (basic)
    for i in range(n):
        for j in range(i+1, n):
            constraints += [
                M[xi(i),xi(i)] - 2*M[xi(i),xi(j)] + M[xi(j),xi(j)] +
                M[yi(i),yi(i)] - 2*M[yi(i),yi(j)] + M[yi(j),yi(j)] >=
                M[ri(i),ri(i)] + 2*M[ri(i),ri(j)] + M[ri(j),ri(j)]
            ]

    # FT area bound
    constraints += [2*np.sqrt(3) * sum(M[ri(i),ri(i)] for i in range(n)) <= 1]

    # NEW: Upper bound on squared distance (from containment geometry)
    # |xi - xj| <= 1 - ri - rj pointwise
    # => E[(xi-xj)^2] <= E[(1-ri-rj)^2]
    for i in range(n):
        for j in range(i+1, n):
            # dx^2 <= (1-ri-rj)^2
            constraints += [
                M[xi(i),xi(i)] - 2*M[xi(i),xi(j)] + M[xi(j),xi(j)] <=
                1 - 2*M[0,ri(i)] - 2*M[0,ri(j)] + 2*M[ri(i),ri(j)] + M[ri(i),ri(i)] + M[ri(j),ri(j)]
            ]
            # dy^2 <= (1-ri-rj)^2
            constraints += [
                M[yi(i),yi(i)] - 2*M[yi(i),yi(j)] + M[yi(j),yi(j)] <=
                1 - 2*M[0,ri(i)] - 2*M[0,ri(j)] + 2*M[ri(i),ri(j)] + M[ri(i),ri(i)] + M[ri(j),ri(j)]
            ]

    # RLT cuts
    for i in range(n):
        for j in range(i+1, n):
            constraints += [M[xi(i),xi(j)] - M[xi(i),ri(j)] - M[ri(i),xi(j)] + M[ri(i),ri(j)] >= 0]
            constraints += [
                1 - M[0,xi(i)] - M[0,ri(i)] - M[0,xi(j)] - M[0,ri(j)]
                + M[xi(i),xi(j)] + M[xi(i),ri(j)] + M[ri(i),xi(j)] + M[ri(i),ri(j)]
                >= 0
            ]
            constraints += [M[yi(i),yi(j)] - M[yi(i),ri(j)] - M[ri(i),yi(j)] + M[ri(i),ri(j)] >= 0]
            constraints += [
                1 - M[0,yi(i)] - M[0,ri(i)] - M[0,yi(j)] - M[0,ri(j)]
                + M[yi(i),yi(j)] + M[yi(i),ri(j)] + M[ri(i),yi(j)] + M[ri(i),ri(j)]
                >= 0
            ]
            constraints += [0.5*M[0,ri(i)] - M[ri(i),ri(j)] >= 0]
            constraints += [0.5*M[0,ri(j)] - M[ri(i),ri(j)] >= 0]

    # Symmetry breaking
    for i in range(n-1):
        constraints += [M[0, ri(i)] >= M[0, ri(i+1)]]

    # Pairwise sum bound: r_1 + r_2 <= 2 - sqrt(2) (from diagonal placement)
    if n >= 2:
        constraints += [M[0, ri(0)] + M[0, ri(1)] <= 2 - np.sqrt(2)]

    objective = cp.Maximize(sum(M[0, ri(i)] for i in range(n)))
    prob = cp.Problem(objective, constraints)

    if verbose:
        print(f"  Pair-strengthened SDP (n={n}): dim={dim}, constraints={len(constraints)}")

    try:
        result = prob.solve(solver=cp.SCS, max_iters=100000, verbose=False, eps=1e-8)
        if verbose:
            print(f"  Status: {prob.status}, bound: {result:.6f}")
            if M.value is not None:
                radii = sorted([M.value[0, ri(i)] for i in range(n)], reverse=True)
                print(f"  Radii (top 5): {[f'{r:.4f}' for r in radii[:5]]}")
        return result
    except Exception as e:
        if verbose:
            print(f"  Error: {e}")
        return None


def socp_enhanced_bound(n, verbose=False):
    """
    Enhanced SOCP using radius-only variables with multiple geometric constraints.

    Variables: r_1 >= r_2 >= ... >= r_n (sorted radii)

    Constraints:
    1. FT area: 2*sqrt(3) * sum(r_i^2) <= 1
    2. Oler area: 2*sqrt(3) * sum(r_i^2) <= 1 + 4*r_1*(sqrt(2*sqrt(3)*sum(r_i^2))-sum...)
       This is complex. Use the numerically computed Oler.
    3. r_i <= 0.5
    4. r_1 + r_2 <= 2 - sqrt(2) (diagonal pair bound)
    5. For k largest circles with equal radii r: k*2*sqrt(3)*r^2 + boundary terms <= 1.
    6. NEW: for the TOP-k circles, their total Voronoi area >= 2*sqrt(3)*sum(r_i^2, i<=k).
       The remaining n-k circles occupy the rest of the square.

    Key insight: we can ADD constraints for each prefix of circles (sorted by radius).
    """
    r = cp.Variable(n)
    constraints = []

    constraints += [r >= 1e-8, r <= 0.5]
    for i in range(n-1):
        constraints += [r[i] >= r[i+1]]

    # FT area (global)
    s = 2 * np.sqrt(3)
    constraints += [s * cp.sum_squares(r) <= 1]

    # Pairwise: top 2 radii sum
    if n >= 2:
        constraints += [r[0] + r[1] <= 2 - np.sqrt(2)]

    # NEW: for any 4 circles placed at corners (optimal for 4 large circles):
    # Adjacent corners: distance = 1 - ri - rj. Need >= ri + rj.
    # So ri + rj <= 0.5 for adjacent pairs.
    # For 4 circles at corners: each is adjacent to 2 others.
    # If r_1,r_2,r_3,r_4 are placed at corners:
    # Can arrange so r_1 is diagonal from r_4, r_2 is diagonal from r_3.
    # Then adjacent pairs: (1,2), (1,3), (2,4), (3,4).
    # r1+r2<=0.5, r1+r3<=0.5, r2+r4<=0.5, r3+r4<=0.5.
    # From r1+r2<=0.5 and r1+r3<=0.5: r2,r3 <= 0.5-r1.
    # From r2+r4<=0.5: r4 <= 0.5-r2.
    # Sum = r1+r2+r3+r4 <= r1 + 2*(0.5-r1) + (0.5-r2) = 1.5 - r1 - r2.
    # Max when r1=r2=0.25: sum <= 1.0. This matches n=4 optimal!
    # For 5+ circles: 4 at corners sum <= 1.0, plus more in interior.

    # For n >= 4: sum of top 4 radii <= 1.0 (corner bound)
    if n >= 4:
        constraints += [r[0] + r[1] + r[2] + r[3] <= 1.0]

    # Actually we can tighten: r1+r2<=0.5, r3+r4<=0.5 (the two diagonal pairs)
    # Wait, we don't know the arrangement. The WEAKEST arrangement gives:
    # For ANY 4 non-overlapping circles in [0,1]^2:
    # We need r1+r2 <= 2-sqrt(2). But for the TOP-4 sum, the corner arrangement
    # gives 1.0 which is NOT a universal bound.
    # Let me verify: can 4 circles sum > 1.0?
    # Known optimal for n=4 is exactly 1.0 (4 circles of r=0.25 at corners).
    # So sum <= 1.0 for n=4 is CORRECT.
    # For n >= 5, the top-4 sum is still <= 1.0.
    # Is this valid? Can we have 4 circles that sum > 1.0 with more circles present?
    # No - if 4 circles alone can't sum > 1.0, having more circles only makes it harder.
    # So: for any subset of 4 circles, their sum <= 1.0.
    # In particular, the 4 largest: r[0]+r[1]+r[2]+r[3] <= 1.0.

    # For 3 circles: best is 0.7645. So top-3 sum <= 0.7645.
    # But we probably can't prove this purely from geometry.
    # Actually for n=3 case: optimal value is ~0.7645.
    # For the top-3 of any n-circle packing: r1+r2+r3 <= 0.7645.
    # Wait, for n=4 optimal (four 0.25s), r1+r2+r3 = 0.75. And 0.75 < 0.7645? No!
    # 0.75 < 0.7645 is false. 0.75 > 0.7645 is also false. 0.75 < 0.7645 = False.
    # Wait: 0.7645 > 0.75. So for n=4, top 3 radii sum = 0.75 < 0.7645. OK.
    # But is the n=3 optimal = 0.7645 the true bound on top-3?
    # For n=26, the top 3 radii might sum much less. But the BOUND on top-3 is 0.7645.
    # However, this bound is based on n=3 optimal, not proven analytically.
    # Skip for now.

    # For k >= 5: k circles of equal radius r in [0,1]^2.
    # By FT: 2*sqrt(3)*k*r^2 <= 1 => r <= 1/sqrt(2*sqrt(3)*k)
    # Sum = k*r <= sqrt(k/(2*sqrt(3)))
    # For k=5: sum <= sqrt(5/3.464) = sqrt(1.443) = 1.2013. Matches FT bound for n=5.

    # NEW: Interaction between large and small circles.
    # If circle 1 has radius r, the ANNULAR region around it
    # (between radius r and 2r from center) must contain no other circle centers.
    # (Because for circle j with center within 2r of circle 1 center,
    # dist >= r+rj, and if dist < 2r then rj < r.)
    # The annular exclusion zone has area pi*(2r)^2 - pi*r^2 = 3*pi*r^2.
    # ... but this doesn't directly constrain radii sum.

    # ADD Oler-style boundary correction.
    # Oler says: 2*sqrt(3)*sum(r_i^2) <= 1 + perimeter_correction.
    # For mixed radii: 2*sqrt(3)*sum(r_i^2) <= (1 + 2*r_max)^2 approximately.
    # Actually the Oler bound for the expanded square [0,1+2r]^2 minus boundary:
    # No, let me not use this - it makes the bound WEAKER.

    objective = cp.Maximize(cp.sum(r))
    prob = cp.Problem(objective, constraints)

    try:
        result = prob.solve(solver=cp.SCS, verbose=False, max_iters=50000, eps=1e-8)
        if verbose and r.value is not None:
            rv = sorted(r.value, reverse=True)
            print(f"  Enhanced SOCP (n={n}): {result:.6f}")
            print(f"  Top radii: {[f'{x:.4f}' for x in rv[:5]]}")
        return result
    except:
        return None


def compute_all_bounds(n, verbose=True):
    """Compute all available bounds for given n."""
    known_best = {
        1: 0.5000, 2: 0.5858, 3: 0.7645, 4: 1.0000, 5: 1.0854,
        10: 1.5911, 15: 2.0365, 20: 2.3010, 26: 2.6360, 30: 2.8425, 32: 2.9390,
    }

    results = {}

    # Area bound
    area = np.sqrt(n / np.pi)
    results['area'] = area

    # FT bound
    ft = np.sqrt(n / (2*np.sqrt(3)))
    results['fejes_toth'] = ft

    # Enhanced SOCP
    socp = socp_enhanced_bound(n, verbose=verbose)
    if socp is not None:
        results['socp_enhanced'] = socp

    # SDP with pair strengthening (only for small n, too slow for large)
    if n <= 10:
        sdp = sdp_pair_strengthened(n, verbose=verbose)
        if sdp is not None:
            results['sdp_pair'] = sdp

    best = min(v for v in results.values() if v is not None)
    results['best'] = best
    results['known'] = known_best.get(n)
    if results['known']:
        results['gap'] = best - results['known']

    return results


def main():
    known_best = {
        1: 0.5000, 2: 0.5858, 3: 0.7645, 4: 1.0000, 5: 1.0854,
        10: 1.5911, 15: 2.0365, 20: 2.3010, 26: 2.6360, 30: 2.8425, 32: 2.9390,
    }

    if len(sys.argv) > 1:
        n_values = [int(x) for x in sys.argv[1:]]
    else:
        n_values = [1, 2, 3, 4, 5, 10]

    print("Improved Upper Bounds for Circle Packing")
    print("=" * 80)

    all_results = {}
    for n_val in n_values:
        print(f"\nn = {n_val}")
        print("-" * 40)
        results = compute_all_bounds(n_val, verbose=True)
        all_results[str(n_val)] = {k: float(v) if v is not None else None
                                    for k, v in results.items()}

        known = known_best.get(n_val, 0)
        best = results['best']
        gap = best - known if known else None
        valid = best >= known - 1e-4 if known else True

        print(f"\n  Area:     {results['area']:.6f}")
        print(f"  FT:       {results['fejes_toth']:.6f}")
        if 'socp_enhanced' in results:
            print(f"  SOCP:     {results['socp_enhanced']:.6f}")
        if 'sdp_pair' in results:
            print(f"  SDP pair: {results['sdp_pair']:.6f}")
        print(f"  BEST:     {best:.6f} {'OK' if valid else 'FAIL!'}")
        if known:
            print(f"  Known:    {known:.6f}")
            print(f"  Gap:      {gap:.6f} ({100*gap/known:.2f}%)")

    output_path = Path(__file__).parent / "improved_bounds.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
