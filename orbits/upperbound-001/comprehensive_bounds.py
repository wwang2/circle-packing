"""
Comprehensive upper bounds for circle packing sum-of-radii.

Combines multiple bounding techniques:
1. Area bound: sum(pi*r_i^2) <= 1 => sum(r_i) <= sqrt(n/pi)
2. Fejes Toth: sum(2*sqrt(3)*r_i^2) <= 1 => sum(r_i) <= sqrt(n/(2*sqrt(3)))
3. Oler SOCP: mixed-radius optimization with FT + boundary terms
4. SOCP with geometric cuts (pair, top-k sum)
5. SDP with RLT cuts (for small n)

The best bound for each n is the MINIMUM of all valid upper bounds.

For n=26: current best bound is FT = 2.7396, known optimal = 2.636, gap = 3.93%.

NEW APPROACHES to try:
- Counting argument: how many circles can have radius >= r?
- Recursive partitioning: split square, bound each part
- Independence number: the max weighted independent set in a "conflict graph"
"""

import numpy as np
import cvxpy as cp
import json
import sys
from pathlib import Path


def fejes_toth_bound(n):
    """FT: sum(r_i) <= sqrt(n/(2*sqrt(3)))"""
    return np.sqrt(n / (2*np.sqrt(3)))


def area_bound(n):
    """Area: sum(r_i) <= sqrt(n/pi)"""
    return np.sqrt(n / np.pi)


def socp_geometric_bound(n, verbose=False):
    """
    SOCP with geometric cuts on sorted radii.

    Key geometric constraints:
    1. FT area: 2*sqrt(3)*sum(r_i^2) <= 1
    2. Pair: r_1+r_2 <= 2-sqrt(2) ≈ 0.5858
    3. Top-4: r_1+r_2+r_3+r_4 <= 1.0
    4. r_i <= 0.5
    5. NEW: for k circles of radius >= r, need k*2*sqrt(3)*r^2 <= 1
       => the k-th largest radius r_k <= 1/sqrt(2*sqrt(3)*k)
       This gives r_k <= 1/sqrt(3.464*k)
       k=1: r_1 <= 0.537 (weaker than r_1 <= 0.5)
       k=2: r_2 <= 0.380
       k=3: r_3 <= 0.310
       k=4: r_4 <= 0.269
       k=5: r_5 <= 0.240
       These are valid because if k circles have r >= r_k, then
       k * 2*sqrt(3)*r_k^2 <= sum(2*sqrt(3)*r_i^2) <= 1.
    6. NEW: sum of top-k radii <= sum of optimal k-circle packing
       This uses known/computed upper bounds for smaller n.
       E.g., top-2 sum <= UB(2), top-3 sum <= UB(3), etc.
    7. NEW: Counting constraint.
       Let f(k) = max sum(r_i) for k circles in [0,1]^2.
       Then for our n circles: sum(r_i) = sum(r_i, i<=k) + sum(r_i, i>k)
       <= f(k) + (n-k) * r_{k+1}
       <= f(k) + (n-k) * 1/sqrt(2*sqrt(3)*(k+1))
       Optimize over k to get the tightest bound.
    """
    # Known upper bounds for small k (from our computations)
    # Using the tightest bounds we've computed
    known_ub = {
        1: 0.5000,
        2: 0.5858,  # 2-sqrt(2)
        3: 0.8787,  # our SOCP bound (true optimal is 0.7645)
        4: 1.0000,  # exact (4 corners)
    }
    # For larger k, use FT
    def ub(k):
        if k in known_ub:
            return known_ub[k]
        return fejes_toth_bound(k)

    r = cp.Variable(n)
    s = 2 * np.sqrt(3)

    constraints = []
    constraints += [r >= 0, r <= 0.5]

    # Ordering
    for i in range(n-1):
        constraints += [r[i] >= r[i+1]]

    # FT area
    constraints += [s * cp.sum_squares(r) <= 1]

    # Pair bound
    if n >= 2:
        constraints += [r[0] + r[1] <= 2 - np.sqrt(2)]

    # Top-4 bound
    if n >= 4:
        constraints += [r[0] + r[1] + r[2] + r[3] <= 1.0]

    # Individual radius bounds from FT
    for k in range(1, min(n+1, 20)):
        # k-th largest (0-indexed: r[k-1]) satisfies: k * s * r[k-1]^2 <= 1
        # So r[k-1] <= 1/sqrt(s*k)
        max_r_k = 1.0 / np.sqrt(s * k)
        if max_r_k < 0.5:
            constraints += [r[k-1] <= max_r_k]

    # Top-k sum constraints using known bounds
    for k in [2, 3, 4]:
        if k <= n:
            constraints += [cp.sum(r[:k]) <= ub(k)]

    # NEW: Tail bound. For i > k, r_i <= r_k.
    # sum(r_i, i>k) <= (n-k)*r_k <= (n-k)/sqrt(s*(k+1))  [from FT on k+1]
    # This is already captured by ordering + individual bounds.

    # NEW: Mixed FT bound.
    # sum(r_i, i>k) satisfies FT: s*sum(r_i^2, i>k) <= 1 - s*sum(r_i^2, i<=k)
    # And sum(r_i, i>k) <= sqrt((n-k) * sum(r_i^2, i>k)) by CS
    # <= sqrt((n-k) * (1 - s*sum(r_i^2, i<=k)) / s)
    # This is a VALID but NONLINEAR constraint.
    # We can linearize: for fixed sum(r_i^2, i<=k) = Q_k:
    # sum(r_i, i>k) <= sqrt((n-k)*(1-s*Q_k)/s)
    # And sum(r_i, i<=k) + sum(r_i, i>k) <= sum(r_i, i<=k) + sqrt((n-k)*(1-s*Q_k)/s)
    # Optimize over allocation of area between top-k and tail.

    # APPROACH: Parametric bound.
    # For a given split at index k:
    # sum(r_i) = S_k + T_k where S_k = sum(r_i, i<=k), T_k = sum(r_i, i>k)
    # S_k <= ub(k) (geometric bound for k circles)
    # T_k <= sqrt((n-k)*Q_tail/1) where Q_tail = sum(r_i^2, i>k)  [by CS]
    # Also Q_tail <= (1 - s*Q_top) / s where Q_top = sum(r_i^2, i<=k)  [from FT on all n]
    # Hmm, FT says s*(Q_top + Q_tail) <= 1, so Q_tail <= (1-s*Q_top)/s
    # T_k <= sqrt((n-k)*(1-s*Q_top)/(s))
    # S_k <= sqrt(k*Q_top) [by CS on top-k]
    # So: sum(r) = S_k + T_k <= sqrt(k*Q_top) + sqrt((n-k)*(1-s*Q_top)/s)
    # Optimize over Q_top:
    # d/dQ_top [sqrt(k*Q) + sqrt((n-k)*(1-s*Q)/s)] = k/(2*sqrt(k*Q)) - (n-k)/(2*sqrt((n-k)*(1-s*Q)/s)) * 1
    # Wait, let f(Q) = sqrt(k*Q) + sqrt((n-k)/s * (1-s*Q))
    #       = sqrt(k*Q) + sqrt((n-k)/s - (n-k)*Q)
    #       = sqrt(k*Q) + sqrt((n-k)*(1/s - Q))
    # f'(Q) = k/(2*sqrt(k*Q)) - (n-k)/(2*sqrt((n-k)*(1/s-Q)))
    #       = sqrt(k)/(2*sqrt(Q)) - sqrt(n-k)/(2*sqrt(1/s-Q))
    # Set to 0: sqrt(k)/sqrt(Q) = sqrt(n-k)/sqrt(1/s-Q)
    # k*(1/s - Q) = (n-k)*Q
    # k/s = k*Q + (n-k)*Q = n*Q
    # Q* = k/(n*s)
    # f(Q*) = sqrt(k^2/(n*s)) + sqrt((n-k)*(1/s - k/(n*s)))
    #       = k/sqrt(n*s) + sqrt((n-k)*(n-k)/(n*s))
    #       = k/sqrt(n*s) + (n-k)/sqrt(n*s)
    #       = n/sqrt(n*s) = sqrt(n/s)
    # This is just the FT bound! The parametric split doesn't help.

    # The CS inequality is tight when all r_i are equal, which is what FT assumes.
    # To get a TIGHTER bound, we need constraints that break this equality.

    # The geometric constraints (pair bound, top-4 bound) DO break equality for small n.
    # For n=26, the pair bound r_1+r_2 <= 0.5858 is not binding when r_i ~ 0.105.

    # WHAT IF we add MANY more geometric constraints?
    # For k=2..n: the top-k circles must fit in [0,1]^2.
    # The optimal packing of k circles gives sum(r_i, top-k) <= f(k).
    # If we knew f(k) for all k, this would be very constraining.

    # APPROACH: compute f(k) for k=1..26 and add constraints.
    # f(1) = 0.5, f(2) = 0.5858, f(3) ~ 0.7645, f(4) = 1.0, f(5) ~ 1.0854
    # For k > 5, f(k) is only known from optimization, not analytically.
    # We can USE our upper bounds for f(k) (which are valid upper bounds on sum(r_i) for k circles).

    # For this SOCP, we already use known_ub for k=1..4.
    # For k=5..25, use FT as the upper bound on top-k sum.
    for k in range(5, n):
        ft_k = fejes_toth_bound(k)
        constraints += [cp.sum(r[:k]) <= ft_k]

    objective = cp.Maximize(cp.sum(r))
    prob = cp.Problem(objective, constraints)

    try:
        result = prob.solve(solver=cp.SCS, verbose=False, max_iters=50000, eps=1e-8)
        if verbose and r.value is not None:
            rv = r.value
            print(f"  SOCP geometric (n={n}): {result:.6f}")
            print(f"  Top radii: {[f'{x:.4f}' for x in rv[:min(5,n)]]}")
            print(f"  Bottom radii: {[f'{x:.4f}' for x in rv[max(0,n-3):]]}")
        return result
    except Exception as e:
        if verbose:
            print(f"  SOCP error: {e}")
        return None


def counting_bound(n, verbose=False):
    """
    Counting/threshold bound.

    For threshold t > 0, let k = number of circles with r_i >= t.
    Then: sum(r_i) = sum(r_i, r_i >= t) + sum(r_i, r_i < t)
    First part: <= sum of top-k radii
    Second part: < (n-k)*t

    Also: k <= floor(1/(2*sqrt(3)*t^2))  [FT: k circles of radius >= t need area k*2*sqrt(3)*t^2 <= 1]
    And: sum(r_i, r_i >= t) <= FT(k) = sqrt(k/(2*sqrt(3)))

    So: sum(r_i) <= sqrt(k/(2*sqrt(3))) + (n-k)*t
    where k <= 1/(2*sqrt(3)*t^2)

    Substituting k = 1/(2*sqrt(3)*t^2):
    sum(r_i) <= sqrt(1/(2*sqrt(3)*t^2) / (2*sqrt(3))) + (n - 1/(2*sqrt(3)*t^2))*t
    = 1/(2*sqrt(3)*t) + n*t - 1/(2*sqrt(3)*t)
    = n*t

    Hmm, that gives n*t which is trivially large. The issue is that FT(k) = sqrt(k/(2*sqrt(3)))
    is already using CS, and plugging k from FT gives the same thing.

    Let me try: for threshold t, k circles with r >= t:
    k * 2*sqrt(3)*t^2 <= 1 (FT on these k circles alone)
    sum(r_i, r_i >= t) <= some UB depending on k
    sum(r_i, r_i < t) < (n-k)*t

    For the first part, use the TIGHTER bound that accounts for packing geometry.
    For k <= 4, we have tight bounds.

    Let me optimize over integer k:
    """
    s = 2 * np.sqrt(3)

    known_ub = {
        1: 0.5000,
        2: 0.5858,
        3: 0.8787,  # our SOCP (conservative)
        4: 1.0000,
    }

    best = float('inf')
    best_k = -1

    for k in range(1, n):
        # Top-k sum bound
        if k in known_ub:
            top_k_ub = known_ub[k]
        else:
            top_k_ub = fejes_toth_bound(k)

        # r_{k+1} bound (k+1-th largest): (k+1)*s*r_{k+1}^2 <= 1
        r_k1_max = 1.0 / np.sqrt(s * (k+1))

        # Tail: n-k circles each <= r_{k+1}
        tail_ub = (n - k) * r_k1_max

        total = top_k_ub + tail_ub
        if total < best:
            best = total
            best_k = k

    if verbose:
        print(f"  Counting bound (n={n}): {best:.6f} (split at k={best_k})")

    return best


def recursive_bound(n, verbose=False):
    """
    Recursive splitting bound.

    Split n circles into two groups of roughly equal size.
    Each group occupies a "region" of the square.
    For group 1 (n1 circles) in a sub-region of area A1:
      FT: 2*sqrt(3)*sum(r_i^2) <= A1
      CS: sum(r_i) <= sqrt(n1/(2*sqrt(3))) * sqrt(A1)  [FT in sub-region of area A1]

    For group 2 (n2 = n-n1) in remaining area A2 = 1-A1:
      sum(r_i) <= sqrt(n2/(2*sqrt(3))) * sqrt(A2)

    Total: sqrt(n1*A1/(2*sqrt(3))) + sqrt(n2*A2/(2*sqrt(3)))
    = 1/sqrt(2*sqrt(3)) * (sqrt(n1*A1) + sqrt(n2*A2))
    subject to A1 + A2 = 1, n1 + n2 = n

    Optimize: sqrt(n1*A) + sqrt(n2*(1-A)) over A in [0,1] and n1+n2=n.
    By CS: max is achieved at A = n1/n, giving sqrt(n1^2/n) + sqrt(n2^2/n) = (n1+n2)/sqrt(n) = sqrt(n).
    So total = sqrt(n/(2*sqrt(3))) = FT bound.

    Same bound again! Splitting doesn't help when using CS + FT per sub-region.

    To improve: use GEOMETRIC constraints per sub-region (like the square boundary within each region).
    But sub-regions are rectangular, not square, and the boundary effect depends on shape.

    For a RECTANGULAR sub-region of width w and height h:
    FT with Oler: 2*sqrt(3)*sum(r_i^2) <= (w+2r_max)(h+2r_max)*(something)...
    Actually for a w x h rectangle, the Oler bound is:
    n * 2*sqrt(3)*r^2 <= w*h + 2*(w+h)*r + pi*r^2
    For mixed radii: 2*sqrt(3)*sum(r_i^2) <= w*h + 2*(w+h)*r_max + pi*r_max^2
    This gives a WEAKER bound (RHS > w*h for r_max > 0).

    TIGHTER approach for sub-region:
    For a w x h rectangle, the CONTAINMENT constraint means r_i <= min(w/2, h/2).
    This limits the maximum radius per sub-region.
    Also: packing in a thin strip (w << h) forces radii to be small.

    Let's try STRIPS of width w = 1/m:
    Each strip is w x 1 = (1/m) x 1.
    Max radius in strip: r <= w/2 = 1/(2m).
    Assign n_k circles to strip k, sum(n_k) = n.
    Per strip: FT gives sum(r_i^2 in strip k) <= 1/(2*sqrt(3)) * w * 1 = w/(2*sqrt(3))
    NO: FT for a w x 1 rectangle:
    sum(2*sqrt(3)*r_i^2) <= w*1 = w  (area = w)
    sum(r_i^2) <= w / (2*sqrt(3))
    CS: sum(r_i in strip k) <= sqrt(n_k * w / (2*sqrt(3)))
    Total: sum over strips <= sum_k sqrt(n_k * w / (2*sqrt(3)))

    Wait, but circles can extend OUTSIDE their strip. A circle centered in strip k
    with radius r extends r on each side. If r > w/2, it extends into neighboring strips.
    Even if r < w/2, the circle partially enters neighboring strips.

    For an UPPER BOUND, we IGNORE the between-strip interactions (this is a relaxation).
    This means we over-count: a circle in strip k uses area in neighboring strips
    that we don't account for.

    But we also have the CONTAINMENT constraint: circles are contained in [0,1]^2.
    A circle centered at x in strip k has r <= min(x, 1-x). So r <= w for a circle
    near the center of its strip, but can be larger if the strip is near the edge.
    Actually, r <= min(x, 1-x, y, 1-y) <= 0.5 always.

    For the strip approach to help, we need the constraint r <= w/2 or similar.
    But this is NOT valid: a circle centered in a strip can have radius > strip width.
    Its center is in the strip, but the circle extends outside.

    HOWEVER: for the sub-problem within each strip, the NON-OVERLAP constraint
    between circles IN THE SAME STRIP is handled. And a circle of large radius
    in strip k prevents circles in neighboring strips from being too close.

    I'll skip this recursive approach since it reduces to FT.
    """
    return fejes_toth_bound(n)


def combined_socp_bound(n, verbose=False):
    """
    Combined SOCP with ALL known constraints.

    This is our best effort at a tight SOCP bound.
    """
    s = 2 * np.sqrt(3)

    # Known EXACT optimal values (provably optimal or best known lower bounds)
    known_optimal = {
        1: 0.5000, 2: 0.5858, 3: 0.7645, 4: 1.0000, 5: 1.0854,
        10: 1.5911, 26: 2.6360,
    }

    # Upper bounds we've proven (or can prove) for top-k
    # For k=1: r_1 <= 0.5 (trivial containment)
    # For k=2: r_1+r_2 <= 2-sqrt(2) ≈ 0.5858 (diagonal placement)
    # For k=4: r_1+r_2+r_3+r_4 <= 1.0 (corner placement)
    # For k=3: need to compute. Currently our bound is 0.8787.
    # For k >= 5: FT is our best bound on top-k sum.

    # Can we prove a tighter bound for k=3?
    # 3 circles: the optimal arrangement has sum = 0.7645.
    # Can we PROVE sum <= 0.7645 (or something close)?
    # 3 circles: at least one pair is "adjacent" (not diagonal).
    # For adjacent pair: distance = |xi-xj|+|yi-yj| >= ri+rj (L1 lower bound on L2).
    # Hmm, L1 is not a lower bound on L2 in general. L2 >= max(|dx|,|dy|) >= (|dx|+|dy|)/sqrt(2).
    # So dist >= (ri+rj) gives max(|dx|,|dy|) >= (ri+rj)/sqrt(2).
    # For 3 circles: by pigeonhole, at least 2 must be in the same half (left or right).
    # Those 2 have |xi-xj| <= 0.5. So:
    # dist_ij^2 = (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2.
    # (yi-yj)^2 >= (ri+rj)^2 - 0.25.
    # This only constrains when ri+rj > 0.5.
    # For 3 equal circles of r: pairs have sum 2r, need 2r > 0.5 => r > 0.25.
    # If r > 0.25: at least 2 circles in same half need vertical separation >= sqrt(4r^2-0.25).
    # And they need centers in [r, 1-r]^2, so vertical range is 1-2r.
    # Those 2 circles need vertical gap >= sqrt(4r^2-0.25) + 2r (including radii).
    # Hmm wait: vertical distance between centers >= sqrt((2r)^2 - (0.5)^2) = sqrt(4r^2-0.25).
    # But centers in [r, 1-r] in y, so range = 1-2r.
    # Need: sqrt(4r^2-0.25) <= 1-2r. Square: 4r^2-0.25 <= 1-4r+4r^2. So -0.25 <= 1-4r. 4r <= 1.25. r <= 0.3125.
    # For r=0.3: sqrt(0.36-0.25) = sqrt(0.11) = 0.332. Range = 0.4. 0.332 <= 0.4. OK.
    # For r=0.31: sqrt(0.3844-0.25) = sqrt(0.1344) = 0.367. Range = 0.38. 0.367 <= 0.38. OK barely.
    # For r=0.3125: sqrt(0.390625-0.25) = sqrt(0.140625) = 0.375. Range = 0.375. Exactly equal.
    # For r>0.3125: impossible to have 2 circles in same half.
    # So for 3 equal circles: r <= 0.3125 (if 2 in same half).
    # But 3 circles can be in a triangular arrangement avoiding same-half constraint.
    # Max r for equilateral triangle in [0,1]^2: limited by square size.

    # This analysis gives r <= 0.3125 for same-half constraint.
    # 3 * 0.3125 = 0.9375. Our SOCP gives 0.8787. So the same-half argument doesn't help directly.

    # For the SOCP, let me add the constraint that for ANY pair:
    # max distance between centers <= sqrt(2)*(1-ri-rj)
    # (both centers in [ri,1-ri] x [ri,1-ri], max distance = diagonal of shrunk square)
    # Wait: for circles i,j with possibly different radii:
    # xi in [ri, 1-ri], xj in [rj, 1-rj].
    # max |xi-xj| = max(1-ri-rj, 1-ri-rj) = 1-ri-rj (when one is at ri, other at 1-rj or vice versa).
    # max dist = sqrt(2) * (1 - max(ri,rj) - min(ri,rj)) = sqrt(2)*(1-ri-rj).
    # Wait, NOT exactly. xi in [ri, 1-ri], xj in [rj, 1-rj].
    # max(xi-xj) = (1-ri) - rj = 1-ri-rj.
    # max(xj-xi) = (1-rj) - ri = 1-ri-rj. Same.
    # max |xi-xj| = 1-ri-rj.
    # Similarly max |yi-yj| = 1-ri-rj.
    # max dist = sqrt(2*(1-ri-rj)^2) = sqrt(2)*(1-ri-rj).

    # For the NON-OVERLAP constraint: dist >= ri+rj.
    # So: ri+rj <= dist <= sqrt(2)*(1-ri-rj).
    # ri+rj <= sqrt(2)*(1-ri-rj) = sqrt(2) - sqrt(2)*(ri+rj).
    # (1+sqrt(2))*(ri+rj) <= sqrt(2).
    # ri+rj <= sqrt(2)/(1+sqrt(2)) = sqrt(2)*(sqrt(2)-1) = 2-sqrt(2) ≈ 0.5858.
    # This is the PAIR BOUND we already have!

    # What about TRIPLE bound? For 3 circles:
    # All pairwise: ri+rj <= 2-sqrt(2) for each pair.
    # Sum of pairs: 2*(r1+r2+r3) <= 3*(2-sqrt(2)) = 6-3*sqrt(2) ≈ 1.757.
    # r1+r2+r3 <= (6-3*sqrt(2))/2 ≈ 0.879. This matches our SOCP result!
    # But the TRUE optimal for 3 circles is 0.7645, so there's still a gap.

    # The pair bound alone gives the 3-sum bound of 0.879.
    # To improve, we need TRIPLE-specific constraints.
    # E.g., for 3 circles, the distances form a triangle in the plane.
    # The triangle must fit in [0,1]^2.

    # For 3 circles with equal radii r placed at maximum spread:
    # Equilateral triangle with side = 2r. Centers at distance 2r from each other.
    # The equilateral triangle with side 2r has height sqrt(3)*r.
    # Must fit in [r, 1-r]^2 (center containment).
    # Width of triangle = 2r, height = sqrt(3)*r.
    # Need: 2r <= 1-2r => r <= 0.25.
    # Hmm that gives 3*0.25 = 0.75. But optimal is 0.7645!
    # So the equilateral arrangement is not optimal for n=3.

    # The true n=3 optimal uses UNEQUAL radii. The constraint analysis is complex.

    r = cp.Variable(n)
    constraints = []

    constraints += [r >= 0, r <= 0.5]

    # Ordering
    for i in range(n-1):
        constraints += [r[i] >= r[i+1]]

    # FT area
    constraints += [s * cp.sum_squares(r) <= 1]

    # ALL pairwise sums: ri + rj <= 2 - sqrt(2)
    pair_limit = 2 - np.sqrt(2)
    for i in range(min(n, 10)):  # only for top-10 to keep manageable
        for j in range(i+1, min(n, 10)):
            constraints += [r[i] + r[j] <= pair_limit]

    # Top-4 sum
    if n >= 4:
        constraints += [r[0] + r[1] + r[2] + r[3] <= 1.0]

    # Individual radius bounds from FT
    for k in range(1, min(n+1, 30)):
        max_r_k = 1.0 / np.sqrt(s * k)
        if max_r_k < 0.5:
            constraints += [r[k-1] <= max_r_k]

    # Top-k sum bounds from FT
    for k in range(5, n):
        ft_k = fejes_toth_bound(k)
        constraints += [cp.sum(r[:k]) <= ft_k]

    # NEW: constraint on sum of r_i * r_j (cross products)
    # From pairwise: (ri+rj)^2 <= 2*(1-ri-rj)^2 = 2-4(ri+rj)+2(ri+rj)^2
    # (ri+rj)^2 - 2(ri+rj)^2 <= 2-4(ri+rj)
    # -(ri+rj)^2 <= 2-4(ri+rj)
    # Already captured by pair bound.

    objective = cp.Maximize(cp.sum(r))
    prob = cp.Problem(objective, constraints)

    try:
        result = prob.solve(solver=cp.SCS, verbose=False, max_iters=100000, eps=1e-9)
        if verbose:
            print(f"  Combined SOCP (n={n}): {result:.6f}")
            if r.value is not None:
                rv = r.value
                print(f"  Top radii: {[f'{x:.4f}' for x in rv[:min(6,n)]]}")
                if n > 6:
                    print(f"  Bottom radii: {[f'{x:.4f}' for x in rv[n-3:]]}")
                total_sq = sum(x**2 for x in rv)
                print(f"  sum(r^2) = {total_sq:.6f}, FT limit = {1/s:.6f}")
                print(f"  FT utilization: {total_sq * s:.4f}")
        return result
    except Exception as e:
        if verbose:
            print(f"  Error: {e}")
        return None


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

    print("Comprehensive Upper Bounds for Circle Packing (sum of radii)")
    print("=" * 90)
    print(f"{'n':>3} | {'Area':>8} | {'FT':>8} | {'SOCP':>8} | {'Count':>8} | {'BEST':>8} | {'Known':>8} | {'Gap':>8} | {'Gap%':>6}")
    print("-" * 90)

    all_results = {}
    for n_val in n_values:
        a = area_bound(n_val)
        ft = fejes_toth_bound(n_val)
        socp = combined_socp_bound(n_val, verbose=(n_val in [3, 10, 26]))
        cnt = counting_bound(n_val, verbose=False)

        bounds = [a, ft]
        if socp is not None:
            bounds.append(socp)
        if cnt is not None:
            bounds.append(cnt)

        best = min(bounds)
        known = known_best.get(n_val, 0)
        gap = best - known if known else 0
        gap_pct = 100*gap/known if known else 0

        valid = best >= known - 1e-3 if known else True

        print(f"{n_val:3d} | {a:8.4f} | {ft:8.4f} | "
              f"{socp if socp else 0:8.4f} | {cnt:8.4f} | "
              f"{best:8.4f} | {known:8.4f} | {gap:8.4f} | {gap_pct:5.1f}% "
              f"{'OK' if valid else '**FAIL**'}")

        all_results[str(n_val)] = {
            'area': float(a), 'fejes_toth': float(ft),
            'socp_combined': float(socp) if socp else None,
            'counting': float(cnt),
            'best': float(best), 'known': float(known),
            'gap': float(gap), 'gap_pct': float(gap_pct),
        }

    output_path = Path(__file__).parent / "comprehensive_bounds.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
