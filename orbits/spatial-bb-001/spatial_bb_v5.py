"""
Spatial Branch-and-Bound v5: Cutting-plane LP with selective McCormick.

Key lessons from v1-v4:
- Full McCormick (v1): 1131 vars, root LP = 13.0 on full box -- too loose
- Distance bounds (v2,v3): INVALID relaxation because d_min(i,j) constraint
  r_i+r_j <= d_min couples to specific center positions, which is wrong when
  centers are free variables
- Full-variable LP with FT (v4): root LP = FT = 2.7396, no pruning because
  branching one variable at a time doesn't help -- the FT bound doesn't depend
  on center positions

The fundamental difficulty:
  FT bound = 2.7396 holds REGARDLESS of how you partition the center space,
  because the FT constraint only depends on radii. To beat FT, you need
  non-overlap constraints, but those involve quadratic terms in centers.

New approach: SELECTIVE McCormick cuts.

Instead of introducing auxiliary variables for ALL quadratic terms, we:
1. Start with the FT LP (radii only, 26 vars)
2. Find the LP-optimal radii
3. Check: are these radii achievable by some center placement?
4. If not: find the most violated non-overlap pair and add a McCormick
   linearization of that specific non-overlap constraint
5. Re-solve and repeat

The McCormick cut for a specific pair (i,j) on a box adds linear constraints
that relate the center positions to the radii. This is a cutting-plane approach
that lazily adds only the constraints that matter.

BUT WAIT -- the key insight is that the FT bound is TIGHT for equal radii.
The LP solution will have near-equal radii. The non-overlap constraints don't
help unless we can show that the LP radii CANNOT be packed.

For the LP radii to be packable, they must satisfy:
  There exist (x_i, y_i) in [r_i, 1-r_i]^2 s.t. ||(x_i,y_i)-(x_j,y_j)|| >= r_i+r_j

This is a feasibility problem. If the radii are too large to pack, that
proves the LP bound is not achievable.

Strategy: use the PACKING FEASIBILITY as a separation oracle.
- Solve FT LP -> get r*
- Check if r* can be packed (this is the original packing problem!)
- If yes: the UB = sum(r*) is achievable -> tightening impossible with just FT
- If no: use the infeasibility proof to generate a cutting plane

For n=26 with equal radii r*=0.1054, the packing is definitely feasible
(hex packing fits). So FT alone cannot be tightened by adding non-overlap
constraints at the root -- we confirmed FT IS achievable for equal radii.

This means: to beat FT, we MUST branch. But branching on what?

THE REAL APPROACH: Branch on RADII, not centers.

Branch on r_i to force non-equal radii distributions. On a box where
some radii are forced to be large and others small, the FT bound is
looser but the PACKING constraints become tighter.

Wait -- actually, the FT bound already gives the tightest result for
equal radii. Making radii unequal (as the actual solution does) makes
sum(r) SMALLER. So the maximum of sum(r) subject to FT is at equal radii.
The non-overlap constraints force the actual solution to be non-equal.

So the strategy should be:
1. The FT bound IS the root LP bound (2.7396) and it's NOT achievable
   because you can't pack 26 circles of radius 0.1054 in [0,1]^2.
   Actually -- CAN you? Let's check.

2. If you CAN pack them: FT is tight and we can't improve. The only way
   to get a better UB is via a non-FT argument.

3. If you CANNOT pack them: the maximum packable equal-radius configuration
   gives a tighter UB.

Let's check option 2: can 26 circles of radius 0.1054 fit in [0,1]^2?
"""

import numpy as np
import json
import time
import os
from scipy.optimize import linprog, minimize
import pickle


N = 26
INCUMBENT = 2.6359830865


def load_incumbent():
    sol_path = "/Users/wujiewang/code/circle-packing/research/solutions/mobius-001/solution_n26.json"
    with open(sol_path) as f:
        data = json.load(f)
    circles = np.array(data["circles"])
    return circles[:, 0], circles[:, 1], circles[:, 2]


def can_pack_equal_radii(n, r, max_attempts=20):
    """
    Check if n circles of radius r can be packed in [0,1]^2.

    Uses random initialization + gradient descent on overlap penalty.
    If any attempt succeeds with zero overlap, returns True.
    """
    box_lo = r
    box_hi = 1.0 - r

    if box_lo >= box_hi:
        return False, None

    best_violation = float('inf')
    best_centers = None

    for attempt in range(max_attempts):
        # Random initial centers in [r, 1-r]^2
        np.random.seed(attempt * 137 + 42)
        centers = np.random.uniform(box_lo, box_hi, size=(n, 2))

        # Optimize: minimize maximum overlap
        def penalty(flat_centers):
            c = flat_centers.reshape(n, 2)
            pen = 0.0
            for i in range(n):
                for j in range(i+1, n):
                    d = np.sqrt(np.sum((c[i] - c[j])**2))
                    overlap = 2*r - d
                    if overlap > 0:
                        pen += overlap**2
                # Boundary
                for k in range(2):
                    if c[i, k] < r:
                        pen += (r - c[i, k])**2
                    if c[i, k] > 1 - r:
                        pen += (c[i, k] - (1-r))**2
            return pen

        result = minimize(penalty, centers.flatten(), method='L-BFGS-B',
                         bounds=[(box_lo, box_hi)] * (2*n),
                         options={'maxiter': 2000, 'ftol': 1e-15})

        violation = result.fun
        if violation < best_violation:
            best_violation = violation
            best_centers = result.x.reshape(n, 2)

        if violation < 1e-12:
            return True, result.x.reshape(n, 2)

    return best_violation < 1e-8, best_centers


def max_equal_radius_packable(n, tol=1e-6):
    """
    Binary search for the maximum radius r such that n circles of radius r
    can be packed in [0,1]^2.

    If max_r * n > FT_ub, then FT is tight.
    If max_r * n < FT_ub, we get a tighter bound than FT.
    """
    lo, hi = 0.0, 0.5

    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        packable, _ = can_pack_equal_radii(n, mid, max_attempts=10)
        if packable:
            lo = mid
        else:
            hi = mid

    return lo


def compute_improved_ub():
    """
    Try to improve the UB by combining FT with packing feasibility arguments.

    Approach: the FT bound assumes equal radii r* = sqrt(1/(n*2*sqrt(3))).
    If these radii CAN be packed, FT is tight and we can't improve.
    If they CANNOT be packed, we binary search for the maximum packable
    equal radius, giving UB = n * r_max.
    """
    n = N
    coeff_ft = 2.0 * np.sqrt(3.0)
    r_ft = np.sqrt(1.0 / (n * coeff_ft))
    ft_ub = n * r_ft

    print(f"FT optimal equal radius: r* = {r_ft:.6f}")
    print(f"FT UB: {ft_ub:.6f}")
    print(f"Can 26 circles of r={r_ft:.6f} fit in [0,1]^2?")

    packable, centers = can_pack_equal_radii(n, r_ft, max_attempts=50)
    print(f"  Result: {'YES' if packable else 'NO'}")

    if packable:
        print(f"\nFT is tight: 26 circles of radius {r_ft:.6f} fit in the unit square.")
        print(f"This means the FT bound cannot be improved by non-overlap constraints")
        print(f"for equal-radius configurations.")

        # But the ACTUAL optimum has non-equal radii.
        # Can we use the FT + inequality structure to get a tighter bound?
        # The answer is: not easily, because FT already gives the max of
        # sum(r) subject to sum(c*r^2) <= 1, and the optimizer chooses equal radii.

        # However: if we add the constraint that the radii must be PACKABLE,
        # the optimum might shift. The packability constraint is:
        #   There exist valid centers for these radii.
        # This is a complex feasibility condition.

        # One approach: for a given set of radii, what's the maximum sum?
        # This is the original problem! So the "UB tightening" reduces to
        # solving the original problem, which is circular.

        # Alternative: use SPECIFIC packability tests.
        # E.g., for the top-k largest circles, check if they fit together.

        return ft_ub, 'ft_is_tight'

    else:
        # FT is NOT tight: the equal-radius packing doesn't fit.
        # Find the max packable equal radius.
        print(f"\nFT is NOT tight. Binary searching for max packable equal radius...")
        r_max = max_equal_radius_packable(n, tol=1e-5)
        ub_equal = n * r_max
        print(f"Max packable equal radius: {r_max:.6f}")
        print(f"UB from equal-radius packing: {ub_equal:.6f}")

        return ub_equal, 'equal_radius_ub'


def lp_ub_with_packing_cuts(n=N):
    """
    Compute UB by combining FT with packing-theoretic constraints.

    Key constraints beyond FT:
    1. r_i + r_j <= pair_bound for ALL pairs
    2. For any k circles: sum of k largest radii <= UB_k
       where UB_k is the max sum(r) for k circles in [0,1]^2
    3. The sum of any sqrt(n) largest radii is bounded by
       the area they need

    These are ADDITIONAL valid inequalities that may tighten FT.
    """
    coeff_ft = 2.0 * np.sqrt(3.0)
    r_inc_x, r_inc_y, r_inc = load_incumbent()

    # LP: max sum(r_i), r sorted in decreasing order (WLOG by relabeling)
    # We use the trick: we can add ordering constraints since the problem
    # is symmetric in circle labels.

    # Actually, we CANNOT add ordering constraints without also ordering
    # the center positions. The problem is that the circles are distinguishable
    # by their positions. So we skip ordering.

    # Approach: add packing-theoretic cuts to the FT LP.

    c = -np.ones(n)

    rows = []
    rhs_list = []

    # FT tangent cuts at multiple points
    for r_ref in [r_inc, np.full(n, 0.1054), np.full(n, 0.05), np.full(n, 0.08),
                  np.full(n, 0.12), np.full(n, 0.15), np.full(n, 0.20)]:
        row = np.zeros(n)
        for i in range(n):
            row[i] = 2.0 * coeff_ft * r_ref[i]
        rows.append(row)
        rhs_list.append(1.0 + coeff_ft * np.sum(r_ref**2))

    # Pair bound: r_i + r_j <= 2 - sqrt(2)
    pair_bound = 2.0 - np.sqrt(2.0)
    for i in range(n):
        for j in range(i+1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            rows.append(row)
            rhs_list.append(pair_bound)

    # Individual radius bound: r_i <= 0.5
    # Already in variable bounds

    # Strengthened individual bound: in a packing of n circles,
    # the largest circle has r_1 <= ... (depends on n)
    # For n >= 5, one can show r_max < 1/(2*sqrt(3)) from density arguments
    # but this is similar to FT applied to a single circle.

    A_ub = np.array(rows)
    b_ub = np.array(rhs_list)

    bounds = [(0.0, 0.5)] * n

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if result.success:
        return -result.fun, result.x
    return None, None


def triple_packing_bound():
    """
    Compute a tighter UB using TRIPLE packing constraints.

    For any three circles i, j, k:
    They must fit in [0,1]^2 with pairwise non-overlap.
    This gives: r_i + r_j + r_k <= UB_3(r_i, r_j, r_k)

    For equal radii: 3 circles of radius r fit in [0,1]^2 iff
    r <= ... (geometric condition).

    The tightest version: for any triple, the three circles need
    a CONNECTED region of the square. The sum r_i + r_j + r_k
    is bounded by the triple's geometry.

    Simpler version: use the PAIR bound iteratively.
    For triple (i,j,k):
      r_i + r_j <= d_max(2 circles in [0,1]^2) = 2 - sqrt(2)
      r_j + r_k <= 2 - sqrt(2)
      r_i + r_k <= 2 - sqrt(2)
    Adding all three: 2*(r_i + r_j + r_k) <= 3*(2 - sqrt(2))
    So: r_i + r_j + r_k <= 1.5*(2 - sqrt(2)) = 3 - 1.5*sqrt(2) ~ 0.879

    But this is just the sum of pair bounds divided by 2, which the LP
    already captures from the pair constraints.
    """
    n = N

    # We can get a tighter triple bound from actual packing:
    # max r_1 + r_2 + r_3 for 3 circles in [0,1]^2 with equal radii
    # is 3 * r_max_3 where r_max_3 = max_equal_radius_packable(3).
    # But with unequal radii, it's the LP value.

    # The LP with pair bounds already gives:
    # max sum(r_i) s.t. r_i + r_j <= 0.586 for all pairs, r_i in [0, 0.5]
    # For n=26: this is a linear program
    c = -np.ones(n)

    rows = []
    rhs_list = []

    pair_bound = 2.0 - np.sqrt(2.0)  # ~0.5858
    for i in range(n):
        for j in range(i+1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            rows.append(row)
            rhs_list.append(pair_bound)

    A_ub = np.array(rows)
    b_ub = np.array(rhs_list)
    bounds = [(0.0, 0.5)] * n

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if result.success:
        print(f"LP with only pair bounds: UB = {-result.fun:.6f}")
        print(f"  Radii: {result.x[:5]}...")
    return -result.fun if result.success else None


def strengthen_with_k_clique_bounds():
    """
    Use k-circle packing bounds to strengthen the LP.

    For k circles in [0,1]^2, the maximum sum(r_i) is bounded.
    The EXACT bound for small k is computable:
    - k=1: sum=0.5
    - k=2: sum = 2-sqrt(2) ~ 0.586
    - k=3: sum ~ 0.765 (from known 3-circle packing)
    - k=4: sum = 1.0 (four corners)
    - k=5: sum ~ 1.085

    We can add: for any subset S of size k, sum_{i in S} r_i <= UB_k

    For k=2 (pair bound), we already have this.
    For k=3,4,5: we can add "k-clique" constraints.

    But there are C(26, k) such subsets, which is huge.
    We only add the ones that are tight at the FT solution (equal radii).

    At equal radii r*=0.1054, any 5 circles sum to 0.527 which is below
    the k=5 bound of ~1.085. So these constraints won't bite!

    The FT optimum at equal radii has sum(r)=2.74 with r=0.1054.
    The pair bound 0.586 >> 2*0.1054=0.2108. So pair bounds are slack.
    ALL the k-clique bounds are slack at the FT optimum.

    This proves: the only constraint that binds at the FT optimum is FT itself.
    No packing-theoretic constraint can tighten FT at the root.
    """
    pass


def voronoi_based_ub():
    """
    Try to compute an improved UB using Voronoi-cell based area arguments.

    Fejes-Toth's bound comes from: each circle of radius r occupies area
    at least 2*sqrt(3)*r^2 (the area of the hexagonal Voronoi cell).
    Total area <= 1 gives sum(2*sqrt(3)*r_i^2) <= 1.

    Improvement idea: boundary circles have LARGER Voronoi cells than
    the hex-packing minimum, because some of their Voronoi region extends
    outside the square. This "boundary waste" means they actually occupy
    more than 2*sqrt(3)*r^2 of the square's area.

    For a circle at position (x, y) with radius r, the portion of its
    Voronoi cell inside [0,1]^2 is at least:
    - Interior: 2*sqrt(3)*r^2 (standard hex cell)
    - Near wall: larger, because the Voronoi cell is truncated by the wall

    The Kepler/FT boundary correction is:
    For a circle touching one wall (say x=r):
    Its Voronoi cell includes all points closer to it than to any other circle
    AND closer to it than to the wall reflection. The reflection trick means
    the cell is at least a HALF of 2*sqrt(3)*r^2... no, this goes the wrong way.

    Actually, the boundary HELPS packing (you can push circles against walls),
    so the boundary correction LOOSENS the bound, not tightens it.
    This was already discovered in upperbound-001.

    The Oler-Groemer correction: area >= sum(2*sqrt(3)*r_i^2) - 4*sqrt(3)*sum(r_i)*???
    This gives a WEAKER (higher) UB. So boundary corrections don't help.

    Conclusion: Voronoi-based improvements to FT go the WRONG way for UB.
    """
    pass


def compute_non_overlap_infeasibility_certificate():
    """
    Instead of trying to compute an improved UB from above, let's see if
    we can characterize WHY the FT bound is 2.7396 and the actual optimum
    is only 2.636.

    The gap is 0.104, or 3.9%. Where does this gap come from?

    The FT optimum is equal radii r*=0.1054. The sum is 26*0.1054 = 2.7396.
    The actual optimum has non-equal radii summing to 2.636.

    Question: CAN 26 circles of r=0.1054 be packed in [0,1]^2?

    Effective side of the containment square: 1 - 2*0.1054 = 0.7893
    This is the space available for centers: [0.1054, 0.8946]^2

    Centers must be pairwise separated by >= 2*0.1054 = 0.2108.

    This is equivalent to: can you place 26 points in [0, 0.7893]^2 with
    pairwise distance >= 0.2108?

    Divide the box into a grid: 0.7893 / 0.2108 = 3.74 rows/columns.
    A 4x4 grid gives 16 positions with separation 0.7893/3 = 0.263 > 0.2108. OK.
    A 5x5 grid gives 25 positions with separation 0.7893/4 = 0.197 < 0.2108. Not enough!
    A 4x7 hex grid? Let's check...

    For hex packing: row spacing = sqrt(3)/2 * d = sqrt(3)/2 * 0.2108 = 0.1822
    Rows that fit: floor(0.7893 / 0.1822) + 1 = 5 rows
    Columns in each row: floor(0.7893 / 0.2108) + 1 = 4 or 5 (alternating)
    Total: ~5*4 + some offsets = ~22-23 circles

    Hmm, that might not be enough. Let me compute more carefully.
    """
    n = N
    coeff_ft = 2.0 * np.sqrt(3.0)
    r_ft = np.sqrt(1.0 / (n * coeff_ft))
    d_min = 2 * r_ft  # minimum distance between centers

    L = 1.0 - 2 * r_ft  # effective side length for centers

    print(f"Equal-radius analysis:")
    print(f"  r* = {r_ft:.6f}")
    print(f"  d_min = {d_min:.6f}")
    print(f"  L (center box side) = {L:.6f}")
    print(f"  Grid: {L/d_min:.2f} circles per row/column")

    # Hex packing count
    row_spacing = np.sqrt(3) / 2 * d_min
    n_rows = int(np.floor(L / row_spacing)) + 1
    n_cols_even = int(np.floor(L / d_min)) + 1
    n_cols_odd = int(np.floor((L - d_min/2) / d_min)) + 1

    total_hex = 0
    for row in range(n_rows):
        if row % 2 == 0:
            total_hex += n_cols_even
        else:
            total_hex += n_cols_odd

    print(f"  Hex packing: {n_rows} rows, ~{n_cols_even}/{n_cols_odd} cols, total={total_hex}")

    # Check if 26 equal circles CAN fit
    print(f"\n  Can {n} circles of r={r_ft:.6f} fit?")
    packable, centers = can_pack_equal_radii(n, r_ft, max_attempts=100)
    print(f"  Result: {'YES' if packable else 'NO (or could not find arrangement)'}")

    if not packable:
        # Binary search for the max n that fits with radius r_ft
        print(f"\n  Binary search: max n for r={r_ft:.6f}:")
        for test_n in [20, 22, 24, 25, 26, 27, 28]:
            ok, _ = can_pack_equal_radii(test_n, r_ft, max_attempts=30)
            print(f"    n={test_n}: {'fits' if ok else 'does NOT fit'}")

    # Also: binary search for max equal radius with n=26
    print(f"\n  Binary search: max r for n={n}:")
    r_max = max_equal_radius_packable(n, tol=1e-5)
    print(f"  r_max = {r_max:.6f}")
    print(f"  UB from equal-r: {n * r_max:.6f}")
    print(f"  Compare: FT = {n * r_ft:.6f}, Incumbent = {INCUMBENT:.10f}")

    return r_max, n * r_max


def run_full_analysis():
    """Run comprehensive analysis."""
    print("=" * 70)
    print("SPATIAL B&B v5: CUTTING-PLANE + PACKING FEASIBILITY ANALYSIS")
    print("=" * 70)

    n = N
    coeff_ft = 2.0 * np.sqrt(3.0)
    r_ft = np.sqrt(1.0 / (n * coeff_ft))
    ft_ub = n * r_ft

    print(f"\nReference bounds:")
    print(f"  FT UB:     {ft_ub:.6f}")
    print(f"  Incumbent: {INCUMBENT:.10f}")
    print(f"  Gap:       {(ft_ub - INCUMBENT):.6f} ({(ft_ub/INCUMBENT-1)*100:.2f}%)")

    # Phase 1: LP with pair bounds only
    print(f"\n--- Phase 1: LP with pair bounds ---")
    pb_ub = triple_packing_bound()

    # Phase 2: LP with FT + pair bounds
    print(f"\n--- Phase 2: LP with FT + pair bounds ---")
    fp_ub, fp_sol = lp_ub_with_packing_cuts()
    if fp_ub:
        print(f"UB = {fp_ub:.6f}")

    # Phase 3: Equal-radius feasibility analysis
    print(f"\n--- Phase 3: Equal-radius feasibility ---")
    r_max, eq_ub = compute_non_overlap_infeasibility_certificate()

    # Phase 4: UB improvement via packing feasibility
    print(f"\n--- Phase 4: Packing-based UB ---")
    ub_final, method = compute_improved_ub()
    print(f"\nBest UB achieved: {ub_final:.6f} (method: {method})")

    return {
        'ft_ub': ft_ub,
        'pair_bound_ub': pb_ub,
        'ft_pair_ub': fp_ub,
        'equal_r_max': r_max,
        'equal_r_ub': eq_ub,
        'best_ub': ub_final,
        'incumbent': INCUMBENT,
    }


if __name__ == '__main__':
    results = run_full_analysis()

    # Save
    results_path = os.path.join(os.path.dirname(__file__), 'results_v5.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"  FT UB:           {results['ft_ub']:.6f}")
    print(f"  Pair bound UB:   {results['pair_bound_ub']:.6f}")
    print(f"  FT+pair UB:      {results['ft_pair_ub']:.6f}")
    print(f"  Equal-r UB:      {results['equal_r_ub']:.6f}")
    print(f"  Best UB:         {results['best_ub']:.6f}")
    print(f"  Incumbent:       {results['incumbent']:.10f}")
    print(f"{'='*70}")
