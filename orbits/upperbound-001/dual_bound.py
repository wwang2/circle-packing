"""
Dual upper bound via weight functions.

The idea: find a non-negative weight function w(x,y) on [0,1]^2 such that
for ANY circle of radius r centered at (cx,cy) inside [0,1]^2:
    integral_{circle} w(x,y) dx dy >= r  (or >= f(r) for some f)

Then: sum_i r_i <= sum_i integral_{circle_i} w(x,y) dx dy
                <= integral_{[0,1]^2} w(x,y) dx dy  (since circles don't overlap)

This gives: sum r_i <= integral w.

We want to MINIMIZE integral w subject to the constraint that
integral_C w >= r for every valid circle C of radius r at (cx,cy).

This is an infinite-dimensional LP (dual of the packing problem).

APPROACH: Approximate w with basis functions and solve a finite LP/SDP.

Basis: w(x,y) = sum_k a_k * phi_k(x,y)
Constraint: for every (cx, cy, r) with r <= cx <= 1-r, r <= cy <= 1-r:
    sum_k a_k * integral_{B(cx,cy,r)} phi_k(x,y) dx dy >= r

This must hold for ALL valid (cx, cy, r). Discretize the (cx, cy, r) space.

For polynomial basis phi_k(x,y) = x^a * y^b:
integral over circle of radius r at (cx,cy) of x^a * y^b can be computed
analytically (for small degrees) using polar coordinates.
"""

import numpy as np
from scipy.optimize import linprog
import json
import sys
from pathlib import Path


def circle_integral_monomial(cx, cy, r, a, b):
    """
    Compute integral of x^a * y^b over disk of radius r centered at (cx, cy).

    Uses the fact that integral over disk of radius r at origin:
    integral r*cos(t) + cx)^a * (r*sin(t) + cy)^b * r dr dt

    For general (a,b), we can use numerical integration.
    For a=b=0: integral = pi*r^2.
    """
    # Numerical integration using polar coordinates
    # Use Gauss quadrature
    from scipy import integrate

    def integrand_r_theta(theta, rr):
        xx = cx + rr * np.cos(theta)
        yy = cy + rr * np.sin(theta)
        return xx**a * yy**b * rr  # Jacobian r

    result, _ = integrate.dblquad(
        integrand_r_theta,
        0, r,                    # r from 0 to R
        0, 2*np.pi,             # theta from 0 to 2pi
    )
    return result


def circle_integral_monomial_fast(cx, cy, r, a, b):
    """
    Fast computation of integral of x^a * y^b over disk B(cx,cy,r).

    Using the substitution x = cx + r*u*cos(t), y = cy + r*v*sin(t):
    Integral = r^2 * integral_0^1 integral_0^{2pi} (cx+r*s*cos(t))^a * (cy+r*s*sin(t))^b * s dt ds

    For small a,b, expand using binomial theorem.
    """
    if a == 0 and b == 0:
        return np.pi * r**2

    # Use numerical quadrature on the unit disk
    # Sample points on unit disk using concentric rings
    n_r = 8
    n_t = 16
    total = 0.0
    for ir in range(n_r):
        rr = (ir + 0.5) / n_r  # radius in [0,1]
        w_r = 2 * rr / n_r     # weight (ring area = 2*pi*r*dr, but we handle pi below)
        for it in range(n_t):
            theta = 2 * np.pi * (it + 0.5) / n_t
            xx = cx + r * rr * np.cos(theta)
            yy = cy + r * rr * np.sin(theta)
            total += w_r * xx**a * yy**b / n_t
    return np.pi * r**2 * total  # Wait, need to be more careful.

    # Actually, integral = r^2 * int_0^1 int_0^{2pi} f(cx+r*s*cos(t), cy+r*s*sin(t)) * s dt ds
    # = r^2 * sum over (s,t) quadrature of f(x,y)*s * (1/n_r) * (2*pi/n_t)


def circle_integral_basis(cx, cy, r, basis_type='polynomial', max_degree=2):
    """
    Compute integrals of basis functions over a disk.

    Returns a vector of integrals, one per basis function.
    """
    if basis_type == 'polynomial':
        # Basis: 1, x, y, x^2, xy, y^2, x^3, ...
        integrals = []
        for deg in range(max_degree + 1):
            for a in range(deg, -1, -1):
                b = deg - a
                integrals.append(_circle_integral_poly(cx, cy, r, a, b))
        return np.array(integrals)
    else:
        raise ValueError(f"Unknown basis type: {basis_type}")


def _circle_integral_poly(cx, cy, r, a, b):
    """
    Integral of x^a * y^b over disk B(cx, cy, r).

    Analytical for small degrees using the formula:
    int_{B(0,0,r)} (u+cx)^a * (v+cy)^b du dv

    Expand (u+cx)^a = sum_i C(a,i) cx^(a-i) u^i
           (v+cy)^b = sum_j C(b,j) cy^(b-j) v^j

    int_{B(0,0,r)} u^i * v^j du dv = 0 if i or j is odd.
    For even i=2p, j=2q:
    = r^(2p+2q+2) * pi * (2p-1)!! * (2q-1)!! / ((2p+2q+2) * (2p)!! * (2q)!!)
    Wait, this is getting complex. Let me use numerical quadrature.
    """
    from scipy import integrate

    def integrand(yy, xx):
        return xx**a * yy**b

    # Integration over disk: x in [cx-r, cx+r], y in [cy-sqrt(r^2-(x-cx)^2), ...]
    def y_lower(xx):
        d = r**2 - (xx - cx)**2
        if d < 0:
            return cy
        return cy - np.sqrt(d)

    def y_upper(xx):
        d = r**2 - (xx - cx)**2
        if d < 0:
            return cy
        return cy + np.sqrt(d)

    result, _ = integrate.dblquad(integrand, cx - r, cx + r, y_lower, y_upper,
                                   epsabs=1e-10, epsrel=1e-10)
    return result


def dual_bound_polynomial(n, max_degree=4, n_samples=500, verbose=False):
    """
    Find the best polynomial weight function for bounding sum of radii.

    Minimize integral_{[0,1]^2} w(x,y) dx dy
    subject to:
      w(x,y) >= 0 for all (x,y) in [0,1]^2  (hard to enforce, relax)
      integral_{B(cx,cy,r)} w(x,y) dx dy >= r for all valid (cx,cy,r)

    Discretize: sample valid circles (cx, cy, r) and enforce the constraint
    at the sample points.

    Variables: coefficients a_k of basis functions.
    """
    # Generate basis functions: polynomials up to given degree
    basis = []
    for deg in range(max_degree + 1):
        for a in range(deg, -1, -1):
            b = deg - a
            basis.append((a, b))
    n_basis = len(basis)

    if verbose:
        print(f"Polynomial basis degree {max_degree}: {n_basis} functions")

    # Total integral of each basis function over [0,1]^2
    # integral_0^1 integral_0^1 x^a * y^b dx dy = 1/((a+1)*(b+1))
    c_obj = np.array([1.0 / ((a+1) * (b+1)) for a, b in basis])

    # Sample valid circles
    np.random.seed(42)
    samples = []

    # Uniform sampling of (cx, cy, r)
    for _ in range(n_samples):
        r = np.random.uniform(0.01, 0.5)
        cx = np.random.uniform(r, 1 - r)
        cy = np.random.uniform(r, 1 - r)
        samples.append((cx, cy, r))

    # Also add structured samples: small, medium, large circles at various positions
    for r in [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        for cx in np.linspace(r, 1-r, 8):
            for cy in np.linspace(r, 1-r, 8):
                samples.append((cx, cy, r))

    if verbose:
        print(f"Number of circle samples: {len(samples)}")

    # Compute integrals for each sample
    # A_ij = integral of basis j over sample circle i
    A = np.zeros((len(samples), n_basis))
    b_rhs = np.zeros(len(samples))

    for i, (cx, cy, r) in enumerate(samples):
        for j, (a, b) in enumerate(basis):
            A[i, j] = _circle_integral_poly(cx, cy, r, a, b)
        b_rhs[i] = r

    if verbose:
        print(f"Computing integrals... done")
        print(f"LP size: {A.shape}")

    # Solve LP: min c^T x  s.t. A x >= b, x free (no sign constraint on coefficients)
    # linprog minimizes c^T x with A_ub x <= b_ub
    # We need A x >= b, i.e., -A x <= -b
    result = linprog(c_obj, A_ub=-A, b_ub=-b_rhs, method='highs')

    if result.success:
        bound_per_circle = result.fun  # This is the min integral of w
        # The bound says: for ANY packing of n circles, sum r_i <= n * bound_per_circle
        # Wait, that's wrong. The bound says:
        # sum r_i <= sum integral_{C_i} w <= integral_{square} w = result.fun
        # So the upper bound on sum r_i IS result.fun, regardless of n!
        # That can't be right... unless we scale w.

        # Actually, the LP minimizes integral(w) subject to integral_C w >= r for all C.
        # This means: for any circle of radius r, integral_C w >= r.
        # For n non-overlapping circles: sum r_i <= sum integral_{C_i} w <= integral w.
        # So sum r_i <= integral w = result.fun.
        # This bound is INDEPENDENT of n!

        # For this to be useful, we need result.fun to be finite and meaningful.
        if verbose:
            print(f"  LP status: {result.message}")
            print(f"  Optimal integral(w) = {result.fun:.6f}")
            print(f"  This gives: sum r_i <= {result.fun:.6f} for ANY n")
            print(f"  Coefficients: {result.x}")

        return result.fun, result.x
    else:
        if verbose:
            print(f"  LP failed: {result.message}")
        return None, None


def dual_bound_scaled(n, max_degree=4, n_r_samples=30, n_pos_samples=20, verbose=False):
    """
    A better formulation: find w(x,y) that minimizes integral(w)
    subject to: for each valid circle (cx,cy,r), integral_C w >= r.

    This gives bound: sum r_i <= integral(w) for any packing.

    But we can also parameterize differently:
    For each circle, we need integral_C w >= r.
    The tightest bound comes from the w that gives the smallest integral.

    Key issue: if w is a low-degree polynomial, it may not be non-negative,
    making the argument invalid... unless we use the constraint more carefully.

    Actually, the argument is:
    sum r_i <= sum integral_{C_i} w (by the constraint)
    But sum integral_{C_i} w = integral_{union C_i} w (since circles don't overlap)
                              <= integral_{[0,1]^2} w  (ONLY IF w >= 0!)

    So we NEED w >= 0 on [0,1]^2. This is a polynomial non-negativity constraint.
    We can enforce it approximately by sampling.

    Or: use SOS (sum-of-squares) to enforce w >= 0.
    w(x,y) = sum of squares polynomial => w >= 0 everywhere.
    This is an SDP constraint.

    For now, let me enforce w >= 0 at sample points.
    """
    basis = []
    for deg in range(max_degree + 1):
        for a in range(deg, -1, -1):
            b = deg - a
            basis.append((a, b))
    n_basis = len(basis)

    if verbose:
        print(f"Dual bound with degree-{max_degree} polynomial ({n_basis} terms)")

    c_obj = np.array([1.0 / ((a+1) * (b+1)) for a, b in basis])

    # Sample circles
    samples = []
    r_values = np.linspace(0.005, 0.5, n_r_samples)
    for r in r_values:
        pos_values = np.linspace(r, 1-r, n_pos_samples)
        for cx in pos_values:
            for cy in pos_values:
                samples.append((cx, cy, r))

    if verbose:
        print(f"  Circle samples: {len(samples)}")

    # Non-negativity sample points for w
    nn_samples = []
    nn_grid = np.linspace(0, 1, 25)
    for xx in nn_grid:
        for yy in nn_grid:
            nn_samples.append((xx, yy))

    if verbose:
        print(f"  Non-negativity samples: {len(nn_samples)}")

    # Build constraint matrix
    # Constraint 1: integral_C w >= r for each circle sample
    A_circ = np.zeros((len(samples), n_basis))
    b_circ = np.zeros(len(samples))
    for i, (cx, cy, r) in enumerate(samples):
        for j, (a, b) in enumerate(basis):
            A_circ[i, j] = _circle_integral_poly(cx, cy, r, a, b)
        b_circ[i] = r

    # Constraint 2: w(x,y) >= 0 at sample points
    A_nn = np.zeros((len(nn_samples), n_basis))
    for i, (xx, yy) in enumerate(nn_samples):
        for j, (a, b) in enumerate(basis):
            A_nn[i, j] = xx**a * yy**b

    # Combined: -A x <= -b (flipped for linprog)
    A_ub = np.vstack([-A_circ, -A_nn])
    b_ub = np.concatenate([-b_circ, np.zeros(len(nn_samples))])

    if verbose:
        print(f"  Total LP constraints: {A_ub.shape[0]}")
        print(f"  Computing circle integrals...")

    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, method='highs')

    if result.success:
        if verbose:
            print(f"  LP solved: integral(w) = {result.fun:.6f}")
            print(f"  Upper bound on sum(r_i) for ANY n: {result.fun:.6f}")
            coeffs = result.x
            print(f"  Coefficients:")
            for k, ((a,b), c) in enumerate(zip(basis, coeffs)):
                if abs(c) > 1e-6:
                    print(f"    x^{a}*y^{b}: {c:.6f}")
        return result.fun, result.x
    else:
        if verbose:
            print(f"  LP failed: {result.message}")
        return None, None


def simple_dual_bound(n, verbose=False):
    """
    Simplest dual bound using CONSTANT weight function.

    w(x,y) = C (constant).
    Constraint: for any circle of radius r, integral_C w >= r.
    integral_C w = C * pi * r^2 >= r => C >= 1/(pi*r).

    This must hold for ALL valid r in (0, 0.5].
    The binding constraint is at the SMALLEST r (r -> 0+):
    C >= 1/(pi*r) -> infinity as r -> 0.

    So constant weight doesn't work! The constraint is impossible for r -> 0.

    SOLUTION: the bound should be: integral w >= f(r) where f(r) = r.
    For w = C: C*pi*r^2 >= r => C >= 1/(pi*r).
    This goes to infinity. So any fixed w can't satisfy integral >= r for arbitrarily small r.

    THE FIX: We're maximizing sum of n specific radii. We don't need the
    bound to hold for r -> 0. The radii are bounded below by... well, they
    can be arbitrarily small.

    Actually, the dual approach needs modification. Instead of requiring
    integral_C w >= r for ALL r, we note that in an optimal packing with n circles,
    the smallest radius has some lower bound.

    Alternatively, use a DIFFERENT dual:
    w(x,y) such that for any point set of n non-overlapping circles:
    sum r_i <= integral w.

    This is exactly: for each circle, integral_C w >= r_i.
    If w >= 0 and this holds, then sum r_i <= integral w (since circles don't overlap).

    The issue is that integral_C w >= r fails for small r.
    So the LP is infeasible with this constraint for small r.

    FIX: use w(x,y) that grows near boundaries (to handle boundary effects)
    but also handle the small-r issue by noting that small circles contribute
    little to the sum.

    Actually, a better formulation:

    Use integral_C w >= r only for r >= r_min.
    For r < r_min, we have at most n circles, contributing at most n*r_min.
    So: sum r_i <= integral w + n*r_min.

    We want to minimize integral w + n*r_min over w and r_min.
    This trades off the "cutoff" error with the weight function quality.
    """
    best_bound = float('inf')
    best_rmin = None

    for r_min in [0.01, 0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2]:
        # For constant w: C * pi * r^2 >= r for r >= r_min
        # => C >= 1/(pi*r_min)
        C = 1.0 / (np.pi * r_min)
        integral_w = C  # integral of constant C over [0,1]^2
        bound = integral_w + n * r_min
        if verbose:
            print(f"  r_min={r_min:.2f}: C={C:.4f}, integral={integral_w:.4f}, "
                  f"correction={n*r_min:.4f}, total={bound:.4f}")
        if bound < best_bound:
            best_bound = bound
            best_rmin = r_min

    if verbose:
        print(f"  Best constant-w bound: {best_bound:.4f} (r_min={best_rmin})")

    return best_bound


def optimized_dual_bound(n, max_degree=2, n_r_samples=20, n_pos_samples=10,
                          r_min=0.02, verbose=False):
    """
    Optimized dual bound with polynomial weight function.

    Minimize integral(w) + n * r_min
    subject to:
      w(x,y) >= 0 on [0,1]^2 (at sample points)
      integral_C w >= r for all valid circles with r >= r_min
    """
    basis = []
    for deg in range(max_degree + 1):
        for a in range(deg, -1, -1):
            b = deg - a
            basis.append((a, b))
    n_basis = len(basis)

    c_obj = np.array([1.0 / ((a+1) * (b+1)) for a, b in basis])

    # Sample circles with r >= r_min
    samples = []
    r_values = np.linspace(r_min, 0.5, n_r_samples)
    for r in r_values:
        margin = max(r, 0.001)
        pos_values = np.linspace(margin, 1-margin, n_pos_samples)
        for cx in pos_values:
            for cy in pos_values:
                if cx >= r and cx <= 1-r and cy >= r and cy <= 1-r:
                    samples.append((cx, cy, r))

    # Non-negativity samples
    nn_grid = np.linspace(0, 1, 20)
    nn_samples = [(xx, yy) for xx in nn_grid for yy in nn_grid]

    # Build constraint matrices
    A_circ = np.zeros((len(samples), n_basis))
    b_circ = np.zeros(len(samples))
    for i, (cx, cy, r) in enumerate(samples):
        for j, (a, b) in enumerate(basis):
            A_circ[i, j] = _circle_integral_poly(cx, cy, r, a, b)
        b_circ[i] = r

    A_nn = np.zeros((len(nn_samples), n_basis))
    for i, (xx, yy) in enumerate(nn_samples):
        for j, (a, b) in enumerate(basis):
            A_nn[i, j] = xx**a * yy**b

    A_ub = np.vstack([-A_circ, -A_nn])
    b_ub = np.concatenate([-b_circ, np.zeros(len(nn_samples))])

    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, method='highs')

    if result.success:
        bound = result.fun + n * r_min
        if verbose:
            print(f"  degree={max_degree}, r_min={r_min:.3f}: integral(w)={result.fun:.4f}, "
                  f"bound={bound:.4f}")
        return bound, result.x
    else:
        if verbose:
            print(f"  LP failed (degree={max_degree}, r_min={r_min})")
        return None, None


def search_best_dual_bound(n, verbose=True):
    """Search over parameters for the best dual bound."""
    if verbose:
        print(f"\nDual bound search for n={n}")
        print(f"{'='*50}")

    best_bound = float('inf')
    best_params = None

    # Try constant weight
    cb = simple_dual_bound(n, verbose=verbose)
    if cb < best_bound:
        best_bound = cb
        best_params = "constant"

    # Try polynomial weights with various degrees and r_min
    for degree in [2, 3, 4]:
        for r_min in [0.01, 0.02, 0.05, 0.08, 0.1]:
            try:
                bound, coeffs = optimized_dual_bound(
                    n, max_degree=degree, r_min=r_min,
                    n_r_samples=15, n_pos_samples=8, verbose=verbose
                )
                if bound is not None and bound < best_bound:
                    best_bound = bound
                    best_params = f"poly_deg{degree}_rmin{r_min}"
            except Exception as e:
                if verbose:
                    print(f"  Error with deg={degree}, r_min={r_min}: {e}")

    if verbose:
        print(f"\nBest dual bound for n={n}: {best_bound:.6f} ({best_params})")
        print(f"Area bound: {np.sqrt(n/np.pi):.6f}")
        print(f"Fejes Toth: {np.sqrt(n/(2*np.sqrt(3))):.6f}")

    return best_bound


if __name__ == "__main__":
    if len(sys.argv) > 1:
        n_values = [int(x) for x in sys.argv[1:]]
    else:
        n_values = [4, 10, 26]

    known_best = {
        1: 0.5000, 2: 0.5858, 3: 0.7645, 4: 1.0000, 5: 1.0854,
        10: 1.5911, 15: 2.0365, 20: 2.3010, 26: 2.6360,
    }

    for n in n_values:
        bound = search_best_dual_bound(n, verbose=True)
        if n in known_best:
            gap = bound - known_best[n]
            print(f"Gap from best known: {gap:.4f} ({100*gap/known_best[n]:.2f}%)")
