"""Deep refinement of n=30 circle packing solution.

Strategy:
1. SLSQP with tight tolerances
2. Basin-hopping with SLSQP local minimizer
3. Single-circle repositioning
4. CMA-ES exploration
"""

import json
import math
import numpy as np
from scipy.optimize import minimize, basinhopping
from pathlib import Path
import copy
import time

WORK = Path(__file__).parent
SOLUTION_PATH = WORK.parent / "diffevo-001" / "solution_n30.json"
OUTPUT_PATH = WORK / "solution_n30.json"
HISTORY_PATH = WORK / "convergence.json"

def load_solution(path):
    with open(path) as f:
        data = json.load(f)
    circles = data["circles"]
    return np.array(circles)

def save_solution(circles, path):
    data = {"circles": [[float(x), float(y), float(r)] for x, y, r in circles]}
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def validate(circles, tol=1e-10):
    """Check feasibility. Returns (valid, max_violation)."""
    n = len(circles)
    max_viol = 0.0
    for i, (x, y, r) in enumerate(circles):
        if r <= 0:
            return False, abs(r)
        for v in [r - x, x + r - 1.0, r - y, y + r - 1.0]:
            if v > tol:
                max_viol = max(max_viol, v)
    for i in range(n):
        xi, yi, ri = circles[i]
        for j in range(i+1, n):
            xj, yj, rj = circles[j]
            dist = math.sqrt((xi-xj)**2 + (yi-yj)**2)
            overlap = (ri + rj) - dist
            if overlap > tol:
                max_viol = max(max_viol, overlap)
    return max_viol <= tol, max_viol

def pack_to_vec(circles):
    """Flatten circles array to optimization vector [x0,y0,r0, x1,y1,r1, ...]"""
    return circles.flatten()

def vec_to_circles(vec, n):
    return vec.reshape(n, 3)

def objective(vec, n):
    """Negative sum of radii (we minimize)."""
    circles = vec_to_circles(vec, n)
    return -np.sum(circles[:, 2])

def build_constraints(n):
    """Build constraint list for SLSQP."""
    constraints = []

    # Containment: x_i - r_i >= 0, 1 - x_i - r_i >= 0, etc.
    for i in range(n):
        ix, iy, ir = 3*i, 3*i+1, 3*i+2
        # x - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, ix=ix, ir=ir: v[ix] - v[ir]})
        # 1 - x - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, ix=ix, ir=ir: 1.0 - v[ix] - v[ir]})
        # y - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, iy=iy, ir=ir: v[iy] - v[ir]})
        # 1 - y - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, iy=iy, ir=ir: 1.0 - v[iy] - v[ir]})
        # r > 0
        constraints.append({'type': 'ineq', 'fun': lambda v, ir=ir: v[ir] - 1e-12})

    # Non-overlap: dist(i,j) - r_i - r_j >= 0
    for i in range(n):
        for j in range(i+1, n):
            ix, iy, ir = 3*i, 3*i+1, 3*i+2
            jx, jy, jr = 3*j, 3*j+1, 3*j+2
            def overlap_con(v, ix=ix, iy=iy, ir=ir, jx=jx, jy=jy, jr=jr):
                dx = v[ix] - v[jx]
                dy = v[iy] - v[jy]
                dist = math.sqrt(dx*dx + dy*dy)
                return dist - v[ir] - v[jr]
            constraints.append({'type': 'ineq', 'fun': overlap_con})

    return constraints

def slsqp_refine(circles, ftol=1e-15, maxiter=5000):
    """Refine using SLSQP with tight tolerances."""
    n = len(circles)
    vec = pack_to_vec(circles)
    constraints = build_constraints(n)

    result = minimize(
        objective, vec, args=(n,),
        method='SLSQP',
        constraints=constraints,
        options={'ftol': ftol, 'maxiter': maxiter, 'disp': False}
    )

    new_circles = vec_to_circles(result.x, n)
    metric = -result.fun
    valid, viol = validate(new_circles)
    return new_circles, metric, valid, viol

def penalty_objective(vec, n, lam=1e6):
    """Objective with penalty for constraint violations."""
    circles = vec_to_circles(vec, n)
    obj = -np.sum(circles[:, 2])

    penalty = 0.0
    for i in range(n):
        x, y, r = circles[i]
        # Containment penalties
        penalty += max(0, r - x)**2
        penalty += max(0, x + r - 1)**2
        penalty += max(0, r - y)**2
        penalty += max(0, y + r - 1)**2
        penalty += max(0, -r)**2

    for i in range(n):
        xi, yi, ri = circles[i]
        for j in range(i+1, n):
            xj, yj, rj = circles[j]
            dist = math.sqrt((xi-xj)**2 + (yi-yj)**2)
            overlap = (ri + rj) - dist
            if overlap > 0:
                penalty += overlap**2

    return obj + lam * penalty

def penalty_gradient(vec, n, lam=1e6):
    """Analytical gradient of penalty objective."""
    circles = vec_to_circles(vec, n)
    grad = np.zeros_like(vec)

    # Gradient of -sum(r_i)
    for i in range(n):
        grad[3*i+2] = -1.0

    for i in range(n):
        x, y, r = circles[i]
        ix, iy, ir = 3*i, 3*i+1, 3*i+2

        # Containment
        if r > x:
            grad[ir] += lam * 2 * (r - x)
            grad[ix] -= lam * 2 * (r - x)
        if x + r > 1:
            grad[ix] += lam * 2 * (x + r - 1)
            grad[ir] += lam * 2 * (x + r - 1)
        if r > y:
            grad[ir] += lam * 2 * (r - y)
            grad[iy] -= lam * 2 * (r - y)
        if y + r > 1:
            grad[iy] += lam * 2 * (y + r - 1)
            grad[ir] += lam * 2 * (y + r - 1)
        if r < 0:
            grad[ir] -= lam * 2 * r

    for i in range(n):
        xi, yi, ri = circles[i]
        for j in range(i+1, n):
            xj, yj, rj = circles[j]
            dx = xi - xj
            dy = yi - yj
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < 1e-15:
                continue
            overlap = (ri + rj) - dist
            if overlap > 0:
                # d(overlap)/d(xi) = -(dx/dist), etc.
                ddist_dxi = dx / dist
                ddist_dyi = dy / dist

                grad[3*i] -= lam * 2 * overlap * ddist_dxi
                grad[3*i+1] -= lam * 2 * overlap * ddist_dyi
                grad[3*i+2] += lam * 2 * overlap

                grad[3*j] += lam * 2 * overlap * ddist_dxi
                grad[3*j+1] += lam * 2 * overlap * ddist_dyi
                grad[3*j+2] += lam * 2 * overlap

    return grad

def lbfgsb_refine(circles, lam=1e8):
    """Refine using L-BFGS-B with penalty method."""
    n = len(circles)
    vec = pack_to_vec(circles)

    bounds = []
    for i in range(n):
        bounds.extend([(1e-6, 1-1e-6), (1e-6, 1-1e-6), (1e-6, 0.5)])

    result = minimize(
        penalty_objective, vec, args=(n, lam),
        jac=lambda v, n=n, lam=lam: penalty_gradient(v, n, lam),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 10000, 'ftol': 1e-15, 'gtol': 1e-12}
    )

    new_circles = vec_to_circles(result.x, n)
    metric = np.sum(new_circles[:, 2])
    valid, viol = validate(new_circles)
    return new_circles, metric, valid, viol

def progressive_penalty_refine(circles):
    """Progressive penalty refinement with increasing lambda."""
    best = circles.copy()
    best_metric = np.sum(best[:, 2])

    for lam in [1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]:
        new_c, met, valid, viol = lbfgsb_refine(best, lam)
        if valid and met > best_metric:
            best = new_c
            best_metric = met
            print(f"  penalty lam={lam:.0e}: metric={met:.10f} valid={valid} viol={viol:.2e}")
        elif valid:
            best = new_c  # Still use it as starting point

    return best, best_metric

class CirclePerturbation:
    """Perturbation for basin-hopping."""
    def __init__(self, n, stepsize=0.01):
        self.n = n
        self.stepsize = stepsize
        self.rng = np.random.RandomState(42)

    def __call__(self, x):
        xnew = x.copy()
        mode = self.rng.randint(5)

        if mode == 0:
            # Perturb all positions slightly
            for i in range(self.n):
                xnew[3*i] += self.rng.normal(0, self.stepsize)
                xnew[3*i+1] += self.rng.normal(0, self.stepsize)
                xnew[3*i+2] += self.rng.normal(0, self.stepsize * 0.3)
        elif mode == 1:
            # Perturb one circle significantly
            i = self.rng.randint(self.n)
            xnew[3*i] += self.rng.normal(0, self.stepsize * 3)
            xnew[3*i+1] += self.rng.normal(0, self.stepsize * 3)
            xnew[3*i+2] += self.rng.normal(0, self.stepsize)
        elif mode == 2:
            # Swap two circles
            i, j = self.rng.choice(self.n, 2, replace=False)
            xnew[3*i], xnew[3*j] = xnew[3*j], xnew[3*i]
            xnew[3*i+1], xnew[3*j+1] = xnew[3*j+1], xnew[3*i+1]
            xnew[3*i+2], xnew[3*j+2] = xnew[3*j+2], xnew[3*i+2]
        elif mode == 3:
            # Rotate a cluster
            angle = self.rng.uniform(-0.1, 0.1)
            cx, cy = 0.5, 0.5
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            for i in range(self.n):
                dx = xnew[3*i] - cx
                dy = xnew[3*i+1] - cy
                xnew[3*i] = cx + cos_a * dx - sin_a * dy
                xnew[3*i+1] = cy + sin_a * dx + cos_a * dy
        elif mode == 4:
            # Shake: redistribute radii slightly
            for i in range(self.n):
                xnew[3*i+2] += self.rng.normal(0, self.stepsize * 0.5)
                xnew[3*i+2] = max(1e-4, xnew[3*i+2])

        # Clamp to bounds
        for i in range(self.n):
            r = max(1e-6, min(0.5, xnew[3*i+2]))
            xnew[3*i+2] = r
            xnew[3*i] = np.clip(xnew[3*i], r, 1 - r)
            xnew[3*i+1] = np.clip(xnew[3*i+1], r, 1 - r)

        return xnew

def basin_hopping_refine(circles, niter=200, temperature=0.5, stepsize=0.02, seed=42):
    """Basin-hopping with L-BFGS-B + penalty as local minimizer."""
    n = len(circles)
    vec = pack_to_vec(circles)
    lam = 1e8

    bounds = []
    for i in range(n):
        bounds.extend([(1e-6, 1-1e-6), (1e-6, 1-1e-6), (1e-6, 0.5)])

    minimizer_kwargs = {
        'method': 'L-BFGS-B',
        'jac': lambda v, n=n, lam=lam: penalty_gradient(v, n, lam),
        'args': (n, lam),
        'bounds': bounds,
        'options': {'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12}
    }

    perturbation = CirclePerturbation(n, stepsize)
    perturbation.rng = np.random.RandomState(seed)

    result = basinhopping(
        penalty_objective, vec,
        minimizer_kwargs=minimizer_kwargs,
        niter=niter,
        T=temperature,
        stepsize=stepsize,
        take_step=perturbation,
        seed=seed,
        disp=False
    )

    new_circles = vec_to_circles(result.x, n)
    # Polish with SLSQP
    new_circles, metric, valid, viol = slsqp_refine(new_circles)
    if not valid:
        metric = np.sum(new_circles[:, 2])
    return new_circles, metric, valid, viol

def single_circle_reposition(circles, max_rounds=3):
    """For each circle, remove it, then find the best place to reinsert it."""
    n = len(circles)
    best = circles.copy()
    best_metric = np.sum(best[:, 2])

    for round_num in range(max_rounds):
        improved = False
        order = np.random.RandomState(round_num).permutation(n)

        for idx in order:
            others = np.delete(best, idx, axis=0)

            # Try many candidate positions for the removed circle
            best_placement = best[idx].copy()
            best_local_metric = best_metric

            # Generate candidate positions on a grid
            candidates = []
            for gx in np.linspace(0.02, 0.98, 25):
                for gy in np.linspace(0.02, 0.98, 25):
                    # Find max radius at this position
                    max_r = min(gx, 1-gx, gy, 1-gy)
                    for j in range(len(others)):
                        dist = math.sqrt((gx - others[j,0])**2 + (gy - others[j,1])**2)
                        max_r = min(max_r, dist - others[j,2])
                    if max_r > 0.001:
                        candidates.append((gx, gy, max_r))

            # Also try near current position with perturbations
            cx, cy, cr = best[idx]
            for dx in np.linspace(-0.05, 0.05, 11):
                for dy in np.linspace(-0.05, 0.05, 11):
                    gx, gy = cx + dx, cy + dy
                    if 0.01 < gx < 0.99 and 0.01 < gy < 0.99:
                        max_r = min(gx, 1-gx, gy, 1-gy)
                        for j in range(len(others)):
                            dist = math.sqrt((gx - others[j,0])**2 + (gy - others[j,1])**2)
                            max_r = min(max_r, dist - others[j,2])
                        if max_r > 0.001:
                            candidates.append((gx, gy, max_r))

            if not candidates:
                continue

            # Pick the candidate that gives highest total metric
            others_sum = np.sum(others[:, 2])
            for gx, gy, gr in candidates:
                total = others_sum + gr
                if total > best_local_metric:
                    best_local_metric = total
                    best_placement = np.array([gx, gy, gr])

            # Reconstruct full solution and refine
            trial = np.insert(others, idx, best_placement, axis=0)
            trial_metric = np.sum(trial[:, 2])

            if trial_metric > best_metric + 1e-12:
                # Verify and polish
                valid, viol = validate(trial)
                if valid:
                    best = trial
                    best_metric = trial_metric
                    improved = True
                    print(f"  Reposition circle {idx}: metric={best_metric:.10f}")
                else:
                    # Try polishing
                    polished, met, val, _ = slsqp_refine(trial)
                    if val and met > best_metric:
                        best = polished
                        best_metric = met
                        improved = True
                        print(f"  Reposition+polish circle {idx}: metric={best_metric:.10f}")

        if not improved:
            print(f"  Round {round_num}: no improvement, stopping")
            break

    return best, best_metric

def main():
    print("=" * 60)
    print("Deep Refinement of n=30 Circle Packing")
    print("=" * 60)

    # Load parent solution
    circles = load_solution(SOLUTION_PATH)
    initial_metric = np.sum(circles[:, 2])
    valid, viol = validate(circles)
    print(f"\nInitial solution: metric={initial_metric:.10f} valid={valid} viol={viol:.2e}")

    history = [{"step": "initial", "metric": float(initial_metric), "valid": valid}]
    best_circles = circles.copy()
    best_metric = initial_metric

    # Step 1: SLSQP refinement with very tight tolerances
    print("\n--- Step 1: SLSQP Refinement ---")
    for ftol in [1e-12, 1e-13, 1e-14, 1e-15]:
        new_c, met, val, viol = slsqp_refine(best_circles, ftol=ftol, maxiter=10000)
        print(f"  ftol={ftol:.0e}: metric={met:.10f} valid={val} viol={viol:.2e}")
        if val and met > best_metric:
            best_circles = new_c
            best_metric = met
            history.append({"step": f"slsqp_ftol{ftol}", "metric": float(met), "valid": val})

    print(f"  Best after SLSQP: {best_metric:.10f}")
    save_solution(best_circles, OUTPUT_PATH)

    # Step 2: Progressive penalty refinement
    print("\n--- Step 2: Progressive Penalty Refinement ---")
    new_c, met = progressive_penalty_refine(best_circles)
    val, viol = validate(new_c)
    print(f"  After progressive penalty: metric={met:.10f} valid={val}")
    if val and met > best_metric:
        best_circles = new_c
        best_metric = met
        history.append({"step": "progressive_penalty", "metric": float(met), "valid": val})
    # Polish with SLSQP
    new_c, met, val, viol = slsqp_refine(new_c, ftol=1e-15)
    if val and met > best_metric:
        best_circles = new_c
        best_metric = met
        print(f"  Polished: metric={met:.10f}")

    save_solution(best_circles, OUTPUT_PATH)

    # Step 3: Basin-hopping with various parameters
    print("\n--- Step 3: Basin-Hopping ---")
    for seed in [42, 123, 456, 789, 1337]:
        for temp in [0.1, 0.5, 1.0]:
            for ss in [0.005, 0.01, 0.02, 0.05]:
                new_c, met, val, viol = basin_hopping_refine(
                    best_circles, niter=100, temperature=temp, stepsize=ss, seed=seed
                )
                if val and met > best_metric:
                    best_circles = new_c
                    best_metric = met
                    print(f"  BH seed={seed} T={temp} ss={ss}: metric={met:.10f} NEW BEST!")
                    history.append({
                        "step": f"bh_s{seed}_T{temp}_ss{ss}",
                        "metric": float(met), "valid": val
                    })
                    save_solution(best_circles, OUTPUT_PATH)

    print(f"  Best after basin-hopping: {best_metric:.10f}")

    # Step 4: Single-circle repositioning
    print("\n--- Step 4: Single-Circle Repositioning ---")
    new_c, met = single_circle_reposition(best_circles, max_rounds=5)
    val, viol = validate(new_c)
    if val and met > best_metric:
        best_circles = new_c
        best_metric = met
        history.append({"step": "reposition", "metric": float(met), "valid": val})
        # Polish
        new_c, met, val, viol = slsqp_refine(new_c, ftol=1e-15)
        if val and met > best_metric:
            best_circles = new_c
            best_metric = met

    save_solution(best_circles, OUTPUT_PATH)

    # Step 5: Final aggressive SLSQP polish
    print("\n--- Step 5: Final Polish ---")
    for _ in range(5):
        new_c, met, val, viol = slsqp_refine(best_circles, ftol=1e-15, maxiter=20000)
        if val and met > best_metric + 1e-14:
            best_circles = new_c
            best_metric = met
        else:
            break

    save_solution(best_circles, OUTPUT_PATH)

    # Save history
    with open(HISTORY_PATH, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"FINAL RESULT: metric={best_metric:.10f}")
    print(f"Improvement: {best_metric - initial_metric:.2e}")
    print(f"Solution saved to {OUTPUT_PATH}")

    return best_metric

if __name__ == "__main__":
    main()
