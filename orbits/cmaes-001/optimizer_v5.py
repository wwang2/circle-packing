"""CMA-ES v5: High-precision polish with multiple optimizers.

Try different scipy optimizers and formulations to squeeze out more precision.
Also try augmented Lagrangian and trust-constr methods.
"""

import json
import math
import os
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

WORKDIR = os.path.dirname(os.path.abspath(__file__))

def log(msg):
    print(msg, flush=True)

def load_solution(path):
    with open(path) as f:
        data = json.load(f)
    circles = data.get("circles", data)
    return np.array(circles)

def save_solution(circles, path):
    data = {"circles": [[float(x), float(y), float(r)] for x, y, r in circles]}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def compute_violation(circles):
    n = len(circles)
    total_viol = 0.0
    max_viol = 0.0
    for i in range(n):
        x, y, r = circles[i]
        for v in [r - x, x + r - 1.0, r - y, y + r - 1.0, -r]:
            if v > 0:
                total_viol += v
                max_viol = max(max_viol, v)
    for i in range(n):
        xi, yi, ri = circles[i]
        for j in range(i + 1, n):
            xj, yj, rj = circles[j]
            dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)
            overlap = (ri + rj) - dist
            if overlap > 0:
                total_viol += overlap
                max_viol = max(max_viol, overlap)
    return total_viol, max_viol

def slsqp_polish(circles, maxiter=10000, ftol=1e-16):
    n = len(circles)
    x0 = circles.flatten()
    def neg_sum_radii(x):
        return -np.sum(x.reshape(n, 3)[:, 2])
    constraints = []
    for i in range(n):
        ri = 3 * i + 2; xi = 3 * i; yi = 3 * i + 1
        constraints.append({"type": "ineq", "fun": lambda x, idx=ri: x[idx] - 1e-12})
        constraints.append({"type": "ineq", "fun": lambda x, ix=xi, ir=ri: x[ix] - x[ir]})
        constraints.append({"type": "ineq", "fun": lambda x, ix=xi, ir=ri: 1.0 - x[ix] - x[ir]})
        constraints.append({"type": "ineq", "fun": lambda x, iy=yi, ir=ri: x[iy] - x[ir]})
        constraints.append({"type": "ineq", "fun": lambda x, iy=yi, ir=ri: 1.0 - x[iy] - x[ir]})
    for i in range(n):
        for j in range(i + 1, n):
            def overlap_con(x, ii=i, jj=j):
                xi, yi, ri = x[3*ii], x[3*ii+1], x[3*ii+2]
                xj, yj, rj = x[3*jj], x[3*jj+1], x[3*jj+2]
                return math.sqrt((xi-xj)**2 + (yi-yj)**2) - ri - rj
            constraints.append({"type": "ineq", "fun": overlap_con})
    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])
    res = minimize(neg_sum_radii, x0, method="SLSQP", constraints=constraints,
                   bounds=bounds, options={"maxiter": maxiter, "ftol": ftol, "disp": False})
    result_circles = res.x.reshape(n, 3)
    viol, max_v = compute_violation(result_circles)
    sr = np.sum(result_circles[:, 2])
    return result_circles, sr, viol, max_v

def trust_constr_polish(circles, maxiter=10000):
    """Polish with trust-constr (interior point) method."""
    n = len(circles)
    x0 = circles.flatten()

    def neg_sum_radii(x):
        return -np.sum(x.reshape(n, 3)[:, 2])

    def neg_sum_radii_grad(x):
        g = np.zeros_like(x)
        for i in range(n):
            g[3*i + 2] = -1.0
        return g

    # Build constraint function: all constraints >= 0
    n_pairs = n * (n - 1) // 2
    n_contain = 5 * n  # r>0, x-r>=0, 1-x-r>=0, y-r>=0, 1-y-r>=0
    n_con = n_contain + n_pairs

    def all_constraints(x):
        c = x.reshape(n, 3)
        result = np.zeros(n_con)
        idx = 0
        for i in range(n):
            xi, yi, ri = c[i]
            result[idx] = ri - 1e-12; idx += 1
            result[idx] = xi - ri; idx += 1
            result[idx] = 1.0 - xi - ri; idx += 1
            result[idx] = yi - ri; idx += 1
            result[idx] = 1.0 - yi - ri; idx += 1
        for i in range(n):
            xi, yi, ri = c[i]
            for j in range(i + 1, n):
                xj, yj, rj = c[j]
                dist = math.sqrt((xi-xj)**2 + (yi-yj)**2)
                result[idx] = dist - ri - rj; idx += 1
        return result

    def all_constraints_jac(x):
        c = x.reshape(n, 3)
        jac = np.zeros((n_con, 3*n))
        idx = 0
        for i in range(n):
            jac[idx, 3*i+2] = 1.0; idx += 1
            jac[idx, 3*i] = 1.0; jac[idx, 3*i+2] = -1.0; idx += 1
            jac[idx, 3*i] = -1.0; jac[idx, 3*i+2] = -1.0; idx += 1
            jac[idx, 3*i+1] = 1.0; jac[idx, 3*i+2] = -1.0; idx += 1
            jac[idx, 3*i+1] = -1.0; jac[idx, 3*i+2] = -1.0; idx += 1
        for i in range(n):
            xi, yi, ri = c[i]
            for j in range(i + 1, n):
                xj, yj, rj = c[j]
                dx = xi - xj; dy = yi - yj
                dist = math.sqrt(dx**2 + dy**2)
                if dist > 1e-15:
                    jac[idx, 3*i] = dx / dist
                    jac[idx, 3*i+1] = dy / dist
                    jac[idx, 3*i+2] = -1.0
                    jac[idx, 3*j] = -dx / dist
                    jac[idx, 3*j+1] = -dy / dist
                    jac[idx, 3*j+2] = -1.0
                idx += 1
        return jac

    con = NonlinearConstraint(all_constraints, 0.0, np.inf, jac=all_constraints_jac)

    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])

    try:
        res = minimize(neg_sum_radii, x0, method="trust-constr",
                      jac=neg_sum_radii_grad,
                      constraints=con, bounds=bounds,
                      options={"maxiter": maxiter, "gtol": 1e-15, "verbose": 0})
        result_circles = res.x.reshape(n, 3)
        viol, max_v = compute_violation(result_circles)
        sr = np.sum(result_circles[:, 2])
        return result_circles, sr, viol, max_v
    except Exception as e:
        log(f"  trust-constr failed: {e}")
        return circles, np.sum(circles[:, 2]), *compute_violation(circles)

def augmented_lagrangian(circles, n_outer=20, maxiter_inner=2000):
    """Augmented Lagrangian method for higher precision."""
    n = len(circles)
    x = circles.flatten().copy()

    n_pairs = n * (n - 1) // 2
    n_contain = 5 * n
    n_con = n_contain + n_pairs

    # Initialize multipliers and penalty
    lam = np.zeros(n_con)
    mu = 10.0

    for outer in range(n_outer):
        def auglag_obj(x_flat):
            c = x_flat.reshape(n, 3)
            obj = -np.sum(c[:, 2])

            # Evaluate constraints
            cons = np.zeros(n_con)
            idx = 0
            for i in range(n):
                xi, yi, ri = c[i]
                cons[idx] = ri - 1e-12; idx += 1
                cons[idx] = xi - ri; idx += 1
                cons[idx] = 1.0 - xi - ri; idx += 1
                cons[idx] = yi - ri; idx += 1
                cons[idx] = 1.0 - yi - ri; idx += 1
            for i in range(n):
                xi, yi, ri = c[i]
                for j in range(i + 1, n):
                    xj, yj, rj = c[j]
                    dist = math.sqrt((xi-xj)**2 + (yi-yj)**2)
                    cons[idx] = dist - ri - rj; idx += 1

            # Augmented Lagrangian terms
            for k in range(n_con):
                if cons[k] < lam[k] / mu:
                    obj += -lam[k] * cons[k] + 0.5 * mu * cons[k]**2
                else:
                    obj += -0.5 * lam[k]**2 / mu

            return obj

        bounds = []
        for i in range(n):
            bounds.extend([(0.001, 0.999), (0.001, 0.999), (0.001, 0.499)])

        res = minimize(auglag_obj, x, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": maxiter_inner, "ftol": 1e-16})
        x = res.x

        # Update multipliers
        c = x.reshape(n, 3)
        cons = np.zeros(n_con)
        idx = 0
        for i in range(n):
            xi, yi, ri = c[i]
            cons[idx] = ri - 1e-12; idx += 1
            cons[idx] = xi - ri; idx += 1
            cons[idx] = 1.0 - xi - ri; idx += 1
            cons[idx] = yi - ri; idx += 1
            cons[idx] = 1.0 - yi - ri; idx += 1
        for i in range(n):
            xi, yi, ri = c[i]
            for j in range(i + 1, n):
                xj, yj, rj = c[j]
                dist = math.sqrt((xi-xj)**2 + (yi-yj)**2)
                cons[idx] = dist - ri - rj; idx += 1

        lam = np.maximum(0, lam - mu * cons)
        mu = min(mu * 2.0, 1e8)

        sr = np.sum(c[:, 2])
        viol, max_v = compute_violation(c)
        if outer % 5 == 0:
            log(f"    AL iter {outer}: sr={sr:.10f}, max_v={max_v:.2e}, mu={mu:.0e}")

    result_circles = x.reshape(n, 3)
    viol, max_v = compute_violation(result_circles)
    sr = np.sum(result_circles[:, 2])
    return result_circles, sr, viol, max_v

def main():
    parent_path = os.path.join(WORKDIR, "..", "sa-001", "solution_n26.json")
    parent_circles = load_solution(parent_path)
    parent_sr = np.sum(parent_circles[:, 2])

    cur_path = os.path.join(WORKDIR, "solution_n26.json")
    if os.path.exists(cur_path):
        cur_circles = load_solution(cur_path)
        cur_sr = np.sum(cur_circles[:, 2])
        viol, mv = compute_violation(cur_circles)
        if mv < 1e-10 and cur_sr >= parent_sr:
            init_circles = cur_circles
        else:
            init_circles = parent_circles
    else:
        init_circles = parent_circles

    init_sr = np.sum(init_circles[:, 2])
    log(f"Starting from: sum_radii={init_sr:.12f}")

    best_sr = init_sr
    best_circles = init_circles.copy()

    def try_update(circles, sr, mv, label):
        nonlocal best_sr, best_circles
        if mv < 1e-10 and sr > best_sr:
            best_sr = sr
            best_circles = circles.copy()
            log(f"  *** NEW BEST ({label}): {best_sr:.12f} ***")
            save_solution(best_circles, cur_path)
            return True
        return False

    # === Method 1: High-precision SLSQP ===
    log("\n=== Method 1: High-precision SLSQP (ftol=1e-16, maxiter=20000) ===")
    for run in range(3):
        c, sr, v, mv = slsqp_polish(best_circles, maxiter=20000, ftol=1e-16)
        log(f"  Run {run}: sr={sr:.12f}, max_v={mv:.2e}")
        try_update(c, sr, mv, f"slsqp-{run}")

    # === Method 2: trust-constr (interior point) ===
    log("\n=== Method 2: trust-constr ===")
    c, sr, v, mv = trust_constr_polish(best_circles, maxiter=10000)
    log(f"  trust-constr: sr={sr:.12f}, max_v={mv:.2e}")
    try_update(c, sr, mv, "trust-constr")

    # Polish trust-constr result with SLSQP
    c2, sr2, v2, mv2 = slsqp_polish(c, maxiter=10000, ftol=1e-16)
    log(f"  trust-constr+SLSQP: sr={sr2:.12f}, max_v={mv2:.2e}")
    try_update(c2, sr2, mv2, "trust-constr+slsqp")

    # === Method 3: Augmented Lagrangian from parent ===
    log("\n=== Method 3: Augmented Lagrangian ===")
    c, sr, v, mv = augmented_lagrangian(parent_circles, n_outer=20, maxiter_inner=3000)
    log(f"  AL result: sr={sr:.12f}, max_v={mv:.2e}")
    # Polish
    c2, sr2, v2, mv2 = slsqp_polish(c, maxiter=10000, ftol=1e-16)
    log(f"  AL+SLSQP: sr={sr2:.12f}, max_v={mv2:.2e}")
    try_update(c2, sr2, mv2, "AL+slsqp")

    # === Method 4: Iterated SLSQP with tiny perturbation ===
    log("\n=== Method 4: Micro-perturbation + SLSQP ===")
    rng = np.random.RandomState(42424)
    for attempt in range(50):
        perturbed = best_circles.copy()
        # Very tiny perturbation
        perturbed[:, :2] += rng.normal(0, 1e-4, size=(len(best_circles), 2))
        perturbed[:, 2] *= (1 + rng.normal(0, 1e-5, size=len(best_circles)))
        perturbed[:, 2] = np.clip(perturbed[:, 2], 0.005, 0.495)
        perturbed[:, 0] = np.clip(perturbed[:, 0], perturbed[:, 2], 1 - perturbed[:, 2])
        perturbed[:, 1] = np.clip(perturbed[:, 1], perturbed[:, 2], 1 - perturbed[:, 2])

        c, sr, v, mv = slsqp_polish(perturbed, maxiter=10000, ftol=1e-16)
        if try_update(c, sr, mv, f"micro-{attempt}"):
            pass
        if attempt % 10 == 0:
            log(f"  Attempt {attempt}: sr={sr:.12f}, max_v={mv:.2e}, best={best_sr:.12f}")

    # === Final ===
    log(f"\n{'='*60}")
    log(f"FINAL: sum_radii = {best_sr:.12f}")
    log(f"Parent:           {parent_sr:.12f}")
    log(f"Delta:            {best_sr - parent_sr:+.14f}")
    viol, max_v = compute_violation(best_circles)
    log(f"Validation: total_viol={viol:.2e}, max_viol={max_v:.2e}")
    save_solution(best_circles, cur_path)

if __name__ == "__main__":
    main()
