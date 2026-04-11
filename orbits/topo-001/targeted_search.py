"""
Targeted topology search: Focus on specific topology modifications
that are more likely to yield improvements.

Key observations from the parent solution:
- Circles 10 and 13 are the smallest (r~0.069) and act as "fillers"
- The solution has ~4-fold symmetry
- All contacts are extremely tight (gaps < 3e-10)

Strategies:
1. Remove the two smallest circles and re-optimize 24 circles, then add 2 back
2. Try packing with one fewer size class (more uniform radii)
3. Force asymmetric arrangements by fixing some circles and optimizing the rest
4. Use the COBYLA optimizer which handles constraints differently than SLSQP
5. Try trust-constr method
"""

import json
import numpy as np
from scipy.optimize import minimize
import os
import time

WORKDIR = os.path.dirname(os.path.abspath(__file__))
N = 26

def load_solution(path):
    with open(path) as f:
        data = json.load(f)
    circles = np.array(data["circles"])
    return circles[:, 0], circles[:, 1], circles[:, 2]

def save_solution(x, y, r, path):
    circles = [[float(x[i]), float(y[i]), float(r[i])] for i in range(len(x))]
    with open(path, 'w') as f:
        json.dump({"circles": circles}, f, indent=2)

def is_feasible(x, y, r, tol=1e-8):
    n = len(x)
    for i in range(n):
        if r[i] < -tol: return False
        if x[i] - r[i] < -tol or 1 - x[i] - r[i] < -tol: return False
        if y[i] - r[i] < -tol or 1 - y[i] - r[i] < -tol: return False
    for i in range(n):
        for j in range(i+1, n):
            if (x[i]-x[j])**2 + (y[i]-y[j])**2 < (r[i]+r[j])**2 - tol:
                return False
    return True

def build_constraints(n):
    constraints = []
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i, n=n: p[i] - p[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i, n=n: 1 - p[i] - p[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i, n=n: p[n+i] - p[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i, n=n: 1 - p[n+i] - p[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i, n=n: p[2*n+i] - 1e-6})
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j, n=n: (
                    (p[i]-p[j])**2 + (p[n+i]-p[n+j])**2 - (p[2*n+i]+p[2*n+j])**2
                )
            })
    return constraints

def optimize_slsqp(x0, y0, r0, maxiter=10000):
    n = len(x0)
    params0 = np.concatenate([x0, y0, r0])
    constraints = build_constraints(n)
    result = minimize(
        lambda p, n=n: -np.sum(p[2*n:3*n]),
        params0, method='SLSQP', constraints=constraints,
        options={'maxiter': maxiter, 'ftol': 1e-15, 'disp': False}
    )
    x = result.x[:n]
    y = result.x[n:2*n]
    r = result.x[2*n:3*n]
    return x, y, r, np.sum(r), result.success

def optimize_cobyla(x0, y0, r0, maxiter=20000):
    """COBYLA might find different local optima than SLSQP."""
    n = len(x0)
    params0 = np.concatenate([x0, y0, r0])

    cons = []
    for i in range(n):
        cons.append(lambda p, i=i, n=n: p[i] - p[2*n+i])
        cons.append(lambda p, i=i, n=n: 1 - p[i] - p[2*n+i])
        cons.append(lambda p, i=i, n=n: p[n+i] - p[2*n+i])
        cons.append(lambda p, i=i, n=n: 1 - p[n+i] - p[2*n+i])
        cons.append(lambda p, i=i, n=n: p[2*n+i] - 1e-6)
    for i in range(n):
        for j in range(i+1, n):
            cons.append(lambda p, i=i, j=j, n=n: (
                np.sqrt((p[i]-p[j])**2 + (p[n+i]-p[n+j])**2) - p[2*n+i] - p[2*n+j]
            ))

    result = minimize(
        lambda p, n=n: -np.sum(p[2*n:3*n]),
        params0, method='COBYLA',
        constraints=[{'type': 'ineq', 'fun': c} for c in cons],
        options={'maxiter': maxiter, 'rhobeg': 0.01, 'catol': 1e-12}
    )

    x = result.x[:n]
    y = result.x[n:2*n]
    r = result.x[2*n:3*n]
    return x, y, r, np.sum(r), result.success

def optimize_trust_constr(x0, y0, r0, maxiter=5000):
    """Trust-region constrained optimizer."""
    n = len(x0)
    params0 = np.concatenate([x0, y0, r0])

    from scipy.optimize import NonlinearConstraint

    def constraint_func(p):
        n = len(p) // 3
        x, y, r = p[:n], p[n:2*n], p[2*n:]
        vals = []
        # Containment: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0, r >= 0
        for i in range(n):
            vals.append(x[i] - r[i])
            vals.append(1 - x[i] - r[i])
            vals.append(y[i] - r[i])
            vals.append(1 - y[i] - r[i])
            vals.append(r[i])
        # Non-overlap: dist^2 - (ri+rj)^2 >= 0
        for i in range(n):
            for j in range(i+1, n):
                vals.append((x[i]-x[j])**2 + (y[i]-y[j])**2 - (r[i]+r[j])**2)
        return np.array(vals)

    n_contain = 5 * n
    n_nonoverlap = n * (n-1) // 2
    n_total = n_contain + n_nonoverlap

    nlc = NonlinearConstraint(constraint_func, 0, np.inf)

    result = minimize(
        lambda p: -np.sum(p[2*n:3*n]),
        params0, method='trust-constr',
        constraints=[nlc],
        options={'maxiter': maxiter, 'gtol': 1e-15, 'verbose': 0}
    )

    x = result.x[:n]
    y = result.x[n:2*n]
    r = result.x[2*n:3*n]
    return x, y, r, np.sum(r), result.success

def multi_optimizer_polish(x0, y0, r0):
    """Try multiple optimizers on the same starting point."""
    best_metric = np.sum(r0)
    best_x, best_y, best_r = x0.copy(), y0.copy(), r0.copy()

    # SLSQP
    x, y, r, m, s = optimize_slsqp(x0, y0, r0, maxiter=15000)
    if s and is_feasible(x, y, r) and m > best_metric:
        best_metric, best_x, best_y, best_r = m, x, y, r

    # COBYLA
    x, y, r, m, s = optimize_cobyla(x0, y0, r0, maxiter=30000)
    if s and is_feasible(x, y, r) and m > best_metric:
        best_metric, best_x, best_y, best_r = m, x, y, r

    # Trust-constr
    try:
        x, y, r, m, s = optimize_trust_constr(x0, y0, r0, maxiter=3000)
        if s and is_feasible(x, y, r) and m > best_metric:
            best_metric, best_x, best_y, best_r = m, x, y, r
    except Exception:
        pass

    # Chain: SLSQP -> COBYLA -> SLSQP
    x, y, r, m, s = optimize_slsqp(best_x, best_y, best_r, maxiter=15000)
    if s and is_feasible(x, y, r) and m > best_metric:
        best_metric, best_x, best_y, best_r = m, x, y, r

    return best_x, best_y, best_r, best_metric

def remove_and_reinsert(x0, y0, r0, remove_indices):
    """Remove circles, optimize remainder, then greedily add them back."""
    n = len(x0)
    keep = [i for i in range(n) if i not in remove_indices]
    n_keep = len(keep)

    # Optimize the remaining circles
    xk = x0[keep]
    yk = y0[keep]
    rk = r0[keep]

    xk, yk, rk, mk, sk = optimize_slsqp(xk, yk, rk, maxiter=10000)

    if not sk:
        return x0, y0, r0, np.sum(r0)

    # Now greedily add removed circles back
    x_out = list(xk)
    y_out = list(yk)
    r_out = list(rk)

    rng = np.random.RandomState(42)

    for idx in remove_indices:
        best_r = 0
        best_pos = None

        # Try many positions
        for _ in range(2000):
            px = rng.uniform(0.01, 0.99)
            py = rng.uniform(0.01, 0.99)

            max_r = min(px, 1-px, py, 1-py)
            for j in range(len(x_out)):
                dist = np.sqrt((px - x_out[j])**2 + (py - y_out[j])**2)
                max_r = min(max_r, dist - r_out[j])
                if max_r < 0.005:
                    break

            if max_r > best_r:
                best_r = max_r
                best_pos = (px, py)

        if best_pos is not None and best_r > 0.005:
            x_out.append(best_pos[0])
            y_out.append(best_pos[1])
            r_out.append(best_r)
        else:
            x_out.append(rng.uniform(0.1, 0.9))
            y_out.append(rng.uniform(0.1, 0.9))
            r_out.append(0.01)

    x_out = np.array(x_out)
    y_out = np.array(y_out)
    r_out = np.array(r_out)

    # Final optimization
    x_out, y_out, r_out, m_out, s_out = optimize_slsqp(x_out, y_out, r_out, maxiter=10000)

    if s_out and is_feasible(x_out, y_out, r_out):
        return x_out, y_out, r_out, m_out
    else:
        return x0, y0, r0, np.sum(r0)

def main():
    t0 = time.time()
    parent_path = os.path.join(WORKDIR, '..', 'nlp-001', 'solution_n26.json')
    x0, y0, r0 = load_solution(parent_path)
    parent_metric = np.sum(r0)
    print(f"Parent metric: {parent_metric:.10f}")

    best_metric = parent_metric
    best_x, best_y, best_r = x0.copy(), y0.copy(), r0.copy()

    # ====== Strategy 1: Multi-optimizer polish ======
    print("\n=== Multi-Optimizer Polish ===")
    x1, y1, r1, m1 = multi_optimizer_polish(x0, y0, r0)
    print(f"  Multi-polish: {m1:.10f}")
    if m1 > best_metric:
        best_metric, best_x, best_y, best_r = m1, x1, y1, r1

    # ====== Strategy 2: Remove-and-reinsert ======
    print("\n=== Remove-and-Reinsert ===")

    # Sort by radius to identify candidates for removal
    r_order = np.argsort(r0)
    print(f"  Smallest circles: {list(r_order[:6])} with radii {[f'{r0[i]:.6f}' for i in r_order[:6]]}")

    # Try removing the smallest circles
    for k in [1, 2, 3]:
        remove = list(r_order[:k])
        print(f"  Removing {k} smallest: {remove}")
        xr, yr, rr, mr = remove_and_reinsert(x0, y0, r0, remove)
        print(f"    Result: {mr:.10f}")
        if mr > best_metric:
            best_metric, best_x, best_y, best_r = mr, xr, yr, rr
            print(f"    IMPROVED!")

    # Try removing various individual circles
    for i in range(N):
        xr, yr, rr, mr = remove_and_reinsert(x0, y0, r0, [i])
        if mr > best_metric:
            print(f"  Remove [{i}] (r={r0[i]:.6f}): IMPROVED to {mr:.10f}")
            best_metric, best_x, best_y, best_r = mr, xr, yr, rr

    elapsed = time.time() - t0
    print(f"  Remove-reinsert done, {elapsed:.0f}s, best={best_metric:.10f}")

    # ====== Strategy 3: Scaled copies of the solution ======
    print("\n=== Scaled/Transformed Solutions ===")
    rng = np.random.RandomState(42)

    for trial in range(20):
        x2, y2, r2 = x0.copy(), y0.copy(), r0.copy()

        # Small scaling perturbations
        # Scale x coordinates
        sx = 1 + rng.normal(0, 0.01)
        sy = 1 + rng.normal(0, 0.01)
        x2 = 0.5 + (x2 - 0.5) * sx
        y2 = 0.5 + (y2 - 0.5) * sy

        # Uniform radius scaling
        rs = 1 + rng.normal(0, 0.005)
        r2 *= rs

        r2 = np.maximum(r2, 0.01)
        x2 = np.clip(x2, r2+0.001, 1-r2-0.001)
        y2 = np.clip(y2, r2+0.001, 1-r2-0.001)

        x2, y2, r2, m2, s2 = optimize_slsqp(x2, y2, r2, maxiter=10000)
        if s2 and is_feasible(x2, y2, r2) and m2 > best_metric:
            print(f"  Scaled trial {trial}: IMPROVED to {m2:.10f}")
            best_metric, best_x, best_y, best_r = m2, x2, y2, r2

    elapsed = time.time() - t0
    print(f"  Scaled solutions done, {elapsed:.0f}s, best={best_metric:.10f}")

    # ====== Strategy 4: COBYLA from various perturbations ======
    print("\n=== COBYLA Exploration ===")
    for seed in range(10):
        rng = np.random.RandomState(seed + 2000)
        x2 = x0 + rng.normal(0, 0.03, N)
        y2 = y0 + rng.normal(0, 0.03, N)
        r2 = r0 * (1 + rng.normal(0, 0.02, N))
        r2 = np.maximum(r2, 0.01)
        x2 = np.clip(x2, r2+0.001, 1-r2-0.001)
        y2 = np.clip(y2, r2+0.001, 1-r2-0.001)

        x2, y2, r2, m2, s2 = optimize_cobyla(x2, y2, r2, maxiter=30000)
        if s2 and is_feasible(x2, y2, r2):
            if m2 > best_metric:
                print(f"  COBYLA seed={seed}: IMPROVED to {m2:.10f}")
                best_metric, best_x, best_y, best_r = m2, x2, y2, r2
            # Also polish with SLSQP
            x3, y3, r3, m3, s3 = optimize_slsqp(x2, y2, r2, maxiter=10000)
            if s3 and is_feasible(x3, y3, r3) and m3 > best_metric:
                print(f"  COBYLA+SLSQP seed={seed}: IMPROVED to {m3:.10f}")
                best_metric, best_x, best_y, best_r = m3, x3, y3, r3

    elapsed = time.time() - t0
    print(f"  COBYLA done, {elapsed:.0f}s, best={best_metric:.10f}")

    # Save final result
    save_solution(best_x, best_y, best_r, os.path.join(WORKDIR, 'solution_n26_targeted.json'))

    print(f"\n=== FINAL ===")
    print(f"Parent:  {parent_metric:.10f}")
    print(f"Best:    {best_metric:.10f}")
    print(f"Delta:   {best_metric - parent_metric:.2e}")
    print(f"Time:    {time.time()-t0:.0f}s")

if __name__ == '__main__':
    main()
