"""
Final polish: take the best solution and run ultra-aggressive SLSQP
with very tight tolerances and many iterations.
Also try trust-constr solver for comparison.
"""

import numpy as np
from scipy.optimize import minimize
import json
import sys
import time
import math
from pathlib import Path


def load_solution(path):
    with open(path) as f:
        data = json.load(f)
    circles = data["circles"]
    n = len(circles)
    centers = np.array([[c[0], c[1]] for c in circles])
    radii = np.array([c[2] for c in circles])
    return centers, radii, n


def cons_joint(x, n):
    c = x[:2*n].reshape((n, 2))
    r = x[2*n:]
    constraints = []
    for i in range(n):
        for j in range(i+1, n):
            constraints.append(np.linalg.norm(c[i]-c[j]) - r[i] - r[j])
        constraints.extend([c[i,0]-r[i], 1-c[i,0]-r[i], c[i,1]-r[i], 1-c[i,1]-r[i]])
    return np.array(constraints)


def check_valid(centers, radii, tol=1e-10):
    n = len(centers)
    for i in range(n):
        r = radii[i]
        if r <= 0 or r-centers[i,0] > tol or centers[i,0]+r-1 > tol or \
           r-centers[i,1] > tol or centers[i,1]+r-1 > tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            if radii[i]+radii[j]-np.linalg.norm(centers[i]-centers[j]) > tol:
                return False
    return True


def repair(centers, radii, tol=1e-10):
    n = len(radii)
    for _ in range(200):
        changed = False
        for i in range(n):
            r = radii[i]
            r_new = min(r, centers[i,0]-tol, 1-centers[i,0]-tol,
                       centers[i,1]-tol, 1-centers[i,1]-tol)
            if r_new < r:
                radii[i] = max(r_new, 1e-8); changed = True
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(centers[i]-centers[j])
                if radii[i]+radii[j] > dist-tol and radii[i]+radii[j] > 0:
                    scale = max((dist-2*tol)/(radii[i]+radii[j]), 0.01)
                    if scale < 1:
                        radii[i] *= scale; radii[j] *= scale; changed = True
        if not changed:
            break
    return radii


def main():
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path

    centers, radii, n = load_solution(input_path)
    initial_metric = np.sum(radii)
    print(f"Loaded n={n}, metric={initial_metric:.10f}", flush=True)

    best_metric = initial_metric
    best_x = np.concatenate([centers.flatten(), radii])
    bounds = [(0.001, 0.999)] * (2*n) + [(0.001, 0.4)] * n

    # Round 1: Ultra-long SLSQP runs
    print("\nRound 1: SLSQP with increasing iterations", flush=True)
    x = best_x.copy()
    for max_it, ftol in [(5000, 1e-13), (10000, 1e-14), (20000, 1e-15), (50000, 1e-16)]:
        t0 = time.time()
        try:
            res = minimize(
                lambda x: -np.sum(x[2*n:]), x,
                method='SLSQP',
                constraints={'type': 'ineq', 'fun': lambda x: cons_joint(x, n)},
                bounds=bounds,
                options={'maxiter': max_it, 'ftol': ftol}
            )
            c = res.x[:2*n].reshape((n, 2))
            r = res.x[2*n:]
            r = repair(c, r)
            metric = np.sum(r)
            valid = check_valid(c, r)
            dt = time.time() - t0
            print(f"  iter={max_it}, ftol={ftol}: {metric:.10f} {'VALID' if valid else 'INV'} [{dt:.1f}s] nit={res.nit}", flush=True)
            if valid and metric > best_metric:
                best_metric = metric
                best_x = np.concatenate([c.flatten(), r])
            x = res.x
        except Exception as e:
            print(f"  iter={max_it}: ERR {e}", flush=True)

    # Round 2: Try from slightly different starting points (micro-perturbation)
    print(f"\nRound 2: Micro-perturbation polish ({best_metric:.10f})", flush=True)
    rng = np.random.RandomState(12345)
    for att in range(30):
        x_pert = best_x.copy()
        # Very small perturbation - just positions
        x_pert[:2*n] += rng.normal(0, 0.001, 2*n)
        x_pert[:2*n] = np.clip(x_pert[:2*n], 0.01, 0.99)
        try:
            res = minimize(
                lambda x: -np.sum(x[2*n:]), x_pert,
                method='SLSQP',
                constraints={'type': 'ineq', 'fun': lambda x: cons_joint(x, n)},
                bounds=bounds,
                options={'maxiter': 20000, 'ftol': 1e-16}
            )
            c = res.x[:2*n].reshape((n, 2))
            r = res.x[2*n:]
            r = repair(c, r)
            metric = np.sum(r)
            if check_valid(c, r) and metric > best_metric + 1e-11:
                best_metric = metric
                best_x = np.concatenate([c.flatten(), r])
                print(f"  #{att+1}: {metric:.10f} ** IMPROVED **", flush=True)
        except:
            pass

    # Round 3: Try with radii perturbation too
    print(f"\nRound 3: Radii perturbation ({best_metric:.10f})", flush=True)
    for att in range(20):
        x_pert = best_x.copy()
        x_pert[2*n:] *= rng.uniform(0.98, 1.02, n)
        x_pert[2*n:] = np.clip(x_pert[2*n:], 0.001, 0.4)
        try:
            res = minimize(
                lambda x: -np.sum(x[2*n:]), x_pert,
                method='SLSQP',
                constraints={'type': 'ineq', 'fun': lambda x: cons_joint(x, n)},
                bounds=bounds,
                options={'maxiter': 20000, 'ftol': 1e-16}
            )
            c = res.x[:2*n].reshape((n, 2))
            r = res.x[2*n:]
            r = repair(c, r)
            metric = np.sum(r)
            if check_valid(c, r) and metric > best_metric + 1e-11:
                best_metric = metric
                best_x = np.concatenate([c.flatten(), r])
                print(f"  #{att+1}: {metric:.10f} ** IMPROVED **", flush=True)
        except:
            pass

    print(f"\nFinal: {best_metric:.10f}", flush=True)

    # Save
    c = best_x[:2*n].reshape((n, 2))
    r = best_x[2*n:]
    circles = [[c[i,0], c[i,1], r[i]] for i in range(n)]
    with open(output_path, 'w') as f:
        json.dump({"circles": circles}, f, indent=2)
    print(f"Saved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
