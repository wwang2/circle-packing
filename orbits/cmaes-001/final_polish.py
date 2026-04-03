"""Final attempt: Clean high-precision polish of parent solution."""

import json
import math
import os
import numpy as np
from scipy.optimize import minimize

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
    max_viol = 0.0
    for i in range(n):
        x, y, r = circles[i]
        for v in [r - x, x + r - 1.0, r - y, y + r - 1.0, -r]:
            if v > 0:
                max_viol = max(max_viol, v)
    for i in range(n):
        xi, yi, ri = circles[i]
        for j in range(i + 1, n):
            xj, yj, rj = circles[j]
            dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)
            overlap = (ri + rj) - dist
            if overlap > 0:
                max_viol = max(max_viol, overlap)
    return max_viol

def slsqp_polish(circles, maxiter=20000):
    n = len(circles)
    x0 = circles.flatten()
    def neg_sum_radii(x):
        return -np.sum(x.reshape(n, 3)[:, 2])
    constraints = []
    for i in range(n):
        ri = 3 * i + 2; xi = 3 * i; yi = 3 * i + 1
        constraints.append({"type": "ineq", "fun": lambda x, idx=ri: x[idx] - 1e-14})
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
        bounds.extend([(1e-6, 1-1e-6), (1e-6, 1-1e-6), (1e-6, 0.5-1e-6)])
    res = minimize(neg_sum_radii, x0, method="SLSQP", constraints=constraints,
                   bounds=bounds, options={"maxiter": maxiter, "ftol": 1e-16})
    return res.x.reshape(n, 3)

def main():
    # Start fresh from parent
    parent_path = os.path.join(WORKDIR, "..", "sa-001", "solution_n26.json")
    parent = load_solution(parent_path)
    parent_sr = np.sum(parent[:, 2])
    parent_mv = compute_violation(parent)
    log(f"Parent: sr={parent_sr:.14f}, max_v={parent_mv:.2e}")

    # Polish parent directly
    polished = slsqp_polish(parent, maxiter=50000)
    pol_sr = np.sum(polished[:, 2])
    pol_mv = compute_violation(polished)
    log(f"Polished parent: sr={pol_sr:.14f}, max_v={pol_mv:.2e}")

    # Polish again
    polished2 = slsqp_polish(polished, maxiter=50000)
    pol2_sr = np.sum(polished2[:, 2])
    pol2_mv = compute_violation(polished2)
    log(f"Double polish: sr={pol2_sr:.14f}, max_v={pol2_mv:.2e}")

    # Pick best valid solution
    candidates = [
        (parent, parent_sr, parent_mv, "parent"),
        (polished, pol_sr, pol_mv, "polished"),
        (polished2, pol2_sr, pol2_mv, "double-polish"),
    ]

    # Load current best too
    cur_path = os.path.join(WORKDIR, "solution_n26.json")
    if os.path.exists(cur_path):
        cur = load_solution(cur_path)
        cur_sr = np.sum(cur[:, 2])
        cur_mv = compute_violation(cur)
        candidates.append((cur, cur_sr, cur_mv, "current"))
        log(f"Current: sr={cur_sr:.14f}, max_v={cur_mv:.2e}")

    best = None
    best_sr = 0
    for c, sr, mv, label in candidates:
        log(f"  Candidate {label}: sr={sr:.14f}, mv={mv:.2e}, valid={mv < 1e-10}")
        if mv < 1e-10 and sr > best_sr:
            best = c
            best_sr = sr
            best_label = label

    if best is not None:
        log(f"\nBest valid: {best_label} with sr={best_sr:.14f}")
        save_solution(best, cur_path)
    else:
        log("No valid solution found!")

if __name__ == "__main__":
    main()
