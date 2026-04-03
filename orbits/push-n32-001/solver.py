"""
Fast n=32 circle packing solver.
Maximize sum of radii for 32 circles in [0,1]^2.

Uses SLSQP (which handles constraints natively) with many restarts
from perturbations + swaps. Also uses penalty-based basin-hopping
with analytical gradients for speed.
"""

import json
import math
import numpy as np
from scipy.optimize import minimize, basinhopping
from pathlib import Path
import time

N = 32
NDIM = 3 * N

BASE_DIR = Path(__file__).parent
BEST_FILE = BASE_DIR / "solution_n32.json"
OUTPUT_FILE = BASE_DIR / "best_n32.json"

def log(msg):
    print(msg, flush=True)

def load_solution(path):
    with open(path) as f:
        data = json.load(f)
    circles = data.get("circles", data)
    return np.array(circles).flatten()

def save_solution(x, path):
    c = x.reshape(-1, 3)
    data = {"circles": [[float(v) for v in row] for row in c]}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# ---- Constraints for SLSQP ----

def ineq_constraints(x):
    c = x.reshape(-1, 3)
    n = len(c)
    xs, ys, rs = c[:, 0], c[:, 1], c[:, 2]
    cont = np.concatenate([xs - rs, 1.0 - xs - rs, ys - rs, 1.0 - ys - rs, rs - 1e-6])
    overlaps = np.empty(n * (n - 1) // 2)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            overlaps[k] = math.sqrt(dx * dx + dy * dy) - rs[i] - rs[j]
            k += 1
    return np.concatenate([cont, overlaps])

def is_valid(x, tol=1e-10):
    c = x.reshape(-1, 3)
    xs, ys, rs = c[:, 0], c[:, 1], c[:, 2]
    if np.any(xs - rs < -tol) or np.any(1.0 - xs - rs < -tol):
        return False
    if np.any(ys - rs < -tol) or np.any(1.0 - ys - rs < -tol):
        return False
    if np.any(rs < -tol):
        return False
    for i in range(N):
        dx = xs[i] - xs[i+1:]
        dy = ys[i] - ys[i+1:]
        dists_sq = dx * dx + dy * dy
        min_dists = rs[i] + rs[i+1:]
        if np.any(dists_sq < (min_dists - tol) ** 2):
            # Double-check with sqrt
            dists = np.sqrt(dists_sq)
            if np.any(dists - min_dists < -tol):
                return False
    return True

def get_metric(x):
    return np.sum(x[2::3])

def make_bounds():
    bounds = []
    for _ in range(N):
        bounds.append((0.005, 0.995))
        bounds.append((0.005, 0.995))
        bounds.append((0.005, 0.5))
    return bounds

BOUNDS = make_bounds()

# ---- SLSQP optimizer ----

def neg_sum_radii(x):
    return -np.sum(x[2::3])

NEG_JAC = np.zeros(NDIM)
NEG_JAC[2::3] = -1.0

def slsqp_opt(x0, maxiter=5000):
    result = minimize(
        neg_sum_radii, x0, method='SLSQP',
        jac=lambda x: NEG_JAC,
        bounds=BOUNDS,
        constraints=[{"type": "ineq", "fun": ineq_constraints}],
        options={'maxiter': maxiter, 'ftol': 1e-16, 'disp': False}
    )
    return result.x, -result.fun

# ---- Penalized objective with analytical gradient ----

def penalized_obj_and_grad(x, penalty):
    """Compute penalized objective and its analytical gradient."""
    c = x.reshape(-1, 3)
    xs, ys, rs = c[:, 0], c[:, 1], c[:, 2]

    obj = -np.sum(rs)
    grad = np.zeros(NDIM)
    grad[2::3] = -1.0  # d(-sum_r)/dr_i = -1

    # Containment: x_i - r_i >= 0 => violation when x_i - r_i < 0
    # penalty * (x_i - r_i)^2 when violated
    for i in range(N):
        ix, iy, ir = 3*i, 3*i+1, 3*i+2

        # left: x - r >= 0
        v = xs[i] - rs[i]
        if v < 0:
            obj += penalty * v * v
            grad[ix] += penalty * 2 * v       # dv/dx = 1
            grad[ir] += penalty * 2 * v * -1  # dv/dr = -1

        # right: 1 - x - r >= 0
        v = 1.0 - xs[i] - rs[i]
        if v < 0:
            obj += penalty * v * v
            grad[ix] += penalty * 2 * v * -1
            grad[ir] += penalty * 2 * v * -1

        # bottom: y - r >= 0
        v = ys[i] - rs[i]
        if v < 0:
            obj += penalty * v * v
            grad[iy] += penalty * 2 * v
            grad[ir] += penalty * 2 * v * -1

        # top: 1 - y - r >= 0
        v = 1.0 - ys[i] - rs[i]
        if v < 0:
            obj += penalty * v * v
            grad[iy] += penalty * 2 * v * -1
            grad[ir] += penalty * 2 * v * -1

    # Non-overlap: dist(i,j) - r_i - r_j >= 0
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = math.sqrt(dx * dx + dy * dy)
            gap = dist - rs[i] - rs[j]
            if gap < 0:
                obj += penalty * gap * gap
                if dist > 1e-12:
                    # d(gap)/d(x_i) = dx/dist, d(gap)/d(x_j) = -dx/dist
                    # d(gap)/d(y_i) = dy/dist, d(gap)/d(y_j) = -dy/dist
                    # d(gap)/d(r_i) = -1, d(gap)/d(r_j) = -1
                    dg = penalty * 2 * gap
                    grad[3*i]   += dg * dx / dist
                    grad[3*i+1] += dg * dy / dist
                    grad[3*i+2] += dg * (-1.0)
                    grad[3*j]   += dg * (-dx / dist)
                    grad[3*j+1] += dg * (-dy / dist)
                    grad[3*j+2] += dg * (-1.0)

    return obj, grad

def penalty_lbfgsb(x0, penalty=1e5, maxiter=500):
    result = minimize(
        penalized_obj_and_grad, x0, args=(penalty,),
        method='L-BFGS-B', jac=True, bounds=BOUNDS,
        options={'maxiter': maxiter, 'ftol': 1e-15}
    )
    return result.x

def progressive_penalty_opt(x0):
    x = x0.copy()
    for penalty in [1e2, 1e3, 1e4, 1e5, 1e6]:
        x = penalty_lbfgsb(x, penalty=penalty, maxiter=400)
    x_pol, metric = slsqp_opt(x, maxiter=3000)
    return x_pol, metric

# ---- Initialization helpers ----

def greedy_init(rng, n_samples=5000):
    circles = []
    for k in range(N):
        best_r = 0
        best_pos = None
        for _ in range(n_samples):
            cx = rng.uniform(0.01, 0.99)
            cy = rng.uniform(0.01, 0.99)
            max_r = min(cx, 1 - cx, cy, 1 - cy)
            for (ox, oy, orr) in circles:
                d = math.sqrt((cx - ox) ** 2 + (cy - oy) ** 2)
                max_r = min(max_r, d - orr)
            if max_r > best_r:
                best_r = max_r
                best_pos = [cx, cy, max_r]
        if best_pos and best_pos[2] > 0.001:
            circles.append(best_pos)
        else:
            r = 0.002
            circles.append([rng.uniform(r + 0.01, 1 - r - 0.01),
                           rng.uniform(r + 0.01, 1 - r - 0.01), r])
    return np.array(circles).flatten()

def perturb(x, rng, scale=0.05):
    c = x.reshape(-1, 3).copy()
    num = rng.integers(1, N + 1)
    idx = rng.choice(N, size=num, replace=False)
    c[idx, 0] += rng.normal(0, scale, num)
    c[idx, 1] += rng.normal(0, scale, num)
    if rng.random() < 0.3:
        c[idx, 2] *= (1 + rng.normal(0, 0.1, num))
        c[idx, 2] = np.maximum(c[idx, 2], 0.005)
    for i in range(N):
        r = np.clip(c[i, 2], 0.005, 0.5)
        c[i, 2] = r
        c[i, 0] = np.clip(c[i, 0], r + 1e-4, 1 - r - 1e-4)
        c[i, 1] = np.clip(c[i, 1], r + 1e-4, 1 - r - 1e-4)
    return c.flatten()

# ---- Basin-hopping with penalty (analytical grad => fast) ----

class PenaltyStep:
    def __init__(self, stepsize=0.08, rng=None):
        self.stepsize = stepsize
        self.rng = rng or np.random.default_rng()

    def __call__(self, x):
        c = x.reshape(-1, 3).copy()
        strategy = self.rng.integers(5)
        if strategy == 0:
            c[:, 0] += self.rng.normal(0, self.stepsize * 0.3, N)
            c[:, 1] += self.rng.normal(0, self.stepsize * 0.3, N)
        elif strategy == 1:
            k = self.rng.integers(2, max(3, N // 3))
            idx = self.rng.choice(N, size=k, replace=False)
            c[idx, 0] += self.rng.normal(0, self.stepsize, k)
            c[idx, 1] += self.rng.normal(0, self.stepsize, k)
        elif strategy == 2:
            i, j = self.rng.choice(N, size=2, replace=False)
            c[i, 0], c[j, 0] = c[j, 0], c[i, 0]
            c[i, 1], c[j, 1] = c[j, 1], c[i, 1]
        elif strategy == 3:
            k = self.rng.integers(1, N + 1)
            idx = self.rng.choice(N, size=k, replace=False)
            c[idx, 0] += self.rng.normal(0, self.stepsize * 0.5, k)
            c[idx, 1] += self.rng.normal(0, self.stepsize * 0.5, k)
            c[idx, 2] *= (1 + self.rng.normal(0, 0.08, k))
            c[idx, 2] = np.maximum(c[idx, 2], 0.005)
        elif strategy == 4:
            sorted_idx = np.argsort(c[:, 2])
            small = sorted_idx[:3]
            c[small, 2] *= 0.5
        for i in range(N):
            r = np.clip(c[i, 2], 0.005, 0.5)
            c[i, 2] = r
            c[i, 0] = np.clip(c[i, 0], r + 1e-4, 1 - r - 1e-4)
            c[i, 1] = np.clip(c[i, 1], r + 1e-4, 1 - r - 1e-4)
        return c.flatten()

def run():
    start_time = time.time()

    best_x = load_solution(BEST_FILE)
    best_metric = get_metric(best_x)
    log(f"Loaded starting solution: metric = {best_metric:.10f}")

    # Initial polish
    x_pol, m_pol = slsqp_opt(best_x, maxiter=10000)
    if is_valid(x_pol) and m_pol > best_metric + 1e-12:
        best_x = x_pol.copy()
        best_metric = m_pol
        log(f"Initial polish: {best_metric:.10f}")

    improvements = []

    def try_update(x, source):
        nonlocal best_x, best_metric
        if is_valid(x):
            m = get_metric(x)
            if m > best_metric + 1e-12:
                imp = m - best_metric
                best_metric = m
                best_x = x.copy()
                improvements.append((source, m))
                log(f"  *** NEW BEST: {m:.10f} (+{imp:.2e}) [{source}]")
                return True
        return False

    # ================================================================
    # Phase 1: Basin-hopping with penalty + analytical grad (20 seeds)
    # ================================================================
    log(f"\n{'='*60}")
    log("Phase 1: Basin-hopping penalty L-BFGS-B (20 seeds x 500 hops)")
    log(f"{'='*60}")

    for seed_idx in range(20):
        seed = seed_idx * 137 + 42
        rng = np.random.default_rng(seed)
        t0 = time.time()

        if seed_idx < 14:
            scale = 0.01 + 0.1 * (seed_idx / 14)
            x0 = perturb(best_x, rng, scale=scale)
        else:
            x0 = greedy_init(rng, n_samples=3000)

        minimizer_kwargs = {
            'method': 'L-BFGS-B',
            'jac': True,
            'bounds': BOUNDS,
            'args': (1e5,),
            'options': {'maxiter': 200, 'ftol': 1e-14}
        }

        step_fn = PenaltyStep(stepsize=0.1, rng=rng)

        try:
            result = basinhopping(
                penalized_obj_and_grad, x0,
                minimizer_kwargs=minimizer_kwargs,
                niter=500,
                T=0.5,
                stepsize=0.1,
                take_step=step_fn,
                seed=int(seed),
                niter_success=80,
            )
            # Polish result with SLSQP
            x_pol, m = slsqp_opt(result.x, maxiter=5000)
            try_update(x_pol, f"bh-s{seed_idx}")
        except Exception as e:
            log(f"  Seed {seed_idx} error: {e}")

        dt = time.time() - t0
        log(f"  Seed {seed_idx+1}/20 done ({dt:.0f}s), best={best_metric:.10f}")

    log(f"\nAfter Phase 1: {best_metric:.10f} ({time.time()-start_time:.0f}s)")

    # ================================================================
    # Phase 2: Multi-start progressive penalty (300 starts)
    # ================================================================
    log(f"\n{'='*60}")
    log("Phase 2: Multi-start progressive penalty (300 starts)")
    log(f"{'='*60}")

    for trial in range(300):
        if trial % 50 == 0:
            log(f"  Trial {trial}/300, best={best_metric:.10f}, time={time.time()-start_time:.0f}s")

        rng = np.random.default_rng(trial * 31 + 20000)

        if trial < 200:
            scale = rng.exponential(0.03)
            x0 = perturb(best_x, rng, scale=scale)
        else:
            x0 = greedy_init(rng, n_samples=3000)

        try:
            x_opt, m = progressive_penalty_opt(x0)
            try_update(x_opt, f"ms-{trial}")
        except Exception:
            pass

    log(f"\nAfter Phase 2: {best_metric:.10f} ({time.time()-start_time:.0f}s)")

    # ================================================================
    # Phase 3: Fine-grained SLSQP (500 trials)
    # ================================================================
    log(f"\n{'='*60}")
    log("Phase 3: Fine-grained local search (500 trials)")
    log(f"{'='*60}")

    for trial in range(500):
        if trial % 100 == 0:
            log(f"  Trial {trial}/500, best={best_metric:.10f}")

        rng = np.random.default_rng(trial * 17 + 50000)
        scale = rng.exponential(0.005)
        x0 = perturb(best_x, rng, scale=scale)

        try:
            x_opt, m = slsqp_opt(x0, maxiter=3000)
            try_update(x_opt, f"fine-{trial}")
        except Exception:
            pass

    log(f"\nAfter Phase 3: {best_metric:.10f} ({time.time()-start_time:.0f}s)")

    # ================================================================
    # Phase 4: Swap + SLSQP (200 trials)
    # ================================================================
    log(f"\n{'='*60}")
    log("Phase 4: Swap neighborhood (200 trials)")
    log(f"{'='*60}")

    for trial in range(200):
        if trial % 50 == 0:
            log(f"  Trial {trial}/200, best={best_metric:.10f}")

        rng = np.random.default_rng(trial * 53 + 70000)
        c = best_x.reshape(-1, 3).copy()
        num_swap = rng.integers(2, 5)
        idx = rng.choice(N, size=num_swap, replace=False)
        perm = rng.permutation(num_swap)
        pos_save = c[idx, :2].copy()
        for k in range(num_swap):
            c[idx[k], :2] = pos_save[perm[k]]
        for i in range(N):
            r = c[i, 2]
            c[i, 0] = np.clip(c[i, 0], r + 1e-4, 1 - r - 1e-4)
            c[i, 1] = np.clip(c[i, 1], r + 1e-4, 1 - r - 1e-4)

        try:
            x_opt, m = slsqp_opt(c.flatten(), maxiter=5000)
            try_update(x_opt, f"swap-{trial}")
        except Exception:
            pass

    log(f"\nAfter Phase 4: {best_metric:.10f} ({time.time()-start_time:.0f}s)")

    # Final polish
    log("\nFinal SLSQP polish...")
    x_final, m_final = slsqp_opt(best_x, maxiter=20000)
    try_update(x_final, "final-polish")

    save_solution(best_x, OUTPUT_FILE)

    elapsed = time.time() - start_time
    log(f"\n{'='*60}")
    log(f"FINAL BEST: {best_metric:.10f}")
    log(f"Total improvements: {len(improvements)}")
    log(f"Total time: {elapsed:.1f}s")
    for src, m in improvements:
        log(f"  {src}: {m:.10f}")
    log(f"Saved to {OUTPUT_FILE}")

    return best_metric

if __name__ == "__main__":
    run()
