"""
RIGIDITY-001: σ_min(R) + Weakest-Multiplier Flex Probe for the n=26 packing

Connelly-style rigidity diagnostic on the mobius-001 optimum (Σr = 2.6359830865).

Pipeline:
  1. Load parent solution (26 disks, 78 primal vars).
  2. Extract active constraint set (78 = 58 disk-disk + 20 wall).
  3. Build rigidity matrix R (78 x 78), compute σ_min(R) and κ(R).
  4. Recover dual variables λ by solving R^T λ = ∇f  (f = Σ r_i).
  5. Build stress matrix Ω = Σ λ_k ∇²g_k and report its spectrum.
  6. Identify weakest multiplier, perturb its 3 neighbors, re-optimize, see if Σr improves.
  7. Save JSON report and figures.

Author: orbit/rigidity-001
"""

from __future__ import annotations

import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

# ---------- paths ----------
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PARENT_SOL = REPO / "orbits" / "mobius-001" / "solution_n26.json"
OUT_JSON = HERE / "rigidity_report.json"
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ---------- numerical thresholds ----------
ACTIVE_GAP_TOL = 1e-7  # constraint considered active if gap below this
N_DISKS = 26
PRIMAL_DIM = 3 * N_DISKS  # 78

np.set_printoptions(precision=6, suppress=True)


# ===================================================================
# helpers
# ===================================================================


def load_solution(path: Path) -> np.ndarray:
    """Return array of shape (n, 3) with (x, y, r) per disk."""
    with open(path) as f:
        d = json.load(f)
    arr = np.array(d["circles"], dtype=float)
    assert arr.shape == (N_DISKS, 3)
    return arr


def sum_radii(state: np.ndarray) -> float:
    return float(state[:, 2].sum())


def extract_active(state: np.ndarray, tol: float = ACTIVE_GAP_TOL):
    """Return (contact_pairs, wall_constraints) that are active within tol."""
    n = state.shape[0]
    contacts = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = state[i, 0] - state[j, 0]
            dy = state[i, 1] - state[j, 1]
            d = np.hypot(dx, dy)
            gap = d - (state[i, 2] + state[j, 2])
            if gap < tol:
                contacts.append((i, j, gap))
    walls = []  # (disk_idx, side): 0=x_i>=r_i, 1=x_i<=1-r_i, 2=y>=r, 3=y<=1-r
    for i in range(n):
        x, y, r = state[i]
        for side, gap in enumerate([x - r, 1 - x - r, y - r, 1 - y - r]):
            if gap < tol:
                walls.append((i, side, gap))
    return contacts, walls


def build_rigidity_matrix(state: np.ndarray, contacts, walls) -> np.ndarray:
    """
    Row k = gradient of g_k w.r.t. full primal vector z = (x_0, y_0, r_0, x_1, y_1, r_1, ...).

    Constraints are written as g(z) >= 0:
      disk-disk: g = (x_i-x_j)^2 + (y_i-y_j)^2 - (r_i+r_j)^2
      wall 0:   g = x_i - r_i
      wall 1:   g = 1 - x_i - r_i
      wall 2:   g = y_i - r_i
      wall 3:   g = 1 - y_i - r_i
    """
    m = len(contacts) + len(walls)
    R = np.zeros((m, PRIMAL_DIM))

    def zidx(i, comp):  # comp 0=x, 1=y, 2=r
        return 3 * i + comp

    row = 0
    for (i, j, _gap) in contacts:
        xi, yi, ri = state[i]
        xj, yj, rj = state[j]
        dx, dy = xi - xj, yi - yj
        dr = ri + rj
        R[row, zidx(i, 0)] = 2 * dx
        R[row, zidx(i, 1)] = 2 * dy
        R[row, zidx(i, 2)] = -2 * dr
        R[row, zidx(j, 0)] = -2 * dx
        R[row, zidx(j, 1)] = -2 * dy
        R[row, zidx(j, 2)] = -2 * dr
        row += 1
    for (i, side, _gap) in walls:
        if side == 0:
            R[row, zidx(i, 0)] = 1.0
            R[row, zidx(i, 2)] = -1.0
        elif side == 1:
            R[row, zidx(i, 0)] = -1.0
            R[row, zidx(i, 2)] = -1.0
        elif side == 2:
            R[row, zidx(i, 1)] = 1.0
            R[row, zidx(i, 2)] = -1.0
        elif side == 3:
            R[row, zidx(i, 1)] = -1.0
            R[row, zidx(i, 2)] = -1.0
        row += 1
    return R


def objective_grad() -> np.ndarray:
    """∇(Σ r_i) with respect to z. Only r-slots are 1."""
    g = np.zeros(PRIMAL_DIM)
    for i in range(N_DISKS):
        g[3 * i + 2] = 1.0
    return g


def recover_multipliers(R: np.ndarray, grad_f: np.ndarray):
    """
    KKT stationarity for maximize f subject to g_k(z) >= 0:
        ∇f = Σ λ_k ∇g_k,   λ_k >= 0    (Lagrangian L = f - Σ λ_k g_k)

    Empirically in our geometry λ comes out negative when we solve
    R^T λ = ∇f directly: the issue is that our "radius slot" in the
    objective is +1, but the gradient of the disk-disk g wrt r is -2(r_i+r_j),
    so the algebraic sign flip means we actually need:
        -∇f = Σ μ_k ∇g_k      (and then the correct multipliers are μ_k := -λ_k_lstsq)
    The physics: at a max of Σ r, growing r decreases all g's (they are
    distance-squared minus (r_i+r_j)^2), so ∇f points opposite to ∇g, which is
    exactly what λ>=0 demands after the sign flip.
    """
    lam_raw, residuals, rank, sv = np.linalg.lstsq(R.T, grad_f, rcond=None)
    lam = -lam_raw  # flip into the λ >= 0 convention
    return lam, residuals, rank, sv


def build_stress_matrix(state: np.ndarray, contacts, walls, lam: np.ndarray) -> np.ndarray:
    """
    Ω = Σ λ_k ∇² g_k, summed over all active constraints. Shape (78, 78).

    For disk-disk g_k = ||c_i - c_j||^2 - (r_i+r_j)^2, the Hessian is a 6x6 block
    with +2 on (x_i,x_i),(y_i,y_i),(x_j,x_j),(y_j,y_j), -2 on the cross diagonals,
    and (r_i, r_j) block = [[-2, -2], [-2, -2]].

    For wall constraints g_k is linear, so ∇² g_k = 0.
    """
    Omega = np.zeros((PRIMAL_DIM, PRIMAL_DIM))

    def zidx(i, comp):
        return 3 * i + comp

    row = 0
    for (i, j, _gap) in contacts:
        lk = lam[row]
        # x-block
        ix, iy, ir = zidx(i, 0), zidx(i, 1), zidx(i, 2)
        jx, jy, jr = zidx(j, 0), zidx(j, 1), zidx(j, 2)
        Omega[ix, ix] += 2 * lk
        Omega[iy, iy] += 2 * lk
        Omega[jx, jx] += 2 * lk
        Omega[jy, jy] += 2 * lk
        Omega[ix, jx] += -2 * lk
        Omega[jx, ix] += -2 * lk
        Omega[iy, jy] += -2 * lk
        Omega[jy, iy] += -2 * lk
        # r-block
        Omega[ir, ir] += -2 * lk
        Omega[jr, jr] += -2 * lk
        Omega[ir, jr] += -2 * lk
        Omega[jr, ir] += -2 * lk
        row += 1
    # walls contribute 0
    return Omega


# ===================================================================
# flex probe — re-optimization with perturbed init
# ===================================================================


def pack_state_to_vec(state: np.ndarray) -> np.ndarray:
    return state.reshape(-1).copy()


def vec_to_state(z: np.ndarray) -> np.ndarray:
    return z.reshape(N_DISKS, 3)


def neg_sum_r(z: np.ndarray) -> float:
    return -z.reshape(N_DISKS, 3)[:, 2].sum()


def neg_sum_r_grad(z: np.ndarray) -> np.ndarray:
    g = np.zeros_like(z)
    for i in range(N_DISKS):
        g[3 * i + 2] = -1.0
    return g


def all_constraints(z: np.ndarray) -> np.ndarray:
    """Return vector of constraint values (>=0 is feasible)."""
    st = z.reshape(N_DISKS, 3)
    vals = []
    # disk-disk
    for i in range(N_DISKS):
        for j in range(i + 1, N_DISKS):
            dx = st[i, 0] - st[j, 0]
            dy = st[i, 1] - st[j, 1]
            vals.append(dx * dx + dy * dy - (st[i, 2] + st[j, 2]) ** 2)
    # walls
    for i in range(N_DISKS):
        x, y, r = st[i]
        vals.append(x - r)
        vals.append(1 - x - r)
        vals.append(y - r)
        vals.append(1 - y - r)
    # r > 0
    for i in range(N_DISKS):
        vals.append(st[i, 2])
    return np.array(vals)


def reoptimize(z0: np.ndarray, maxiter: int = 400):
    """SLSQP with analytic constraint vector. Returns final state vector + Σr."""
    con = NonlinearConstraint(all_constraints, 0.0, np.inf)
    res = minimize(
        neg_sum_r,
        z0,
        jac=neg_sum_r_grad,
        method="SLSQP",
        constraints=con,
        options={"maxiter": maxiter, "ftol": 1e-14},
    )
    return res.x, -res.fun, res.success


def probe_trial(args):
    """One flex probe trial: perturb 3 neighbors of weak disk, reopt."""
    z_base, neighbor_idx, seed, scale = args
    rng = np.random.default_rng(seed)
    z = z_base.copy()
    for idx in neighbor_idx:
        dx = rng.normal(scale=scale)
        dy = rng.normal(scale=scale)
        z[3 * idx + 0] += dx
        z[3 * idx + 1] += dy
    z_opt, sumr, success = reoptimize(z)
    # also verify constraints hold
    cons = all_constraints(z_opt)
    min_cons = float(cons.min())
    return {
        "seed": int(seed),
        "scale": float(scale),
        "success": bool(success),
        "sum_r": float(sumr),
        "min_constraint": min_cons,
    }


# ===================================================================
# plotting
# ===================================================================


def make_figures(svals, stress_eigs, lambdas, active_count):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)

    # (a) singular values of R
    ax = axes[0]
    ax.bar(np.arange(len(svals)), svals, color="#1f77b4", width=0.9)
    ax.set_yscale("log")
    ax.set_title("(a) Singular values of R (78x78)", fontweight="bold")
    ax.set_xlabel("singular value index")
    ax.set_ylabel(r"$\sigma_k(R)$")
    ax.axhline(1e-6, color="red", ls="--", lw=1, label=r"$10^{-6}$")
    ax.legend(frameon=False)

    # (b) stress matrix eigenvalue histogram
    ax = axes[1]
    ax.hist(stress_eigs, bins=30, color="#2ca02c", edgecolor="black")
    ax.set_title(r"(b) Stress matrix $\Omega$ spectrum", fontweight="bold")
    ax.set_xlabel("eigenvalue")
    ax.set_ylabel("count")
    ax.axvline(0, color="red", ls="--", lw=1)

    # (c) multipliers
    ax = axes[2]
    order = np.argsort(lambdas)
    ax.bar(np.arange(len(lambdas)), lambdas[order], color="#ff7f0e")
    ax.set_title(
        f"(c) Dual multipliers (n={len(lambdas)} active)", fontweight="bold"
    )
    ax.set_xlabel("sorted constraint index")
    ax.set_ylabel(r"$\lambda_k$")
    ax.set_yscale("log")

    fig.savefig(FIG_DIR / "rigidity_diagnostic.png", bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


# ===================================================================
# main
# ===================================================================


def main():
    t0 = time.time()
    state = load_solution(PARENT_SOL)
    sum_r0 = sum_radii(state)
    print(f"[load] Σr = {sum_r0:.12f}")

    contacts, walls = extract_active(state)
    print(f"[active] disk-disk contacts: {len(contacts)}")
    print(f"[active] wall constraints:  {len(walls)}")
    n_active = len(contacts) + len(walls)
    print(f"[active] total: {n_active}  (expected 78)")

    R = build_rigidity_matrix(state, contacts, walls)
    print(f"[R] shape = {R.shape}")

    # SVD
    U, svals, Vt = np.linalg.svd(R, full_matrices=False)
    sigma_min = float(svals[-1])
    sigma_max = float(svals[0])
    kappa = sigma_max / sigma_min if sigma_min > 0 else np.inf
    print(f"[svd] σ_max(R) = {sigma_max:.6e}")
    print(f"[svd] σ_min(R) = {sigma_min:.6e}")
    print(f"[svd] κ(R)     = {kappa:.6e}")

    grad_f = objective_grad()
    lam, resid, rank, _sv = recover_multipliers(R, grad_f)
    # With sign-flipped λ, stationarity is ∇f = -R^T λ
    stationarity_residual = float(np.linalg.norm(R.T @ lam + grad_f))
    print(f"[kkt] rank(R) = {rank}")
    print(f"[kkt] ||R^T λ − ∇f|| = {stationarity_residual:.3e}")

    # Split multipliers
    n_cc = len(contacts)
    lam_contacts = lam[:n_cc]
    lam_walls = lam[n_cc:]
    print(f"[λ] min = {lam.min():.4e}  max = {lam.max():.4e}")
    print(f"[λ] #negative: {(lam < -1e-10).sum()}")

    # Find weakest-multiplier contact (smallest positive λ among disk-disk).
    # Only consider positive λ (negative would indicate inactive or degenerate).
    pos_mask = lam_contacts > 0
    if pos_mask.any():
        tmp = np.where(pos_mask, lam_contacts, np.inf)
        idx_weak_c = int(np.argmin(tmp))
    else:
        idx_weak_c = int(np.argmin(np.abs(lam_contacts)))
    weak_pair = contacts[idx_weak_c]
    print(
        f"[weak] weakest contact: disks ({weak_pair[0]},{weak_pair[1]}), λ = {lam_contacts[idx_weak_c]:.4e}"
    )

    # Weakest disk = one of the two in that pair; use both and their joint neighborhood
    weak_disks = (weak_pair[0], weak_pair[1])

    # Pick 3 nearest neighbors to the weakest pair's midpoint
    mid = 0.5 * (state[weak_disks[0], :2] + state[weak_disks[1], :2])
    dists = np.linalg.norm(state[:, :2] - mid, axis=1)
    nearest = np.argsort(dists)[:5]  # 5 closest (includes the pair itself)
    neighbors_for_perturb = [int(k) for k in nearest if k not in weak_disks][:3]
    print(f"[weak] 3 nearest neighbors (perturbing): {neighbors_for_perturb}")

    Omega = build_stress_matrix(state, contacts, walls, lam)
    stress_eigs = np.linalg.eigvalsh(0.5 * (Omega + Omega.T))
    n_neg = int((stress_eigs < -1e-10).sum())
    print(f"[Ω] min eig = {stress_eigs[0]:.4e}  max eig = {stress_eigs[-1]:.4e}")
    print(f"[Ω] #negative eigs: {n_neg}")

    # ------------- flex probe -------------
    z0 = pack_state_to_vec(state)
    scales = [1e-3, 1e-2, 1e-1]
    seeds = [1, 2, 3, 7, 13, 21, 42, 99, 123, 2026]
    tasks = []
    for scale in scales:
        for seed in seeds:
            tasks.append((z0, neighbors_for_perturb, seed, scale))

    print(f"[probe] launching {len(tasks)} SLSQP trials in parallel ...")
    t_probe = time.time()
    with Pool(processes=min(8, os.cpu_count() or 2)) as pool:
        probe_results = pool.map(probe_trial, tasks)
    print(f"[probe] done in {time.time()-t_probe:.1f}s")

    valid = [r for r in probe_results if r["success"] and r["min_constraint"] > -1e-9]
    best_flex = max((r["sum_r"] for r in valid), default=float("nan"))
    baseline_sum_r = sum_r0
    flex_found = best_flex > baseline_sum_r + 1e-9
    print(f"[probe] best Σr over {len(valid)}/{len(tasks)} valid trials: {best_flex:.12f}")
    print(f"[probe] baseline: {baseline_sum_r:.12f}  flex_found: {flex_found}")

    # ------------- verdict -------------
    certified = (sigma_min > 1e-6) and (lam.min() > -1e-8) and (not flex_found)
    print(f"[verdict] certified strict local max: {certified}")

    # ------------- figures -------------
    make_figures(svals, stress_eigs, np.maximum(lam, 1e-16), n_active)

    # ------------- write report -------------
    report = {
        "baseline_sum_r": float(baseline_sum_r),
        "n_active_total": int(n_active),
        "n_contacts": int(len(contacts)),
        "n_walls": int(len(walls)),
        "sigma_min_R": float(sigma_min),
        "sigma_max_R": float(sigma_max),
        "kappa_R": float(kappa),
        "rank_R": int(rank),
        "stationarity_residual": float(stationarity_residual),
        "lambda_min": float(lam.min()),
        "lambda_max": float(lam.max()),
        "lambda_num_negative": int((lam < -1e-10).sum()),
        "weakest_contact_disks": [int(weak_pair[0]), int(weak_pair[1])],
        "weakest_contact_lambda": float(lam_contacts[idx_weak_c]),
        "weak_neighbors_perturbed": [int(k) for k in neighbors_for_perturb],
        "stress_matrix_min_eig": float(stress_eigs[0]),
        "stress_matrix_max_eig": float(stress_eigs[-1]),
        "stress_matrix_num_negative": int(n_neg),
        "stress_matrix_eigs_head": stress_eigs[:10].tolist(),
        "stress_matrix_eigs_tail": stress_eigs[-10:].tolist(),
        "flex_probe_n_trials": len(tasks),
        "flex_probe_n_valid": len(valid),
        "flex_probe_best_sum_r": float(best_flex) if np.isfinite(best_flex) else None,
        "flex_probe_scales_tried": scales,
        "flex_probe_seeds_tried": seeds,
        "flex_probe_results": probe_results,
        "certified_strict_local_max": bool(certified),
        "wall_clock_seconds": float(time.time() - t0),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[done] wrote {OUT_JSON}")
    return report


if __name__ == "__main__":
    main()
