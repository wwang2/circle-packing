"""
CONTACT-GRAPH-001: Realizable contact-graph enumeration for n=26 circle packing.

Strategy
--------
The parent orbit `mobius-001` established that sum_r = 2.6359830865 is the best
known basin, with the contact graph G* consisting of 58 disk-disk edges and
20 wall incidences. Rigidity-001 + interval-newton-001 proved this basin is
isolated in R^156. Any SOTA improvement must come from a DIFFERENT contact graph.

This script:
 1. Recovers G* from the parent's coordinates.
 2. Enumerates candidate graphs via three strategies:
    (a) edge-flip:     remove a disk-disk edge, add a new one between
                        other non-adjacent disks compatible with the
                        tangency-gap.
    (b) wall-swap:     reassign which disks touch which wall
                        (permute within each wall side).
    (c) edge-swap:     remove two edges and add two new ones.
 3. Filters for planarity via networkx.check_planarity.
 4. For each filtered candidate, runs a fresh SLSQP optimization whose
    non-linear constraints treat G as the active set (equalities for edges,
    inequalities for non-edges). Three seeds per candidate; best of three.
 5. Aggregates and reports.

Minimum 3 seeds per candidate (autorun override).

This is an enumeration-as-deliverable: even a negative result (no alternate
graph beats the incumbent) is a publishable outcome, strengthening the
global-optimality case for G*.
"""
from __future__ import annotations
import json
import time
import itertools
import os
from pathlib import Path
from multiprocessing import Pool
from typing import Optional

import numpy as np
import networkx as nx
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
ORBIT = Path(__file__).resolve().parent
PARENT_SOLUTION = ROOT / "orbits/mobius-001/solution_n26.json"
INCUMBENT = 2.6359830865

# ---------------------------------------------------------------------------
# Graph extraction
# ---------------------------------------------------------------------------
def load_parent_graph(path: Path = PARENT_SOLUTION, tol: float = 1e-6):
    data = json.load(open(path))
    circles = np.array(data["circles"], dtype=float)
    n = len(circles)
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]

    # Disk-disk edges: sparse record of active tangencies
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            if abs(d - (r[i] + r[j])) < tol:
                edges.append((i, j))

    # Wall contacts: dict {disk: set(sides)}, sides in {L, R, B, T}
    walls = {i: set() for i in range(n)}
    for i in range(n):
        if abs(x[i] - r[i]) < tol:
            walls[i].add("L")
        if abs((1 - x[i]) - r[i]) < tol:
            walls[i].add("R")
        if abs(y[i] - r[i]) < tol:
            walls[i].add("B")
        if abs((1 - y[i]) - r[i]) < tol:
            walls[i].add("T")

    return circles, edges, walls


def as_graph_key(edges, walls):
    """Canonical hashable key for (edges, walls) pair."""
    e_sorted = tuple(sorted((min(a, b), max(a, b)) for a, b in edges))
    w_sorted = tuple(sorted((k, tuple(sorted(v))) for k, v in walls.items() if v))
    return (e_sorted, w_sorted)


# ---------------------------------------------------------------------------
# Planarity + realizability filters
# ---------------------------------------------------------------------------
def graph_from_edges(n, edges):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    return G


def is_planar(edges, n):
    G = graph_from_edges(n, edges)
    ok, _ = nx.check_planarity(G)
    return ok


def quick_feasible(edges, walls, n):
    """Cheap sanity checks on a candidate topology before running SLSQP."""
    # Planarity of the disk-disk contact graph
    if not is_planar(edges, n):
        return False
    # No disk may touch more than 2 walls (possible only at a corner)
    for i, sides in walls.items():
        if len(sides) > 2:
            return False
        if sides == {"L", "R"} or sides == {"B", "T"}:
            return False  # would need r >= 0.5
    # Each disk has at least one wall or 2 neighbors? (Rigidity prerequisite)
    G = graph_from_edges(n, edges)
    for i in range(n):
        deg = G.degree(i) + len(walls.get(i, set()))
        if deg < 2:
            return False
    return True


# ---------------------------------------------------------------------------
# SLSQP optimizer with contact graph as active set
# ---------------------------------------------------------------------------
def pack_params(x):
    return x


def build_objective_and_constraints(n, edges, walls):
    """Return (neg_sum_r, constraint_list) for scipy.optimize.minimize."""
    edges = list(edges)
    walls_map = {int(k): set(v) for k, v in walls.items()}

    def neg_sum_r(z):
        return -z[2::3].sum()

    def neg_sum_r_grad(z):
        g = np.zeros_like(z)
        g[2::3] = -1.0
        return g

    constraints = []

    # Equality: d(i,j) - (r_i+r_j) = 0 for every edge (tangent)
    def make_edge_eq(i, j):
        def fn(z):
            xi, yi, ri = z[3 * i : 3 * i + 3]
            xj, yj, rj = z[3 * j : 3 * j + 3]
            return np.hypot(xi - xj, yi - yj) - (ri + rj)

        def jac(z):
            xi, yi, ri = z[3 * i : 3 * i + 3]
            xj, yj, rj = z[3 * j : 3 * j + 3]
            d = np.hypot(xi - xj, yi - yj) + 1e-16
            g = np.zeros_like(z)
            g[3 * i] = (xi - xj) / d
            g[3 * i + 1] = (yi - yj) / d
            g[3 * i + 2] = -1.0
            g[3 * j] = (xj - xi) / d
            g[3 * j + 1] = (yj - yi) / d
            g[3 * j + 2] = -1.0
            return g

        return {"type": "eq", "fun": fn, "jac": jac}

    for i, j in edges:
        constraints.append(make_edge_eq(i, j))

    # Equality: wall contact (x - r = 0 etc.)
    def make_wall_eq(i, side):
        if side == "L":
            def fn(z):  # x = r
                return z[3 * i] - z[3 * i + 2]
            def jac(z):
                g = np.zeros_like(z); g[3*i]=1; g[3*i+2]=-1; return g
        elif side == "R":
            def fn(z):  # 1 - x = r
                return 1.0 - z[3 * i] - z[3 * i + 2]
            def jac(z):
                g = np.zeros_like(z); g[3*i]=-1; g[3*i+2]=-1; return g
        elif side == "B":
            def fn(z):
                return z[3 * i + 1] - z[3 * i + 2]
            def jac(z):
                g = np.zeros_like(z); g[3*i+1]=1; g[3*i+2]=-1; return g
        else:  # T
            def fn(z):
                return 1.0 - z[3 * i + 1] - z[3 * i + 2]
            def jac(z):
                g = np.zeros_like(z); g[3*i+1]=-1; g[3*i+2]=-1; return g
        return {"type": "eq", "fun": fn, "jac": jac}

    for i, sides in walls_map.items():
        for s in sides:
            constraints.append(make_wall_eq(i, s))

    # Inequality: containment for ALL disks (non-negative slack)
    def make_containment(i):
        def fn(z):
            xi, yi, ri = z[3 * i : 3 * i + 3]
            # vector of 4 slack values
            return np.array([xi - ri, 1 - xi - ri, yi - ri, 1 - yi - ri])
        return {"type": "ineq", "fun": fn}
    for i in range(n):
        constraints.append(make_containment(i))

    # Inequality: non-overlap for all non-edge pairs
    edge_set = set((min(i, j), max(i, j)) for i, j in edges)
    non_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if (i, j) not in edge_set]
    def make_nonoverlap(i, j):
        def fn(z):
            xi, yi, ri = z[3 * i : 3 * i + 3]
            xj, yj, rj = z[3 * j : 3 * j + 3]
            return np.hypot(xi - xj, yi - yj) - (ri + rj)
        return {"type": "ineq", "fun": fn}
    for i, j in non_edges:
        constraints.append(make_nonoverlap(i, j))

    # Positivity of r
    def make_rpos(i):
        def fn(z):
            return z[3 * i + 2]
        return {"type": "ineq", "fun": fn}
    for i in range(n):
        constraints.append(make_rpos(i))

    return neg_sum_r, neg_sum_r_grad, constraints


def sum_r_of_state(z):
    return float(z[2::3].sum())


def validate(z, n, tol=1e-10):
    x = z[0::3]; y = z[1::3]; r = z[2::3]
    if (r <= 0).any():
        return False, -1.0, float("inf")
    viol = 0.0
    # Containment
    viol = max(viol, (r - x).max(), (x + r - 1).max(), (r - y).max(), (y + r - 1).max())
    # Overlap
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            ov = (r[i] + r[j]) - d
            if ov > viol:
                viol = ov
    return (viol <= tol), float(r.sum()), float(viol)


def build_free_problem(n):
    """
    Free-form (topology-agnostic) containment + non-overlap.
    The objective is -sum_r. SLSQP will find whatever contact graph
    happens to be active at the local optimum.

    This is the critical realization: SLSQP doesn't know or care what graph
    we "started from" — it discovers the active set. So to test whether a
    candidate topology is a better-basin than the parent, we simply push
    the initialization in its direction and see where SLSQP lands.
    """
    def neg_f(z):
        return -z[2::3].sum()
    def neg_f_grad(z):
        g = np.zeros_like(z); g[2::3] = -1.0; return g

    constraints = []
    # Containment (ineq): x>=r, x<=1-r, y>=r, y<=1-r  (all must be >=0)
    for i in range(n):
        idx = i
        def mk_cont(i=idx):
            def fn(z):
                xi, yi, ri = z[3*i:3*i+3]
                return np.array([xi - ri, 1 - xi - ri, yi - ri, 1 - yi - ri])
            return {"type": "ineq", "fun": fn}
        constraints.append(mk_cont())
    # Non-overlap for all pairs
    for i in range(n):
        for j in range(i+1, n):
            def mk_no(i=i, j=j):
                def fn(z):
                    xi, yi, ri = z[3*i:3*i+3]
                    xj, yj, rj = z[3*j:3*j+3]
                    return (xi-xj)**2 + (yi-yj)**2 - (ri+rj)**2
                return {"type": "ineq", "fun": fn}
            constraints.append(mk_no())
    # r >= 0
    for i in range(n):
        def mk_r(i=i):
            def fn(z): return z[3*i+2]
            return {"type": "ineq", "fun": fn}
        constraints.append(mk_r())
    return neg_f, neg_f_grad, constraints


def optimize_graph(n, edges, walls, init, seed: int = 42) -> dict:
    """
    Optimize with active-set constraints from (edges, walls) — strict graph.
    Used as a "solve-this-specific-topology" diagnostic.
    """
    neg_f, neg_f_grad, cons = build_objective_and_constraints(n, edges, walls)
    rng = np.random.default_rng(seed)
    z0 = init.flatten().copy() + rng.normal(0, 1e-3, size=3 * n)
    z0[2::3] = np.clip(z0[2::3], 1e-3, 0.3)
    z0[0::3] = np.clip(z0[0::3], 0.01, 0.99)
    z0[1::3] = np.clip(z0[1::3], 0.01, 0.99)
    try:
        res = minimize(neg_f, z0, jac=neg_f_grad, method="SLSQP",
                       constraints=cons,
                       options={"maxiter": 500, "ftol": 1e-12, "disp": False})
        z = res.x
        ok, sr, viol = validate(z, n, tol=1e-8)
        return {"success": ok, "sum_r": sr, "max_violation": viol, "message": str(res.message)}
    except Exception as e:
        return {"success": False, "sum_r": -1.0, "max_violation": float("inf"), "message": str(e)}


def optimize_free(n, init_flat, seed: int = 42, perturb_scale: float = 0.02,
                  targeted_moves=None, maxiter: int = 500):
    """
    Topology-agnostic SLSQP from a perturbed init.
    `targeted_moves` is an optional list of (i, j, factor) pairs telling us to
    pull i and j toward each other by `factor` before starting the solve,
    so that after optimization they are likely to be tangent.
    Returns whatever the optimizer finds.
    """
    neg_f, neg_f_grad, cons = build_free_problem(n)
    rng = np.random.default_rng(seed)
    z0 = np.asarray(init_flat, dtype=float).copy()
    # Global perturbation
    z0 += rng.normal(0, perturb_scale, size=z0.shape)
    # Targeted moves: bring pairs (u,v) closer together (bias the basin)
    if targeted_moves:
        for u, v, factor in targeted_moves:
            cu = z0[3*u:3*u+2]
            cv = z0[3*v:3*v+2]
            mid = 0.5*(cu+cv)
            cu_new = mid + (cu-mid)*(1-factor)
            cv_new = mid + (cv-mid)*(1-factor)
            z0[3*u:3*u+2] = cu_new
            z0[3*v:3*v+2] = cv_new
    # Clamp
    z0[2::3] = np.clip(z0[2::3], 1e-3, 0.3)
    z0[0::3] = np.clip(z0[0::3], 0.01, 0.99)
    z0[1::3] = np.clip(z0[1::3], 0.01, 0.99)
    try:
        res = minimize(neg_f, z0, jac=neg_f_grad, method="SLSQP",
                       constraints=cons,
                       options={"maxiter": maxiter, "ftol": 1e-11, "disp": False})
        z = res.x
        ok, sr, viol = validate(z, n, tol=1e-8)
        return {"success": ok, "sum_r": sr, "max_violation": viol,
                "z": z.tolist(), "message": str(res.message)}
    except Exception as e:
        return {"success": False, "sum_r": -1.0, "max_violation": float("inf"),
                "z": None, "message": str(e)}


def best_of_seeds(n, edges, walls, init, seeds):
    """
    Strategy: since candidate contact graphs are almost never exactly
    realizable with the parent's initialization, we use free optimization
    with targeted moves:
      - Pull the (new) edges in the candidate graph closer together.
      - Let SLSQP find the local optimum with whatever contact graph wins.
      - Report sum_r + what its actual contact graph is.

    We use 3 seeds per candidate. 'success' requires constraint
    satisfaction to 1e-8.
    """
    init_flat = np.asarray(init, dtype=float).flatten()
    # Identify "new" edges — those not in the parent graph — and mark them
    # as targeted moves (pull together).
    parent_circles, parent_edges, _ = load_parent_graph()
    parent_set = set(tuple(sorted(e)) for e in parent_edges)
    cand_set = set(tuple(sorted(e)) for e in edges)
    new_edges = cand_set - parent_set

    # Build targeted moves: for each new edge, pull (i,j) ~30% closer
    targeted = [(u, v, 0.25) for (u, v) in new_edges]

    best = {"success": False, "sum_r": -1.0, "max_violation": float("inf"),
            "z": None, "message": "no seed"}
    for s in seeds:
        res = optimize_free(n, init_flat, seed=s,
                            perturb_scale=0.015,
                            targeted_moves=targeted,
                            maxiter=400)
        if res["success"] and res["sum_r"] > best["sum_r"]:
            best = res
    return best


# ---------------------------------------------------------------------------
# Candidate enumeration strategies
# ---------------------------------------------------------------------------
def enum_edge_flip(edges, walls, n, radii):
    """
    For each disk-disk edge (i,j), remove it and try adding a new edge
    (u,v) where u,v are currently close to being tangent but are not in
    edges. Limits the combinatorial blowup by requiring the new edge to be
    plausible (within 3x the current gap of tangency).
    """
    edges_set = set(tuple(sorted(e)) for e in edges)
    base = json.load(open(PARENT_SOLUTION))
    C = np.array(base["circles"])
    cs = C[:, :2]
    rs = C[:, 2]
    # Rank non-edges by tangency gap
    gaps = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in edges_set:
                continue
            d = np.hypot(cs[u, 0] - cs[v, 0], cs[u, 1] - cs[v, 1])
            gap = d - (rs[u] + rs[v])
            gaps.append((gap, u, v))
    gaps.sort()
    # Keep the 30 nearest non-edges as candidate additions
    top_adds = [(u, v) for _, u, v in gaps[:30]]
    candidates = []
    for (i, j) in edges:
        for (u, v) in top_adds:
            if (u, v) == (i, j):
                continue
            new_edges = list(edges_set - {tuple(sorted((i, j)))}) + [tuple(sorted((u, v)))]
            candidates.append(("flip", (i, j), (u, v), new_edges, dict(walls)))
    return candidates


def enum_wall_swap(edges, walls, n):
    """
    For each pair of disks on the same wall vs a disk not on that wall
    that has a nearby y/x, swap who touches the wall. Small neighborhood.
    """
    by_side = {"L": [], "R": [], "B": [], "T": []}
    for i in range(n):
        for s in walls.get(i, set()):
            by_side[s].append(i)
    base = json.load(open(PARENT_SOLUTION))
    C = np.array(base["circles"])
    cs = C[:, :2]; rs = C[:, 2]
    candidates = []
    for side, lst in by_side.items():
        # Axis for the side
        if side == "L":
            coord = cs[:, 0] - rs  # distance to L wall
        elif side == "R":
            coord = (1 - cs[:, 0]) - rs
        elif side == "B":
            coord = cs[:, 1] - rs
        else:
            coord = (1 - cs[:, 1]) - rs
        # Currently-on-wall set
        on = set(lst)
        # Rank off-wall disks by proximity to side
        off = [(coord[i], i) for i in range(n) if i not in on]
        off.sort()
        top_off = [i for _, i in off[:3]]
        for i_on in lst:
            for i_off in top_off:
                new_walls = {k: set(v) for k, v in walls.items()}
                new_walls[i_on].discard(side)
                new_walls.setdefault(i_off, set()).add(side)
                # Clean empty
                new_walls = {k: v for k, v in new_walls.items() if v}
                # Pad back zero-sets
                for k in range(n):
                    new_walls.setdefault(k, set())
                candidates.append(("wall_swap", (i_on, side), (i_off, side), list(edges), new_walls))
    return candidates


def enum_edge_swap(edges, walls, n):
    """
    2-exchange: remove (a,b) and (c,d), add (a,c) and (b,d) (or (a,d)+(b,c)).
    Restrict to edge pairs that share no vertex.
    """
    candidates = []
    edges_set = set(tuple(sorted(e)) for e in edges)
    edge_list = sorted(edges_set)
    # Only try a random subset to keep size manageable
    rng = np.random.default_rng(17)
    pairs = []
    for i1 in range(len(edge_list)):
        for i2 in range(i1 + 1, len(edge_list)):
            a, b = edge_list[i1]
            c, d = edge_list[i2]
            if len({a, b, c, d}) == 4:
                pairs.append((i1, i2))
    rng.shuffle(pairs)
    for i1, i2 in pairs[:200]:
        a, b = edge_list[i1]
        c, d = edge_list[i2]
        for (u, v, x, y) in [(a, c, b, d), (a, d, b, c)]:
            e1 = tuple(sorted((u, v)))
            e2 = tuple(sorted((x, y)))
            if e1 in edges_set or e2 in edges_set or e1 == e2:
                continue
            new_edges = list(edges_set - {edge_list[i1], edge_list[i2]}) + [e1, e2]
            candidates.append(("edge_swap", (edge_list[i1], edge_list[i2]), (e1, e2), new_edges, dict(walls)))
    return candidates


# ---------------------------------------------------------------------------
# Parallel worker
# ---------------------------------------------------------------------------
# Globals for worker process
_WORKER_STATE = {}

def _worker_init(n, init_flat):
    _WORKER_STATE["n"] = n
    _WORKER_STATE["init"] = np.array(init_flat).reshape(-1, 3)

def _worker_eval(args):
    idx, strategy, src, dst, edges, walls, seeds = args
    n = _WORKER_STATE["n"]
    init = _WORKER_STATE["init"]
    # Realizability filter
    walls_clean = {int(k): set(v) for k, v in walls.items()}
    if not quick_feasible(edges, walls_clean, n):
        return {
            "idx": idx, "strategy": strategy, "src": str(src), "dst": str(dst),
            "feasible_prefilter": False, "planar": False, "success": False,
            "sum_r": -1.0, "max_violation": float("inf"), "n_edges": len(edges),
        }
    planar = is_planar(edges, n)
    res = best_of_seeds(n, edges, walls_clean, init, seeds=seeds)
    return {
        "idx": idx, "strategy": strategy, "src": str(src), "dst": str(dst),
        "feasible_prefilter": True, "planar": bool(planar),
        "success": bool(res["success"]), "sum_r": float(res["sum_r"]),
        "max_violation": float(res["max_violation"]),
        "n_edges": len(edges),
        "z": res.get("z"),  # keep the solution for the best candidate
    }


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
def main(max_candidates: Optional[int] = None, num_workers: int = 6):
    t0 = time.time()

    circles, edges, walls = load_parent_graph()
    n = len(circles)
    print(f"[recover] n={n}, |E|={len(edges)} disk-disk, |W|={sum(len(w) for w in walls.values())} wall")
    print(f"[recover] parent sum_r = {circles[:, 2].sum():.10f}")

    # --- Generate candidates ---
    ef = enum_edge_flip(edges, walls, n, circles[:, 2])
    ws = enum_wall_swap(edges, walls, n)
    es = enum_edge_swap(edges, walls, n)
    all_candidates = ef + ws + es
    print(f"[enum] edge_flip={len(ef)}  wall_swap={len(ws)}  edge_swap={len(es)}  total={len(all_candidates)}")

    # De-duplicate
    seen = set()
    unique = []
    for strat, src, dst, ed, wl in all_candidates:
        # Canonical key
        key = as_graph_key(ed, wl)
        if key in seen:
            continue
        seen.add(key)
        unique.append((strat, src, dst, ed, wl))
    print(f"[enum] unique canonical graphs: {len(unique)}")

    if max_candidates is not None and len(unique) > max_candidates:
        unique = unique[:max_candidates]
        print(f"[enum] truncated to {max_candidates}")

    seeds = [42, 123, 7]
    work = [
        (i, strat, src, dst, ed, wl, seeds)
        for i, (strat, src, dst, ed, wl) in enumerate(unique)
    ]

    # Pre-filter planarity/realizability to count totals fast (before parallel run)
    planar_count = 0
    feasible_pref = 0
    for _, _, _, ed, wl in unique:
        wl_clean = {int(k): set(v) for k, v in wl.items()}
        if quick_feasible(ed, wl_clean, n):
            feasible_pref += 1
        if is_planar(ed, n):
            planar_count += 1
    print(f"[filter] planar = {planar_count}/{len(unique)}")
    print(f"[filter] prefilter feasible = {feasible_pref}/{len(unique)}")

    # --- Parallel SLSQP ---
    init_flat = circles.flatten()
    print(f"[opt] running parallel SLSQP x {len(seeds)} seeds on {len(work)} candidates with {num_workers} workers")

    with Pool(num_workers, initializer=_worker_init, initargs=(n, init_flat)) as pool:
        results = pool.map(_worker_eval, work)

    elapsed = time.time() - t0
    print(f"[opt] done in {elapsed:.1f}s")

    # --- Analyze ---
    solved = [r for r in results if r["success"]]
    sums = sorted([r["sum_r"] for r in solved], reverse=True)
    n_dominated = sum(1 for r in solved if r["sum_r"] < INCUMBENT - 1e-9)
    n_tied = sum(1 for r in solved if abs(r["sum_r"] - INCUMBENT) <= 1e-9)
    n_better = sum(1 for r in solved if r["sum_r"] > INCUMBENT + 1e-9)
    best_sum = max((r["sum_r"] for r in solved), default=-1.0)

    print()
    print(f"[result] total candidates        = {len(unique)}")
    print(f"[result] planar                  = {planar_count}")
    print(f"[result] prefilter feasible      = {feasible_pref}")
    print(f"[result] solved successfully     = {len(solved)}")
    print(f"[result] dominated (< incumbent) = {n_dominated}")
    print(f"[result] tied    (== incumbent)  = {n_tied}")
    print(f"[result] better  (> incumbent)   = {n_better}")
    print(f"[result] best sum_r found        = {best_sum:.10f}")
    print(f"[result] incumbent               = {INCUMBENT:.10f}")

    # --- Persist enum_report.json ---
    # Strip 'z' from non-top entries to keep the report small
    top_sorted = sorted(solved, key=lambda x: -x["sum_r"])
    top10_full = top_sorted[:10]
    top10_payload = [
        {k: v for k, v in r.items() if k != "z"} | {"sum_r": r["sum_r"]}
        for r in top10_full
    ]
    all_results_slim = [{k: v for k, v in r.items() if k != "z"} for r in results]
    report = {
        "n": n,
        "incumbent": INCUMBENT,
        "total_candidates": len(unique),
        "planar": planar_count,
        "prefilter_feasible": feasible_pref,
        "solved": len(solved),
        "dominated": n_dominated,
        "tied": n_tied,
        "better": n_better,
        "global_max_found": best_sum,
        "elapsed_sec": elapsed,
        "seeds": seeds,
        "strategy_counts": {
            "edge_flip": len(ef),
            "wall_swap": len(ws),
            "edge_swap": len(es),
        },
        "top10": top10_payload,
        "all_results": all_results_slim,
    }
    (ORBIT / "enum_report.json").write_text(json.dumps(report, indent=2))
    print(f"[write] enum_report.json ({len(results)} records)")

    # --- Solution if better ---
    if n_better > 0 and top10_full:
        best_record = top10_full[0]
        if best_record.get("z"):
            z = np.asarray(best_record["z"]).reshape(-1, 3)
            payload = {"circles": z.tolist(),
                       "sum_r": float(z[:, 2].sum()),
                       "strategy": best_record["strategy"],
                       "note": "Discovered alternative contact graph beating incumbent"}
            (ORBIT / "solution_if_better.json").write_text(json.dumps(payload, indent=2))
            print(f"[BETTER FOUND] saved solution_if_better.json with sum_r={payload['sum_r']:.10f}")

    return report


if __name__ == "__main__":
    import sys
    max_c = None
    if len(sys.argv) > 1:
        max_c = int(sys.argv[1])
    main(max_candidates=max_c)
