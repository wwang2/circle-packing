"""
Genetic Topology Search for Circle Packing n=26.

Strategy: Search the DISCRETE space of contact graph topologies using a
genetic algorithm. Each topology defines which circles are tangent to which
other circles or walls. Given a topology, we solve for positions+radii.

Key insight: the known optimum has 78 active constraints = 78 variables (zero DOF).
We need to find a DIFFERENT topology that yields a higher sum of radii.
"""

import json
import numpy as np
from scipy.optimize import minimize, fsolve
import os
import sys
import time
import random
from itertools import combinations
import copy

WORKDIR = os.path.dirname(os.path.abspath(__file__))
N = 26  # number of circles
SEED = 42

# Wall indices: 0=left, 1=right, 2=bottom, 3=top
WALL_LEFT, WALL_RIGHT, WALL_BOTTOM, WALL_TOP = 0, 1, 2, 3


def load_solution(path):
    with open(path) as f:
        data = json.load(f)
    circles = np.array(data["circles"])
    return circles[:, 0], circles[:, 1], circles[:, 2]


def save_solution(x, y, r, path):
    circles = [[float(x[i]), float(y[i]), float(r[i])] for i in range(len(x))]
    with open(path, 'w') as f:
        json.dump({"circles": circles}, f, indent=2)


def extract_topology(x, y, r, tol=1e-6):
    """Extract contact graph from a solution."""
    n = len(x)
    contacts = set()  # (i, j) pairs for circle-circle
    walls = {}  # i -> set of wall indices

    for i in range(n):
        walls[i] = set()
        if abs(x[i] - r[i]) < tol:
            walls[i].add(WALL_LEFT)
        if abs(1 - x[i] - r[i]) < tol:
            walls[i].add(WALL_RIGHT)
        if abs(y[i] - r[i]) < tol:
            walls[i].add(WALL_BOTTOM)
        if abs(1 - y[i] - r[i]) < tol:
            walls[i].add(WALL_TOP)

    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            gap = dist - (r[i] + r[j])
            if abs(gap) < tol:
                contacts.add((i, j))

    return contacts, walls


class Topology:
    """Represents a contact graph topology."""
    def __init__(self, n=N):
        self.n = n
        self.contacts = set()  # set of (i,j) with i<j
        self.walls = {i: set() for i in range(n)}  # i -> set of wall indices
        self.fitness = 0.0
        self.solution = None  # (x, y, r) if solved

    def copy(self):
        t = Topology(self.n)
        t.contacts = set(self.contacts)
        t.walls = {i: set(w) for i, w in self.walls.items()}
        t.fitness = self.fitness
        t.solution = self.solution
        return t

    def n_constraints(self):
        n_wall = sum(len(w) for w in self.walls.values())
        return len(self.contacts) + n_wall

    def add_contact(self, i, j):
        if i > j: i, j = j, i
        if i != j and 0 <= i < self.n and 0 <= j < self.n:
            self.contacts.add((i, j))

    def remove_contact(self, i, j):
        if i > j: i, j = j, i
        self.contacts.discard((i, j))

    def add_wall(self, i, wall):
        self.walls[i].add(wall)

    def remove_wall(self, i, wall):
        self.walls[i].discard(wall)

    def get_constraint_list(self):
        """Return list of active constraints in the same format as kkt_refine.py."""
        active = []
        for i in range(self.n):
            for w in sorted(self.walls[i]):
                if w == WALL_LEFT:
                    active.append(('wall_L', i))
                elif w == WALL_RIGHT:
                    active.append(('wall_R', i))
                elif w == WALL_BOTTOM:
                    active.append(('wall_B', i))
                elif w == WALL_TOP:
                    active.append(('wall_T', i))
        for i, j in sorted(self.contacts):
            active.append(('contact', i, j))
        return active

    def fingerprint(self):
        """Hash for deduplication."""
        parts = []
        for i in range(self.n):
            for w in sorted(self.walls[i]):
                parts.append(f"w{i}_{w}")
        for i, j in sorted(self.contacts):
            parts.append(f"c{i}_{j}")
        return hash(tuple(parts))


def topology_from_solution(x, y, r, tol=1e-6):
    """Create a Topology from a known solution."""
    contacts, walls = extract_topology(x, y, r, tol)
    t = Topology(len(x))
    t.contacts = contacts
    t.walls = walls
    t.fitness = np.sum(r)
    t.solution = (x.copy(), y.copy(), r.copy())
    return t


def solve_topology_slsqp(topo, x0, y0, r0, max_iter=500):
    """Given a topology, optimize positions+radii using SLSQP.

    The topology defines which constraints are ACTIVE (equality).
    We maximize sum(r) subject to:
    - ALL constraints >= 0 (containment + non-overlap)
    - The topology contacts are ENCOURAGED to be active (but not forced)

    We use the topology as a warm-start guide.
    """
    n = topo.n

    def pack_vars(x, y, r):
        return np.concatenate([x, y, r])

    def unpack_vars(v):
        return v[:n], v[n:2*n], v[2*n:3*n]

    def objective(v):
        return -np.sum(v[2*n:3*n])

    def obj_grad(v):
        g = np.zeros(3*n)
        g[2*n:3*n] = -1.0
        return g

    constraints = []

    # Containment: x_i - r_i >= 0, 1-x_i-r_i >= 0, y_i-r_i >= 0, 1-y_i-r_i >= 0
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[i] - v[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[n+i] - v[2*n+i]})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[n+i] - v[2*n+i]})

    # Non-overlap: dist^2 >= (r_i + r_j)^2
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j: (
                    (v[i]-v[j])**2 + (v[n+i]-v[n+j])**2 - (v[2*n+i]+v[2*n+j])**2
                )
            })

    # Positive radii
    bounds = [(0.01, 0.99)]*n + [(0.01, 0.99)]*n + [(0.001, 0.5)]*n

    v0 = pack_vars(x0, y0, r0)

    result = minimize(
        objective, v0, method='SLSQP',
        constraints=constraints, bounds=bounds,
        jac=obj_grad,
        options={'maxiter': max_iter, 'ftol': 1e-14, 'disp': False}
    )

    x, y, r = unpack_vars(result.x)
    return x, y, r, -result.fun


def generate_initial_positions(n, seed=None):
    """Generate random initial positions for n circles in unit square."""
    rng = np.random.RandomState(seed)
    # Start with small circles at random positions
    r = np.full(n, 0.03)
    x = rng.uniform(0.05, 0.95, n)
    y = rng.uniform(0.05, 0.95, n)
    return x, y, r


def perturb_solution(x, y, r, strength=0.05, seed=None):
    """Perturb a solution by moving circles slightly."""
    rng = np.random.RandomState(seed)
    n = len(x)
    x2 = x.copy() + rng.uniform(-strength, strength, n)
    y2 = y.copy() + rng.uniform(-strength, strength, n)
    r2 = r.copy() * rng.uniform(0.8, 1.2, n)

    # Clamp to valid range
    r2 = np.clip(r2, 0.01, 0.49)
    x2 = np.clip(x2, r2 + 0.001, 1 - r2 - 0.001)
    y2 = np.clip(y2, r2 + 0.001, 1 - r2 - 0.001)

    return x2, y2, r2


def mutate_topology(topo, n_changes=5, rng=None):
    """Mutate a topology by changing multiple contacts simultaneously."""
    if rng is None:
        rng = np.random.RandomState()

    t = topo.copy()
    t.fitness = 0.0
    t.solution = None

    for _ in range(n_changes):
        action = rng.randint(4)

        if action == 0:
            # Add a random circle-circle contact
            i, j = rng.randint(0, N, 2)
            if i != j:
                t.add_contact(i, j)

        elif action == 1:
            # Remove a random circle-circle contact
            if t.contacts:
                contact = list(t.contacts)[rng.randint(len(t.contacts))]
                t.remove_contact(*contact)

        elif action == 2:
            # Add a random wall contact
            i = rng.randint(N)
            wall = rng.randint(4)
            t.add_wall(i, wall)

        elif action == 3:
            # Remove a random wall contact
            candidates = [(i, w) for i in range(N) for w in t.walls[i]]
            if candidates:
                i, w = candidates[rng.randint(len(candidates))]
                t.remove_wall(i, w)

    return t


def crossover_topologies(t1, t2, rng=None):
    """Cross two topologies: take contacts from each parent."""
    if rng is None:
        rng = np.random.RandomState()

    child = Topology(N)

    # For each possible contact, take from one parent
    all_contacts = t1.contacts | t2.contacts
    for c in all_contacts:
        if rng.random() < 0.5:
            if c in t1.contacts:
                child.contacts.add(c)
        else:
            if c in t2.contacts:
                child.contacts.add(c)

    # For walls, similarly
    for i in range(N):
        all_walls = t1.walls[i] | t2.walls[i]
        for w in all_walls:
            if rng.random() < 0.5:
                if w in t1.walls[i]:
                    child.walls[i].add(w)
            else:
                if w in t2.walls[i]:
                    child.walls[i].add(w)

    return child


def optimize_from_scratch(x0, y0, r0, max_restarts=3):
    """Run SLSQP from given initial point, return best (x, y, r, metric)."""
    n = len(x0)
    best_metric = 0.0
    best_sol = None

    for restart in range(max_restarts):
        if restart > 0:
            x0p, y0p, r0p = perturb_solution(x0, y0, r0, strength=0.02, seed=restart*1000)
        else:
            x0p, y0p, r0p = x0.copy(), y0.copy(), r0.copy()

        # Build constraints
        constraints = []
        for i in range(n):
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2*n+i]})
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[i] - v[2*n+i]})
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[n+i] - v[2*n+i]})
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[n+i] - v[2*n+i]})

        for i in range(n):
            for j in range(i+1, n):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, j=j: (
                        (v[i]-v[j])**2 + (v[n+i]-v[n+j])**2 - (v[2*n+i]+v[2*n+j])**2
                    )
                })

        bounds = [(0.01, 0.99)]*n + [(0.01, 0.99)]*n + [(0.001, 0.5)]*n

        def objective(v):
            return -np.sum(v[2*n:3*n])

        v0 = np.concatenate([x0p, y0p, r0p])

        result = minimize(
            objective, v0, method='SLSQP',
            constraints=constraints, bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-14, 'disp': False}
        )

        x, y, r = result.x[:n], result.x[n:2*n], result.x[2*n:3*n]
        metric = -result.fun

        if is_feasible(x, y, r) and metric > best_metric:
            best_metric = metric
            best_sol = (x.copy(), y.copy(), r.copy())

    return best_sol, best_metric


def is_feasible(x, y, r, tol=1e-8):
    n = len(x)
    for i in range(n):
        if r[i] <= 0: return False
        if x[i] - r[i] < -tol or 1 - x[i] - r[i] < -tol: return False
        if y[i] - r[i] < -tol or 1 - y[i] - r[i] < -tol: return False
    for i in range(n):
        for j in range(i+1, n):
            dist2 = (x[i]-x[j])**2 + (y[i]-y[j])**2
            if dist2 < (r[i]+r[j])**2 - tol*2*(r[i]+r[j]):
                return False
    return True


def generate_structured_init(pattern, n=26, seed=42):
    """Generate structured initial configurations."""
    rng = np.random.RandomState(seed)

    if pattern == 'hex':
        # Hexagonal close packing
        r0 = 0.08
        x, y, r = [], [], []
        row = 0
        while len(x) < n:
            cols_in_row = 6 if row % 2 == 0 else 5
            x_offset = 0.0 if row % 2 == 0 else r0
            for col in range(cols_in_row):
                if len(x) >= n:
                    break
                cx = r0 + x_offset + col * 2 * r0 * 1.05
                cy = r0 + row * r0 * 1.8
                if cx + r0 <= 1 and cy + r0 <= 1:
                    x.append(cx)
                    y.append(cy)
                    r.append(r0)
            row += 1
        while len(x) < n:
            x.append(rng.uniform(0.1, 0.9))
            y.append(rng.uniform(0.1, 0.9))
            r.append(0.03)
        return np.array(x[:n]), np.array(y[:n]), np.array(r[:n])

    elif pattern == 'concentric':
        # Concentric rings: 1 + 6 + 12 + 7
        x, y, r = [0.5], [0.5], [0.13]
        # Inner ring of 6
        for k in range(6):
            angle = k * np.pi / 3 + rng.uniform(-0.1, 0.1)
            rad = 0.25
            x.append(0.5 + rad * np.cos(angle))
            y.append(0.5 + rad * np.sin(angle))
            r.append(0.09)
        # Middle ring of 12
        for k in range(12):
            angle = k * np.pi / 6 + rng.uniform(-0.05, 0.05)
            rad = 0.40
            x.append(0.5 + rad * np.cos(angle))
            y.append(0.5 + rad * np.sin(angle))
            r.append(0.07)
        # Outer ring of 7
        for k in range(7):
            angle = k * 2 * np.pi / 7 + rng.uniform(-0.1, 0.1)
            rad = 0.45
            x.append(0.5 + rad * np.cos(angle))
            y.append(0.5 + rad * np.sin(angle))
            r.append(0.04)
        x, y, r = np.array(x[:n]), np.array(y[:n]), np.array(r[:n])
        # Clamp
        r = np.clip(r, 0.01, 0.49)
        x = np.clip(x, r+0.001, 1-r-0.001)
        y = np.clip(y, r+0.001, 1-r-0.001)
        return x, y, r

    elif pattern == 'grid_5x5_plus1':
        # 5x5 grid + 1 in center
        x, y, r = [], [], []
        for row in range(5):
            for col in range(5):
                cx = 0.1 + col * 0.2
                cy = 0.1 + row * 0.2
                x.append(cx)
                y.append(cy)
                r.append(0.08)
        x.append(0.5)
        y.append(0.5)
        r.append(0.05)
        return np.array(x[:n]), np.array(y[:n]), np.array(r[:n])

    elif pattern == 'diagonal_bands':
        # Circles arranged in diagonal bands
        x, y, r = [], [], []
        idx = 0
        for band in range(8):
            n_in_band = min(4, n - idx)
            for k in range(n_in_band):
                if idx >= n:
                    break
                t = (k + 0.5) / n_in_band
                cx = t * 0.9 + 0.05
                cy = (band * 0.12 + t * 0.08) % 0.9 + 0.05
                x.append(cx)
                y.append(cy)
                r.append(0.06 + rng.uniform(-0.02, 0.02))
                idx += 1
        while len(x) < n:
            x.append(rng.uniform(0.1, 0.9))
            y.append(rng.uniform(0.1, 0.9))
            r.append(0.03)
        return np.array(x[:n]), np.array(y[:n]), np.array(r[:n])

    elif pattern == 'asymmetric':
        # Intentionally asymmetric: more circles on one side
        x, y, r = [], [], []
        # 16 circles in left half
        for k in range(16):
            x.append(rng.uniform(0.05, 0.50))
            y.append(rng.uniform(0.05, 0.95))
            r.append(0.05 + rng.uniform(-0.02, 0.02))
        # 10 circles in right half, bigger
        for k in range(10):
            x.append(rng.uniform(0.50, 0.95))
            y.append(rng.uniform(0.05, 0.95))
            r.append(0.08 + rng.uniform(-0.02, 0.02))
        return np.array(x[:n]), np.array(y[:n]), np.array(r[:n])

    elif pattern == 'two_big':
        # Start with 2 big circles + 24 smaller
        x, y, r = [], [], []
        x.append(0.3); y.append(0.5); r.append(0.25)
        x.append(0.75); y.append(0.5); r.append(0.20)
        for k in range(24):
            angle = rng.uniform(0, 2*np.pi)
            rad = rng.uniform(0.3, 0.5)
            cx = 0.5 + rad * np.cos(angle)
            cy = 0.5 + rad * np.sin(angle)
            cx = np.clip(cx, 0.05, 0.95)
            cy = np.clip(cy, 0.05, 0.95)
            x.append(cx)
            y.append(cy)
            r.append(0.04 + rng.uniform(0, 0.03))
        return np.array(x[:n]), np.array(y[:n]), np.array(r[:n])

    elif pattern == 'corner_focused':
        # Big circles in corners, small ones fill gaps
        x, y, r = [], [], []
        corners = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
        for cx, cy in corners:
            x.append(cx); y.append(cy); r.append(0.14)
        # Mid-edges
        edges = [(0.5, 0.08), (0.5, 0.92), (0.08, 0.5), (0.92, 0.5)]
        for cx, cy in edges:
            x.append(cx); y.append(cy); r.append(0.08)
        # Fill rest
        for k in range(n - 8):
            x.append(rng.uniform(0.1, 0.9))
            y.append(rng.uniform(0.1, 0.9))
            r.append(0.05 + rng.uniform(-0.02, 0.02))
        return np.array(x[:n]), np.array(y[:n]), np.array(r[:n])

    else:
        # Random
        return generate_initial_positions(n, seed)


def multi_start_optimize(n_starts=50, seed=42):
    """Run optimization from many diverse starting points."""
    rng = np.random.RandomState(seed)

    patterns = ['hex', 'concentric', 'grid_5x5_plus1', 'diagonal_bands',
                'asymmetric', 'two_big', 'corner_focused']

    best_metric = 0.0
    best_sol = None
    results = []

    # Load known best for comparison
    known_path = os.path.join(WORKDIR, '..', 'topo-001', 'solution_n26.json')
    if os.path.exists(known_path):
        xk, yk, rk = load_solution(known_path)
        known_metric = np.sum(rk)
    else:
        known_metric = 0

    for i in range(n_starts):
        t0 = time.time()

        if i < len(patterns):
            pattern = patterns[i]
            x0, y0, r0 = generate_structured_init(pattern, N, seed=seed+i)
        elif i < len(patterns) + 10 and known_metric > 0:
            # Perturbed known best with LARGE perturbation
            strength = 0.05 + 0.15 * (i - len(patterns)) / 10
            x0, y0, r0 = perturb_solution(xk, yk, rk, strength=strength, seed=seed+i)
        else:
            # Random
            x0, y0, r0 = generate_initial_positions(N, seed=seed+i)

        sol, metric = optimize_from_scratch(x0, y0, r0, max_restarts=1)
        elapsed = time.time() - t0

        if sol is not None:
            topo = topology_from_solution(*sol)
            fp = topo.fingerprint()
            results.append((metric, fp, elapsed))

            if metric > best_metric:
                best_metric = metric
                best_sol = sol
                print(f"  [{i+1}/{n_starts}] NEW BEST: {metric:.10f} "
                      f"(contacts={len(topo.contacts)}, walls={sum(len(w) for w in topo.walls.values())}) "
                      f"[{elapsed:.1f}s]")
            else:
                if i < 20 or metric > best_metric - 0.01:
                    print(f"  [{i+1}/{n_starts}] metric={metric:.10f} [{elapsed:.1f}s]")
        else:
            results.append((0, 0, elapsed))
            if i < 20:
                print(f"  [{i+1}/{n_starts}] FAILED [{elapsed:.1f}s]")

    return best_sol, best_metric, results


def swap_circles_in_solution(x, y, r, pairs_to_swap, seed=42):
    """Swap positions of circle pairs to create new arrangement."""
    x2, y2, r2 = x.copy(), y.copy(), r.copy()
    for i, j in pairs_to_swap:
        x2[i], x2[j] = x2[j], x2[i]
        y2[i], y2[j] = y2[j], y2[i]
        # Keep radii in place (don't swap) - this breaks topology
    return x2, y2, r2


def targeted_topology_search(base_x, base_y, base_r, n_attempts=100, seed=42):
    """
    More targeted approach: take the known best, make specific structural changes.

    Key idea: swap circle assignments. In the known solution, circle i has
    specific neighbors. If we swap which physical positions circles occupy,
    we get a different topology.
    """
    rng = np.random.RandomState(seed)
    n = len(base_x)
    base_metric = np.sum(base_r)

    best_metric = base_metric
    best_sol = (base_x.copy(), base_y.copy(), base_r.copy())

    print(f"\nTargeted topology search (base metric={base_metric:.10f})")

    for attempt in range(n_attempts):
        # Strategy: permute 3-6 circles then re-optimize
        n_swap = rng.randint(2, 7)
        indices = rng.choice(n, n_swap, replace=False)
        perm = indices.copy()
        rng.shuffle(perm)

        x2, y2, r2 = base_x.copy(), base_y.copy(), base_r.copy()
        # Move circles to permuted positions but keep original radii
        for orig, new in zip(indices, perm):
            x2[new] = base_x[orig]
            y2[new] = base_y[orig]
            # Assign slightly perturbed radius
            r2[new] = base_r[orig] * rng.uniform(0.9, 1.1)

        # Add some random perturbation
        x2 += rng.uniform(-0.02, 0.02, n)
        y2 += rng.uniform(-0.02, 0.02, n)
        r2 = np.clip(r2, 0.01, 0.49)
        x2 = np.clip(x2, r2+0.001, 1-r2-0.001)
        y2 = np.clip(y2, r2+0.001, 1-r2-0.001)

        sol, metric = optimize_from_scratch(x2, y2, r2, max_restarts=1)

        if sol is not None and metric > best_metric:
            best_metric = metric
            best_sol = sol
            topo = topology_from_solution(*sol)
            print(f"  [attempt {attempt+1}] NEW BEST: {metric:.10f} "
                  f"(swapped {n_swap} circles, contacts={len(topo.contacts)})")
        elif attempt < 10 and sol is not None:
            print(f"  [attempt {attempt+1}] metric={metric:.10f}")

    return best_sol, best_metric


def reflect_and_optimize(base_x, base_y, base_r, seed=42):
    """Try reflections, rotations, and partial reflections."""
    rng = np.random.RandomState(seed)
    n = len(base_x)
    base_metric = np.sum(base_r)

    best_metric = base_metric
    best_sol = (base_x.copy(), base_y.copy(), base_r.copy())

    transforms = []

    # Full reflections
    transforms.append(('reflect_x', lambda x, y: (1-x, y)))
    transforms.append(('reflect_y', lambda x, y: (x, 1-y)))
    transforms.append(('reflect_diag', lambda x, y: (y, x)))
    transforms.append(('rotate_90', lambda x, y: (y, 1-x)))
    transforms.append(('rotate_180', lambda x, y: (1-x, 1-y)))

    # Partial transforms: reflect only some circles
    for _ in range(20):
        k = rng.randint(5, 15)
        subset = rng.choice(n, k, replace=False)
        transforms.append(('partial_reflect_x_' + str(k),
                          lambda x, y, s=subset: partial_transform(x, y, s, lambda a, b: (1-a, b))))

    print(f"\nReflection/rotation search (base metric={base_metric:.10f})")

    for name, transform in transforms:
        x2, y2 = base_x.copy(), base_y.copy()

        if name.startswith('partial'):
            x2, y2 = transform(x2, y2)
        else:
            x2, y2 = transform(x2, y2)

        r2 = base_r.copy()
        r2 = np.clip(r2, 0.01, 0.49)
        x2 = np.clip(x2, r2+0.001, 1-r2-0.001)
        y2 = np.clip(y2, r2+0.001, 1-r2-0.001)

        sol, metric = optimize_from_scratch(x2, y2, r2, max_restarts=1)

        if sol is not None and metric > best_metric:
            best_metric = metric
            best_sol = sol
            print(f"  [{name}] NEW BEST: {metric:.10f}")
        elif metric > base_metric - 0.01:
            print(f"  [{name}] metric={metric:.10f}")

    return best_sol, best_metric


def partial_transform(x, y, subset, transform_fn):
    x2, y2 = x.copy(), y.copy()
    tx, ty = transform_fn(x[subset], y[subset])
    x2[subset] = tx
    y2[subset] = ty
    return x2, y2


def remove_reinsert_search(base_x, base_y, base_r, n_attempts=50, seed=42):
    """Remove 2-4 circles, re-optimize remaining, then try to insert them differently."""
    rng = np.random.RandomState(seed)
    n = len(base_x)
    base_metric = np.sum(base_r)

    best_metric = base_metric
    best_sol = (base_x.copy(), base_y.copy(), base_r.copy())

    print(f"\nRemove-reinsert search (base metric={base_metric:.10f})")

    for attempt in range(n_attempts):
        n_remove = rng.randint(2, 5)
        to_remove = sorted(rng.choice(n, n_remove, replace=False))

        # Keep the remaining circles
        keep = [i for i in range(n) if i not in to_remove]
        x_keep = base_x[keep]
        y_keep = base_y[keep]
        r_keep = base_r[keep]

        # Try to place removed circles in random positions
        x2 = np.zeros(n)
        y2 = np.zeros(n)
        r2 = np.zeros(n)

        x2[keep] = x_keep
        y2[keep] = y_keep
        r2[keep] = r_keep

        for idx in to_remove:
            # Random position, small radius
            x2[idx] = rng.uniform(0.05, 0.95)
            y2[idx] = rng.uniform(0.05, 0.95)
            r2[idx] = 0.02

        sol, metric = optimize_from_scratch(x2, y2, r2, max_restarts=1)

        if sol is not None and metric > best_metric:
            best_metric = metric
            best_sol = sol
            print(f"  [attempt {attempt+1}] NEW BEST: {metric:.10f} (removed {to_remove})")
        elif attempt < 5 and sol is not None:
            print(f"  [attempt {attempt+1}] metric={metric:.10f}")

    return best_sol, best_metric


def main():
    t0 = time.time()
    np.random.seed(SEED)
    random.seed(SEED)

    # Load known best
    known_path = os.path.join(WORKDIR, '..', 'topo-001', 'solution_n26.json')
    xk, yk, rk = load_solution(known_path)
    known_metric = np.sum(rk)

    print(f"Known best metric: {known_metric:.10f}")

    # Extract known topology
    known_topo = topology_from_solution(xk, yk, rk)
    print(f"Known topology: {len(known_topo.contacts)} circle-circle, "
          f"{sum(len(w) for w in known_topo.walls.values())} wall contacts")
    print(f"Total constraints: {known_topo.n_constraints()}")

    overall_best_metric = known_metric
    overall_best_sol = (xk.copy(), yk.copy(), rk.copy())

    # Phase 1: Multi-start from diverse initializations
    print("\n" + "="*60)
    print("PHASE 1: Multi-start optimization from diverse initializations")
    print("="*60)
    sol1, metric1, results1 = multi_start_optimize(n_starts=40, seed=SEED)
    if metric1 > overall_best_metric:
        overall_best_metric = metric1
        overall_best_sol = sol1
        print(f"Phase 1 improved: {overall_best_metric:.10f}")

    # Count unique topologies
    fps = set(r[1] for r in results1 if r[0] > 2.5)
    print(f"Unique topologies with metric > 2.5: {len(fps)}")

    # Phase 2: Targeted topology changes
    print("\n" + "="*60)
    print("PHASE 2: Targeted topology search (permute circles)")
    print("="*60)
    sol2, metric2 = targeted_topology_search(xk, yk, rk, n_attempts=80, seed=SEED+100)
    if metric2 > overall_best_metric:
        overall_best_metric = metric2
        overall_best_sol = sol2
        print(f"Phase 2 improved: {overall_best_metric:.10f}")

    # Phase 3: Reflections and rotations
    print("\n" + "="*60)
    print("PHASE 3: Reflection/rotation search")
    print("="*60)
    sol3, metric3 = reflect_and_optimize(xk, yk, rk, seed=SEED+200)
    if metric3 > overall_best_metric:
        overall_best_metric = metric3
        overall_best_sol = sol3
        print(f"Phase 3 improved: {overall_best_metric:.10f}")

    # Phase 4: Remove-reinsert
    print("\n" + "="*60)
    print("PHASE 4: Remove-reinsert search")
    print("="*60)
    sol4, metric4 = remove_reinsert_search(xk, yk, rk, n_attempts=50, seed=SEED+300)
    if metric4 > overall_best_metric:
        overall_best_metric = metric4
        overall_best_sol = sol4
        print(f"Phase 4 improved: {overall_best_metric:.10f}")

    # Save best solution
    sol_path = os.path.join(WORKDIR, 'solution_n26.json')
    save_solution(*overall_best_sol, sol_path)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"FINAL RESULT: metric={overall_best_metric:.10f}")
    print(f"Known best:   metric={known_metric:.10f}")
    print(f"Improvement:  {overall_best_metric - known_metric:.2e}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Solution saved to: {sol_path}")

    return overall_best_metric


if __name__ == '__main__':
    main()
