"""Run optimizer across multiple seeds, keep best solution."""

import json
import numpy as np
from pathlib import Path
import time
import sys
import functools
import importlib

print = functools.partial(print, flush=True)

# Import solver from optimizer_v3
sys.path.insert(0, str(Path(__file__).parent))
from optimizer_v3 import solve, validate, save_solution

def run_multi_seed(n=26, seeds=None, n_starts=25, timeout_per_seed=480, total_timeout=3600):
    if seeds is None:
        seeds = [42, 123, 7, 314, 271, 1337, 2024, 55, 89, 144]

    best_metric = 0.0
    best_solution = None
    best_seed = None
    t0 = time.time()

    # Try to load existing best
    sol_path = Path(__file__).parent / f"solution_n{n}_best.json"
    if sol_path.exists():
        with open(sol_path) as f:
            data = json.load(f)
        circles = data['circles']
        if validate(circles):
            best_metric = sum(c[2] for c in circles)
            best_solution = circles
            print(f"Loaded existing best: {best_metric:.6f}")

    for seed in seeds:
        elapsed = time.time() - t0
        if elapsed > total_timeout:
            print(f"Total timeout reached")
            break

        remaining = total_timeout - elapsed
        seed_timeout = min(timeout_per_seed, remaining)

        print(f"\n{'='*60}")
        print(f"Seed {seed} (elapsed={elapsed:.0f}s, budget={seed_timeout:.0f}s)")
        print(f"{'='*60}")

        solution, metric, _ = solve(n=n, n_starts=n_starts, seed=seed,
                                     verbose=True, timeout=seed_timeout)

        if solution is not None and validate(solution) and metric > best_metric:
            best_metric = metric
            best_solution = solution
            best_seed = seed
            print(f"\n*** GLOBAL BEST: {best_metric:.6f} (seed={seed}) ***")
            # Save immediately
            save_solution(best_solution, sol_path)
            save_solution(best_solution, Path(__file__).parent / f"solution_n{n}.json")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"FINAL: best_metric={best_metric:.6f}, best_seed={best_seed}, time={elapsed:.0f}s")
    print(f"{'='*60}")

    if best_solution is not None:
        save_solution(best_solution, Path(__file__).parent / f"solution_n{n}.json")
        save_solution(best_solution, sol_path)

    return best_metric

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 26
    total_timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 2400

    seeds = [123, 42, 7, 314, 271, 1337, 2024, 55, 89, 144,
             500, 777, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]

    run_multi_seed(n=n, seeds=seeds, n_starts=25, timeout_per_seed=450,
                   total_timeout=total_timeout)
