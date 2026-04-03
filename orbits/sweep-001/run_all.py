"""Run solver for all target n values sequentially."""
import sys
import os
import numpy as np
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solver import solve_n, save_solution, validate_solution, vec_to_pack

np.random.seed(12345)

targets = {
    29: 2.790,
    31: 2.889,
    24: 2.530,
    25: 2.587,
    27: 2.685,
}

out_dir = os.path.dirname(os.path.abspath(__file__))
results = {}

for n, sota in targets.items():
    print(f"\n{'#'*60}")
    print(f"# TARGET n={n}, SOTA={sota}")
    print(f"{'#'*60}")
    t0 = time.time()

    vec, metric = solve_n(n, num_starts=100, bh_hops=300)

    if vec is not None:
        filepath = os.path.join(out_dir, f"solution_n{n}.json")
        save_solution(vec, n, filepath)
        results[n] = metric
        elapsed = time.time() - t0
        print(f"\n>>> n={n}: metric={metric:.10f}, SOTA={sota:.3f}, ratio={metric/sota:.4f}, time={elapsed:.0f}s")
    else:
        print(f"\n>>> n={n}: FAILED")
        results[n] = 0

print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
for n, sota in targets.items():
    m = results.get(n, 0)
    print(f"  n={n}: metric={m:.10f}  SOTA={sota:.3f}  ratio={m/sota:.4f}")
