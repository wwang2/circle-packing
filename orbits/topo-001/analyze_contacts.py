"""Quick analysis of parent solution's contact graph."""
import json
import numpy as np
import os

WORKDIR = os.path.dirname(os.path.abspath(__file__))

def load_solution(path):
    with open(path) as f:
        data = json.load(f)
    circles = np.array(data["circles"])
    return circles[:, 0], circles[:, 1], circles[:, 2]

parent_path = os.path.join(WORKDIR, '..', 'nlp-001', 'solution_n26.json')
x, y, r = load_solution(parent_path)
n = len(x)

print(f"n={n}, sum_r={sum(r):.10f}")
print(f"\nCircle sizes (sorted):")
order = np.argsort(r)[::-1]
for idx in order:
    print(f"  [{idx:2d}] r={r[idx]:.6f} pos=({x[idx]:.4f}, {y[idx]:.4f})")

# Contact graph with various tolerances
for tol in [1e-6, 1e-5, 1e-4, 1e-3]:
    cc = []
    wc = []
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            gap = dist - (r[i] + r[j])
            if gap < tol:
                cc.append((i, j, gap))
        gaps = {'L': x[i]-r[i], 'R': 1-x[i]-r[i], 'B': y[i]-r[i], 'T': 1-y[i]-r[i]}
        for wall, gap in gaps.items():
            if gap < tol:
                wc.append((i, wall, gap))
    print(f"\ntol={tol}: {len(cc)} circle-circle, {len(wc)} wall contacts")

    if tol == 1e-4:
        print("  Circle-circle contacts:")
        for i, j, gap in sorted(cc, key=lambda t: t[2]):
            print(f"    {i:2d}-{j:2d}: gap={gap:.2e}")
        print("  Wall contacts:")
        for i, wall, gap in sorted(wc, key=lambda t: t[2]):
            print(f"    {i:2d}-{wall}: gap={gap:.2e}")

        # Degree
        degree = np.zeros(n, dtype=int)
        for i, j, _ in cc:
            degree[i] += 1
            degree[j] += 1
        for i, _, _ in wc:
            degree[i] += 1
        print(f"  Degrees: {list(degree)}")
        print(f"  Total contacts: {len(cc) + len(wc)}")
        print(f"  Circles with degree 0: {list(np.where(degree==0)[0])}")
