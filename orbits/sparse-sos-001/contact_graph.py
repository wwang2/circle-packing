"""
Build the active contact graph from the mobius-001 optimum.
Identifies disk-disk contacts and disk-wall contacts.

Output:
  contacts.json: {
    "n": 26,
    "x*": [..], "y*": [..], "r*": [..],
    "disk_disk_edges": [(i,j), ...],
    "disk_wall_contacts": [(i, side), ...],  # side in {"L","R","B","T"}
    "num_contacts": 78,
  }
"""
import json
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent
PARENT_SOLN = HERE.parent / "mobius-001" / "solution_n26.json"
OUT = HERE / "contacts.json"

TOL = 1e-7  # active-constraint tolerance (mobius-001 reports ~1e-10 feasibility)


def main():
    data = json.loads(PARENT_SOLN.read_text())
    circles = np.asarray(data["circles"], dtype=float)  # (n, 3) -> x,y,r
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    n = len(r)

    # Disk-disk contacts: (xi-xj)^2 + (yi-yj)^2 == (ri+rj)^2
    disk_disk = []
    for i in range(n):
        for j in range(i + 1, n):
            d2 = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2
            rsum = r[i] + r[j]
            slack = np.sqrt(d2) - rsum
            if slack < TOL:
                disk_disk.append((i, j, float(slack)))

    # Disk-wall contacts
    wall = []
    for i in range(n):
        # Left wall: x == r
        if abs(x[i] - r[i]) < TOL:
            wall.append((i, "L", float(x[i] - r[i])))
        # Right wall: x + r == 1
        if abs(x[i] + r[i] - 1.0) < TOL:
            wall.append((i, "R", float(1.0 - x[i] - r[i])))
        # Bottom wall: y == r
        if abs(y[i] - r[i]) < TOL:
            wall.append((i, "B", float(y[i] - r[i])))
        # Top wall: y + r == 1
        if abs(y[i] + r[i] - 1.0) < TOL:
            wall.append((i, "T", float(1.0 - y[i] - r[i])))

    # Summaries
    total = len(disk_disk) + len(wall)
    sum_r = float(np.sum(r))
    print(f"n={n}  sum_r={sum_r:.10f}")
    print(f"disk-disk contacts: {len(disk_disk)}")
    print(f"disk-wall contacts: {len(wall)}")
    print(f"total active constraints: {total}")

    # Sort & print a few
    disk_disk.sort(key=lambda t: t[2])
    wall.sort(key=lambda t: abs(t[2]))
    print("\nDisk-disk (tightest 5):", disk_disk[:5])
    print("Disk-wall (tightest 5):", wall[:5])
    print("Disk-disk (loosest 3):", disk_disk[-3:])

    out = {
        "n": n,
        "x": x.tolist(),
        "y": y.tolist(),
        "r": r.tolist(),
        "sum_r": sum_r,
        "disk_disk_edges": [[i, j] for (i, j, s) in disk_disk],
        "disk_disk_slack": [s for (i, j, s) in disk_disk],
        "disk_wall_contacts": [[i, side] for (i, side, s) in wall],
        "disk_wall_slack": [s for (i, side, s) in wall],
        "num_disk_disk": len(disk_disk),
        "num_disk_wall": len(wall),
        "num_contacts": total,
        "tol": TOL,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
