"""Visualize n=32 circle packing solution and analyze contacts."""

import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

WORKDIR = Path(__file__).parent
TOL = 1e-6  # Contact tolerance


def load_solution(path):
    with open(path) as f:
        data = json.load(f)
    return np.array(data.get("circles", data))


def analyze_contacts(circles, tol=TOL):
    """Analyze contact graph of the packing."""
    n = len(circles)
    contacts_cc = []  # circle-circle
    contacts_cw = []  # circle-wall

    for i in range(n):
        xi, yi, ri = circles[i]
        # Wall contacts
        if abs(xi - ri) < tol:
            contacts_cw.append((i, 'left'))
        if abs(1 - xi - ri) < tol:
            contacts_cw.append((i, 'right'))
        if abs(yi - ri) < tol:
            contacts_cw.append((i, 'bottom'))
        if abs(1 - yi - ri) < tol:
            contacts_cw.append((i, 'top'))

    for i in range(n):
        xi, yi, ri = circles[i]
        for j in range(i+1, n):
            xj, yj, rj = circles[j]
            dist = math.sqrt((xi-xj)**2 + (yi-yj)**2)
            if abs(dist - ri - rj) < tol:
                contacts_cc.append((i, j))

    dof = 3 * n - len(contacts_cc) - len(contacts_cw)

    return contacts_cc, contacts_cw, dof


def main():
    circles = load_solution(WORKDIR / "solution_n32_initial.json")
    n = len(circles)
    metric = sum(c[2] for c in circles)

    contacts_cc, contacts_cw, dof = analyze_contacts(circles)

    print(f"N = {n}")
    print(f"Sum of radii = {metric:.10f}")
    print(f"Circle-circle contacts: {len(contacts_cc)}")
    print(f"Circle-wall contacts: {len(contacts_cw)}")
    print(f"Total contacts: {len(contacts_cc) + len(contacts_cw)}")
    print(f"Variables: {3*n}")
    print(f"DOF = {dof}")

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', linewidth=2))

    # Draw circles
    for i, (x, y, r) in enumerate(circles):
        circle = plt.Circle((x, y), r, fill=True, facecolor='lightblue',
                           edgecolor='navy', linewidth=0.8, alpha=0.7)
        ax.add_patch(circle)
        ax.text(x, y, str(i), ha='center', va='center', fontsize=5, fontweight='bold')

    # Draw contact lines
    for i, j in contacts_cc:
        xi, yi, _ = circles[i]
        xj, yj, _ = circles[j]
        ax.plot([xi, xj], [yi, yj], 'r-', linewidth=0.3, alpha=0.5)

    ax.set_title(f'N=32 Circle Packing\nSum of radii = {metric:.10f}\n'
                f'Contacts: {len(contacts_cc)} CC + {len(contacts_cw)} CW = '
                f'{len(contacts_cc)+len(contacts_cw)}, DOF = {dof}',
                fontsize=10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.tight_layout()
    plt.savefig(WORKDIR / 'figures' / 'n32_packing.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved figure to figures/n32_packing.png")

    # Radius distribution
    radii = sorted(circles[:, 2], reverse=True)
    print(f"\nRadius distribution:")
    print(f"  Max: {radii[0]:.6f}")
    print(f"  Min: {radii[-1]:.6f}")
    print(f"  Mean: {np.mean(radii):.6f}")
    print(f"  Std: {np.std(radii):.6f}")


if __name__ == "__main__":
    main()
