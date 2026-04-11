#!/usr/bin/env python3
"""Generate figures for mobius-001 orbit."""

import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from pathlib import Path

WORKTREE = Path("/Users/wujiewang/code/circle-packing/.worktrees/mobius-001")
OUTPUT_DIR = WORKTREE / "orbits/mobius-001"
FIG_DIR = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "monospace",
    "font.monospace": ["DejaVu Sans Mono", "Menlo", "Consolas", "Monaco"],
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlepad": 8.0,
    "axes.labelpad": 4.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def load_solution(path):
    with open(path) as f:
        return np.array(json.load(f)["circles"])


def get_contacts(circles, tol=1e-4):
    n = len(circles)
    contacts = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = circles[i, 0] - circles[j, 0]
            dy = circles[i, 1] - circles[j, 1]
            d = math.sqrt(dx * dx + dy * dy)
            gap = d - circles[i, 2] - circles[j, 2]
            if abs(gap) < tol:
                contacts.append((i, j))
    return contacts


def main():
    sol = load_solution(OUTPUT_DIR / "solution_n26.json")
    n = len(sol)
    contacts = get_contacts(sol)

    # Figure: 2-panel layout
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), constrained_layout=True)

    # Panel 1: Circle packing visualization
    ax = axes[0]
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.set_title("n=26 Circle Packing", fontweight='bold')

    # Unit square
    sq = Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', linewidth=1.5)
    ax.add_patch(sq)

    # Color by radius
    radii = sol[:, 2]
    norm = plt.Normalize(radii.min(), radii.max())
    cmap = plt.cm.get_cmap('GnBu')

    for i in range(n):
        x, y, r = sol[i]
        color = cmap(norm(r))
        circle = Circle((x, y), r, facecolor=color, edgecolor='#333333',
                         linewidth=0.8, alpha=0.85)
        ax.add_patch(circle)
        # Label with index
        ax.text(x, y, str(i), ha='center', va='center', fontsize=7,
                fontweight='bold', color='#222222')

    # Draw contact edges
    for i, j in contacts:
        ax.plot([sol[i, 0], sol[j, 0]], [sol[i, 1], sol[j, 1]],
                'k-', alpha=0.1, linewidth=0.5)

    # Wall contacts
    wall_contacts = 0
    for i in range(n):
        x, y, r = sol[i]
        if abs(x - r) < 1e-4: wall_contacts += 1
        if abs(1 - x - r) < 1e-4: wall_contacts += 1
        if abs(y - r) < 1e-4: wall_contacts += 1
        if abs(1 - y - r) < 1e-4: wall_contacts += 1

    ax.text(0.02, -0.01, f"sum(r) = {np.sum(radii):.10f}",
            transform=ax.transData, fontsize=10, fontweight='bold',
            verticalalignment='top')
    ax.text(0.02, -0.035, f"{len(contacts)} contacts + {wall_contacts} walls = "
            f"{len(contacts) + wall_contacts} active",
            transform=ax.transData, fontsize=9, color='#555555',
            verticalalignment='top')

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(False)

    # Panel 2: Radius distribution + contact degree
    ax2 = axes[1]

    # Compute contact degree for each circle
    degree = np.zeros(n, dtype=int)
    for i, j in contacts:
        degree[i] += 1
        degree[j] += 1
    # Add wall contacts
    for i in range(n):
        x, y, r = sol[i]
        if abs(x - r) < 1e-4: degree[i] += 1
        if abs(1 - x - r) < 1e-4: degree[i] += 1
        if abs(y - r) < 1e-4: degree[i] += 1
        if abs(1 - y - r) < 1e-4: degree[i] += 1

    # Sort by radius
    order = np.argsort(-radii)
    sorted_radii = radii[order]
    sorted_degree = degree[order]
    sorted_idx = order

    colors = [cmap(norm(r)) for r in sorted_radii]
    bars = ax2.bar(range(n), sorted_radii, color=colors, edgecolor='#555555',
                   linewidth=0.5, alpha=0.85)

    # Annotate degree
    for i in range(n):
        ax2.text(i, sorted_radii[i] + 0.002, str(sorted_degree[i]),
                 ha='center', va='bottom', fontsize=7, color='#333333')

    ax2.set_xlabel("Circle (sorted by radius)")
    ax2.set_ylabel("Radius")
    ax2.set_title("Radius Distribution (numbers = contact degree)", fontweight='bold')
    ax2.set_xticks(range(0, n, 5))

    # Add summary stats
    stats_text = (
        f"n = {n}\n"
        f"sum(r) = {np.sum(radii):.10f}\n"
        f"mean(r) = {np.mean(radii):.4f}\n"
        f"std(r) = {np.std(radii):.4f}\n"
        f"min(r) = {np.min(radii):.4f}\n"
        f"max(r) = {np.max(radii):.4f}\n"
        f"mean(deg) = {np.mean(degree):.1f}"
    )
    ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes,
             fontsize=9, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8,
                       edgecolor='#cccccc'))

    fig.savefig(FIG_DIR / "packing_n26.png", dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close()

    print(f"Saved: {FIG_DIR / 'packing_n26.png'}", flush=True)


if __name__ == "__main__":
    main()
