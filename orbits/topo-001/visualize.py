"""Visualize the circle packing solution with contact graph overlay."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as MplCircle
import os

WORKDIR = os.path.dirname(os.path.abspath(__file__))

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
        data = json.load(f)
    return np.array(data["circles"])

def compute_contacts(circles, tol=1e-4):
    n = len(circles)
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
    cc, wc = [], []
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
    return cc, wc

def main():
    sol_path = os.path.join(WORKDIR, 'solution_n26.json')
    if not os.path.exists(sol_path):
        sol_path = os.path.join(WORKDIR, '..', 'nlp-001', 'solution_n26.json')

    circles = load_solution(sol_path)
    n = len(circles)
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
    metric = np.sum(r)

    cc, wc = compute_contacts(circles)

    # Compute degree
    degree = np.zeros(n, dtype=int)
    for i, j, _ in cc:
        degree[i] += 1
        degree[j] += 1
    for i, _, _ in wc:
        degree[i] += 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    # Left: Circle packing with contact graph
    ax = axes[0]
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.set_title(f'n=26 Packing (sum_r={metric:.10f})', fontweight='bold')

    # Unit square
    sq = plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(sq)

    # Color by radius
    r_norm = (r - r.min()) / (r.max() - r.min() + 1e-10)
    cmap = plt.cm.GnBu

    for i in range(n):
        color = cmap(0.3 + 0.6 * r_norm[i])
        circle = MplCircle((x[i], y[i]), r[i], fill=True, facecolor=color,
                          edgecolor='#2c3e50', linewidth=0.8, alpha=0.85)
        ax.add_patch(circle)
        ax.text(x[i], y[i], str(i), ha='center', va='center',
               fontsize=6, fontweight='bold', color='#2c3e50')

    # Draw contact edges
    for i, j, _ in cc:
        ax.plot([x[i], x[j]], [y[i], y[j]], 'r-', linewidth=0.3, alpha=0.4)

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Right: Degree and radius distribution
    ax2 = axes[1]

    # Radius histogram
    order = np.argsort(r)[::-1]
    colors = [cmap(0.3 + 0.6 * r_norm[i]) for i in order]
    bars = ax2.barh(range(n), r[order], color=colors, edgecolor='#2c3e50', linewidth=0.5)

    # Annotate with degree
    for idx, i in enumerate(order):
        ax2.text(r[i] + 0.002, idx, f'd={degree[i]}', va='center',
                fontsize=7, color='#666')

    ax2.set_yticks(range(n))
    ax2.set_yticklabels([str(i) for i in order], fontsize=7)
    ax2.set_xlabel('Radius')
    ax2.set_ylabel('Circle index')
    ax2.set_title(f'Radii & Degree ({len(cc)} contacts, {len(wc)} wall)',
                  fontweight='bold')
    ax2.invert_yaxis()

    fig_path = os.path.join(WORKDIR, 'figures', 'packing_n26.png')
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {fig_path}")

if __name__ == '__main__':
    main()
