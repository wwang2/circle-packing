"""Plot circle packing solution."""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
import sys

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

def plot_packing(solution_path, output_path):
    with open(solution_path) as f:
        data = json.load(f)
    circles = data['circles']
    n = len(circles)
    total_r = sum(c[2] for c in circles)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Draw unit square
    square = patches.Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='black',
                                facecolor='#f8f8f8', zorder=0)
    ax.add_patch(square)

    # Color by radius
    radii = [c[2] for c in circles]
    r_min, r_max = min(radii), max(radii)
    cmap = plt.cm.viridis

    for i, (x, y, r) in enumerate(circles):
        norm_r = (r - r_min) / (r_max - r_min + 1e-10)
        color = cmap(norm_r)
        circle = patches.Circle((x, y), r, linewidth=0.8, edgecolor='black',
                                facecolor=color, alpha=0.7, zorder=1)
        ax.add_patch(circle)
        # Label with index
        if r > 0.04:
            ax.text(x, y, str(i+1), ha='center', va='center', fontsize=7,
                   fontweight='bold', color='white', zorder=2)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.set_title(f'n={n} circles, sum(r) = {total_r:.6f}', fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(r_min, r_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, label='radius')

    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    sol = sys.argv[1] if len(sys.argv) > 1 else "orbits/basin-001/solution_n26.json"
    out = sys.argv[2] if len(sys.argv) > 2 else "orbits/basin-001/figures/packing_n26.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plot_packing(sol, out)
