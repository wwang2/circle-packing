"""Visualize the best circle packing solution."""

import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path


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


def main():
    solution_path = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / "solution_n26.json")
    output_path = sys.argv[2] if len(sys.argv) > 2 else str(Path(__file__).parent / "figures/packing_n26.png")

    with open(solution_path) as f:
        data = json.load(f)
    circles = data["circles"]
    n = len(circles)
    total = sum(c[2] for c in circles)

    # Sort by radius for coloring
    radii = [c[2] for c in circles]
    r_min, r_max = min(radii), max(radii)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True,
                             gridspec_kw={'width_ratios': [1, 0.6]})

    # Left: circle packing visualization
    ax = axes[0]
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.set_title(f'Circle Packing n={n}, sum(r)={total:.10f}', fontweight='bold')

    # Draw unit square
    square = plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(square)

    # Color map
    cmap = plt.cm.GnBu
    norm = plt.Normalize(r_min * 0.9, r_max * 1.1)

    for i, (x, y, r) in enumerate(circles):
        color = cmap(norm(r))
        circle = Circle((x, y), r, facecolor=color, edgecolor='black',
                        linewidth=0.5, alpha=0.7)
        ax.add_patch(circle)
        # Label with index
        if r > 0.06:
            ax.text(x, y, f'{i}', ha='center', va='center', fontsize=7, fontweight='bold')

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Right: radius distribution
    ax2 = axes[1]
    sorted_radii = sorted(radii, reverse=True)
    colors = [cmap(norm(r)) for r in sorted_radii]
    ax2.barh(range(n), sorted_radii, color=colors, edgecolor='black', linewidth=0.3)
    ax2.set_xlabel('Radius')
    ax2.set_ylabel('Circle (sorted)')
    ax2.set_title('Radius Distribution', fontweight='bold')
    ax2.invert_yaxis()

    # Add stats text
    stats = f'n={n}\nsum(r)={total:.6f}\nmin(r)={r_min:.6f}\nmax(r)={r_max:.6f}\nstd(r)={np.std(radii):.6f}'
    ax2.text(0.95, 0.95, stats, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
