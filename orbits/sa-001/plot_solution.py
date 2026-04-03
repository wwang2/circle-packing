"""Generate visualization of the circle packing solution."""

import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
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
    script_dir = Path(__file__).parent
    solution_path = script_dir / "solution_n26.json"

    with open(solution_path) as f:
        data = json.load(f)
    circles = data["circles"]
    n = len(circles)
    total = sum(c[2] for c in circles)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True,
                             gridspec_kw={'width_ratios': [1, 0.8]})

    # Left: Circle packing visualization
    ax = axes[0]
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.set_title(f'Circle Packing n={n}, sum(r) = {total:.6f}', fontweight='bold')

    # Draw unit square
    square = Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(square)

    # Color circles by radius
    radii = [c[2] for c in circles]
    r_min, r_max = min(radii), max(radii)
    cmap = plt.cm.GnBu

    for i, (x, y, r) in enumerate(circles):
        norm_r = (r - r_min) / (r_max - r_min) if r_max > r_min else 0.5
        color = cmap(0.3 + 0.6 * norm_r)
        circle = Circle((x, y), r, fill=True, facecolor=color,
                        edgecolor='#2c3e50', linewidth=0.8, alpha=0.85)
        ax.add_patch(circle)
        # Label with index
        if r > 0.05:
            ax.text(x, y, str(i), ha='center', va='center',
                   fontsize=7, fontweight='bold', color='#2c3e50')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, alpha=0.15)

    # Right: Radius distribution
    ax2 = axes[1]
    sorted_radii = sorted(radii, reverse=True)
    colors = [cmap(0.3 + 0.6 * (r - r_min) / (r_max - r_min)) for r in sorted_radii]

    bars = ax2.barh(range(n), sorted_radii, color=colors, edgecolor='#2c3e50',
                    linewidth=0.5, alpha=0.85)
    ax2.set_xlabel('Radius')
    ax2.set_ylabel('Circle (sorted by radius)')
    ax2.set_title('Radius Distribution', fontweight='bold')
    ax2.invert_yaxis()

    # Add stats
    stats_text = (
        f'n = {n}\n'
        f'sum(r) = {total:.6f}\n'
        f'mean(r) = {total/n:.6f}\n'
        f'max(r) = {r_max:.6f}\n'
        f'min(r) = {r_min:.6f}\n'
        f'SOTA ~ 2.6360'
    )
    ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                     edgecolor='#cccccc', alpha=0.9))

    fig_path = script_dir / "figures" / "solution_n26.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved to {fig_path}")
    plt.close()


if __name__ == "__main__":
    main()
