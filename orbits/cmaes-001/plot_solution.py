"""Plot the circle packing solution."""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
    return np.array(data.get("circles", data))

def main():
    sol_path = os.path.join(WORKDIR, "solution_n26.json")
    circles = load_solution(sol_path)
    n = len(circles)
    sum_r = np.sum(circles[:, 2])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True,
                              gridspec_kw={'width_ratios': [1, 0.6]})

    # Left: circle packing visualization
    ax = axes[0]
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.set_title(f'Circle Packing n={n}, sum(r)={sum_r:.10f}', fontweight='bold')

    # Unit square
    square = patches.Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='black',
                                facecolor='none', zorder=3)
    ax.add_patch(square)

    # Color by radius
    radii = circles[:, 2]
    norm = plt.Normalize(radii.min(), radii.max())
    cmap = plt.cm.GnBu

    for i, (x, y, r) in enumerate(circles):
        color = cmap(norm(r))
        circle = plt.Circle((x, y), r, facecolor=color, edgecolor='black',
                           linewidth=0.8, alpha=0.85, zorder=2)
        ax.add_patch(circle)
        # Label with index
        fontsize = max(6, min(10, int(r * 80)))
        ax.text(x, y, str(i), ha='center', va='center', fontsize=fontsize,
               fontweight='bold', zorder=4)

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Right: radius distribution
    ax2 = axes[1]
    sorted_r = np.sort(radii)[::-1]
    colors = [cmap(norm(r)) for r in sorted_r]
    ax2.barh(range(n), sorted_r, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Radius')
    ax2.set_ylabel('Circle (sorted)')
    ax2.set_title('Radius Distribution', fontweight='bold')
    ax2.invert_yaxis()
    ax2.axvline(x=np.mean(radii), color='red', linestyle='--', alpha=0.7,
                label=f'mean={np.mean(radii):.4f}')
    ax2.legend()

    # Add annotation
    fig.suptitle(f'CMA-ES Orbit: n=26, metric={sum_r:.10f}', fontweight='bold', fontsize=14)

    fig_path = os.path.join(WORKDIR, "figures", "solution_n26.png")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {fig_path}")

if __name__ == "__main__":
    main()
