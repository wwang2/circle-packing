"""Generate visualization figure for diffevo-001 results."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import json
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


def load_solution(path):
    with open(path) as f:
        data = json.load(f)
    return data['circles']


def plot_packing(ax, circles, title):
    """Plot a circle packing on the given axes."""
    # Draw unit square
    square = patches.Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='black',
                                facecolor='#f8f8f8', zorder=0)
    ax.add_patch(square)

    # Color map
    n = len(circles)
    cmap = plt.cm.GnBu
    radii = [r for _, _, r in circles]
    r_min, r_max = min(radii), max(radii)

    for i, (cx, cy, r) in enumerate(circles):
        # Color based on radius
        if r_max > r_min:
            norm_r = (r - r_min) / (r_max - r_min)
        else:
            norm_r = 0.5
        color = cmap(0.3 + 0.6 * norm_r)

        circle = patches.Circle((cx, cy), r, linewidth=0.5, edgecolor='#333333',
                                facecolor=color, alpha=0.85, zorder=1)
        ax.add_patch(circle)

    metric = sum(r for _, _, r in circles)
    ax.set_title(f"{title}\nsum(r) = {metric:.6f}", fontweight='bold')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.grid(False)


def main():
    outdir = Path(__file__).parent
    fig_dir = outdir / 'figures'
    fig_dir.mkdir(exist_ok=True)

    # Load solutions
    solutions = {}
    for n in [10, 28, 30, 32]:
        path = outdir / f'solution_n{n}.json'
        if path.exists():
            solutions[n] = load_solution(path)

    if not solutions:
        print("No solutions found")
        return

    # Create grid figure
    n_plots = len(solutions)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5.5),
                             constrained_layout=True)
    if n_plots == 1:
        axes = [axes]

    sota = {10: 1.591, 28: 2.737, 30: 2.842, 32: 2.939}

    for ax, (n, circles) in zip(axes, sorted(solutions.items())):
        metric = sum(r for _, _, r in circles)
        sota_val = sota.get(n, 0)
        pct = (metric / sota_val * 100) if sota_val else 0
        title = f"n={n} ({pct:.1f}% of SOTA)"
        plot_packing(ax, circles, title)

    fig.suptitle("diffevo-001: Circle Packing Results\nL-BFGS-B penalty + basin-hopping refinement",
                 fontweight='bold', fontsize=14, y=1.02)

    outpath = fig_dir / 'results.png'
    fig.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved to {outpath}")
    plt.close()


if __name__ == '__main__':
    main()
