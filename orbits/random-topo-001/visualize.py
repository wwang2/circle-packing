"""Visualize search results and best solution for random-topo-001."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

def main():
    # Load best solution
    sol_path = os.path.join(WORKDIR, '..', 'topo-001', 'solution_n26.json')
    circles = load_solution(sol_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    # Left panel: Best solution visualization
    ax = axes[0]
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.set_title('Best n=26 Solution (metric=2.6360)', fontweight='bold')

    # Unit square
    rect = plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(rect)

    # Color by radius
    radii = circles[:, 2]
    norm = plt.Normalize(vmin=radii.min(), vmax=radii.max())
    cmap = plt.cm.get_cmap('GnBu')

    for i, (cx, cy, r) in enumerate(circles):
        color = cmap(norm(r))
        circle = plt.Circle((cx, cy), r, fill=True, facecolor=color,
                           edgecolor='black', linewidth=0.8, alpha=0.85)
        ax.add_patch(circle)
        if r > 0.09:
            ax.text(cx, cy, f'{i}', ha='center', va='center', fontsize=7,
                   fontweight='bold')

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Right panel: Search results summary
    ax2 = axes[1]

    # Data from all 3 search runs
    search_labels = [
        'v1: Random inits\n(1000+ configs)',
        'v1: Perturbations\n(600 perturbations)',
        'v3: Apollonius\n(200 configs)',
        'v3: Stripes\n(80 configs)',
        'v3: Spirals\n(80 configs)',
        'v3: Cantrell-style\n(200 configs)',
        'v3: Max-hole\n(200 configs)',
    ]

    # Best metrics from each strategy (from log data)
    best_metrics = [
        2.6359773948,  # v1 random
        2.6359773948,  # v1 perturbations
        2.6279045349,  # v3 apollonius
        2.6310935895,  # v3 stripes
        2.60,          # v3 spirals (lower)
        2.6359773948,  # v3 cantrell
        2.6293005838,  # v3 maxhole
    ]

    colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    y_pos = np.arange(len(search_labels))
    bars = ax2.barh(y_pos, best_metrics, color=colors, alpha=0.8, height=0.6)

    # Reference line for best known
    ax2.axvline(x=2.6359830849, color='red', linestyle='--', linewidth=2,
                label=f'Best known: 2.6360')

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(search_labels, fontsize=9)
    ax2.set_xlabel('Best metric (sum of radii)')
    ax2.set_title('Search Strategy Comparison', fontweight='bold')
    ax2.set_xlim(2.58, 2.645)
    ax2.legend(loc='lower right')

    # Add metric labels on bars
    for bar, metric in zip(bars, best_metrics):
        ax2.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height()/2,
                f'{metric:.4f}', va='center', fontsize=9)

    fig.suptitle('random-topo-001: Massive Random Topology Search (n=26)',
                fontweight='bold', fontsize=14)

    fig_path = os.path.join(WORKDIR, 'figures', 'search_results.png')
    fig.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {fig_path}")
    plt.close()

if __name__ == '__main__':
    main()
