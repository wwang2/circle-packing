"""Generate figure showing the best packing and search results."""

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
    return data["circles"]


def draw_packing(ax, circles, title=""):
    """Draw circle packing in unit square."""
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')

    # Draw unit square
    square = plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(square)

    # Color circles by radius
    radii = [c[2] for c in circles]
    r_min, r_max = min(radii), max(radii)

    cmap = plt.cm.GnBu
    for i, (cx, cy, cr) in enumerate(circles):
        norm_r = (cr - r_min) / (r_max - r_min) if r_max > r_min else 0.5
        color = cmap(0.3 + 0.6 * norm_r)
        circle = plt.Circle((cx, cy), cr, fill=True, facecolor=color,
                            edgecolor='#2c3e50', linewidth=0.8, alpha=0.8)
        ax.add_patch(circle)
        if cr > 0.06:
            ax.text(cx, cy, f'{i}', ha='center', va='center', fontsize=7,
                   fontweight='bold', color='#2c3e50')

    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')


def main():
    sol_path = os.path.join(WORKDIR, 'solution_n26.json')
    circles = load_solution(sol_path)
    metric = sum(c[2] for c in circles)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), constrained_layout=True,
                              gridspec_kw={'width_ratios': [1, 1]})

    # Left: packing visualization
    draw_packing(axes[0], circles,
                 title=f'n=26 Packing (metric={metric:.10f})')

    # Right: search summary
    ax = axes[1]

    strategies = [
        'v4: Fast SLSQP\n(2000 starts)',
        'v5: Diff Evolution\n(100 pop, 500 gen)',
        'v7: Basin Hopping\n(300 iter)',
        'v6: Constructive\n(biscuit/sym)',
        'topo-001: Multi-start\n(500+ inits)',
    ]
    best_metrics = [2.6359830849, 2.6359830849, 2.6359830849, 2.6099, 2.6359830849]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']

    y_pos = np.arange(len(strategies))
    bars = ax.barh(y_pos, best_metrics, color=colors, alpha=0.7, edgecolor='#2c3e50', linewidth=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(strategies, fontsize=10)
    ax.set_xlabel('Best Metric (sum of radii)')
    ax.set_title('Search Strategy Comparison', fontweight='bold')

    # Reference line
    ax.axvline(x=2.6359830849, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(2.636, len(strategies)-0.5, 'Known best\n2.63598', fontsize=9,
            color='red', alpha=0.7, ha='center')

    ax.set_xlim(2.55, 2.65)

    # Add value labels
    for bar, val in zip(bars, best_metrics):
        ax.text(max(val, 2.56) + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9, fontweight='bold')

    fig_path = os.path.join(WORKDIR, 'figures', 'search_results.png')
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to {fig_path}")


if __name__ == '__main__':
    main()
