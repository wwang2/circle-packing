#!/usr/bin/env python3
"""Generate visualization figures for mobius-001 orbit."""

import json
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

def draw_packing(ax, circles, title, metric_str):
    """Draw a circle packing on the given axes."""
    n = len(circles)

    # Draw unit square
    square = Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='#333333',
                       facecolor='#fafafa', zorder=0)
    ax.add_patch(square)

    # Color by radius
    radii = circles[:, 2]
    norm = plt.Normalize(radii.min(), radii.max())
    cmap = plt.cm.GnBu

    for i in range(n):
        x, y, r = circles[i]
        color = cmap(norm(r))
        circle = Circle((x, y), r, facecolor=color, edgecolor='#333333',
                        linewidth=0.8, alpha=0.85, zorder=1)
        ax.add_patch(circle)

        # Add radius text for larger circles
        if r > 0.05:
            ax.text(x, y, f'{r:.3f}', ha='center', va='center',
                   fontsize=7, fontweight='bold', color='#222222', zorder=2)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.set_title(f'{title}\n{metric_str}', fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')


def main():
    # Load solutions
    solutions = {}

    n26_path = OUTPUT_DIR / "solution_n26.json"
    if n26_path.exists():
        solutions['n26'] = load_solution(n26_path)

    n32_path = OUTPUT_DIR / "solution_n32.json"
    if n32_path.exists():
        solutions['n32'] = load_solution(n32_path)

    n_plots = len(solutions)
    if n_plots == 0:
        print("No solutions found!")
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 7),
                             constrained_layout=True)
    if n_plots == 1:
        axes = [axes]

    idx = 0
    if 'n26' in solutions:
        c = solutions['n26']
        sr = sum(c[:, 2])
        draw_packing(axes[idx], c, f'n=26 Circle Packing',
                    f'sum(r) = {sr:.10f}')
        idx += 1

    if 'n32' in solutions:
        c = solutions['n32']
        sr = sum(c[:, 2])
        draw_packing(axes[idx], c, f'n=32 Circle Packing',
                    f'sum(r) = {sr:.10f}')
        idx += 1

    fig.suptitle('mobius-001: Circle Packing Optimization',
                fontweight='bold', fontsize=14, y=1.02)

    fig.savefig(FIG_DIR / 'packings.png', dpi=200, bbox_inches='tight',
               facecolor='white')
    plt.close()
    print(f"Saved packings.png")

    # Also create a convergence/comparison figure
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)

    # Data from experiments
    attempts = {
        'n=26': [
            ('NLP baseline', 2.6246),
            ('Basin-hop', 2.6308),
            ('Topo search', 2.6360),
            ('Mobius deform', 2.6360),
            ('KKT refine', 2.6360),
            ('Tolerance squeeze', 2.6360),
            ('Graph topo (3000)', 2.6360),
            ('Edge flip', 2.6360),
        ],
        'n=32': [
            ('Ring init', 2.9132),
            ('Grid init', 2.9194),
            ('Random init', 2.9213),
            ('Basin-hop 1', 2.9334),
            ('Basin-hop 2', 2.9396),
        ],
    }

    colors = {'n=26': '#2166ac', 'n=32': '#b2182b'}

    for label, data in attempts.items():
        names = [d[0] for d in data]
        values = [d[1] for d in data]
        ax2.plot(range(len(values)), values, 'o-', color=colors[label],
                label=label, markersize=6, linewidth=2)
        # Annotate last point
        ax2.annotate(f'{values[-1]:.4f}',
                    (len(values)-1, values[-1]),
                    textcoords="offset points", xytext=(10, 5),
                    fontsize=10, fontweight='bold', color=colors[label])

    ax2.set_ylabel('Sum of Radii', fontweight='bold')
    ax2.set_xlabel('Optimization Stage', fontweight='bold')
    ax2.set_title('Convergence History', fontweight='bold')
    ax2.legend(loc='lower right')

    fig2.savefig(FIG_DIR / 'convergence.png', dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print(f"Saved convergence.png")


if __name__ == "__main__":
    main()
