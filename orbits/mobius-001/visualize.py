"""Visualization for mobius-001 results."""

import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

WORKTREE = Path("/Users/wujiewang/code/circle-packing/.worktrees/mobius-001")
OUTPUT_DIR = WORKTREE / "orbits/mobius-001"


def plot_packing(circles, ax, title="Circle Packing"):
    """Draw a circle packing on the given axes."""
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.add_patch(patches.Rectangle((0, 0), 1, 1, fill=False,
                                    edgecolor='black', linewidth=2))

    cmap = plt.cm.GnBu
    radii = [c[2] for c in circles]
    r_min, r_max = min(radii), max(radii)

    for i, (x, y, r) in enumerate(circles):
        norm_r = (r - r_min) / (r_max - r_min + 1e-10)
        color = cmap(0.3 + 0.6 * norm_r)
        circle = plt.Circle((x, y), r, fill=True, facecolor=color,
                            edgecolor='black', linewidth=0.5, alpha=0.8)
        ax.add_patch(circle)
        if r > 0.05:
            ax.text(x, y, f'{i}', ha='center', va='center', fontsize=7)

    ax.set_title(title, fontweight='bold')


def plot_results():
    """Create summary figure."""
    # Load solution
    sol_path = OUTPUT_DIR / "solution_n26.json"
    if not sol_path.exists():
        print("No solution found")
        return

    with open(sol_path) as f:
        data = json.load(f)
    circles = data["circles"]
    metric = sum(c[2] for c in circles)

    # Load metrics if available
    metrics_path = OUTPUT_DIR / "search_metrics.json"
    has_metrics = metrics_path.exists()
    if has_metrics:
        with open(metrics_path) as f:
            metrics_data = json.load(f)

    if has_metrics:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

        # Left: packing
        plot_packing(circles, axes[0],
                    f"n=26 packing  sum(r)={metric:.10f}")

        # Right: metric distribution
        ax = axes[1]
        all_m = sorted(metrics_data.get('all_metrics', []), reverse=True)
        if all_m:
            ax.hist(all_m, bins=30, color='steelblue', alpha=0.7, edgecolor='white')
            ax.axvline(metrics_data.get('base', 0), color='red', linestyle='--',
                      linewidth=2, label=f"Base: {metrics_data.get('base', 0):.6f}")
            ax.axvline(metrics_data.get('best', 0), color='green', linestyle='-',
                      linewidth=2, label=f"Best: {metrics_data.get('best', 0):.6f}")
            ax.set_xlabel("Sum of radii")
            ax.set_ylabel("Count")
            ax.set_title("Search metric distribution", fontweight='bold')
            ax.legend()
    else:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7), constrained_layout=True)
        plot_packing(circles, ax, f"n=26 packing  sum(r)={metric:.10f}")

    fig.savefig(OUTPUT_DIR / "figures" / "packing_summary.png",
               dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved figure to {OUTPUT_DIR / 'figures' / 'packing_summary.png'}")
    plt.close()


if __name__ == "__main__":
    plot_results()
