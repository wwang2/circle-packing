"""Visualize circle packing solution and convergence."""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

def plot_packing(solution_path, output_path):
    """Plot the circle packing."""
    with open(solution_path) as f:
        data = json.load(f)
    circles = data["circles"] if "circles" in data else data

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Unit square
    ax.add_patch(patches.Rectangle((0, 0), 1, 1, fill=False,
                                     edgecolor='black', linewidth=2))

    # Circles
    colors = plt.cm.tab20(np.linspace(0, 1, len(circles)))
    total_r = 0
    for i, (x, y, r) in enumerate(circles):
        circle = plt.Circle((x, y), r, fill=True, facecolor=colors[i],
                            edgecolor='black', linewidth=0.5, alpha=0.7)
        ax.add_patch(circle)
        ax.text(x, y, f'{i}', ha='center', va='center', fontsize=6)
        total_r += r

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.set_title(f'Circle Packing n={len(circles)}, sum(r)={total_r:.10f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved packing plot to {output_path}")


def plot_convergence(results_path, output_path):
    """Plot convergence from results summary."""
    if not Path(results_path).exists():
        print("No results summary found, skipping convergence plot.")
        return

    with open(results_path) as f:
        results = json.load(f)

    # Sort by metric descending
    metrics = [r[1] for r in results]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.bar(range(len(metrics)), metrics, color='steelblue', alpha=0.7)
    ax.axhline(y=2.6359830849, color='red', linestyle='--', label='Current best (2.6360)')
    ax.set_xlabel('Trial (sorted by metric)')
    ax.set_ylabel('Sum of radii')
    ax.set_title('JAX Soft-Body Optimizer: All Trial Results')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved convergence plot to {output_path}")


if __name__ == "__main__":
    workdir = Path(__file__).parent
    figdir = workdir / "figures"
    figdir.mkdir(exist_ok=True)

    sol_path = workdir / "solution_n26.json"
    if sol_path.exists():
        plot_packing(str(sol_path), str(figdir / "packing_n26.png"))

    results_path = workdir / "results_summary.json"
    plot_convergence(str(results_path), str(figdir / "convergence.png"))
