"""Generate figure comparing upper bounds vs known best solutions."""

import numpy as np
import matplotlib.pyplot as plt
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
    "legend.fontsize": 10,
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


known_best = {
    1: 0.5000, 2: 0.5858, 3: 0.7645, 4: 1.0000, 5: 1.0854,
    10: 1.5911, 15: 2.0365, 20: 2.3010, 26: 2.6360, 30: 2.8425, 32: 2.9390,
}

n_values = sorted(known_best.keys())
known = [known_best[n] for n in n_values]

# Compute bounds
n_fine = np.arange(1, 33)
area_bounds = np.sqrt(n_fine / np.pi)
ft_bounds = np.sqrt(n_fine / (2 * np.sqrt(3)))

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

# Panel 1: Absolute values
ax = axes[0]
ax.fill_between(n_fine, ft_bounds, area_bounds, alpha=0.15, color='tab:blue',
                label='Gap: area vs Fejes Toth')
ax.plot(n_fine, area_bounds, 'b--', lw=1.5, label='Area bound: sqrt(n/pi)')
ax.plot(n_fine, ft_bounds, 'b-', lw=2, label='Fejes Toth: sqrt(n/(2*sqrt(3)))')
ax.scatter(n_values, known, c='tab:red', s=60, zorder=5, label='Best known solution')

# Highlight n=26
ax.axvline(26, color='gray', ls=':', alpha=0.5)
ax.annotate(f'n=26\nUB={ft_bounds[25]:.3f}\nBest={known_best[26]:.3f}',
           xy=(26, ft_bounds[25]), xytext=(28, 2.2),
           fontsize=9,
           arrowprops=dict(arrowstyle='->', color='gray'),
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

ax.set_xlabel('Number of circles (n)')
ax.set_ylabel('Sum of radii')
ax.set_title('Upper Bounds vs Best Known Solutions', fontweight='bold')
ax.legend(loc='upper left')
ax.set_xlim(0, 33)
ax.set_ylim(0, 3.5)

# Panel 2: Relative gap
ax = axes[1]
area_gaps = [(np.sqrt(n/np.pi) - known_best[n]) / known_best[n] * 100 for n in n_values]
ft_gaps = [(np.sqrt(n/(2*np.sqrt(3))) - known_best[n]) / known_best[n] * 100 for n in n_values]

ax.bar([x - 0.3 for x in range(len(n_values))], area_gaps, width=0.4,
       color='tab:blue', alpha=0.4, label='Area bound gap')
ax.bar([x + 0.1 for x in range(len(n_values))], ft_gaps, width=0.4,
       color='tab:blue', alpha=0.8, label='Fejes Toth gap')

ax.set_xticks(range(len(n_values)))
ax.set_xticklabels([str(n) for n in n_values], fontsize=9)
ax.set_xlabel('Number of circles (n)')
ax.set_ylabel('Gap from best known (%)')
ax.set_title('Optimality Gap of Upper Bounds', fontweight='bold')
ax.legend(loc='upper right')

# Annotate n=26 gap
idx_26 = n_values.index(26)
ax.annotate(f'{ft_gaps[idx_26]:.1f}%', xy=(idx_26 + 0.1, ft_gaps[idx_26]),
           xytext=(idx_26 + 0.5, ft_gaps[idx_26] + 5), fontsize=9,
           arrowprops=dict(arrowstyle='->', color='gray'))

fig_path = Path(__file__).parent / "figures" / "upper_bounds_comparison.png"
fig_path.parent.mkdir(exist_ok=True)
plt.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved figure to {fig_path}")
plt.close()
