"""Visualize the n=30 circle packing solution and analyze contacts."""
import json, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path

WORK = Path(__file__).parent

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

def load(path):
    with open(path) as f:
        return np.array(json.load(f)["circles"])

def analyze_contacts(circles, tol=1e-6):
    """Find contact pairs (circle-circle and circle-wall)."""
    n = len(circles)
    contacts_cc = []
    contacts_wall = []

    for i in range(n):
        x, y, r = circles[i]
        # Wall contacts
        if abs(x - r) < tol: contacts_wall.append((i, 'left'))
        if abs(1 - x - r) < tol: contacts_wall.append((i, 'right'))
        if abs(y - r) < tol: contacts_wall.append((i, 'bottom'))
        if abs(1 - y - r) < tol: contacts_wall.append((i, 'top'))

    for i in range(n):
        xi, yi, ri = circles[i]
        for j in range(i+1, n):
            xj, yj, rj = circles[j]
            dist = math.sqrt((xi-xj)**2 + (yi-yj)**2)
            gap = dist - ri - rj
            if abs(gap) < tol:
                contacts_cc.append((i, j, gap))

    return contacts_cc, contacts_wall

# Load solution
circles = load(WORK / "solution_n30.json")
n = len(circles)
metric = np.sum(circles[:, 2])

# Analyze contacts
cc, cw = analyze_contacts(circles, tol=1e-5)

print(f"n = {n}")
print(f"Sum of radii = {metric:.10f}")
print(f"\nCircle-circle contacts: {len(cc)}")
for i, j, gap in cc:
    print(f"  {i}-{j}: gap={gap:.2e}")
print(f"\nWall contacts: {len(cw)}")
for i, wall in cw:
    print(f"  Circle {i} touches {wall} wall")

# Degrees of freedom analysis
total_contacts = len(cc) + len(cw)
dof = 3*n - total_contacts
print(f"\nTotal contacts: {total_contacts}")
print(f"Variables: {3*n}")
print(f"Degrees of freedom: {dof}")
print(f"{'Likely optimal (0 DOF)' if dof <= 0 else f'{dof} DOF remaining'}")

# Create figure with 2 panels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)

# Panel 1: Circle packing visualization
ax1.set_xlim(-0.02, 1.02)
ax1.set_ylim(-0.02, 1.02)
ax1.set_aspect('equal')
ax1.set_title(f'n=30 Circle Packing\nSum of radii = {metric:.10f}', fontweight='bold')

# Draw unit square
square = plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', linewidth=2)
ax1.add_patch(square)

# Color by radius
radii = circles[:, 2]
cmap = plt.cm.GnBu
norm = plt.Normalize(radii.min(), radii.max())

for i, (x, y, r) in enumerate(circles):
    color = cmap(norm(r))
    circle_patch = Circle((x, y), r, facecolor=color, edgecolor='black',
                          linewidth=0.5, alpha=0.8)
    ax1.add_patch(circle_patch)
    # Label
    ax1.text(x, y, str(i), ha='center', va='center', fontsize=6, fontweight='bold')

# Draw contact lines
for i, j, gap in cc:
    xi, yi, ri = circles[i]
    xj, yj, rj = circles[j]
    ax1.plot([xi, xj], [yi, yj], 'r-', linewidth=0.3, alpha=0.5)

ax1.set_xlabel('x')
ax1.set_ylabel('y')

# Panel 2: Radius distribution
radii_sorted = np.sort(radii)[::-1]
colors = [cmap(norm(r)) for r in radii_sorted]
bars = ax2.barh(range(n), radii_sorted, color=colors, edgecolor='black', linewidth=0.3)
ax2.set_xlabel('Radius')
ax2.set_ylabel('Circle index (sorted)')
ax2.set_title(f'Radius Distribution\n{len(cc)} circle-circle + {len(cw)} wall contacts = {total_contacts} total',
              fontweight='bold')
ax2.invert_yaxis()

# Add stats text
stats_text = (
    f"Contacts: {total_contacts}\n"
    f"DOF: {dof}\n"
    f"Mean r: {np.mean(radii):.6f}\n"
    f"Std r: {np.std(radii):.6f}\n"
    f"Min r: {np.min(radii):.6f}\n"
    f"Max r: {np.max(radii):.6f}"
)
ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes,
         verticalalignment='top', horizontalalignment='right',
         fontsize=10, family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

fig.savefig(WORK / "figures" / "n30_packing.png", dpi=200, bbox_inches='tight', facecolor='white')
print(f"\nFigure saved to {WORK / 'figures' / 'n30_packing.png'}")
