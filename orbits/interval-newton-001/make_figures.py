"""Generate certificate figures:

Panel (a) — the certified packing: 26 disks with contact graph overlay, in the
            unit square, colored by radius.
Panel (b) — ε-sweep chart: max K-image width (log scale) vs box radius ε,
            with a shaded "contracted" region where K ⊂ interior(B).
Panel (c) — per-coordinate enclosure width at the smallest certified ε:
            shows that every primal/dual coordinate is pinned to ~1e-12 or
            tighter.
"""
from __future__ import annotations

import json
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)

mpl.rcParams.update({
    "font.family": "monospace",
    "font.monospace": ["DejaVu Sans Mono", "Menlo", "Consolas", "Monaco"],
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlepad": 10.0,
    "axes.labelpad": 4.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox_inches": "tight",
    "savefig.pad_inches": 0.2,
    "figure.dpi": 150,
})


def load():
    cert = json.loads((HERE / "certificate.json").read_text())
    polished = json.loads((HERE / "kkt_point_polished.json").read_text())
    box = json.loads((HERE / "krawczyk_box.json").read_text())
    return cert, polished, box


def plot_packing(ax, polished):
    n = polished["n"]
    x = np.array(polished["x"])
    y = np.array(polished["y"])
    r = np.array(polished["r"])
    contacts_dd = polished["contacts_dd"]
    contacts_wall = polished["contacts_wall"]

    # Unit square
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], "-", color="#333", lw=1.5)

    # Disks colored by radius
    cmap = plt.cm.GnBu
    rmin, rmax = r.min(), r.max()
    for i in range(n):
        color = cmap(0.25 + 0.65 * (r[i] - rmin) / max(rmax - rmin, 1e-12))
        circ = plt.Circle((x[i], y[i]), r[i], facecolor=color,
                          edgecolor="#1f4e79", lw=0.9, zorder=2)
        ax.add_patch(circ)

    # Contact graph edges (disk-disk)
    for i, j in contacts_dd:
        ax.plot([x[i], x[j]], [y[i], y[j]], "-", color="#b22222",
                lw=0.8, alpha=0.7, zorder=3)

    # Wall contact marks
    for i, side in contacts_wall:
        if side == "L":
            ax.plot([0, x[i]], [y[i], y[i]], "-", color="#b22222", lw=0.8,
                    alpha=0.7, zorder=3)
        elif side == "R":
            ax.plot([1, x[i]], [y[i], y[i]], "-", color="#b22222", lw=0.8,
                    alpha=0.7, zorder=3)
        elif side == "B":
            ax.plot([x[i], x[i]], [0, y[i]], "-", color="#b22222", lw=0.8,
                    alpha=0.7, zorder=3)
        elif side == "T":
            ax.plot([x[i], x[i]], [1, y[i]], "-", color="#b22222", lw=0.8,
                    alpha=0.7, zorder=3)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_title("(a) Certified packing  n=26\n58 disk-disk + 20 wall contacts",
                 fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(False)


def plot_sweep(ax, cert):
    sweep = cert["sweep"]
    eps_arr = np.array([s["eps"] for s in sweep])
    widths = np.array([s["K_widths_max"] for s in sweep])
    contracted = np.array([s["contracted"] for s in sweep])
    row_norms = np.array([s["rowsum_ImCJ"] for s in sweep])

    # Shade contracted region
    if contracted.any():
        eps_good = eps_arr[contracted]
        ax.axvspan(eps_good.min() / 3, eps_good.max() * 3, alpha=0.1,
                   color="#2e7d32", label="contracted region")

    ax.loglog(eps_arr, widths, "o-", color="#1f77b4", markersize=8,
              lw=1.8, label="max K-image width")
    ax.loglog(eps_arr, eps_arr, "--", color="#888", lw=1.0,
              label="y = ε  (box width)")
    ax.loglog(eps_arr, row_norms, "s--", color="#d4502a", markersize=7,
              lw=1.4, alpha=0.8, label=r"$\|I - C J(B)\|_{row sum}$")

    # Mark failures with red X
    if (~contracted).any():
        ax.loglog(eps_arr[~contracted], widths[~contracted], "X",
                  color="#c62828", markersize=16, markeredgewidth=3,
                  label="contraction failed")

    ax.set_xlabel(r"box radius  $\varepsilon$")
    ax.set_ylabel("magnitude")
    ax.set_title("(b) Krawczyk ε-sweep", fontweight="bold")
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    ax.grid(True, alpha=0.25, which="both")


def plot_enclosure_widths(ax, box, polished):
    """Show per-coordinate K-enclosure widths for the smallest certified eps."""
    K_lo = np.array(box["K_lo"])
    K_hi = np.array(box["K_hi"])
    widths = K_hi - K_lo
    n = polished["n"]
    m = polished["m"]
    labels = ([f"x{i+1}" for i in range(n)] +
              [f"y{i+1}" for i in range(n)] +
              [f"r{i+1}" for i in range(n)] +
              [f"λ{k+1}" for k in range(m)])
    groups = (np.array(
        [0] * n + [1] * n + [2] * n + [3] * m))
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]
    group_names = ["x", "y", "r", "λ"]

    idx = np.arange(len(widths))
    for g in range(4):
        mask = groups == g
        ax.semilogy(idx[mask], widths[mask], "o", color=colors[g], markersize=3.5,
                    label=group_names[g])
    ax.axhline(2 * box["eps"], color="#888", ls="--", lw=1.0,
               label=f"input box width 2ε = {2*box['eps']:.0e}")
    ax.set_xlabel("coordinate index  (x₁..x₂₆, y₁..y₂₆, r₁..r₂₆, λ₁..λ₇₈)")
    ax.set_ylabel("K-enclosure width")
    ax.set_title(f"(c) Certified per-coordinate tightness   ε = {box['eps']:.0e}",
                 fontweight="bold")
    ax.legend(loc="upper right", ncol=2, frameon=False, fontsize=10)
    ax.set_xlim(-2, len(widths) + 2)
    ax.grid(True, alpha=0.25, which="both")


def main():
    cert, polished, box = load()

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1.15, 1.3]})
    plot_packing(axes[0], polished)
    plot_sweep(axes[1], cert)
    plot_enclosure_widths(axes[2], box, polished)

    # Header
    fig.suptitle(
        f"Krawczyk rigorous certificate for n=26 circle packing local KKT point\n"
        f"sum_r ∈ [{cert['sum_r_lower_bound']:.13f}, {cert['sum_r_upper_bound']:.13f}]   "
        f"(width {cert['sum_r_width']:.2e})",
        fontsize=13, fontweight="bold",
    )

    fig.savefig(FIG_DIR / "certificate_panel.png")
    plt.close(fig)
    print(f"Wrote {FIG_DIR / 'certificate_panel.png'}")


if __name__ == "__main__":
    main()
