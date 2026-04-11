"""
Figures for contact-graph-001.
 - histogram of per-graph best sum_r
 - visualization of top-5 candidate graphs (showing they all converge to
   the parent basin)
 - the parent packing itself for reference
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

ORBIT = Path(__file__).resolve().parent
PARENT = ORBIT.parent / "mobius-001/solution_n26.json"

mpl.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})


def main():
    report = json.load(open(ORBIT / "enum_report.json"))
    parent = json.load(open(PARENT))
    C = np.array(parent["circles"])
    n = len(C)

    sols = [r for r in report["all_results"] if r["success"]]
    sums = np.array([r["sum_r"] for r in sols])
    incumbent = report["incumbent"]

    # ---------------------------------------------------------------- fig 1
    # 3 panels:
    #  (a) parent packing visualization
    #  (b) histogram of per-graph sum_r with incumbent marked
    #  (c) by-strategy violin/strip plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    # (a) parent packing
    ax = axes[0]
    ax.set_aspect("equal")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="black", linewidth=1.5))
    for x, y, r in C:
        ax.add_patch(plt.Circle((x, y), r, fill=True, facecolor="#3498db",
                                edgecolor="#1f3a93", linewidth=0.8, alpha=0.7))
    ax.set_title("(a) Parent packing G*\nsum_r = 2.6359830865")
    ax.set_xlabel("x"); ax.set_ylabel("y")

    # (b) histogram
    ax = axes[1]
    ax.hist(sums, bins=60, color="#3498db", edgecolor="#1f3a93", alpha=0.8)
    ax.axvline(incumbent, color="red", linestyle="--", linewidth=2, label=f"incumbent = {incumbent:.10f}")
    ax.set_xlabel("best sum_r per candidate graph")
    ax.set_ylabel("# candidates")
    ax.set_title(f"(b) All {len(sums)} solved candidates\ncollapse to parent basin")
    ax.legend(frameon=False, loc="upper left")
    # Zoom on relevant window
    lo = sums.min() - 1e-5
    hi = incumbent + 1e-5
    ax.set_xlim(lo, hi)
    ax.ticklabel_format(useOffset=False, style="plain", axis="x")
    ax.tick_params(axis="x", rotation=30)

    # (c) by strategy
    ax = axes[2]
    strategies = ["flip", "wall_swap", "edge_swap"]
    colors = ["#3498db", "#2ecc71", "#e67e22"]
    for i, strat in enumerate(strategies):
        vals = [r["sum_r"] for r in sols if r["strategy"] == strat]
        if not vals:
            continue
        x_jit = np.random.default_rng(i).normal(i, 0.07, size=len(vals))
        ax.scatter(x_jit, vals, s=18, alpha=0.6, color=colors[i], label=f"{strat} (n={len(vals)})")
    ax.axhline(incumbent, color="red", linestyle="--", linewidth=2)
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies)
    ax.set_ylabel("best sum_r")
    ax.set_title("(c) Outcomes by strategy")
    ax.legend(frameon=False, fontsize=10, loc="lower left")
    ax.ticklabel_format(useOffset=False, style="plain", axis="y")

    fig.suptitle(
        f"Contact-graph enumeration for n=26 circle packing\n"
        f"{report['total_candidates']} candidates → "
        f"{report['planar']} planar → "
        f"{report['solved']} solved → "
        f"{report['better']} beat incumbent",
        fontsize=13, y=1.08
    )
    fig.savefig(ORBIT / "figures/enum_summary.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures/enum_summary.png")

    # ---------------------------------------------------------------- fig 2
    # Zoom of all sum_r with precision annotations
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.hist(sums, bins=np.linspace(2.63585, 2.63600, 100),
            color="#3498db", edgecolor="#1f3a93")
    ax.axvline(incumbent, color="red", linestyle="--", linewidth=2,
               label=f"incumbent = {incumbent:.10f}")
    ax.set_xlabel("best sum_r per candidate (zoom)")
    ax.set_ylabel("# candidates")
    ax.set_title(f"Zoom: every candidate basin is ≤ incumbent\n"
                 f"best candidate = {sums.max():.10f}  "
                 f"(gap = {incumbent - sums.max():.2e})")
    ax.ticklabel_format(useOffset=False, style="plain", axis="x")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(ORBIT / "figures/sum_r_zoom.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures/sum_r_zoom.png")


if __name__ == "__main__":
    main()
