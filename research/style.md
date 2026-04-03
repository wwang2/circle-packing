---
alwaysApply: false
---
# Plotting Style Guide

When creating matplotlib figures, follow this consistent style:

## Layout Philosophy
- **Prefer grid layouts.** When presenting multiple related plots, arrange them as subplots in a grid rather than separate figures.
- Use `GridSpec` for complex layouts with custom width/height ratios.
- Use `fig, axes = plt.subplots(nrows, ncols, figsize=(...), constrained_layout=True)` as the default pattern.
- Aim for information density — one well-organized grid figure is better than 5 separate PNGs.
- For orbit comparisons: one row per strategy, columns for different metrics or views.

## Font & Typography
- `font.family`: "monospace"
- `font.monospace`: ["DejaVu Sans Mono", "Menlo", "Consolas", "Monaco"]
- `font.size`: 11-12
- Use `fontweight='bold'` for titles and important labels

## Axes & Spines
- `axes.spines.top`: False
- `axes.spines.right`: False
- `axes.grid`: True
- `grid.alpha`: 0.2-0.25
- `grid.linewidth`: 0.5-0.7
- `xtick.direction`: "out"
- `ytick.direction`: "out"
- `legend.frameon`: False

## Figure Setup
- Use `constrained_layout=True` for multi-panel figures
- Default DPI: 150-200 for normal, 300+ for publication
- `figure.facecolor` and `savefig.facecolor`: "white"

## Color Schemes
- Primary: "GnBu", "Blues", "Greens", "viridis"
- Diverging: "RdBu_r" for signed values
- Discrete: `tab20` or explicit hex with semantic meaning

## Output
- Save with `bbox_inches='tight'`, `facecolor='white'`

## Standard rcParams Block

```python
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
```

When writing orbit logs:
- Lead with the result, not the method.
- Include one key figure per orbit.
- Keep it under 500 words unless warranted.
