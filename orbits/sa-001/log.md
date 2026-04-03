---
strategy: multi-start-three-stage-optimization
status: complete
eval_version: eval-v1
metric: 2.6359830849
issue: 3
parent: null
---

# SA-001: Multi-Start Three-Stage Optimization

## Result
**metric = 2.6359830849** (valid, n=26) -- matches OpenEvolve SOTA (~2.6360).

## Approach Evolution

### V1: Pure SA (abandoned)
Pure Python SA was too slow. 500k iters per start timed out.

### V2: Multi-start local search + SLSQP
- 50 diverse initializations (hex grid, random, grid-varied, mixed sizes)
- Push-apart + greedy radius growth
- Local search (300 perturbation trials)
- SLSQP polish
- Result: **2.6244** -> **2.6317** after basin-hopping

### V3: Greedy placement + radius distributions
- 60 candidate radius distributions (power-law, uniform, known-good)
- Greedy largest-gap placement
- Result: **2.6317** (small improvement from different init)

### V4: Three-stage optimization (final approach)
Inspired by OpenEvolve methodology:
1. Stage 1: Position optimization with L-BFGS-B + penalty (300 iters)
2. Stage 2: Radii optimization with SLSQP (500 iters)
3. Stage 3: Joint optimization with SLSQP (3x3000 iters)

Seven initialization patterns: corners+edges+center, hybrid, billiard,
greedy-largest-gap, diagonal-symmetric, grid-optimized.

The **hybrid pattern** (large center + concentric rings + corner circles)
at seed 42424+55*7 found the best basin: **2.6360**.

### Refinement
Perturbation + re-polish improved to **2.6359830849**.
Three further attempts found no improvement (stuck at local optimum).

## Key Insights
- Pure SA in Python is impractical for n=26 -- too slow per iteration
- Initialization topology matters more than optimization budget
- Three-stage (position -> radii -> joint) outperforms single-stage SLSQP
- The hybrid pattern (center + rings) finds better basins than grid-based inits
- Once in the right basin, SLSQP converges precisely; perturbations can't escape
- Diagonal symmetry is a strong feature of optimal n=26 packings

## Seeds
- Base seed: 42424
- Best init: pattern_hybrid(n=26, seed=42424+55*7=42809)
- Refinement seed: 88888

## Files
- `optimizer_v4.py` -- main optimizer (three-stage)
- `improve.py` -- perturbation refinement loop
- `solution_n26.json` -- best solution
- `figures/solution_n26.png` -- visualization
- `run.sh` -- reproduction script
