---
strategy: multi-start-nlp-slsqp
status: complete
eval_version: v1
metric: 2.6359830823
issue: 4
parent: null
---

## Multi-Start NLP with SLSQP + Penalty Shaping

### Result
**metric=2.6359830823** for n=26 (VALID). Matches OpenEvolve SOTA (~2.63598).

### Approach
3-stage nonlinear programming pipeline adapted from OpenEvolve:
1. **Diverse initializations** (132 patterns: OpenEvolve ring/hybrid/specialized/greedy/grid/corner/billiard + Poisson disk + random + perturbed variants)
2. **Stage 1:** Position optimization with L-BFGS-B (progressive penalty)
3. **Stage 2:** Radius optimization with SLSQP (constrained)
4. **Stage 3:** Joint optimization with SLSQP (all 3n variables, up to 10000 iterations)
5. **Basin-hopping** perturbation search from best solution

### Key Insight
The **concentric ring initialization** (1 center + 8 inner ring + 12 middle ring + 4 corners + 1 extra) consistently finds the best topological basin. Random/Poisson/hex initializations plateau around 2.62-2.63.

### Results

| Attempt | n | Metric | Notes |
|---------|---|--------|-------|
| V1 | 26 | 2.6233 | Basic multi-start + SLSQP polish |
| V1+refine | 26 | 2.6246 | Basin-hopping on V1 |
| V3 | 26 | 2.6273 | 136 inits, more Poisson disk |
| V4 | 26 | 2.6308 | Topology-aware inits + basin-hopping |
| V5 | 26 | 2.6308 | 400 parallel inits, no improvement |
| V6 | 26 | **2.6360** | OpenEvolve patterns, ring init wins |
| Polish | 26 | 2.6360 | Ultra-tight SLSQP, no improvement |
| n=10 | 10 | 1.5910 | Matches known best |

### What Worked
- Ring-based initialization from OpenEvolve (1+8+12+4+1 = 26)
- 3-stage optimization: L-BFGS-B for positions, SLSQP for radii, SLSQP for joint
- Wider radius bounds [0.01, 0.25] allow optimizer to find variable-size packings

### What Did Not Work
- Random/Poisson disk initializations: too far from optimal topology
- Single-circle relocation: solution too rigid
- Pair swaps: no improvement at this optimum
- Micro-perturbation polish: convergent after a few SLSQP iterations
- Massive parallel search (400 random inits): diminishing returns

### Seeds
All random seeds are deterministic and documented in optimizer_v6.py. Basin-hopping seed=42.
