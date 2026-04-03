---
strategy: aggressive-basin-hopping-n32
status: done
eval_version: eval-v1
metric: 2.9395727728
issue: 15
parent: mobius-001
---
# PUSH-N32-001: Aggressive Basin-Hopping for n=32

## solver.py run (2026-04-03)

Ran massive optimization campaign:
- Phase 1: Basin-hopping with penalty L-BFGS-B (20 seeds x 500 hops, analytical gradients) - 407s
- Phase 2: Multi-start progressive penalty L-BFGS-B + SLSQP polish (300 starts) - 471s
- Phase 3: Fine-grained SLSQP local search (500 trials) - 355s
- Phase 4: Swap neighborhood search (200 trials) - 120s
- Final SLSQP polish

Total: 1353s (~23 min), 10,000+ basin-hopping iterations + 1000 multi-start trials.

**Result: No improvement found. Current solution at 2.9395727728 is a strong local optimum.**

This is consistent with the SOTA target of "2.939+" for n=32. The solution appears to be at or very near the global optimum.
