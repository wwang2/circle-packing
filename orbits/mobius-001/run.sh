#!/bin/bash
# Reproduce mobius-001 experiments from seed
# n=26: metric=2.6359830865 (confirmed global optimum)
# n=32: metric=2.9395727728 (multi-start + basin hopping + squeeze)
set -e
cd "$(dirname "$0")/../.."

echo "=== n=26 Phase 1: Multi-start search ==="
uv run python orbits/mobius-001/search_v2.py

echo ""
echo "=== n=26 Phase 2: KKT refinement ==="
uv run python orbits/mobius-001/kkt_refine.py

echo ""
echo "=== n=26 Phase 3: Tolerance squeeze ==="
uv run python orbits/mobius-001/squeeze.py

echo ""
echo "=== n=32 Phase 1: Multi-start optimizer ==="
uv run python orbits/mobius-001/n32_optimizer.py

echo ""
echo "=== n=32 Phase 2: Intensive refinement ==="
uv run python orbits/mobius-001/n32_refine.py

echo ""
echo "=== n=32 Phase 3: Tolerance squeeze ==="
uv run python orbits/mobius-001/n32_squeeze.py

echo ""
echo "=== Generate figures ==="
uv run python orbits/mobius-001/make_figures_v2.py

echo ""
echo "=== Evaluate final solutions ==="
uv run python research/eval/evaluator.py orbits/mobius-001/solution_n26.json
uv run python research/eval/evaluator.py orbits/mobius-001/solution_n32.json
