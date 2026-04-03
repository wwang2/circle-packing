#!/bin/bash
# Reproduce gentopo-001 experiments
cd "$(dirname "$0")"
cd ../..  # repo root

echo "=== v4: Fast SLSQP (2000 starts) ==="
uv run python orbits/gentopo-001/v4_fast_slsqp.py

echo ""
echo "=== v9: Discrete Topology Search ==="
uv run python orbits/gentopo-001/v9_discrete_topo.py

echo ""
echo "=== Evaluate solution ==="
uv run python research/eval/evaluator.py orbits/gentopo-001/solution_n26.json
