#!/bin/bash
# Reproduce gentopo-001 experiments
cd "$(dirname "$0")"
cd ../..  # repo root

echo "=== Phase 1: Genetic Topology Search ==="
uv run python orbits/gentopo-001/gentopo_search.py

echo ""
echo "=== Evaluate solution ==="
uv run python research/eval/evaluator.py orbits/gentopo-001/solution_n26.json
