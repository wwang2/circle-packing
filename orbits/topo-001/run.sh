#!/bin/bash
# Reproduce topo-001 experiments from seed
# Run from repo root: bash orbits/topo-001/run.sh
set -e

echo "=== TOPO-001: Contact Graph Topology Search ==="
echo "Running precision search (analytical Jacobian, ~160s)..."
uv run python orbits/topo-001/precision_search.py

echo ""
echo "Running global search (dual_annealing + basinhopping, ~300s)..."
uv run python orbits/topo-001/global_search.py

echo ""
echo "Evaluating best solution..."
uv run python research/eval/evaluator.py orbits/topo-001/solution_n26.json

echo ""
echo "Generating visualization..."
uv run python orbits/topo-001/visualize.py
