#!/bin/bash
# Reproduce random-topo-001 experiments
# Usage: cd repo_root && bash orbits/random-topo-001/run.sh

set -e

echo "=== random-topo-001: Massive Random Topology Search ==="

# Search v1: Random inits + perturbations (1600+ configs)
echo "Running search v1..."
uv run python orbits/random-topo-001/search.py

# Search v3: Structured inits (760 configs)
echo "Running search v3..."
PYTHONUNBUFFERED=1 uv run python orbits/random-topo-001/search_v3.py

# Evaluate best solution
echo "Evaluating..."
uv run python research/eval/evaluator.py orbits/random-topo-001/solution_n26.json

# Visualize
echo "Generating figure..."
uv run python orbits/random-topo-001/visualize.py

echo "Done."
