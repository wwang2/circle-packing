#!/bin/bash
# Reproduce the n=30 and n=32 refinement experiments
# Starting from diffevo-001's solutions, refine with multiple methods
set -e
cd "$(dirname "$0")/../.."

echo "=== N=30: SLSQP Refinement ==="
uv run python orbits/refine-n30-001/step1_slsqp.py

echo ""
echo "=== N=30: Evaluate ==="
uv run python research/eval/evaluator.py orbits/refine-n30-001/solution_n30.json

echo ""
echo "=== N=32: Full Optimization Pipeline ==="
uv run python orbits/refine-n30-001/optimize_n32.py

echo ""
echo "=== N=32: Topology Search ==="
uv run python orbits/refine-n30-001/topology_search_n32.py

echo ""
echo "=== N=32: Evaluate ==="
uv run python research/eval/evaluator.py orbits/refine-n30-001/solution_n32.json

echo ""
echo "=== Visualize ==="
uv run python orbits/refine-n30-001/visualize.py
uv run python orbits/refine-n30-001/visualize_n32.py
