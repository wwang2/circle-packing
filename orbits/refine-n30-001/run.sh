#!/bin/bash
# Reproduce the n=30 refinement experiment
# Starting from diffevo-001's solution, refine with SLSQP
set -e
cd "$(dirname "$0")/../.."

echo "=== Step 1: SLSQP Refinement ==="
uv run python orbits/refine-n30-001/step1_slsqp.py

echo ""
echo "=== Evaluate ==="
uv run python research/eval/evaluator.py orbits/refine-n30-001/solution_n30.json

echo ""
echo "=== Visualize ==="
uv run python orbits/refine-n30-001/visualize.py
