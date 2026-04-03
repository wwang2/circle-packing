#!/bin/bash
# Reproduce CMA-ES optimization for circle packing n=26
# Requires: uv, cma package
set -e

cd "$(dirname "$0")/../.."
uv run python orbits/cmaes-001/optimizer.py
echo ""
echo "Evaluating solution..."
uv run python research/eval/evaluator.py orbits/cmaes-001/solution_n26.json
