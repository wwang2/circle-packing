#!/bin/bash
# Reproduce CMA-ES optimization for circle packing n=26
# Best approach was v2 (CMA-ES + perturbation polish)
# Requires: uv, cma package
set -e

cd "$(dirname "$0")/../.."
echo "Running CMA-ES optimizer v2..."
uv run python orbits/cmaes-001/optimizer_v2.py
echo ""
echo "Evaluating solution..."
uv run python research/eval/evaluator.py orbits/cmaes-001/solution_n26.json
echo ""
echo "Generating figure..."
uv run python orbits/cmaes-001/plot_solution.py
