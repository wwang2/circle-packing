#!/bin/bash
# Reproduce the JAX soft-body circle packing experiment
# Usage: cd to repo root, then: bash orbits/jax-soft-001/run.sh

set -e

echo "Running JAX soft-body optimizer..."
uv run python orbits/jax-soft-001/optimizer.py

echo ""
echo "Evaluating solution..."
uv run python research/eval/evaluator.py orbits/jax-soft-001/solution_n26.json

echo ""
echo "Generating visualization..."
uv run python orbits/jax-soft-001/visualize.py
