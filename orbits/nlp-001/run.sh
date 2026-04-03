#!/bin/bash
# Reproduce NLP-001 circle packing optimization
# Best result: n=26, metric=2.63598 (matches OpenEvolve SOTA)
# Usage: bash orbits/nlp-001/run.sh
set -e
cd "$(dirname "$0")/../.."

echo "=== NLP-001: OpenEvolve-style 3-stage optimizer for n=26 ==="
echo "This takes ~20-25 minutes with 132 initializations + basin-hopping."
uv run python orbits/nlp-001/optimizer_v6.py "orbits/nlp-001/solution_n26.json"

echo ""
echo "=== Evaluating ==="
uv run python research/eval/evaluator.py "orbits/nlp-001/solution_n26.json"

echo ""
echo "=== Generating figure ==="
uv run python orbits/nlp-001/visualize.py
