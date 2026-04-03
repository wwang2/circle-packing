#!/bin/bash
# Reproduce NLP-001 circle packing optimization
# Usage: bash orbits/nlp-001/run.sh [n] [num_inits]
set -e
cd "$(dirname "$0")/../.."

N=${1:-26}
INITS=${2:-80}

echo "=== NLP-001: Multi-start NLP optimizer for n=$N ==="
uv run python orbits/nlp-001/optimizer.py "$N" "orbits/nlp-001/solution_n${N}.json"
echo ""
echo "=== Evaluating ==="
uv run python research/eval/evaluator.py "orbits/nlp-001/solution_n${N}.json"
