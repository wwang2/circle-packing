#!/bin/bash
# Reproduce SA circle packing experiment for n=26
# Usage: bash orbits/sa-001/run.sh [n] [sa_iters] [seed]
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

N="${1:-26}"
SA_ITERS="${2:-1500000}"
SEED="${3:-42}"

cd "$REPO_DIR"

echo "Running SA optimizer: n=$N, iters=$SA_ITERS, seed=$SEED"
uv run python "$SCRIPT_DIR/optimizer.py" "$N" "$SA_ITERS" "$SEED"

echo ""
echo "Evaluating solution..."
uv run python research/eval/evaluator.py "$SCRIPT_DIR/solution_n${N}.json"
