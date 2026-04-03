#!/bin/bash
# Reproduce circle packing experiment for n=26
# Usage: bash orbits/sa-001/run.sh [num_trials] [seed]
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

NUM_TRIALS="${1:-80}"
SEED="${2:-42424}"

cd "$REPO_DIR"

echo "=== Phase 1: Multi-start three-stage optimization ==="
uv run python "$SCRIPT_DIR/optimizer_v4.py" "$NUM_TRIALS" "$SEED"

echo ""
echo "=== Phase 2: Perturbation refinement ==="
uv run python "$SCRIPT_DIR/improve.py" 100 88888

echo ""
echo "=== Phase 3: Visualization ==="
uv run python "$SCRIPT_DIR/plot_solution.py"

echo ""
echo "=== Evaluation ==="
uv run python research/eval/evaluator.py "$SCRIPT_DIR/solution_n26.json"
