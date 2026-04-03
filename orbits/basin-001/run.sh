#!/bin/bash
# Basin-hopping optimizer for circle packing
# Usage: bash run.sh [n] [seed] [starts] [timeout]
set -e
cd "$(dirname "$0")/../.."
PYTHONUNBUFFERED=1 uv run python orbits/basin-001/optimizer_v3.py "${1:-26}" "${2:-42}" "${3:-25}" "${4:-480}"
echo ""
echo "Evaluating solution..."
uv run python research/eval/evaluator.py "orbits/basin-001/solution_n${1:-26}.json"
