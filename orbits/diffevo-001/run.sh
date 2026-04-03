#!/bin/bash
# Reproduce diffevo-001 experiments
# Usage: bash run.sh [n]
# Full run: bash run.sh all
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR/../.."

run_one() {
    local N=$1
    local STARTS=${2:-50}
    echo "=== Solving n=$N with $STARTS starts ==="
    uv run python -u orbits/diffevo-001/solver.py "$N" "$STARTS"
    echo "=== Refining n=$N ==="
    uv run python -u orbits/diffevo-001/refine.py "orbits/diffevo-001/solution_n${N}.json" 300
    echo "=== Evaluating n=$N ==="
    uv run python research/eval/evaluator.py "orbits/diffevo-001/solution_n${N}.json"
    echo ""
}

if [ "${1:-all}" = "all" ]; then
    run_one 10 30
    run_one 28 30
    run_one 30 30
    run_one 32 50
    echo "=== Generating figures ==="
    uv run python orbits/diffevo-001/plot.py
else
    run_one "$1" "${2:-30}"
fi
