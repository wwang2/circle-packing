#!/bin/bash
# Reproduce diffevo-001 experiments
# Usage: bash run.sh [n] [num_seeds] [maxiter] [popsize]
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR/../.."

N=${1:-10}
SEEDS=${2:-5}
MAXITER=${3:-2000}
POPSIZE=${4:-40}

echo "=== Differential Evolution: n=$N, seeds=$SEEDS, maxiter=$MAXITER, popsize=$POPSIZE ==="
uv run python orbits/diffevo-001/solver.py "$N" "$SEEDS" "$MAXITER" "$POPSIZE"

echo ""
echo "=== Evaluating ==="
uv run python research/eval/evaluator.py "orbits/diffevo-001/solution_n${N}.json"
