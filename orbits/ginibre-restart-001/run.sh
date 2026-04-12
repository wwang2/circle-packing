#!/usr/bin/env bash
# Reproduce the Ginibre DPP multi-start experiment
# Usage: bash run.sh [--quick]  (--quick for 10 restarts, default 500)

set -euo pipefail
cd "$(dirname "$0")"

echo "=== Ginibre DPP Multi-Start Circle Packing ==="
echo "Working dir: $(pwd)"

# Copy the known best solution for fallback
if [ ! -f best_solution.json ]; then
    cp ../../research/solutions/mobius-001/solution_n26.json best_solution.json 2>/dev/null || true
fi

# Run the campaign
python3 solver.py "$@"

# Run evaluator on the best solution
echo ""
echo "=== Running evaluator ==="
python3 ../../research/eval/evaluator.py solver.py
