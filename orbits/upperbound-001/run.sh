#!/bin/bash
# Reproduce upper bound computations for circle packing sum-of-radii.
# Run from the orbit directory.
set -e

cd "$(dirname "$0")"

echo "=== Corrected Upper Bounds for Circle Packing ==="
echo ""

echo "--- Final corrected bounds (all methods, validated) ---"
python3 corrected_bounds.py

echo ""
echo "Done. Results in corrected_bounds.json, figure in figures/upper_bounds_summary.png"
