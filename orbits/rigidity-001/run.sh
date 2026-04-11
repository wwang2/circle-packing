#!/usr/bin/env bash
# Reproduce the rigidity-001 diagnostic.
set -euo pipefail
cd "$(dirname "$0")"
python solution.py
