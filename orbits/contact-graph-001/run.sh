#!/usr/bin/env bash
# One-command reproducer for orbit/contact-graph-001.
#
# Expects Python 3 with numpy, scipy, networkx, matplotlib.
set -euo pipefail

cd "$(dirname "$0")/../.."

echo "[run.sh] Running contact-graph enumeration..."
python3 orbits/contact-graph-001/solution.py

echo "[run.sh] Generating figures..."
python3 orbits/contact-graph-001/make_figures.py

echo "[run.sh] Done. See orbits/contact-graph-001/enum_report.json"
