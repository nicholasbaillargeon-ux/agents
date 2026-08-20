#!/usr/bin/env bash
# The whole suite, then the benchmark gates cross-referenced against
# BENCHMARKS.md so a documented gate with no test behind it fails loudly.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "== full suite =="
.venv/bin/python -m pytest -p no:warnings "$@"

echo
echo "== benchmark gates (BENCHMARKS.md) =="
.venv/bin/python scripts/benchmark-report.py
