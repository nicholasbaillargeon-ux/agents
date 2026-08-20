#!/usr/bin/env bash
# The whole suite, then the benchmark gates on their own so the count is visible.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "== full suite =="
.venv/bin/python -m pytest -p no:warnings "$@"

echo
echo "== benchmark gates (BENCHMARKS.md) =="
.venv/bin/python -m pytest -p no:warnings -m benchmark "$@"
