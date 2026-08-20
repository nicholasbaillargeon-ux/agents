#!/usr/bin/env bash
# Write a research brief for every symbol on the watchlist, one at a time.
# Sequential on purpose: EDGAR publishes a 10 req/s limit and the point of the
# rate limiter in netcache.py is to stay well under it.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

WATCHLIST="${AGENTS_WATCHLIST:-$(grep -E '^AGENTS_WATCHLIST=' .env | cut -d= -f2-)}"
IFS=',' read -ra SYMBOLS <<< "$WATCHLIST"

for symbol in "${SYMBOLS[@]}"; do
  echo "== $symbol"
  .venv/bin/agents research "$symbol" || echo "  (failed, continuing)"
done
