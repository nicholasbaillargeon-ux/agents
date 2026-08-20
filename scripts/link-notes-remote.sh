#!/usr/bin/env bash
# Point the research-notes repo at any git remote and push what is already there.
#
#   scripts/link-notes-remote.sh git@github.com:you/research-notes.git
#
# The remote must already exist and be empty (GitHub will not create it for you;
# `gh repo create you/research-notes --private` does, if you have the CLI).
#
# Nothing about the agents changes: briefs are committed locally either way, and
# `NotesRepo` treats a failed push as a degradation on the run rather than a lost
# commit. This only adds the push.
set -euo pipefail

REMOTE="${1:?usage: link-notes-remote.sh <git-remote-url>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NOTES="${AGENTS_DATA_DIR:-$HERE/data}/research-notes"

[ -d "$NOTES/.git" ] || { echo "no notes repo at $NOTES — run an agent first"; exit 1; }

git -C "$NOTES" remote remove origin 2>/dev/null || true
git -C "$NOTES" remote add origin "$REMOTE"
git -C "$NOTES" push -u origin main

sed -i '/^AGENTS_GIT_REMOTE=/d' "$HERE/.env"
echo "AGENTS_GIT_REMOTE=$REMOTE" >> "$HERE/.env"

echo
echo "pushed $(git -C "$NOTES" rev-list --count main) commits; AGENTS_GIT_REMOTE written to .env"
echo "restart the dashboard to show the remote:  sudo systemctl restart agents-web"
