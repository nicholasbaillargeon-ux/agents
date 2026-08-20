#!/usr/bin/env bash
# Point the research-notes repo at a Gitea remote and push what is already there.
#
# Everything the agents do works without this: briefs are committed locally
# either way, and `NotesRepo` treats a dead remote as a degradation rather than
# a failure. This just adds the push.
#
#   scripts/link-gitea.sh <user> <repo> [base-url]
#
# Requires GITEA_TOKEN in the environment (Settings -> Applications -> Generate
# Token, scope `write:repository`). The token is written into .env, which is
# gitignored — it is not committed anywhere.
set -euo pipefail

USER_NAME="${1:?usage: link-gitea.sh <user> <repo> [base-url]}"
REPO="${2:?usage: link-gitea.sh <user> <repo> [base-url]}"
BASE="${3:-http://localhost:3000}"
: "${GITEA_TOKEN:?set GITEA_TOKEN first}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NOTES="${AGENTS_DATA_DIR:-$HERE/data}/research-notes"

# Create the repo if it is not there yet; an existing one is not an error.
code=$(curl -s -o /tmp/gitea-create.json -w '%{http_code}' \
  -X POST "$BASE/api/v1/user/repos" \
  -H "Authorization: token $GITEA_TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"name\":\"$REPO\",\"private\":true,\"description\":\"Machine-written research notes from agents_work\"}")
case "$code" in
  201) echo "created $USER_NAME/$REPO" ;;
  409) echo "$USER_NAME/$REPO already exists" ;;
  *)   echo "gitea returned $code:"; cat /tmp/gitea-create.json; exit 1 ;;
esac

REMOTE="${BASE/:\/\//://$USER_NAME:$GITEA_TOKEN@}/$USER_NAME/$REPO.git"

git -C "$NOTES" remote remove origin 2>/dev/null || true
git -C "$NOTES" remote add origin "$REMOTE"
git -C "$NOTES" push -u origin main

# Persist for the agents, replacing any previous value.
sed -i '/^AGENTS_GIT_REMOTE=/d' "$HERE/.env"
echo "AGENTS_GIT_REMOTE=$REMOTE" >> "$HERE/.env"

echo "pushed, and AGENTS_GIT_REMOTE written to .env"
echo "restart the timers to pick it up:  sudo systemctl restart agents-web"
