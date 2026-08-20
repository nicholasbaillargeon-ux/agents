#!/usr/bin/env bash
# Install (or reinstall) the systemd units. Idempotent.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNITS=(agents-web.service agents-briefing.service agents-briefing.timer
       agents-scout.service agents-scout.timer)

for unit in "${UNITS[@]}"; do
  sudo install -m 0644 "$HERE/$unit" "/etc/systemd/system/$unit"
done

sudo systemctl daemon-reload
sudo systemctl enable --now agents-web.service
sudo systemctl enable --now agents-briefing.timer
sudo systemctl enable --now agents-scout.timer

echo
systemctl --no-pager --lines=0 status agents-web.service | head -4
echo
systemctl list-timers --no-pager 'agents-*' || true
echo
echo "Dashboard: http://$(hostname -I | awk '{print $1}'):8110"
