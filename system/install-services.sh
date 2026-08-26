#!/usr/bin/env bash
#
# system/install-services.sh
# ===========================
# Copies this repo's systemd unit files into /etc/systemd/system/ and
# reloads systemd so the changes take effect. Run this after `git pull`
# whenever a .service/.timer file here changes (e.g. the 2026-08-26
# User=nvidia -> User=pi fix).
#
# Usage:
#   sudo bash system/install-services.sh
#
# Does NOT enable or start anything that isn't already enabled — existing
# timers keep their current enabled/disabled state and just pick up the
# updated unit file on their next scheduled fire. See system/info for the
# enable/start commands for a unit that isn't running yet.
#
# Does restart the two momentum-sleeve services that were failing at the
# systemd USER step (see MEMORY / commit 322d2cc), so the fix takes effect
# immediately rather than waiting for their next scheduled run.

set -euo pipefail

if [ "$EUID" -ne 0 ]; then
    echo "Run with sudo: sudo bash system/install-services.sh" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Copying unit files to /etc/systemd/system/ ..."
cp -v "$SCRIPT_DIR"/stockscanner-*.service /etc/systemd/system/
cp -v "$SCRIPT_DIR"/stockscanner-*.timer /etc/systemd/system/

echo "Reloading systemd ..."
systemctl daemon-reload

echo "Restarting momentum-monitor and momentum-pipeline to pick up the User= fix now ..."
systemctl restart stockscanner-momentum-monitor.service stockscanner-momentum-pipeline.service

echo
echo "Done. Verify with:"
echo "  journalctl -u stockscanner-momentum-monitor.service -n 20 --no-pager"
echo "  journalctl -u stockscanner-momentum-pipeline.service -n 20 --no-pager"
echo "  systemctl list-timers --all | grep stockscanner"
