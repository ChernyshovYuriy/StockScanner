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
#
# The `cp -v .../stockscanner-*.service|.timer` globs below already pick up
# any new unit file added to this directory (e.g. the 2026-09 Kangaroo Tail
# sleeve's 6 files, or the Scanner Board's 2) with no change needed here —
# but per the "does NOT enable/start anything not already enabled" rule
# above, a BRAND NEW service's timer still needs a one-time manual enable
# on first deploy (see system/info); this script alone won't start it
# running.

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
echo
echo "First time deploying a brand-new service's units (e.g. Kangaroo Tail, or"
echo "the Scanner Board)? This script does not enable/start them — run the"
echo "enable --now commands in system/info for that service once."
