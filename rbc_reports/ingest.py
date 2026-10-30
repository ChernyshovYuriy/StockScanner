"""
rbc_reports/ingest.py
=====================
Parses every saved digest (*.html) in config.RBC_REPORTS_DIR into
data/rbc_reports.db. Idempotent -- a file already ingested (by name) is
not re-read, and a report already stored is skipped, so re-running over
the whole folder is safe and cheap. Non-digest HTML files are skipped
with a note.

  python -m rbc_reports.ingest                  # ingest RBC_REPORTS_DIR
  python -m rbc_reports.ingest --dir PATH       # another folder
  python -m rbc_reports.ingest --dry-run        # parse + print, write nothing
"""
from __future__ import annotations

import argparse
from pathlib import Path

from config import RBC_REPORTS_DIR
from rbc_reports import store
from rbc_reports.parser import is_digest, parse_digest
from time_utils import market_now


def ingest_dir(folder: Path, conn=None, dry_run: bool = False) -> dict:
    """Returns {"files", "digests", "reports_seen", "reports_new",
    "already", "skipped"} -- `already` counts files ingested on an earlier
    run (not re-read)."""
    stats = {"files": 0, "digests": 0, "reports_seen": 0, "reports_new": 0, "already": 0, "skipped": []}
    done = set()
    if not dry_run:
        conn = conn or store.connect()
        done = store.ingested_files(conn)
    now = market_now().isoformat()
    for path in sorted(Path(folder).glob("*.html")):
        stats["files"] += 1
        if path.name in done:
            stats["already"] += 1
            continue
        page = path.read_text(encoding="utf-8", errors="replace")
        if not is_digest(page):
            stats["skipped"].append(path.name)
            continue
        try:
            digest = parse_digest(page)
        except ValueError as e:
            stats["skipped"].append(f"{path.name} ({e})")
            continue
        stats["digests"] += 1
        stats["reports_seen"] += len(digest.items)
        if not dry_run:
            stats["reports_new"] += store.save_digest(conn, digest, path.name, now, commit=False)
    if not dry_run:
        conn.commit()
    return stats


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Ingest saved RBC Insight digest emails.")
    p.add_argument("--dir", type=Path, default=RBC_REPORTS_DIR)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    if not args.dir.is_dir():
        print(f"No such folder: {args.dir}")
        return 1
    s = ingest_dir(args.dir, dry_run=args.dry_run)
    print(f"{s['files']} file(s), {s['digests']} digest(s), {s['reports_seen']} report(s) parsed"
          + ("" if args.dry_run else f", {s['reports_new']} new"))
    for name in s["skipped"]:
        print(f"  skipped: {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
