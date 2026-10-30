"""
rbc_reports/gmail_fetch.py
==========================
Pulls the RBC Insight digest emails straight from the user's Gmail over
IMAP, so they no longer have to be saved by hand. Each digest's HTML is
written into config.RBC_REPORTS_DIR (one file per digest, named by its
"as of" time and alert name -- an existing file is left alone), and
ingest.py then parses that folder as usual.

Read-only on the mailbox: it searches "[Gmail]/All Mail" (so an archived
or labelled digest is still found) for messages FROM
config.RBC_GMAIL_FROM only, opens the folder with EXAMINE (read-only) and
fetches with BODY.PEEK[] -- nothing is marked read, moved or deleted.

Credentials: the same GMAIL_SENDER / GMAIL_APP_PASSWORD pair from .env
that send_report.py sends with (the digests arrive in that same inbox; a
Gmail app password works for IMAP too). Read from .env by this module
itself, same self-contained pattern as demand_signals/darkpool.py's FINRA
creds. Unset -> fetch_digests() returns a "not configured" result, never
raises.

  python -m rbc_reports.gmail_fetch                 # last RBC_GMAIL_FETCH_DAYS days, then ingest
  python -m rbc_reports.gmail_fetch --days 60       # a wider window
  python -m rbc_reports.gmail_fetch --all           # every RBC digest in the mailbox
  python -m rbc_reports.gmail_fetch --dry-run       # list what would be saved, write nothing
"""
from __future__ import annotations

import argparse
import email
import imaplib
import os
import re
from datetime import timedelta
from email import policy
from pathlib import Path

from config import RBC_GMAIL_FETCH_DAYS, RBC_GMAIL_FROM, RBC_REPORTS_DIR
from rbc_reports.parser import is_digest, parse_digest
from time_utils import market_now

try:
    from dotenv import load_dotenv

    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass  # dotenv not installed -- that's fine, falls through to os.environ

GMAIL_USER = os.environ.get("GMAIL_SENDER", "")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")

IMAP_HOST = "imap.gmail.com"
MAILBOX = '"[Gmail]/All Mail"'


def is_configured() -> bool:
    return bool(GMAIL_USER and GMAIL_APP_PASSWORD) and "@" in GMAIL_USER


def digest_filename(as_of: str, subscription: str) -> str:
    """e.g. rbc_digest_20261029_1901_rbc-capital-markets-all-reports.html"""
    slug = re.sub(r"[^a-z0-9]+", "-", subscription.lower()).strip("-") or "digest"
    return f"rbc_digest_{as_of[:10].replace('-', '')}_{as_of[11:16].replace(':', '')}_{slug}.html"


def _html_body(raw: bytes) -> str | None:
    """The digest's HTML part. Decoded from its raw bytes rather than via
    get_content(): RBC's HTML part declares no MIME charset (only a
    <meta charset=utf-8> inside the HTML), and get_content() then falls
    back to ASCII, turning every curly quote into replacement characters."""
    msg = email.message_from_bytes(raw, policy=policy.default)
    part = msg.get_body(preferencelist=("html",))
    if part is None:
        return None
    body = part.get_payload(decode=True) or b""
    charset = part.get_content_charset() or "utf-8"
    try:
        return body.decode(charset, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


def fetch_digests(days: int | None = RBC_GMAIL_FETCH_DAYS, folder: Path = RBC_REPORTS_DIR,
                  dry_run: bool = False, imap_factory=imaplib.IMAP4_SSL) -> dict:
    """Saves every not-yet-saved RBC digest from the last `days` days
    (None = all) into `folder`. Returns {"configured", "messages",
    "saved", "existing", "skipped"} -- `saved` lists the new filenames.
    `imap_factory` is a test seam."""
    result = {"configured": is_configured(), "messages": 0, "saved": [], "existing": 0, "skipped": 0}
    if not result["configured"]:
        return result

    criteria = f'(FROM "{RBC_GMAIL_FROM}")'
    if days:
        since = (market_now() - timedelta(days=days)).strftime("%d-%b-%Y")
        criteria = f'(FROM "{RBC_GMAIL_FROM}" SINCE {since})'

    folder = Path(folder)
    imap = imap_factory(IMAP_HOST)
    try:
        imap.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        status, _ = imap.select(MAILBOX, readonly=True)
        if status != "OK":
            raise RuntimeError(f"cannot open {MAILBOX}")
        status, data = imap.search(None, criteria)
        if status != "OK":
            raise RuntimeError(f"IMAP search failed: {criteria}")
        ids = data[0].split()
        result["messages"] = len(ids)
        for msg_id in ids:
            status, parts = imap.fetch(msg_id, "(BODY.PEEK[])")
            raw = next((p[1] for p in parts if isinstance(p, tuple)), None) if status == "OK" else None
            page = _html_body(raw) if raw else None
            if not page or not is_digest(page):
                result["skipped"] += 1
                continue
            try:
                digest = parse_digest(page)
            except ValueError:
                result["skipped"] += 1
                continue
            path = folder / digest_filename(digest.as_of, digest.subscription)
            if path.exists():
                result["existing"] += 1
                continue
            if not dry_run:
                folder.mkdir(parents=True, exist_ok=True)
                path.write_text(page, encoding="utf-8")
            result["saved"].append(path.name)
    finally:
        try:
            imap.logout()
        except Exception:
            pass
    return result


def main(argv=None) -> int:
    from rbc_reports.ingest import ingest_dir

    p = argparse.ArgumentParser(description="Fetch RBC Insight digest emails from Gmail, then ingest them.")
    p.add_argument("--days", type=int, default=RBC_GMAIL_FETCH_DAYS)
    p.add_argument("--all", action="store_true", help="every RBC digest in the mailbox")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    r = fetch_digests(days=None if args.all else args.days, dry_run=args.dry_run)
    if not r["configured"]:
        print("Gmail not configured: set GMAIL_SENDER and GMAIL_APP_PASSWORD in .env")
        return 1
    print(f"{r['messages']} RBC message(s): {len(r['saved'])} new digest(s), "
          f"{r['existing']} already saved, {r['skipped']} not a digest")
    for name in r["saved"]:
        print(f"  {'would save' if args.dry_run else 'saved'}: {name}")
    if not args.dry_run and RBC_REPORTS_DIR.is_dir():
        s = ingest_dir(RBC_REPORTS_DIR)
        print(f"ingest: {s['digests']} digest(s), {s['reports_new']} new report(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
