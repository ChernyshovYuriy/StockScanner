"""
send_report.py
==============
Sends the latest HTML report as a rich-HTML email via Gmail SMTP.

Requirements
------------
Gmail requires an **App Password** (not your regular password).
Steps to create one:
  1. Go to https://myaccount.google.com/security
  2. Enable 2-Step Verification if not already on
  3. Search for "App passwords" → create one (name it e.g. "TSX Pipeline")
  4. Copy the 16-character password into GMAIL_APP_PASSWORD below
     (or set it as an env var — see CONFIG section)

Usage
-----
  python send_report.py                          # sends the newest report in alerts/
  python send_report.py --file path/to/file.html # send a specific file
  python send_report.py --date 20260301          # send report for a specific date (YYYYMMDD)
  python send_report.py --dry-run                # validate config & file without sending

Install (nothing beyond stdlib needed for sending):
  pip install python-dotenv          # optional — only if you use a .env file
"""

import argparse
import os
import smtplib
import socket
import sys
from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from dotenv import load_dotenv

from time_utils import market_now, market_today_str

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  — edit these or set the matching environment variables
# ─────────────────────────────────────────────────────────────────────────────

GMAIL_SENDER = os.environ.get("GMAIL_SENDER", "your_address@gmail.com")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "xxxx xxxx xxxx xxxx")  # 16-char App Password
GMAIL_RECIPIENT = os.environ.get("GMAIL_RECIPIENT", "your_address@gmail.com")

# Where auto_pipeline.py drops its reports
ALERTS_DIR = Path(os.environ.get("ALERTS_DIR", "alerts"))

# Subject line template  —  {date} is replaced with today's date string
SUBJECT_TEMPLATE = "📈 TSX Pipeline Report — {date}"

# SMTP settings (do not change for Gmail)
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587  # STARTTLS

# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL: load .env file if present  (pip install python-dotenv)
# ─────────────────────────────────────────────────────────────────────────────

try:
    _env_path = Path(__file__).parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
        # Re-read after load so env vars set in .env take effect
        GMAIL_SENDER = os.environ.get("GMAIL_SENDER", GMAIL_SENDER)
        GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", GMAIL_APP_PASSWORD)
        GMAIL_RECIPIENT = os.environ.get("GMAIL_RECIPIENT", GMAIL_RECIPIENT)
        ALERTS_DIR = Path(os.environ.get("ALERTS_DIR", str(ALERTS_DIR)))
except ImportError:
    pass  # dotenv not installed — that's fine


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SendConfig:
    file: str
    date: str | None
    dry_run: bool
    alerts_dir: str


# ─────────────────────────────────────────────────────────────────────────────
# FILE DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

def find_report(alerts_dir: Path, date_str: str | None = None) -> Path:
    """
    Return path to the HTML report to send.
    - If date_str (YYYYMMDD) is given, looks for report_{date_str}.html or
      any .html file in alerts_dir containing that date string.
    - Otherwise returns the most recently modified .html file in alerts_dir.
    Also checks the repo root and a 'shared_reports' dir as fallbacks.
    """
    search_dirs = [
        alerts_dir,
        Path("."),
        Path("shared_reports"),
    ]

    candidates: list[Path] = []

    for d in search_dirs:
        if not d.exists():
            continue
        for p in d.iterdir():
            if p.suffix.lower() == ".html":
                if date_str:
                    if date_str in p.name:
                        candidates.append(p)
                else:
                    candidates.append(p)

    if not candidates:
        raise FileNotFoundError(
            f"No HTML report found.\n"
            f"  Searched: {', '.join(str(d) for d in search_dirs)}\n"
            f"  Date filter: {date_str or 'none (latest)'}\n"
            f"  Run auto_pipeline.py with --shared-report-file first."
        )

    # Prefer the most recently modified
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_config() -> list[str]:
    """Return list of config problems (empty = all good)."""
    problems = []

    if GMAIL_SENDER == "your_address@gmail.com":
        problems.append("GMAIL_SENDER is still the placeholder — set your Gmail address.")
    elif "@gmail.com" not in GMAIL_SENDER:
        problems.append(f"GMAIL_SENDER '{GMAIL_SENDER}' doesn't look like a Gmail address.")

    if GMAIL_APP_PASSWORD in ("xxxx xxxx xxxx xxxx", "", None):
        problems.append(
            "GMAIL_APP_PASSWORD is not set.\n"
            "  Create one at: https://myaccount.google.com/apppasswords"
        )
    else:
        # App passwords are 16 chars (spaces are optional formatting)
        clean = GMAIL_APP_PASSWORD.replace(" ", "")
        if len(clean) != 16:
            problems.append(
                f"GMAIL_APP_PASSWORD looks wrong (expected 16 chars, got {len(clean)}). "
                "Re-copy it from Google — spaces are OK."
            )

    if GMAIL_RECIPIENT == "your_address@gmail.com" and GMAIL_SENDER == "your_address@gmail.com":
        problems.append("GMAIL_RECIPIENT is still the placeholder — set your destination address.")

    return problems


# ─────────────────────────────────────────────────────────────────────────────
# EMAIL BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_message(html_path: Path, sender: str, recipient: str) -> MIMEMultipart:
    """Build a MIME multipart message with plain-text fallback + HTML body."""
    date_str = market_today_str()
    subject = SUBJECT_TEMPLATE.format(date=date_str)

    html_content = html_path.read_text(encoding="utf-8")

    # Plain-text fallback (very minimal — the HTML is the real content)
    plain_text = (
        f"TSX Auto Entry Pipeline — Daily Report\n"
        f"Date: {date_str}\n\n"
        f"This email contains an HTML report. Please view it in an HTML-capable email client.\n\n"
        f"Report file: {html_path.name}\n"
        f"Generated on: {market_now()}\n"
    )

    msg = MIMEMultipart("alternative")
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject

    # Attach plain first, HTML second (email clients prefer last match)
    msg.attach(MIMEText(plain_text, "plain", "utf-8"))
    msg.attach(MIMEText(html_content, "html", "utf-8"))

    return msg


# ─────────────────────────────────────────────────────────────────────────────
# SEND
# ─────────────────────────────────────────────────────────────────────────────

def send_email(msg: MIMEMultipart, dry_run: bool = False) -> None:
    """Connect to Gmail SMTP and deliver the message."""
    if dry_run:
        print(f"  [dry-run] Would send '{msg['Subject']}'")
        print(f"  [dry-run]   From : {msg['From']}")
        print(f"  [dry-run]   To   : {msg['To']}")
        print(f"  [dry-run]   Size : {len(msg.as_string()):,} bytes")
        return

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(GMAIL_SENDER, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_SENDER, GMAIL_RECIPIENT, msg.as_string())

    except smtplib.SMTPAuthenticationError:
        print(
            "\n❌  Gmail authentication failed.\n"
            "   Make sure you are using an App Password, NOT your regular Gmail password.\n"
            "   Create / reset at: https://myaccount.google.com/apppasswords\n"
            "   Also confirm that 2-Step Verification is enabled on the account.",
            file=sys.stderr,
        )
        sys.exit(1)

    except smtplib.SMTPException as e:
        print(f"\n❌  SMTP error: {e}", file=sys.stderr)
        sys.exit(1)

    except (socket.timeout, OSError) as e:
        print(
            f"\n❌  Network error: {e}\n"
            f"   Check internet connectivity on the Jetson Nano.\n"
            f"   Test with: ping smtp.gmail.com",
            file=sys.stderr,
        )
        sys.exit(1)


def send_report(cfg: SendConfig):
    alerts_dir = Path(cfg.alerts_dir)

    print("─" * 55)
    print("  TSX Pipeline — Gmail Sender")
    print("─" * 55)

    # ── 1. Validate config ───────────────────────────────────────────────────
    print("\n[1/3] Checking configuration...")
    problems = validate_config()
    if problems:
        print("\n❌  Configuration errors found:\n")
        for p in problems:
            print(f"  • {p}")
        print(
            "\nEdit the CONFIG section at the top of send_report.py\n"
            "or set the corresponding environment variables / .env file.\n"
        )
        sys.exit(1)
    print(f"  Sender    : {GMAIL_SENDER}")
    print(f"  Recipient : {GMAIL_RECIPIENT}")
    print(f"  SMTP      : {SMTP_HOST}:{SMTP_PORT}")

    # ── 2. Find report file ──────────────────────────────────────────────────
    print("\n[2/3] Locating report file...")
    try:
        if cfg.file:
            report_path = Path(cfg.file)
            if not report_path.exists():
                print(f"❌  File not found: {report_path}", file=sys.stderr)
                sys.exit(1)
        else:
            report_path = find_report(alerts_dir, cfg.date)
    except FileNotFoundError as e:
        print(f"❌  {e}", file=sys.stderr)
        sys.exit(1)

    size_kb = report_path.stat().st_size / 1024
    print(f"  File : {report_path}")
    print(f"  Size : {size_kb:.1f} KB")

    # ── 3. Build & send ──────────────────────────────────────────────────────
    print(f"\n[3/3] {'[dry-run] Validating' if cfg.dry_run else 'Sending'} email...")
    msg = build_message(report_path, GMAIL_SENDER, GMAIL_RECIPIENT)
    send_email(msg, dry_run=cfg.dry_run)

    if cfg.dry_run:
        print("\n✅  Dry-run complete — no email sent.")
    else:
        print(f"\n✅  Report sent to {GMAIL_RECIPIENT}")
        print(f"    Subject : {msg['Subject']}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send the TSX pipeline HTML report via Gmail."
    )
    parser.add_argument(
        "--file", "-f",
        default="report/report.html",
        help="Path to a specific HTML report file to send.",
    )
    parser.add_argument(
        "--date", "-d", default=None,
        help="Date string (YYYYMMDD) to find the right report, e.g. 20260301",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate config and locate the file without actually sending.",
    )
    parser.add_argument(
        "--alerts-dir", default=None,
        help=f"Override the alerts directory (default: {ALERTS_DIR})",
    )
    args = parser.parse_args()

    cfg = SendConfig(
        file=args.file,
        date=args.date,
        dry_run=args.dry_run,
        alerts_dir=args.alerts_dir
    )

    send_report(cfg)


if __name__ == "__main__":
    main()
