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
# TRANSACTION EMAIL — buy / sell activity notification
# ─────────────────────────────────────────────────────────────────────────────

def build_transaction_html(
    buys: list,
    sells: list,
    cash_before: float,
    cash_after: float,
    open_positions_count: int,
) -> str:
    """
    Build a Gmail-compatible HTML email for trade activity.

    buys  : list of buy-record dicts with keys: ticker, shares, entry_price, pattern
    sells : list of sell-record dicts with keys: ticker, shares, entry_price,
            sell_price, proceeds, pnl_$, pnl_%, reason
    """
    from time_utils import market_today_str
    date_str = market_today_str()

    # ── Palette (matches report_html.py light theme) ──────────────────────────
    TEXT       = "#1a1d2e"
    MUTED      = "#5a6080"
    BORDER     = "#dde1ea"
    PAGE_BG    = "#f4f6f9"
    WHITE      = "#ffffff"
    GREEN      = "#0a7c4e"
    GREEN_BG   = "#f0fdf7"
    RED        = "#c0152f"
    RED_BG     = "#fff0f2"
    HEADER_BG  = "#1a1d2e"
    FONT       = "Arial,Helvetica,sans-serif"

    # ── Header label ──────────────────────────────────────────────────────────
    if buys and sells:
        icon, subtitle = "🔄", f"{len(buys)} bought · {len(sells)} sold"
    elif buys:
        n = len(buys)
        icon, subtitle = "🟢", f"{n} position{'s' if n != 1 else ''} opened"
    else:
        n = len(sells)
        icon, subtitle = "🔴", f"{n} position{'s' if n != 1 else ''} closed"

    # ── Summary card ──────────────────────────────────────────────────────────
    cash_delta  = cash_after - cash_before
    delta_sign  = "+" if cash_delta >= 0 else ""
    delta_color = GREEN if cash_delta >= 0 else RED

    extra_rows = ""
    if buys:
        total_cost = sum(float(r.get("shares", 0)) * float(r.get("entry_price", 0)) for r in buys)
        extra_rows += f"<li><b>Total invested</b>: ${total_cost:,.2f}</li>"
    if sells:
        total_proceeds = sum(float(r.get("proceeds", 0)) for r in sells)
        total_pnl      = sum(float(r.get("pnl_$", 0))   for r in sells)
        pnl_color = GREEN if total_pnl >= 0 else RED
        pnl_sign  = "+" if total_pnl >= 0 else ""
        extra_rows += (
            f"<li><b>Total proceeds</b>: ${total_proceeds:,.2f}</li>"
            f"<li><b>Realised P&amp;L</b>: "
            f"<span style='color:{pnl_color};font-weight:bold'>"
            f"{pnl_sign}${abs(total_pnl):,.2f}</span></li>"
        )

    summary = f"""
    <table width="100%" cellpadding="0" cellspacing="0" border="0"
     style="border-collapse:collapse;background:{WHITE};border:1px solid {BORDER};margin-bottom:16px">
    <tr><td style="padding:20px 24px;font-family:{FONT};color:{TEXT}">
      <div style="font-size:12px;font-weight:bold;color:{MUTED};margin-bottom:10px;
                  text-transform:uppercase;letter-spacing:.5px">Summary</div>
      <ul style="margin:0;padding-left:18px;line-height:1.9;font-size:14px">
        <li><b>Cash before</b>: ${cash_before:,.2f}</li>
        <li><b>Cash after</b>&nbsp;: ${cash_after:,.2f}
          &nbsp;<span style="color:{delta_color};font-weight:bold">
            ({delta_sign}${abs(cash_delta):,.2f})</span></li>
        {extra_rows}
        <li><b>Open positions</b>: {open_positions_count}</li>
      </ul>
    </td></tr></table>"""

    # ── Buys table ────────────────────────────────────────────────────────────
    TH = (f"padding:8px 12px;font-family:{FONT};font-size:11px;font-weight:bold;"
          f"text-transform:uppercase;color:{MUTED}")
    buys_section = ""
    if buys:
        rows = ""
        for i, r in enumerate(buys):
            bg      = WHITE if i % 2 == 0 else PAGE_BG
            ticker  = r.get("ticker", "")
            shares  = float(r.get("shares", 0))
            price   = float(r.get("entry_price", 0))
            cost    = round(shares * price, 2)
            pattern = r.get("pattern") or "—"
            rows += f"""
            <tr style="background:{bg}">
              <td style="padding:10px 12px;font-family:{FONT};font-size:13px;
                         font-weight:bold;color:{TEXT}">{ticker}</td>
              <td style="padding:10px 12px;font-family:{FONT};font-size:13px;
                         color:{TEXT};text-align:right">{shares:.0f}</td>
              <td style="padding:10px 12px;font-family:{FONT};font-size:13px;
                         color:{TEXT};text-align:right">${price:.4f}</td>
              <td style="padding:10px 12px;font-family:{FONT};font-size:13px;
                         color:{TEXT};text-align:right">${cost:,.2f}</td>
              <td style="padding:10px 12px;font-family:{FONT};font-size:13px;
                         color:{MUTED}">{pattern}</td>
            </tr>"""
        buys_section = f"""
        <table width="100%" cellpadding="0" cellspacing="0" border="0"
         style="border-collapse:collapse;background:{WHITE};border:1px solid {BORDER};margin-bottom:16px">
          <tr style="background:{GREEN_BG}">
            <td colspan="5" style="padding:12px 16px;font-family:{FONT};
                                   font-size:14px;font-weight:bold;color:{GREEN}">
              🟢 Positions Opened</td></tr>
          <tr style="background:{PAGE_BG}">
            <th style="{TH};text-align:left">Ticker</th>
            <th style="{TH};text-align:right">Shares</th>
            <th style="{TH};text-align:right">Price</th>
            <th style="{TH};text-align:right">Cost</th>
            <th style="{TH};text-align:left">Pattern</th>
          </tr>
          {rows}
        </table>"""

    # ── Sells table ───────────────────────────────────────────────────────────
    sells_section = ""
    if sells:
        rows = ""
        for i, r in enumerate(sells):
            bg          = WHITE if i % 2 == 0 else PAGE_BG
            ticker      = r.get("ticker", "")
            shares      = float(r.get("shares", 0))
            entry_price = float(r.get("entry_price", 0))
            sell_price  = float(r.get("sell_price", 0))
            pnl_dollars = float(r.get("pnl_$", 0))
            pnl_pct     = float(r.get("pnl_%", 0))
            reason      = r.get("reason", "")
            pc          = GREEN if pnl_dollars >= 0 else RED
            ps          = "+" if pnl_dollars >= 0 else "-"
            rows += f"""
            <tr style="background:{bg}">
              <td style="padding:10px 12px;font-family:{FONT};font-size:13px;
                         font-weight:bold;color:{TEXT}">{ticker}</td>
              <td style="padding:10px 12px;font-family:{FONT};font-size:13px;
                         color:{TEXT};text-align:right">{shares:.0f}</td>
              <td style="padding:10px 12px;font-family:{FONT};font-size:13px;
                         color:{TEXT};text-align:right">${entry_price:.4f}</td>
              <td style="padding:10px 12px;font-family:{FONT};font-size:13px;
                         color:{TEXT};text-align:right">${sell_price:.4f}</td>
              <td style="padding:10px 12px;font-family:{FONT};font-size:13px;
                         font-weight:bold;color:{pc};text-align:right">{ps}${abs(pnl_dollars):,.2f}</td>
              <td style="padding:10px 12px;font-family:{FONT};font-size:13px;
                         color:{pc};text-align:right">{ps}{abs(pnl_pct):.2f}%</td>
              <td style="padding:10px 12px;font-family:{FONT};font-size:12px;
                         color:{MUTED}">{reason}</td>
            </tr>"""
        sells_section = f"""
        <table width="100%" cellpadding="0" cellspacing="0" border="0"
         style="border-collapse:collapse;background:{WHITE};border:1px solid {BORDER};margin-bottom:16px">
          <tr style="background:{RED_BG}">
            <td colspan="7" style="padding:12px 16px;font-family:{FONT};
                                   font-size:14px;font-weight:bold;color:{RED}">
              🔴 Positions Closed</td></tr>
          <tr style="background:{PAGE_BG}">
            <th style="{TH};text-align:left">Ticker</th>
            <th style="{TH};text-align:right">Shares</th>
            <th style="{TH};text-align:right">Entry</th>
            <th style="{TH};text-align:right">Sell</th>
            <th style="{TH};text-align:right">P&amp;L $</th>
            <th style="{TH};text-align:right">P&amp;L %</th>
            <th style="{TH};text-align:left">Reason</th>
          </tr>
          {rows}
        </table>"""

    # ── Full document ─────────────────────────────────────────────────────────
    return f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:{PAGE_BG}">
<table width="100%" cellpadding="0" cellspacing="0" border="0" style="background:{PAGE_BG}">
<tr><td align="center" style="padding:20px 8px">
<table width="600" cellpadding="0" cellspacing="0" border="0" style="max-width:600px;width:100%">

  <tr><td style="background:{HEADER_BG};padding:20px 24px;border-radius:4px 4px 0 0">
    <table width="100%" cellpadding="0" cellspacing="0" border="0">
    <tr>
      <td style="font-family:{FONT};font-size:18px;font-weight:bold;color:#ffffff">
        {icon} Trade Activity</td>
      <td align="right" style="font-family:{FONT};font-size:12px;color:#9099b8">
        {date_str}</td>
    </tr>
    <tr><td colspan="2" style="font-family:{FONT};font-size:13px;color:#9099b8;padding-top:4px">
        {subtitle}</td></tr>
    </table>
  </td></tr>

  <tr><td style="background:{PAGE_BG};padding:16px 0">
    {summary}
    {buys_section}
    {sells_section}
  </td></tr>

  <tr><td style="padding:8px 0 16px;text-align:center;
                 font-family:{FONT};font-size:11px;color:#9099b8">
    Virtual transactions only — not financial advice
  </td></tr>

</table>
</td></tr></table>
</body>
</html>"""


def send_transaction_email(
    buys: list,
    sells: list,
    cash_before: float,
    cash_after: float,
    open_positions_count: int = 0,
    label: str = "TSX",
) -> None:
    """
    Send a trade activity notification email.

    Silently skips if Gmail is not configured (so unconfigured systems
    don't crash on every trade).  Always call after DB writes are committed —
    never in dry_run paths.

    label: subject-line prefix, default "TSX" reproduces the exact historical
    subject text. The momentum sleeve (momentum_buy.py / momentum_monitor.py)
    passes label="Momentum" so its emails are never confused with the core
    sleeve's.
    """
    if not buys and not sells:
        return

    problems = validate_config()
    if problems:
        print("  [transaction email] skipped — Gmail not configured.")
        return

    from time_utils import market_today_str
    date_str = market_today_str()

    if buys and sells:
        subject = f"🔄 {label} — {len(buys)} bought, {len(sells)} sold — {date_str}"
    elif buys:
        n = len(buys)
        subject = f"🟢 {label} — {n} position{'s' if n != 1 else ''} opened — {date_str}"
    else:
        n = len(sells)
        subject = f"🔴 {label} — {n} position{'s' if n != 1 else ''} closed — {date_str}"

    html_content = build_transaction_html(buys, sells, cash_before, cash_after, open_positions_count)

    plain = (
        f"TSX Trade Activity — {date_str}\n"
        f"Cash: ${cash_before:,.2f} → ${cash_after:,.2f}\n"
        f"Positions opened : {len(buys)}\n"
        f"Positions closed : {len(sells)}\n"
    )

    msg = MIMEMultipart("alternative")
    msg["From"]    = GMAIL_SENDER
    msg["To"]      = GMAIL_RECIPIENT
    msg["Subject"] = subject
    msg.attach(MIMEText(plain, "plain", "utf-8"))
    msg.attach(MIMEText(html_content, "html", "utf-8"))

    send_email(msg)
    print(f"  ✓ Transaction email sent → {GMAIL_RECIPIENT}")


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
            f"   Check internet connectivity on the host running this service.\n"
            f"   Test with: ping smtp.gmail.com",
            file=sys.stderr,
        )
        sys.exit(1)


def send_text_email(subject: str, body: str) -> bool:
    """
    Send a plain-text email through the shared Gmail transport.

    Used by the EDGAR collector service for its plain-text digest — it reuses the
    same SMTP connection, credentials, and error handling as the HTML reports
    rather than a fresh smtplib implementation.  Returns False (without raising)
    when Gmail is not configured, so an unconfigured host degrades to
    "logged, not sent" instead of crashing.
    """
    problems = validate_config()
    if problems:
        print("  [text email] skipped — Gmail not configured.")
        return False

    msg = MIMEMultipart("alternative")
    msg["From"] = GMAIL_SENDER
    msg["To"] = GMAIL_RECIPIENT
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

    send_email(msg)
    return True


def send_report(cfg: SendConfig):
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
            alerts_dir = Path(cfg.alerts_dir)
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
