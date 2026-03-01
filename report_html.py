"""
report_html.py
==============
Builds a fully inline-styled HTML report suitable for email delivery.
Used by auto_pipeline.py (writes the opening + pipeline section)
and position_monitor.py (appends the positions section + closes the document).

Email-safe constraints enforced throughout:
  - All styles are inline (no <style> blocks — Gmail / Outlook strip them)
  - No external fonts or CDN resources
  - Tables use border-collapse with explicit cell padding
  - Uses web-safe monospace stack for numbers
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# PALETTE  (dark finance terminal)
# ─────────────────────────────────────────────────────────────────────────────

C = {
    "bg": "#0f1117",
    "surface": "#1a1d27",
    "surface2": "#22263a",
    "border": "#2e3250",
    "border_light": "#3d4470",

    "text": "#e8eaf0",
    "text_muted": "#7b82a8",
    "text_dim": "#4a506e",

    "confirmed": "#ff4d6d",
    "confirmed_bg": "#2a0d14",
    "confirmed_bd": "#7a1628",

    "pivot": "#f5a623",
    "pivot_bg": "#241a06",
    "pivot_bd": "#7a5210",

    "forming": "#34d399",
    "forming_bg": "#062316",
    "forming_bd": "#0f6640",

    "sell": "#ff4d6d",
    "sell_bg": "#2a0d14",
    "hold": "#34d399",
    "hold_bg": "#062316",

    "expired": "#6b7280",
    "expired_bg": "#18191e",

    "accent": "#6c8eff",
    "gold": "#f5c842",
    "white": "#ffffff",
}

FONT_BODY = "'Georgia', 'Times New Roman', serif"
FONT_MONO = "'Courier New', 'Lucida Console', monospace"
FONT_LABEL = "'Verdana', 'Geneva', sans-serif"


# ─────────────────────────────────────────────────────────────────────────────
# LOW-LEVEL STYLE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _s(**kwargs) -> str:
    """Convert keyword args to an inline style string."""
    parts = []
    for k, v in kwargs.items():
        css_prop = k.replace("_", "-")
        parts.append(f"{css_prop}:{v}")
    return "; ".join(parts)


def _td(content: str, style: str = "", align: str = "left") -> str:
    base = _s(
        padding="8px 14px",
        border_bottom=f"1px solid {C['border']}",
        font_family=FONT_MONO,
        font_size="13px",
        color=C["text"],
        white_space="nowrap",
        text_align=align,
    )
    combined = f"{base}; {style}" if style else base
    return f"<td style='{combined}'>{content}</td>"


def _th(content: str, align: str = "left") -> str:
    style = _s(
        padding="9px 14px",
        background=C["surface2"],
        color=C["text_muted"],
        font_family=FONT_LABEL,
        font_size="10px",
        font_weight="bold",
        letter_spacing="0.08em",
        text_transform="uppercase",
        border_bottom=f"2px solid {C['border_light']}",
        white_space="nowrap",
        text_align=align,
    )
    return f"<th style='{style}'>{content}</th>"


def _badge(text: str, color: str, bg: str, border: str) -> str:
    style = _s(
        display="inline-block",
        padding="2px 9px",
        border_radius="4px",
        background=bg,
        color=color,
        border=f"1px solid {border}",
        font_family=FONT_LABEL,
        font_size="10px",
        font_weight="bold",
        letter_spacing="0.07em",
        text_transform="uppercase",
    )
    return f"<span style='{style}'>{text}</span>"


# ─────────────────────────────────────────────────────────────────────────────
# STATE / STATUS HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _state_badge(state: str) -> str:
    mapping = {
        "CONFIRMED": (_badge("● CONFIRMED", C["confirmed"], C["confirmed_bg"], C["confirmed_bd"])),
        "AT_PIVOT": (_badge("◆ AT PIVOT", C["pivot"], C["pivot_bg"], C["pivot_bd"])),
        "FORMING": (_badge("○ FORMING", C["forming"], C["forming_bg"], C["forming_bd"])),
        "ACTIVE": (_badge("▶ ACTIVE", C["accent"], "#0d1635", "#2040a0")),
        "FAILED": (_badge("✕ FAILED", C["expired"], C["expired_bg"], "#3a3d50")),
        "EXPIRED": (_badge("— EXPIRED", C["expired"], C["expired_bg"], "#3a3d50")),
    }
    return mapping.get(state, _badge(state, C["text_muted"], C["surface"], C["border"]))


def _row_bg(state: str) -> str:
    return {
        "CONFIRMED": C["confirmed_bg"],
        "AT_PIVOT": C["pivot_bg"],
        "FORMING": C["forming_bg"],
        "SELL": C["sell_bg"],
        "HOLD": C["bg"],
    }.get(state, C["bg"])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION HEADINGS
# ─────────────────────────────────────────────────────────────────────────────

def _section_header(title: str, subtitle: str = "", color: str = "") -> str:
    color = color or C["accent"]
    bar_style = _s(
        display="block",
        width="3px",
        background=color,
        border_radius="2px",
        margin_right="12px",
        flex_shrink="0",
        align_self="stretch",
        min_height="24px",
    )
    title_style = _s(
        font_family=FONT_LABEL,
        font_size="13px",
        font_weight="bold",
        letter_spacing="0.08em",
        text_transform="uppercase",
        color=color,
        margin="0 0 2px 0",
    )
    sub_style = _s(
        font_family=FONT_BODY,
        font_size="12px",
        color=C["text_muted"],
        margin="0",
    )
    sub_html = f"<p style='{sub_style}'>{subtitle}</p>" if subtitle else ""
    return f"""
    <div style='{_s(display="flex", align_items="stretch", margin_bottom="12px")}'>
      <span style='{bar_style}'></span>
      <div>
        <p style='{title_style}'>{title}</p>
        {sub_html}
      </div>
    </div>"""


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE SIGNAL TABLE
# ─────────────────────────────────────────────────────────────────────────────

_PIPELINE_COLS = [
    ("Ticker", "ticker", "left"),
    ("Pattern", "pattern", "left"),
    ("State", "state", "left"),
    ("Price", "price", "right"),
    ("Entry", "entry", "right"),
    ("Stop", "stop", "right"),
    ("Risk", "risk_pct", "right"),
    ("Target 2R", "target_2R", "right"),
    ("R:R", "R:R", "right"),
    ("Shares", "shares", "right"),
    ("Days", "screener_days", "right"),
]


def _pipeline_table(alerts: List[Dict[str, Any]]) -> str:
    if not alerts:
        return ""

    header_row = "".join(_th(label, align) for label, _, align in _PIPELINE_COLS)
    thead = f"<thead><tr>{header_row}</tr></thead>"

    rows = []
    for a in alerts:
        state = str(a.get("state", ""))
        row_bg = _row_bg(state)

        cells = []
        for _label, key, align in _PIPELINE_COLS:
            raw = a.get(key, "—")
            if key == "state":
                cell = _td(_state_badge(state),
                           _s(background=row_bg, border_bottom=f"1px solid {C['border']}"),
                           align)
            elif key == "ticker":
                ticker_style = _s(
                    font_weight="bold",
                    font_size="13px",
                    color=C["white"],
                    letter_spacing="0.04em",
                )
                cell = _td(f"<span style='{ticker_style}'>{raw}</span>",
                           _s(background=row_bg, border_bottom=f"1px solid {C['border']}"),
                           align)
            elif key == "R:R":
                try:
                    rr_val = float(str(raw))
                    rr_color = C["forming"] if rr_val >= 3 else (C["pivot"] if rr_val >= 2 else C["confirmed"])
                except (ValueError, TypeError):
                    rr_color = C["text"]
                cell = _td(f"<span style='color:{rr_color};font-weight:bold'>{raw}</span>",
                           _s(background=row_bg, border_bottom=f"1px solid {C['border']}"),
                           align)
            else:
                cell = _td(str(raw) if raw is not None else "—",
                           _s(background=row_bg, border_bottom=f"1px solid {C['border']}"),
                           align)
            cells.append(cell)

        rows.append(f"<tr>{''.join(cells)}</tr>")

        # Detail row
        detail = str(a.get("detail", ""))
        if detail:
            detail_style = _s(
                padding="4px 14px 10px 14px",
                font_family=FONT_BODY,
                font_size="12px",
                color=C["text_muted"],
                background=row_bg,
                border_bottom=f"1px solid {C['border']}",
                font_style="italic",
            )
            rows.append(
                f"<tr><td colspan='{len(_PIPELINE_COLS)}' style='{detail_style}'>"
                f"↳ {detail}</td></tr>"
            )

    tbody = f"<tbody>{''.join(rows)}</tbody>"
    table_style = _s(
        width="100%",
        border_collapse="collapse",
        border="1px solid " + C["border"],
        border_radius="6px",
        overflow="hidden",
    )
    return f"<table style='{table_style}'>{thead}{tbody}</table>"


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL DB SUMMARY TABLE  (compact, full state overview)
# ─────────────────────────────────────────────────────────────────────────────

def _db_table(db_rows: List[Dict[str, Any]]) -> str:
    if not db_rows:
        return "<p style='color:#4a506e;font-style:italic;font-size:13px'>No signals in database.</p>"

    cols = [
        ("Ticker", "ticker", "left"),
        ("Pattern", "pattern", "left"),
        ("State", "state", "left"),
        ("Last Seen", "last_seen", "left"),
        ("Days", "days_in_state", "right"),
        ("Entry", "entry", "right"),
        ("Stop", "stop", "right"),
        ("Detail", "detail", "left"),
    ]

    header_row = "".join(_th(label, align) for label, _, align in cols)
    thead = f"<thead><tr>{header_row}</tr></thead>"

    rows = []
    for rec in db_rows:
        state = str(rec.get("state", ""))
        row_bg = _row_bg(state) if state in ("CONFIRMED", "AT_PIVOT", "FORMING") else C["bg"]

        cells = []
        for _label, key, align in cols:
            raw = rec.get(key, "—")
            if raw is None or str(raw) in ("nan", "NaT", "None"):
                raw = "—"
            if key == "state":
                cell = _td(_state_badge(state),
                           _s(background=row_bg, border_bottom=f"1px solid {C['border']}"), align)
            elif key == "ticker":
                cell = _td(f"<b style='color:{C['white']}'>{raw}</b>",
                           _s(background=row_bg, border_bottom=f"1px solid {C['border']}"), align)
            else:
                cell = _td(str(raw),
                           _s(background=row_bg, border_bottom=f"1px solid {C['border']}"), align)
            cells.append(cell)

        rows.append(f"<tr>{''.join(cells)}</tr>")

    tbody = f"<tbody>{''.join(rows)}</tbody>"
    table_style = _s(
        width="100%",
        border_collapse="collapse",
        border="1px solid " + C["border"],
    )
    return f"<table style='{table_style}'>{thead}{tbody}</table>"


# ─────────────────────────────────────────────────────────────────────────────
# POSITION MONITOR TABLE
# ─────────────────────────────────────────────────────────────────────────────

_POS_COLS = [
    ("Ticker", "ticker", "left"),
    ("Entry Date", "entry_date", "left"),
    ("Entry $", "entry_price", "right"),
    ("Last $", "last_close", "right"),
    ("PnL %", "pnl_%", "right"),
    ("PnL $", "pnl_$", "right"),
    ("Max PnL%", "max_pnl_%", "right"),
    ("Stop", "stop_price", "right"),
    ("ATR14", "ATR14", "right"),
    ("R×", "R_mult", "right"),
    ("Days", "tdays", "right"),
    ("Status", "status", "left"),
    ("Reason", "reason", "left"),
]


def _positions_table(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "<p style='color:#4a506e;font-style:italic;font-size:13px'>No positions found.</p>"

    header_row = "".join(_th(label, align) for label, _, align in _POS_COLS)
    thead = f"<thead><tr>{header_row}</tr></thead>"

    body_rows = []
    for rec in rows:
        status = str(rec.get("status", "HOLD")).upper()
        is_sell = status == "SELL"
        row_bg = C["sell_bg"] if is_sell else C["bg"]

        cells = []
        for _label, key, align in _POS_COLS:
            raw = rec.get(key, "—")
            if raw is None or str(raw) in ("nan", "None"):
                raw = "—"

            extra_style = _s(background=row_bg, border_bottom=f"1px solid {C['border']}")

            if key == "ticker":
                cell = _td(f"<b style='color:{C['white']}'>{raw}</b>", extra_style, align)

            elif key == "status":
                if status == "SELL":
                    badge = _badge("✕ SELL", C["sell"], C["sell_bg"], C["confirmed_bd"])
                elif status == "HOLD":
                    badge = _badge("✓ HOLD", C["forming"], C["forming_bg"], C["forming_bd"])
                else:
                    badge = _badge(status, C["text_muted"], C["surface"], C["border"])
                cell = _td(badge, extra_style, align)

            elif key == "pnl_%":
                try:
                    val = float(str(raw))
                    color = C["forming"] if val > 0 else C["confirmed"]
                    cell = _td(
                        f"<span style='color:{color};font-weight:bold'>{val:+.2f}%</span>",
                        extra_style, align)
                except (ValueError, TypeError):
                    cell = _td(str(raw), extra_style, align)

            elif key == "pnl_$":
                try:
                    val = float(str(raw).replace("$", "").replace(",", ""))
                    color = C["forming"] if val > 0 else C["confirmed"]
                    cell = _td(
                        f"<span style='color:{color}'>${val:+,.2f}</span>",
                        extra_style, align)
                except (ValueError, TypeError):
                    cell = _td(str(raw), extra_style, align)

            elif key == "R_mult":
                try:
                    val = float(str(raw))
                    color = C["forming"] if val >= 2 else (C["pivot"] if val >= 1 else C["confirmed"])
                    cell = _td(f"<span style='color:{color}'>{val:.2f}R</span>",
                               extra_style, align)
                except (ValueError, TypeError):
                    cell = _td(str(raw), extra_style, align)

            else:
                cell = _td(str(raw), extra_style, align)
            cells.append(cell)

        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    tbody = f"<tbody>{''.join(body_rows)}</tbody>"
    table_style = _s(
        width="100%",
        border_collapse="collapse",
        border="1px solid " + C["border"],
    )
    return f"<table style='{table_style}'>{thead}{tbody}</table>"


# ─────────────────────────────────────────────────────────────────────────────
# STAT PILLS  (summary numbers in the header)
# ─────────────────────────────────────────────────────────────────────────────

def _stat_pill(label: str, value: str, color: str = "") -> str:
    color = color or C["accent"]
    wrap = _s(
        display="inline-block",
        margin="4px 8px 4px 0",
        padding="6px 16px",
        background=C["surface2"],
        border=f"1px solid {C['border_light']}",
        border_radius="4px",
        text_align="center",
    )
    val_style = _s(
        display="block",
        font_family=FONT_MONO,
        font_size="20px",
        font_weight="bold",
        color=color,
        line_height="1",
    )
    lbl_style = _s(
        display="block",
        font_family=FONT_LABEL,
        font_size="9px",
        letter_spacing="0.1em",
        text_transform="uppercase",
        color=C["text_muted"],
        margin_top="4px",
    )
    return f"<span style='{wrap}'><span style='{val_style}'>{value}</span><span style='{lbl_style}'>{label}</span></span>"


# ─────────────────────────────────────────────────────────────────────────────
# DIVIDER
# ─────────────────────────────────────────────────────────────────────────────

def _divider() -> str:
    border_color = C["border"]
    return f"<hr style='border:none;border-top:1px solid {border_color};margin:28px 0;'/>"


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API — called from auto_pipeline.py
# ─────────────────────────────────────────────────────────────────────────────

def write_pipeline_report(
        path: str,
        date_str: str,
        account_size: float,
        risk_pct: float,
        n_tracked: int,
        alerts: List[Dict[str, Any]],
        db_records: List[Dict[str, Any]],
) -> None:
    """
    Write (overwrite) the shared HTML report with the pipeline section.
    Leaves the </body></html> open — position_monitor will append its section
    and call close_html_report() to seal it.
    """
    confirmed = [a for a in alerts if a.get("state") == "CONFIRMED"]
    at_pivot = [a for a in alerts if a.get("state") == "AT_PIVOT"]
    forming = [a for a in alerts if a.get("state") == "FORMING"]

    # ── outer wrapper ────────────────────────────────────────────────────────
    outer = _s(
        max_width="960px",
        margin="0 auto",
        background=C["bg"],
        color=C["text"],
        font_family=FONT_BODY,
        padding="32px 24px",
    )
    card = _s(
        background=C["surface"],
        border=f"1px solid {C['border']}",
        border_radius="8px",
        padding="24px 28px",
        margin_bottom="24px",
    )
    h1_style = _s(
        font_family=FONT_LABEL,
        font_size="22px",
        font_weight="bold",
        letter_spacing="0.05em",
        color=C["white"],
        margin="0 0 4px 0",
        text_transform="uppercase",
    )
    sub_style = _s(
        font_family=FONT_BODY,
        font_size="13px",
        color=C["text_muted"],
        margin="0 0 20px 0",
    )

    # ── stats row ────────────────────────────────────────────────────────────
    pills = (
            _stat_pill("Confirmed", str(len(confirmed)), C["confirmed"]) +
            _stat_pill("At Pivot", str(len(at_pivot)), C["pivot"]) +
            _stat_pill("Forming", str(len(forming)), C["forming"]) +
            _stat_pill("Tickers", str(n_tracked), C["accent"]) +
            _stat_pill("Account", f"${account_size:,.0f}", C["gold"]) +
            _stat_pill("Risk/Trade", f"{risk_pct}%", C["text"])
    )

    # ── alert sections ───────────────────────────────────────────────────────
    sections_html = ""

    if confirmed:
        sections_html += _section_header(
            "🔴  Confirmed — Enter Tomorrow Open",
            "Pattern breakout confirmed with volume. Act at next session open.",
            C["confirmed"]
        )
        sections_html += _pipeline_table(confirmed)
        sections_html += "<br/>"

    if at_pivot:
        sections_html += _section_header(
            "🟡  At Pivot — Place Buy-Stop",
            "Setup at trigger level. Place a buy-stop 1¢ above pivot and watch for volume.",
            C["pivot"]
        )
        sections_html += _pipeline_table(at_pivot)
        sections_html += "<br/>"

    if forming:
        sections_html += _section_header(
            "🟢  Forming — Watchlist",
            "Pattern building. No action yet — check again tomorrow.",
            C["forming"]
        )
        sections_html += _pipeline_table(forming)
        sections_html += "<br/>"

    if not alerts:
        _no_sig_style = _s(color=C["text_muted"], font_style="italic", font_size="14px")
        sections_html = (
            f"<p style='{_no_sig_style}'>"
            "No actionable signals today. Patterns still forming.</p>"
        )

    # ── signal DB section ────────────────────────────────────────────────────
    db_section = ""
    if db_records:
        db_section = f"""
        <div style='{card}'>
          {_section_header("Signal Database — Full State", f"{len(db_records)} tracked signals", C["text_muted"])}
          {_db_table(db_records)}
        </div>"""

    # ── invalidation rules ───────────────────────────────────────────────────
    rule_style = _s(
        font_family=FONT_MONO,
        font_size="12px",
        color=C["text_muted"],
        line_height="1.8",
        margin="8px 0 0 0",
    )
    rules = f"""
    <div style='{card}'>
      {_section_header("Invalidation Rules", "", C["text_dim"])}
      <p style='{rule_style}'>
        <b style='color:{C["text"]}'>VCP</b> &nbsp;&nbsp;— Close below initial stop<br/>
        <b style='color:{C["text"]}'>PB&nbsp;&nbsp;</b> &nbsp;— Close below 50 EMA on above-average volume<br/>
        <b style='color:{C["text"]}'>BASE</b> — Close back inside base after confirmed breakout
      </p>
    </div>"""

    disclaimer_style = _s(
        font_family=FONT_LABEL,
        font_size="10px",
        color=C["text_dim"],
        text_align="center",
        padding_top="8px",
        letter_spacing="0.04em",
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>TSX Pipeline Report — {date_str}</title>
</head>
<body style="margin:0;padding:20px;background:{C['bg']};">
<div style='{outer}'>

  <!-- ═══════════════════ PIPELINE SECTION ═══════════════════ -->
  <div style='{card}'>
    <h1 style='{h1_style}'>🇨🇦  TSX Auto Entry Pipeline</h1>
    <p style='{sub_style}'>Daily Report &nbsp;·&nbsp; {date_str}</p>
    <div style='margin-bottom:20px'>{pills}</div>
    {_divider()}
    {sections_html}
  </div>

  {db_section}
  {rules}
  <p style='{disclaimer_style}'>⚠ Educational / research use only. This is not financial advice.</p>

  {_divider()}
  <!-- POSITION_MONITOR_PLACEHOLDER -->
"""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with p.open("w", encoding="utf-8") as f:
        f.write(html)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API — called from position_monitor.py
# ─────────────────────────────────────────────────────────────────────────────

def append_positions_report(
        path: str,
        date_str: str,
        rows: List[Dict[str, Any]],
) -> None:
    """
    Append the position monitor section to an existing HTML report file,
    then close the HTML document.  If the file does not exist yet, a
    standalone HTML document is created instead.
    """
    sell_rows = [r for r in rows if str(r.get("status", "")).upper() == "SELL"]
    hold_rows = [r for r in rows if str(r.get("status", "")).upper() == "HOLD"]
    other_rows = [r for r in rows if str(r.get("status", "")).upper() not in ("SELL", "HOLD")]

    sorted_rows = sell_rows + hold_rows + other_rows

    card = _s(
        background=C["surface"],
        border=f"1px solid {C['border']}",
        border_radius="8px",
        padding="24px 28px",
        margin_bottom="24px",
    )
    h1_style = _s(
        font_family=FONT_LABEL,
        font_size="22px",
        font_weight="bold",
        letter_spacing="0.05em",
        color=C["white"],
        margin="0 0 4px 0",
        text_transform="uppercase",
    )
    sub_style = _s(
        font_family=FONT_BODY,
        font_size="13px",
        color=C["text_muted"],
        margin="0 0 20px 0",
    )

    pills = (
            _stat_pill("Positions", str(len(rows)), C["accent"]) +
            _stat_pill("SELL", str(len(sell_rows)), C["confirmed"]) +
            _stat_pill("HOLD", str(len(hold_rows)), C["forming"])
    )

    pnl_total: Optional[float] = None
    try:
        vals = [float(r.get("pnl_$", 0) or 0) for r in rows if r.get("pnl_$") not in (None, "—", "")]
        pnl_total = sum(vals)
    except Exception:
        pass

    if pnl_total is not None:
        pnl_color = C["forming"] if pnl_total >= 0 else C["confirmed"]
        pills += _stat_pill("Unrealised P&L", f"${pnl_total:+,.2f}", pnl_color)

    pos_section = f"""
  <!-- ═══════════════════ POSITION MONITOR ═══════════════════ -->
  <div style='{card}'>
    <h1 style='{h1_style}'>📊  Position Monitor</h1>
    <p style='{sub_style}'>Daily Report &nbsp;·&nbsp; {date_str}</p>
    <div style='margin-bottom:20px'>{pills}</div>
    {_divider()}
    {_positions_table(sorted_rows)}
  </div>

</div>
</body>
</html>
"""

    import os
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # Replace placeholder if pipeline wrote it, otherwise just append before </body>
        if "<!-- POSITION_MONITOR_PLACEHOLDER -->" in content:
            content = content.replace("<!-- POSITION_MONITOR_PLACEHOLDER -->", pos_section)
        elif "</body>" in content:
            content = content.replace("</body>", pos_section + "</body>")
        else:
            content += pos_section

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    else:
        # Standalone — no prior pipeline section
        outer = _s(
            max_width="960px",
            margin="0 auto",
            background=C["bg"],
            color=C["text"],
            font_family=FONT_BODY,
            padding="32px 24px",
        )
        standalone = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Position Monitor — {date_str}</title>
</head>
<body style="margin:0;padding:20px;background:{C['bg']};">
<div style='{outer}'>
{pos_section}
</div>
</body>
</html>
"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(standalone)
