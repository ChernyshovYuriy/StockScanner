"""
report_html.py
==============
Light-theme HTML email report — works reliably in Gmail.

Why light theme: Gmail forcibly overrides dark backgrounds with white,
making white text invisible. Light theme (dark text on white) is the only
approach that renders correctly in Gmail without workarounds.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

from schema_keys import POSITION_COL_ENTRY_DATE, POSITION_COL_ENTRY_PRICE, POSITION_COL_LAST_CLOSE, \
    POSITION_COL_PNL_DOLLARS, \
    POSITION_COL_PNL_PCT, POSITION_COL_REASON, POSITION_COL_STATUS, SIGNAL_COL_DAYS_IN_STATE, SIGNAL_COL_DETAIL, \
    SIGNAL_COL_ENTRY, SIGNAL_COL_LAST_SEEN, SIGNAL_COL_PATTERN, SIGNAL_COL_RISK_PCT, SIGNAL_COL_STATE, SIGNAL_COL_STOP, \
    SIGNAL_COL_TICKER

# ─────────────────────────────────────────────────────────────────────────────
# PALETTE  — light theme, high contrast
# ─────────────────────────────────────────────────────────────────────────────

C = {
    "page_bg": "#f4f6f9",
    "card_bg": "#ffffff",
    "border": "#dde1ea",
    "border_dark": "#b0b8cc",

    "text": "#1a1d2e",
    "text_muted": "#5a6080",
    "text_dim": "#9099b8",

    # Signal states
    "confirmed": "#c0152f",
    "confirmed_bg": "#fff0f2",
    "confirmed_border": "#f5a0ab",
    "confirmed_bar": "#e8183a",

    "pivot": "#b85c00",
    "pivot_bg": "#fff8ee",
    "pivot_border": "#f5c97a",
    "pivot_bar": "#e07b10",

    "forming": "#0a7c4e",
    "forming_bg": "#f0fdf7",
    "forming_border": "#7dd5b0",
    "forming_bar": "#12a368",

    "sell": "#c0152f",
    "sell_bg": "#fff0f2",
    "hold": "#0a7c4e",
    "hold_bg": "#f0fdf7",

    "accent": "#2f55d4",
    "gold": "#9a6c00",
    "white": "#ffffff",

    # Header band
    "header_bg": "#1a1d2e",
    "header_text": "#ffffff",
    "header_sub": "#9099b8",
}

FONT = "Arial,Helvetica,sans-serif"
FONT_MONO = "Courier New,Courier,monospace"


def c(key: str) -> str:
    """Shorthand for C[key] — avoids escaped quotes inside f-strings."""
    return C[key]


# ─────────────────────────────────────────────────────────────────────────────
# PRIMITIVES
# ─────────────────────────────────────────────────────────────────────────────

def _spacer(h: int = 12) -> str:
    return (f"<table width='100%' cellpadding='0' cellspacing='0' border='0'>"
            f"<tr><td height='{h}' style='font-size:1px;line-height:1px'>&nbsp;</td></tr>"
            f"</table>")


def _divider(color: str = "") -> str:
    color = color or C["border"]
    return (f"<table width='100%' cellpadding='0' cellspacing='0' border='0'"
            f" style='margin:16px 0'>"
            f"<tr><td height='1' bgcolor='{color}'"
            f" style='font-size:1px;line-height:1px;background:{color}'>&nbsp;</td></tr>"
            f"</table>")


def _card(inner: str, bg: str = "") -> str:
    bg = bg or C["card_bg"]
    return (
        f"<table width='100%' cellpadding='0' cellspacing='0' border='0'"
        f" bgcolor='{bg}'"
        f" style='border-collapse:collapse;background:{bg};"
        f"border:1px solid {C['border']};margin-bottom:16px'>"
        f"<tr><td bgcolor='{bg}' style='padding:20px 24px;background:{bg}'>"
        f"{inner}"
        f"</td></tr></table>"
    )


def _tinted_card(inner: str, bg: str, border: str) -> str:
    return (
        f"<table width='100%' cellpadding='0' cellspacing='0' border='0'"
        f" bgcolor='{bg}'"
        f" style='border-collapse:collapse;background:{bg};"
        f"border:1px solid {border};margin-bottom:16px'>"
        f"<tr><td bgcolor='{bg}' style='padding:16px 20px;background:{bg}'>"
        f"{inner}"
        f"</td></tr></table>"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEXT
# ─────────────────────────────────────────────────────────────────────────────

def _h1(text: str, color: str = "") -> str:
    color = color or C["header_text"]
    return (f"<p style='margin:0 0 4px 0;padding:0;font-family:{FONT};"
            f"font-size:20px;font-weight:bold;color:{color}'>{text}</p>")


def _h2(text: str, color: str = "") -> str:
    color = color or C["text"]
    return (f"<p style='margin:0 0 10px 0;padding:0;font-family:{FONT};"
            f"font-size:15px;font-weight:bold;color:{color}'>{text}</p>")


def _subline(text: str, color: str = "") -> str:
    color = color or C["header_sub"]
    return (f"<p style='margin:0 0 16px 0;padding:0;font-family:{FONT};"
            f"font-size:12px;color:{color}'>{text}</p>")


def _label(text: str, color: str = "") -> str:
    color = color or C["text_muted"]
    return (f"<span style='font-family:{FONT};font-size:10px;"
            f"font-weight:bold;letter-spacing:0.08em;"
            f"text-transform:uppercase;color:{color}'>{text}</span>")


# ─────────────────────────────────────────────────────────────────────────────
# BADGE
# ─────────────────────────────────────────────────────────────────────────────

def _badge(text: str, color: str, bg: str, border: str) -> str:
    return (f"<span style='padding:2px 8px;background:{bg};color:{color};"
            f"border:1px solid {border};font-family:{FONT};font-size:10px;"
            f"font-weight:bold;letter-spacing:0.06em;text-transform:uppercase'>"
            f"{text}</span>")


def _state_badge(state: str) -> str:
    m = {
        "CONFIRMED": _badge("&#9679; CONFIRMED", C["confirmed"], C["confirmed_bg"], C["confirmed_border"]),
        "AT_PIVOT": _badge("&#9670; AT PIVOT", C["pivot"], C["pivot_bg"], C["pivot_border"]),
        "FORMING": _badge("&#9675; FORMING", C["forming"], C["forming_bg"], C["forming_border"]),
        "ACTIVE": _badge("&#9654; ACTIVE", C["accent"], "#eef2ff", "#b0bfff"),
        "FAILED": _badge("&#10005; FAILED", "#666", "#f5f5f5", "#ccc"),
        "EXPIRED": _badge("&mdash; EXPIRED", "#999", "#f9f9f9", "#ddd"),
    }
    return m.get(state, _badge(state, C["text_muted"], "#f5f5f5", "#ccc"))


def _status_badge(status: str) -> str:
    status = status.upper()
    if status == "SELL":
        return _badge("&#10005; SELL", C["sell"], C["sell_bg"], C["confirmed_border"])
    if status == "HOLD":
        return _badge("&#10003; HOLD", C["hold"], C["hold_bg"], C["forming_border"])
    return _badge(status, "#666", "#f5f5f5", "#ccc")


def _row_bg(state: str) -> str:
    return {
        "CONFIRMED": C["confirmed_bg"],
        "AT_PIVOT": C["pivot_bg"],
        "FORMING": C["forming_bg"],
        "SELL": C["sell_bg"],
    }.get(state, C["card_bg"])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION HEADER with left color bar
# ─────────────────────────────────────────────────────────────────────────────

def _section_header(title: str, subtitle: str = "", bar_color: str = "") -> str:
    bar_color = bar_color or C["accent"]
    sub_html = ""
    if subtitle:
        sub_html = (f"<p style='margin:2px 0 0 0;padding:0;font-family:{FONT};"
                    f"font-size:11px;color:{c("text_muted")}'>{subtitle}</p>")
    return (
        f"<table width='100%' cellpadding='0' cellspacing='0' border='0'"
        f" style='margin-bottom:12px'>"
        f"<tr>"
        f"<td width='4' bgcolor='{bar_color}'"
        f" style='width:4px;padding:0;background:{bar_color};font-size:1px'>&nbsp;</td>"
        f"<td style='padding:2px 0 2px 10px;background:{c("card_bg")}'>"
        f"<p style='margin:0;padding:0;font-family:{FONT};font-size:12px;"
        f"font-weight:bold;letter-spacing:0.07em;text-transform:uppercase;"
        f"color:{bar_color}'>{title}</p>"
        f"{sub_html}"
        f"</td></tr></table>"
    )


# ─────────────────────────────────────────────────────────────────────────────
# STAT PILLS
# ─────────────────────────────────────────────────────────────────────────────

def _stat_pill(label: str, value: str, color: str = "") -> str:
    """Single pill as two <td> elements (pill + spacer). Wrap in <table><tr>."""
    color = color or C["accent"]
    return (
        f"<td bgcolor='{c("card_bg")}'"
        f" style='background:{c("card_bg")};padding:10px 16px;"
        f"text-align:center;border:1px solid {c("border")};vertical-align:middle'>"
        f"<p style='margin:0;padding:0;font-family:{FONT_MONO};"
        f"font-size:22px;font-weight:bold;color:{color};line-height:1'>{value}</p>"
        f"<p style='margin:4px 0 0 0;padding:0;font-family:{FONT};font-size:9px;"
        f"letter-spacing:0.1em;text-transform:uppercase;color:{c("text_muted")}'>"
        f"{label}</p>"
        f"</td>"
        f"<td style='padding:0;width:8px;font-size:1px'>&nbsp;</td>"
    )


def _pills_row(*args: Tuple[str, str, str]) -> str:
    cells = "".join(_stat_pill(lbl, val, col) for lbl, val, col in args)
    return (
        f"<table cellpadding='0' cellspacing='0' border='0'"
        f" style='margin-bottom:16px'>"
        f"<tr>{cells}</tr></table>"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TABLE CELLS
# ─────────────────────────────────────────────────────────────────────────────

def _th(text: str, align: str = "left") -> str:
    return (
        f"<th align='{align}' bgcolor='{c("page_bg")}'"
        f" style='background:{c("page_bg")};padding:7px 12px;"
        f"font-family:{FONT};font-size:10px;font-weight:bold;"
        f"letter-spacing:0.08em;text-transform:uppercase;color:{c("text_muted")};"
        f"white-space:nowrap;border-bottom:2px solid {c("border_dark")};"
        f"text-align:{align}'>{text}</th>"
    )


def _td(content: str, bg: str = "", align: str = "left",
        bold: bool = False, color: str = "") -> str:
    bg = bg or C["card_bg"]
    color = color or C["text"]
    fw = "bold" if bold else "normal"
    return (
        f"<td bgcolor='{bg}' align='{align}'"
        f" style='background:{bg};padding:8px 12px;"
        f"font-family:{FONT_MONO};font-size:12px;color:{color};"
        f"font-weight:{fw};white-space:nowrap;"
        f"border-bottom:1px solid {c("border")};text-align:{align}'>"
        f"{content}</td>"
    )


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE SIGNAL TABLE
# ─────────────────────────────────────────────────────────────────────────────

_PIPELINE_COLS = [
    ("Ticker", SIGNAL_COL_TICKER, "left"),
    ("Pattern", SIGNAL_COL_PATTERN, "left"),
    ("State", SIGNAL_COL_STATE, "left"),
    ("Price", "price", "right"),
    ("Entry", SIGNAL_COL_ENTRY, "right"),
    ("Stop", SIGNAL_COL_STOP, "right"),
    ("Risk", SIGNAL_COL_RISK_PCT, "right"),
    ("Target 2R", "target_2R", "right"),
    ("R:R", "R:R", "right"),
    ("Shares", "shares", "right"),
    ("Days", "screener_days", "right"),
]


def _pipeline_table(alerts: List[Dict[str, Any]]) -> str:
    if not alerts:
        return ""
    thead = ("<thead><tr>" +
             "".join(_th(lbl, al) for lbl, _, al in _PIPELINE_COLS) +
             "</tr></thead>")
    rows = []
    for a in alerts:
        state = str(a.get(SIGNAL_COL_STATE, ""))
        bg = _row_bg(state)
        cells = []
        for _lbl, key, al in _PIPELINE_COLS:
            raw = str(a.get(key) or "&mdash;")
            if key == SIGNAL_COL_STATE:
                cells.append(_td(_state_badge(state), bg, al))
            elif key == SIGNAL_COL_TICKER:
                cells.append(_td(raw, bg, al, bold=True))
            elif key == "R:R":
                try:
                    v = float(raw)
                    col = C["forming"] if v >= 3 else (C["pivot"] if v >= 2 else C["confirmed"])
                except (ValueError, TypeError):
                    col = C["text"]
                cells.append(_td(
                    f"<span style='color:{col};font-weight:bold'>{raw}</span>",
                    bg, al))
            else:
                cells.append(_td(raw, bg, al))
        rows.append(f"<tr>{''.join(cells)}</tr>")

        detail = str(a.get(SIGNAL_COL_DETAIL, ""))
        if detail:
            n = len(_PIPELINE_COLS)
            rows.append(
                f"<tr><td colspan='{n}' bgcolor='{bg}'"
                f" style='background:{bg};padding:2px 12px 8px 12px;"
                f"font-family:{FONT};font-size:11px;color:{c("text_muted")};"
                f"font-style:italic;border-bottom:1px solid {c("border")};"
                f"white-space:normal'>&#8627; {detail}</td></tr>"
            )

    return (f"<table width='100%' cellpadding='0' cellspacing='0' border='0'"
            f" style='border-collapse:collapse;border:1px solid {c("border")}'>"
            f"{thead}<tbody>{''.join(rows)}</tbody></table>")


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL DB TABLE
# ─────────────────────────────────────────────────────────────────────────────

_DB_COLS = [
    ("Ticker", SIGNAL_COL_TICKER, "left"),
    ("Pattern", SIGNAL_COL_PATTERN, "left"),
    ("State", SIGNAL_COL_STATE, "left"),
    ("Last Seen", SIGNAL_COL_LAST_SEEN, "left"),
    ("Days", SIGNAL_COL_DAYS_IN_STATE, "right"),
    ("Entry", SIGNAL_COL_ENTRY, "right"),
    ("Stop", SIGNAL_COL_STOP, "right"),
    ("Detail", SIGNAL_COL_DETAIL, "left"),
]


def _db_table(db_rows: List[Dict[str, Any]]) -> str:
    if not db_rows:
        return (f"<p style='font-family:{FONT};font-size:12px;"
                f"color:{c("text_muted")};font-style:italic'>No signals in database.</p>")
    thead = ("<thead><tr>" +
             "".join(_th(lbl, al) for lbl, _, al in _DB_COLS) +
             "</tr></thead>")
    rows = []
    for rec in db_rows:
        state = str(rec.get(SIGNAL_COL_STATE, ""))
        bg = _row_bg(state) if state in ("CONFIRMED", "AT_PIVOT", "FORMING") else C["card_bg"]
        cells = []
        for _lbl, key, al in _DB_COLS:
            raw = rec.get(key)
            if raw is None or str(raw) in ("nan", "NaT", "None"):
                raw = "&mdash;"
            raw = str(raw)
            if key == SIGNAL_COL_STATE:
                cells.append(_td(_state_badge(state), bg, al))
            elif key == SIGNAL_COL_TICKER:
                cells.append(_td(raw, bg, al, bold=True))
            else:
                cells.append(_td(raw, bg, al))
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return (f"<table width='100%' cellpadding='0' cellspacing='0' border='0'"
            f" style='border-collapse:collapse;border:1px solid {c("border")}'>"
            f"{thead}<tbody>{''.join(rows)}</tbody></table>")


# ─────────────────────────────────────────────────────────────────────────────
# POSITION TABLE
# ─────────────────────────────────────────────────────────────────────────────

_POS_COLS = [
    ("Ticker", SIGNAL_COL_TICKER, "left"),
    ("Entry Date", POSITION_COL_ENTRY_DATE, "left"),
    ("Entry $", POSITION_COL_ENTRY_PRICE, "right"),
    ("Last $", POSITION_COL_LAST_CLOSE, "right"),
    ("PnL %", POSITION_COL_PNL_PCT, "right"),
    ("PnL $", POSITION_COL_PNL_DOLLARS, "right"),
    ("Max PnL%", "max_pnl_%", "right"),
    ("Stop", "stop_price", "right"),
    ("ATR14", "ATR14", "right"),
    ("Rx", "R_mult", "right"),
    ("Days", "tdays", "right"),
    ("Status", POSITION_COL_STATUS, "left"),
    ("Reason", POSITION_COL_REASON, "left"),
]


def _positions_table(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return (f"<p style='font-family:{FONT};font-size:12px;"
                f"color:{c("text_muted")};font-style:italic'>No positions found.</p>")
    thead = ("<thead><tr>" +
             "".join(_th(lbl, al) for lbl, _, al in _POS_COLS) +
             "</tr></thead>")
    body_rows = []
    for rec in rows:
        status = str(rec.get(POSITION_COL_STATUS, "HOLD")).upper()
        bg = C["sell_bg"] if status == "SELL" else C["card_bg"]
        cells = []
        for _lbl, key, al in _POS_COLS:
            raw = rec.get(key)
            if raw is None or str(raw) in ("nan", "None"):
                raw = "&mdash;"
            if key == SIGNAL_COL_TICKER:
                cells.append(_td(str(raw), bg, al, bold=True))
            elif key == POSITION_COL_STATUS:
                cells.append(_td(_status_badge(status), bg, al))
            elif key == POSITION_COL_PNL_PCT:
                try:
                    v = float(str(raw))
                    col = C["forming"] if v > 0 else C["confirmed"]
                    cells.append(_td(
                        f"<span style='color:{col};font-weight:bold'>{v:+.2f}%</span>",
                        bg, al))
                except (ValueError, TypeError):
                    cells.append(_td(str(raw), bg, al))
            elif key == POSITION_COL_PNL_DOLLARS:
                try:
                    v = float(str(raw).replace("$", "").replace(",", ""))
                    col = C["forming"] if v > 0 else C["confirmed"]
                    cells.append(_td(
                        f"<span style='color:{col}'>${v:+,.2f}</span>",
                        bg, al))
                except (ValueError, TypeError):
                    cells.append(_td(str(raw), bg, al))
            elif key == "R_mult":
                try:
                    v = float(str(raw))
                    col = C["forming"] if v >= 2 else (C["pivot"] if v >= 1 else C["confirmed"])
                    cells.append(_td(
                        f"<span style='color:{col}'>{v:.2f}R</span>",
                        bg, al))
                except (ValueError, TypeError):
                    cells.append(_td(str(raw), bg, al))
            else:
                cells.append(_td(str(raw), bg, al))
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    return (f"<table width='100%' cellpadding='0' cellspacing='0' border='0'"
            f" style='border-collapse:collapse;border:1px solid {c("border")}'>"
            f"{thead}<tbody>{''.join(body_rows)}</tbody></table>")


# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT SKELETON
# ─────────────────────────────────────────────────────────────────────────────

def _doc_open(title: str) -> str:
    return (
        f"<!DOCTYPE html>\n<html lang='en'>\n<head>\n"
        f"  <meta charset='UTF-8'/>\n"
        f"  <meta name='viewport' content='width=device-width,initial-scale=1'/>\n"
        f"  <title>{title}</title>\n</head>\n"
        f"<body style='margin:0;padding:0;background:{c("page_bg")}'>\n"
        f"<table width='100%' cellpadding='0' cellspacing='0' border='0'"
        f" bgcolor='{c("page_bg")}'"
        f" style='border-collapse:collapse;background:{c("page_bg")}'>\n"
        f"<tr><td bgcolor='{c("page_bg")}' style='padding:24px 16px;background:{c("page_bg")}'>\n"
        f"<table width='100%' cellpadding='0' cellspacing='0' border='0'"
        f" style='border-collapse:collapse;max-width:960px;margin:0 auto'>\n"
        f"<tr><td bgcolor='{c("page_bg")}' style='background:{c("page_bg")}'>\n"
    )


def _doc_close() -> str:
    return "\n</td></tr></table>\n</td></tr></table>\n</body>\n</html>"


def _header_banner(title: str, subtitle: str) -> str:
    """Dark header band — this is the ONLY dark element; text is explicitly white."""
    return (
        f"<table width='100%' cellpadding='0' cellspacing='0' border='0'"
        f" bgcolor='{c("header_bg")}'"
        f" style='border-collapse:collapse;background:{c("header_bg")};margin-bottom:16px'>"
        f"<tr><td bgcolor='{c("header_bg")}'"
        f" style='padding:20px 24px;background:{c("header_bg")}'>"
        f"<p style='margin:0 0 4px 0;padding:0;font-family:{FONT};"
        f"font-size:22px;font-weight:bold;color:{c("header_text")}'>{title}</p>"
        f"<p style='margin:0;padding:0;font-family:{FONT};"
        f"font-size:12px;color:{c("header_sub")}'>{subtitle}</p>"
        f"</td></tr></table>"
    )


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
    confirmed = [a for a in alerts if a.get("state") == "CONFIRMED"]
    at_pivot = [a for a in alerts if a.get("state") == "AT_PIVOT"]
    forming = [a for a in alerts if a.get("state") == "FORMING"]

    pills = _pills_row(
        ("Confirmed", str(len(confirmed)), C["confirmed"]),
        ("At Pivot", str(len(at_pivot)), C["pivot"]),
        ("Forming", str(len(forming)), C["forming"]),
        ("Tickers", str(n_tracked), C["accent"]),
        ("Account", f"${account_size:,.0f}", C["gold"]),
        ("Risk/Trade", f"{risk_pct}%", C["text"]),
    )

    sections = ""

    if confirmed:
        inner = (
                _section_header("Confirmed &mdash; Enter Tomorrow Open",
                                "Pattern breakout confirmed with volume. Act at next session open.",
                                C["confirmed_bar"]) +
                _pipeline_table(confirmed)
        )
        sections += _tinted_card(inner, C["confirmed_bg"], C["confirmed_border"])
        sections += _spacer(8)

    if at_pivot:
        inner = (
                _section_header("At Pivot &mdash; Place Buy-Stop",
                                "Setup at trigger level. Place a buy-stop 1 cent above pivot; watch for volume.",
                                C["pivot_bar"]) +
                _pipeline_table(at_pivot)
        )
        sections += _tinted_card(inner, C["pivot_bg"], C["pivot_border"])
        sections += _spacer(8)

    if forming:
        inner = (
                _section_header("Forming &mdash; Watchlist",
                                "Pattern building. No action yet &mdash; check again tomorrow.",
                                C["forming_bar"]) +
                _pipeline_table(forming)
        )
        sections += _tinted_card(inner, C["forming_bg"], C["forming_border"])
        sections += _spacer(8)

    if not alerts:
        sections = _card(
            f"<p style='font-family:{FONT};font-size:14px;"
            f"color:{c("text_muted")};font-style:italic;margin:0'>"
            f"No actionable signals today. Patterns still forming.</p>"
        )

    # Signal DB
    db_section = ""
    if db_records:
        db_inner = (
                _h2("Signal Database &mdash; Full State") +
                _db_table(db_records)
        )
        db_section = _card(db_inner)

    # Invalidation rules
    rules_rows = ""
    for name, desc in [
        ("VCP", "Close below initial stop"),
        ("PB", "Close below 50 EMA on above-average volume"),
        ("BASE", "Close back inside base after confirmed breakout"),
    ]:
        rules_rows += (
            f"<tr>"
            f"<td style='padding:4px 14px 4px 0;font-family:{FONT_MONO};"
            f"font-size:12px;color:{c("text")};font-weight:bold;white-space:nowrap;"
            f"vertical-align:top;background:{c("card_bg")}'>{name}</td>"
            f"<td style='padding:4px 0;font-family:{FONT};font-size:12px;"
            f"color:{c("text_muted")};background:{c("card_bg")}'>&mdash; {desc}</td>"
            f"</tr>"
        )
    rules_card = _card(
        _h2("Invalidation Rules") +
        f"<table cellpadding='0' cellspacing='0' border='0'"
        f" style='border-collapse:collapse'>{rules_rows}</table>"
    )

    disclaimer = (
        f"<p style='margin:0;padding:8px 0 0 0;font-family:{FONT};font-size:10px;"
        f"color:{c("text_dim")};text-align:center'>"
        f"&#9888; Educational / research use only. This is not financial advice.</p>"
    )

    body = (
            _header_banner(
                "&#127464;&#127462; TSX Auto Entry Pipeline",
                f"Daily Report &nbsp;&middot;&nbsp; {date_str}"
            ) +
            _card(pills) +
            sections +
            db_section +
            rules_card +
            disclaimer +
            _spacer(20) +
            "<!-- POSITION_MONITOR_PLACEHOLDER -->"
    )

    html = _doc_open(f"TSX Pipeline Report &mdash; {date_str}") + body

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API — called from position_monitor.py
# ─────────────────────────────────────────────────────────────────────────────

def append_positions_report(
        path: str,
        date_str: str,
        rows: List[Dict[str, Any]],
) -> None:
    sell_rows = [r for r in rows if str(r.get("status", "")).upper() == "SELL"]
    hold_rows = [r for r in rows if str(r.get("status", "")).upper() == "HOLD"]
    other_rows = [r for r in rows if str(r.get("status", "")).upper() not in ("SELL", "HOLD")]
    sorted_rows = sell_rows + hold_rows + other_rows

    pnl_pill: tuple = ()
    try:
        total = sum(float(r.get("pnl_$", 0) or 0)
                    for r in rows if r.get("pnl_$") not in (None, "&mdash;", ""))
        col = C["forming"] if total >= 0 else C["confirmed"]
        pnl_pill = (("Unrealised P&L", f"${total:+,.2f}", col),)
    except Exception:
        pass

    pills = _pills_row(
        ("Positions", str(len(rows)), C["accent"]),
        ("SELL", str(len(sell_rows)), C["confirmed"]),
        ("HOLD", str(len(hold_rows)), C["forming"]),
        *pnl_pill,
    )

    pos_section = (
            _header_banner(
                "&#128202; Position Monitor",
                f"Daily Report &nbsp;&middot;&nbsp; {date_str}"
            ) +
            _card(pills) +
            _card(_positions_table(sorted_rows))
    )

    # Each day's block ends with a marker so the funds-state card can be
    # injected into the right place by _write_position_report.
    pos_block = _spacer(8) + pos_section + "<!-- MONITOR_DAY_END -->"

    # We use a stable anchor <!-- MONITOR_SECTION_TOP --> so that each new day
    # is prepended (inserted right after the anchor) instead of appended.
    # This keeps the newest report at the top of the email.
    SECTION_ANCHOR = "<!-- MONITOR_SECTION_TOP -->"
    PLACEHOLDER = "<!-- POSITION_MONITOR_PLACEHOLDER -->"

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        if PLACEHOLDER in content:
            # First monitor run: replace the pipeline placeholder and plant anchor.
            content = content.replace(
                PLACEHOLDER,
                SECTION_ANCHOR + pos_block + _doc_close()
            )
        elif SECTION_ANCHOR in content:
            # Subsequent runs: prepend before the previous day's block.
            content = content.replace(
                SECTION_ANCHOR,
                SECTION_ANCHOR + pos_block
            )
        elif "</body>" in content:
            content = content.replace("</body>", pos_block + "</body>")
        else:
            content += pos_block + _doc_close()
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    else:
        html = (
                _doc_open(f"Position Monitor &mdash; {date_str}") +
                SECTION_ANCHOR + pos_block +
                _doc_close()
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
