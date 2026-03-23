"""
backtest_report.py
==================
Phase 5 — HTML report generator for BacktestResults.

Produces a single self-contained HTML file with:
  1. Header banner — period, initial capital, final equity
  2. Stat pills    — total return, max drawdown, win rate, profit factor,
                     avg hold, total trades
  3. Equity curve  — inline SVG line chart (portfolio vs XIU benchmark)
  4. Drawdown chart — inline SVG area chart
  5. Trade log      — sortable HTML table with colour-coded P&L
  6. Per-ticker stats — wins / losses / avg P&L per ticker
  7. Monthly returns heatmap — calendar grid

No external JS or CSS dependencies — renders in any email client or browser.
Reuses the colour palette from report_html.py for visual consistency.

Usage
-----
    from backtest_report import write_backtest_report
    write_backtest_report(results, "report/backtest_2023.html")

    # Or with a benchmark equity series for comparison:
    write_backtest_report(results, "report/backtest_2023.html",
                          benchmark_equity=xiu_series)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd
import numpy as np

from backtest_runner import BacktestResults

# ── reuse palette from existing report module ─────────────────────────────────
from report_html import C, FONT, FONT_MONO


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def write_backtest_report(
    results:           BacktestResults,
    path:              str,
    benchmark_equity:  Optional[pd.Series] = None,
    title:             str = "Backtest Report",
) -> None:
    """
    Write a self-contained HTML backtest report to `path`.

    Parameters
    ----------
    results          : BacktestResults from BacktestRunner.run()
    path             : output file path (created / overwritten)
    benchmark_equity : optional pd.Series (DatetimeIndex → float) of benchmark
                       daily closes for overlay on the equity chart.
                       If None, only the portfolio curve is shown.
    title            : page / email subject title
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    eq = results.equity_curve_df()
    tl = results.trade_log_df()

    html = "\n".join([
        _doc_open(title),
        _header_banner(results),
        _stat_pills(results, eq, tl),
        _equity_chart(eq, benchmark_equity),
        _drawdown_chart(eq),
        _monthly_heatmap(eq),
        _trade_table(tl),
        _per_ticker_stats(tl),
        _disclaimer(),
        _doc_close(),
    ])

    Path(path).write_text(html, encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT SKELETON  (matches report_html.py style exactly)
# ─────────────────────────────────────────────────────────────────────────────

def _doc_open(title: str) -> str:
    return (
        f"<!DOCTYPE html><html lang='en'><head>"
        f"<meta charset='UTF-8'/>"
        f"<meta name='viewport' content='width=device-width,initial-scale=1'/>"
        f"<title>{title}</title></head>"
        f"<body style='margin:0;padding:0;background:{C['page_bg']}'>"
        f"<table width='100%' cellpadding='0' cellspacing='0' border='0' "
        f"bgcolor='{C['page_bg']}' style='border-collapse:collapse;"
        f"background:{C['page_bg']}'>"
        f"<tr><td bgcolor='{C['page_bg']}' style='padding:24px 16px;"
        f"background:{C['page_bg']}'>"
        f"<table width='100%' cellpadding='0' cellspacing='0' border='0' "
        f"style='border-collapse:collapse;max-width:1000px;margin:0 auto'>"
        f"<tr><td bgcolor='{C['page_bg']}' style='background:{C['page_bg']}'>"
    )


def _doc_close() -> str:
    return (
        "<p style='margin:24px 0 0 0;padding:0;font-family:{f};font-size:10px;"
        "color:{d};text-align:center'>&#9888; Educational / research use only. "
        "Not financial advice.</p>"
        "</td></tr></table></td></tr></table></body></html>"
    ).format(f=FONT, d=C["text_dim"])


def _spacer(h: int = 12) -> str:
    return (
        f"<table width='100%' cellpadding='0' cellspacing='0' border='0'>"
        f"<tr><td height='{h}' style='font-size:1px;line-height:1px'>&nbsp;"
        f"</td></tr></table>"
    )


def _card(inner: str) -> str:
    bg = C["card_bg"]
    return (
        f"<table width='100%' cellpadding='0' cellspacing='0' border='0' "
        f"bgcolor='{bg}' style='border-collapse:collapse;background:{bg};"
        f"border:1px solid {C['border']};margin-bottom:16px'>"
        f"<tr><td bgcolor='{bg}' style='padding:20px 24px;background:{bg}'>"
        f"{inner}</td></tr></table>"
    )


def _section_title(text: str, bar_color: str = "") -> str:
    bar_color = bar_color or C["accent"]
    return (
        f"<table width='100%' cellpadding='0' cellspacing='0' border='0' "
        f"style='margin-bottom:12px'><tr>"
        f"<td width='4' bgcolor='{bar_color}' style='width:4px;padding:0;"
        f"background:{bar_color};font-size:1px'>&nbsp;</td>"
        f"<td style='padding:2px 0 2px 10px'>"
        f"<p style='margin:0;padding:0;font-family:{FONT};font-size:12px;"
        f"font-weight:bold;letter-spacing:.07em;text-transform:uppercase;"
        f"color:{bar_color}'>{text}</p>"
        f"</td></tr></table>"
    )


# ─────────────────────────────────────────────────────────────────────────────
# HEADER BANNER
# ─────────────────────────────────────────────────────────────────────────────

def _header_banner(results: BacktestResults) -> str:
    cfg = results.cfg
    eq  = results.equity_curve_df()
    end_eq = float(eq["total_equity"].iloc[-1]) if not eq.empty else cfg.initial_cash
    ret_pct = (end_eq / cfg.initial_cash - 1.0) * 100.0 if cfg.initial_cash > 0 else 0.0
    sign    = "+" if ret_pct >= 0 else ""
    col     = C["forming"] if ret_pct >= 0 else C["confirmed"]

    return (
        f"<table width='100%' cellpadding='0' cellspacing='0' border='0' "
        f"bgcolor='{C['header_bg']}' style='border-collapse:collapse;"
        f"background:{C['header_bg']};margin-bottom:16px'>"
        f"<tr><td bgcolor='{C['header_bg']}' style='padding:20px 24px;"
        f"background:{C['header_bg']}'>"
        f"<p style='margin:0 0 4px 0;font-family:{FONT};font-size:22px;"
        f"font-weight:bold;color:{C['header_text']}'>&#128202; Backtest Report</p>"
        f"<p style='margin:0;font-family:{FONT};font-size:12px;"
        f"color:{C['header_sub']}'>{cfg.start_date} &rarr; {cfg.end_date}"
        f" &nbsp;&middot;&nbsp; "
        f"Capital: ${cfg.initial_cash:,.0f}"
        f" &nbsp;&middot;&nbsp; "
        f"Return: <span style='color:{col};font-weight:bold'>"
        f"{sign}{ret_pct:.2f}%</span></p>"
        f"</td></tr></table>"
    )


# ─────────────────────────────────────────────────────────────────────────────
# STAT PILLS
# ─────────────────────────────────────────────────────────────────────────────

def _stat_pills(results: BacktestResults, eq: pd.DataFrame,
                tl: pd.DataFrame) -> str:
    cfg = results.cfg

    # Compute stats
    end_eq   = float(eq["total_equity"].iloc[-1]) if not eq.empty else cfg.initial_cash
    ret_pct  = (end_eq / cfg.initial_cash - 1.0) * 100.0 if cfg.initial_cash > 0 else 0.0

    equity   = eq["total_equity"]
    roll_max = equity.cummax()
    dd       = (equity - roll_max) / roll_max * 100
    max_dd   = float(dd.min()) if not eq.empty else 0.0

    n        = len(tl)
    win_rate = 0.0
    pf       = 0.0
    avg_hold = 0.0
    if n > 0:
        wins     = tl[tl["pnl"] > 0]
        losses   = tl[tl["pnl"] <= 0]
        win_rate = len(wins) / n * 100
        gp       = float(wins["pnl"].sum())   if not wins.empty   else 0.0
        gl       = abs(float(losses["pnl"].sum())) if not losses.empty else 0.0
        pf       = gp / gl if gl > 0 else float("inf")
        avg_hold = float(tl["holding_days"].mean())

    ret_col  = C["forming"]   if ret_pct >= 0  else C["confirmed"]
    dd_col   = C["text_muted"] if max_dd > -5   else C["confirmed"]
    wr_col   = C["forming"]   if win_rate >= 50 else C["confirmed"]
    pf_str   = f"{pf:.2f}"   if pf != float("inf") else "&infin;"

    def pill(label: str, value: str, color: str) -> str:
        return (
            f"<td bgcolor='{C['card_bg']}' style='background:{C['card_bg']};"
            f"padding:10px 16px;text-align:center;"
            f"border:1px solid {C['border']};vertical-align:middle'>"
            f"<p style='margin:0;font-family:{FONT_MONO};font-size:22px;"
            f"font-weight:bold;color:{color};line-height:1'>{value}</p>"
            f"<p style='margin:4px 0 0 0;font-family:{FONT};font-size:9px;"
            f"letter-spacing:.1em;text-transform:uppercase;"
            f"color:{C['text_muted']}'>{label}</p>"
            f"</td>"
            f"<td style='width:8px;font-size:1px'>&nbsp;</td>"
        )

    pills = "".join([
        pill("Total Return",   f"{'+'if ret_pct>=0 else ''}{ret_pct:.2f}%",   ret_col),
        pill("Max Drawdown",   f"{max_dd:.2f}%",                               dd_col),
        pill("Win Rate",       f"{win_rate:.1f}%",                             wr_col),
        pill("Profit Factor",  pf_str,                                         C["accent"]),
        pill("Avg Hold (d)",   f"{avg_hold:.1f}",                              C["text"]),
        pill("Total Trades",   str(n),                                         C["gold"]),
    ])

    return (
        _spacer(4) +
        f"<table cellpadding='0' cellspacing='0' border='0' "
        f"style='margin-bottom:16px'><tr>{pills}</tr></table>"
    )


# ─────────────────────────────────────────────────────────────────────────────
# EQUITY CURVE  (inline SVG)
# ─────────────────────────────────────────────────────────────────────────────

def _equity_chart(eq: pd.DataFrame,
                  benchmark_equity: Optional[pd.Series] = None) -> str:
    if eq.empty:
        return ""

    W, H   = 960, 260
    PAD_L  = 72
    PAD_R  = 20
    PAD_T  = 20
    PAD_B  = 36
    CW     = W - PAD_L - PAD_R
    CH     = H - PAD_T - PAD_B

    values = eq["total_equity"].values.astype(float)
    dates  = eq["date"].tolist()
    n      = len(values)

    # Normalise portfolio to 100
    base   = values[0] if values[0] != 0 else 1.0
    norm   = values / base * 100.0

    all_vals = list(norm)

    # Normalise benchmark if provided
    bench_norm = None
    if benchmark_equity is not None and len(benchmark_equity) > 0:
        bv = benchmark_equity.values.astype(float)
        bb = bv[0] if bv[0] != 0 else 1.0
        bench_norm = bv / bb * 100.0
        all_vals += list(bench_norm)

    y_min  = min(all_vals) * 0.995
    y_max  = max(all_vals) * 1.005
    y_rng  = max(y_max - y_min, 0.01)

    def x(i: int) -> float:
        return PAD_L + i / max(n - 1, 1) * CW

    def y(v: float) -> float:
        return PAD_T + CH - (v - y_min) / y_rng * CH

    def polyline(vals, color: str, width: int = 2) -> str:
        pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(vals))
        return (f"<polyline points='{pts}' fill='none' stroke='{color}' "
                f"stroke-width='{width}' stroke-linejoin='round' "
                f"stroke-linecap='round'/>")

    # Fill area under portfolio curve
    fill_pts = (
        f"{x(0):.1f},{PAD_T+CH:.1f} " +
        " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(norm)) +
        f" {x(n-1):.1f},{PAD_T+CH:.1f}"
    )

    # Y axis gridlines & labels (5 lines)
    grid_lines = ""
    y_steps = 5
    for k in range(y_steps + 1):
        yv  = y_min + k / y_steps * y_rng
        yp  = y(yv)
        grid_lines += (
            f"<line x1='{PAD_L}' y1='{yp:.1f}' x2='{W-PAD_R}' y2='{yp:.1f}' "
            f"stroke='{C['border']}' stroke-width='1'/>"
            f"<text x='{PAD_L-6}' y='{yp+4:.1f}' text-anchor='end' "
            f"font-family='{FONT_MONO}' font-size='9' "
            f"fill='{C['text_muted']}'>{yv:.0f}</text>"
        )

    # X axis labels (monthly, ~6 labels)
    x_labels = ""
    step = max(n // 6, 1)
    for i in range(0, n, step):
        d   = dates[i]
        lbl = str(d)[:7]          # YYYY-MM
        xp  = x(i)
        x_labels += (
            f"<text x='{xp:.1f}' y='{PAD_T+CH+18}' text-anchor='middle' "
            f"font-family='{FONT}' font-size='9' "
            f"fill='{C['text_muted']}'>{lbl}</text>"
        )

    # Baseline at 100
    y100 = y(100.0)
    baseline = (
        f"<line x1='{PAD_L}' y1='{y100:.1f}' x2='{W-PAD_R}' y2='{y100:.1f}' "
        f"stroke='{C['border_dark']}' stroke-width='1' stroke-dasharray='4,3'/>"
    )

    # Legend
    c_accent = C["accent"]
    c_text   = C["text"]
    c_muted  = C["text_muted"]
    legend = (
        f"<circle cx='{PAD_L+10}' cy='{PAD_T+10}' r='4' fill='{c_accent}'/>"
        f"<text x='{PAD_L+18}' y='{PAD_T+14}' font-family='{FONT}' "
        f"font-size='10' fill='{c_text}'>Portfolio</text>"
    )
    if bench_norm is not None:
        legend += (
            f"<circle cx='{PAD_L+90}' cy='{PAD_T+10}' r='4' "
            f"fill='{c_muted}'/>"
            f"<text x='{PAD_L+98}' y='{PAD_T+14}' font-family='{FONT}' "
            f"font-size='10' fill='{c_muted}'>Benchmark</text>"
        )

    svg = (
        f"<svg viewBox='0 0 {W} {H}' xmlns='http://www.w3.org/2000/svg' "
        f"style='width:100%;max-width:{W}px;display:block'>"
        f"<rect width='{W}' height='{H}' fill='{C['card_bg']}'/>"
        f"{grid_lines}"
        f"{baseline}"
        f"<polygon points='{fill_pts}' fill='{C['accent']}' opacity='0.08'/>"
        f"{polyline(norm, C['accent'], 2)}"
        + (polyline(bench_norm, C["text_muted"], 1) if bench_norm is not None else "")
        + f"{x_labels}{legend}"
        f"</svg>"
    )

    inner = _section_title("Equity Curve (indexed to 100)", C["accent"]) + svg
    return _card(inner)


# ─────────────────────────────────────────────────────────────────────────────
# DRAWDOWN CHART  (inline SVG)
# ─────────────────────────────────────────────────────────────────────────────

def _drawdown_chart(eq: pd.DataFrame) -> str:
    if eq.empty:
        return ""

    W, H   = 960, 140
    PAD_L  = 52
    PAD_R  = 20
    PAD_T  = 16
    PAD_B  = 28
    CW     = W - PAD_L - PAD_R
    CH     = H - PAD_T - PAD_B

    equity   = eq["total_equity"].values.astype(float)
    roll_max = np.maximum.accumulate(equity)
    dd       = (equity - roll_max) / np.where(roll_max != 0, roll_max, 1) * 100.0
    dates    = eq["date"].tolist()
    n        = len(dd)

    y_min  = min(float(dd.min()) * 1.05, -0.1)
    y_max  = 0.5
    y_rng  = max(y_max - y_min, 0.01)

    def x(i: int) -> float:
        return PAD_L + i / max(n - 1, 1) * CW

    def yp(v: float) -> float:
        return PAD_T + CH - (v - y_min) / y_rng * CH

    # Fill area
    zero_y = yp(0.0)
    fill_pts = (
        f"{x(0):.1f},{zero_y:.1f} " +
        " ".join(f"{x(i):.1f},{yp(v):.1f}" for i, v in enumerate(dd)) +
        f" {x(n-1):.1f},{zero_y:.1f}"
    )

    # Y gridlines
    grids = ""
    for kv in [0, y_min / 2, y_min]:
        ypp = yp(kv)
        grids += (
            f"<line x1='{PAD_L}' y1='{ypp:.1f}' x2='{W-PAD_R}' y2='{ypp:.1f}' "
            f"stroke='{C['border']}' stroke-width='1'/>"
            f"<text x='{PAD_L-4}' y='{ypp+4:.1f}' text-anchor='end' "
            f"font-family='{FONT_MONO}' font-size='9' "
            f"fill='{C['text_muted']}'>{kv:.1f}%</text>"
        )

    # X labels
    x_lbls = ""
    step = max(n // 6, 1)
    for i in range(0, n, step):
        lbl = str(dates[i])[:7]
        x_lbls += (
            f"<text x='{x(i):.1f}' y='{PAD_T+CH+16}' text-anchor='middle' "
            f"font-family='{FONT}' font-size='9' "
            f"fill='{C['text_muted']}'>{lbl}</text>"
        )

    pts = " ".join(f"{x(i):.1f},{yp(v):.1f}" for i, v in enumerate(dd))

    svg = (
        f"<svg viewBox='0 0 {W} {H}' xmlns='http://www.w3.org/2000/svg' "
        f"style='width:100%;max-width:{W}px;display:block'>"
        f"<rect width='{W}' height='{H}' fill='{C['card_bg']}'/>"
        f"{grids}"
        f"<polygon points='{fill_pts}' fill='{C['confirmed']}' opacity='0.18'/>"
        f"<polyline points='{pts}' fill='none' stroke='{C['confirmed']}' "
        f"stroke-width='1.5' stroke-linejoin='round'/>"
        f"<line x1='{PAD_L}' y1='{zero_y:.1f}' x2='{W-PAD_R}' y2='{zero_y:.1f}' "
        f"stroke='{C['border_dark']}' stroke-width='1'/>"
        f"{x_lbls}"
        f"</svg>"
    )

    inner = _section_title("Drawdown (%)", C["confirmed"]) + svg
    return _card(inner)


# ─────────────────────────────────────────────────────────────────────────────
# MONTHLY RETURNS HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

def _monthly_heatmap(eq: pd.DataFrame) -> str:
    if eq.empty:
        return ""

    eq2   = eq.copy()
    eq2["date"] = pd.to_datetime(eq2["date"])
    eq2 = eq2.set_index("date")

    # Resample to month-end equity; compute monthly returns
    monthly = eq2["total_equity"].resample("ME").last()
    rets    = monthly.pct_change().dropna() * 100.0

    if rets.empty:
        return ""

    years  = sorted(rets.index.year.unique())
    months = list(range(1, 13))
    MONTH_ABBR = ["Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec"]

    # Build grid
    def cell_color(v: float) -> str:
        if pd.isna(v):
            return C["page_bg"]
        if v > 4:
            return "#0a5e35"
        if v > 2:
            return "#12a368"
        if v > 0.5:
            return "#7dd5b0"
        if v > -0.5:
            return C["card_bg"]
        if v > -2:
            return "#f5a0ab"
        if v > -4:
            return "#e8183a"
        return "#8b0000"

    def cell_text_color(v: float) -> str:
        if pd.isna(v):
            return C["text_muted"]
        if abs(v) > 2:
            return "#ffffff"
        return C["text"]

    CW, CH_ROW = 56, 28
    PAD = 6
    c_muted = C["text_muted"]
    c_text  = C["text"]

    header_row = (
        f"<tr><td style='padding:4px 6px;font-family:{FONT};font-size:10px;"
        f"font-weight:bold;color:{c_muted}'></td>"
        + "".join(
            f"<td style='padding:4px;text-align:center;font-family:{FONT};"
            f"font-size:10px;font-weight:bold;color:{c_muted}'>"
            f"{m}</td>"
            for m in MONTH_ABBR
        )
        + f"<td style='padding:4px 6px;font-family:{FONT};font-size:10px;"
        f"font-weight:bold;color:{c_muted}'>Full Year</td></tr>"
    )

    data_rows = ""
    for yr in years:
        yr_rets = rets[rets.index.year == yr]
        yr_map  = {r.month: r for r in yr_rets.index}
        yr_vals = [yr_rets.get(yr_map[m]) if m in yr_map else float("nan")
                   for m in months]

        # Annual return from monthly
        valid   = [v for v in yr_vals if not np.isnan(v)]
        annual  = (np.prod([1 + v/100 for v in valid]) - 1) * 100 if valid else float("nan")
        ann_str = f"{annual:+.1f}%" if not np.isnan(annual) else "—"
        ann_col = C["forming"] if (not np.isnan(annual) and annual >= 0) else C["confirmed"]

        cells = "".join(
            f"<td style='padding:2px;text-align:center'>"
            f"<div style='background:{cell_color(v)};color:{cell_text_color(v)};"
            f"font-family:{FONT_MONO};font-size:10px;padding:4px 2px;"
            f"text-align:center;min-width:{CW}px'>"
            f"{'—' if np.isnan(v) else f'{v:+.1f}%'}</div></td>"
            for v in yr_vals
        )

        data_rows += (
            f"<tr><td style='padding:4px 6px;font-family:{FONT_MONO};font-size:11px;"
            f"font-weight:bold;color:{c_text}'>{yr}</td>"
            f"{cells}"
            f"<td style='padding:4px 6px;font-family:{FONT_MONO};font-size:11px;"
            f"font-weight:bold;color:{ann_col}'>{ann_str}</td></tr>"
        )

    table = (
        f"<table cellpadding='0' cellspacing='2' border='0' "
        f"style='border-collapse:separate;border-spacing:2px'>"
        f"{header_row}{data_rows}</table>"
    )

    inner = _section_title("Monthly Returns", C["gold"]) + table
    return _card(inner)


# ─────────────────────────────────────────────────────────────────────────────
# TRADE LOG TABLE
# ─────────────────────────────────────────────────────────────────────────────

def _trade_table(tl: pd.DataFrame) -> str:
    if tl.empty:
        return _card(
            _section_title("Trade Log", C["accent"]) +
            f"<p style='font-family:{FONT};font-size:12px;"
            f"color:{C['text_muted']};font-style:italic'>No closed trades.</p>"
        )

    cols = [
        ("Ticker",       "ticker"),
        ("Entry",        "entry_date"),
        ("Exit",         "sell_date"),
        ("Entry $",      "entry_price"),
        ("Exit $",       "sell_price"),
        ("Shares",       "shares"),
        ("P&L $",        "pnl"),
        ("P&L %",        "pnl_pct"),
        ("Hold (d)",     "holding_days"),
    ]

    def th(label: str) -> str:
        return (
            f"<th align='left' bgcolor='{C['page_bg']}' style='"
            f"background:{C['page_bg']};padding:7px 10px;font-family:{FONT};"
            f"font-size:10px;font-weight:bold;letter-spacing:.08em;"
            f"text-transform:uppercase;color:{C['text_muted']};"
            f"border-bottom:2px solid {C['border_dark']}'>{label}</th>"
        )

    def td(content: str, align: str = "left", bold: bool = False,
           color: str = "") -> str:
        fw  = "bold" if bold else "normal"
        col = color or C["text"]
        return (
            f"<td align='{align}' bgcolor='{C['card_bg']}' style='"
            f"background:{C['card_bg']};padding:7px 10px;"
            f"font-family:{FONT_MONO};font-size:12px;color:{col};"
            f"font-weight:{fw};border-bottom:1px solid {C['border']}'>"
            f"{content}</td>"
        )

    thead = "<thead><tr>" + "".join(th(l) for l, _ in cols) + "</tr></thead>"

    rows = ""
    for _, row in tl.sort_values("entry_date", ascending=False).iterrows():
        pnl     = float(row["pnl"])
        pnl_pct = float(row["pnl_pct"])
        col     = C["forming"] if pnl >= 0 else C["confirmed"]
        sign    = "+" if pnl >= 0 else ""
        cells = (
            td(str(row["ticker"]), bold=True) +
            td(str(row["entry_date"])[:10]) +
            td(str(row["sell_date"])[:10]) +
            td(f"${float(row['entry_price']):.4f}", "right") +
            td(f"${float(row['sell_price']):.4f}", "right") +
            td(str(int(row["shares"])), "right") +
            td(f"<span style='color:{col};font-weight:bold'>"
               f"{sign}${abs(pnl):,.2f}</span>", "right") +
            td(f"<span style='color:{col}'>{sign}{pnl_pct:.2f}%</span>",
               "right") +
            td(str(int(row["holding_days"])), "right")
        )
        rows += f"<tr>{cells}</tr>"

    table = (
        f"<table width='100%' cellpadding='0' cellspacing='0' border='0' "
        f"style='border-collapse:collapse;border:1px solid {C['border']}'>"
        f"{thead}<tbody>{rows}</tbody></table>"
    )

    inner = _section_title(f"Trade Log ({len(tl)} trades)", C["accent"]) + table
    return _card(inner)


# ─────────────────────────────────────────────────────────────────────────────
# PER-TICKER STATS
# ─────────────────────────────────────────────────────────────────────────────

def _per_ticker_stats(tl: pd.DataFrame) -> str:
    if tl.empty:
        return ""

    stats = []
    for tkr, grp in tl.groupby("ticker"):
        wins   = grp[grp["pnl"] > 0]
        losses = grp[grp["pnl"] <= 0]
        stats.append({
            "ticker":    tkr,
            "trades":    len(grp),
            "wins":      len(wins),
            "losses":    len(losses),
            "win_rate":  round(len(wins) / len(grp) * 100, 1),
            "total_pnl": round(float(grp["pnl"].sum()), 2),
            "avg_pnl":   round(float(grp["pnl"].mean()), 2),
            "avg_hold":  round(float(grp["holding_days"].mean()), 1),
        })

    stats.sort(key=lambda x: x["total_pnl"], reverse=True)

    cols = [
        ("Ticker",    "ticker"),
        ("Trades",    "trades"),
        ("Wins",      "wins"),
        ("Losses",    "losses"),
        ("Win Rate",  "win_rate"),
        ("Total P&L", "total_pnl"),
        ("Avg P&L",   "avg_pnl"),
        ("Avg Hold",  "avg_hold"),
    ]

    def th(label: str) -> str:
        return (
            f"<th align='left' bgcolor='{C['page_bg']}' style='"
            f"background:{C['page_bg']};padding:7px 10px;font-family:{FONT};"
            f"font-size:10px;font-weight:bold;letter-spacing:.08em;"
            f"text-transform:uppercase;color:{C['text_muted']};"
            f"border-bottom:2px solid {C['border_dark']}'>{label}</th>"
        )

    def td_cell(content: str, align: str = "left", color: str = "") -> str:
        col = color or C["text"]
        return (
            f"<td align='{align}' bgcolor='{C['card_bg']}' style='"
            f"background:{C['card_bg']};padding:7px 10px;"
            f"font-family:{FONT_MONO};font-size:12px;color:{col};"
            f"border-bottom:1px solid {C['border']}'>{content}</td>"
        )

    thead = "<thead><tr>" + "".join(th(l) for l, _ in cols) + "</tr></thead>"
    rows  = ""
    for s in stats:
        pnl_col = C["forming"] if s["total_pnl"] >= 0 else C["confirmed"]
        sign    = "+" if s["total_pnl"] >= 0 else ""
        asign   = "+" if s["avg_pnl"] >= 0 else ""
        rows += (
            f"<tr>"
            + td_cell(f"<strong>{s['ticker']}</strong>")
            + td_cell(str(s["trades"]), "right")
            + td_cell(str(s["wins"]),   "right", C["forming"])
            + td_cell(str(s["losses"]), "right", C["confirmed"])
            + td_cell(f"{s['win_rate']:.1f}%", "right")
            + td_cell(
                f"<span style='color:{pnl_col};font-weight:bold'>"
                f"{sign}${abs(s['total_pnl']):,.2f}</span>", "right")
            + td_cell(
                f"<span style='color:{pnl_col}'>"
                f"{asign}${abs(s['avg_pnl']):,.2f}</span>", "right")
            + td_cell(f"{s['avg_hold']:.1f}d", "right")
            + "</tr>"
        )

    table = (
        f"<table width='100%' cellpadding='0' cellspacing='0' border='0' "
        f"style='border-collapse:collapse;border:1px solid {C['border']}'>"
        f"{thead}<tbody>{rows}</tbody></table>"
    )

    inner = _section_title("Per-Ticker Statistics", C["pivot"]) + table
    return _card(inner)


# ─────────────────────────────────────────────────────────────────────────────
# DISCLAIMER
# ─────────────────────────────────────────────────────────────────────────────

def _disclaimer() -> str:
    return (
        f"<p style='margin:8px 0 0 0;font-family:{FONT};font-size:10px;"
        f"color:{C['text_dim']};text-align:center'>"
        f"&#9888; Past performance does not predict future results. "
        f"Educational / research use only. Not financial advice.</p>"
        + _spacer(16)
    )
