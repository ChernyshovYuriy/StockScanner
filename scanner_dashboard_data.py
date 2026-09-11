"""
scanner_dashboard_data.py
===========================
Read-only assembly of the Ticker Indicator Board (scanner_board/) for the
web dashboard's /scanner route — Phase 5, see scanner_board/PLAN.md.

Same isolation pattern as triple_screen_tracker_dashboard_data.py: opens
its OWN read-only sqlite3 connection straight at SCANNER_BOARD_DB_PATH —
never imports scanner_board/store.py's connect() (a write-shaped
self-migrate the dashboard's long-running process shouldn't do on every
page load). Purely a display layer over whatever scanner_pipeline.py has
already computed; never fetches or computes anything itself.

Visual convention (see scanner_board/PLAN.md Phase 5): every labeled
column is rendered as a colored badge, using ONE shared word->color lookup
(_WORD_BADGE below) reused across every column rather than a different
color rule per column — the same "single shared logic" discipline the
rest of scanner_board/ was built under. The one column that can't share
that lookup is value_zone: Ch.22 says buying "Above" the value zone means
overpaying, not that anything bullish is happening — the opposite of what
"Above" means on the price_vs_ma column right next to it — so value_zone
is deliberately rendered as plain text in the template instead of forcing
a same-word badge that would silently contradict its neighbour.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from typing import Dict, List, Optional

from config import DASHBOARD_SNAPSHOT_CACHE_TTL_SECONDS, SCANNER_BOARD_DB_PATH
from scanner_board.store import LABEL_COLUMNS

_cache_lock = threading.Lock()
_cache: Dict[str, object] = {"ts": 0.0, "state": None}

# One shared label -> dashboard.css badge class lookup, reused for every
# Board column except value_zone (see module docstring). Every word here
# is an actual Enum .value from scanner_board/thesis_rules.py or
# scanner_board/triple_screen.py — see those modules for what each word
# means and which book chapter it's from. Words not listed here (there
# shouldn't be any, given every possible Enum value is enumerated) fall
# back to badge-neutral rather than raising.
_WORD_BADGE = {
    # Slope / trend direction (Ch.22-24, 28-30's various *_trend columns)
    "Rising": "badge-bullish", "Falling": "badge-bearish", "Flat": "badge-neutral",
    "Up": "badge-bullish", "Down": "badge-bearish",
    # price_vs_ma (Ch.22) only — NOT value_zone, see module docstring
    "Above": "badge-bullish", "Below": "badge-bearish", "At": "badge-neutral",
    # macd_cross / di_bias / force_long_bias (Ch.23, 24, 30)
    "Bull": "badge-bullish", "Bear": "badge-bearish", "Neutral": "badge-neutral",
    # trend_health (Ch.23's flagship thesis)
    "Safe": "badge-bullish", "Weak": "badge-bearish", "Unclear": "badge-neutral",
    # macd_hist_extreme (Ch.23)
    "New High": "badge-bullish", "New Low": "badge-bearish", "No": "badge-neutral",
    # volume_vs_avg (Ch.28)
    "High": "badge-bullish", "Low": "badge-bearish", "Normal": "badge-neutral",
    # force_short_zone (Ch.30) — raw sign only, see thesis_rules.force_short_term_zone
    "Positive": "badge-bullish", "Negative": "badge-bearish", "Zero": "badge-neutral",
    # every *_divergence column (Ch.23, 26, 27, 29, 30)
    "Bullish": "badge-bullish", "Bearish": "badge-bearish", "None": "badge-neutral",
    # rsi_zone / stoch_zone (Ch.25-27) — a caution/opportunity zone, not an
    # outright bull/bear call, so a milder tint than the categories above
    "Overbought": "badge-mild_bearish", "Oversold": "badge-mild_bullish",
    # adx_regime (Ch.24) — ADX has no directional sign of its own; Waking
    # Up/Overheated are worth calling out as "watch this" rather than
    # colored bull/bear
    "Choppy": "badge-neutral", "Trending": "badge-neutral", "Unknown": "badge-neutral",
    "Waking Up": "badge-highlight", "Overheated": "badge-highlight",
    # impulse_daily / impulse_weekly (Ch.40) — Elder's own color words
    "Green": "badge-bullish", "Red": "badge-bearish", "Blue": "badge-impulse-blue",
    # triple_screen (Ch.39)
    "Stand aside": "badge-neutral", "Go long setup": "badge-bullish", "Go short setup": "badge-bearish",
}

# value_zone is rendered as plain text (see module docstring) — excluded
# from the per-row badge dict entirely so a template bug can't accidentally
# badge it with a color that means the opposite thing on this column.
_BADGE_COLUMNS = tuple(c for c in LABEL_COLUMNS if c != "value_zone")


def badge_class(label: Optional[str]) -> str:
    if label is None:
        return "badge-neutral"
    return _WORD_BADGE.get(label, "badge-neutral")


# ▲▼– glyphs (PLAN.md's visual spec) for every column whose value is
# literally a slope/trend direction — Rising/Falling/Flat or Up/Down (Flat
# never appears on weekly_trend/daily_trend, but the dash covers it
# defensively rather than leaving a blank). Deliberately NOT applied to
# every badge column — a glyph only makes sense where "up" is a real
# direction, not e.g. Bull/Bear or Overbought/Oversold.
_SLOPE_GLYPH = {"Rising": "▲", "Falling": "▼", "Flat": "–", "Up": "▲", "Down": "▼"}
_GLYPH_COLUMNS = ("ma_slope", "macd_hist_slope", "adx_trend", "obv_trend", "ad_trend",
                   "weekly_trend", "daily_trend")


def slope_glyph(label: Optional[str]) -> str:
    if label is None:
        return ""
    return _SLOPE_GLYPH.get(label, "")


def _read_latest() -> List[dict]:
    """Every ticker's most recent snapshot. [] if the DB doesn't exist yet
    (scanner_pipeline.py hasn't run) — same "not yet available" convention
    as triple_screen_tracker_dashboard_data.py's DB-missing guard."""
    if not SCANNER_BOARD_DB_PATH.exists():
        return []
    conn = sqlite3.connect(f"file:{SCANNER_BOARD_DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        run_date_row = conn.execute("SELECT MAX(run_date) FROM board_snapshot").fetchone()
        run_date = run_date_row[0] if run_date_row else None
        if run_date is None:
            return []
        cur = conn.execute(
            "SELECT * FROM board_snapshot WHERE run_date = ? ORDER BY ticker", [run_date])
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def _build_scanner_state() -> Dict[str, object]:
    rows = _read_latest()
    for row in rows:
        row["badges"] = {col: badge_class(row.get(col)) for col in _BADGE_COLUMNS}
        row["glyphs"] = {col: slope_glyph(row.get(col)) for col in _GLYPH_COLUMNS}
    run_date = rows[0]["run_date"] if rows else None
    return {"rows": rows, "run_date": run_date}


def build_scanner_state() -> Dict[str, object]:
    """TTL-cached wrapper — same rationale as
    triple_screen_tracker_dashboard_data.build_triple_screen_tracker_state()."""
    with _cache_lock:
        now = time.monotonic()
        if _cache["state"] is not None and now - _cache["ts"] < DASHBOARD_SNAPSHOT_CACHE_TTL_SECONDS:
            return _cache["state"]

        state = _build_scanner_state()
        _cache["state"] = state
        _cache["ts"] = now
        return state
