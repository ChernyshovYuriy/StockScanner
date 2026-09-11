"""
scanner_board/store.py
========================
Persistence (SQLite) for the Ticker Indicator Board — Phase 4, see
scanner_board/PLAN.md. Same convention as edgar/store.py /
demand_signals/store.py / triple_screen_tracker/store.py: one schema
string, connect() self-migrates, plain functions — no ORM.

One row per (ticker, run_date): a daily snapshot, not an append-only
ledger — a re-run on the same day replaces that day's row (INSERT OR
REPLACE) rather than accumulating duplicates. No capital/positions table
here at all, unlike every paper sleeve's db.py-based store — this package
never touches db.py or any data/*.db used by an actual sleeve.

Column list single-source-of-truth: NUMERIC_COLUMNS/LABEL_COLUMNS below
must match scanner_board.row.compute_row()'s own dict keys exactly (tested
in tests/test_scanner_board_store.py) — the schema is generated from these
two tuples, not hand-typed a second time, so there is exactly one place
that lists every Board column.
"""
from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Optional

# In StockScanner the path comes from config.py (SCANNER_BOARD_DB_PATH);
# the fallback keeps this package runnable standalone, same as
# triple_screen_tracker/store.py's own fallback.
try:
    from config import SCANNER_BOARD_DB_PATH as DB_PATH
except Exception:
    from pathlib import Path
    DB_PATH = Path(__file__).resolve().parent.parent / "scanner_board.db"

# Every numeric (REAL) column scanner_board.row.compute_row() produces,
# beyond "ticker" (the primary key's other half) and "price"/"pct_chg".
NUMERIC_COLUMNS = (
    "price", "pct_chg",
    "ma22", "ma50", "ma200",
    "macd_line", "macd_signal", "macd_hist",
    "plus_di", "minus_di", "adx", "atr", "atr_pct",
    "rsi", "stoch_k", "stoch_d",
    "volume", "obv", "ad", "force_short", "force_long",
)

# Every labeled (TEXT) column scanner_board.row.compute_row() produces.
LABEL_COLUMNS = (
    "ma_slope", "price_vs_ma", "value_zone",
    "macd_cross", "macd_hist_slope", "trend_health", "macd_hist_extreme", "macd_hist_divergence",
    "di_bias", "adx_trend", "adx_regime",
    "rsi_zone", "rsi_divergence", "stoch_zone", "stoch_divergence",
    "volume_vs_avg", "obv_trend", "obv_divergence", "ad_trend", "ad_divergence",
    "force_short_zone", "force_short_divergence", "force_long_bias", "force_long_divergence",
    "impulse_daily", "impulse_weekly", "weekly_trend", "daily_trend", "triple_screen",
)

# The full set of columns compute_row() must return (besides "ticker").
ROW_VALUE_COLUMNS = NUMERIC_COLUMNS + LABEL_COLUMNS


def _column_defs() -> str:
    lines = ["ticker TEXT NOT NULL", "run_date TEXT NOT NULL"]
    lines += [f"{c} REAL" for c in NUMERIC_COLUMNS]
    lines += [f"{c} TEXT" for c in LABEL_COLUMNS]
    lines.append("updated_at TEXT NOT NULL")
    lines.append("PRIMARY KEY (ticker, run_date)")
    return ",\n    ".join(lines)


_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS board_snapshot (
    {_column_defs()}
);
CREATE INDEX IF NOT EXISTS idx_board_snapshot_run_date ON board_snapshot(run_date);
"""


def connect(db_path=DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    conn.row_factory = sqlite3.Row
    return conn


def upsert_rows(conn: sqlite3.Connection, run_date: str,
                 rows: List[Dict[str, Any]], updated_at: str) -> None:
    """Write every row in one transaction (a full-universe run is hundreds
    of tickers — one commit at the end, not one per ticker, the same
    rationale market_data_cache.py's own bulk sync documents)."""
    columns = ("ticker", "run_date") + ROW_VALUE_COLUMNS + ("updated_at",)
    placeholders = ",".join("?" * len(columns))
    sql = f"INSERT OR REPLACE INTO board_snapshot ({','.join(columns)}) VALUES ({placeholders})"
    conn.execute("BEGIN")
    try:
        for row in rows:
            values = [row.get("ticker"), run_date] + \
                     [row.get(c) for c in ROW_VALUE_COLUMNS] + [updated_at]
            conn.execute(sql, values)
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def latest_run_date(conn: sqlite3.Connection) -> Optional[str]:
    row = conn.execute("SELECT MAX(run_date) FROM board_snapshot").fetchone()
    return row[0] if row and row[0] is not None else None


def rows_for_run_date(conn: sqlite3.Connection, run_date: str) -> List[Dict[str, Any]]:
    cur = conn.execute(
        "SELECT * FROM board_snapshot WHERE run_date = ? ORDER BY ticker", [run_date])
    return [dict(r) for r in cur.fetchall()]


def latest_rows(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """The dashboard's own read path (Phase 5): every ticker's most recent
    snapshot. Empty list on a brand-new, never-run DB rather than an
    error."""
    run_date = latest_run_date(conn)
    if run_date is None:
        return []
    return rows_for_run_date(conn, run_date)
