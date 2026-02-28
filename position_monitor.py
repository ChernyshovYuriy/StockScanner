"""
position_monitor.py

Reads positions.csv (format: ticker,entry_date,entry_price,shares),
downloads daily OHLCV (Yahoo Finance via yfinance), computes ATR-based risk controls,
and prints a HOLD/SELL table + writes a daily log CSV.

No CLI args; just run from IDE.

Dependencies:
  pip install pandas yfinance

Notes:
- Uses daily bars.
- Exit logic (defaults tuned for ~1–2 week holds):
    * Initial stop = entry - 1.5 * ATR(14)
    * Chandelier trail = HH_since_entry - 2.5 * ATR(14)
    * Profit giveback: if max_profit >= 3% and current <= max_profit - 2% => SELL
    * Time stop: if >= 7 trading days and current profit < +0.5% => SELL
- "Stop hit" can be evaluated by today's LOW crossing stop (more realistic) or by CLOSE below stop.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("Missing dependency: yfinance. Install with: pip install yfinance", file=sys.stderr)
    raise

# -----------------------------
# Configuration (edit as needed)
# -----------------------------

# Files
POSITIONS_CSV = Path("positions/positions.csv")
DATA_DIR = Path("data_cache")  # optional local cache directory (CSV per ticker)
LOGS_DIR = Path("logs")

# Data range
ATR_PERIOD = 14
LOOKBACK_DAYS_BEFORE_ENTRY = 80  # ensures enough bars pre-entry for ATR
MIN_BARS_REQUIRED = 25

# Risk rules (weekly-ish hold)
INITIAL_STOP_ATR_K = 1.5
CHAND_TRAIL_ATR_K = 2.5

# Profit giveback rule
GIVEBACK_ACTIVATE_PCT = 3.0  # activate once max profit >= 3%
GIVEBACK_ALLOW_PCT = 2.0  # allow giveback of 2% from peak

# Time stop
TIME_STOP_DAYS = 7  # trading days since entry
TIME_STOP_MIN_PROFIT_PCT = 0.5  # require at least +0.5% by then

# Trigger mode for stop comparison:
#   "low"  -> if today's low <= stop => treat as stop hit (more realistic)
#   "close"-> if today's close < stop => stop hit (more conservative for EOD-only execution)
STOP_TRIGGER = "low"  # "low" or "close"

# Whether to cache downloaded data locally per ticker
ENABLE_CACHE = True


# -----------------------------
# Helpers
# -----------------------------

@dataclass
class Position:
    ticker: str
    entry_date: date
    entry_price: float
    shares: float


def parse_positions_csv(path: Path) -> list[Position]:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {path.resolve()}")

    df = pd.read_csv(path)
    required = {"ticker", "entry_date", "entry_price", "shares"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    positions: list[Position] = []
    for _, row in df.iterrows():
        ticker = str(row["ticker"]).strip()
        if not ticker:
            continue
        entry_date = pd.to_datetime(row["entry_date"]).date()
        entry_price = float(row["entry_price"])
        shares = float(row["shares"])
        positions.append(Position(ticker, entry_date, entry_price, shares))
    return positions


def wilder_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """
    ATR via Wilder's smoothing.
    df must have columns: High, Low, Close
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)

    # Wilder smoothing (RMA): ATR_t = (ATR_{t-1}*(n-1) + TR_t)/n
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    return atr


def trading_days_since_entry(df: pd.DataFrame, entry_dt: pd.Timestamp) -> int:
    """Count number of bars from entry date to last bar inclusive (approx trading days)."""
    if df.empty:
        return 0
    mask = df.index.normalize() >= entry_dt.normalize()
    return int(mask.sum())


def download_ohlc(ticker: str, start: date, end: Optional[date] = None) -> pd.DataFrame:
    """
    Download daily OHLCV from Yahoo via yfinance.
    Returns DataFrame indexed by DatetimeIndex (timezone-naive), columns: Open/High/Low/Close/Adj Close/Volume.
    """
    end_dt = end or (date.today() + timedelta(days=1))
    df = yf.download(
        tickers=ticker,
        start=start.isoformat(),
        end=(end_dt.isoformat()),
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if df is None or df.empty:
        return pd.DataFrame()

    # yfinance sometimes returns multiindex columns if multiple tickers; ensure flat for single ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Standardize index
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def compute_signals(pos: Position, df: pd.DataFrame) -> Dict[str, object]:
    """
    Compute stops and exit signals for a single position.
    Returns dict suitable for a summary table.
    """
    entry_dt = pd.Timestamp(pos.entry_date)

    # Keep only relevant bars; require entry date present or later data
    df = df.copy()
    df = df.dropna(subset=["High", "Low", "Close"])
    if df.empty:
        return {"ticker": pos.ticker, "status": "NO_DATA", "reason": "No OHLC data"}

    # Use only bars up to last available
    last_bar = df.iloc[-1]
    last_date = df.index[-1].date()

    # ATR
    df["ATR"] = wilder_atr(df, ATR_PERIOD)

    # Find bars from entry onwards
    after_entry = df[df.index.normalize() >= entry_dt.normalize()]
    if after_entry.empty:
        return {
            "ticker": pos.ticker,
            "status": "NO_DATA",
            "reason": f"No bars on/after entry_date ({pos.entry_date})",
        }

    # Highest high since entry for chandelier (intraday - intentional for trailing stop)
    hh_since_entry = float(after_entry["High"].max())

    # Latest ATR (use last available value; if NaN, fall back)
    atr_latest = float(df["ATR"].iloc[-1]) if pd.notna(df["ATR"].iloc[-1]) else float("nan")
    if pd.isna(atr_latest):
        # Try last non-NaN
        atr_non_nan = df["ATR"].dropna()
        atr_latest = float(atr_non_nan.iloc[-1]) if not atr_non_nan.empty else float("nan")

    if pd.isna(atr_latest):
        return {
            "ticker": pos.ticker,
            "status": "NO_ATR",
            "reason": "ATR could not be computed (insufficient history)",
        }

    # Stops
    initial_stop = pos.entry_price - INITIAL_STOP_ATR_K * atr_latest
    chandelier_stop = hh_since_entry - CHAND_TRAIL_ATR_K * atr_latest
    stop_price = max(initial_stop, chandelier_stop)

    # PnL
    last_close = float(last_bar["Close"])
    last_low = float(last_bar["Low"])
    pnl_pct = (last_close / pos.entry_price - 1.0) * 100.0

    # Max PnL since entry: use Close to avoid inflating peak with intraday wicks
    peak_price = float(after_entry["Close"].max())
    max_pnl_pct = (peak_price / pos.entry_price - 1.0) * 100.0

    # Time in trade (trading days)
    tdays = trading_days_since_entry(df, entry_dt)

    # Exit conditions
    reasons = []
    sell = False

    # Stop trigger
    if STOP_TRIGGER.lower() == "low":
        if last_low <= stop_price:
            sell = True
            reasons.append(f"STOP_HIT(low<=stop {stop_price:.2f})")
    else:
        if last_close < stop_price:
            sell = True
            reasons.append(f"STOP_HIT(close<stop {stop_price:.2f})")

    # Giveback rule
    if max_pnl_pct >= GIVEBACK_ACTIVATE_PCT:
        if pnl_pct <= (max_pnl_pct - GIVEBACK_ALLOW_PCT):
            sell = True
            reasons.append(f"GIVEBACK({max_pnl_pct:.1f}%-> {pnl_pct:.1f}%)")

    # Time stop
    if tdays >= TIME_STOP_DAYS and pnl_pct < TIME_STOP_MIN_PROFIT_PCT:
        sell = True
        reasons.append(f"TIME_STOP({tdays}d, pnl {pnl_pct:.1f}%)")

    status = "SELL" if sell else "HOLD"
    reason = "; ".join(reasons) if reasons else "OK"

    # Extra: "R" estimate (risk units) using initial stop distance
    risk_per_share = pos.entry_price - initial_stop
    r_multiple = (last_close - pos.entry_price) / risk_per_share if risk_per_share > 0 else float("nan")

    return {
        "ticker": pos.ticker,
        "entry_date": pos.entry_date.isoformat(),
        "entry_price": round(pos.entry_price, 4),
        "shares": pos.shares,
        "last_date": last_date.isoformat(),
        "last_close": round(last_close, 4),
        "pnl_%": round(pnl_pct, 2),
        "pnl_$": round((last_close - pos.entry_price) * pos.shares, 2),
        "max_pnl_%": round(max_pnl_pct, 2),
        "ATR14": round(atr_latest, 4),
        "initial_stop": round(initial_stop, 4),
        "chandelier_stop": round(chandelier_stop, 4),
        "stop_price": round(stop_price, 4),
        "tdays": tdays,
        "R_mult": (round(r_multiple, 2) if pd.notna(r_multiple) else None),
        "status": status,
        "reason": reason,
    }


def load_or_fetch_data(ticker: str, start: date) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_DIR / f"{ticker.replace('/', '_')}.csv"

    cached = pd.DataFrame()
    if ENABLE_CACHE and cache_path.exists():
        try:
            cached = pd.read_csv(cache_path, parse_dates=["Date"]).set_index("Date")
            cached.index = pd.to_datetime(cached.index)
            cached = cached.sort_index()
        except Exception:
            cached = pd.DataFrame()

    fetched = download_ohlc(ticker, start=start, end=None)

    if fetched.empty and not cached.empty:
        df = cached
    elif not cached.empty and not fetched.empty:
        df = pd.concat([cached, fetched])
        df = df[~df.index.duplicated(keep="last")]  # keep newest for same date
        df = df.sort_index()
    else:
        df = fetched

    if ENABLE_CACHE and not df.empty:
        out = df.copy()
        out.insert(0, "Date", out.index)
        out.to_csv(cache_path, index=False)

    return df


def main() -> None:
    positions = parse_positions_csv(POSITIONS_CSV)
    if not positions:
        print("No positions found in positions.csv")
        return

    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for pos in positions:
        # Download enough data for ATR + entry window
        start = (pd.Timestamp(pos.entry_date) - pd.Timedelta(days=LOOKBACK_DAYS_BEFORE_ENTRY)).date()
        df = load_or_fetch_data(pos.ticker, start=start)

        # Basic validation
        if df.empty or len(df) < MIN_BARS_REQUIRED:
            rows.append({
                "ticker": pos.ticker,
                "status": "NO_DATA",
                "reason": f"Insufficient bars ({len(df)})",
            })
            continue

        # Ensure expected columns exist
        needed_cols = {"High", "Low", "Close"}
        if not needed_cols.issubset(df.columns):
            rows.append({
                "ticker": pos.ticker,
                "status": "BAD_DATA",
                "reason": f"Missing columns: {sorted(needed_cols - set(df.columns))}",
            })
            continue

        rows.append(compute_signals(pos, df))

    out_df = pd.DataFrame(rows)

    # Sort: SELL first, then biggest pnl at top
    if "status" in out_df.columns:
        out_df["__status_rank"] = out_df["status"].map({"SELL": 0, "HOLD": 1}).fillna(9)
        if "pnl_%" in out_df.columns:
            out_df = out_df.sort_values(by=["__status_rank", "pnl_%"], ascending=[True, False])
        else:
            out_df = out_df.sort_values(by=["__status_rank"], ascending=[True])
        out_df = out_df.drop(columns=["__status_rank"])

    # Print nicely
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 50)
    print("\n=== POSITION MONITOR ===")
    print(out_df.to_string(index=False))

    # Write daily log
    today_str = date.today().isoformat()
    log_path = LOGS_DIR / f"position_monitor_{today_str}.csv"
    out_df.to_csv(log_path, index=False)
    print(f"\nSaved log: {log_path.resolve()}")


if __name__ == "__main__":
    main()
