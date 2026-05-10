"""
Automated Entry Pipeline — TSX Swing Trading
=============================================
Reads daily screener CSV outputs, tracks tickers across days,
detects entry patterns, and alerts on signal state transitions.

Directory layout (auto-created on first run):
  <base_dir>/
    screener_outputs/     ← drop your daily top-N CSVs here
    signal_db/
      signal_history.csv  ← persistent signal state across days (auto-managed)
    alerts/
      alerts_YYYYMMDD.csv ← actionable alerts for each run
      report_YYYYMMDD.txt ← human-readable daily report

CSV naming convention (flexible — any of these work):
  top10_tsx_20260218_1430.csv   ← from canadian_stock_screener.py
  cad_swing_candidates.json     ← from swing_tickers.py  (JSON also supported)
  my_tickers_2026-02-18.csv     ← any CSV with a "Ticker" or "symbol" column

Signal State Machine:
  FORMING  →  AT_PIVOT  →  CONFIRMED  →  ACTIVE / FAILED

Alert priority:
  🔴 URGENT   — state jumped to CONFIRMED today (enter tomorrow open)
  🟡 WATCH    — state reached AT_PIVOT today (place buy-stop)
  🟢 FORMING  — pattern building, check again tomorrow
  ⚫ EXPIRED  — pattern failed / invalidated

Run:
  python auto_pipeline.py                  # single run
  python auto_pipeline.py --schedule       # runs daily at market close (4pm ET)
  python auto_pipeline.py --ticker CNQ.TO  # debug single ticker
"""

import argparse
import json
import re
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from colorama import Fore, Style, init
from tabulate import tabulate

from config import ALERTS_PATH, FUNDS_PATH, SCREENER_OUT_PATH, OUT_PATH
from funds import read_funds
from report_html import write_pipeline_report
from schema_keys import SIGNAL_DB_COLS, SIGNAL_COL_ALERT_SENT, SIGNAL_COL_CONSECUTIVE_SCREENER_DAYS, SIGNAL_COL_DETAIL, \
    SIGNAL_COL_DAYS_IN_STATE, SIGNAL_COL_ENTRY, SIGNAL_COL_FIRST_SEEN, SIGNAL_COL_LAST_SEEN, SIGNAL_COL_PATTERN, \
    SIGNAL_COL_PIVOT_PRICE, SIGNAL_COL_RISK_PCT, SIGNAL_COL_SCREENER_DAYS, SIGNAL_COL_STATE, SIGNAL_COL_STOP, \
    SIGNAL_COL_TARGET_2R, SIGNAL_COL_TARGET_3R, SIGNAL_COL_TICKER, INTENT_COL_ALERT_STATE, INTENT_COL_CREATED_AT, \
    INTENT_COL_ENTRY_PRICE_PLANNED, INTENT_COL_PRIORITY, INTENT_COL_REASON, INTENT_COL_RR, INTENT_COL_SIGNAL_DATE, \
    INTENT_COL_STATUS, INTENT_COL_STOP_PRICE, INTENT_COL_TARGET_PRICE, INTENT_REQUIRED_COLS
from time_utils import market_today, date_to_iso_extended, date_to_iso_basic, market_now

warnings.filterwarnings("ignore")
init(autoreset=True)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    # Directory layout
    base_dir: str = "."  # root — all sub-dirs created here
    screener_subdir: str = SCREENER_OUT_PATH  # drop daily CSVs here
    db_subdir: str = OUT_PATH / "signal_db"  # signal history lives here
    alerts_subdir: str = ALERTS_PATH  # daily alert output

    # Ticker promotion rules
    min_days_in_screener: int = 1  # days a ticker must appear before being tracked
    lookback_csv_days: int = 10  # how many past CSVs to consider for persistence
    max_tracked_tickers: int = 40  # cap to avoid excessive API calls

    # Entry detection params
    account_size: float = 0.0  # set from data/funds at runtime
    risk_per_trade_pct: float = 1.0
    atr_period: int = 14
    atr_stop_mult: float = 1.5
    max_stop_pct: float = 7.0   # cap stop distance; prevents tiny positions on wide stops
    price_data_days: int = 400  # history window for pattern detection

    # Alert filtering
    min_rr: float = 2.0  # minimum risk:reward to include in alerts
    alert_on_forming: bool = True  # include FORMING signals (lower priority)

    # Market regime filter — block new buys when TSX benchmark is below its
    # 200-day SMA.  Uses XIU.TO as the TSX proxy.  Enabled by default so the
    # pipeline does not generate buy signals into a downtrending market.
    regime_filter: bool = True
    regime_benchmark: str = "XIU.TO"
    regime_sma_period: int = 200

    # Scheduling
    schedule_time: str = "16:30"  # HH:MM ET — after TSX close

    # Reporting
    shared_report_path: Optional[str] = None  # optional single-file report output

    # Buy candidates
    candidates_queue_path: Optional[str] = None  # optional single-file for buy candidates


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL STATES  (ordered — higher index = more advanced)
# ─────────────────────────────────────────────────────────────────────────────

STATE_FORMING = "FORMING"
STATE_AT_PIVOT = "AT_PIVOT"
STATE_CONFIRMED = "CONFIRMED"
STATE_ACTIVE = "ACTIVE"  # entered trade
STATE_FAILED = "FAILED"  # pattern invalidated
STATE_EXPIRED = "EXPIRED"  # disappeared from screener / stale

STATE_ORDER = [STATE_FORMING, STATE_AT_PIVOT, STATE_CONFIRMED,
               STATE_ACTIVE, STATE_FAILED, STATE_EXPIRED]

ALERT_EMOJI = {
    STATE_CONFIRMED: "🔴",
    STATE_AT_PIVOT: "🟡",
    STATE_FORMING: "🟢",
    STATE_ACTIVE: "🔵",
    STATE_FAILED: "⚫",
    STATE_EXPIRED: "⚫",
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — technical indicators (self-contained, no external dep)
# ─────────────────────────────────────────────────────────────────────────────

def _ema(s, n):   return s.ewm(span=n, adjust=False).mean()


def _sma(s, n):   return s.rolling(n).mean()


def _atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    # FIX: use Wilder's smoothing (alpha=1/period) — NOT ewm(span=period).
    # span=14 gives alpha≈0.133; Wilder's uses alpha=1/14≈0.071 (half the decay
    # speed). Using span produced ATR values ~30-50% higher than TradingView /
    # TC2000, causing over-wide stops and undersized positions.
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def _slope(series, lookback=10):
    y = series.dropna().values
    if len(y) < lookback:
        return 0.0
    y = y[-lookback:]
    x = np.arange(len(y), dtype=float)
    denom = ((x - x.mean()) ** 2).sum()
    if denom == 0:
        return 0.0
    sl = float(((x - x.mean()) * (y - y.mean())).sum() / denom)
    return sl / (y.mean() + 1e-9)  # normalized


def _rolling_vol_pct(close, n):
    return close.pct_change().rolling(n).std() * 100


# ─────────────────────────────────────────────────────────────────────────────
# CSV SCANNER — reads screener output directory
# ─────────────────────────────────────────────────────────────────────────────

def _extract_date_from_filename(path: Path) -> Optional[datetime]:
    """Try to parse YYYYMMDD or YYYY-MM-DD from filename."""
    stem = path.stem
    # Try YYYYMMDD
    m = re.search(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})", stem)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    return None


def _read_screener_file(path: Path) -> List[str]:
    """
    Read tickers from a screener output file.
    Supports: CSV with 'Ticker' or 'symbol' column, JSON array of records.
    Returns list of ticker strings.
    """
    tickers = []
    try:
        if path.suffix.lower() == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for rec in data:
                    for col in ("symbol", "Ticker", "ticker"):
                        if col in rec and rec[col]:
                            tickers.append(str(rec[col]).strip().upper())
                            break
        else:
            df = pd.read_csv(path)
            for col in ("Ticker", "symbol", "ticker", "TICKER", "Symbol"):
                if col in df.columns:
                    tickers = df[col].dropna().astype(str).str.strip().str.upper().tolist()
                    break
    except Exception as e:
        print(f"  {Fore.RED}Could not read {path.name}: {e}{Style.RESET_ALL}")
    return [t for t in tickers if t and t != "NAN"]


def scan_screener_dir(screener_dir: Path,
                      lookback_days: int = 10) -> Dict[datetime, List[str]]:
    """
    Scan directory for screener output files.
    Returns {date: [tickers]} for the last `lookback_days` files.
    """
    cutoff = datetime.today() - timedelta(days=lookback_days + 5)
    results: Dict[datetime, List[str]] = {}

    files = sorted(
        [p for p in screener_dir.iterdir()
         if p.suffix.lower() in (".csv", ".json")],
        key=lambda p: p.stat().st_mtime
    )

    for path in files:
        date = _extract_date_from_filename(path)
        if date is None:
            # Fall back to file modification time
            date = datetime.fromtimestamp(path.stat().st_mtime).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        if date < cutoff:
            continue
        tickers = _read_screener_file(path)
        if tickers:
            # If multiple files on same date, merge
            if date in results:
                results[date] = list(dict.fromkeys(results[date] + tickers))
            else:
                results[date] = tickers
            print(f"  {Fore.CYAN}Read{Style.RESET_ALL} {path.name} "
                  f"→ {len(tickers)} tickers ({date_to_iso_extended(date)})")

    return results


def build_ticker_persistence(history: Dict[datetime, List[str]],
                             min_days: int,
                             max_tickers: int) -> List[Tuple[str, int]]:
    """
    Count how many days each ticker appeared in screener output.
    Returns [(ticker, day_count)] sorted by count desc, capped at max_tickers.

    Tickers appearing on MORE recent days are weighted higher:
    weight = 1.0 + 0.2 * (day_index / total_days)  so recent days matter more.
    """
    sorted_dates = sorted(history.keys())
    n = len(sorted_dates)
    counts: Dict[str, float] = {}

    for i, date in enumerate(sorted_dates):
        weight = 1.0 + 0.2 * (i / max(n - 1, 1))  # 1.0 → 1.2 for most recent
        for ticker in history[date]:
            counts[ticker] = counts.get(ticker, 0) + weight

    # Convert to integer-equivalent day count for filtering
    raw_counts: Dict[str, int] = {}
    for ticker, weighted in counts.items():
        raw_counts[ticker] = sum(
            1 for tickers in history.values() if ticker in tickers
        )

    qualified = [
        (t, raw_counts[t]) for t in counts
        if raw_counts[t] >= min_days
    ]
    # Sort by weighted score (not raw count) for ranking
    qualified.sort(key=lambda x: counts[x[0]], reverse=True)
    return qualified[:max_tickers]


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL DB — persistent state across daily runs
# ─────────────────────────────────────────────────────────────────────────────

def load_signal_db(db_path: Path) -> pd.DataFrame:
    if db_path.exists():
        try:
            db = pd.read_csv(db_path)
            for col in (SIGNAL_COL_FIRST_SEEN, SIGNAL_COL_LAST_SEEN):
                if col in db.columns:
                    # Keep seen columns as calendar dates (no time/tz semantics).
                    # This avoids tz-aware/naive mixing while matching the business
                    # meaning of these fields: "seen on day X".
                    day_text = db[col].astype(str).str.extract(r"(\d{4}-\d{2}-\d{2})", expand=False)
                    db[col] = pd.to_datetime(day_text, errors="coerce").dt.date
            return db
        except Exception:
            pass
    return pd.DataFrame(columns=SIGNAL_DB_COLS)


def save_signal_db(db: pd.DataFrame, db_path: Path) -> None:
    db.to_csv(db_path, index=False)


def expire_missing_tickers(db: pd.DataFrame,
                           active_tickers: List[str],
                           today) -> pd.DataFrame:
    """
    Mark tickers that have disappeared from screener as EXPIRED
    if they haven't been seen for > 2 days and aren't ACTIVE trades.
    """
    if db.empty:
        return db
    db = db.copy()
    # TODO: Review this hack with date:
    today_date = today.date() if isinstance(today, datetime) else today
    for idx, row in db.iterrows():
        if row[SIGNAL_COL_TICKER] not in active_tickers and row[SIGNAL_COL_STATE] not in (STATE_ACTIVE, STATE_EXPIRED, STATE_FAILED):
            last = row[SIGNAL_COL_LAST_SEEN]
            if pd.isna(last):
                continue
            # TODO: Review this hack with date:
            last_date = last.date() if isinstance(last, datetime) else last
            gap = (today_date - last_date).days
            if gap > 2:
                db.at[idx, SIGNAL_COL_STATE] = STATE_EXPIRED
                db.at[idx, SIGNAL_COL_DETAIL] = f"Left screener after {gap}d"
    return db


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY PATTERN DETECTORS  (adapted from entry_detector.py)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_vcp(close, high, low, volume) -> Optional[Dict]:
    ma150 = _sma(close, 150)
    if pd.isna(ma150.iloc[-1]) or close.iloc[-1] < ma150.iloc[-1]:
        return None

    window = 60
    h = high.iloc[-window:]
    l = low.iloc[-window:]

    peaks = [
        (i, h.iloc[i]) for i in range(2, len(h) - 2)
        if h.iloc[i] >= h.iloc[i - 1] and h.iloc[i] >= h.iloc[i + 1]
           and h.iloc[i] >= h.iloc[i - 2] and h.iloc[i] >= h.iloc[i + 2]
    ]
    troughs = [
        (i, l.iloc[i]) for i in range(2, len(l) - 2)
        if l.iloc[i] <= l.iloc[i - 1] and l.iloc[i] <= l.iloc[i + 1]
           and l.iloc[i] <= l.iloc[i - 2] and l.iloc[i] <= l.iloc[i + 2]
    ]

    if len(peaks) < 2 or not troughs:
        return None

    contractions = []
    for pk_idx, pk_val in peaks:
        after = [(ti, tv) for ti, tv in troughs if ti > pk_idx]
        if not after:
            continue
        tr_val = after[0][1]
        contractions.append((pk_val, tr_val, (pk_val - tr_val) / pk_val * 100))

    if len(contractions) < 2:
        return None

    contracting = all(
        contractions[i][2] < contractions[i - 1][2] * 0.85
        for i in range(1, len(contractions))
    )
    if not contracting:
        return None

    cur_vol = _rolling_vol_pct(close, 10).iloc[-1]
    if cur_vol > 4.0:
        return None

    pivot = peaks[-1][1]
    last_close = close.iloc[-1]
    avg_vol = volume.iloc[-50:].mean()
    vol_ratio = volume.iloc[-1] / (avg_vol + 1e-9)

    if last_close > pivot and vol_ratio >= 1.5:
        state = STATE_CONFIRMED
        detail = f"VCP breakout confirmed — vol {vol_ratio:.1f}x avg"
    elif last_close > pivot:
        state = STATE_AT_PIVOT
        detail = f"VCP breakout above ${pivot:.2f} — low vol ({vol_ratio:.1f}x), wait"
    elif last_close >= pivot * 0.98:
        state = STATE_AT_PIVOT
        detail = f"VCP at pivot ${pivot:.2f} — place buy-stop 1¢ above"
    else:
        gap = (pivot - last_close) / last_close * 100
        state = STATE_FORMING
        detail = f"VCP forming — {len(contractions)} contractions, {gap:.1f}% to pivot"

    return {"pattern": "VCP", "pivot": round(pivot, 2),
            "state": state, "detail": detail}


def _detect_ema_pullback(close, high, low, volume) -> List[Dict]:
    ma150 = _sma(close, 150)
    if pd.isna(ma150.iloc[-1]) or close.iloc[-1] < ma150.iloc[-1]:
        return []

    results = []
    last = close.iloc[-1]
    prev = close.iloc[-2]

    for label, e in [("EMA21", _ema(close, 21)), ("EMA50", _ema(close, 50))]:
        ev = e.iloc[-1]
        if _slope(e.iloc[-10:]) <= 0:
            continue
        touched = low.iloc[-3:].min() <= ev * 1.01
        reclaiming = last > ev and prev <= ev * 1.005
        avg_vol = volume.iloc[-20:].mean()
        vol_ok = volume.iloc[-1] >= avg_vol * 0.8

        if not touched:
            continue

        if reclaiming and vol_ok:
            state = STATE_CONFIRMED
            detail = f"PB reclaim {label} at ${ev:.2f}"
        else:
            state = STATE_AT_PIVOT
            detail = f"Touching {label} (${ev:.2f}) — wait for close above"

        results.append({"pattern": f"PB-{label}", "pivot": round(ev, 2),
                        "state": state, "detail": detail})
    return results


def _detect_base_breakout(close, high, low, volume) -> Optional[Dict]:
    if len(close) < 90:
        return None
    BASE_BARS = 40
    base_high = high.iloc[-BASE_BARS:-1].max()
    base_low = low.iloc[-BASE_BARS:-1].min()
    base_range = (base_high - base_low) / base_low * 100
    if base_range > 20:
        return None

    last = close.iloc[-1]
    avg_vol = volume.iloc[-50:].mean()
    vol_ratio = volume.iloc[-1] / (avg_vol + 1e-9)

    if last > base_high and vol_ratio >= 1.5:
        state = STATE_CONFIRMED
        detail = f"Base breakout above ${base_high:.2f} — vol {vol_ratio:.1f}x"
    elif last > base_high:
        state = STATE_AT_PIVOT
        detail = f"Breakout above ${base_high:.2f} — vol weak ({vol_ratio:.1f}x)"
    elif last >= base_high * 0.98:
        state = STATE_AT_PIVOT
        detail = f"At base top ${base_high:.2f} — watch for vol surge"
    else:
        gap = (base_high - last) / last * 100
        state = STATE_FORMING
        detail = f"In base (range {base_range:.1f}%) — {gap:.1f}% to breakout"

    return {"pattern": "BASE", "pivot": round(base_high, 2),
            "state": state, "detail": detail}


def detect_all_patterns(ticker: str, df: pd.DataFrame) -> List[Dict]:
    """Run all 3 detectors, return list of pattern dicts sorted by state priority."""
    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()
    volume = df["Volume"].squeeze()

    patterns = []
    v = _detect_vcp(close, high, low, volume)
    if v:   patterns.append(v)
    patterns.extend(_detect_ema_pullback(close, high, low, volume))
    b = _detect_base_breakout(close, high, low, volume)
    if b:   patterns.append(b)

    priority = {STATE_CONFIRMED: 0, STATE_AT_PIVOT: 1,
                STATE_FORMING: 2, STATE_FAILED: 3}
    patterns.sort(key=lambda x: priority.get(x["state"], 9))
    return patterns


# ─────────────────────────────────────────────────────────────────────────────
# RISK / SIZING
# ─────────────────────────────────────────────────────────────────────────────

def _find_resistance(high, entry: float, min_rr_distance: float = 2.0,
                     risk: float = 0.0, lookback: int = 252) -> Optional[float]:
    """
    Return the nearest swing-high *above* entry within the last `lookback` bars,
    but only if it is at least `min_rr_distance * risk` above entry.
    Resistance closer than that would produce R:R < min_rr_distance, so we
    ignore it and let the caller fall back to an ATR-based target instead.
    Falls back to None if no qualifying resistance is found.
    """
    h = high.iloc[-lookback:]
    min_target = entry + min_rr_distance * risk if risk > 0 else entry
    candidates = []
    for i in range(2, len(h) - 2):
        v = h.iloc[i]
        if v > min_target and v >= h.iloc[i - 1] and v >= h.iloc[i + 1] \
                and v >= h.iloc[i - 2] and v >= h.iloc[i + 2]:
            candidates.append(v)
    return float(min(candidates)) if candidates else None


def compute_levels(close, high, low, entry: float,
                   atr_period: int, atr_mult: float,
                   max_stop_pct: float = 7.0) -> Dict:
    atr_val = _atr(high, low, close, atr_period).iloc[-1]
    atr_stop = entry - atr_mult * atr_val
    swing_low = low.iloc[-20:].min()
    raw_stop = min(atr_stop, swing_low * 0.99)

    # Cap stop distance so wide stops don't shrink positions to insignificance
    min_stop = entry * (1 - max_stop_pct / 100)
    stop = max(raw_stop, min_stop)

    risk = max(entry - stop, 0.01)

    # Only use resistance as target if it is at least 2R above entry,
    # otherwise fall back to ATR projection (keeps R:R >= 2.0 by default).
    resistance = _find_resistance(high, entry, min_rr_distance=2.0, risk=risk)
    if resistance is not None:
        target_2r = resistance  # real supply ceiling at least 2R away
        target_3r = entry + 3 * risk  # ATR projection beyond resistance
    else:
        target_2r = entry + 2 * risk  # ATR-only fallback
        target_3r = entry + 3 * risk

    return {
        "entry": round(entry, 2),
        "stop": round(stop, 2),
        "risk_pct": round(risk / entry * 100, 2),
        "target_2r": round(target_2r, 2),
        "target_3r": round(target_3r, 2),
        "atr": round(atr_val, 2),
        "resistance_based": resistance is not None,  # flag for transparency
    }


def compute_position_size(account: float, risk_pct: float,
                          entry: float, stop: float) -> Dict:
    dollar_risk = account * (risk_pct / 100)
    per_share_risk = max(entry - stop, 0.01)
    shares = int(dollar_risk / per_share_risk)
    return {
        "shares": shares,
        "position_$": round(shares * entry, 2),
        "risk_$": round(dollar_risk, 2),
        "acct_pct": round(shares * entry / account * 100, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# STATE TRANSITION LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def state_transition_label(old: str, new: str) -> str:
    """Describe what changed between yesterday and today."""
    if old == new:
        return f"held {new}"
    if old == STATE_FORMING and new == STATE_AT_PIVOT:
        return "🟡 ADVANCING: FORMING → AT_PIVOT"
    if old == STATE_AT_PIVOT and new == STATE_CONFIRMED:
        return "🔴 BREAKOUT: AT_PIVOT → CONFIRMED"
    if old == STATE_FORMING and new == STATE_CONFIRMED:
        return "⚠ JUMP: FORMING → CONFIRMED (possible chase — verify volume)"
    if new in (STATE_FAILED, STATE_EXPIRED):
        return f"⚫ INVALIDATED: {old} → {new}"
    return f"{old} → {new}"


def invalidation_check(ticker: str, df: pd.DataFrame, db_row: pd.Series) -> bool:
    """
    Returns True if the pattern should be marked FAILED.
    Checks pattern-specific invalidation rules.
    """
    close = df["Close"].squeeze()
    low = df["Low"].squeeze()
    volume = df["Volume"].squeeze()
    pattern = str(db_row.get("pattern", ""))

    # FIX: exclude today (iloc[-21:-1]) so a distribution-day spike doesn't
    # inflate the baseline and prevent the invalidation from firing.
    avg_vol = volume.iloc[-21:-1].mean()
    today_vol = volume.iloc[-1]
    last_close = close.iloc[-1]

    pivot = float(db_row.get("pivot_price", 0) or 0)

    if pattern == "VCP":
        # Failed: close below last contraction low (use stop as proxy)
        stop = float(db_row.get("stop", 0) or 0)
        if stop > 0 and last_close < stop:
            return True

    elif pattern.startswith("PB"):
        # Failed: close below 50 EMA on above-avg volume
        e50 = _ema(close, 50).iloc[-1]
        if last_close < e50 and today_vol > avg_vol * 1.2:
            return True

    elif pattern == "BASE":
        # Failed: close back inside base after breakout
        if db_row.get("state") in (STATE_CONFIRMED, STATE_AT_PIVOT) and pivot > 0:
            if last_close < pivot * 0.98:
                return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# MARKET REGIME FILTER
# ─────────────────────────────────────────────────────────────────────────────

def _is_market_in_uptrend(benchmark: str, sma_period: int = 200) -> bool:
    """
    Return True when benchmark last close >= its sma_period-day SMA.
    Uses XIU.TO (TSX 60 ETF) as the TSX proxy by default.
    Returns True (permissive) on any data failure so a bad Yahoo fetch never
    blocks the whole pipeline.
    """
    try:
        df = yf.download(
            tickers=benchmark,
            period=f"{sma_period + 20}d",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if df is None or df.empty:
            print(f"  {Fore.YELLOW}[regime] No data for {benchmark} — filter disabled{Style.RESET_ALL}")
            return True
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        close = df["Close"].dropna()
        if len(close) < sma_period:
            print(f"  {Fore.YELLOW}[regime] Insufficient history for {benchmark} — filter disabled{Style.RESET_ALL}")
            return True
        sma = float(close.rolling(sma_period).mean().iloc[-1])
        last = float(close.iloc[-1])
        uptrend = last >= sma
        status = f"{Fore.GREEN}UPTREND ✓" if uptrend else f"{Fore.RED}DOWNTREND ✗ — buy signals suppressed"
        print(f"  [regime] {benchmark}: last={last:.2f}  SMA{sma_period}={sma:.2f}  → {status}{Style.RESET_ALL}")
        return uptrend
    except Exception as exc:
        print(f"  {Fore.YELLOW}[regime] Error checking {benchmark}: {exc} — filter disabled{Style.RESET_ALL}")
        return True


# ─────────────────────────────────────────────────────────────────────────────
# CORE PIPELINE RUN
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(cfg: PipelineConfig) -> pd.DataFrame:
    # Use calendar-day values for signal timestamps (no tz arithmetic).
    today = market_today()

    # ── Setup dirs ───────────────────────────────────────────────────────────
    base = Path(cfg.base_dir)
    screener_dir = base / cfg.screener_subdir
    db_dir = base / cfg.db_subdir
    alerts_dir = base / cfg.alerts_subdir
    for d in (screener_dir, db_dir, alerts_dir):
        d.mkdir(parents=True, exist_ok=True)

    db_path = db_dir / "signal_history.csv"

    print(f"\n{'=' * 65}")
    print(f"  {Fore.YELLOW}🤖  AUTO ENTRY PIPELINE  —  {date_to_iso_extended(today)}{Style.RESET_ALL}")
    print(f"{'=' * 65}\n")

    # ── Step 1: Scan screener CSVs ───────────────────────────────────────────
    print(f"{Fore.CYAN}[1/5] Scanning screener outputs...{Style.RESET_ALL}")
    history = scan_screener_dir(screener_dir, cfg.lookback_csv_days)

    if not history:
        print(f"\n{Fore.RED}No screener files found in: {screener_dir}{Style.RESET_ALL}")
        print(f"  Drop your daily CSV/JSON files there and re-run.\n")
        _write_empty_report(alerts_dir, today, cfg.shared_report_path)
        return pd.DataFrame()

    print(f"  Found {len(history)} day(s) of screener data")

    # ── Step 2: Build tracked ticker list ───────────────────────────────────
    print(f"\n{Fore.CYAN}[2/5] Building ticker universe...{Style.RESET_ALL}")
    ticker_persistence = build_ticker_persistence(
        history, cfg.min_days_in_screener, cfg.max_tracked_tickers
    )
    tracked = [t for t, _ in ticker_persistence]

    if not tracked:
        print(f"  {Fore.YELLOW}No tickers met min_days_in_screener={cfg.min_days_in_screener}{Style.RESET_ALL}")
        return pd.DataFrame()

    print(f"  Tracking {len(tracked)} tickers "
          f"(min {cfg.min_days_in_screener} screener day(s) required)")
    for t, days in ticker_persistence[:10]:
        print(f"    {t:<14} appeared {days}d")
    if len(ticker_persistence) > 10:
        print(f"    ... and {len(ticker_persistence) - 10} more")

    # ── Step 3: Load signal DB & expire missing ──────────────────────────────
    print(f"\n{Fore.CYAN}[3/5] Loading signal history...{Style.RESET_ALL}")
    db = load_signal_db(db_path)
    db = expire_missing_tickers(db, tracked, today)
    print(f"  DB has {len(db)} existing signals")

    # ── Step 4: Download price data & run detectors ──────────────────────────
    print(f"\n{Fore.CYAN}[4/5] Downloading price data & detecting patterns...{Style.RESET_ALL}")

    start_dt = date_to_iso_extended(today - timedelta(days=cfg.price_data_days))
    end_dt = date_to_iso_extended(today + timedelta(days=1))

    alerts: List[Dict] = []

    for ticker in tracked:
        print(f"  {ticker:<14}", end=" ", flush=True)
        try:
            raw = yf.download(
                ticker,
                start=start_dt,
                end=end_dt,
                auto_adjust=True,
                progress=False
            )
            if raw.empty or len(raw) < 60:
                print(f"{Fore.YELLOW}insufficient data{Style.RESET_ALL}")
                continue

            raw.index = pd.to_datetime(raw.index).tz_localize(None)

            close = raw["Close"].squeeze()
            high = raw["High"].squeeze()
            low = raw["Low"].squeeze()
            last_price = float(close.iloc[-1])

            # Check invalidation for existing signals
            existing = db[db[SIGNAL_COL_TICKER] == ticker]
            for idx, ex_row in existing.iterrows():
                if ex_row[SIGNAL_COL_STATE] in (STATE_ACTIVE, STATE_AT_PIVOT, STATE_CONFIRMED):
                    if invalidation_check(ticker, raw, ex_row):
                        db.at[idx, SIGNAL_COL_STATE] = STATE_FAILED
                        db.at[idx, SIGNAL_COL_DETAIL] = "Invalidation rule triggered"
                        db.at[idx, SIGNAL_COL_LAST_SEEN] = today
                        print(f"{Fore.RED}FAILED{Style.RESET_ALL} ", end="")

            # Run pattern detectors
            patterns = detect_all_patterns(ticker, raw)

            if not patterns:
                print(f"{Fore.WHITE}no pattern{Style.RESET_ALL}")
                continue

            best = patterns[0]
            pattern = best["pattern"]
            state = best["state"]
            pivot = best.get("pivot", last_price)
            detail = best["detail"]

            # Compute trade levels
            entry = round(pivot * 1.005, 2)  # tiny buffer above pivot
            levels = compute_levels(close, high, low, entry,
                                    cfg.atr_period, cfg.atr_stop_mult,
                                    cfg.max_stop_pct)
            sizing = compute_position_size(
                cfg.account_size, cfg.risk_per_trade_pct,
                levels["entry"], levels["stop"]
            )
            # FIX: compute R:R from the actual target vs risk, not from
            # target_2r which was always exactly 2× risk (always = 2.0).
            # Now target_2r is resistance-derived so this reflects the real
            # market structure. Resistance-absent fallback stays at 2.0 but is
            # correctly flagged via levels["resistance_based"].
            rr = (levels["target_2r"] - levels["entry"]) / max(levels["entry"] - levels["stop"], 0.01)

            # ── Update or insert DB record ───────────────────────────────────
            match = db[(db[SIGNAL_COL_TICKER] == ticker) & (db[SIGNAL_COL_PATTERN] == pattern)]

            if match.empty:
                # New signal
                new_row = {
                    SIGNAL_COL_TICKER: ticker,
                    SIGNAL_COL_PATTERN: pattern,
                    SIGNAL_COL_STATE: state,
                    SIGNAL_COL_FIRST_SEEN: today,
                    SIGNAL_COL_LAST_SEEN: today,
                    SIGNAL_COL_DAYS_IN_STATE: 1,
                    SIGNAL_COL_CONSECUTIVE_SCREENER_DAYS: dict(ticker_persistence).get(ticker, 1),
                    SIGNAL_COL_ENTRY: levels[SIGNAL_COL_ENTRY],
                    SIGNAL_COL_STOP: levels[SIGNAL_COL_STOP],
                    SIGNAL_COL_TARGET_2R: levels[SIGNAL_COL_TARGET_2R],
                    SIGNAL_COL_TARGET_3R: levels[SIGNAL_COL_TARGET_3R],
                    SIGNAL_COL_RISK_PCT: levels[SIGNAL_COL_RISK_PCT],
                    SIGNAL_COL_PIVOT_PRICE: pivot,
                    SIGNAL_COL_DETAIL: detail,
                    SIGNAL_COL_ALERT_SENT: False,
                }
                db = pd.concat([db, pd.DataFrame([new_row])], ignore_index=True)
                transition = f"NEW {state}"
                print(f"{Fore.GREEN}NEW {state}{Style.RESET_ALL}", end="")
            else:
                idx = match.index[0]
                old_state = db.at[idx, SIGNAL_COL_STATE]
                transition = state_transition_label(old_state, state)

                db.at[idx, SIGNAL_COL_STATE] = state
                db.at[idx, SIGNAL_COL_LAST_SEEN] = today
                db.at[idx, SIGNAL_COL_DETAIL] = detail
                db.at[idx, SIGNAL_COL_ENTRY] = levels[SIGNAL_COL_ENTRY]
                db.at[idx, SIGNAL_COL_STOP] = levels[SIGNAL_COL_STOP]
                db.at[idx, SIGNAL_COL_TARGET_2R] = levels[SIGNAL_COL_TARGET_2R]
                db.at[idx, SIGNAL_COL_TARGET_3R] = levels[SIGNAL_COL_TARGET_3R]
                db.at[idx, SIGNAL_COL_RISK_PCT] = levels[SIGNAL_COL_RISK_PCT]
                db.at[idx, SIGNAL_COL_PIVOT_PRICE] = pivot
                db.at[idx, SIGNAL_COL_CONSECUTIVE_SCREENER_DAYS] = dict(ticker_persistence).get(ticker, 1)

                if old_state == state:
                    db.at[idx, SIGNAL_COL_DAYS_IN_STATE] = int(db.at[idx, SIGNAL_COL_DAYS_IN_STATE] or 1) + 1
                else:
                    db.at[idx, SIGNAL_COL_DAYS_IN_STATE] = 1
                    db.at[idx, SIGNAL_COL_ALERT_SENT] = False  # re-alert on state change

                col = Fore.GREEN if state == STATE_CONFIRMED else (
                    Fore.YELLOW if state == STATE_AT_PIVOT else Fore.WHITE)
                print(f"{col}{transition}{Style.RESET_ALL}", end="")

            # Build alert record
            if state in (STATE_CONFIRMED, STATE_AT_PIVOT) or cfg.alert_on_forming:
                # FIX: AT_PIVOT now bypasses the min_rr filter just like
                # CONFIRMED. Previously only CONFIRMED had a bypass, so any
                # AT_PIVOT signal with rr < min_rr was silently dropped —
                # the user would never know a setup was approaching its pivot.
                if rr >= cfg.min_rr or state in (STATE_CONFIRMED, STATE_AT_PIVOT):
                    alerts.append({
                        SIGNAL_COL_TICKER: ticker,
                        SIGNAL_COL_PATTERN: pattern,
                        SIGNAL_COL_STATE: state,
                        "emoji": ALERT_EMOJI.get(state, ""),
                        "transition": transition,
                        "price": f"${last_price:.2f}",
                        SIGNAL_COL_ENTRY: f"${levels[SIGNAL_COL_ENTRY]:.2f}",
                        SIGNAL_COL_STOP: f"${levels[SIGNAL_COL_STOP]:.2f}",
                        SIGNAL_COL_RISK_PCT: f"{levels[SIGNAL_COL_RISK_PCT]:.1f}%",
                        "target_2R": f"${levels['target_2r']:.2f}",
                        "target_3R": f"${levels['target_3r']:.2f}",
                        "R:R": f"{rr:.1f}",
                        "shares": sizing["shares"],
                        "position_$": f"${sizing['position_$']:,.0f}",
                        SIGNAL_COL_SCREENER_DAYS: dict(ticker_persistence).get(ticker, 1),
                        SIGNAL_COL_DETAIL: detail,
                    })

            print()  # newline after ticker status
            time.sleep(0.2)

        except Exception as e:
            print(f"{Fore.RED}error: {e}{Style.RESET_ALL}")

    # ── Step 5: Save DB & write outputs ─────────────────────────────────────
    print(f"\n{Fore.CYAN}[5/5] Saving results...{Style.RESET_ALL}")
    save_signal_db(db, db_path)

    df_alerts = pd.DataFrame(alerts) if alerts else pd.DataFrame()

    if not df_alerts.empty:
        # Sort: CONFIRMED first, then AT_PIVOT, then FORMING; within group by screener_days desc
        state_order = {STATE_CONFIRMED: 0, STATE_AT_PIVOT: 1, STATE_FORMING: 2}
        df_alerts["_sort"] = df_alerts[SIGNAL_COL_STATE].map(state_order).fillna(3)
        df_alerts = df_alerts.sort_values(["_sort", SIGNAL_COL_SCREENER_DAYS],
                                          ascending=[True, False])
        df_alerts = df_alerts.drop(columns=["_sort"])

    # Write alert CSV
    alert_csv = alerts_dir / f"alerts_{date_to_iso_basic(today)}.csv"
    df_alerts.to_csv(alert_csv, index=False)

    candidates_queue_path = cfg.candidates_queue_path
    if candidates_queue_path:
        # ── Market regime check ──────────────────────────────────────────────
        market_ok = True
        if cfg.regime_filter:
            print(f"\n  Checking market regime ({cfg.regime_benchmark} vs SMA{cfg.regime_sma_period})...")
            market_ok = _is_market_in_uptrend(cfg.regime_benchmark, cfg.regime_sma_period)

        candidates_queue = []
        if market_ok:
            confirmed = df_alerts[df_alerts[SIGNAL_COL_STATE] == STATE_CONFIRMED]
            for index, row in confirmed.iterrows():
                candidates_queue.append({
                    SIGNAL_COL_TICKER: row.get(SIGNAL_COL_TICKER, ""),
                    INTENT_COL_ALERT_STATE: row.get(SIGNAL_COL_STATE, ""),
                    INTENT_COL_PRIORITY: len(candidates_queue) + 1,
                    SIGNAL_COL_PATTERN: row.get(SIGNAL_COL_PATTERN, ""),
                    INTENT_COL_ENTRY_PRICE_PLANNED: row.get(SIGNAL_COL_ENTRY, ""),
                    INTENT_COL_STOP_PRICE: row.get(SIGNAL_COL_STOP, ""),
                    INTENT_COL_TARGET_PRICE: row.get("target_2R", ""),
                    INTENT_COL_RR: row.get("R:R", ""),
                })
        else:
            print(f"  {Fore.RED}[regime] Market below SMA{cfg.regime_sma_period} — "
                  f"candidates queue left empty (no new buys){Style.RESET_ALL}")

        _write_candidates_queue(candidates_queue, candidates_queue_path)
        print(f"  Queue   → {candidates_queue_path} ({len(candidates_queue)} ticker(s))")

    # Write human-readable report
    _write_report(df_alerts, db, alerts_dir, today, cfg, len(tracked), cfg.shared_report_path)

    # ── Print summary ────────────────────────────────────────────────────────
    _print_summary(df_alerts, today, len(tracked))

    return df_alerts


# ─────────────────────────────────────────────────────────────────────────────
# REPORT WRITERS
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(df_alerts: pd.DataFrame, today: datetime, n_tracked: int):
    print(f"\n{'=' * 65}")
    print(f"  {Fore.GREEN}📊  DAILY ALERT SUMMARY — {date_to_iso_extended(today)}{Style.RESET_ALL}")
    print(f"{'=' * 65}\n")

    if df_alerts.empty:
        print(f"  {Fore.YELLOW}No actionable signals today. Patterns still forming.{Style.RESET_ALL}")
        return

    display_cols = [
        "emoji", SIGNAL_COL_TICKER, SIGNAL_COL_PATTERN, SIGNAL_COL_STATE,
        "price", SIGNAL_COL_ENTRY, SIGNAL_COL_STOP, SIGNAL_COL_RISK_PCT,
        "target_2R", "R:R", "shares", SIGNAL_COL_SCREENER_DAYS,
    ]
    available = [c for c in display_cols if c in df_alerts.columns]
    print(tabulate(df_alerts[available], headers="keys",
                   tablefmt="rounded_outline", showindex=False))

    confirmed = df_alerts[df_alerts[SIGNAL_COL_STATE] == STATE_CONFIRMED]
    at_pivot = df_alerts[df_alerts[SIGNAL_COL_STATE] == STATE_AT_PIVOT]

    print(f"\n  {Fore.RED}🔴 CONFIRMED (enter tomorrow open): {len(confirmed)}{Style.RESET_ALL}")
    print(f"  {Fore.YELLOW}🟡 AT PIVOT  (place buy-stop):       {len(at_pivot)}{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}🟢 FORMING   (watchlist):             "
          f"{len(df_alerts) - len(confirmed) - len(at_pivot)}{Style.RESET_ALL}")
    print(f"\n  Tickers scanned today: {n_tracked}")

    print(f"\n{Fore.YELLOW}Detail:{Style.RESET_ALL}")
    for _, row in df_alerts.iterrows():
        em = row.get("emoji", "")
        print(f"  {em} {row[SIGNAL_COL_TICKER]:<12} {row[SIGNAL_COL_PATTERN]:<10} {row[SIGNAL_COL_DETAIL]}")


def _write_candidates_queue(intents: List[Dict[str, object]], output_path: str) -> None:
    """Write structured entry intents to the queue CSV file."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    now = market_now()
    created_at = now.isoformat()
    signal_date = now.date().isoformat()
    rows = []
    for priority, intent in enumerate(intents, start=1):
        rows.append({
            SIGNAL_COL_TICKER: intent.get(SIGNAL_COL_TICKER, ""),
            INTENT_COL_SIGNAL_DATE: signal_date,
            INTENT_COL_ALERT_STATE: intent.get(INTENT_COL_ALERT_STATE, STATE_CONFIRMED),
            INTENT_COL_PRIORITY: intent.get(INTENT_COL_PRIORITY, priority),
            SIGNAL_COL_PATTERN: intent.get(SIGNAL_COL_PATTERN, ""),
            INTENT_COL_ENTRY_PRICE_PLANNED: intent.get(INTENT_COL_ENTRY_PRICE_PLANNED, ""),
            INTENT_COL_STOP_PRICE: intent.get(INTENT_COL_STOP_PRICE, ""),
            INTENT_COL_TARGET_PRICE: intent.get(INTENT_COL_TARGET_PRICE, ""),
            INTENT_COL_RR: intent.get(INTENT_COL_RR, ""),
            INTENT_COL_STATUS: "pending",
            INTENT_COL_REASON: "",
            INTENT_COL_CREATED_AT: created_at,
        })
    pd.DataFrame(rows, columns=INTENT_REQUIRED_COLS).to_csv(out, index=False)


def _write_report(df_alerts: pd.DataFrame, db: pd.DataFrame,
                  alerts_dir: Path, today: datetime,
                  cfg: PipelineConfig, n_tracked: int,
                  shared_report_path: Optional[str] = None):
    lines = []
    lines.append(f"TSX AUTO ENTRY PIPELINE — Daily Report")
    lines.append(f"Date    : {date_to_iso_extended(today)}")
    lines.append(f"Account : ${cfg.account_size:,.0f}   Risk/trade: {cfg.risk_per_trade_pct}%")
    lines.append(f"Tracked : {n_tracked} tickers")
    lines.append("=" * 65)

    if df_alerts.empty:
        lines.append("No actionable signals today.")
    else:
        for state_label, state in [
            ("CONFIRMED — Enter tomorrow open", STATE_CONFIRMED),
            ("AT PIVOT — Place buy-stop above pivot", STATE_AT_PIVOT),
            ("FORMING — Watch, check tomorrow", STATE_FORMING),
        ]:
            subset = df_alerts[df_alerts[SIGNAL_COL_STATE] == state]
            if subset.empty:
                continue
            lines.append(f"\n{state_label}:")
            lines.append("-" * 50)
            for _, row in subset.iterrows():
                lines.append(
                    f"  {row[SIGNAL_COL_TICKER]:<12} {row[SIGNAL_COL_PATTERN]:<8} "
                    f"Entry:{row[SIGNAL_COL_ENTRY]}  Stop:{row[SIGNAL_COL_STOP]}  "
                    f"T1:{row['target_2R']}  R:R:{row['R:R']}  "
                    f"Shares:{row['shares']}  ({row[SIGNAL_COL_SCREENER_DAYS]}d in screener)"
                )
                lines.append(f"    → {row[SIGNAL_COL_DETAIL]}")

    lines.append("\n" + "=" * 65)
    lines.append("FULL SIGNAL DB STATE:")
    lines.append("-" * 50)
    if not db.empty:
        for _, row in db.sort_values(SIGNAL_COL_STATE).iterrows():
            lines.append(
                f"  {row[SIGNAL_COL_TICKER]:<12} {row[SIGNAL_COL_PATTERN]:<10} "
                f"[{row[SIGNAL_COL_STATE]}]  last:{str(row.get(SIGNAL_COL_LAST_SEEN, ''))[:10]}"
                f"  days_in_state:{int(row.get(SIGNAL_COL_DAYS_IN_STATE, 1) or 1)}"
            )

    lines.append("\nINVALIDATION RULES:")
    lines.append("  VCP  : exit if price closes below stop")
    lines.append("  PB   : exit if price closes below 50 EMA on above-avg volume")
    lines.append("  BASE : exit if price closes back inside base after breakout")
    lines.append("\n⚠  Educational only. Not financial advice.")

    report_path = alerts_dir / f"report_{date_to_iso_basic(today)}.txt"
    report_text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    if shared_report_path:
        date_str = date_to_iso_extended(today)
        alerts_list = df_alerts.to_dict("records") if not df_alerts.empty else []
        db_records = db.to_dict("records") if not db.empty else []

        # Normalise DB records so None/NaT values survive JSON serialisation
        for rec in db_records:
            for k, v in rec.items():
                if pd.isna(v) if not isinstance(v, (list, dict)) else False:
                    rec[k] = "—"
                else:
                    rec[k] = str(v) if not isinstance(v, (int, float, bool)) else v

        write_pipeline_report(
            path=shared_report_path,
            date_str=date_str,
            account_size=cfg.account_size,
            risk_pct=cfg.risk_per_trade_pct,
            n_tracked=n_tracked,
            alerts=alerts_list,
            db_records=db_records,
        )
        print(f"  HTML    → {shared_report_path}")

    alerts_fname = f"alerts_{date_to_iso_basic(today)}.csv"
    print(f"  Report  → {report_path}")
    print(f"  Alerts  → {alerts_dir / alerts_fname}")
    print(f"  DB      → signal_db/signal_history.csv")


def _write_empty_report(alerts_dir: Path, today: datetime,
                        shared_report_path: Optional[str] = None):
    text = f"No screener files found — {date_to_iso_extended(today)}\n"
    path = alerts_dir / f"report_{date_to_iso_basic(today)}.txt"
    with open(path, "w") as f:
        f.write(text)

    if shared_report_path:
        # Write a minimal HTML page so position_monitor can still append to it
        write_pipeline_report(
            path=shared_report_path,
            date_str=date_to_iso_extended(today),
            account_size=0,
            risk_pct=0,
            n_tracked=0,
            alerts=[],
            db_records=[],
        )


# ─────────────────────────────────────────────────────────────────────────────
# SCHEDULER — runs daily at market close
# ─────────────────────────────────────────────────────────────────────────────

def run_scheduler(cfg: PipelineConfig):
    """
    Lightweight scheduler — no external dependency (no APScheduler needed).
    Runs the pipeline once per day at cfg.schedule_time (HH:MM, local time).
    """
    print(f"\n{Fore.CYAN}Scheduler active — will run daily at "
          f"{cfg.schedule_time} (local time).{Style.RESET_ALL}")
    print("Press Ctrl+C to stop.\n")

    last_run_date = None

    while True:
        now = datetime.now()
        scheduled_h, scheduled_m = map(int, cfg.schedule_time.split(":"))
        scheduled_today = now.replace(hour=scheduled_h, minute=scheduled_m,
                                      second=0, microsecond=0)

        if now >= scheduled_today and last_run_date != now.date():
            try:
                run_pipeline(cfg)
                last_run_date = now.date()
            except Exception as e:
                print(f"{Fore.RED}Pipeline error: {e}{Style.RESET_ALL}")

        # Sleep until next check (every 60s)
        next_check = 60 - now.second
        print(f"\r  Next check in {next_check}s — "
              f"last run: {last_run_date or 'never'}   ", end="", flush=True)
        time.sleep(next_check)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Auto Entry Pipeline — TSX Swing Trading"
    )
    parser.add_argument("--base-dir", default=".", help="Root directory")
    parser.add_argument("--account", default=None, type=float,
                        help="Account size override (default: read from data/funds)")
    parser.add_argument("--max-stop", default=7.0, type=float,
                        help="Max stop distance as %% of entry price (default: 7.0)")
    parser.add_argument("--min-days", default=1, type=int,
                        help="Min days ticker must appear in screener (default: 1)")
    parser.add_argument("--max-tickers", default=40, type=int)
    parser.add_argument("--schedule", action="store_true",
                        help="Run as daily scheduler at market close")
    parser.add_argument("--schedule-time", default="16:30",
                        help="HH:MM to run daily (default: 16:30)")
    parser.add_argument("--ticker", default=None,
                        help="Debug: analyse a single ticker and exit")
    parser.add_argument("--shared-report-file",
                        default="report/report.html",
                        help="Optional single report file path that pipeline writes to")
    parser.add_argument("--candidates_queue",
                        default="positions/candidates",
                        help="Optional output file for tickers that jumped to CONFIRMED today")
    args = parser.parse_args()

    if args.account is not None:
        account_size = args.account
    else:
        funds_path = Path(args.base_dir) / FUNDS_PATH
        account_size = read_funds(funds_path)
        if account_size <= 0:
            print(
                f"{Fore.YELLOW}Warning: funds file returned ${account_size:,.2f} — "
                f"position sizing will be zero. Use --account to override.{Style.RESET_ALL}"
            )

    cfg = PipelineConfig(
        base_dir=args.base_dir,
        account_size=account_size,
        risk_per_trade_pct=args.risk,
        max_stop_pct=args.max_stop,
        min_days_in_screener=args.min_days,
        max_tracked_tickers=args.max_tickers,
        schedule_time=args.schedule_time,
        shared_report_path=args.shared_report_file,
        candidates_queue_path=args.candidates_queue
    )

    # Single ticker debug mode
    if args.ticker:
        print(f"\nDebug mode: analysing {args.ticker}")
        end = market_now()
        start = end - timedelta(days=cfg.price_data_days)
        raw = yf.download(
            args.ticker,
            start=date_to_iso_extended(start),
            end=date_to_iso_extended(end),
            auto_adjust=True,
            progress=False
        )
        if raw.empty:
            print("No data returned.")
            return
        patterns = detect_all_patterns(args.ticker, raw)
        if patterns:
            for p in patterns:
                print(f"  {p['pattern']:<10} [{p['state']}]  {p['detail']}")
        else:
            print("  No pattern detected.")
        return

    if args.schedule:
        run_scheduler(cfg)
    else:
        run_pipeline(cfg)


if __name__ == "__main__":
    main()
