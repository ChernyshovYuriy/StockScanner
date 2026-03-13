"""
position_monitor.py

Reads a positions CSV (format: ticker,entry_date,entry_price,shares),
downloads OHLCV data, computes ATR-based risk controls, and prints a
HOLD/SELL table + writes a daily log CSV.

Run modes
---------
Pre-close  (e.g. 3:30 PM ET, market still open):
    python position_monitor.py --execute-sells
    - Fetches today's live intraday snapshot (5-min bars) so that last_low
      and last_close reflect what has already happened *today*, not yesterday.
    - Any SELL signals are acted on immediately: the position is removed from
      the positions file and the proceeds are added back to the funds file.

Post-close / EOD (e.g. 4:30 PM ET, after market close):
    python position_monitor.py
    - Uses the completed daily bar (same behaviour as before).
    - No positions are removed; output is informational only.
    - Typically followed by the screener + pipeline run to find tomorrow's buys.

Exit logic (defaults tuned for ~1-2 week swing holds):
    * Initial stop     = entry - 1.5 * ATR(14)
    * Chandelier trail = HH_since_entry - 2.5 * ATR(14)
    * Profit giveback  : if max_profit >= 3% and current <= max_profit - 2% => SELL
    * Time stop        : if >= 7 trading days and profit < +0.5% => SELL

Dependencies:
    pip install pandas yfinance colorama
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from colorama import Fore, Style, init

from report_html import append_positions_report
from time_utils import date_to_iso_basic, market_now, market_today

init(autoreset=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Default file paths (all overridable via CLI)
DEFAULT_POSITIONS_CSV = Path("positions/own.csv")
DEFAULT_FUNDS_FILE = Path("data/funds")
DATA_DIR = Path("data_cache")
LOGS_DIR = Path("logs")

# ATR / stop parameters
ATR_PERIOD = 14
LOOKBACK_DAYS_BEFORE_ENTRY = 80  # enough pre-entry bars for a stable ATR
MIN_BARS_REQUIRED = 25
INITIAL_STOP_ATR_K = 1.5
CHAND_TRAIL_ATR_K = 2.5

# Profit giveback rule
GIVEBACK_ACTIVATE_PCT = 3.0  # arm the rule once max profit hits 3%
GIVEBACK_ALLOW_PCT = 2.0  # tolerate up to 2% pullback from peak

# Time stop
TIME_STOP_DAYS = 7  # trading days
TIME_STOP_MIN_PROFIT_PCT = 0.5  # require +0.5% by day 7

# Stop trigger mode:
#   "low"   -> fire if today's low  <= stop  (catches intraday breach)
#   "close" -> fire if today's close < stop  (more conservative, EOD-only)
STOP_TRIGGER = "low"

# Cache daily bars locally to avoid redundant downloads
ENABLE_CACHE = True

# TSX timezone
TSX_TZ = ZoneInfo("America/Toronto")

# TSX regular session hours (ET)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MIN = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MIN = 0


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Position:
    ticker: str
    entry_date: date
    entry_price: float
    shares: float


@dataclass
class TodayBar:
    """Live intraday snapshot for the current session."""
    low: float
    close: float  # latest traded price (last 5-min close)
    high: float  # session high so far
    source: str  # e.g. "5m-intraday"


# ─────────────────────────────────────────────────────────────────────────────
# MARKET HOURS CHECK
# ─────────────────────────────────────────────────────────────────────────────

def is_market_open() -> bool:
    """Return True if TSX is currently in its regular session (9:30–16:00 ET)."""
    now = market_now(TSX_TZ)
    open_min = MARKET_OPEN_HOUR * 60 + MARKET_OPEN_MIN
    close_min = MARKET_CLOSE_HOUR * 60 + MARKET_CLOSE_MIN
    now_min = now.hour * 60 + now.minute
    return open_min <= now_min < close_min


# ─────────────────────────────────────────────────────────────────────────────
# INTRADAY SNAPSHOT  (used during pre-close run)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_intraday_snapshot(ticker: str) -> Optional[TodayBar]:
    """
    Fetch today's 5-min bars and return a TodayBar with:
      - low   : the session low so far  (used for stop-hit check)
      - close : the latest 5-min close  (used for PnL / giveback)
      - high  : the session high so far

    Returns None on any failure; caller falls back to completed daily bar.
    """
    try:
        df = yf.download(
            tickers=ticker,
            period="1d",
            interval="5m",
            auto_adjust=True,
            progress=False,
        )
        if df is None or df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.dropna(subset=["High", "Low", "Close"])
        if df.empty:
            return None

        return TodayBar(
            low=float(df["Low"].min()),
            close=float(df["Close"].iloc[-1]),
            high=float(df["High"].max()),
            source="5m-intraday",
        )

    except Exception as e:
        print(f"    {Fore.YELLOW}Intraday snapshot failed for {ticker}: {e}{Style.RESET_ALL}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# DAILY OHLC DOWNLOAD + CACHE
# ─────────────────────────────────────────────────────────────────────────────

def download_ohlc(ticker: str, start: date, end: Optional[date] = None) -> pd.DataFrame:
    """Download daily OHLCV bars from Yahoo Finance."""
    end_dt = end or (date.today() + timedelta(days=1))
    df = yf.download(
        tickers=ticker,
        start=start.isoformat(),
        end=end_dt.isoformat(),
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.copy()
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def load_or_fetch_data(ticker: str, start: date) -> pd.DataFrame:
    """Load daily bars from local cache if available, else download and cache."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_DIR / f"{ticker.replace('/', '_')}.csv"

    cached = pd.DataFrame()
    if ENABLE_CACHE and cache_path.exists():
        try:
            cached = (
                pd.read_csv(cache_path, parse_dates=["Date"])
                .set_index("Date")
                .pipe(lambda d: d.set_index(pd.to_datetime(d.index)))
                .sort_index()
            )
        except Exception:
            cached = pd.DataFrame()

    fetched = download_ohlc(ticker, start=start)

    if fetched.empty and not cached.empty:
        df = cached
    elif not cached.empty and not fetched.empty:
        df = (
            pd.concat([cached, fetched])
            .pipe(lambda d: d[~d.index.duplicated(keep="last")])
            .sort_index()
        )
    else:
        df = fetched

    if ENABLE_CACHE and not df.empty:
        out = df.copy()
        out.insert(0, "Date", out.index)
        out.to_csv(cache_path, index=False)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# ATR HELPER
# ─────────────────────────────────────────────────────────────────────────────

def wilder_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Wilder's smoothed ATR (alpha = 1/period)."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev = close.shift(1)

    tr = pd.concat(
        [(high - low).abs(), (high - prev).abs(), (low - prev).abs()],
        axis=1,
    ).max(axis=1)

    return tr.ewm(alpha=1 / period, adjust=False).mean()


def trading_days_since_entry(df: pd.DataFrame, entry_dt: pd.Timestamp) -> int:
    """Count bars from entry date to last bar (inclusive)."""
    if df.empty:
        return 0
    return int((df.index.normalize() >= entry_dt.normalize()).sum())


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_signals(
        pos: Position,
        df: pd.DataFrame,
        today_bar: Optional[TodayBar] = None,
) -> Dict[str, object]:
    """
    Compute stops and exit signals for a single position.

    Parameters
    ----------
    pos       : the open position
    df        : daily OHLCV history (used for ATR, chandelier, max-PnL history)
    today_bar : optional live intraday snapshot; when provided, last_low and
                last_close are taken from this instead of the last daily bar.
                This is what makes the pre-close run see today's intraday move.
    """
    entry_dt = pd.Timestamp(pos.entry_date)

    df = df.copy().dropna(subset=["High", "Low", "Close"])
    if df.empty:
        return {"ticker": pos.ticker, "status": "NO_DATA", "reason": "No OHLC data"}

    # ── ATR on daily history ──────────────────────────────────────────────────
    df["ATR"] = wilder_atr(df, ATR_PERIOD)

    atr_latest = float(df["ATR"].iloc[-1]) if pd.notna(df["ATR"].iloc[-1]) else float("nan")
    if pd.isna(atr_latest):
        atr_non_nan = df["ATR"].dropna()
        atr_latest = float(atr_non_nan.iloc[-1]) if not atr_non_nan.empty else float("nan")

    if pd.isna(atr_latest):
        return {"ticker": pos.ticker, "status": "NO_ATR",
                "reason": "ATR could not be computed (insufficient history)"}

    # ── Bars from entry onwards ───────────────────────────────────────────────
    after_entry = df[df.index.normalize() >= entry_dt.normalize()]
    if after_entry.empty:
        return {"ticker": pos.ticker, "status": "NO_DATA",
                "reason": f"No bars on/after entry_date ({pos.entry_date})"}

    # ── Stop levels ───────────────────────────────────────────────────────────
    # Chandelier anchors to the highest HIGH since entry (uses daily wicks, intentional)
    hh_since_entry = float(after_entry["High"].max())
    initial_stop = pos.entry_price - INITIAL_STOP_ATR_K * atr_latest
    chandelier_stop = hh_since_entry - CHAND_TRAIL_ATR_K * atr_latest
    stop_price = max(initial_stop, chandelier_stop)

    # ── Price data: prefer live intraday snapshot over last daily bar ─────────
    if today_bar is not None:
        last_low = today_bar.low
        last_close = today_bar.close
        price_source = today_bar.source
        last_date = market_now(TSX_TZ).date()
    else:
        last_bar = df.iloc[-1]
        last_low = float(last_bar["Low"])
        last_close = float(last_bar["Close"])
        price_source = "daily"
        last_date = df.index[-1].date()

    # ── PnL ──────────────────────────────────────────────────────────────────
    pnl_pct = (last_close / pos.entry_price - 1.0) * 100.0
    # Peak uses daily Close history to avoid inflating with intraday wicks
    peak_price = float(after_entry["Close"].max())
    max_pnl_pct = (peak_price / pos.entry_price - 1.0) * 100.0

    tdays = trading_days_since_entry(df, entry_dt)

    # ── Exit conditions ───────────────────────────────────────────────────────
    reasons: List[str] = []
    sell = False

    # 1. Stop hit
    if STOP_TRIGGER.lower() == "low":
        if last_low <= stop_price:
            sell = True
            reasons.append(f"STOP_HIT(low {last_low:.2f} <= stop {stop_price:.2f})")
    else:
        if last_close < stop_price:
            sell = True
            reasons.append(f"STOP_HIT(close {last_close:.2f} < stop {stop_price:.2f})")

    # 2. Profit giveback
    if max_pnl_pct >= GIVEBACK_ACTIVATE_PCT:
        if pnl_pct <= (max_pnl_pct - GIVEBACK_ALLOW_PCT):
            sell = True
            reasons.append(f"GIVEBACK(peak {max_pnl_pct:.1f}% → now {pnl_pct:.1f}%)")

    # 3. Time stop
    if tdays >= TIME_STOP_DAYS and pnl_pct < TIME_STOP_MIN_PROFIT_PCT:
        sell = True
        reasons.append(f"TIME_STOP({tdays}d, pnl {pnl_pct:.1f}%)")

    status = "SELL" if sell else "HOLD"
    reason = "; ".join(reasons) if reasons else "OK"

    risk_per_share = pos.entry_price - initial_stop
    r_multiple = (
        (last_close - pos.entry_price) / risk_per_share
        if risk_per_share > 0 else float("nan")
    )

    return {
        "ticker": pos.ticker,
        "entry_date": pos.entry_date.isoformat(),
        "entry_price": round(pos.entry_price, 4),
        "shares": pos.shares,
        "last_date": last_date.isoformat(),
        "price_source": price_source,
        "last_close": round(last_close, 4),
        "last_low": round(last_low, 4),
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


# ─────────────────────────────────────────────────────────────────────────────
# POSITIONS FILE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

POSITIONS_COLS = ["ticker", "entry_date", "entry_price", "shares"]


def parse_positions_csv(path: Path) -> list[Position]:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {path.resolve()}")

    df = pd.read_csv(path)
    missing = set(POSITIONS_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    positions: list[Position] = []
    for _, row in df.iterrows():
        ticker = str(row["ticker"]).strip()
        if not ticker:
            continue
        positions.append(Position(
            ticker=ticker,
            entry_date=pd.to_datetime(row["entry_date"]).date(),
            entry_price=float(row["entry_price"]),
            shares=float(row["shares"]),
        ))
    return positions


# ─────────────────────────────────────────────────────────────────────────────
# FUNDS FILE HELPERS  (matches virtual_buy.py implementation exactly)
# ─────────────────────────────────────────────────────────────────────────────

def read_funds(path: Path) -> float:
    """Read available capital from a plain-text funds file."""
    if not path.exists():
        print(f"{Fore.YELLOW}Funds file not found: {path}{Style.RESET_ALL}")
        return 0.0

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            return float(line.replace(",", "").replace("$", ""))
        except ValueError:
            print(f"{Fore.YELLOW}Could not parse funds value '{line}' in {path}{Style.RESET_ALL}")
            return 0.0

    return 0.0


def write_funds(path: Path, amount: float) -> None:
    """Overwrite the funds file with the new balance, preserving comment lines."""
    comments: list[str] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("#"):
                comments.append(line)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(comments + [f"{amount:.2f}"]) + "\n", encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# VIRTUAL SELL EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def execute_virtual_sells(
        sell_rows: List[Dict],
        positions_path: Path,
        funds_path: Path,
        dry_run: bool = False,
) -> None:
    """
    For each SELL signal:
      1. Remove the position row from the positions CSV.
      2. Compute proceeds = shares * last_close.
      3. Add proceeds back to the funds file.
      4. Append the closed trade to logs/sells_YYYYMMDD.csv.
    """
    if not sell_rows:
        return

    print(f"\n{'─' * 60}")
    print(f"  {Fore.RED}💸  Virtual Sell Execution{Style.RESET_ALL}")
    print(f"{'─' * 60}")

    # ── Load current positions ────────────────────────────────────────────────
    if not positions_path.exists():
        print(f"{Fore.RED}Positions file not found: {positions_path}{Style.RESET_ALL}")
        return

    pos_df = pd.read_csv(positions_path)

    # ── Summarise each sell ───────────────────────────────────────────────────
    total_proceeds = 0.0
    sold_records: List[Dict] = []

    for row in sell_rows:
        ticker = row["ticker"]
        sell_price = float(row["last_close"])
        shares = float(row["shares"])
        proceeds = round(sell_price * shares, 2)
        pnl_dollars = round(float(row.get("pnl_$", 0)), 2)
        pnl_pct = round(float(row.get("pnl_%", 0)), 2)
        reason = row.get("reason", "")

        color = Fore.GREEN if pnl_dollars >= 0 else Fore.RED
        sign = "+" if pnl_dollars >= 0 else ""

        print(
            f"  {Fore.CYAN}{ticker:<14}{Style.RESET_ALL}"
            f"{shares:.0f} sh @ ${sell_price:.4f} = ${proceeds:,.2f}  "
            f"PnL: {color}{sign}${pnl_dollars:,.2f} ({sign}{pnl_pct:.2f}%){Style.RESET_ALL}"
            f"  [{reason}]"
        )

        total_proceeds += proceeds
        sold_records.append({
            "ticker": ticker,
            "entry_date": row.get("entry_date", ""),
            "entry_price": row.get("entry_price", ""),
            "shares": shares,
            "sell_date": market_now(TSX_TZ).date().isoformat(),
            "sell_price": sell_price,
            "proceeds": proceeds,
            "pnl_$": pnl_dollars,
            "pnl_%": pnl_pct,
            "reason": reason,
        })

    print(f"\n  Total proceeds : ${total_proceeds:,.2f}")

    if dry_run:
        print(f"\n  {Fore.CYAN}[DRY RUN] No files written.{Style.RESET_ALL}\n")
        return

    # ── Remove sold tickers from positions CSV ────────────────────────────────
    sold_tickers = {r["ticker"] for r in sell_rows}
    remaining_df = pos_df[~pos_df["ticker"].isin(sold_tickers)]
    remaining_df.to_csv(positions_path, index=False)

    print(
        f"\n  {Fore.GREEN}✓ Removed {len(sold_tickers)} position(s) from "
        f"{positions_path.resolve()}{Style.RESET_ALL}"
    )
    print(f"    Remaining open positions: {len(remaining_df)}")

    # ── Update funds file ─────────────────────────────────────────────────────
    current_funds = read_funds(funds_path)
    new_funds = current_funds + total_proceeds
    write_funds(funds_path, new_funds)

    print(
        f"  {Fore.GREEN}✓ Funds: "
        f"${current_funds:,.2f} + ${total_proceeds:,.2f} "
        f"→ ${new_funds:,.2f}{Style.RESET_ALL}"
    )

    # ── Append to sells log ───────────────────────────────────────────────────
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    today_str = date_to_iso_basic(market_today())
    sells_log = LOGS_DIR / f"sells_{today_str}.csv"

    sells_df = pd.DataFrame(sold_records)
    write_hdr = not sells_log.exists() or sells_log.stat().st_size == 0
    sells_df.to_csv(sells_log, mode="a", index=False, header=write_hdr)

    print(f"  {Fore.GREEN}✓ Sells logged → {sells_log.resolve()}{Style.RESET_ALL}")
    print(f"{'─' * 60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# REPORT HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _append_shared_report(shared_report_file: str, out_df: pd.DataFrame) -> None:
    date_str = date_to_iso_basic(market_today())
    rows = out_df.to_dict("records") if not out_df.empty else []
    append_positions_report(path=shared_report_file, date_str=date_str, rows=rows)
    print(f"Appended HTML positions section → {Path(shared_report_file).resolve()}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Position Monitor — HOLD/SELL signals with optional virtual sell execution.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--positions", "-p",
        default=str(DEFAULT_POSITIONS_CSV),
        help=f"Path to positions CSV (default: {DEFAULT_POSITIONS_CSV})",
    )
    parser.add_argument(
        "--funds", "-f",
        default=str(DEFAULT_FUNDS_FILE),
        help=f"Path to funds plain-text file (default: {DEFAULT_FUNDS_FILE})",
    )
    parser.add_argument(
        "--execute-sells",
        action="store_true",
        help=(
            "Execute SELL signals: remove sold positions from the positions file "
            "and add proceeds back to the funds file. "
            "Intended for the pre-close run (~3:30 PM ET). "
            "Without this flag the run is read-only (informational only)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --execute-sells: print what would happen without writing anything.",
    )
    parser.add_argument(
        "--force-intraday",
        action="store_true",
        help=(
            "Always fetch live intraday 5-min data regardless of market hours. "
            "Useful for testing outside regular session."
        ),
    )
    parser.add_argument(
        "--shared-report-file",
        default="report/report.html",
        help="Optional HTML report file that monitor appends its section to.",
    )
    args = parser.parse_args()

    positions_path = Path(args.positions)
    funds_path = Path(args.funds)

    # ── Determine run mode ────────────────────────────────────────────────────
    use_intraday = args.force_intraday or is_market_open()

    print(f"\n{'=' * 65}")
    print(f"  {Fore.YELLOW}📊  Position Monitor{Style.RESET_ALL}")
    if use_intraday:
        print(f"  {Fore.CYAN}Mode: PRE-CLOSE  (live 5-min intraday data){Style.RESET_ALL}")
    else:
        print(f"  {Fore.WHITE}Mode: POST-CLOSE  (completed daily bars){Style.RESET_ALL}")
    if args.execute_sells:
        tag = "[DRY RUN] " if args.dry_run else ""
        print(f"  {Fore.RED}{tag}Sell execution: ENABLED{Style.RESET_ALL}")
    print(f"{'=' * 65}\n")

    # ── Load positions ────────────────────────────────────────────────────────
    try:
        positions = parse_positions_csv(positions_path)
    except FileNotFoundError:
        print(f"{Fore.RED}Positions file not found: {positions_path.resolve()}{Style.RESET_ALL}")
        print("  Run virtual_buy.py first to populate positions.")
        sys.exit(0)

    if not positions:
        print("No positions found — nothing to monitor.")
        sys.exit(0)

    print(f"  Loaded {len(positions)} open position(s) from {positions_path}\n")

    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Analyse each position ─────────────────────────────────────────────────
    rows: List[Dict] = []

    for pos in positions:
        print(f"  {Fore.CYAN}{pos.ticker:<14}{Style.RESET_ALL}", end=" ", flush=True)

        start = (
                pd.Timestamp(pos.entry_date) - pd.Timedelta(days=LOOKBACK_DAYS_BEFORE_ENTRY)
        ).date()

        df = load_or_fetch_data(pos.ticker, start=start)

        if df.empty or len(df) < MIN_BARS_REQUIRED:
            print(f"{Fore.YELLOW}insufficient data ({len(df)} bars){Style.RESET_ALL}")
            rows.append({"ticker": pos.ticker, "status": "NO_DATA",
                         "reason": f"Insufficient bars ({len(df)})"})
            continue

        needed = {"High", "Low", "Close"}
        if not needed.issubset(df.columns):
            missing = sorted(needed - set(df.columns))
            print(f"{Fore.RED}missing columns: {missing}{Style.RESET_ALL}")
            rows.append({"ticker": pos.ticker, "status": "BAD_DATA",
                         "reason": f"Missing columns: {missing}"})
            continue

        # Fetch live intraday snapshot when market is open
        today_bar: Optional[TodayBar] = None
        if use_intraday:
            today_bar = fetch_intraday_snapshot(pos.ticker)
            if today_bar is not None:
                print(
                    f"{Fore.CYAN}[live low={today_bar.low:.2f} "
                    f"close={today_bar.close:.2f}]{Style.RESET_ALL} ",
                    end="",
                )

        result = compute_signals(pos, df, today_bar=today_bar)

        status = result.get("status", "")
        color = Fore.RED if status == "SELL" else Fore.GREEN
        print(
            f"{color}{status}{Style.RESET_ALL}  "
            f"pnl={result.get('pnl_%', '?')}%  "
            f"stop={result.get('stop_price', '?')}  "
            f"{result.get('reason', '')}"
        )

        rows.append(result)

    # ── Build output DataFrame ────────────────────────────────────────────────
    out_df = pd.DataFrame(rows)

    if "status" in out_df.columns:
        out_df["__rank"] = out_df["status"].map({"SELL": 0, "HOLD": 1}).fillna(9)
        asc = [True, False] if "pnl_%" in out_df.columns else [True]
        cols = ["__rank", "pnl_%"] if "pnl_%" in out_df.columns else ["__rank"]
        out_df = out_df.sort_values(cols, ascending=asc).drop(columns=["__rank"])

    # ── Print table ───────────────────────────────────────────────────────────
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 50)
    print(f"\n{'=' * 65}")
    print(out_df.to_string(index=False))
    print(f"{'=' * 65}\n")

    # ── Write daily log ────────────────────────────────────────────────────────
    today_str = date_to_iso_basic(market_today())
    log_path = LOGS_DIR / f"position_monitor_{today_str}.csv"
    out_df.to_csv(log_path, index=False)
    print(f"Log saved → {log_path.resolve()}")

    # ── Execute sells if requested ────────────────────────────────────────────
    if args.execute_sells:
        sell_rows = [
            r for r in rows
            if r.get("status") == "SELL"
               and r.get("last_close") is not None
               and r.get("shares") is not None
        ]

        if sell_rows:
            execute_virtual_sells(
                sell_rows=sell_rows,
                positions_path=positions_path,
                funds_path=funds_path,
                dry_run=args.dry_run,
            )
        else:
            print(f"\n  {Fore.GREEN}✅  No SELL signals — all positions held.{Style.RESET_ALL}")
    else:
        sell_count = len([r for r in rows if r.get("status") == "SELL"])
        if sell_count:
            print(
                f"\n  {Fore.YELLOW}⚠  {sell_count} SELL signal(s) detected. "
                f"Re-run with --execute-sells to act on them.{Style.RESET_ALL}"
            )

    # ── Append to shared HTML report ──────────────────────────────────────────
    if args.shared_report_file:
        _append_shared_report(args.shared_report_file, out_df)


if __name__ == "__main__":
    main()
