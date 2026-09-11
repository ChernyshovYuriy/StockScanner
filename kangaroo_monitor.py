"""
kangaroo_monitor.py
=====================
Position monitoring / virtual exits for the Kangaroo Tail sleeve — separate
DB/capital from every other sleeve (see config.py KANGAROO_* and
CLAUDE.md).

Exit logic here is deliberately NOT position_monitor.py's chandelier
trail — this sleeve trades a fixed, defined-risk setup (stop at the tail's
low, target at KANGAROO_RR_TARGET × that risk, matching exactly what Phase
2's R-multiple backtest validated), so compute_kangaroo_signal() below is a
new, self-contained function rather than an ExitParams override of
compute_signals() (which the momentum/macro sleeves use — that reuse only
works because their exits are still chandelier-shaped). It does reuse
position_monitor.py's fetch_intraday_snapshot(), load_or_fetch_data(),
trading_days_since_entry(), and execute_virtual_sells() by import — those
pieces are exit-rule-agnostic.

A stop/target hit is filled at the stop/target price itself, matching the
R-multiple backtest's own fill assumption (Low<=sl -> -1R, High>=tp -> +rr R)
— real slippage is NOT modelled, same caveat as kangaroo_buy.py's trigger
fill.

Run modes: same --mode pre-close / post-close semantics as
position_monitor.py. kangaroo_buy.py (the breakout watcher) is meant to run
in the same pre-close slot, not virtual_buy.py's 9:45 AM — see its
docstring.

Usage
-----
  python kangaroo_monitor.py --mode pre-close
  python kangaroo_monitor.py --mode post-close
  python kangaroo_monitor.py --mode pre-close --dry-run
"""

from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd
from colorama import Fore, Style, init

from concurrent_utils import acquire_lock
from config import (
    KANGAROO_ALERTS_PATH,
    KANGAROO_DB_PATH,
    KANGAROO_MAX_HOLD_DAYS,
    KANGAROO_REPORT_POSITION_PATH,
    PositionMonitorMode,
)
from db import get_cash, get_open_positions_df, init_db
from log_utils import log
from market_data import TodayBar
from position_monitor import (
    LOGS_PATH,
    LOOKBACK_DAYS_BEFORE_ENTRY,
    MIN_BARS_REQUIRED,
    execute_virtual_sells,
    fetch_intraday_snapshot,
    load_or_fetch_data,
    trading_days_since_entry,
)
from report_html import append_positions_report
from schema_keys import (
    POSITION_COL_ENTRY_DATE,
    POSITION_COL_ENTRY_PRICE,
    POSITION_COL_LAST_CLOSE,
    POSITION_COL_LAST_LOW,
    POSITION_COL_PNL_DOLLARS,
    POSITION_COL_PNL_PCT,
    POSITION_COL_REASON,
    POSITION_COL_SHARES,
    POSITION_COL_STATUS,
    POSITION_COL_TRADING_DAYS,
    SIGNAL_COL_TICKER,
)
from send_report import send_report, SendConfig
from time_utils import date_to_iso_basic, is_market_open, market_now, market_today

init(autoreset=True)

TSX_TZ = ZoneInfo("America/Toronto")


@dataclass
class KangarooPosition:
    ticker: str
    entry_date: date
    entry_price: float
    shares: float
    stop_price: float
    target_price: float


def _parse_kangaroo_positions() -> list[KangarooPosition]:
    df = get_open_positions_df()
    if df.empty:
        return []
    positions: list[KangarooPosition] = []
    for _, row in df.iterrows():
        ticker = str(row[SIGNAL_COL_TICKER]).strip()
        if not ticker or ticker.lower() == "nan":
            continue
        stop_price = row.get("stop_price")
        target_price = row.get("target_price")
        if pd.isna(stop_price) or pd.isna(target_price):
            # A position missing either level can't be evaluated against
            # this sleeve's fixed stop/target rules — surfaced as BAD_DATA
            # by compute_kangaroo_signal() rather than silently held forever.
            stop_price = None
            target_price = None
        positions.append(KangarooPosition(
            ticker=ticker,
            entry_date=pd.to_datetime(row[POSITION_COL_ENTRY_DATE]).date(),
            entry_price=float(row[POSITION_COL_ENTRY_PRICE]),
            shares=float(row[POSITION_COL_SHARES]),
            stop_price=float(stop_price) if stop_price is not None else None,
            target_price=float(target_price) if target_price is not None else None,
        ))
    return positions


def compute_kangaroo_signal(
        pos: KangarooPosition,
        df: pd.DataFrame,
        today_bar: Optional[TodayBar] = None,
        max_hold_days: int = KANGAROO_MAX_HOLD_DAYS,
) -> Dict[str, object]:
    """Fixed stop / fixed target / time-stop exit — see module docstring
    for why this isn't compute_signals(). Stop is checked before target on
    a bar that hits both, matching the R-multiple backtest's own order."""
    if pos.entry_price <= 0:
        return {SIGNAL_COL_TICKER: pos.ticker, POSITION_COL_STATUS: "BAD_DATA",
                POSITION_COL_REASON: f"Invalid entry_price ({pos.entry_price})"}
    if pos.stop_price is None or pos.target_price is None:
        return {SIGNAL_COL_TICKER: pos.ticker, POSITION_COL_STATUS: "BAD_DATA",
                POSITION_COL_REASON: "Missing stop_price/target_price"}

    df = df.dropna(subset=["High", "Low", "Close"])
    if df.empty:
        return {SIGNAL_COL_TICKER: pos.ticker, POSITION_COL_STATUS: "NO_DATA", POSITION_COL_REASON: "No OHLC data"}

    entry_dt = pd.Timestamp(pos.entry_date)
    if today_bar is not None:
        last_low = today_bar.low
        last_high = today_bar.high
        last_close = today_bar.close
        price_source = today_bar.source
        last_date = market_now(TSX_TZ).date()
    else:
        last_bar = df.iloc[-1]
        last_low = float(last_bar["Low"])
        last_high = float(last_bar["High"])
        last_close = float(last_bar["Close"])
        price_source = "daily"
        last_date = df.index[-1].date()

    tdays = trading_days_since_entry(df, entry_dt)

    reasons: List[str] = []
    sell = False
    fill_price = last_close

    if last_low <= pos.stop_price:
        sell = True
        fill_price = pos.stop_price
        reasons.append(f"STOP_HIT(low {last_low:.2f} <= stop {pos.stop_price:.2f})")
    elif last_high >= pos.target_price:
        sell = True
        fill_price = pos.target_price
        reasons.append(f"TARGET_HIT(high {last_high:.2f} >= target {pos.target_price:.2f})")
    elif tdays >= max_hold_days:
        sell = True
        fill_price = last_close
        reasons.append(f"TIME_STOP({tdays}d)")

    status = "SELL" if sell else "HOLD"
    reason = "; ".join(reasons) if reasons else "OK"
    display_price = fill_price if sell else last_close
    pnl_pct = (display_price / pos.entry_price - 1.0) * 100.0
    pnl_dollars = (display_price - pos.entry_price) * pos.shares

    return {
        SIGNAL_COL_TICKER: pos.ticker,
        POSITION_COL_ENTRY_DATE: pos.entry_date.isoformat(),
        POSITION_COL_ENTRY_PRICE: round(pos.entry_price, 4),
        POSITION_COL_SHARES: pos.shares,
        "last_date": last_date.isoformat(),
        "price_source": price_source,
        POSITION_COL_LAST_CLOSE: round(display_price, 4),
        POSITION_COL_LAST_LOW: round(last_low, 4),
        POSITION_COL_PNL_PCT: round(pnl_pct, 2),
        POSITION_COL_PNL_DOLLARS: round(pnl_dollars, 2),
        "stop_price": round(pos.stop_price, 4),
        "target_price": round(pos.target_price, 4),
        POSITION_COL_TRADING_DAYS: tdays,
        POSITION_COL_STATUS: status,
        POSITION_COL_REASON: reason,
    }


def __run_send_report():
    cfg = SendConfig(
        file=KANGAROO_REPORT_POSITION_PATH,
        date=None,
        dry_run=False,
        alerts_dir=KANGAROO_ALERTS_PATH,
    )
    send_report(cfg)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Kangaroo Tail Sleeve Position Monitor")
    parser.add_argument("--mode", choices=["pre-close", "post-close"], default="pre-close")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    mode = PositionMonitorMode.PRE_CLOSE if args.mode == "pre-close" else PositionMonitorMode.POST_CLOSE

    service = "kangaroo_monitor"
    run_id = uuid.uuid4().hex

    try:
        lock_path, lock_file = acquire_lock(service)
    except BlockingIOError:
        log(service, run_id, "skip_already_running")
        sys.exit(0)

    log(service, run_id, "start", mode=mode)

    init_db(path=KANGAROO_DB_PATH)
    dry_run = args.dry_run
    funds_before = get_cash()
    funds_after = funds_before
    funds_gained = 0.0
    realized_pnl = 0.0

    market_open = is_market_open()
    use_intraday = mode == PositionMonitorMode.PRE_CLOSE and market_open
    execute_sells = mode == PositionMonitorMode.PRE_CLOSE and market_open

    if mode == PositionMonitorMode.PRE_CLOSE and not market_open:
        log(service, run_id, "sells_suppressed_market_closed")
        print(f"  {Fore.YELLOW}⚠  Market closed — running informational only, sells suppressed.{Style.RESET_ALL}")

    print(f"\n{'=' * 65}")
    print(f"  {Fore.YELLOW}🦘  Kangaroo Tail Sleeve — Position Monitor{Style.RESET_ALL}")
    print(f"  Exit: fixed stop/target, time_stop={KANGAROO_MAX_HOLD_DAYS}d")
    print(f"{'=' * 65}\n")

    positions = _parse_kangaroo_positions()
    if not positions:
        print("No positions found — nothing to monitor.")
        lock_file.close()
        sys.exit(0)

    print(f"  Loaded {len(positions)} open position(s) from database\n")
    LOGS_PATH.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for pos in positions:
        print(f"  {Fore.CYAN}{pos.ticker:<14}{Style.RESET_ALL}", end=" ", flush=True)
        start = (pd.Timestamp(pos.entry_date) - pd.Timedelta(days=LOOKBACK_DAYS_BEFORE_ENTRY)).date()
        df = load_or_fetch_data(pos.ticker, start=start)

        if df.empty or len(df) < MIN_BARS_REQUIRED:
            print(f"{Fore.YELLOW}insufficient data ({len(df)} bars){Style.RESET_ALL}")
            rows.append({SIGNAL_COL_TICKER: pos.ticker, POSITION_COL_STATUS: "NO_DATA",
                         POSITION_COL_REASON: f"Insufficient bars ({len(df)})"})
            continue

        needed = {"High", "Low", "Close"}
        if not needed.issubset(df.columns):
            missing = sorted(needed - set(df.columns))
            print(f"{Fore.RED}missing columns: {missing}{Style.RESET_ALL}")
            rows.append({SIGNAL_COL_TICKER: pos.ticker, POSITION_COL_STATUS: "BAD_DATA",
                         POSITION_COL_REASON: f"Missing columns: {missing}"})
            continue

        today_bar: Optional[TodayBar] = None
        if use_intraday:
            today_bar = fetch_intraday_snapshot(pos.ticker)
            if today_bar is not None:
                print(f"{Fore.CYAN}[live low={today_bar.low:.2f} high={today_bar.high:.2f} "
                      f"close={today_bar.close:.2f}]{Style.RESET_ALL} ", end="")

        result = compute_kangaroo_signal(pos, df, today_bar=today_bar)
        status = result.get(POSITION_COL_STATUS, "")
        color = Fore.RED if status == "SELL" else Fore.GREEN
        print(f"{color}{status}{Style.RESET_ALL}  pnl={result.get(POSITION_COL_PNL_PCT, '?')}%  "
              f"stop={result.get('stop_price', '?')}  target={result.get('target_price', '?')}  "
              f"{result.get(POSITION_COL_REASON, '')}")
        rows.append(result)

    out_df = pd.DataFrame(rows)
    if "status" in out_df.columns:
        out_df["__rank"] = out_df["status"].map({"SELL": 0, "HOLD": 1}).fillna(9)
        asc = [True, False] if "pnl_%" in out_df.columns else [True]
        cols = ["__rank", "pnl_%"] if "pnl_%" in out_df.columns else ["__rank"]
        out_df = out_df.sort_values(cols, ascending=asc).drop(columns=["__rank"])

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 50)
    print(f"\n{'=' * 65}")
    print(out_df.to_string(index=False))
    print(f"{'=' * 65}\n")

    today_str = date_to_iso_basic(market_today())
    log_path = LOGS_PATH / f"kangaroo_monitor_{today_str}.csv"
    out_df.to_csv(log_path, index=False)
    print(f"Log saved → {log_path.resolve()}")

    if execute_sells:
        sell_rows = [
            r for r in rows
            if r.get(POSITION_COL_STATUS) == "SELL"
               and r.get(POSITION_COL_LAST_CLOSE) is not None
               and r.get(POSITION_COL_SHARES) is not None
        ]
        if sell_rows:
            funds_state = execute_virtual_sells(sell_rows=sell_rows, dry_run=dry_run, label="Kangaroo Tail")
            funds_before = funds_state.get("funds_before", funds_before)
            funds_after = funds_state.get("funds_after", funds_after)
            funds_gained = funds_state.get("funds_gained", 0.0)
            realized_pnl = funds_state.get("realized_pnl", 0.0)
        else:
            print(f"\n  {Fore.GREEN}✅  No SELL signals — all positions held.{Style.RESET_ALL}")
    else:
        sell_count = len([r for r in rows if r.get(POSITION_COL_STATUS) == "SELL"])
        if sell_count:
            print(f"\n  {Fore.YELLOW}⚠  {sell_count} SELL signal(s) detected (informational only this run).{Style.RESET_ALL}")

    report_file = str(Path(KANGAROO_REPORT_POSITION_PATH))
    date_str = date_to_iso_basic(market_today())
    rows_for_report = out_df.to_dict("records") if not out_df.empty else []
    append_positions_report(path=report_file, date_str=date_str, rows=rows_for_report)

    __run_send_report()

    log(service, run_id, "completed", positions=len(positions), mode=mode)
    lock_file.close()


if __name__ == "__main__":
    main()
