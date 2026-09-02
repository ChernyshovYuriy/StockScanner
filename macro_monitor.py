"""
macro_monitor.py
=================
Position monitoring / virtual exits for the macro conviction sleeve —
separate DB/capital from the core sleeve's position_monitor.py (see
config.py MACRO_* and CLAUDE.md).

Reuses position_monitor.py's compute_signals(), parse_positions_from_db(),
fetch_intraday_snapshot(), load_or_fetch_data(), execute_virtual_sells() by
import — same "import, don't fork" pattern as momentum_monitor.py. No
ExitParams override: this sleeve's positions come from the core sleeve's own
already-ATR-disciplined candidates (via macro_buy.py), so there's no ATR-
relaxation rationale to justify different stop widths the way momentum's
wide chandelier trail has. Concentration here is expressed through sizing
and the entry regime-gate (macro_buy.py), not through wider stops — this
module passes no exit_params=, reusing position_monitor.py's validated
defaults verbatim.

NEW logic (doesn't exist anywhere else in this repo): a regime-flip kill
switch. After the normal per-position exit evaluation, if today's macro
regime (macro_regime.get_macro_regime()) reads "risk_off", every position
still in HOLD status is force-liquidated (capital preservation first) —
Druckenmiller's "cut losers/change your mind fast" trait, mechanized as
"the backdrop turned hostile, get out regardless of any individual stop."
A position already flagged SELL for a real stop/giveback/time-stop reason
keeps its own reason; NO_DATA/BAD_DATA rows are left alone (no price to
force-sell at). The status override always happens (useful in the report
even off-hours); whether it actually reaches execute_virtual_sells() still
depends on the existing pre-close/market-open gate, same as every other
exit path here.

db.py's global DB_PATH means parse_positions_from_db() etc. operate on
whichever DB init_db() last pointed at in this process — set once, below,
before anything else runs.

Run modes: same --mode pre-close / post-close semantics as
position_monitor.py.

Usage
-----
  python macro_monitor.py --mode pre-close
  python macro_monitor.py --mode post-close
  python macro_monitor.py --mode pre-close --dry-run
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from colorama import Fore, Style, init

from concurrent_utils import acquire_lock
from config import (
    MACRO_ALERTS_PATH,
    MACRO_DB_PATH,
    MACRO_REPORT_POSITION_PATH,
    PositionMonitorMode,
)
from db import get_cash, init_db
from log_utils import log
from macro_regime import get_macro_regime
from position_monitor import (
    LOGS_PATH,
    MIN_BARS_REQUIRED,
    LOOKBACK_DAYS_BEFORE_ENTRY,
    TodayBar,
    compute_signals,
    execute_virtual_sells,
    fetch_intraday_snapshot,
    load_or_fetch_data,
    parse_positions_from_db,
)
from report_html import append_positions_report
from schema_keys import POSITION_COL_LAST_CLOSE, POSITION_COL_REASON, POSITION_COL_SHARES, POSITION_COL_STATUS, \
    SIGNAL_COL_TICKER
from send_report import send_report, SendConfig
from time_utils import date_to_iso_basic, is_market_open, market_today

init(autoreset=True)


def _apply_regime_flip(rows: List[Dict], regime: dict) -> None:
    """Mutate `rows` in place: force every currently-HOLD row to SELL with a
    `macro_regime_flip` reason when `regime` reads risk_off — the new logic
    described in the module docstring, isolated here so it's directly unit-
    testable without going through the full I/O-heavy main(). A no-op
    unless regime['label'] == 'risk_off'. Rows already SELL (a real stop/
    giveback/time-stop hit) keep their own reason; NO_DATA/BAD_DATA rows are
    left alone (no price to force-sell at)."""
    if regime.get("label") != "risk_off":
        return
    for r in rows:
        if r.get(POSITION_COL_STATUS) == "HOLD":
            r[POSITION_COL_STATUS] = "SELL"
            r[POSITION_COL_REASON] = f"macro_regime_flip(composite={regime.get('composite')})"


def __run_send_report():
    cfg = SendConfig(
        file=MACRO_REPORT_POSITION_PATH,
        date=None,
        dry_run=False,
        alerts_dir=MACRO_ALERTS_PATH,
    )
    send_report(cfg)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Macro Conviction Sleeve Position Monitor")
    parser.add_argument("--mode", choices=["pre-close", "post-close"], default="pre-close")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    mode = PositionMonitorMode.PRE_CLOSE if args.mode == "pre-close" else PositionMonitorMode.POST_CLOSE

    service = "macro_monitor"
    run_id = uuid.uuid4().hex

    try:
        lock_path, lock_file = acquire_lock(service)
    except BlockingIOError:
        log(service, run_id, "skip_already_running")
        sys.exit(0)

    log(service, run_id, "start", mode=mode)

    init_db(path=MACRO_DB_PATH)
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
    print(f"  {Fore.BLUE}🎯  Macro Conviction Sleeve — Position Monitor{Style.RESET_ALL}")
    print(f"{'=' * 65}\n")

    positions = parse_positions_from_db()
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
                print(f"{Fore.CYAN}[live low={today_bar.low:.2f} close={today_bar.close:.2f}]{Style.RESET_ALL} ", end="")

        result = compute_signals(pos, df, today_bar=today_bar, planned_stop=pos.stop_price)

        status = result.get(POSITION_COL_STATUS, "")
        color = Fore.RED if status == "SELL" else Fore.GREEN
        print(f"{color}{status}{Style.RESET_ALL}  pnl={result.get('pnl_%', '?')}%  "
              f"stop={result.get('stop_price', '?')}  {result.get('reason', '')}")
        rows.append(result)

    # ── Regime-flip kill switch: force-liquidate every HOLD position when the
    #    macro backdrop has turned hostile, regardless of individual stop
    #    status. Applied unconditionally (useful in the report/log even
    #    off-hours); whether it reaches execute_virtual_sells() still depends
    #    on the execute_sells gate below, same as every other exit path. ──
    regime = get_macro_regime()
    print(f"\n  Macro regime : {regime['label'].upper()} "
          f"(composite={regime['composite']}, votes={regime['votes']})")
    if regime["label"] == "risk_off":
        print(f"  {Fore.RED}⚠  Macro regime is RISK_OFF — force-liquidating all held positions "
              f"(capital preservation).{Style.RESET_ALL}")
        log(service, run_id, "regime_flip_liquidation", composite=regime["composite"], votes=regime["votes"])
    _apply_regime_flip(rows, regime)

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
    log_path = LOGS_PATH / f"macro_monitor_{today_str}.csv"
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
            funds_state = execute_virtual_sells(sell_rows=sell_rows, dry_run=dry_run, label="Macro")
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

    report_file = str(Path(MACRO_REPORT_POSITION_PATH))
    date_str = date_to_iso_basic(market_today())
    rows_for_report = out_df.to_dict("records") if not out_df.empty else []
    append_positions_report(path=report_file, date_str=date_str, rows=rows_for_report)

    __run_send_report()

    log(service, run_id, "completed", positions=len(positions), mode=mode)
    lock_file.close()


if __name__ == "__main__":
    main()
