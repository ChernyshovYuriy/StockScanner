"""
kangaroo_pipeline.py
======================
Daily scan for the Kangaroo Tail sleeve — a fully isolated, breakout-entry,
defined-risk paper account (own DB/capital; see config.py KANGAROO_* and
CLAUDE.md). Built on research/kangaroo_tail/ (Phase 1 detector; Phase 2
verified via a 16-fold walk-forward + R-multiple breakout backtest — see
research/kangaroo_tail/KANGAROO_TAIL_VERIFICATION_FINDINGS.md).

Unlike every other sleeve, an entry here is NOT "buy at next open off a
confirmed intent" — a Kangaroo Tail detection is a PENDING BREAKOUT setup:
wait for price to actually trade through the tail candle's own high (a
buy-stop trigger) before risking capital, with a stop at the tail's low and
a target at KANGAROO_RR_TARGET × that risk. This script only detects fresh
tail signals and queues them as PENDING intents; kangaroo_buy.py watches
live intraday price for the trigger.

Pending intents can (and normally do) survive across multiple daily runs
until they trigger or expire (KANGAROO_MAX_HOLD_DAYS trading days with no
breakout) — unlike the core/momentum sleeves, where every pipeline run's
intents are a fresh one-shot snapshot of that day's state machine. This
script therefore does its own carry-forward: still-valid pending intents
from previous runs are re-included alongside today's fresh detections
before calling db.save_intents() (which replaces only PENDING rows), and an
expired intent is flipped to SKIPPED first so it is excluded from that
replace and its history is preserved (see _carry_forward_intents()).

Run manually or via system/stockscanner-kangaroo-pipeline.{service,timer}
(~16:30 ET, same slot as main.py/momentum_pipeline.py — after that day's
bar closes, so the tail's own High/Low are final, not still-forming).

Usage
-----
  python kangaroo_pipeline.py
"""

from __future__ import annotations

import sys
import uuid

import pandas as pd
from colorama import Fore, Style, init

from concurrent_utils import acquire_lock
from config import (
    CAN_TICKERS_URL,
    KANGAROO_DB_PATH,
    KANGAROO_INITIAL_CAPITAL,
    KANGAROO_MAX_HOLD_DAYS,
    KANGAROO_RR_TARGET,
)
from db import get_cash, get_open_positions_df, init_db, load_pending_intents, mark_intent_skipped, save_intents, set_cash
from log_utils import log
from market_data_cache import sync_and_load
from position_monitor import trading_days_since_entry
from research.kangaroo_tail.batch import load_tickers
from research.kangaroo_tail.detector import detect_kangaroo_tail
from research.kangaroo_tail.types import TailConfig
from time_utils import market_today

init(autoreset=True)

# Shipped defaults (see research/kangaroo_tail/types.py) — the Phase 2
# "sweet_spot" config. Not overridden here so a future default change is
# picked up automatically, same as every detector call site in this module.
KANGAROO_CONFIG = TailConfig()

# Fetch this many extra calendar days of history so the detector's own
# lookback/ATR/volume-average windows are all satisfied for every ticker.
_FETCH_LOOKBACK_DAYS = 150


def _carry_forward_intents(bars_by_ticker: dict[str, pd.DataFrame]) -> list[dict]:
    """Load existing PENDING intents, expire the ones that have gone
    KANGAROO_MAX_HOLD_DAYS trading days without triggering (flipping them to
    SKIPPED so their history survives), and return the rest as plain dicts
    ready to be merged with today's fresh detections and passed to
    save_intents().
    """
    pending_df = load_pending_intents()
    survivors: list[dict] = []
    if pending_df.empty:
        return survivors

    today = market_today().date()
    for _, row in pending_df.iterrows():
        ticker = str(row["ticker"]).strip().upper()
        try:
            signal_dt = pd.Timestamp(row["signal_date"])
        except (ValueError, TypeError):
            survivors.append(row.drop(labels=["id"]).to_dict())
            continue

        bars = bars_by_ticker.get(ticker)
        tdays = trading_days_since_entry(bars.sort_index(), signal_dt) if bars is not None and not bars.empty else None
        if tdays is not None and tdays > KANGAROO_MAX_HOLD_DAYS:
            mark_intent_skipped(int(row["id"]), "breakout_not_triggered_expired")
            print(f"  {Fore.YELLOW}{ticker:<10}{Style.RESET_ALL} pending breakout expired "
                  f"({tdays}d, signal {signal_dt.date()}) — skipped")
            continue
        survivors.append(row.drop(labels=["id"]).to_dict())
    return survivors


def _scan_for_new_signals(tickers: list[str], bars_by_ticker: dict[str, pd.DataFrame],
                          already_pending_or_owned: set[str]) -> list[dict]:
    """Run the Phase 1 detector's live, last-bar-only check against every
    ticker not already pending/owned, and build one intent dict per fresh
    detection."""
    new_intents: list[dict] = []

    for ticker in tickers:
        if ticker in already_pending_or_owned:
            continue
        bars = bars_by_ticker.get(ticker)
        if bars is None or bars.empty:
            continue
        signal = detect_kangaroo_tail(ticker, bars, KANGAROO_CONFIG)
        if signal is None:
            continue

        risk = signal.high - signal.low
        if risk <= 0:
            continue
        target = signal.high + KANGAROO_RR_TARGET * risk

        print(f"  {Fore.GREEN}{ticker:<10}{Style.RESET_ALL} fresh Kangaroo Tail "
              f"({signal.date.date()})  trigger=${signal.high:.2f}  stop=${signal.low:.2f}  "
              f"target=${target:.2f}")

        new_intents.append({
            "ticker": ticker,
            "signal_date": signal.date.date().isoformat(),
            "alert_state": "PENDING_BREAKOUT",
            "priority": 1,
            "pattern": "kangaroo_tail",
            "entry_price_planned": round(signal.high, 4),
            "stop_price": round(signal.low, 4),
            "target_price": round(target, 4),
            "rr": KANGAROO_RR_TARGET,
        })
    return new_intents


def main() -> None:
    service = "kangaroo_pipeline"
    run_id = uuid.uuid4().hex

    try:
        lock_path, lock_file = acquire_lock(service)
    except BlockingIOError:
        log(service, run_id, "skip_already_running")
        sys.exit(0)

    log(service, run_id, "start", lock_file=str(lock_path))
    try:
        print(f"\n{'=' * 65}")
        print(f"  {Fore.YELLOW}🦘  Kangaroo Tail Sleeve — Daily Scan{Style.RESET_ALL}")
        print(f"{'=' * 65}\n")

        init_db(path=KANGAROO_DB_PATH)
        if get_cash() <= 0:
            set_cash(KANGAROO_INITIAL_CAPITAL)
            print(f"  Initialised Kangaroo Tail account with ${KANGAROO_INITIAL_CAPITAL:,.2f}")

        tickers = load_tickers(CAN_TICKERS_URL)
        print(f"  {len(tickers)} tickers in universe\n")

        end = market_today().strftime("%Y-%m-%d")
        start = (market_today() - pd.Timedelta(days=_FETCH_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        print(f"{Fore.CYAN}[1/3] Fetching daily bars...{Style.RESET_ALL}")
        bars_by_ticker = sync_and_load(tickers, start=start, end=end)

        # Also fetch bars for any currently-pending ticker that has fallen
        # out of the universe (e.g. delisted from the published list) so
        # its expiry check below still has data to work with.
        pending_df = load_pending_intents()
        missing = sorted(set(pending_df["ticker"].str.upper()) - set(bars_by_ticker)) if not pending_df.empty else []
        if missing:
            bars_by_ticker.update(sync_and_load(missing, start=start, end=end))

        print(f"\n{Fore.CYAN}[2/3] Carrying forward / expiring pending breakout intents...{Style.RESET_ALL}")
        survivors = _carry_forward_intents(bars_by_ticker)
        print(f"  {len(survivors)} pending intent(s) still valid")

        print(f"\n{Fore.CYAN}[3/3] Scanning for fresh Kangaroo Tail signals...{Style.RESET_ALL}")
        owned = set(get_open_positions_df()["ticker"].str.upper())
        already = {str(d["ticker"]).upper() for d in survivors} | owned
        new_intents = _scan_for_new_signals(tickers, bars_by_ticker, already)
        print(f"  {len(new_intents)} fresh signal(s)")

        save_intents(survivors + new_intents)
        print(f"\n{Fore.GREEN}✓ {len(survivors) + len(new_intents)} pending breakout intent(s) queued{Style.RESET_ALL}")

        log(service, run_id, "completed", survivors=len(survivors), new_signals=len(new_intents))

    finally:
        lock_file.close()


if __name__ == "__main__":
    main()
