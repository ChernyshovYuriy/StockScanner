"""
momentum_pipeline.py
=====================
End-of-day pipeline for the momentum sleeve — a separate, isolated paper
account (own DB, own capital, own dashboard page; see config.py MOMENTUM_*
and CLAUDE.md). Mirrors main.py's structure but:

  - builds its own universe each run (relaxed ATR ceiling — see
    build_momentum_universe()) instead of consuming the already
    ATR-filtered CAN_TICKERS_URL
  - has no regime filter: a broad-index gate would block exactly the
    sector-specific rallies (e.g. gold/silver while XIU.TO is flat) this
    sleeve exists to catch
  - writes to data/momentum.db, never data/trading.db

Run manually or via system/stockscanner-momentum-pipeline.{service,timer}.
Does not import or call main.py — the two pipelines must stay independent,
same as the existing three-service rule in CLAUDE.md.

Usage
-----
  python momentum_pipeline.py
"""

from __future__ import annotations

import sys
import uuid

from colorama import Fore, Style

from auto_pipeline import PipelineConfig, run_pipeline
from canadian_stock_screener import DataManager, StockScreener, CONFIG, display_results, save_results
from concurrent_utils import acquire_lock
from config import (
    CACHE_PATH,
    MOMENTUM_ALERTS_PATH,
    MOMENTUM_DB_PATH,
    MOMENTUM_MAX_ATR_PCT,
    MOMENTUM_RAW_TICKERS_URL,
    MOMENTUM_REPORT_PATH,
    MOMENTUM_RISK_PER_TRADE_PCT,
    MOMENTUM_SCREENER_OUT_PATH,
    MOMENTUM_UNIVERSE_OUT_PATH,
)
from db import get_cash, init_db, set_cash
from log_utils import log
from send_report import SendConfig, send_report
from swing_tickers import Thresholds, UniverseBuilderConfig, read_tickers, run_universe_builder


# ─────────────────────────────────────────────────────────────────────────────
# UNIVERSE — relaxed ATR ceiling vs the core sleeve's swing_tickers.py run
# ─────────────────────────────────────────────────────────────────────────────

def build_momentum_universe() -> list[str]:
    """
    Run swing_tickers.py's own universe builder against the raw, pre-filter
    ticker source with MOMENTUM_MAX_ATR_PCT instead of the core sleeve's hard
    0.05 ceiling — this is the actual unblock that lets EDR.TO-style vertical
    movers into the universe (they never even reach the pattern detectors
    otherwise). NEO (.NE) interlisting duplicates are dropped first — same
    underlying security as the .TO/.V/.CN listing, just double the download
    work. Other gates (liquidity, above_50d, staleness) are kept as-is.
    """
    raw = read_tickers(MOMENTUM_RAW_TICKERS_URL)
    tickers = [t for t in raw if not t.endswith(".NE")]

    # UniverseBuilderConfig.tickers_source is a URL/path (read_tickers() calls
    # .startswith() on it) — unlike DataManager it can't take an in-memory
    # list, so the NE-deduped list is written to a local file first.
    raw_source_path = CACHE_PATH / "momentum_raw_tickers.txt"
    CACHE_PATH.mkdir(parents=True, exist_ok=True)
    raw_source_path.write_text("\n".join(tickers), encoding="utf-8")

    cfg = UniverseBuilderConfig(
        tickers_source=str(raw_source_path),
        benchmark="XIU.TO",
        out_file_path=str(MOMENTUM_UNIVERSE_OUT_PATH),
        out_one_line_file_path=str(MOMENTUM_UNIVERSE_OUT_PATH) + "_one_line",
        out_rejected_file_path=str(MOMENTUM_UNIVERSE_OUT_PATH) + "_rejected.csv",
        period="1y",
        interval="1d",
        auto_adjust=True,
        batch_size=80,
        sleep_seconds=1.0,
        thresholds=Thresholds(
            min_price=1.0,
            min_avg_dollar_vol_20=1_000_000.0,
            max_atr_pct_14=MOMENTUM_MAX_ATR_PCT,
            max_one_day_drop_126=-0.15,
            require_above_50d=True,
            prefer_above_200d=True,
            max_stale_days=5,
        ),
    )
    df_tradable, _df_rejected = run_universe_builder(cfg)
    return df_tradable["symbol"].tolist() if not df_tradable.empty else []


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STEPS
# ─────────────────────────────────────────────────────────────────────────────

def __run_stock_screener(tickers: list[str]):
    data_manager = DataManager(tickers)
    screener = StockScreener(CONFIG, data_manager)

    results = screener.run(force_refresh=True)

    if not results.empty:
        display_results(results, CONFIG, data_manager)
        save_results(results, MOMENTUM_SCREENER_OUT_PATH)
    else:
        print(f"{Fore.RED}No results generated. Check the momentum universe and data connection.{Style.RESET_ALL}")


def __run_pipeline():
    cfg = PipelineConfig(
        screener_subdir=MOMENTUM_SCREENER_OUT_PATH,
        alerts_subdir=MOMENTUM_ALERTS_PATH,
        risk_per_trade_pct=MOMENTUM_RISK_PER_TRADE_PCT,
        regime_filter=False,  # see module docstring
        enable_momentum_breakout=False,  # walk-forward showed no benefit (2026-08) — code stays opt-in
        shared_report_path=MOMENTUM_REPORT_PATH,
    )
    run_pipeline(cfg)


def __run_send_report():
    cfg = SendConfig(
        file=MOMENTUM_REPORT_PATH,
        date=None,
        dry_run=False,
        alerts_dir=MOMENTUM_ALERTS_PATH,
    )
    send_report(cfg)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    service = "momentum_pipeline"
    run_id = uuid.uuid4().hex

    try:
        lock_path, lock_file = acquire_lock(service)
    except BlockingIOError:
        log(service, run_id, "skip_already_running")
        sys.exit(0)

    log(service, run_id, "start", lock_file=str(lock_path))
    try:
        print(f"\n{'=' * 65}")
        print(f"  {Fore.YELLOW}🚀  Momentum Sleeve — Daily Run{Style.RESET_ALL}")
        print(f"{'=' * 65}\n")

        init_db(path=MOMENTUM_DB_PATH)
        if get_cash() <= 0:
            # First-time init for this sleeve's DB — see CLAUDE.md init pattern.
            from config import MOMENTUM_INITIAL_CAPITAL
            set_cash(MOMENTUM_INITIAL_CAPITAL)
            print(f"  Initialised momentum account with ${MOMENTUM_INITIAL_CAPITAL:,.2f}")

        print(f"{Fore.CYAN}[1/2] Building relaxed-ATR universe...{Style.RESET_ALL}")
        tickers = build_momentum_universe()
        print(f"  {len(tickers)} tickers in momentum universe\n")

        if not tickers:
            print(f"{Fore.RED}Empty momentum universe — skipping this run.{Style.RESET_ALL}")
            log(service, run_id, "empty_universe")
            sys.exit(0)

        print(f"{Fore.CYAN}[2/2] Running screener + entry pipeline...{Style.RESET_ALL}")
        __run_stock_screener(tickers)
        __run_pipeline()

        __run_send_report()

        log(service, run_id, "completed")

    finally:
        lock_file.close()
