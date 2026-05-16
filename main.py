import sys
import uuid

import yfinance as yf
from colorama import Fore, Style

from auto_pipeline import PipelineConfig, run_pipeline
from canadian_stock_screener import DataManager, StockScreener, CONFIG, display_results, save_results
from concurrent_utils import acquire_lock
from config import ALERTS_PATH
from config import CAN_TICKERS_PATH, CAN_TICKERS_ONE_LINE_PATH, CAN_TICKERS_REJECTED_PATH, SCREENER_OUT_PATH, \
    CAN_TICKERS_UNIVERSE_PATH, REPORT_PATH
from log_utils import log
from send_report import SendConfig, send_report
from swing_tickers import UniverseBuilderConfig, Thresholds, run_universe_builder

# ─────────────────────────────────────────────────────────────────────────────
# REGIME FILTER
# ─────────────────────────────────────────────────────────────────────────────
BENCHMARK = "XIU.TO"
REGIME_SMA_PERIOD = 200  # days


def __check_regime() -> bool:
    """
    Return True when the TSX benchmark is in a bull regime.

    Rule: XIU.TO last close >= its 200-day simple moving average.

    In a bear regime the live system still runs position_monitor to manage
    existing positions, but skips signal generation and new buy intents.
    This avoids opening new long positions into a declining market — the
    single biggest cause of losses in the 2022-2023 backtest period.

    Returns True (bull / allow trades) on any data error so a transient
    network failure never silently prevents the system from running.
    """
    try:
        df = yf.download(
            BENCHMARK,
            period="1y",
            auto_adjust=True,
            progress=False,
        )
        if df is None or df.empty:
            print(f"{Fore.YELLOW}  Regime check: no data for {BENCHMARK} — defaulting to BULL{Style.RESET_ALL}")
            return True

        if isinstance(df.columns, yf.core.frame.Column if hasattr(yf, 'core') else type(df.columns)):
            pass
        close = df["Close"].squeeze().dropna()
        if len(close) < REGIME_SMA_PERIOD:
            print(f"{Fore.YELLOW}  Regime check: insufficient history — defaulting to BULL{Style.RESET_ALL}")
            return True

        sma200 = float(close.rolling(REGIME_SMA_PERIOD).mean().iloc[-1])
        last = float(close.iloc[-1])
        in_bull = last >= sma200
        pct_diff = (last / sma200 - 1) * 100

        color = Fore.GREEN if in_bull else Fore.RED
        label = "BULL ✓" if in_bull else "BEAR ✗"
        print(f"  {color}Regime: {label}  |  {BENCHMARK} {last:.2f}  "
              f"vs 200d SMA {sma200:.2f}  ({pct_diff:+.1f}%){Style.RESET_ALL}")
        return in_bull

    except Exception as e:
        print(f"{Fore.YELLOW}  Regime check failed ({e}) — defaulting to BULL{Style.RESET_ALL}")
        return True


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STEPS
# ─────────────────────────────────────────────────────────────────────────────

def __build_swing_tickers():
    config = UniverseBuilderConfig(
        tickers_path=CAN_TICKERS_UNIVERSE_PATH,
        benchmark="XIU.TO",
        out_file_path=CAN_TICKERS_PATH,
        out_one_line_file_path=CAN_TICKERS_ONE_LINE_PATH,
        out_rejected_file_path=CAN_TICKERS_REJECTED_PATH,
        period="1y",
        interval="1d",
        auto_adjust=True,
        batch_size=80,
        sleep_seconds=1.0,
        thresholds=Thresholds(
            min_price=1.0,
            min_avg_dollar_vol_20=1_000_000.0,
            max_atr_pct_14=0.05,
            max_one_day_drop_126=-0.15,
            require_above_50d=True,
            prefer_above_200d=True,
            max_stale_days=5,
        ),
    )
    run_universe_builder(config)


def __run_stock_screener():
    data_manager = DataManager(CAN_TICKERS_PATH)
    screener = StockScreener(CONFIG, data_manager)

    results = screener.run(force_refresh=True)

    if not results.empty:
        display_results(results, CONFIG, data_manager)
        save_results(results, SCREENER_OUT_PATH)
    else:
        print(f"{Fore.RED}No results generated. Check your tickers and data connection.{Style.RESET_ALL}")


def __run_pipeline():
    cfg = PipelineConfig(
        shared_report_path=REPORT_PATH,
    )
    run_pipeline(cfg)


def __run_send_report():
    cfg = SendConfig(
        file=REPORT_PATH,
        date=None,
        dry_run=False,
        alerts_dir=ALERTS_PATH
    )
    send_report(cfg)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    service = "main"
    run_id = uuid.uuid4().hex

    try:
        lock_path, lock_file = acquire_lock(service)
    except BlockingIOError:
        log(service, run_id, "skip_already_running")
        sys.exit(0)

    log(service, run_id, "start", lock_file=str(lock_path))
    try:
        print(f"\n{'=' * 65}")
        print(f"  {Fore.YELLOW}🇨🇦  TSX Swing Trading System — Daily Run{Style.RESET_ALL}")
        print(f"{'=' * 65}\n")

        # ── Step 1: Market regime check ───────────────────────────────────────
        print(f"{Fore.CYAN}[1/4] Checking market regime...{Style.RESET_ALL}")
        bull_regime = __check_regime()

        if not bull_regime:
            # Bear regime: manage existing positions only, skip new signal generation
            print(f"\n{Fore.YELLOW}  Bear regime detected — skipping new signal generation.")
            print(f"  Existing positions will still be monitored by position_monitor.py{Style.RESET_ALL}")
            log(service, run_id, "bear_regime_skip_signals")
            # Send a minimal report so the daily email still arrives
            __run_send_report()
            log(service, run_id, "completed_bear_regime")
            sys.exit(0)

        print(f"\n{Fore.GREEN}  Bull regime confirmed — proceeding with full pipeline.{Style.RESET_ALL}\n")

        # ── Step 2: Build universe ────────────────────────────────────────────
        print(f"{Fore.CYAN}[2/4] Building swing ticker universe...{Style.RESET_ALL}")
        __build_swing_tickers()

        # ── Step 3: Score and rank ────────────────────────────────────────────
        print(f"\n{Fore.CYAN}[3/4] Running stock screener...{Style.RESET_ALL}")
        __run_stock_screener()

        # ── Step 4: Detect patterns and emit buy intents ──────────────────────
        print(f"\n{Fore.CYAN}[4/4] Running entry pipeline...{Style.RESET_ALL}")
        __run_pipeline()

        # ── Send report ───────────────────────────────────────────────────────
        __run_send_report()

        log(service, run_id, "completed")

    finally:
        lock_file.close()
