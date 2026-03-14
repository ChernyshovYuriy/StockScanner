import sys
import sys
import uuid

from colorama import Fore, Style

from auto_pipeline import PipelineConfig, run_pipeline
from canadian_stock_screener import DataManager, StockScreener, CONFIG, display_results, save_results
from concurrent_utils import acquire_lock
from config import ALERTS_PATH
from config import CAN_TICKERS_PATH, CAN_TICKERS_ONE_LINE_PATH, CAN_TICKERS_REJECTED_PATH, SCREENER_OUT_PATH, \
    CAN_TICKERS_UNIVERSE_PATH, REPORT_PATH, CANDIDATES_QUEUE_PATH
from log_utils import log
from send_report import SendConfig, send_report
from swing_tickers import UniverseBuilderConfig, Thresholds, run_universe_builder


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
    # Initialize components
    data_manager = DataManager(CAN_TICKERS_PATH)
    screener = StockScreener(CONFIG, data_manager)

    # Run screening
    results = screener.run(force_refresh=True)

    # Display and save
    if not results.empty:
        display_results(results, CONFIG, data_manager)
        save_results(results, SCREENER_OUT_PATH)
    else:
        print(f"{Fore.RED}No results generated. Check your tickers and data connection.{Style.RESET_ALL}")


def __run_pipeline():
    cfg = PipelineConfig(
        shared_report_path=REPORT_PATH,
        candidates_queue_path=CANDIDATES_QUEUE_PATH
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
        # Prepare tickers list for swing trading:
        __build_swing_tickers()
        # Run the screener
        __run_stock_screener()
        # Run the pipeline
        __run_pipeline()
        # Send the report
        __run_send_report()
        log(service, run_id, "completed")
    finally:
        lock_file.close()
