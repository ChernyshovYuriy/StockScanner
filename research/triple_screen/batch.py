"""
Batch layer: run the Triple Screen engine across a list of tickers via a
DataProvider, returning a structured per-ticker result (signal plus all
three intermediate screen verdicts, so a result can be audited, not just
trusted), plus a CLI entry point that pulls the ticker list from
CAN_TICKERS_URL (or an override), runs the engine against real yfinance data
(yfinance_provider.YFinanceDataProvider), and prints the results as a table
(tabulate, matching research/elder_ray.py's own console-output convention).

run_batch()/format_results_table() are provider-agnostic (tested against a
fake provider -- see tests/test_triple_screen_batch.py); the CLI's own
provider choice (yfinance_provider.py) is the concrete seam the source build
spec's Constraints left open for "later" -- now wired in.
"""
import argparse
import urllib.request
from typing import Mapping, Sequence

from tabulate import tabulate

from config import CAN_TICKERS_URL
from .data_provider import DataProvider
from .engine import SignalEngine
from .reference_impl import EMASlopeTrendScreen, ForceIndexEntryScreen, PriorBarTriggerScreen, TripleScreenEngine
from .types import SignalResult, TimeframeConfig
from .yfinance_provider import YFinanceDataProvider


def load_tickers(source: str) -> list[str]:
    """Load a ticker list from a URL (one ticker per line, '#' comments
    skipped) or a local file -- mirrors canadian_stock_screener.py's
    DataManager._load_tickers() convention, so CAN_TICKERS_URL (the repo's
    single source of truth for the TSX universe, config.py) works here
    exactly as it does in main.py/canadian_stock_screener.py. This is only
    the ticker *list* -- a separate concern from the OHLCV DataProvider,
    which stays out of scope (see module docstring).
    """
    if source.startswith("http://") or source.startswith("https://"):
        with urllib.request.urlopen(source) as resp:
            content = resp.read().decode("utf-8")
        return [line.strip() for line in content.splitlines()
                if line.strip() and not line.strip().startswith("#")]
    with open(source, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]


def build_default_engine() -> TripleScreenEngine:
    """The reference engine, wired with the three default screen implementations."""
    return TripleScreenEngine(
        trend_screen=EMASlopeTrendScreen(),
        entry_screen=ForceIndexEntryScreen(),
        trigger_screen=PriorBarTriggerScreen(),
    )


def run_batch(tickers: Sequence[str], provider: DataProvider, engine: SignalEngine,
              positions: Mapping[str, bool] | None = None,
              timeframes: TimeframeConfig = TimeframeConfig()) -> dict[str, SignalResult]:
    """Evaluate `engine` for every ticker in `tickers`, fetching each one's
    bars from `provider`. `positions` maps ticker -> currently-open (True)
    or flat (False); a ticker missing from `positions` defaults to flat.
    """
    positions = positions or {}
    results: dict[str, SignalResult] = {}
    for ticker in tickers:
        bars = provider.get_bars(ticker, timeframes)
        results[ticker] = engine.evaluate(bars, position_open=positions.get(ticker, False))
    return results


# Actionable-first ordering: new entries, then exits, then an active
# position needing no action, then nothing happening.
_SIGNAL_SORT_ORDER = {"BUY": 0, "SELL": 1, "HOLD": 2, "WAIT": 3}


def format_results_table(results: Mapping[str, SignalResult]) -> str:
    """Render a per-ticker results map as a printable table -- the final
    signal plus all three intermediate verdicts, so the output itself is
    auditable rather than a bare signal column. Rows are grouped by signal
    (BUY, SELL, HOLD, WAIT, in that order); within a group, tickers keep
    their original `results` order.
    """
    rows = [{
        "ticker": ticker,
        "signal": result.signal.value,
        "trend": result.trend.direction.value,
        "pullback": result.entry.pullback_present,
        "trigger": result.trigger.triggered,
    } for ticker, result in results.items()]
    rows.sort(key=lambda row: _SIGNAL_SORT_ORDER[row["signal"]])
    return tabulate(rows, headers="keys", tablefmt="github", showindex=False)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Elder Triple Screen signals for a list of tickers")
    parser.add_argument("tickers", nargs="*",
                         help="explicit tickers, e.g. AAPL MSFT SLF.TO (overrides --tickers-url)")
    parser.add_argument("--tickers-url", default=CAN_TICKERS_URL,
                         help="ticker-list URL or local file, one per line (default: config.CAN_TICKERS_URL)")
    parser.add_argument("--limit", type=int, default=None,
                         help="only process the first N tickers (e.g. for a quick smoke run)")
    args = parser.parse_args(argv)

    tickers = args.tickers or load_tickers(args.tickers_url)
    if args.limit is not None:
        tickers = tickers[:args.limit]
    print(f"{len(tickers)} tickers loaded "
          f"({'explicit args' if args.tickers else args.tickers_url})")

    engine = build_default_engine()
    provider = YFinanceDataProvider()
    results = run_batch(tickers, provider, engine)
    print()
    print(format_results_table(results))


if __name__ == "__main__":
    main()
