"""
Batch layer: scan a list of tickers' full history (via market_data_cache's
existing OHLCV cache -- no separate yfinance wrapper needed, since Phase 1's
scope is fixed to the CAN_TICKERS_URL universe that cache already serves)
for every bullish Kangaroo Tail detect_kangaroo_tail()/scan_history() would
have fired, and print them as a table. Purely a Phase 1 sanity-check tool
for eyeballing detections against a real chart before trusting the
detector -- no persistence, no email (see __init__.py for phase scope).

    python -m research.kangaroo_tail.batch AAPL SLF.TO
    python -m research.kangaroo_tail.batch --start 2025-01-01 --end 2026-09-10
"""
import argparse
import urllib.request
from typing import Sequence

from tabulate import tabulate

from config import CAN_TICKERS_URL
from market_data_cache import sync_and_load
from time_utils import market_today_str
from .detector import confirms_next_bar, scan_history
from .types import TailConfig, TailSignal

# Default lookback window for the CLI: generous enough to clear
# TailConfig's default indicator warmup (ATR/lookback/volume_lookback, all
# <= 20 bars) many times over, so a short fetch never silently starves the
# detector of history.
_DEFAULT_LOOKBACK_DAYS = 400


def load_tickers(source: str) -> list[str]:
    """Load a ticker list from a URL (one ticker per line, '#' comments
    skipped) or a local file -- mirrors
    research/triple_screen/batch.py's own load_tickers() (reimplemented
    independently rather than imported, since the two research tools are
    otherwise unrelated)."""
    if source.startswith("http://") or source.startswith("https://"):
        with urllib.request.urlopen(source) as resp:
            content = resp.read().decode("utf-8")
        return [line.strip() for line in content.splitlines()
                if line.strip() and not line.strip().startswith("#")]
    with open(source, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]


def run_batch(tickers: Sequence[str], start: str, end: str,
              config: TailConfig = TailConfig()) -> tuple[dict[str, list[TailSignal]], dict]:
    """Fetch `tickers`' bars for [start, end] via the shared market data
    cache once, and return (every ticker's detected signals, the fetched
    bars themselves -- so a caller that also needs the raw bars, e.g. to
    check next-bar confirmation, doesn't have to fetch a second time). A
    ticker whose fetch or scan raises is skipped (printed as a WARNING)
    rather than aborting the whole batch -- same isolation guarantee as
    research/triple_screen/batch.py's run_batch(), for the same reason
    (one bad symbol shouldn't take down every other ticker's result).
    """
    results: dict[str, list[TailSignal]] = {}
    try:
        bars_by_ticker = sync_and_load(list(tickers), start=start, end=end)
    except Exception as e:
        print(f"WARNING: batch fetch failed ({type(e).__name__}: {e})")
        return results, {}
    for ticker in tickers:
        bars = bars_by_ticker.get(ticker)
        if bars is None or bars.empty:
            print(f"WARNING: skipping {ticker} (no data)")
            continue
        try:
            results[ticker] = scan_history(ticker, bars, config)
        except Exception as e:
            print(f"WARNING: skipping {ticker} ({type(e).__name__}: {e})")
    return results, bars_by_ticker


def format_signals_table(results: dict[str, list[TailSignal]],
                          bars_by_ticker: dict, config: TailConfig = TailConfig()) -> str:
    """Render every detected signal across all tickers as one table, most
    recent first, including whether the following bar (if already in the
    fetched window) confirmed under each of confirms_next_bar()'s two
    threshold variants -- so a Phase 1 eyeball pass can already see how
    same-day vs. next-bar-confirmed would have differed, ahead of Phase 2's
    proper walk-forward comparison.
    """
    rows = []
    for ticker, signals in results.items():
        bars = bars_by_ticker.get(ticker)
        for signal in signals:
            confirmed_high = confirmed_mid = None
            if bars is not None:
                later = bars.sort_index().loc[bars.sort_index().index > signal.date]
                if not later.empty:
                    next_bar = later.iloc[0]
                    confirmed_high = confirms_next_bar(signal, next_bar, use_midpoint=False)
                    confirmed_mid = confirms_next_bar(signal, next_bar, use_midpoint=True)
            rows.append({
                "ticker": ticker,
                "date": signal.date.date(),
                "close": round(signal.close, 2),
                "close_pos_%": round(signal.close_position_pct * 100, 1),
                "wick_%": round(signal.wick_pct * 100, 1),
                "break_atr": round(signal.break_amount / signal.atr, 2),
                "vol_ratio": round(signal.volume_ratio, 2) if signal.volume_ratio is not None else "-",
                "confirmed(high)": confirmed_high,
                "confirmed(mid)": confirmed_mid,
            })
    rows.sort(key=lambda r: (r["date"], r["ticker"]), reverse=True)
    return tabulate(rows, headers="keys", tablefmt="github", showindex=False)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Scan history for bullish Kangaroo Tail detections")
    parser.add_argument("tickers", nargs="*",
                         help="explicit tickers, e.g. AAPL SLF.TO (overrides --tickers-url)")
    parser.add_argument("--tickers-url", default=CAN_TICKERS_URL,
                         help="ticker-list URL or local file, one per line (default: config.CAN_TICKERS_URL)")
    parser.add_argument("--start", default=None,
                         help=f"history start date, YYYY-MM-DD (default: {_DEFAULT_LOOKBACK_DAYS} days back)")
    parser.add_argument("--end", default=None,
                         help="history end date, YYYY-MM-DD (default: today)")
    parser.add_argument("--limit", type=int, default=None,
                         help="only process the first N tickers (e.g. for a quick smoke run)")
    args = parser.parse_args(argv)

    end = args.end or market_today_str()
    start = args.start
    if start is None:
        import datetime
        start = (datetime.date.fromisoformat(end) - datetime.timedelta(days=_DEFAULT_LOOKBACK_DAYS)).isoformat()

    tickers = args.tickers or load_tickers(args.tickers_url)
    if args.limit is not None:
        tickers = tickers[:args.limit]
    print(f"{len(tickers)} tickers loaded "
          f"({'explicit args' if args.tickers else args.tickers_url}), "
          f"scanning {start} .. {end}")

    config = TailConfig()
    results, bars_by_ticker = run_batch(tickers, start, end, config)
    total = sum(len(sigs) for sigs in results.values())
    print(f"\n{total} Kangaroo Tail signal(s) found across {len(results)} ticker(s) with data\n")
    if total:
        print(format_signals_table(results, bars_by_ticker, config))


if __name__ == "__main__":
    main()
