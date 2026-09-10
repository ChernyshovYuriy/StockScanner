"""
Real market-data DataProvider, wrapping yfinance -- the seam Phase 1's
DataProvider interface left open (the source build spec explicitly scoped
this out of Phases 1-3: "define the DataProvider interface and ship one
stub/fake implementation for tests ... leave a clear seam where a real
(e.g. yfinance) provider drops in later"). Fetch convention (auto_adjust,
MultiIndex flattening, dropna, to_datetime index) mirrors
research/elder_ray.py's own fetch_bars() for consistency across the
research package -- reimplemented independently rather than imported,
since the two tools are otherwise unrelated.

A single ticker's fetch failing (bad symbol, network hiccup, delisted) does
NOT abort a batch run: get_bars() returns an empty-but-well-formed PriceData
for that ticker instead of raising, and every screen in this package already
treats too little data as a safe default (FLAT / no-pullback / not-fired --
see indicators.py), not an exception.
"""
import pandas as pd
import yfinance as yf

from .data_provider import DataProvider
from .types import AlignedBars, PriceData, TimeframeConfig

# yfinance interval string per TimeframeConfig's free-form timeframe label.
_YFINANCE_INTERVAL = {"daily": "1d", "weekly": "1wk", "hourly": "60m"}

_EMPTY_BARS = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])


def fetch_bars(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Mirrors research/elder_ray.py's fetch_bars() exactly. Never raises:
    any yfinance/network failure is caught and reported as empty bars,
    which every screen in this package already handles as insufficient
    data (see indicators.py's safe-default contract) rather than a crash.
    """
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    except Exception as e:
        print(f"WARNING: fetch failed for {ticker} ({interval}): {e}")
        return _EMPTY_BARS.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    df.index = pd.to_datetime(df.index)
    return df


class YFinanceDataProvider(DataProvider):
    """`trend_period`/`entry_period` are yfinance history windows (e.g.
    "2y", "1y") for the trend- and entry-timeframe fetches respectively --
    defaults match research/elder_ray.py's own WEEKLY_PERIOD / --period
    defaults, for the same reasoning (comfortable EMA-13 + lag warmup
    margin at each timeframe).
    """

    def __init__(self, trend_period: str = "2y", entry_period: str = "1y"):
        self.trend_period = trend_period
        self.entry_period = entry_period

    def get_bars(self, ticker: str, timeframes: TimeframeConfig = TimeframeConfig()) -> AlignedBars:
        trend_interval = _YFINANCE_INTERVAL.get(timeframes.trend_timeframe)
        entry_interval = _YFINANCE_INTERVAL.get(timeframes.entry_timeframe)
        if trend_interval is None or entry_interval is None:
            raise ValueError(
                f"YFinanceDataProvider has no yfinance interval mapping for "
                f"timeframe(s) {timeframes.trend_timeframe!r}/{timeframes.entry_timeframe!r} "
                f"-- known: {sorted(_YFINANCE_INTERVAL)}"
            )
        trend_bars = fetch_bars(ticker, self.trend_period, trend_interval)
        entry_bars = fetch_bars(ticker, self.entry_period, entry_interval)
        return AlignedBars(
            trend=PriceData(ticker=ticker, timeframe=timeframes.trend_timeframe, bars=trend_bars),
            entry=PriceData(ticker=ticker, timeframe=timeframes.entry_timeframe, bars=entry_bars),
        )
