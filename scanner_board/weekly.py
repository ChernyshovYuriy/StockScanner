"""
scanner_board/weekly.py
=========================
Weekly OHLCV bars from daily bars — Phase 4, see scanner_board/PLAN.md.
Needed because Triple Screen/Impulse (Ch.39/40) are inherently
multi-timeframe: "perform weekly studies each day" (Ch.23) rather than a
second yfinance round-trip, resampled from the same cached daily bars
market_data_cache.py already maintains.

Single-shared-logic rule: canadian_stock_screener.TechnicalIndicators.
weekly_resample() already resamples a Close series to weekly bars AND
decides whether the most recent, still-in-progress week should be dropped
(fewer than 4 trading days contributed so far — see its own docstring).
That drop decision is not re-implemented here for Open/High/Low/Volume: this
module resamples all five OHLCV columns independently, then reindexes the
other four to exactly the set of week-endings weekly_resample() already
decided to keep for Close — so the "how many days makes a week real" rule
has exactly one implementation, not five.
"""
from __future__ import annotations

import pandas as pd

from canadian_stock_screener import TechnicalIndicators


def resample_ohlcv_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Weekly OHLCV bars from a daily OHLCV DataFrame (Open/High/Low/Close/
    Volume columns, ascending DatetimeIndex — the same shape
    market_data_cache.sync_and_load() returns): Open=first, High=max,
    Low=min, Close=last, Volume=sum, week-ending Friday. See module
    docstring for why the partial-current-week rule is applied only once.
    """
    close_weekly = TechnicalIndicators.weekly_resample(daily["Close"])
    open_weekly = daily["Open"].resample("W-FRI").first()
    high_weekly = daily["High"].resample("W-FRI").max()
    low_weekly = daily["Low"].resample("W-FRI").min()
    volume_weekly = daily["Volume"].resample("W-FRI").sum()
    return pd.DataFrame({
        "Open": open_weekly,
        "High": high_weekly,
        "Low": low_weekly,
        "Close": close_weekly,
        "Volume": volume_weekly,
    }).reindex(close_weekly.index)
