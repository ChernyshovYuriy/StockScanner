"""
market_data.py
==============
Market data provider abstraction for the backtest refactor (Phase 2).

Introduces three things:

  MarketDataProvider  — Protocol (structural interface) that both providers
                        must satisfy.  Any object with a matching .download()
                        and .get() signature works — no inheritance required.

  LiveDataProvider    — Wraps the existing yfinance download logic.
                        Used by live / paper-trading mode.
                        Behaviour is identical to the original DataManager
                        download path; this is a pure extraction, not a rewrite.

  HistoricalSliceProvider — Holds a pre-loaded {ticker: DataFrame} dataset
                        (covering the full backtest date range) and serves
                        per-ticker slices truncated at a given as_of date.
                        This is the key lookahead-bias guard: callers can only
                        see data that would have been available at as_of.

Usage — live mode (no change to existing behaviour):
    provider = LiveDataProvider()
    data = provider.download(tickers, days=504)

Usage — backtest mode:
    # Pre-load once for the full date range
    provider = HistoricalSliceProvider.from_yfinance(
        tickers, start="2022-01-01", end="2025-01-01"
    )
    # Per simulated day, get a cutoff-respecting slice
    data = provider.download(tickers, days=504, as_of=pd.Timestamp("2024-06-15"))

DataManager integration:
    DataManager(tickers_file, provider=HistoricalSliceProvider(...))

Non-goals:
  - This module does NOT change any scoring, pattern detection, or sizing logic.
  - It does NOT replace the yfinance dependency for live mode.
  - It does NOT alter how DataManager processes or filters tickers after download.
"""

from __future__ import annotations

import time
import warnings
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PROTOCOL — structural interface
# ─────────────────────────────────────────────────────────────────────────────

class MarketDataProvider:
    """
    Structural interface for market data providers.

    Both LiveDataProvider and HistoricalSliceProvider implement these two
    methods.  Type checkers and tests reference this class; runtime code uses
    duck-typing so no inheritance is required.

    Methods
    -------
    get(ticker, as_of) → pd.DataFrame
        Return OHLCV for a single ticker, containing only bars up to and
        including as_of.  Raises KeyError if ticker is unknown.

    download(tickers, days, as_of) → Dict[str, pd.DataFrame]
        Return a {ticker: DataFrame} dict for all requested tickers.
        Each DataFrame contains only bars that would have been available
        at as_of (or today, for live mode).
        Tickers that fail to load are silently omitted (same as original).
    """

    def get(self, ticker: str, as_of: pd.Timestamp) -> pd.DataFrame:
        raise NotImplementedError

    def download(
        self,
        tickers: List[str],
        days: int,
        as_of: Optional[pd.Timestamp] = None,
    ) -> Dict[str, pd.DataFrame]:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# LIVE DATA PROVIDER
# ─────────────────────────────────────────────────────────────────────────────

class LiveDataProvider(MarketDataProvider):
    """
    Fetches data from Yahoo Finance in real time.

    This is an extraction of the original DataManager.download_data() network
    path.  No logic has been changed — batch size, sleep, quality checks, and
    column normalisation are all preserved exactly.

    The as_of parameter is accepted but ignored: live mode always fetches up
    to today.  This keeps the interface consistent with HistoricalSliceProvider
    so callers never need to branch on provider type.
    """

    def __init__(self, batch_size: int = 30, sleep_seconds: float = 0.5):
        self.batch_size   = batch_size
        self.sleep_seconds = sleep_seconds

    # ------------------------------------------------------------------
    # get() is less useful for live mode (always returns full history)
    # but must exist to satisfy the interface.
    # ------------------------------------------------------------------
    def get(self, ticker: str, as_of: pd.Timestamp) -> pd.DataFrame:
        """Download a single ticker.  as_of is ignored in live mode."""
        raw = yf.download(
            ticker,
            period="2y",
            auto_adjust=True,
            progress=False,
            timeout=10,
        )
        if raw is None or raw.empty:
            raise KeyError(f"LiveDataProvider: no data returned for {ticker!r}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        return raw[["Open", "High", "Low", "Close", "Volume"]].dropna()

    # ------------------------------------------------------------------
    # download() — batch fetch, identical behaviour to original
    # ------------------------------------------------------------------
    def download(
        self,
        tickers: List[str],
        days: int,
        as_of: Optional[pd.Timestamp] = None,
        start_dt: Optional[str] = None,
        end_dt: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Download OHLCV for all tickers in batches.

        Parameters
        ----------
        tickers     : full list including benchmark
        days        : lookback window in calendar days
        as_of       : accepted but ignored (live mode always fetches to today)
        start_dt    : override start date (ISO string); derived from days if None
        end_dt      : override end date (ISO string); defaults to today
        """
        from time_utils import market_today, date_to_iso_extended
        from datetime import timedelta

        if end_dt is None:
            end   = market_today()
            end_dt = date_to_iso_extended(end)
        if start_dt is None:
            from time_utils import market_today
            start = market_today() - timedelta(days=days + 60)
            start_dt = date_to_iso_extended(start)

        data: Dict[str, pd.DataFrame] = {}
        failed: List[str] = []

        for i in range(0, len(tickers), self.batch_size):
            batch = tickers[i : i + self.batch_size]
            try:
                raw = yf.download(
                    batch,
                    start=start_dt,
                    end=end_dt,
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                    timeout=10,
                )
                for ticker in batch:
                    try:
                        if isinstance(raw.columns, pd.MultiIndex):
                            df = pd.DataFrame({
                                "Open":   raw["Open"][ticker],
                                "High":   raw["High"][ticker],
                                "Low":    raw["Low"][ticker],
                                "Close":  raw["Close"][ticker],
                                "Volume": raw["Volume"][ticker],
                            }).dropna()
                        else:
                            if ticker != batch[0]:
                                continue
                            df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()

                        # Same quality gate as original DataManager
                        if len(df) > 200 and df["Close"].iloc[-1] > 0:
                            data[ticker] = df
                        else:
                            failed.append(ticker)
                    except Exception:
                        failed.append(ticker)
            except Exception:
                failed.extend(batch)

            time.sleep(self.sleep_seconds)

        return data


# ─────────────────────────────────────────────────────────────────────────────
# HISTORICAL SLICE PROVIDER
# ─────────────────────────────────────────────────────────────────────────────

class HistoricalSliceProvider(MarketDataProvider):
    """
    Pre-loaded historical data provider that enforces a strict as_of cutoff.

    The full OHLCV dataset for the entire backtest date range is loaded once
    at construction time.  Every subsequent call to get() or download() returns
    only bars with index <= as_of, preventing any form of lookahead bias.

    Construction
    ------------
    # From a pre-built dict (tests, or when you already have the data):
    provider = HistoricalSliceProvider({"RY.TO": df_ry, "XIU.TO": df_xiu})

    # From Yahoo Finance (convenience constructor for backtest runner):
    provider = HistoricalSliceProvider.from_yfinance(
        tickers=["RY.TO", "TD.TO", "XIU.TO"],
        start="2022-01-01",
        end="2025-01-01",
    )
    """

    def __init__(self, data: Dict[str, pd.DataFrame]):
        """
        Parameters
        ----------
        data : dict mapping ticker → full-history DataFrame (DatetimeIndex).
               The DatetimeIndex must be timezone-naive or consistently tz-aware.
        """
        # Normalise all indexes to tz-naive UTC midnight so comparisons with
        # pd.Timestamp as_of values are unambiguous.
        self._data: Dict[str, pd.DataFrame] = {}
        for ticker, df in data.items():
            df = df.copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            self._data[ticker] = df.sort_index()

    # ------------------------------------------------------------------
    # CLASS METHOD — convenience constructor from Yahoo Finance
    # ------------------------------------------------------------------
    @classmethod
    def from_yfinance(
        cls,
        tickers: List[str],
        start: str,
        end: str,
        batch_size: int = 30,
        sleep_seconds: float = 0.5,
    ) -> "HistoricalSliceProvider":
        """
        Pre-fetch the full date range from Yahoo Finance once and return a
        HistoricalSliceProvider backed by that data.

        This should be called once before the backtest loop starts, not inside
        the per-day simulation.
        """
        data: Dict[str, pd.DataFrame] = {}
        failed: List[str] = []

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i : i + batch_size]
            try:
                raw = yf.download(
                    batch,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                    timeout=10,
                )
                for ticker in batch:
                    try:
                        if isinstance(raw.columns, pd.MultiIndex):
                            df = pd.DataFrame({
                                "Open":   raw["Open"][ticker],
                                "High":   raw["High"][ticker],
                                "Low":    raw["Low"][ticker],
                                "Close":  raw["Close"][ticker],
                                "Volume": raw["Volume"][ticker],
                            }).dropna()
                        else:
                            if ticker != batch[0]:
                                continue
                            df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()

                        if not df.empty and df["Close"].iloc[-1] > 0:
                            data[ticker] = df
                        else:
                            failed.append(ticker)
                    except Exception:
                        failed.append(ticker)
            except Exception:
                failed.extend(batch)

            time.sleep(sleep_seconds)

        if failed:
            print(f"  [HistoricalSliceProvider] failed to load: {failed[:5]}"
                  f"{'...' if len(failed) > 5 else ''}")

        return cls(data)

    # ------------------------------------------------------------------
    # get() — single ticker, strictly cutoff-respecting
    # ------------------------------------------------------------------
    def get(self, ticker: str, as_of: pd.Timestamp) -> pd.DataFrame:
        """
        Return all bars for ticker with index <= as_of.

        Parameters
        ----------
        ticker : Yahoo Finance symbol (case-sensitive, must be in dataset)
        as_of  : cutoff timestamp; only bars on or before this date are returned

        Raises
        ------
        KeyError if ticker was not present in the dataset at construction time.
        """
        if ticker not in self._data:
            raise KeyError(
                f"HistoricalSliceProvider: ticker {ticker!r} not in dataset. "
                f"Available: {list(self._data.keys())[:10]}"
            )
        df   = self._data[ticker]
        # Normalise as_of to tz-naive for consistent comparison
        cutoff = pd.Timestamp(as_of).tz_localize(None) if getattr(as_of, "tzinfo", None) else pd.Timestamp(as_of)
        return df.loc[df.index <= cutoff]

    # ------------------------------------------------------------------
    # download() — bulk fetch used by DataManager
    # ------------------------------------------------------------------
    def download(
        self,
        tickers: List[str],
        days: int,
        as_of: Optional[pd.Timestamp] = None,
        **_kwargs,
    ) -> Dict[str, pd.DataFrame]:
        """
        Return {ticker: DataFrame} for all requested tickers, each sliced to
        as_of and further limited to the last `days` calendar days of that slice.

        Tickers missing from the dataset are silently skipped (same behaviour as
        LiveDataProvider and the original DataManager).

        Parameters
        ----------
        tickers : list of tickers to return (subset of the preloaded dataset)
        days    : lookback window; only the tail of each slice is returned
        as_of   : hard cutoff — no bar after this date will be included
        """
        from datetime import timedelta

        if as_of is None:
            # Fallback: use the latest bar available across all tickers
            as_of = max(
                (df.index.max() for df in self._data.values() if not df.empty),
                default=pd.Timestamp.now(),
            )

        cutoff = pd.Timestamp(as_of).tz_localize(None) if getattr(as_of, "tzinfo", None) else pd.Timestamp(as_of)
        earliest = cutoff - timedelta(days=days + 60)

        result: Dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            if ticker not in self._data:
                continue
            df = self._data[ticker]
            sliced = df.loc[(df.index >= earliest) & (df.index <= cutoff)]
            # Apply the same quality gate as LiveDataProvider / original DataManager
            if len(sliced) > 200 and not sliced.empty and sliced["Close"].iloc[-1] > 0:
                result[ticker] = sliced

        return result

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @property
    def tickers(self) -> List[str]:
        """All tickers available in this dataset."""
        return list(self._data.keys())

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return (f"HistoricalSliceProvider("
                f"{len(self._data)} tickers, "
                f"dates={self._date_range()})")

    def _date_range(self) -> str:
        dates = [df.index for df in self._data.values() if not df.empty]
        if not dates:
            return "empty"
        mn = min(d.min() for d in dates).date()
        mx = max(d.max() for d in dates).date()
        return f"{mn} → {mx}"
