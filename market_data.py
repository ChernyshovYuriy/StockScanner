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

Beyond daily OHLCV (get/download), the provider also centralizes the three
other shapes of market data live code needs — a latest quote, an intraday
snapshot, and a sector classification — so every live call site goes through
one abstraction instead of each calling yfinance directly with its own
parameters and its own cache (or none). See get_quote(), get_intraday_snapshot(),
get_sector() below.
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf
from colorama import Fore, Style

from config import CACHE_PATH

warnings.filterwarnings("ignore")


@dataclass
class TodayBar:
    """Live intraday snapshot for the current session."""
    low: float
    close: float  # latest traded price (last 5-min close)
    high: float  # session high so far
    source: str  # e.g. "5m-intraday"


# ─────────────────────────────────────────────────────────────────────────────
# OHLCV CONTRACT  — the internal data structure the rest of the codebase
# relies on. Not yfinance's shape by inheritance: an explicitly owned,
# validated contract that yfinance's own output happens to already satisfy.
# Any future provider adapter is "done" when its output passes validate_ohlcv().
# ─────────────────────────────────────────────────────────────────────────────

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce the internal OHLCV contract: a DatetimeIndex and float
    Open/High/Low/Close/Volume columns, in that order.

    An empty DataFrame always passes through unchanged — many callers treat
    "no data" as a valid, explicitly-checked state, not an error.

    Raises ValueError on a genuine contract violation (missing column, or a
    non-DatetimeIndex). In practice this should never fire for real yfinance
    data, since every LiveDataProvider/HistoricalSliceProvider method already
    slices to exactly these columns before returning — it exists as the
    acceptance test a *different* future provider's adapter would have to
    pass.
    """
    if df.empty:
        return df
    missing = set(OHLCV_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"OHLCV contract violation: missing columns {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("OHLCV contract violation: index must be a DatetimeIndex")
    return df[OHLCV_COLUMNS].astype(float)


# ─────────────────────────────────────────────────────────────────────────────
# SECTOR CACHE  (shared by LiveDataProvider.get_sector and
# HistoricalSliceProvider.get_sector — sector classifications change rarely
# and aren't tied to a backtest as_of cutoff the way price bars are)
# ─────────────────────────────────────────────────────────────────────────────

UNKNOWN_SECTOR = "Unknown"
_SECTOR_CACHE_FILE = Path(CACHE_PATH) / "sector_cache.json"


def _load_sector_cache() -> Dict[str, str]:
    try:
        return json.loads(_SECTOR_CACHE_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_sector_cache(cache: Dict[str, str]) -> None:
    try:
        _SECTOR_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _SECTOR_CACHE_FILE.write_text(json.dumps(cache, indent=1, sort_keys=True))
    except OSError:
        pass  # best-effort — a failed write just means next run re-fetches


_sector_cache: Dict[str, str] = _load_sector_cache()


def _get_sector_cached(ticker: str) -> str:
    """Return the GICS sector for ticker (cached), fetching via yfinance on a
    cache miss. Returns UNKNOWN_SECTOR if the lookup fails or the ticker has
    no sector (e.g. an ETF) — treated as its own bucket by callers, so
    unclassified tickers are still capped among themselves rather than
    bypassing the cap entirely."""
    ticker = ticker.upper()
    if ticker in _sector_cache:
        return _sector_cache[ticker]
    sector = UNKNOWN_SECTOR
    try:
        info = yf.Ticker(ticker).info
        sector = info.get("sector") or UNKNOWN_SECTOR
    except Exception:
        pass
    _sector_cache[ticker] = sector
    _save_sector_cache(_sector_cache)
    return sector

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

    def get_quote(self, ticker: str) -> Optional[float]:
        """Return the latest available price for ticker, or None on failure."""
        raise NotImplementedError

    def get_intraday_snapshot(self, ticker: str) -> Optional["TodayBar"]:
        """Return today's session low/close/high, or None on failure /
        unavailable (e.g. a historical provider, which has no "today")."""
        raise NotImplementedError

    def get_sector(self, ticker: str) -> str:
        """Return the GICS sector for ticker, or UNKNOWN_SECTOR."""
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
    def get(
        self,
        ticker: str,
        as_of: pd.Timestamp,
        start_dt: Optional[str] = None,
        end_dt: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Download a single ticker.  as_of is ignored in live mode.

        start_dt / end_dt (ISO date strings) are an extra, opt-in override for
        callers that need an explicit date range (e.g. an exit computation
        re-fetching from a specific entry date) rather than the default 2y
        lookback — same override pattern already used by download().
        """
        if start_dt is not None or end_dt is not None:
            raw = yf.download(
                ticker,
                start=start_dt,
                end=end_dt,
                interval="1d",
                auto_adjust=True,
                progress=False,
                group_by="column",
            )
        else:
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
        return validate_ohlcv(raw[["Open", "High", "Low", "Close", "Volume"]].dropna())

    # ------------------------------------------------------------------
    # get_quote() — latest available price (~15-min delayed quote)
    # ------------------------------------------------------------------
    def get_quote(self, ticker: str) -> Optional[float]:
        """
        Fetch the latest available market price for a ticker via Yahoo Finance.

        Strategy (in order of preference):
          1. yf.Ticker.fast_info["last_price"]  — fastest, returns the most
             recent delayed quote (~15 min) directly without downloading
             OHLCV bars.
          2. Fallback: download 1-minute bars for the last 1 trading day and
             take the last bar's Close — useful outside regular hours when
             fast_info may return None.

        This intentionally does NOT use daily bars so the price reflects the
        current session, not yesterday's close.
        """
        t = yf.Ticker(ticker)

        try:
            price = t.fast_info["last_price"]
            if price is not None and float(price) > 0:
                return float(price)
        except Exception:
            pass

        try:
            df = yf.download(
                tickers=ticker,
                period="1d",
                interval="1m",
                auto_adjust=True,
                progress=False,
            )
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                close = df["Close"].dropna()
                if not close.empty:
                    return float(close.iloc[-1])
        except Exception as e:
            print(f"  {Fore.RED}{ticker}: fallback download error — {e}{Style.RESET_ALL}")

        return None

    # ------------------------------------------------------------------
    # get_intraday_snapshot() — used during pre-close monitor run
    # ------------------------------------------------------------------
    def get_intraday_snapshot(self, ticker: str) -> Optional[TodayBar]:
        """
        Fetch today's 5-min bars and return a TodayBar with:
          - low   : the session low so far  (used for stop-hit check)
          - close : the latest 5-min close  (used for PnL / giveback)
          - high  : the session high so far

        Returns None on any failure; caller falls back to completed daily bar.
        """
        try:
            df = yf.download(
                tickers=ticker,
                period="1d",
                interval="5m",
                auto_adjust=True,
                progress=False,
            )
            if df is None or df.empty:
                return None

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.dropna(subset=["High", "Low", "Close"])
            if df.empty:
                return None

            return TodayBar(
                low=float(df["Low"].min()),
                close=float(df["Close"].iloc[-1]),
                high=float(df["High"].max()),
                source="5m-intraday",
            )

        except Exception as e:
            print(f"    {Fore.YELLOW}Intraday snapshot failed for {ticker}: {e}{Style.RESET_ALL}")
            return None

    # ------------------------------------------------------------------
    # get_sector() — GICS sector classification (cached to disk)
    # ------------------------------------------------------------------
    def get_sector(self, ticker: str) -> str:
        return _get_sector_cached(ticker)

    # ------------------------------------------------------------------
    # download_batch_with_reasons() — used by swing_tickers.py's universe
    # builder, which needs to know *why* a ticker was excluded (not just
    # whether it loaded) for its rejected-tickers report.
    # ------------------------------------------------------------------
    def download_batch_with_reasons(
        self, tickers: List[str], period: str, interval: str, auto_adjust: bool,
    ) -> "tuple[Dict[str, pd.DataFrame], Dict[str, str]]":
        """
        Batch-fetch OHLCV for tickers, fully normalizing yfinance's raw
        MultiIndex/single-ticker-collapse shape into validated per-ticker
        OHLCV DataFrames — the yfinance-shape awareness (MultiIndex checks,
        the single-ticker collapse yfinance does for a batch of one) stays
        contained here instead of leaking to the caller.

        Returns (data, reasons):
          data    : {ticker: validated OHLCV DataFrame} for tickers that loaded
          reasons : {ticker: "no_data" | "missing_ohlcv" | "all_nan_close"}
                    for any requested ticker NOT in data.

        Live-only: never called by the backtester.
        """
        raw = yf.download(
            tickers=tickers, period=period, interval=interval,
            auto_adjust=auto_adjust, group_by="ticker", threads=True, progress=False,
        )
        data: Dict[str, pd.DataFrame] = {}
        reasons: Dict[str, str] = {}

        if isinstance(raw.columns, pd.MultiIndex):
            for sym in tickers:
                if sym not in raw.columns.get_level_values(0):
                    reasons[sym] = "no_data"
                    continue
                sub = raw[sym].dropna(how="all")
                if sub.empty:
                    reasons[sym] = "no_data"
                    continue
                sub.index = pd.to_datetime(sub.index).tz_localize(None)
                data[sym] = validate_ohlcv(sub[OHLCV_COLUMNS])
        else:
            # yfinance collapses a single-ticker batch to plain columns
            sym = tickers[0]
            sub = raw.dropna(how="all")
            if sub.empty or not set(OHLCV_COLUMNS).issubset(set(sub.columns)):
                reasons[sym] = "missing_ohlcv"
            elif sub["Close"].dropna().empty:
                reasons[sym] = "all_nan_close"
            else:
                sub.index = pd.to_datetime(sub.index).tz_localize(None)
                data[sym] = validate_ohlcv(sub[OHLCV_COLUMNS])

        return data, reasons

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
                            data[ticker] = validate_ohlcv(df)
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
        # pd.Timestamp as_of values are unambiguous, and enforce the OHLCV
        # contract once here (get()/download() only slice rows out of
        # already-stored data, so this is the one place construction-time
        # data needs to satisfy it).
        self._data: Dict[str, pd.DataFrame] = {}
        for ticker, df in data.items():
            df = df.copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            self._data[ticker] = validate_ohlcv(df.sort_index())

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
    # get_quote() — backtest parity: last close at-or-before "now"
    # ------------------------------------------------------------------
    def get_quote(self, ticker: str) -> Optional[float]:
        """Return the most recent close in the preloaded dataset (no as_of
        cutoff is available here — callers needing a specific cutoff should
        use get()/download() with an explicit as_of instead)."""
        df = self._data.get(ticker)
        if df is None or df.empty:
            return None
        close = df["Close"].dropna()
        if close.empty:
            return None
        return float(close.iloc[-1])

    # ------------------------------------------------------------------
    # get_intraday_snapshot() — not applicable to pre-loaded daily bars
    # ------------------------------------------------------------------
    def get_intraday_snapshot(self, ticker: str) -> Optional[TodayBar]:
        """The backtester works off daily bars only; there is no "today"
        intraday snapshot for historical data. Always returns None so
        callers fall back to the completed daily bar, same as a live failure."""
        return None

    # ------------------------------------------------------------------
    # get_sector() — shares the same disk cache as LiveDataProvider
    # ------------------------------------------------------------------
    def get_sector(self, ticker: str) -> str:
        return _get_sector_cached(ticker)

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


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT PROVIDER  — the one instance every live call site defaults to
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PROVIDER = LiveDataProvider()
