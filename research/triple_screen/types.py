"""
Data types for the Triple Screen engine. Phase 1: structure only, no logic --
these are plain (frozen) containers and enums; every behavior lives in the
screens/engine interfaces (screens.py, engine.py) or their later
implementations.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

import pandas as pd


@dataclass(frozen=True)
class PriceData:
    """OHLCV bars for one ticker at one timeframe, sorted ascending by date.

    `timeframe` is an explicit, free-form label ("weekly", "daily", "hourly",
    ...) rather than something inferred from bar spacing -- screens key off
    this label, never off assumptions about bar count or spacing, so the same
    screen code works under a different TimeframeConfig mapping later.

    `bars` columns: Open, High, Low, Close, Volume (matching elder_ray.py's
    own yfinance-sourced convention); DatetimeIndex, ascending, no NaNs.
    """
    ticker: str
    timeframe: str
    bars: pd.DataFrame


@dataclass(frozen=True)
class TimeframeConfig:
    """Which timeframe label maps to Screen 1 (trend) vs. Screens 2/3 (entry
    + trigger). Default is the swing mapping this build targets (weekly
    trend / daily entry); the same two-field shape supports a later
    daily/hourly intraday mapping without any screen or engine change --
    only the labels move. DataProvider implementations key off these labels
    rather than hardcoding "weekly"/"daily" anywhere.
    """
    trend_timeframe: str = "weekly"
    entry_timeframe: str = "daily"


@dataclass(frozen=True)
class AlignedBars:
    """One ticker's bars for both Triple Screen timeframes, as returned by
    DataProvider.get_bars(). "Aligned" means both series come from the same
    ticker as of the same provider cutoff -- it does NOT mean the two
    different-frequency series are row-aligned to each other; that
    point-in-time alignment, where a screen needs it, is the screen's own
    concern (mirroring elder_ray.py's weekly_trend_by_date, which exists for
    exactly this reason: no lookahead into a still-forming trend-timeframe
    bar).
    """
    trend: PriceData
    entry: PriceData


class Direction(Enum):
    """Screen 1's verdict on the trend timeframe."""
    UP = "UP"
    DOWN = "DOWN"
    FLAT = "FLAT"


@dataclass(frozen=True)
class TrendVerdict:
    """Screen 1's result. `indicator_values` carries whatever numeric values
    produced `direction` (e.g. {"ema_slope": ..., "macd_hist": ...}) so the
    verdict is auditable, not just trusted -- required by the source spec's
    "every screen's verdict must be explainable" constraint.
    """
    direction: Direction
    indicator_values: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class EntryVerdict:
    """Screen 2's result: whether a counter-trend pullback is present."""
    pullback_present: bool
    indicator_values: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TriggerVerdict:
    """Screen 3's result: whether the entry trigger has fired."""
    triggered: bool
    indicator_values: Mapping[str, float] = field(default_factory=dict)


class Signal(Enum):
    """Final per-ticker Triple Screen result. See engine.py's module
    docstring for the exact semantics this encodes."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    WAIT = "WAIT"


@dataclass(frozen=True)
class SignalResult:
    """The final Signal plus every intermediate screen verdict, so a result
    can be audited without re-running the screens."""
    signal: Signal
    trend: TrendVerdict
    entry: EntryVerdict
    trigger: TriggerVerdict
