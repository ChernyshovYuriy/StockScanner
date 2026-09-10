"""
Screen 1 = TrendScreen, Screen 2 = EntryScreen, Screen 3 = TriggerScreen.

Phase 1: interfaces only -- each defines what a screen consumes and
returns, not how. Phase 3's concrete indicator math lives in small, pure,
independently-testable functions that these classes call; the interface
itself stays indicator-agnostic.
"""
from abc import ABC, abstractmethod

from .types import Direction, EntryVerdict, PriceData, TrendVerdict, TriggerVerdict


class TrendScreen(ABC):
    """Screen 1 (Trend): consumes the longer timeframe (weekly, by default
    TimeframeConfig) and returns a trend verdict -- UP, DOWN, or FLAT.
    Reference method: slope of a 13-period EMA and/or the trend timeframe's
    MACD-histogram direction. The exact indicator is a documented, swappable
    parameter of the concrete implementation, not fixed by this interface.
    """

    @abstractmethod
    def evaluate(self, trend_bars: PriceData) -> TrendVerdict:
        """`trend_bars.timeframe` is whatever TimeframeConfig.trend_timeframe
        was for this run (default "weekly") -- this screen must not assume
        a specific label or bar spacing beyond what it's handed in
        `trend_bars`.

        Bars too short for this screen's own lookback are not this screen's
        job to escalate as an error (see DataProvider's docstring) -- return
        a defined verdict; the engine decides what a too-short verdict means
        for the final Signal.
        """
        raise NotImplementedError


class EntryScreen(ABC):
    """Screen 2 (Entry pullback): consumes the shorter timeframe (daily, by
    default) plus Screen 1's direction, and returns whether a counter-trend
    pullback is present -- for an UP trend, an oscillator (e.g. Force Index,
    or a short-period oscillator) in its oversold/dipped state; mirrored
    (overbought) for a DOWN trend. A FLAT trend has no defined pullback --
    see Phase 2's truth-table tests for the reviewed, exhaustive resolution.
    """

    @abstractmethod
    def evaluate(self, entry_bars: PriceData, trend: Direction) -> EntryVerdict:
        """`entry_bars.timeframe` is whatever TimeframeConfig.entry_timeframe
        was for this run (default "daily"). `trend` is Screen 1's verdict,
        needed because "pullback" only means something relative to a
        direction to pull back against.
        """
        raise NotImplementedError


class TriggerScreen(ABC):
    """Screen 3 (Trigger): consumes the shorter timeframe (daily, by
    default) plus Screen 1's direction, and returns whether the entry
    trigger has fired -- e.g. price crossing above the prior daily bar's
    high in an uptrend; mirrored (crossing below the prior low) for a
    downtrend. The reference implementation additionally requires the
    crossing to be confirmed by volume and by an indicator reaching a new
    extreme (Elder's true-vs-false-breakout distinction) before it counts
    as fired -- a bare price cross alone is not sufficient there; see
    indicators.prior_bar_breakout.
    """

    @abstractmethod
    def evaluate(self, entry_bars: PriceData, trend: Direction) -> TriggerVerdict:
        raise NotImplementedError
