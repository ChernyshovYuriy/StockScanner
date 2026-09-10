"""
Reference (default) concrete implementations of the three screens plus
SignalEngine -- the "documented, swappable" default the Phase 1 interfaces
call for. A different indicator choice is a different class implementing
the same ABC, not a change to SignalEngine or the ABCs themselves.

These are thin adapters between indicators.py's pure math and the
Screen/SignalEngine ABCs; the ABCs require a class, so "prefer pure
functions over classes where a class isn't earning its place" doesn't argue
against them -- the math itself already lives in pure, independently
testable functions (indicators.py).
"""
from .engine import SignalEngine
from .indicators import ema_slope_direction, force_index_pullback, prior_bar_breakout
from .screens import EntryScreen, TrendScreen, TriggerScreen
from .types import (
    AlignedBars, Direction, EntryVerdict, PriceData, Signal, SignalResult, TrendVerdict, TriggerVerdict,
)


class EMASlopeTrendScreen(TrendScreen):
    """Screen 1 default: EMA slope (see indicators.ema_slope_direction)."""

    def __init__(self, ema_period: int = 13, lag: int = 3):
        self.ema_period = ema_period
        self.lag = lag

    def evaluate(self, trend_bars: PriceData) -> TrendVerdict:
        direction, indicator_values = ema_slope_direction(trend_bars.bars["Close"], self.ema_period, self.lag)
        return TrendVerdict(direction=direction, indicator_values=indicator_values)


class ForceIndexEntryScreen(EntryScreen):
    """Screen 2 default: 2-period Force Index (see indicators.force_index_pullback)."""

    def __init__(self, span: int = 2):
        self.span = span

    def evaluate(self, entry_bars: PriceData, trend: Direction) -> EntryVerdict:
        present, indicator_values = force_index_pullback(entry_bars.bars, trend, self.span)
        return EntryVerdict(pullback_present=present, indicator_values=indicator_values)


class PriorBarTriggerScreen(TriggerScreen):
    """Screen 3 default: prior-bar high/low breakout, confirmed by volume
    and indicator-extreme confirmation (see indicators.prior_bar_breakout)."""

    def __init__(self, volume_lookback: int = 20, volume_multiplier: float = 1.5,
                 momentum_period: int = 10, momentum_lookback: int = 20):
        self.volume_lookback = volume_lookback
        self.volume_multiplier = volume_multiplier
        self.momentum_period = momentum_period
        self.momentum_lookback = momentum_lookback

    def evaluate(self, entry_bars: PriceData, trend: Direction) -> TriggerVerdict:
        fired, indicator_values = prior_bar_breakout(
            entry_bars.bars, trend, self.volume_lookback, self.volume_multiplier,
            self.momentum_period, self.momentum_lookback)
        return TriggerVerdict(triggered=fired, indicator_values=indicator_values)


class TripleScreenEngine(SignalEngine):
    """Composes the three screens + position-awareness input per the exact
    truth table documented in engine.py's module docstring: an OPEN
    position's fate depends on trend alignment alone; a FLAT position's
    fate depends on the full three-screen conjunction alone.
    """

    def __init__(self, trend_screen: TrendScreen, entry_screen: EntryScreen, trigger_screen: TriggerScreen):
        self.trend_screen = trend_screen
        self.entry_screen = entry_screen
        self.trigger_screen = trigger_screen

    def evaluate(self, bars: AlignedBars, position_open: bool) -> SignalResult:
        trend = self.trend_screen.evaluate(bars.trend)
        entry = self.entry_screen.evaluate(bars.entry, trend.direction)
        trigger = self.trigger_screen.evaluate(bars.entry, trend.direction)

        if position_open:
            # trend-flip exit: SELL unconditionally the moment trend reads
            # DOWN, regardless of Screens 2/3 -- Screens 2/3 are an ENTRY
            # filter only (see engine.py docstring); HOLD otherwise (trend
            # still UP, or indecisively FLAT -- neither is a reversal).
            signal = Signal.SELL if trend.direction == Direction.DOWN else Signal.HOLD
        else:
            # only the full three-screen conjunction opens a new BUY;
            # everything else (including a fully-formed bearish setup --
            # long-only, nothing to short) is WAIT.
            if trend.direction == Direction.UP and entry.pullback_present and trigger.triggered:
                signal = Signal.BUY
            else:
                signal = Signal.WAIT

        return SignalResult(signal=signal, trend=trend, entry=entry, trigger=trigger)
