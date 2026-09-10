"""
DataProvider interface for the Triple Screen engine. Phase 1: interface
only, no implementation. A real market-data implementation (e.g. wrapping
yfinance, matching elder_ray.py's own fetch_bars) is explicitly out of scope
for now and is a separate, later module -- this interface is the seam it
drops into. Phase 2's tests use a fake/stub implementation instead, so no
test in this package ever makes a network call.
"""
from abc import ABC, abstractmethod

from .types import AlignedBars, TimeframeConfig


class DataProvider(ABC):
    """Given a ticker, returns its bars for both Triple Screen timeframes,
    aligned to each other (same ticker, same as-of cutoff -- see
    AlignedBars). Concrete implementations own how they fetch/cache/rate-
    limit; nothing else in this package fetches data itself -- screens and
    the engine only ever see bars handed to them (no implicit data fetching
    inside a screen, no lookahead).
    """

    @abstractmethod
    def get_bars(self, ticker: str, timeframes: TimeframeConfig = TimeframeConfig()) -> AlignedBars:
        """Return `ticker`'s bars for `timeframes.trend_timeframe` and
        `timeframes.entry_timeframe`, each sorted ascending, with nothing
        dated after this provider's notion of "now".

        What a concrete implementation raises for "ticker not found" /
        "fetch failed" is its own concern. A short-but-otherwise-valid
        result (too few bars for a screen's lookback) is NOT this
        provider's job to pre-validate -- the engine (Phase 3) is
        responsible for turning an insufficient-data screen result into
        Signal.WAIT, the same safe-default convention elder_ray.py already
        uses for a too-short window.
        """
        raise NotImplementedError
