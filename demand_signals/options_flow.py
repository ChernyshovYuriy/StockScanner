"""
Options-flow proxy: a pluggable Provider interface (one free implementation
to start) deriving an "unusual activity" signal from a public options chain
snapshot -- volume/open-interest ratio and call/put volume skew.

HONEST CEILING: this is a SNAPSHOT of the current chain (volume/OI/bid-ask
at request time), not a trade-by-trade tape -- no sweep or block-trade
detection. See demand_signals/__init__.py's data-limitations note. A paid
feed (e.g. Unusual Whales) plugs in as a second OptionsFlowProvider
implementation without touching callers -- that's the point of the
abstraction: everything downstream only ever sees an OptionsSnapshot.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

try:
    from config import (
        DEMAND_CACHE_PATH as CACHE_PATH,
        DEMAND_OPTIONS_UNUSUAL_VOL_OI_RATIO as UNUSUAL_RATIO,
    )
except Exception:
    CACHE_PATH = Path(__file__).resolve().parent.parent / "demand_signals_cache"
    UNUSUAL_RATIO = 2.0

from demand_signals.http_cache import DiskCache, cache_date
from demand_signals.schema import DemandSignal


@dataclass
class OptionsSnapshot:
    """One expiration's chain summary for one ticker, as a Provider returns it."""

    us_ticker: str
    as_of_date: str
    call_volume: int
    put_volume: int
    call_oi: int
    put_oi: int
    legs: list = field(default_factory=list)  # per-strike detail, kept as evidence


class OptionsFlowProvider(ABC):
    """Pluggable options-data source. Swap YahooOptionsProvider for a paid
    vendor's implementation without touching callers -- build_signals() and
    everything upstream only ever sees an OptionsSnapshot."""

    @abstractmethod
    def snapshot(self, us_ticker: str) -> Optional[OptionsSnapshot]:
        """Today's chain summary for us_ticker, or None if unavailable."""
        ...


class YahooOptionsProvider(OptionsFlowProvider):
    """Free options-chain snapshot via yfinance (already a dependency here,
    already used by market_data.LiveDataProvider). yfinance manages its own
    HTTP session/throttling internally, so this caches the parsed *result*
    rather than a raw HTTP response -- that boundary isn't exposed to us."""

    def __init__(self, cache_dir: Path = None):
        self.cache = DiskCache(cache_dir if cache_dir is not None else Path(CACHE_PATH))

    def snapshot(self, us_ticker: str) -> Optional[OptionsSnapshot]:
        cache_key = f"yahoo_options_{us_ticker}_{cache_date()}.json"
        cached = self.cache.get(cache_key)
        if cached is not None:
            data = json.loads(cached)
        else:
            data = self._fetch(us_ticker)
            if data is None:
                return None
            self.cache.set(cache_key, json.dumps(data))
        return self._to_snapshot(us_ticker, data)

    def _fetch(self, us_ticker: str) -> Optional[dict]:
        import yfinance as yf

        t = yf.Ticker(us_ticker)
        expirations = t.options
        if not expirations:
            return None
        # Nearest expiration only: the closest-dated chain carries the
        # freshest positioning; farther-dated chains are noisier for a
        # same-day "unusual activity" read.
        chain = t.option_chain(expirations[0])
        cols = ["volume", "openInterest", "strike"]
        return {
            "expiration": expirations[0],
            "calls": chain.calls[cols].fillna(0).to_dict("records"),
            "puts": chain.puts[cols].fillna(0).to_dict("records"),
        }

    @staticmethod
    def _to_snapshot(us_ticker: str, data: dict) -> OptionsSnapshot:
        calls, puts = data["calls"], data["puts"]
        return OptionsSnapshot(
            us_ticker=us_ticker,
            as_of_date=date.today().isoformat(),
            call_volume=int(sum(c["volume"] for c in calls)),
            put_volume=int(sum(p["volume"] for p in puts)),
            call_oi=int(sum(c["openInterest"] for c in calls)),
            put_oi=int(sum(p["openInterest"] for p in puts)),
            legs=[{"type": "call", **c} for c in calls] + [{"type": "put", **p} for p in puts],
        )


def build_signals(ticker: str, us_ticker: str, snap: OptionsSnapshot,
                   fetched_at: str) -> list[DemandSignal]:
    """Turn one OptionsSnapshot into 0-3 DemandSignals:

      unusual_call_volume / unusual_put_volume -- fire only when that leg's
      volume/OI clears UNUSUAL_RATIO (both can fire the same day).
      call_put_skew -- always emitted when there's any volume at all;
      direction from which side dominates, strength = |calls-puts|/(calls+puts),
      naturally in [0,1].
    """
    signals = []

    if snap.call_oi > 0:
        call_ratio = snap.call_volume / snap.call_oi
        if call_ratio >= UNUSUAL_RATIO:
            signals.append(DemandSignal(
                ticker=ticker, us_ticker=us_ticker, date=snap.as_of_date,
                source="options_flow", signal_type="unusual_call_volume",
                direction="bullish",
                strength=min(1.0, call_ratio / UNUSUAL_RATIO / 3),
                lag_days=0,
                detail={"call_volume": snap.call_volume, "call_oi": snap.call_oi, "ratio": call_ratio},
                fetched_at=fetched_at,
            ))

    if snap.put_oi > 0:
        put_ratio = snap.put_volume / snap.put_oi
        if put_ratio >= UNUSUAL_RATIO:
            signals.append(DemandSignal(
                ticker=ticker, us_ticker=us_ticker, date=snap.as_of_date,
                source="options_flow", signal_type="unusual_put_volume",
                direction="bearish",
                strength=min(1.0, put_ratio / UNUSUAL_RATIO / 3),
                lag_days=0,
                detail={"put_volume": snap.put_volume, "put_oi": snap.put_oi, "ratio": put_ratio},
                fetched_at=fetched_at,
            ))

    total_vol = snap.call_volume + snap.put_volume
    if total_vol > 0:
        skew = (snap.call_volume - snap.put_volume) / total_vol
        signals.append(DemandSignal(
            ticker=ticker, us_ticker=us_ticker, date=snap.as_of_date,
            source="options_flow", signal_type="call_put_skew",
            direction="bullish" if skew > 0 else ("bearish" if skew < 0 else "neutral"),
            strength=abs(skew),
            lag_days=0,
            detail={"call_volume": snap.call_volume, "put_volume": snap.put_volume, "skew": skew},
            fetched_at=fetched_at,
        ))

    return signals
