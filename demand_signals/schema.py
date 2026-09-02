"""
The normalized "demand signal" record all four sources produce, and its
flat SQLite row shape (see store.py). See demand_signals/__init__.py for
what strength/direction mean per source and the honest data-limitations note.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Literal

Source = Literal["edgar_insider", "finra_darkpool", "finra_short_volume", "options_flow"]
Direction = Literal["bullish", "bearish", "neutral"]

_VALID_SOURCES = {"edgar_insider", "finra_darkpool", "finra_short_volume", "options_flow"}
_VALID_DIRECTIONS = {"bullish", "bearish", "neutral"}


@dataclass(frozen=True)
class DemandSignal:
    """One normalized signal for one ticker on one date.

    ticker      : symbol as the screener knows it (CAN or US)
    us_ticker   : symbol actually queried -- == ticker for US names, resolved
                  via ticker_map.get_us_ticker() for interlisted CAN names
    date        : ISO date the signal is FOR (not fetch date) -- FINRA
                  week-ending date, Form 4 txn date, short-volume file date,
                  or options snapshot date
    source      : 'edgar_insider' | 'finra_darkpool' | 'finra_short_volume'
                  | 'options_flow'
    signal_type : e.g. 'insider_buy', 'darkpool_ratio_rising',
                  'short_volume_covering', 'unusual_call_volume',
                  'unusual_put_volume'
    direction   : 'bullish' | 'bearish' | 'neutral'
    strength    : 0.0-1.0, source-specific magnitude (see each source's
                  module docstring for exactly how it's computed)
    lag_days    : structural staleness of this signal type -- callers use
                  this to weight/discount, not just the strength value
    detail      : source-specific raw fields, kept as evidence (mirrors
                  edgar's activist_filings.raw_text convention)
    fetched_at  : ISO timestamp this record was produced
    """

    ticker: str
    us_ticker: str
    date: str
    source: Source
    signal_type: str
    direction: Direction
    strength: float
    lag_days: int
    detail: dict
    fetched_at: str

    def __post_init__(self):
        if self.source not in _VALID_SOURCES:
            raise ValueError(f"invalid source: {self.source!r}")
        if self.direction not in _VALID_DIRECTIONS:
            raise ValueError(f"invalid direction: {self.direction!r}")
        if not (0.0 <= self.strength <= 1.0):
            raise ValueError(f"strength must be in [0.0, 1.0], got {self.strength}")
        if self.lag_days < 0:
            raise ValueError(f"lag_days must be >= 0, got {self.lag_days}")

    def to_row(self) -> dict:
        """Flat dict matching store.py's demand_signals table columns
        (detail serialized to JSON text, as SQLite has no native dict type)."""
        row = asdict(self)
        row["detail"] = json.dumps(self.detail, default=str)
        return row

    @classmethod
    def from_row(cls, row: dict) -> "DemandSignal":
        """Inverse of to_row() -- reconstruct from a stored SQLite row."""
        row = dict(row)
        row["detail"] = json.loads(row["detail"]) if row.get("detail") else {}
        return cls(**row)
