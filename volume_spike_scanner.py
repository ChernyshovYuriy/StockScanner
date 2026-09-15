"""
volume_spike_scanner.py
========================
On-demand intraday volume-spike scanner. Purely a dashboard feature (see
dashboard_app.py's /volume-spikes route) — no scheduled service, no
systemd unit, no DB, nothing persisted: each page load / refresh click
runs a fresh live scan.

For every ticker in the VOLUME_SPIKE_TICKERS_URL universe, compares
today's volume-so-far against that ticker's own recent average daily
volume, via market_data.py's LiveDataProvider.download_range() — the one
batched fetch with no >200-row quality gate (see its own docstring),
which this simple comparison doesn't need. Only tickers actually spiking
above their own average are returned (see compute_spikes()).

Usage
-----
  python volume_spike_scanner.py                    # full VOLUME_SPIKE_TICKERS_URL universe
  python volume_spike_scanner.py AAPL MSFT SLF.TO     # manual test against explicit tickers
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List, Optional

import pandas as pd

from config import VOLUME_SPIKE_TICKERS_URL
from market_data import DEFAULT_PROVIDER
from research.triple_screen.batch import load_tickers
from time_utils import date_to_iso_extended, market_today

# Trading days of history averaged against today's volume-so-far. 20 ≈ one
# trading month — long enough to smooth out day-to-day noise, short enough
# to reflect the stock's *current* typical activity rather than a stale
# yearly average.
AVG_VOLUME_DAYS = 20

# Calendar-day lookback passed to download_range() — padded well past
# AVG_VOLUME_DAYS trading days to absorb weekends/holidays.
LOOKBACK_CALENDAR_DAYS = 45


@dataclass
class VolumeSpikeRow:
    ticker: str
    current_volume: int
    average_volume: float
    spike_pct: float


def compute_spikes(volume_by_ticker: Dict[str, pd.Series]) -> List[VolumeSpikeRow]:
    """Turn each ticker's daily Volume series (oldest → newest, most recent
    bar being today's, still filling during the session) into a
    VolumeSpikeRow, dropping any ticker at or below its own recent average
    — the feature only wants to surface spikes. Sorted by spike_pct
    descending.
    """
    rows = []
    for ticker, volumes in volume_by_ticker.items():
        if len(volumes) < 2:
            continue
        current = float(volumes.iloc[-1])
        history = volumes.iloc[-(AVG_VOLUME_DAYS + 1):-1]
        if history.empty:
            continue
        average = float(history.mean())
        if average <= 0 or current <= average:
            continue
        spike_pct = (current - average) / average * 100.0
        rows.append(VolumeSpikeRow(
            ticker=ticker,
            current_volume=int(current),
            average_volume=average,
            spike_pct=spike_pct,
        ))
    rows.sort(key=lambda r: r.spike_pct, reverse=True)
    return rows


def scan_volume_spikes(tickers: Optional[List[str]] = None) -> List[VolumeSpikeRow]:
    """Fetch + compute in one call — the entry point both the CLI and the
    dashboard route use."""
    universe = tickers if tickers is not None else load_tickers(VOLUME_SPIKE_TICKERS_URL)
    today = market_today()
    start = date_to_iso_extended(today - timedelta(days=LOOKBACK_CALENDAR_DAYS))
    # end is exclusive in yfinance -- +1 day so today's still-filling bar
    # is actually included, not just history up to yesterday's close.
    end = date_to_iso_extended(today + timedelta(days=1))
    data, _failed = DEFAULT_PROVIDER.download_range(universe, start=start, end=end)
    volume_by_ticker = {ticker: df["Volume"] for ticker, df in data.items()}
    return compute_spikes(volume_by_ticker)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan for intraday volume spikes")
    parser.add_argument("tickers", nargs="*",
                         help="explicit tickers, e.g. AAPL MSFT SLF.TO (overrides VOLUME_SPIKE_TICKERS_URL)")
    args = parser.parse_args()
    result = scan_volume_spikes(args.tickers or None)
    for r in result:
        print(f"{r.ticker:10s} current={r.current_volume:>12,} avg={r.average_volume:>12,.0f} spike={r.spike_pct:+.1f}%")
