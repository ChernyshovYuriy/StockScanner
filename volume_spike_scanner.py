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
which this simple comparison doesn't need. Only tickers BOTH spiking
above their own average volume AND trading up on the day are returned
(see compute_spikes()) — a volume spike alone is directionless (it can
just as easily mean heavy selling), so price direction is a hard filter,
not a separate displayed column.

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


def compute_spikes(data_by_ticker: Dict[str, pd.DataFrame]) -> List[VolumeSpikeRow]:
    """Turn each ticker's daily OHLCV (oldest → newest, most recent bar
    being today's, still filling during the session) into a
    VolumeSpikeRow, keeping only tickers that are BOTH running above
    their own recent average volume AND trading up on the day (today's
    close/last price above yesterday's close) — a volume spike on a down
    day is more likely distribution/selling than the buying-interest
    signal this scanner is for. Sorted by spike_pct descending.
    """
    rows = []
    for ticker, df in data_by_ticker.items():
        if len(df) < 2:
            continue
        current_close = float(df["Close"].iloc[-1])
        prev_close = float(df["Close"].iloc[-2])
        if current_close <= prev_close:
            continue
        current_volume = float(df["Volume"].iloc[-1])
        volume_history = df["Volume"].iloc[-(AVG_VOLUME_DAYS + 1):-1]
        if volume_history.empty:
            continue
        average_volume = float(volume_history.mean())
        if average_volume <= 0 or current_volume <= average_volume:
            continue
        spike_pct = (current_volume - average_volume) / average_volume * 100.0
        rows.append(VolumeSpikeRow(
            ticker=ticker,
            current_volume=int(current_volume),
            average_volume=average_volume,
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
    return compute_spikes(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan for intraday volume spikes")
    parser.add_argument("tickers", nargs="*",
                         help="explicit tickers, e.g. AAPL MSFT SLF.TO (overrides VOLUME_SPIKE_TICKERS_URL)")
    args = parser.parse_args()
    result = scan_volume_spikes(args.tickers or None)
    for r in result:
        print(f"{r.ticker:10s} current={r.current_volume:>12,} avg={r.average_volume:>12,.0f} spike={r.spike_pct:+.1f}%")
