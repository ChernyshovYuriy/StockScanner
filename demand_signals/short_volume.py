"""
FINRA Reg SHO daily consolidated short-sale-volume file -> a "short sellers
stepping back" (or "piling in") signal, per ticker, per day.

Unlike darkpool.py's ATS weekly summary, this file needs NO credentials --
it's a plain, unauthenticated static download (confirmed against the live
endpoint: https://cdn.finra.org/equity/regsho/daily/CNMSshvol{YYYYMMDD}.txt,
pipe-delimited, header "Date|Symbol|ShortVolume|ShortExemptVolume|
TotalVolume|Market"). CNMS is FINRA's cross-market consolidated file (all
TRFs + the ADF/ORF combined) -- the one comparable, single number per
symbol; the six per-venue files (FNQC/FNRA/FNSQ/FNYX/FORF) aren't used here.

HONEST CEILING: like darkpool.py, this is short-sale *volume*, not signed
buy/sell order flow -- there's no way to know from this file alone who was
buying. The read here is indirect: a falling short-volume ratio is treated
as short sellers stepping back (reduced selling pressure / possible
covering), a demand tell, not a direct buy signal. A rising ratio is the
mirror-image bearish read. Unlike darkpool.py, this source DOES emit
'bearish' -- the ATS ratio has no clear directional meaning on its own,
but a rising fraction of volume being sold short is a more direct pressure
signal.

LAG: published the next business day (T+1) -- the reason this source was
added alongside edgar_insider's ~5-day Form 4 deadline instead of relying
on finra_darkpool's ~2-week-lagged weekly ATS ratio for anything time-
sensitive (see demand_signals/__init__.py's honest-ceiling note).
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import requests

try:
    from config import (
        DEMAND_CACHE_PATH as CACHE_PATH,
        DEMAND_SHORTVOL_TREND_DAYS as TREND_DAYS,
        DEMAND_SHORTVOL_STRENGTH_SCALE as STRENGTH_SCALE,
        DEMAND_USER_AGENT as USER_AGENT,
    )
except Exception:
    CACHE_PATH = Path(__file__).resolve().parent.parent / "demand_signals_cache"
    TREND_DAYS = 3
    STRENGTH_SCALE = 0.05
    USER_AGENT = "StockScanner-DemandSignals/0.1 (your-email@example.com)"

from demand_signals.http_cache import CachedClient
from demand_signals.schema import DemandSignal

_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{yyyymmdd}.txt"

# Own CachedClient instance, same "one rate budget per source" convention as
# darkpool.py -- this hits cdn.finra.org, a different host from darkpool.py's
# api.finra.org, so sharing a limiter would incorrectly couple the two.
_client = CachedClient(user_agent=USER_AGENT, cache_dir=Path(CACHE_PATH))


def _fetch_daily_file(d: date) -> str | None:
    """Raw pipe-delimited text of one date's CNMS file, or None if there
    isn't one (weekend, market holiday, or not yet published). Cached by
    date only -- the file covers every symbol, so one fetch serves every
    ticker the collector loop asks about that day, unlike darkpool.py's
    per-ticker requests."""
    if d.weekday() >= 5:
        return None  # no file published for weekends -- don't even try

    cache_key = f"finra_shortvol_{d.strftime('%Y%m%d')}.txt"
    try:
        return _client.request(
            _URL.format(yyyymmdd=d.strftime("%Y%m%d")),
            cache_key=cache_key,
            is_json=False,
        )
    except requests.RequestException:
        return None  # holiday, not yet published, or a transient failure


def _parse_symbol_row(text: str, symbol: str) -> tuple[float, float] | None:
    """(short_shares, total_shares) for `symbol` in one day's file, or None
    if the symbol isn't in it that day (e.g. no short-sale activity)."""
    for line in text.splitlines()[1:]:  # [0] is the header row
        parts = line.split("|")
        if len(parts) < 5 or parts[1] != symbol:
            continue
        try:
            return float(parts[2]), float(parts[4])
        except ValueError:
            return None
    return None


def fetch_daily_short_volume(us_ticker: str, days: int = 10) -> list[dict]:
    """Daily short-volume/total-volume ratio for `us_ticker`, most recent
    `days` trading days, oldest first. Weekends are skipped outright;
    holidays/not-yet-published days are silently skipped (same "no data ==
    no signal" convention as darkpool.py) -- so the returned list can be
    shorter than `days`.

    Each row: {"date": "YYYY-MM-DD", "short_shares": float,
    "total_shares": float, "ratio": float}.
    """
    rows = []
    d = date.today()
    checked = 0
    while checked < days:
        d -= timedelta(days=1)
        if d.weekday() >= 5:
            continue
        checked += 1

        text = _fetch_daily_file(d)
        if text is None:
            continue
        parsed = _parse_symbol_row(text, us_ticker)
        if parsed is None:
            continue
        short_shares, total_shares = parsed
        if total_shares <= 0:
            continue
        rows.append({"date": d.isoformat(), "short_shares": short_shares,
                     "total_shares": total_shares, "ratio": short_shares / total_shares})

    rows.reverse()  # oldest first, matching darkpool.py's ascending order
    return rows


def _consecutive_trend(ratios: list[float], idx: int, days: int, rising: bool) -> bool:
    """True if ratios[idx-days+1 .. idx] is strictly increasing (rising=True)
    or strictly decreasing (rising=False). Same shape as darkpool.py's
    _consecutive_rising, generalized to either direction."""
    if idx + 1 < days:
        return False
    window = ratios[idx - days + 1: idx + 1]
    if rising:
        return all(window[i] < window[i + 1] for i in range(len(window) - 1))
    return all(window[i] > window[i + 1] for i in range(len(window) - 1))


def build_signals(ticker: str, us_ticker: str, daily_rows: list[dict],
                   fetched_at: str) -> list[DemandSignal]:
    """Turn parsed daily short-volume rows into one DemandSignal per day.

    signal_type/direction:
      'short_volume_covering' / bullish -- ratio has FALLEN for TREND_DAYS
        straight sessions (short sellers stepping back).
      'short_volume_pressure' / bearish -- the mirror-image rising run.
      'short_volume_ratio'    / neutral -- neither trend confirmed yet.
    """
    signals = []
    ratios = [r["ratio"] for r in daily_rows]
    for i, row in enumerate(daily_rows):
        if _consecutive_trend(ratios, i, TREND_DAYS, rising=False):
            signal_type, direction = "short_volume_covering", "bullish"
        elif _consecutive_trend(ratios, i, TREND_DAYS, rising=True):
            signal_type, direction = "short_volume_pressure", "bearish"
        else:
            signal_type, direction = "short_volume_ratio", "neutral"

        delta = ratios[i] - ratios[i - 1] if i > 0 else 0.0
        signals.append(DemandSignal(
            ticker=ticker,
            us_ticker=us_ticker,
            date=row["date"],
            source="finra_short_volume",
            signal_type=signal_type,
            direction=direction,
            strength=min(1.0, abs(delta) / STRENGTH_SCALE),
            lag_days=1,  # published the next business day
            detail={"short_shares": row["short_shares"], "total_shares": row["total_shares"],
                    "ratio": row["ratio"]},
            fetched_at=fetched_at,
        ))
    return signals
