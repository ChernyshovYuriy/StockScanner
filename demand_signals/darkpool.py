"""
FINRA ATS (dark pool) weekly volume -> a "rising dark-pool ratio" signal.

FINRA's Query API is free but requires a registered app: OAuth2
client-credentials, not an anonymous GET like SEC EDGAR's endpoints (see
demand_signals/__init__.py's data-limitations note). FINRA_CLIENT_ID /
FINRA_CLIENT_SECRET are read from .env here (same self-contained pattern
send_report.py uses for GMAIL_* -- config.py itself doesn't load .env).
Unconfigured -> fetch_weekly_ats_volume() returns [] and logs why, same
"silently skip if not configured" convention as send_report.py.

HONEST CEILING: this is WEEKLY, published with its own ~2-week lag on top
of the week it covers. Confirmation only, never a live trigger.

The dark-pool ratio needs a denominator (total consolidated volume); rather
than a second raw-HTTP fetch, this reuses market_data.LiveDataProvider
read-only (already a dependency, already used for every other volume/price
need in this repo) to sum daily Volume across the matching week.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import requests

try:
    from config import (
        DEMAND_CACHE_PATH as CACHE_PATH,
        DEMAND_DARKPOOL_RISING_WEEKS as RISING_WEEKS,
        DEMAND_USER_AGENT as USER_AGENT,
    )
except Exception:
    CACHE_PATH = Path(__file__).resolve().parent.parent / "demand_signals_cache"
    RISING_WEEKS = 3
    USER_AGENT = "StockScanner-DemandSignals/0.1 (your-email@example.com)"

from demand_signals.http_cache import CachedClient, cache_date
from demand_signals.schema import DemandSignal

_TOKEN_URL = "https://ews.fip.finra.org/fip/rest/ews/oauth2/access_token"
_QUERY_URL = "https://api.finra.org/data/group/otcMarket/name/weeklySummary"
_ATS_SUMMARY_TYPE = "ATS_W_SMBL_FIRM"

# ── credentials: .env, same self-contained pattern as send_report.py ──────
try:
    from dotenv import load_dotenv

    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass  # dotenv not installed -- that's fine, falls through to os.environ

FINRA_CLIENT_ID = os.environ.get("FINRA_CLIENT_ID", "")
FINRA_CLIENT_SECRET = os.environ.get("FINRA_CLIENT_SECRET", "")

_client = CachedClient(user_agent=USER_AGENT, cache_dir=Path(CACHE_PATH))

# In-memory OAuth2 token cache -- not written to disk (unlike raw data
# responses): a bearer token is short-lived and shouldn't outlive the
# process, let alone be cached-by-key indefinitely like a data response.
_token_cache = {"token": None, "expires_at": 0.0}


def _get_access_token(force: bool = False) -> str | None:
    """Client-credentials OAuth2 token, cached in-memory until near expiry.

    Returns None (never raises) when FINRA_CLIENT_ID/SECRET aren't set, so
    callers can treat "not configured" the same way send_report.py treats
    missing Gmail credentials: skip and log, don't crash the run.

    NOTE: implements the standard OAuth2 client-credentials shape (HTTP
    Basic auth + grant_type=client_credentials); verify against FINRA's
    actual token response once real credentials are available -- this
    hasn't been exercised against the live endpoint.
    """
    import time

    if not FINRA_CLIENT_ID or not FINRA_CLIENT_SECRET:
        return None
    if not force and _token_cache["token"] and time.time() < _token_cache["expires_at"]:
        return _token_cache["token"]

    resp = requests.post(
        _TOKEN_URL,
        params={"grant_type": "client_credentials"},
        auth=(FINRA_CLIENT_ID, FINRA_CLIENT_SECRET),
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    _token_cache["token"] = data["access_token"]
    # Refresh a minute early rather than racing the real expiry.
    # FINRA returns expires_in as a string (observed: "43162"), not a number.
    _token_cache["expires_at"] = time.time() + int(data.get("expires_in", 3600)) - 60
    return _token_cache["token"]


def fetch_weekly_ats_volume(ticker: str, weeks: int = 8) -> list[dict]:
    """Raw FINRA ATS weekly-by-symbol volume rows for `ticker`, most recent
    `weeks` weeks. [] if FINRA credentials aren't configured or the request
    fails -- callers must treat that the same as "no data", not an error.

    Each row: {"week_start": "YYYY-MM-DD", "shares": int}.
    """
    token = _get_access_token()
    if not token:
        return []

    end = date.today()
    start = end - timedelta(weeks=weeks + 1)
    cache_key = f"finra_ats_{ticker}_{weeks}w_{cache_date()}.json"

    try:
        records = _client.request(
            _QUERY_URL,
            method="POST",
            cache_key=cache_key,
            # FINRA's Query API defaults to CSV (text/plain) regardless of
            # the request body; Accept: application/json is required to get
            # back the JSON shape _parse_ats_records() expects.
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            json_body={
                "compareFilters": [
                    {"fieldName": "issueSymbolIdentifier", "fieldValue": ticker, "compareType": "EQUAL"},
                    {"fieldName": "summaryTypeCode", "fieldValue": _ATS_SUMMARY_TYPE, "compareType": "EQUAL"},
                ],
                "dateRangeFilters": [
                    {"fieldName": "weekStartDate", "startDate": start.isoformat(), "endDate": end.isoformat()},
                ],
                "limit": 5000,
            },
        )
    except requests.RequestException:
        return []

    return _parse_ats_records(records)


def _parse_ats_records(records: list[dict]) -> list[dict]:
    """Collapse FINRA's per-venue rows to one total per week (a symbol can
    trade across many ATS venues in the same week; totalWeeklyShareQuantity
    is per venue, so this sums them)."""
    by_week: dict[str, int] = {}
    for r in records:
        week = r.get("weekStartDate")
        shares = r.get("totalWeeklyShareQuantity")
        if week is None or shares is None:
            continue
        by_week[week] = by_week.get(week, 0) + int(shares)
    return [{"week_start": w, "shares": s} for w, s in sorted(by_week.items())]


def _total_weekly_volume(us_ticker: str, week_start: str) -> int | None:
    """Sum of daily consolidated Volume (all venues, lit + dark) over the
    trading week starting `week_start`, via the existing yfinance path.
    None if the fetch fails -- caller skips that week's ratio rather than
    dividing by a wrong number."""
    from market_data import LiveDataProvider

    start = date.fromisoformat(week_start)
    end = start + timedelta(days=6)
    try:
        df = LiveDataProvider().get(us_ticker, as_of=None, start_dt=start.isoformat(),
                                     end_dt=end.isoformat())
    except Exception:
        return None
    if df is None or df.empty:
        return None
    return int(df["Volume"].sum())


def _consecutive_rising(ratios: list[float], idx: int, weeks: int) -> bool:
    """True if ratios[idx-weeks+1 .. idx] is strictly increasing."""
    if idx + 1 < weeks:
        return False
    window = ratios[idx - weeks + 1: idx + 1]
    return all(window[i] < window[i + 1] for i in range(len(window) - 1))


def build_signals(ticker: str, us_ticker: str, ats_weekly: list[dict],
                   fetched_at: str) -> list[DemandSignal]:
    """Turn parsed ATS weekly rows into one DemandSignal per week that has
    both an ATS figure and a resolvable total-volume denominator.

    direction is 'bullish' only for a week that caps a RISING_WEEKS-long
    strictly-increasing run of the ratio; 'neutral' otherwise -- this
    source is never 'bearish' (a falling ratio isn't a documented
    distribution signal, just the absence of this one).
    """
    weeks_with_ratio = []
    for row in ats_weekly:
        total = _total_weekly_volume(us_ticker, row["week_start"])
        if not total:
            continue
        weeks_with_ratio.append((row["week_start"], row["shares"] / total, row["shares"], total))

    signals = []
    ratios = [r for _, r, _, _ in weeks_with_ratio]
    for i, (week_start, ratio, ats_shares, total_shares) in enumerate(weeks_with_ratio):
        rising = _consecutive_rising(ratios, i, RISING_WEEKS)
        delta = ratio - ratios[i - 1] if i > 0 else 0.0
        signals.append(DemandSignal(
            ticker=ticker,
            us_ticker=us_ticker,
            date=week_start,
            source="finra_darkpool",
            signal_type="darkpool_ratio_rising" if rising else "darkpool_ratio",
            direction="bullish" if rising else "neutral",
            strength=min(1.0, max(0.0, delta / 0.05)),
            lag_days=14,  # FINRA's own publication lag behind the week it covers
            detail={"ats_shares": ats_shares, "total_shares": total_shares, "ratio": ratio},
            fetched_at=fetched_at,
        ))
    return signals
