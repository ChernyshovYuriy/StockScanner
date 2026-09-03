"""
macro_regime.py
================
A top-down macro/liquidity "regime" read, loosely mechanizing the part of
Stanley Druckenmiller's approach he's stated most directly: watch central-
bank liquidity and credit conditions before anything else. Feeds
macro_buy.py's entry gate and macro_monitor.py's regime-flip kill switch —
see config.py's MACRO_* block and CLAUDE.md.

Three free FRED (St. Louis Fed) series, each voted +1 (bullish/risk-on),
-1 (bearish/risk-off), or 0 (no confirmed trend) over its own native-
frequency lookback window, using the "consecutive N periods trending" idiom
already established in demand_signals/darkpool.py's _consecutive_rising()
and demand_signals/short_volume.py's _consecutive_trend() -- reimplemented
fresh here (a small pure function, not imported) since this module is
standalone, not part of the demand_signals package (different data, "services
stay independent" — same precedent as edgar_service.py vs demand_signals_service.py):

  T10Y2Y        (10y-2y Treasury yield spread, daily)   -- bullish if RISING
                (steepening / de-inversion = risk appetite improving)
  BAMLH0A0HYM2  (ICE BofA US High Yield OAS, daily)      -- bullish if FALLING
                (credit spread tightening = credit stress easing)
  WALCL         (Fed total assets, weekly)               -- bullish if RISING
                (balance-sheet expansion = the "central bank liquidity"
                 signal Druckenmiller himself has repeatedly cited)

composite = sum(votes) in [-3, +3]; label thresholds at zero:
  composite > 0 -> "risk_on"
  composite < 0 -> "risk_off"
  composite == 0 -> "neutral"
No tunable float threshold is needed -- the vote system self-thresholds.

HONEST CEILING: this is 3 macro-liquidity proxies, not a forecasting model --
no backtest/walk-forward exists for this composite (unlike e.g. the core
sleeve's MAX_POSITIONS_PER_SECTOR or the momentum sleeve's exit params). It
is a coarse regime label, not a market-timing tool: it says "conditions are
broadly supportive/hostile," nothing about any specific ticker. T10Y2Y and
BAMLH0A0HYM2 publish next business day (T+1 lag from FRED, itself sourced
same-day from Treasury/ICE); WALCL publishes weekly (Thursdays, as-of
Wednesday) -- so the "daily" votes can be up to a day stale and the liquidity
vote up to a week stale. Never a live intraday signal.
"""

from __future__ import annotations

import os
from pathlib import Path

import requests

try:
    from config import (
        MACRO_CACHE_PATH as CACHE_PATH,
        MACRO_CREDIT_TREND_DAYS as CREDIT_TREND_DAYS,
        MACRO_CURVE_TREND_DAYS as CURVE_TREND_DAYS,
        MACRO_LIQUIDITY_TREND_WEEKS as LIQUIDITY_TREND_WEEKS,
        MACRO_USER_AGENT as USER_AGENT,
    )
except Exception:
    CACHE_PATH = Path(__file__).resolve().parent / "cache" / "macro_regime"
    CREDIT_TREND_DAYS = 5
    CURVE_TREND_DAYS = 5
    LIQUIDITY_TREND_WEEKS = 3
    USER_AGENT = "StockScanner-MacroRegime/0.1 (your-email@example.com)"

from demand_signals.http_cache import CachedClient, cache_date
from time_utils import market_now

# ── credentials: .env, same self-contained pattern as demand_signals/darkpool.py ──
try:
    from dotenv import load_dotenv

    _env_path = Path(__file__).resolve().parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass  # dotenv not installed -- that's fine, falls through to os.environ

FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# Own CachedClient instance, same "one rate budget per source" convention as
# demand_signals -- api.stlouisfed.org is a new host, never shared with
# FINRA/SEC/Yahoo's limiters.
_client = CachedClient(user_agent=USER_AGENT, cache_dir=Path(CACHE_PATH))

# series_id -> (lookback window in native-frequency readings, True if a
# RISING trend is the bullish direction for this series).
_SERIES = {
    "T10Y2Y": (CURVE_TREND_DAYS, True),
    "BAMLH0A0HYM2": (CREDIT_TREND_DAYS, False),
    "WALCL": (LIQUIDITY_TREND_WEEKS, True),
}


def _fetch_series(series_id: str, lookback_n: int, force: bool = False) -> list[dict] | None:
    """Most recent `lookback_n` observations for `series_id`, oldest first,
    or None on any failure (missing key, network error, bad response) --
    the caller treats a failed series as a 0 vote, not a crash.

    FRED's documented missing-value sentinel is the literal string "." --
    those rows are dropped rather than passed to float().
    """
    if not FRED_API_KEY:
        return None
    url = (
        f"{_BASE_URL}?series_id={series_id}&api_key={FRED_API_KEY}"
        f"&file_type=json&sort_order=desc&limit={lookback_n}"
    )
    cache_key = f"{series_id}_{cache_date()}.json"
    try:
        payload = _client.request(url, cache_key=cache_key, force=force, is_json=True)
    except requests.RequestException:
        return None
    observations = payload.get("observations", []) if isinstance(payload, dict) else []
    rows = []
    for obs in observations:
        raw = obs.get("value")
        if raw in (None, "."):
            continue
        try:
            rows.append({"date": obs.get("date"), "value": float(raw)})
        except (TypeError, ValueError):
            continue
    rows.reverse()  # oldest first, matching demand_signals' ascending convention
    return rows or None


def _consecutive_trend(values: list[float], rising: bool) -> bool:
    """True if `values` (already the exact lookback window, oldest first) is
    strictly increasing (rising=True) or strictly decreasing (rising=False).
    Same shape as demand_signals/short_volume.py's _consecutive_trend(),
    reimplemented here since this module doesn't import from that package."""
    if len(values) < 2:
        return False
    if rising:
        return all(values[i] < values[i + 1] for i in range(len(values) - 1))
    return all(values[i] > values[i + 1] for i in range(len(values) - 1))


def _vote_series(series_id: str, lookback_n: int, rising_is_bullish: bool, force: bool = False) -> tuple[int, dict]:
    """+1 if the series is consecutive-trending toward its bullish direction,
    -1 if trending toward its bearish direction, 0 if flat/mixed/unavailable.
    Also returns a `detail` dict for the caller's transparency/debugging."""
    rows = _fetch_series(series_id, lookback_n, force=force)
    if not rows:
        return 0, {"error": "unavailable (missing FRED_API_KEY or fetch failed)"}

    values = [r["value"] for r in rows]
    latest = values[-1]
    detail = {"latest": latest, "as_of": rows[-1]["date"], "window": values}

    if _consecutive_trend(values, rising=rising_is_bullish):
        return 1, detail
    if _consecutive_trend(values, rising=not rising_is_bullish):
        return -1, detail
    return 0, detail


def get_macro_regime(force: bool = False) -> dict:
    """
    Compute today's composite macro regime reading.

    force: bypass the per-day disk cache and re-fetch every series (passed
    straight through to CachedClient.request's own `force` kwarg) -- not
    used by macro_buy.py/macro_monitor.py in normal operation, since one
    fetch per day is the intended cadence for a T+1/weekly-lagged signal.

    Returns {"label": "risk_on"|"neutral"|"risk_off", "composite": int,
             "votes": {series_id: int}, "detail": {series_id: {...}},
             "fetched_at": iso}. If FRED_API_KEY is unset, every series votes
    0 and the result is "neutral" -- a fail-safe, not an error: both
    macro_buy.py (no buys on non-risk_on) and macro_monitor.py (no forced
    liquidation on non-risk_off) treat this identically to a genuine neutral
    reading, so the sleeve just stays in cash until the key is configured.
    """
    votes: dict[str, int] = {}
    detail: dict[str, dict] = {}

    for series_id, (lookback_n, rising_is_bullish) in _SERIES.items():
        try:
            vote, series_detail = _vote_series(series_id, lookback_n, rising_is_bullish, force=force)
        except Exception as exc:
            # One series' unexpected failure must not sink the whole
            # regime read -- it just contributes no vote.
            vote, series_detail = 0, {"error": str(exc)}
        votes[series_id] = vote
        detail[series_id] = series_detail

    composite = sum(votes.values())
    if composite > 0:
        label = "risk_on"
    elif composite < 0:
        label = "risk_off"
    else:
        label = "neutral"

    return {
        "label": label,
        "composite": composite,
        "votes": votes,
        "detail": detail,
        "fetched_at": market_now().isoformat(),
    }
