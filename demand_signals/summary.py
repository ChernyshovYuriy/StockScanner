"""
Plain-English composite read of one ticker's demand_signals -- condenses the
raw per-source rows (see darkpool.py/options_flow.py/edgar_adapter.py) into
a single label + short reason, for people who don't want to read a table of
signal_type/direction/strength rows. Shared by demand_signals_service.py's
--dry-run printout and the dashboard's /demand tab so both stay in sync.

This is a re-phrasing layer only -- it doesn't compute anything the sources
don't already compute, and it's not a trigger, same "confirmation only"
posture as every source's own honest-ceiling note (see
demand_signals/__init__.py). summarize_ticker() takes plain dicts (each
needing at least source/signal_type/direction/date) so it works equally
against demand_dashboard_data.py's sqlite-row dicts and against live
DemandSignal objects via vars(signal).
"""

from __future__ import annotations

from datetime import date as _date
from typing import Iterable

try:
    from config import DEMAND_DARKPOOL_RISING_WEEKS as RISING_WEEKS
except Exception:
    RISING_WEEKS = 3


def summarize_ticker(rows: Iterable[dict]) -> dict:
    """One ticker's rows -> {"label": str, "reasons": [str, ...]}.

    label is one of: "bullish", "mild bullish", "neutral", "mixed",
    "mild bearish", "bearish", "no data".

    "elevated" evidence (an insider buy, a rising dark-pool trend, or a
    today's-volume spike on either side) outweighs plain directional lean:
    a ticker with only an ordinary call/put skew tops out at "mild
    bullish"/"mild bearish"; only elevated evidence earns the plain
    "bullish"/"bearish" label.
    """
    rows = list(rows)

    insider_buys = [r for r in rows if r["source"] == "edgar_insider"]

    darkpool_rising = None  # most recent darkpool_ratio_rising row, if any
    for r in rows:
        if r["source"] == "finra_darkpool" and r["signal_type"] == "darkpool_ratio_rising":
            if darkpool_rising is None or r["date"] > darkpool_rising["date"]:
                darkpool_rising = r

    skew_row = None
    unusual_call = False
    unusual_put = False
    for r in rows:
        if r["source"] != "options_flow":
            continue
        if r["signal_type"] == "call_put_skew":
            skew_row = r
        elif r["signal_type"] == "unusual_call_volume":
            unusual_call = True
        elif r["signal_type"] == "unusual_put_volume":
            unusual_put = True

    score = 0
    reasons = []

    if insider_buys:
        score += 1
        n = len(insider_buys)
        reasons.append(f"{n} insider buy{'s' if n != 1 else ''} on record")

    if darkpool_rising:
        score += 1
        month = _date.fromisoformat(darkpool_rising["date"]).strftime("%B")
        reasons.append(f"dark-pool ratio rose {RISING_WEEKS}wks straight in {month}")

    if unusual_call and unusual_put:
        reasons.append("unusual volume on both calls and puts today")
    elif unusual_call:
        score += 1
        reasons.append("unusual call volume today")
    elif unusual_put:
        score -= 1
        reasons.append("unusual put volume today")

    if skew_row:
        if skew_row["direction"] == "bullish":
            score += 1
            reasons.append("today's flow leans calls")
        elif skew_row["direction"] == "bearish":
            score -= 1
            reasons.append("today's flow leans puts")
        else:
            reasons.append("today's flow is balanced calls/puts")

    elevated = bool(insider_buys) or bool(darkpool_rising) or unusual_call or unusual_put

    if not reasons:
        return {"label": "no data", "reasons": ["no signals available"]}

    if not elevated:
        reasons.append("no unusual volume, no dark-pool trend, no insider buys")

    if score > 0:
        label = "bullish" if elevated else "mild bullish"
    elif score < 0:
        label = "bearish" if elevated else "mild bearish"
    else:
        label = "mixed" if elevated else "neutral"

    return {"label": label, "reasons": reasons}


def summarize_all(by_ticker: dict) -> dict:
    """{ticker: [row, ...]} -> {ticker: {"label": ..., "reasons": [...]}}."""
    return {ticker: summarize_ticker(rows) for ticker, rows in by_ticker.items()}
