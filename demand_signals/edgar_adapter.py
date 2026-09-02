"""
Adapts EDGAR's own persisted records (insider_buys in edgar.db) into the
common demand_signals schema. READS ONLY -- never re-fetches or duplicates
edgar/'s own SEC EDGAR logic, and never writes back into edgar.db. Callers
pass in an edgar.store connection explicitly (rather than this module
importing edgar.store and opening its own), so edgar_service.py keeps sole
ownership of what "the" EDGAR DB connection is.

Scoped to insider buys only (source='edgar_insider'), matching this
package's reviewed schema. Activist stakes (13D/13G) aren't adapted here --
if wanted later, they'd be a natural additional `source` value, same
extension pattern as ticker_map.py's SEDI note.
"""

from __future__ import annotations

from demand_signals.schema import DemandSignal

try:
    from config import EDGAR_MIN_BUY_VALUE
except Exception:
    EDGAR_MIN_BUY_VALUE = 250_000


def normalize_recent_insider_buys(conn, since_date: str = None) -> list[DemandSignal]:
    """All insider_buys rows in edgar.db (optionally restricted to
    txn_date >= since_date), normalized to DemandSignal.

    Sells aren't in this table at all -- edgar/insiders.py only ever
    isolates transaction-code 'P' purchases -- so direction is always
    bullish, matching edgar's own "sells carry little signal, intentionally
    never flagged" convention.
    """
    from edgar.core import load_cik_to_ticker
    from time_utils import market_now

    cik2tic = load_cik_to_ticker()
    fetched_at = market_now().isoformat()

    if since_date:
        rows = conn.execute(
            "SELECT cik, accession, owner, shares, price, txn_date, "
            "is_officer, is_director FROM insider_buys WHERE txn_date >= ?",
            (since_date,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT cik, accession, owner, shares, price, txn_date, "
            "is_officer, is_director FROM insider_buys"
        ).fetchall()

    signals = []
    for cik, accession, owner, shares, price, txn_date, is_officer, is_director in rows:
        ticker = cik2tic.get(cik)
        if not ticker or not txn_date:
            continue  # can't screen a signal with no resolvable ticker or date
        value = (shares or 0) * (price or 0)
        signals.append(DemandSignal(
            ticker=ticker,
            us_ticker=ticker,  # EDGAR watchlist CIKs are already US filers
            date=txn_date,
            source="edgar_insider",
            signal_type="insider_buy",
            direction="bullish",
            strength=min(1.0, value / (10 * EDGAR_MIN_BUY_VALUE)) if value else 0.0,
            lag_days=5,  # Form 4's own ~5-business-day filing deadline
            detail={"owner": owner, "shares": shares, "price": price, "value": value,
                    "accession": accession, "is_officer": bool(is_officer),
                    "is_director": bool(is_director)},
            fetched_at=fetched_at,
        ))
    return signals
