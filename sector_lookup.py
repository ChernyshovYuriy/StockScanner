"""
sector_lookup.py
=================
Ticker -> GICS sector lookup, used by virtual_buy.py to enforce
MAX_POSITIONS_PER_SECTOR (config.py). Sector classifications change rarely,
so lookups are cached to disk under CACHE_PATH — repeated runs don't re-hit
Yahoo for tickers already seen.

Validated 2026-08 (see backtest_runner.py's max_per_sector/sector_map): a
max-N-per-sector cap on concurrent positions costs nothing in return but
significantly reduces max drawdown in a 16-fold walk-forward, by preventing
correlated same-sector clusters (e.g. Canadian bank earnings week) from
entering and exiting the book together.

The actual fetch + disk cache now live in market_data.py, shared by every
provider (get_sector()), so this module is a thin back-compat wrapper —
callers (and tests, which patch virtual_buy.get_sector directly) are
unaffected.
"""

from __future__ import annotations

from market_data import DEFAULT_PROVIDER, UNKNOWN_SECTOR  # noqa: F401


def get_sector(ticker: str) -> str:
    """Return the GICS sector for ticker (cached), fetching via yfinance on a
    cache miss. Returns UNKNOWN_SECTOR if the lookup fails or the ticker has
    no sector (e.g. an ETF) — treated as its own bucket by callers, so
    unclassified tickers are still capped among themselves rather than
    bypassing the cap entirely."""
    return DEFAULT_PROVIDER.get_sector(ticker)
