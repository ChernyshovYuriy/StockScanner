"""Configuration constants for the conviction_watchlist package.

See conviction_watchlist/__init__.py for the package's overall purpose.
"""
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent

# Raw CAN ticker universe -- the user's own list, refreshed periodically from
# Yahoo Finance (see canadian_stock_screener.py's CAN_TICKERS_URL for the
# same fetch pattern/repo, a different file within it).
TICKERS_URL = "https://raw.githubusercontent.com/ChernyshovYuriy/Financing/refs/heads/main/data/can_tickers_full"

INFO_CACHE_FILE = REPO_ROOT / "cache" / "conviction_watchlist_info_cache.json"
HOLDINGS_FILE = REPO_ROOT / "data" / "conviction_holdings.json"
CANDIDATES_FILE = REPO_ROOT / "data" / "conviction_entry_candidates.json"

MIN_MARKET_CAP_CAD = 2_000_000_000
MIN_PRICE = 5.0
DIP_PCT_OFF_HIGH = 0.20     # buy candidate: price is >=20% below its 52-week high (picked 2026-09-05)
TRAILING_STOP_PCT = 0.25    # sell candidate: price is >=25% below the peak since entry (picked 2026-09-05)
