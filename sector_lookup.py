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
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import yfinance as yf

from config import CACHE_PATH

_CACHE_FILE = Path(CACHE_PATH) / "sector_cache.json"
UNKNOWN_SECTOR = "Unknown"


def _load_cache() -> Dict[str, str]:
    try:
        return json.loads(_CACHE_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache: Dict[str, str]) -> None:
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE.write_text(json.dumps(cache, indent=1, sort_keys=True))
    except OSError:
        pass  # best-effort — a failed write just means next run re-fetches


_cache: Dict[str, str] = _load_cache()


def get_sector(ticker: str) -> str:
    """Return the GICS sector for ticker (cached), fetching via yfinance on a
    cache miss. Returns UNKNOWN_SECTOR if the lookup fails or the ticker has
    no sector (e.g. an ETF) — treated as its own bucket by callers, so
    unclassified tickers are still capped among themselves rather than
    bypassing the cap entirely."""
    ticker = ticker.upper()
    if ticker in _cache:
        return _cache[ticker]
    sector = UNKNOWN_SECTOR
    try:
        info = yf.Ticker(ticker).info
        sector = info.get("sector") or UNKNOWN_SECTOR
    except Exception:
        pass
    _cache[ticker] = sector
    _save_cache(_cache)
    return sector
