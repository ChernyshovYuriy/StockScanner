"""
EDGAR core plumbing: Layer 0 (identity) + Layer 1 (rate-limited fetch + cache).

Shared by every higher layer. SEC fair-access rules:
  - Declare a User-Agent identifying you with contact info (or you get 403'd).
  - Stay under 10 requests/second.
"""

import json
import threading
import time
from collections import deque
from datetime import date
from pathlib import Path

import requests

# ---- CONFIGURE: SEC requires a real contact in the User-Agent ----
# In StockScanner these come from config.py (EDGAR_USER_AGENT / EDGAR_CACHE_PATH).
# The fallbacks keep this package runnable standalone in the edgar sandbox.
try:
    from config import EDGAR_USER_AGENT as USER_AGENT
    from config import EDGAR_CACHE_PATH as CACHE_DIR

    CACHE_DIR = Path(CACHE_DIR)
except Exception:
    USER_AGENT = "EdgarScreener/0.2 (your-email@example.com)"
    CACHE_DIR = Path(__file__).resolve().parent.parent / "edgar_cache"
# ------------------------------------------------------------------

CACHE_DIR.mkdir(parents=True, exist_ok=True)

_RATE = 8
_WINDOW = 1.0


class _RateLimiter:
    def __init__(self, rate=_RATE, window=_WINDOW):
        self.rate, self.window = rate, window
        self.calls = deque()
        self.lock = threading.Lock()

    def acquire(self):
        with self.lock:
            now = time.monotonic()
            while self.calls and now - self.calls[0] > self.window:
                self.calls.popleft()
            if len(self.calls) >= self.rate:
                time.sleep(max(self.window - (now - self.calls[0]), 0))
            self.calls.append(time.monotonic())


_limiter = _RateLimiter()
_session = requests.Session()
_session.headers.update({"User-Agent": USER_AGENT})


def get(url, cache_key=None, force=False, is_json=True):
    """Rate-limited GET with optional on-disk cache."""
    if cache_key:
        cache_file = CACHE_DIR / cache_key
        if cache_file.exists() and not force:
            text = cache_file.read_text()
            return json.loads(text) if is_json else text
    _limiter.acquire()
    resp = _session.get(url, timeout=30)
    resp.raise_for_status()
    text = resp.text
    if cache_key:
        (CACHE_DIR / cache_key).write_text(text)
    return json.loads(text) if is_json else text


def _cache_date():
    """Today's date as a cache-key suffix. A seam so tests can pin it."""
    return date.today().strftime("%Y%m%d")


# ---------------- Layer 0: identity ----------------

_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"


def load_ticker_map(force=False):
    """{TICKER: cik_int}. Date-scoped cache (like fetch_submissions below): a
    plain forever-cache would miss new listings/ticker changes indefinitely
    once fetched once. Ticker/CIK assignments change far less often than
    daily, but the collector only runs once a day, so a daily cache costs
    nothing extra while bounding staleness to at most one day."""
    data = get(_TICKER_URL, cache_key=f"company_tickers_{_cache_date()}.json", force=force)
    return {row["ticker"].upper(): int(row["cik_str"]) for row in data.values()}


def load_cik_to_ticker(force=False):
    """Reverse map {cik_int: TICKER} for labelling scan hits.

    A CIK can have multiple ticker aliases (multiple share classes, e.g.
    Alphabet's GOOGL/GOOG/GOOGM/GOOGN all share one CIK); SEC's source file
    lists the primary/most-traded class first, so keep the first ticker
    seen per CIK rather than the last (a plain {v: k for k, v in ...} dict
    comprehension silently keeps the LAST one, which for Alphabet lands on
    the obscure "GOOGN" alias rather than "GOOG").
    """
    result = {}
    for ticker, cik in load_ticker_map(force=force).items():
        result.setdefault(cik, ticker)
    return result


def cik_for(ticker, ticker_map=None):
    ticker_map = ticker_map or load_ticker_map()
    cik = ticker_map.get(ticker.upper())
    if cik is None:
        raise KeyError(f"{ticker} not in EDGAR ticker map (US filers only)")
    return cik


def cik10(cik_int):
    return f"{int(cik_int):010d}"


# ---------------- Layer 1: shared fetch helpers ----------------

def fetch_company_facts(cik_int, force=False):
    c = cik10(cik_int)
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{c}.json"
    return get(url, cache_key=f"facts_{c}.json", force=force)


def fetch_submissions(cik_int, force=False):
    """The CIK's recent filings, incl. Form 4s -- changes daily, so the cache
    key is date-scoped: a plain per-CIK key would serve the same snapshot
    back forever, and any Form 4 filed after the first-ever fetch for that
    CIK would never be seen again."""
    c = cik10(cik_int)
    url = f"https://data.sec.gov/submissions/CIK{c}.json"
    return get(url, cache_key=f"subs_{c}_{_cache_date()}.json", force=force)
