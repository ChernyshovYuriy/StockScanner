"""
Quality watchlist builder -- see conviction_watchlist/__init__.py for the
package's overall purpose.

Filters the user's full CAN ticker list (fetched from config.TICKERS_URL,
~1000 tickers refreshed periodically from Yahoo Finance) down to a much
smaller set of "quality, large-cap, profitable, main-board" names -- the
type of business meant to be held through a drawdown, as opposed to the
speculative TSXV (.V) / CSE (.CN) / NEO (.NE) junior miners and microcaps
(TTS, CUPR, TLG, ONFO, MSCL) that produced the account's biggest losses.

Quality bar (all must pass, thresholds adjustable via settings.py / the
dashboard's Conviction tab):
  - main board only (.TO) -- Venture/CSE/NEO listings are excluded
    entirely, not scored
  - market cap >= min_market_cap_cad
  - positive trailing EPS (profitable, not a story stock)
  - price >= min_price (avoid low-priced/penny-adjacent names)

No sector exclusion: an earlier version also dropped Basic Materials/Energy
(mostly gold/silver/copper miners and oil & gas producers) as "commodity-
cyclical, not durable businesses." Reverted 2026-09-05 at the user's
explicit request -- sector alone is a weak proxy for quality (many of the
excluded names, e.g. Franco-Nevada, Canadian Natural Resources, are
long-established, consistently profitable businesses by most measures), and
the market-cap/profitability bar above is the intended quality filter.

Fundamentals are cached to disk (one yfinance .info call per ticker is the
slow part) so re-runs after the first are fast. Run standalone:

    python -m conviction_watchlist.quality_filter
"""
import json
import urllib.request

import yfinance as yf

from conviction_watchlist.config import INFO_CACHE_FILE, TICKERS_URL
from conviction_watchlist.settings import load_settings

FALLBACK_TICKERS = ["RY.TO", "TD.TO", "BNS.TO", "BMO.TO", "CM.TO",
                     "ENB.TO", "SU.TO", "CNQ.TO", "CP.TO", "CNR.TO"]


def fetch_can_tickers(url: str = TICKERS_URL) -> list:
    """Fetch the raw CAN ticker universe from a URL. Same fetch pattern as
    canadian_stock_screener.py's _load_tickers: urlopen, decode utf-8, strip
    blank/comment lines, fall back to a small hardcoded list on failure."""
    try:
        with urllib.request.urlopen(url) as resp:
            content = resp.read().decode("utf-8")
        return [ln.strip() for ln in content.splitlines()
                if ln.strip() and not ln.strip().startswith("#")]
    except Exception as e:
        print(f"Error fetching tickers from {url}: {e}. Using fallback tickers.")
        return FALLBACK_TICKERS


def load_cache() -> dict:
    if INFO_CACHE_FILE.exists():
        return json.loads(INFO_CACHE_FILE.read_text())
    return {}


def save_cache(cache: dict) -> None:
    INFO_CACHE_FILE.parent.mkdir(exist_ok=True)
    INFO_CACHE_FILE.write_text(json.dumps(cache, indent=2))


def fetch_info(ticker: str, cache: dict) -> dict:
    if ticker in cache:
        return cache[ticker]
    try:
        info = yf.Ticker(ticker).info
        record = {
            "marketCap": info.get("marketCap"),
            "trailingEps": info.get("trailingEps"),
            "currentPrice": info.get("currentPrice") or info.get("regularMarketPrice"),
            "shortName": info.get("shortName"),
            "sector": info.get("sector"),
        }
    except Exception as e:
        record = {"error": str(e)}
    cache[ticker] = record
    return record


def qualified_tickers(cache: dict = None) -> list:
    """Return the list of tickers passing the quality bar, using whatever is
    already cached (does not fetch new data) and the current settings
    (dashboard-adjustable). Used by entry_screener.py."""
    if cache is None:
        cache = load_cache()
    settings = load_settings()
    min_mc, min_px = settings["min_market_cap_cad"], settings["min_price"]
    out = []
    for ticker, r in cache.items():
        if "error" in r:
            continue
        mc, eps, px = r.get("marketCap"), r.get("trailingEps"), r.get("currentPrice")
        if mc and eps and px and mc >= min_mc and eps > 0 and px >= min_px:
            out.append(ticker)
    return out


def rebuild(progress_cb=None) -> dict:
    """Fetch the raw ticker universe, refresh fundamentals for every
    main-board (.TO) name (cache-backed, so only new/never-fetched tickers
    cost a network call), and return the updated cache. `progress_cb(i,
    total)` is called periodically if given -- the dashboard uses this to
    show a coarse progress note; the CLI's main() prints instead.

    Saves every 25 tickers, not just once at the end -- a slow host (e.g. a
    Raspberry Pi vs. the dev machine this was first tested on) or any
    mid-run hiccup must not lose everything fetched so far, and a second
    click of "Rebuild Quality List" then only has to fetch what's left."""
    tickers = fetch_can_tickers()
    main_board = [t for t in tickers if t.endswith(".TO")]
    cache = load_cache()
    for i, ticker in enumerate(main_board):
        fetch_info(ticker, cache)
        if i % 25 == 0:
            save_cache(cache)
            if progress_cb:
                progress_cb(i, len(main_board))
    save_cache(cache)
    return cache


def main():
    settings = load_settings()
    min_mc, min_px = settings["min_market_cap_cad"], settings["min_price"]

    def progress(i, total):
        print(f"  ...{i}/{total}")

    cache = rebuild(progress_cb=progress)
    errors = sum(1 for r in cache.values() if "error" in r)

    qualified_records = sorted(
        ((t, cache[t]) for t in qualified_tickers(cache)),
        key=lambda r: -(r[1]["marketCap"] or 0)
    )
    print(f"\n{errors} tickers failed to fetch")
    print(f"{len(qualified_records)} tickers pass the quality bar "
          f"(mkt cap >= ${min_mc:,}, EPS>0, price>=${min_px}):\n")
    for ticker, record in qualified_records:
        name = (record.get("shortName") or "")[:35]
        print(f"  {ticker:10} {name:35} mktcap=${record['marketCap']:>15,.0f}  "
              f"eps={record['trailingEps']:>6.2f}  px=${record['currentPrice']:>8.2f}  "
              f"{record.get('sector','')}")


if __name__ == "__main__":
    main()
