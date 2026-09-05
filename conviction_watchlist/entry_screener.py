"""
Daily entry screener -- see conviction_watchlist/__init__.py for the
package's overall purpose.

Checks the quality watchlist (conviction_watchlist.quality_filter) against
the entry rule chosen 2026-09-05: flag a ticker as a buy candidate when its
current price is >= settings["dip_pct_off_high"] below its trailing 52-week
high (adjustable via the dashboard's Conviction tab).

Results are persisted to config.CANDIDATES_FILE with a timestamp so the
dashboard can show the last computed result without re-running the (slow --
one yfinance history call per ticker) scan on every page load.

Run standalone, ideally once a day after close:

    python -m conviction_watchlist.entry_screener
"""
import json
from datetime import datetime, timezone

import yfinance as yf

from conviction_watchlist.config import CANDIDATES_FILE
from conviction_watchlist.quality_filter import load_cache, qualified_tickers
from conviction_watchlist.settings import load_settings


def compute_candidates(progress_cb=None) -> list:
    """Scan every quality-watchlist ticker and return the ones that are
    >= dip_pct_off_high below their 52-week high, sorted deepest-first.
    `progress_cb(i, total)` is called periodically if given."""
    dip_pct_off_high = load_settings()["dip_pct_off_high"]
    tickers = qualified_tickers()
    cache = load_cache()  # for sector, already fetched by the quality filter

    candidates = []
    for i, ticker in enumerate(tickers):
        try:
            hist = yf.Ticker(ticker).history(period="1y", interval="1d", auto_adjust=True)
        except Exception:
            continue
        if hist.empty:
            continue
        high_52w = float(hist["Close"].max())
        current = float(hist["Close"].iloc[-1])
        pct_off_high = (high_52w - current) / high_52w
        if pct_off_high >= dip_pct_off_high:
            sector = cache.get(ticker, {}).get("sector") or "Unknown"
            candidates.append({"ticker": ticker, "current": current,
                                "high_52w": high_52w, "pct_off_high": pct_off_high,
                                "sector": sector})
        if progress_cb and i % 25 == 0:
            progress_cb(i, len(tickers))

    candidates.sort(key=lambda r: -r["pct_off_high"])
    return candidates


def refresh_and_save(progress_cb=None) -> dict:
    """compute_candidates() plus persistence -- the single code path used by
    both the CLI and the dashboard's "Refresh" button, so a CLI run also
    leaves the dashboard with fresh results to show."""
    candidates = compute_candidates(progress_cb=progress_cb)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dip_pct_off_high": load_settings()["dip_pct_off_high"],
        "candidates": candidates,
    }
    CANDIDATES_FILE.parent.mkdir(exist_ok=True)
    CANDIDATES_FILE.write_text(json.dumps(payload, indent=2))
    return payload


def load_last_result() -> dict:
    """Read whatever refresh_and_save() last wrote, without recomputing.
    Returns {"generated_at": None, "candidates": []} if never run."""
    if not CANDIDATES_FILE.exists():
        return {"generated_at": None, "dip_pct_off_high": None, "candidates": []}
    return json.loads(CANDIDATES_FILE.read_text())


def main():
    dip_pct_off_high = load_settings()["dip_pct_off_high"]
    tickers = qualified_tickers()
    print(f"Checking {len(tickers)} quality-watchlist tickers for a "
          f">= {dip_pct_off_high:.0%} pullback off the 52-week high...\n")

    def progress(i, total):
        print(f"  ...{i}/{total}")

    payload = refresh_and_save(progress_cb=progress)
    candidates = payload["candidates"]
    print(f"\n{len(candidates)} buy candidates (>= {dip_pct_off_high:.0%} off 52-week high):\n")
    for c in candidates:
        print(f"  {c['ticker']:10} now=${c['current']:>8.2f}  "
              f"52w high=${c['high_52w']:>8.2f}  off high={c['pct_off_high']:>6.1%}  "
              f"{c.get('sector', '')}")


if __name__ == "__main__":
    main()
