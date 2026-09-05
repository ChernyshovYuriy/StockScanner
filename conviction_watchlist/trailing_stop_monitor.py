"""
Trailing-stop monitor -- see conviction_watchlist/__init__.py for the
package's overall purpose.

Tracks the highest daily close since each position's entry and flags a SELL
once price has fallen settings["trailing_stop_pct"] (25% by default, picked
2026-09-05, adjustable via the dashboard's Conviction tab) from that peak.
Deliberately has NO profit-target exit -- see the
dip-bounce-grid-rejected-2026-09 memory entry: every variant of "sell once
up 5-10%" tested this session lost to just holding, including 0/8 walk-
forward years on AAPL. The only exit trigger here is a real reversal from
the peak, matching the "buy-and-hold sized for conviction" approach the user
actually runs.

Holdings come from holdings_store.py (data/conviction_holdings.json) --
updated by hand, or via the dashboard's Conviction tab, as positions are
opened/closed in the real account. This module never places or simulates a
trade itself, only reports status.

Run standalone, ideally once a day after close:

    python -m conviction_watchlist.trailing_stop_monitor
"""
import yfinance as yf

from conviction_watchlist.holdings_store import load_holdings
from conviction_watchlist.settings import load_settings


def compute_status() -> list:
    """Return per-holding trailing-stop status. Used by both the CLI and the
    dashboard's Conviction tab."""
    trailing_stop_pct = load_settings()["trailing_stop_pct"]
    holdings = load_holdings()
    results = []
    for h in holdings:
        ticker, entry_date, entry_price = h["ticker"], h["entry_date"], h["entry_price"]
        try:
            hist = yf.Ticker(ticker).history(start=entry_date, interval="1d", auto_adjust=True)
        except Exception as e:
            results.append({**h, "error": str(e)})
            continue
        if hist.empty:
            results.append({**h, "error": f"no data since {entry_date}"})
            continue
        peak = float(hist["Close"].max())
        current = float(hist["Close"].iloc[-1])
        drop_from_peak = (peak - current) / peak
        gain_vs_entry = (current - entry_price) / entry_price
        results.append({
            **h,
            "peak": peak,
            "current": current,
            "drop_from_peak": drop_from_peak,
            "gain_vs_entry": gain_vs_entry,
            "flagged": drop_from_peak >= trailing_stop_pct,
        })
    return results


def main():
    trailing_stop_pct = load_settings()["trailing_stop_pct"]
    results = compute_status()
    print(f"Checking {len(results)} holdings against a {trailing_stop_pct:.0%} "
          f"trailing stop from the peak since entry...\n")
    for r in results:
        if "error" in r:
            print(f"  {r['ticker']:10} {r['error']}")
            continue
        flag = "SELL -- trailing stop triggered" if r["flagged"] else "hold"
        print(f"  {r['ticker']:10} [{r.get('account','')}]  entry=${r['entry_price']:>8.2f} ({r['entry_date']})  "
              f"peak=${r['peak']:>8.2f}  now=${r['current']:>8.2f}  off peak={r['drop_from_peak']:>6.1%}  "
              f"vs entry={r['gain_vs_entry']:>+7.1%}  -> {flag}")


if __name__ == "__main__":
    main()
