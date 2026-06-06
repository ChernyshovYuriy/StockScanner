"""
CLI for the EDGAR screener.

  # one-off look at a single name (fundamentals + insider buys)
  python -m edgar.run ticker MU

  # set your watchlist (comma-separated tickers)
  python -m edgar.run watchlist MU,KEY,AMD,PLTR

  # scan recent days for watchlist filings (insiders, activists, 8-Ks)
  python -m edgar.run scan --days 5

  # scan a single day, ALL filers, only activist stakes (no watchlist filter)
  python -m edgar.run scan --days 1 --forms "SC 13D,SC 13D/A" --all

Set USER_AGENT in edgar/core.py to a real contact first, or EDGAR returns 403.
"""

import argparse
from datetime import date, timedelta

from edgar import store
from edgar.core import load_ticker_map, cik_for
from edgar.fundamentals import get_fundamentals
from edgar.insiders import get_recent_insider_activity, open_market_buys, cluster_flag
from edgar.scanner import scan_range


def cmd_ticker(args):
    tmap = load_ticker_map()
    cik = cik_for(args.ticker, tmap)
    print(f"{args.ticker} -> CIK {cik}\n")

    f = get_fundamentals(cik, pin_period=True)
    print(f"FUNDAMENTALS  ({f['entity']})  target period {f.get('target_period', '?')}")
    for m, v in f["metrics"].items():
        print(f"  {m:16s} {v['val']:>20,.2f}   ({v['end']})" if v
              else f"  {m:16s} <not tagged>")
    for flag in f["period_flags"]:
        print(f"  ! period mismatch: {flag}")

    print("\nINSIDER OPEN-MARKET BUYS")
    buys = open_market_buys(get_recent_insider_activity(cik))
    if not buys:
        print("  none in recent Form 4s")
    else:
        for b in buys:
            print(f"  {b['owner']:24s} {b['shares']:>10,.0f} @ {b['price']}  {b['date']}")
        if cluster_flag(buys):
            print("  >> CLUSTER FLAG: 2+ insiders buying")


def cmd_watchlist(args):
    tmap = load_ticker_map()
    pairs = []
    for t in args.tickers.split(","):
        t = t.strip().upper()
        try:
            pairs.append((t, cik_for(t, tmap)))
        except KeyError:
            print(f"  skip {t}: not a US EDGAR filer")
    conn = store.connect()
    store.set_watchlist(conn, pairs)
    print(f"watchlist set: {[t for t, _ in pairs]}")


def cmd_scan(args):
    conn = store.connect()
    wl = None if args.all else store.watchlist_ciks(conn)
    if wl is not None and not wl:
        print("watchlist empty -- set one first or use --all")
        return
    forms = [x.strip() for x in args.forms.split(",")] if args.forms else None
    end = date.today()
    start = end - timedelta(days=args.days)
    hits = scan_range(start, end, watchlist_ciks=wl, forms=forms)
    real = [h for h in hits if "error" not in h]
    store.save_scan_hits(conn, real)
    print(f"{len(real)} filings {start}..{end}\n")
    for h in sorted(real, key=lambda x: x.get("date", "")):
        tic = h.get("ticker") or f"CIK{h['cik']}"
        print(f"  {h['date']}  {tic:8s} {h['form']:10s} {h['category']}")
        print(f"            {h['url']}")


def main():
    p = argparse.ArgumentParser(description="EDGAR screener")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("ticker");
    pt.add_argument("ticker");
    pt.set_defaults(fn=cmd_ticker)
    pw = sub.add_parser("watchlist");
    pw.add_argument("tickers");
    pw.set_defaults(fn=cmd_watchlist)
    ps = sub.add_parser("scan")
    ps.add_argument("--days", type=int, default=3)
    ps.add_argument("--forms", default=None)
    ps.add_argument("--all", action="store_true", help="ignore watchlist, scan all filers")
    ps.set_defaults(fn=cmd_scan)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
