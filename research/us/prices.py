"""
Unadjusted daily bars of the US list, in cache/us_prices.db. The paced, resumable
download and the stop when Yahoo refuses are research/vcp/prices.py's.

    python -m research.us.prices [--until HH:MM] [--retry]
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from research.breakouts.prices import sync_unadjusted
from research.us.universe import load_tickers
from research.vcp import prices as vp

DB = Path(__file__).resolve().parents[2] / "cache" / "us_prices.db"
START, END = "2011-01-01", "2026-11-24"
SLICE = 50


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Unadjusted bars of the US list")
    p.add_argument("--until", help="stop at this time of day (resume later)")
    p.add_argument("--retry", action="store_true")
    a = p.parse_args(argv)
    names = ["SPY", *load_tickers()]
    conn, pacer = vp.connect(str(DB)), vp.Pacer(vp.YAHOO_PER_SECOND)
    try:
        done = vp.statuses(conn)
        todo = [t for t in names if t not in done or (a.retry and done[t] == vp.NO_DATA)]
        for i in range(0, len(todo), SLICE):
            if a.until and datetime.now().strftime("%H:%M") >= a.until:
                print(f"stopped at {a.until}: {len(todo) - i} names left", flush=True)
                return 0
            out = vp.download(conn, todo[i:i + SLICE], START, END, retry=a.retry, sync_fn=sync_unadjusted, pacer=pacer, progress=lambda m: None)
            print(datetime.now().strftime("%H:%M"), f"{min(i + SLICE, len(todo))}/{len(todo)}", out, flush=True)
        return 0
    except vp.YahooBlocked as exc:
        print(f"STOPPED, Yahoo is refusing requests: {exc}. Re-run later to resume.", flush=True)
        return 2
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
