"""
analysis/fetch_reference_data.py -- one-off Yahoo fetch of the reference
data the idle-cash analysis needs, saved to disk so analysis/trade_vs_xiu.py
runs offline and reproducibly:

  ref/xiu_cash_daily.csv   XIU.TO and CASH.TO daily OHLC, freshly downloaded
                           with auto_adjust=True in ONE call -- a consistent
                           total-return basis as of today (the local market
                           cache only tops up its tail, so older bars there
                           miss dividends paid after each ticker's first fetch)
  ref/dividends.csv        cash dividends (ticker, ex_date, amount) for every
                           ticker the live paper account ever held

Paced (~1 request / 3 s) and stops at the first rate-limit error, per the
project's Yahoo rule. All calls go through market_data.DEFAULT_PROVIDER.

Usage:
    python analysis/fetch_reference_data.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from market_data import DEFAULT_PROVIDER  # noqa: E402

REF_DIR = ROOT / "reports" / "idle_cash" / "ref"
START = "2020-09-01"
END = "2026-10-28"           # yfinance end is exclusive
PAUSE_S = 3.0


def main() -> None:
    REF_DIR.mkdir(parents=True, exist_ok=True)

    data, failed = DEFAULT_PROVIDER.download_range(["XIU.TO", "CASH.TO"], START, END)
    if failed:
        sys.exit(f"Benchmark download failed for {failed} -- stopping.")
    frames = [df.assign(ticker=t).rename_axis("date").reset_index() for t, df in data.items()]
    pd.concat(frames).to_csv(REF_DIR / "xiu_cash_daily.csv", index=False)
    for t, df in data.items():
        print(f"{t}: {len(df)} bars {df.index[0].date()}..{df.index[-1].date()}")

    con = duckdb.connect(str(ROOT / "data" / "trading.db"), read_only=True)
    held = [r[0] for r in con.execute("SELECT DISTINCT ticker FROM transactions ORDER BY 1").fetchall()]
    con.close()

    rows = []
    for i, ticker in enumerate(held, 1):
        time.sleep(PAUSE_S)
        try:
            div = DEFAULT_PROVIDER.get_dividends(ticker)
        except Exception as exc:  # rate limit or network: stop, don't hammer
            sys.exit(f"[{i}/{len(held)}] {ticker}: {exc!r} -- stopping, {len(rows)} dividend rows not saved.")
        for ts, amount in div.items():
            rows.append({"ticker": ticker, "ex_date": pd.Timestamp(ts).date(), "amount": float(amount)})
        print(f"[{i}/{len(held)}] {ticker}: {len(div)} dividends")
    pd.DataFrame(rows, columns=["ticker", "ex_date", "amount"]).to_csv(REF_DIR / "dividends.csv", index=False)
    print(f"Saved -> {REF_DIR}")


if __name__ == "__main__":
    main()
