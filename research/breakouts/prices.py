"""
Daily bars without dividend adjustment (split-adjusted only) for the breakout
study, in cache/breakout_prices.db. The paced, resumable download and the stop
when Yahoo refuses are research/vcp/prices.py's.

    python -m research.breakouts.prices --tickers FILE
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

import market_data_cache as mdc
from market_data import LiveDataProvider
from research.vcp import prices as vp

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "cache" / "breakout_prices.db"
PERIOD = "15y"
SEAM = 2.5                       # a one-day rise this large is an unadjusted consolidation (TLRY x9.8, WEED x52)


def sync_unadjusted(tickers: List[str], start: str, end: str, conn=None, force_refresh: bool = False,
                    quiet: bool = True, fetch=None) -> None:
    """Drop-in for market_data_cache.sync_tickers: unadjusted bars in [start, end]."""
    fetch = fetch or LiveDataProvider().download_batch_with_reasons
    data, _ = fetch(tickers, period=PERIOD, interval="1d", auto_adjust=False)
    for t, d in data.items():
        d = d.dropna()
        d = d[(d["Close"] > 0) & (d["Volume"] >= 0)].loc[start:end]
        if len(d):
            mdc._upsert(conn, t, d)


def after_seam(d: pd.DataFrame) -> pd.DataFrame:
    """Bars from the last seam on. Falls are kept: a crash is a real loss."""
    jump = (d["Close"] / d["Close"].shift(1) >= SEAM).to_numpy()
    return d.iloc[jump.nonzero()[0][-1]:] if jump.any() else d


def load(db: Path = DB) -> Dict[str, pd.DataFrame]:
    conn = vp.connect(str(db))
    try:
        return {t: after_seam(d) for t, d in vp.load_all(conn).items()}
    finally:
        conn.close()


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Unadjusted bars for the breakout study")
    p.add_argument("--tickers", required=True)
    p.add_argument("--retry", action="store_true")
    a = p.parse_args(argv)
    names = [t.strip() for t in Path(a.tickers).read_text().splitlines() if t.strip()]
    conn = vp.connect(str(DB))
    try:
        print(vp.download(conn, names, "2011-01-01", "2026-11-24", retry=a.retry, sync_fn=sync_unadjusted,
                          progress=lambda m: print(m, flush=True)))
        return 0
    except vp.YahooBlocked as exc:
        print(f"STOPPED, Yahoo is refusing requests: {exc}. Re-run later to resume.")
        return 2
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
