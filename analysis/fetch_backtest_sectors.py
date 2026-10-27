"""
analysis/fetch_backtest_sectors.py -- one-off Yahoo fetch of the GICS sector
for every ticker the Phase 1 backtest traded (01_trade_vs_xiu/trades_with_excess.csv),
so Phase 5 of the idle-cash plan can slice by sector offline.

Starts from the sector maps already on disk (ref/live_sectors.json,
ref/sector_cache_pi.json) and fetches only the rest via
market_data.DEFAULT_PROVIDER.get_info(), paced at one request / 3 s. Stops at
the first rate-limit error and saves what it has. A ticker Yahoo no longer
knows (delisted) or that has no sector is recorded as "Unknown".

Usage:
    python analysis/fetch_backtest_sectors.py    -> ref/backtest_sectors.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from market_data import DEFAULT_PROVIDER  # noqa: E402

REF = ROOT / "reports" / "idle_cash" / "ref"
OUT = REF / "backtest_sectors.json"
PAUSE_S = 3.0


def main() -> None:
    trades = pd.read_csv(ROOT / "reports" / "idle_cash" / "01_trade_vs_xiu" / "trades_with_excess.csv")
    tickers = sorted(set(trades.loc[trades.source == "backtest", "ticker"]))
    known: dict = {}
    for name in ("sector_cache_pi.json", "live_sectors.json"):
        known.update(json.loads((REF / name).read_text()))
    if OUT.exists():
        known.update(json.loads(OUT.read_text()))
    sectors = {t: known[t] for t in tickers if known.get(t) not in (None, "", "Unknown")}
    todo = [t for t in tickers if t not in sectors]
    print(f"{len(tickers)} tickers, {len(sectors)} already mapped, fetching {len(todo)}", flush=True)
    for i, t in enumerate(todo, 1):
        time.sleep(PAUSE_S)
        try:
            info = DEFAULT_PROVIDER.get_info(t) or {}
        except Exception as exc:
            msg = repr(exc)
            if "429" in msg or "Too Many" in msg or "Rate" in msg:
                print(f"[{i}/{len(todo)}] {t}: rate limited -- stopping", flush=True)
                break
            info = {}
        sectors[t] = info.get("sector") or "Unknown"
        print(f"[{i}/{len(todo)}] {t}: {sectors[t]}", flush=True)
        OUT.write_text(json.dumps(dict(sorted(sectors.items())), indent=1))
    OUT.write_text(json.dumps(dict(sorted(sectors.items())), indent=1))
    print(f"Saved {len(sectors)} -> {OUT}")


if __name__ == "__main__":
    main()
