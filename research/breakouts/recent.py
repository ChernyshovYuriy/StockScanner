"""
Confirmed breakouts in the last sessions: facts about past bars, not forecasts
(they did not beat plain new highs in the backtest, see PLAN.md). Reads
cache/breakout_prices.db, no network.

    python -m research.breakouts.recent            # last 5 sessions, liquid names
    python -m research.breakouts.recent 20 CAE.TO  # last 20, these names
"""
from __future__ import annotations

import sys
from typing import Dict, List, Optional

import pandas as pd

from research.breakouts.backtest import MIN_DOLLAR, MIN_PRICE, XIU
from research.breakouts.detector import scan
from research.breakouts.features import slope200

COLUMNS = ["date", "ticker", "avg200", "line", "confirm_at", "close", "above_pct", "tests", "months", "lfd_low", "target"]


def recent(data: Dict[str, pd.DataFrame], sessions: int = 5, tickers: Optional[List[str]] = None) -> pd.DataFrame:
    cal = data[XIU].index
    first = cal[-sessions]
    rows = []
    for t, d in data.items():
        if t == XIU or (tickers and t not in tickers):
            continue
        dollar = (d["Close"] * d["Volume"]).rolling(20).mean()
        for b in scan(d):
            day = d.index[b.bar]
            if day < first or (not tickers and not (b.close >= MIN_PRICE and dollar.iloc[b.bar] >= MIN_DOLLAR)):
                continue
            slope = slope200(d["Close"].to_numpy(float), b.bar)
            avg200 = "?" if slope != slope else "rising" if slope > 0 else "falling"
            rows.append([day.date().isoformat(), t, avg200, round(b.level, 3), round(b.confirm, 3), round(b.close, 3),
                         round((b.close / b.level - 1) * 100, 1), b.tests, round(b.bars / 21, 1), round(b.lfd_low, 3), round(b.target, 3)])
    return pd.DataFrame(rows, columns=COLUMNS).sort_values(["date", "ticker"], ascending=[False, True]).reset_index(drop=True)


def main(argv: Optional[List[str]] = None) -> int:
    from research.breakouts import prices
    args = sys.argv[1:] if argv is None else argv
    sessions = int(args[0]) if args and args[0].isdigit() else 5
    names = [a.upper() for a in args if not a.isdigit()]
    data = prices.load()
    out = recent(data, sessions, names or None)
    print(f"data to {data[XIU].index[-1].date()}")
    print(out.to_string(index=False) if len(out) else "no confirmed breakouts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
