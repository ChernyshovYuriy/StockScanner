"""
The US stock list: every common stock and ADR in the official Nasdaq Trader symbol
directory (Nasdaq, NYSE, NYSE American), no size filter: a filter on today's size would
pick the past's winners. The tests choose by price and volume on each past day. Also the
S&P 500's past members (fja05680/sp500), by date.

    python -m research.us.universe NASDAQLISTED OTHERLISTED SP500_START_END   # writes the lists below
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

HERE = Path(__file__).parent
TICKERS, SP500 = HERE / "tickers.csv", HERE / "sp500_members.csv"
SP500_FROM = "2011-11-01"

_NOT = re.compile(r"warrant|\brights?\b|\bunits?\b|preferred|preference|\bnotes?\b|debenture|\bbonds?\b|\betn\b|\bfund\b|acquisition|"
                  r"blank check|subordinat|\d%|perpetual|contingent|depositary shares?,? (each )?representing (a |an )?(1/|one-)|interests? in a share", re.I)
_YES = re.compile(r"common stock|common shares?|ordinary shares?|class [a-c]\b|american depositary|\badss?\b|\badrs?\b|shares of beneficial|"
                  r"common units|capital stock|voting shares|subordinate voting", re.I)
_SYMBOL = re.compile(r"[A-Z]{1,5}(\.[A-C])?")
EXCHANGES = {"N": "NYSE", "A": "NYSE American"}


def is_common(symbol: str, name: str) -> bool:
    """A common stock or ADR, judged from its symbol and its official name."""
    text = re.sub(r"subordinate voting", "voting shares", name, flags=re.I)
    return bool(_SYMBOL.fullmatch(symbol) and _YES.search(text) and not _NOT.search(text))


def yahoo(symbol: str) -> str:
    return symbol.replace(".", "-")


def read(path: Path, symbol_col: str) -> pd.DataFrame:
    d = pd.read_csv(path, sep="|", dtype=str, keep_default_na=False)
    d = d[~d[symbol_col].str.startswith("File Creation")]
    return d[(d["Test Issue"] == "N") & (d["ETF"] == "N")]


def build(nasdaq: Path, other: Path) -> pd.DataFrame:
    a, b = read(nasdaq, "Symbol"), read(other, "ACT Symbol")
    b = b[b["Exchange"].isin(EXCHANGES)]
    rows = pd.concat([pd.DataFrame({"symbol": a["Symbol"], "name": a["Security Name"], "exchange": "Nasdaq"}),
                      pd.DataFrame({"symbol": b["ACT Symbol"], "name": b["Security Name"], "exchange": b["Exchange"].map(EXCHANGES)})])
    rows = rows[[is_common(s, n) for s, n in zip(rows["symbol"], rows["name"])]].drop_duplicates("symbol")
    rows.insert(0, "ticker", rows["symbol"].map(yahoo))
    return rows.drop(columns="symbol").sort_values("ticker").reset_index(drop=True)


def members(path: Path) -> pd.DataFrame:
    """S&P 500 spells that reach past SP500_FROM; an open end is today."""
    d = pd.read_csv(path, dtype=str, keep_default_na=False)
    d = d[(d["end_date"] == "") | (d["end_date"] >= SP500_FROM)]
    return pd.DataFrame({"ticker": d["ticker"].map(yahoo), "start": d["start_date"], "end": d["end_date"]}).sort_values(["ticker", "start"]).reset_index(drop=True)


def load_tickers() -> List[str]:
    return sorted(set(pd.read_csv(TICKERS, keep_default_na=False)["ticker"]) | set(pd.read_csv(SP500, keep_default_na=False)["ticker"]))


def main(argv: Optional[List[str]] = None) -> int:
    nasdaq, other, sp = (Path(x) for x in (argv or sys.argv[1:]))
    t, m = build(nasdaq, other), members(sp)
    t.to_csv(TICKERS, index=False)
    m.to_csv(SP500, index=False)
    print(len(t), t["exchange"].value_counts().to_dict(), "| S&P 500 since", SP500_FROM, m["ticker"].nunique(), "names,",
          m[~m["ticker"].isin(t["ticker"])]["ticker"].nunique(), "of them not listed today")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
