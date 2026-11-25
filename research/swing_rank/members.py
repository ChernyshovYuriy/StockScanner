"""
Who is who in the dated index files (holdings.py): one company id across ticker
and name changes, month-end prices from the files themselves, and the matching
Yahoo symbol, accepted only if its returns agree with the files' prices.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from research.swing_rank.holdings import yahoo_symbol

MAX_GAP_DAYS = 400               # the same ticker after a longer absence may be another company
SAME_PRICE = 0.005
MIN_OVERLAP = 6                  # months needed to accept a symbol
MAX_MEDIAN_DIFF = 0.01           # typical monthly disagreement allowed (dividends explain ~0.3%)
MIN_CORR = 0.95
SPLIT_TOLERANCE = 0.20           # on the fund's share count
SPLIT_PRICE_RANGE = (0.5, 1.7)   # the month's real move allowed on top of a split
SPLITS = (1.5, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0)
RENAME_SHARES, RENAME_PRICE = 0.03, 0.20     # same holding under a new ticker and name


def long_table(files: Dict[pd.Timestamp, pd.DataFrame]) -> pd.DataFrame:
    """One row per member per file."""
    parts = [m.reset_index().assign(date=d) for d, m in files.items()]
    return pd.concat(parts, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)


def company_ids(table: pd.DataFrame) -> pd.Series:
    """Joins rows sharing a name, or a ticker without a long absence in between,
    unless both are present in one file at different prices (two securities; the
    December 2011 file lists 82 companies twice, under old and new labels, at one
    price)."""
    parent: Dict[int, int] = {}

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    node: Dict[Tuple[str, str], int] = {}
    for key in zip(table["ticker"], table["name"]):
        node.setdefault(key, len(node))
    for v in node.values():
        parent[v] = v
    dates: Dict[int, Dict[pd.Timestamp, float]] = {v: {} for v in node.values()}
    for t, n, d, p in zip(table["ticker"], table["name"], table["date"], table["price"]):
        dates[node[(t, n)]][d] = p

    def join(a: int, b: int) -> None:
        a, b = find(a), find(b)
        if a == b or any(abs(dates[a][d] / dates[b][d] - 1.0) > SAME_PRICE for d in dates[a].keys() & dates[b].keys()):
            return
        parent[b] = a
        dates[a] = {**dates[b], **dates[a]}

    spans = table.groupby(["ticker", "name"])["date"].agg(["min", "max"]).sort_values("min")
    for _, g in spans.groupby(level="ticker", sort=False):
        keys, ends = list(g.index), g["max"].to_numpy()
        for k in range(1, len(keys)):
            prior = int(np.argmax(ends[:k]))         # the label last seen, not the one that started last
            if (g["min"].iloc[k] - g["max"].iloc[prior]).days <= MAX_GAP_DAYS:
                join(node[keys[prior]], node[keys[k]])
    for _, g in spans.groupby(level="name", sort=False):
        keys = list(g.index)
        for k in range(1, len(keys)):
            join(node[keys[k - 1]], node[keys[k]])
    return pd.Series([find(node[k]) for k in zip(table["ticker"], table["name"])], index=table.index)


def collapse(table: pd.DataFrame) -> pd.DataFrame:
    """One row per company and file: share counts and weights added up."""
    how = {c: "sum" if c in ("shares", "weight") else "first" for c in table.columns if c not in ("id", "date")}
    out = table.groupby(["date", "id"], as_index=False, sort=True).agg(how)
    return out[list(table.columns)].sort_values(["date", "ticker"]).reset_index(drop=True)


def split_factor(price_ratio: float, shares_ratio: float) -> float:
    """What a split (or consolidation) did to the price between two files: 1.0
    unless the fund's share count moved by the opposite of the price, close to a
    usual ratio. 2.0 means a 2-for-1 split."""
    if abs(price_ratio - 1.0) <= 0.25:
        return 1.0
    best = 1.0
    for k in SPLITS:
        for f in (k, 1.0 / k):
            if (abs(shares_ratio / f - 1.0) <= SPLIT_TOLERANCE
                    and SPLIT_PRICE_RANGE[0] <= price_ratio * f <= SPLIT_PRICE_RANGE[1]
                    and (best == 1.0 or abs(shares_ratio / f - 1.0) < abs(shares_ratio / best - 1.0))):
                best = f
    return best


def link_renames(table: pd.DataFrame) -> pd.Series:
    """Company ids with one more join: a company missing from the next month's
    file whose exact holding (share count within 3%, price within 20%, same
    sector) reappears there under a ticker and name never seen before. Only when
    the pairing is one-to-one."""
    ids = table["id"].copy()
    first, last = table.groupby("id")["date"].min(), table.groupby("id")["date"].max()
    files = sorted(table["date"].unique())
    rows = {d: g.drop_duplicates("id").set_index("id") for d, g in table.groupby("date")}
    merged: Dict[int, int] = {}
    for a, b in zip(files, files[1:]):
        gone = rows[a].loc[[c for c in rows[a].index if last[c] == a]]
        new = rows[b].loc[[c for c in rows[b].index if first[c] == b]]
        fits = [(c, n) for c, g in gone.iterrows() for n, h in new.iterrows()
                if abs(h["shares"] / g["shares"] - 1) <= RENAME_SHARES and abs(h["price"] / g["price"] - 1) <= RENAME_PRICE
                and h["sector"] == g["sector"]]
        for c, n in fits:
            if sum(x == c for x, _ in fits) == 1 and sum(y == n for _, y in fits) == 1:
                merged[n] = c

    def root(c: int) -> int:
        while c in merged:
            c = merged[c]
        return c
    return ids.map(root)


def file_returns(table: pd.DataFrame) -> pd.DataFrame:
    """Month-over-month price returns from the files, split-adjusted. Rows: the
    later file's date; columns: company id. Only between neighbouring months."""
    out: Dict[int, Dict[pd.Timestamp, float]] = {}
    for cid, g in table.sort_values("date").groupby("id"):
        g = g.drop_duplicates("date")
        for a, b in zip(g.itertuples(), g.iloc[1:].itertuples()):
            if (b.date - a.date).days <= 40:
                f = split_factor(b.price / a.price, b.shares / a.shares)
                out.setdefault(cid, {})[b.date] = b.price * f / a.price - 1.0
    return pd.DataFrame(out).sort_index()


def candidate_symbols(table: pd.DataFrame) -> List[str]:
    """Every Yahoo symbol any member might have had."""
    ok = [t for t in table["ticker"].unique() if re.fullmatch(r"[A-Z]{1,5}([./ ][A-Z]{1,2})?", t)]
    return sorted({yahoo_symbol(t) for t in ok})


def month_end_returns(close: pd.DataFrame, dates: List[pd.Timestamp]) -> pd.DataFrame:
    """Close-to-close return of every column between consecutive file dates at
    most 40 days apart."""
    at = close.reindex(close.index.union(dates)).ffill(limit=5).loc[dates]
    r = at / at.shift(1) - 1.0
    gap = pd.Series(dates, index=dates).diff().dt.days
    r[(gap > 40).to_numpy()] = np.nan
    return r


def agreement(files: pd.Series, yahoo: pd.Series) -> Tuple[int, float, float]:
    """(months in common, median absolute difference, correlation)."""
    z = pd.concat([files, yahoo], axis=1, sort=True).dropna()
    if len(z) < 3:
        return len(z), np.nan, np.nan
    d = (z.iloc[:, 0] - z.iloc[:, 1]).abs().median()
    return len(z), float(d), float(z.iloc[:, 0].corr(z.iloc[:, 1]))


def accepted(files: pd.Series, yahoo: pd.Series) -> bool:
    n, d, c = agreement(files, yahoo)
    return bool(n >= MIN_OVERLAP and d <= MAX_MEDIAN_DIFF and c >= MIN_CORR)


def match_symbols(table: pd.DataFrame, from_files: pd.DataFrame, yahoo_r: pd.DataFrame) -> Dict[int, str]:
    """{company id: Yahoo symbol}. First the id's own tickers, newest first; then
    any symbol whose returns fit (renamed companies). A symbol serves one id."""
    out: Dict[int, str] = {}
    used: Set[str] = set()
    own: Dict[int, List[str]] = {}
    for cid, g in table.sort_values("date").groupby("id"):
        own[cid] = [yahoo_symbol(t) for t in reversed(list(dict.fromkeys(g["ticker"])))]
    ids = [c for c in own if c in from_files.columns]
    for cid in ids:
        for s in own[cid]:
            if s in yahoo_r.columns and s not in used and accepted(from_files[cid], yahoo_r[s]):
                out[cid] = s
                used.add(s)
                break
    for cid in ids:
        if cid in out:
            continue
        best: Optional[Tuple[float, str]] = None
        for s in yahoo_r.columns:
            if s not in used and accepted(from_files[cid], yahoo_r[s]):
                n, d, c = agreement(from_files[cid], yahoo_r[s])
                if n >= 12 and c >= 0.98 and (best is None or d < best[0]):
                    best = (d, s)
        if best:
            out[cid] = best[1]
            used.add(best[1])
    return out
