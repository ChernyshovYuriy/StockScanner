"""
Point-in-time S&P/TSX Composite membership, from BlackRock's dated holdings
files for XIC (iShares Core S&P/TSX Capped Composite), one per month-end since
December 2006. Every member is there as it was: sector, shares held and closing
price, including names that were later delisted, acquired or renamed.

Files are kept under cache/xic_holdings/ (gitignored), fetched once, slowly.

    python -m research.swing_rank.holdings            # fetch what is missing
"""
from __future__ import annotations

import csv
import io
import time
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[2] / "cache" / "xic_holdings"
URL = ("https://www.blackrock.com/ca/investors/en/products/239837/ishares-sptsx-capped-composite-index-etf/"
       "1464253357814.ajax?fileType=csv&fileName=XIC_holdings&dataType=fund&asOfDate={day}")
FIRST_MONTH = "2006-12"
PAUSE_SECONDS = 6.0
MAX_BACK_DAYS = 5                 # a month-end holiday: try the last this many weekdays (never leaves the month)
MIN_MEMBERS = 150
COLUMNS = ["ticker", "name", "sector", "shares", "price", "weight", "exchange"]


class FetchFailed(RuntimeError):
    """The site stopped giving usable answers; stop instead of hammering it."""


def _number(text: str) -> Optional[float]:
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def parse(text: str) -> tuple:
    """(as-of date or None, members) of one holdings file: equities priced in CAD.
    The Exchange column can't be used: files before about 2017 label interlisted
    members by a foreign listing (RY as New York, Manulife as Hong Kong) while the
    price is Toronto's."""
    rows = list(csv.reader(io.StringIO(text.lstrip("﻿"))))
    as_of = None
    for r in rows[:3]:
        if r and r[0].strip().startswith("Fund Holdings as of") and len(r) > 1:
            as_of = pd.to_datetime(r[1], errors="coerce")
            as_of = None if pd.isna(as_of) else as_of
    header = next((k for k, r in enumerate(rows) if r and r[0].strip() == "Ticker"), None)
    if header is None:
        return as_of, pd.DataFrame(columns=COLUMNS)
    at = {name.strip(): k for k, name in enumerate(rows[header])}
    out = []
    for r in rows[header + 1:]:
        if len(r) < len(at) or r[at["Asset Class"]] != "Equity":
            continue
        price, shares = _number(r[at["Price"]]), _number(r[at["Shares"]])
        if not price or not shares or r[at["Currency"]] != "CAD":
            continue
        out.append((r[at["Ticker"]].strip(), r[at["Name"]].strip(), r[at["Sector"]].strip(),
                    shares, price, _number(r[at["Weight (%)"]]), r[at["Exchange"]].strip()))
    return as_of, pd.DataFrame(out, columns=COLUMNS).drop_duplicates(["ticker", "name"])


def month_ends(first: str, last: pd.Timestamp) -> List[pd.Timestamp]:
    """Last weekday of every complete month from `first` through `last`."""
    ends = pd.date_range(first, last, freq="BME")
    return [d for d in ends if d <= last]


def _get(day: str) -> str:
    request = urllib.request.Request(URL.format(day=day), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as resp:
        return resp.read().decode("utf-8", errors="replace")


def fetch(last: pd.Timestamp, first: str = FIRST_MONTH, cache_dir: Path = CACHE_DIR,
          get: Callable[[str], str] = _get, sleep=time.sleep, progress=print) -> Dict[str, int]:
    """Download every month-end not already saved. A month is saved only if the
    file's own date is the one asked for and it has a plausible member count. A
    recognizable but empty file is a gap in what BlackRock published; three
    months in a row of anything else stop the run."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    have = {p.stem[:6] for p in cache_dir.glob("*.csv")}
    fetched, missing, misses_in_a_row = 0, 0, 0
    for end in month_ends(first, last):
        if end.strftime("%Y%m") in have:
            continue
        ok = gap = False
        for back in range(MAX_BACK_DAYS):
            day = (end - pd.offsets.BDay(back))
            sleep(PAUSE_SECONDS)
            text = get(day.strftime("%Y%m%d"))
            as_of, members = parse(text)
            if as_of == day.normalize() and len(members) >= MIN_MEMBERS:
                (cache_dir / f"{day:%Y%m%d}.csv").write_text(text)
                ok = True
                break
            gap = as_of is None and members.empty and "Fund Holdings as of" in text
        fetched, missing = fetched + ok, missing + (not ok)
        misses_in_a_row = 0 if ok or gap else misses_in_a_row + 1
        if misses_in_a_row >= 3:
            raise FetchFailed(f"three months in a row unavailable, up to {end:%Y-%m}")
        if fetched and fetched % 12 == 0 and ok:
            progress(f"{fetched} fetched, at {end:%Y-%m}")
    return {"fetched": fetched, "missing": missing}


def load(cache_dir: Path = CACHE_DIR) -> Dict[pd.Timestamp, pd.DataFrame]:
    """{as-of date: members indexed by XIC ticker}, oldest first."""
    out = {}
    for p in sorted(cache_dir.glob("*.csv")):
        as_of, members = parse(p.read_text())
        out[as_of] = members.set_index("ticker")
    return out


def yahoo_symbol(ticker: str) -> str:
    """XIC's TECK.B / REI.UN -> Yahoo's TECK-B.TO / REI-UN.TO."""
    return ticker.replace(".", "-").replace("/", "-").replace(" ", "-") + ".TO"


if __name__ == "__main__":
    print(fetch(pd.Timestamp.now().normalize() - pd.Timedelta(days=1), progress=lambda m: print(m, flush=True)))
