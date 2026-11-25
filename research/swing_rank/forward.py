"""
Forward record of S1 (PLAN.md, "Forward record"): each month-end, the ten names
S1 would hold, written down before they could be bought. Paper only: no money, no
service, no database of any sleeve. The list goes to reports/swing_rank/
forward_s1.csv, which is meant to be committed the same day: the commit is the
proof that it was made in advance.

    python -m research.swing_rank.forward record            # after a month's last close
    python -m research.swing_rank.forward record --dry-run  # show the list, write nothing
    python -m research.swing_rank.forward score             # results so far
"""
from __future__ import annotations

import argparse
import time
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import pandas as pd

from research.swing_rank import engine as en
from research.swing_rank import holdings
from research.swing_rank import members as mb
from research.swing_rank.holdings import yahoo_symbol
from research.swing_rank.stocks import COST, DEFAULT, XIU, Scores, StockData
from research.vcp import prices
from time_utils import last_trading_day_on_or_before, market_now, next_trading_day, previous_trading_day

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"
RECORD = ROOT / "reports" / "swing_rank" / "forward_s1.csv"
COLUMNS = ["signal_date", "recorded_at", "entry_date", "file_date", "rank", "ticker", "symbol", "name", "sector",
           "momentum", "close"]
PRICE_START = "2005-01-01"
CLOSE_HOUR = 16.5                # bars are taken as final half an hour after the close
MAX_LATE = 5                     # sessions an entry may be late; after that the month is skipped
MAX_FILE_AGE = 10                # days before the signal the member list may be from
MIN_FRESH = 0.90                 # members whose prices must reach the signal session


class NotNow(RuntimeError):
    """The list can't be recorded at this moment."""


# ── When ─────────────────────────────────────────────────────────────────────

def signal_session(now: datetime) -> date:
    """Last session of the latest month that is over at `now` (after its final close)."""
    today = now.date()
    day = last_trading_day_on_or_before(today)
    if day == today and now.hour + now.minute / 60 < CLOSE_HOUR:
        day = previous_trading_day(day)
    while next_trading_day(day).month == day.month:
        day = previous_trading_day(day)
    return day


def entry_session(now: datetime) -> date:
    """First session that opens after `now`."""
    today = now.date()
    if last_trading_day_on_or_before(today) == today and (now.hour, now.minute) < (9, 30):
        return today
    return next_trading_day(today)


def sessions_late(signal: date, entry: date) -> int:
    """0 when the entry is the session right after the signal."""
    n, day = 0, next_trading_day(signal)
    while day < entry:
        n, day = n + 1, next_trading_day(day)
    return n


# ── What ─────────────────────────────────────────────────────────────────────

def member_file(signal: date, get: Callable[[str], str] = holdings._get, sleep: Callable[[float], None] = time.sleep
                ) -> Tuple[pd.Timestamp, pd.DataFrame]:
    """The latest member list published up to the signal session."""
    for back in range(holdings.MAX_BACK_DAYS):
        day = pd.Timestamp(signal) - pd.offsets.BDay(back)
        sleep(holdings.PAUSE_SECONDS)
        as_of, m = holdings.parse(get(day.strftime("%Y%m%d")))
        if as_of == day and len(m) >= holdings.MIN_MEMBERS:
            return as_of, m.set_index("ticker")
    raise NotNow("no member list published for the last days of the month yet")


def with_next_session(yahoo: Dict[str, pd.DataFrame], signal: date) -> Dict[str, pd.DataFrame]:
    """Bars up to the signal session, and an empty next session so that the month
    counts as complete before that session has traded."""
    last = pd.Timestamp(signal)
    cut = {t: d[d.index <= last] for t, d in yahoo.items()}
    cut = {t: d for t, d in cut.items() if len(d)}
    if cut[XIU].index[-1] != last:
        raise NotNow(f"{XIU} has no bar for {signal} yet")
    cut[XIU] = cut[XIU].reindex(cut[XIU].index.append(pd.DatetimeIndex([next_trading_day(signal)])))
    return cut


def fresh_share(yahoo: Dict[str, pd.DataFrame], symbols: List[str], signal: date) -> float:
    """Share of the symbols Yahoo knows that have a bar on the signal session."""
    known = [s for s in symbols if s in yahoo]
    return sum(pd.Timestamp(signal) in yahoo[s].index for s in known) / max(1, len(symbols))


def held_ids(data: StockData, record: pd.DataFrame) -> Set[int]:
    """Company ids of the last recorded list (ids are renumbered on every load)."""
    if record.empty:
        return set()
    last = record[record["signal_date"] == record["signal_date"].max()]
    out = set()
    for t, d in zip(last["ticker"], pd.to_datetime(last["file_date"])):
        near = data.table[(data.table["ticker"] == t) & (data.table["date"].dt.to_period("M") == d.to_period("M"))]
        if near.empty:
            print(f"warning: held name {t} of {d:%Y-%m-%d} is not in that month's member file; not kept")
        else:
            out.add(int(near["id"].iloc[0]))
    return out


def make_list(data: StockData, signal: date, held: Set[int]) -> pd.DataFrame:
    s = int(data.calendar.get_loc(pd.Timestamp(signal)))
    score, sector = Scores(data).momentum(s), data.sector_of(s)
    file_date = data._file_of[s]
    rows = data.table[data.table["date"] == file_date].drop_duplicates("id").set_index("id")
    out = []
    for k, c in enumerate(en.select(score, held, DEFAULT.top_n, DEFAULT.keep, sector, DEFAULT.max_per_sector), 1):
        symbol = data.symbol.get(c, yahoo_symbol(rows.at[c, "ticker"]))
        close = data.panel.close[symbol].iat[s] if symbol in data.panel.close else rows.at[c, "price"]
        out.append({"signal_date": signal.isoformat(), "file_date": file_date.date().isoformat(), "rank": k,
                    "ticker": rows.at[c, "ticker"], "symbol": symbol, "name": rows.at[c, "name"], "sector": sector[c],
                    "momentum": round(float(score[c]), 4), "close": round(float(close), 4)})
    return pd.DataFrame(out)


def load_record(path: Path = RECORD) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"signal_date": str, "file_date": str, "ticker": str, "symbol": str}, keep_default_na=False
                       ) if path.exists() else pd.DataFrame(columns=COLUMNS)


def append(path: Path, rows: pd.DataFrame, now: datetime, entry: date) -> None:
    """Only ever adds a month; an existing one is never rewritten."""
    old = load_record(path)
    if rows["signal_date"].iloc[0] in set(old["signal_date"]):
        raise NotNow(f"{rows['signal_date'].iloc[0]} is already recorded")
    rows = rows.assign(recorded_at=now.isoformat(timespec="seconds"), entry_date=entry.isoformat())[COLUMNS]
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(path, mode="a", header=not path.exists(), index=False)


def check_time(now: datetime, record: pd.DataFrame) -> date:
    signal, entry = signal_session(now), entry_session(now)
    if signal.isoformat() in set(record["signal_date"]):
        raise NotNow(f"{signal} is already recorded")
    late = sessions_late(signal, entry)
    if late > MAX_LATE:
        raise NotNow(f"the {signal} list is {late} sessions late; the next one can be recorded after the month's last close")
    return signal


def fetch_prices(symbols: List[str], db: Path, end: date) -> Dict[str, pd.DataFrame]:
    conn = prices.connect(str(db))
    try:
        prices.download(conn, sorted(symbols), PRICE_START, end.isoformat())
        return prices.load_all(conn)
    finally:
        conn.close()


def build(signal: date) -> StockData:
    """Downloads what is missing and returns the data as it stood at the signal."""
    holdings.fetch(pd.Timestamp(signal) - pd.Timedelta(days=1))
    files = {d: m for d, m in holdings.load().items() if d <= pd.Timestamp(signal)}
    if max(files).to_period("M") != pd.Timestamp(signal).to_period("M"):
        as_of, m = member_file(signal)
        files[as_of] = m
    newest = max(files)
    if (pd.Timestamp(signal) - newest).days > MAX_FILE_AGE:
        raise NotNow(f"newest member list is from {newest:%Y-%m-%d}")
    table = mb.long_table({newest: files[newest]})
    symbols = mb.candidate_symbols(table)
    yahoo = fetch_prices(symbols, CACHE / f"swing_rank_fwd_{signal:%Y%m%d}.db", signal)
    share = fresh_share(yahoo, symbols, signal)
    if share < MIN_FRESH:
        raise NotNow(f"only {share:.0%} of members have a bar for {signal}")
    return StockData(dict(sorted(files.items())), with_next_session(yahoo, signal))


def record(dry_run: bool = False, path: Path = RECORD, build_fn: Callable[[date], StockData] = build) -> pd.DataFrame:
    old = load_record(path)
    signal = signal_session(market_now()) if dry_run else check_time(market_now(), old)
    data = build_fn(signal)
    rows = make_list(data, signal, held_ids(data, old))
    if len(rows) != DEFAULT.top_n:
        raise NotNow(f"only {len(rows)} names qualified")
    if not dry_run:
        done = market_now()              # the download takes minutes: the entry is the first open after it
        append(path, rows, done, entry_session(done))
    return rows


# ── Results ──────────────────────────────────────────────────────────────────

def results(rec: pd.DataFrame, data: Dict[str, pd.DataFrame], cost: float = COST) -> pd.DataFrame:
    """One row per recorded list that has started: entry open to the next list's
    entry open, the last one to the latest close (complete = False). Costs and
    drifting weights as in the backtest. `missing`: names without a price, counted as 0%."""
    panel = en.build_panel(data, data[XIU].index)
    last = panel.close.ffill()
    months = [(pd.Timestamp(e), list(g.sort_values("rank")["symbol"]))
              for e, g in rec.sort_values("entry_date").groupby("entry_date")]
    months = [(e, names) for e, names in months if e in panel.calendar]
    rows, picks = [], {}
    for k, (e, names) in enumerate(months):
        done = k + 1 < len(months)
        stop = months[k + 1][0] if done else panel.calendar[-1]
        end = panel.execution.loc[stop].fillna(last.loc[stop]) if done else last.loc[stop]
        rows.append(end / panel.execution.loc[e] - 1.0)
        picks[k] = names
    if not rows:
        return pd.DataFrame(columns=["entry", "complete", "net", "xiu", "excess", "missing"])
    r = pd.DataFrame(rows, index=range(len(rows)))
    res = en.run(lambda k, held: en.equal_weights(picks[k]), r, cost)
    out = pd.DataFrame({"entry": [e.date().isoformat() for e, _ in months], "complete": [True] * (len(months) - 1) + [False],
                        "net": res["net"].to_numpy(), "xiu": r[XIU].to_numpy()})
    out["excess"] = out["net"] - out["xiu"]
    out["missing"] = [" ".join(t for t in picks[k] if t not in r.columns or pd.isna(r.at[k, t])) for k in picks]
    return out


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Forward record of S1 (paper)")
    p.add_argument("action", choices=["record", "score"])
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    pd.set_option("display.width", 200)
    try:
        if args.action == "record":
            rows = record(args.dry_run)
            print(rows.to_string(index=False))
            print("(dry run, nothing written)" if args.dry_run else f"Recorded in {RECORD}; commit it today.")
            return 0
        rec = load_record()
        if rec.empty:
            print("Nothing recorded yet.")
            return 0
        today = market_now().date()
        out = results(rec, fetch_prices(sorted(set(rec["symbol"])), CACHE / f"swing_rank_fwd_score_{today:%Y%m%d}.db", today))
        print(out.round(4).to_string(index=False))
        full = out[out["complete"]]
        if len(full):
            print(f"{len(full)} complete months: S1 {(1 + full['net']).prod() - 1:+.2%}, XIU {(1 + full['xiu']).prod() - 1:+.2%}")
        return 0
    except (NotNow, holdings.FetchFailed, prices.YahooBlocked) as exc:
        print(f"STOPPED: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
