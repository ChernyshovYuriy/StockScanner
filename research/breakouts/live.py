"""
Daily scan for confirmed breakouts, emailed the evening they happen (PLAN.md). Facts
about past bars, not advice: no capital, no positions, no orders. Own database,
data/breakouts.db.

    python -m research.breakouts.live auto             # the Pi's timer: scans a finished session once, emails what it finds
    python -m research.breakouts.live auto --dry-run   # scan, print, write and send nothing
    python -m research.breakouts.live list             # breakouts emailed so far

Same detector, prices (split- but not dividend-adjusted) and eligibility as the
backtest. Names: research/breakouts/tickers.txt (the backtest's) to start with;
each run also looks at ROTATE other names of the full list, so a name that becomes
liquid joins within about a month and one that dries up leaves.
"""
from __future__ import annotations

import argparse
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from config import MOMENTUM_RAW_TICKERS_URL
from market_data import LiveDataProvider
from research.breakouts.backtest import MIN_DOLLAR, MIN_HISTORY, MIN_PRICE, XIU
from research.breakouts.detector import DEFAULT, scan
from research.breakouts.features import slope200, stock_features
from research.breakouts.prices import after_seam
from research.vcp.prices import CHUNK, YAHOO_PER_SECOND, Pacer, YahooBlocked, candidate_tickers
from time_utils import last_trading_day_on_or_before, market_now, previous_trading_day

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "breakouts.db"
SEED = Path(__file__).with_name("tickers.txt")
SUBJECT = "Breakouts"
PERIOD = "3y"
CLOSE_HOUR = 16.5                # bars are taken as final half an hour after the close
KEEP_DOLLAR = 500_000.0          # below this 20-day dollar volume a name is only looked at in rotation
ROTATE = 100
CATCH_UP = 5                     # sessions an earlier missed scan is made up for
AGAIN = 60                       # an earlier breakout of the name within this many sessions is mentioned
GONE_DAYS = 14                   # a name without data this long is dropped from the liquid ones
MIN_FRESH = 0.90                 # share of liquid names whose bars must reach the session
NEED = DEFAULT.max_bars + AGAIN + CATCH_UP + 2 * DEFAULT.pivot + 2          # bars the detector can look at

RECORD = """What happened after such breakouts, 2012-2026 (3,024 on today's Canadian names; before costs):
  200-day average rising : 57% up after a month; within two weeks 38% touched +5%, 32% touched -5%
  200-day average falling: 46% up after a month; within two weeks 38% touched +5%, 42% touched -5%
On its own this did not beat buying similar stocks in the backtest. Not advice."""


class NotNow(RuntimeError):
    """The session can't be scanned at this moment."""


# ── State ────────────────────────────────────────────────────────────────────

def connect(path: Path = DB) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS names (ticker TEXT PRIMARY KEY, dollar REAL, seen TEXT, tried TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS breakouts (ticker TEXT NOT NULL, date TEXT NOT NULL, rising INTEGER, slope200 REAL, close REAL,
            line REAL, confirm REAL, tests INTEGER, bars INTEGER, depth REAL, lfd_low REAL, target REAL, vol_x REAL, day_ret REAL,
            atr REAL, previous TEXT, emailed INTEGER NOT NULL DEFAULT 0, found_at TEXT NOT NULL, PRIMARY KEY (ticker, date));
        CREATE TABLE IF NOT EXISTS runs (session TEXT PRIMARY KEY, finished_at TEXT NOT NULL, scanned INTEGER NOT NULL);
    """)
    return conn


def finished(conn: sqlite3.Connection) -> List[str]:
    return [r[0] for r in conn.execute("SELECT session FROM runs ORDER BY session")]


# ── When and what ────────────────────────────────────────────────────────────

def session_at(now: datetime) -> date:
    """The last session that is over at `now`."""
    day = last_trading_day_on_or_before(now.date())
    if day == now.date() and now.hour + now.minute / 60 < CLOSE_HOUR:
        day = previous_trading_day(day)
    return day


def choose(conn: sqlite3.Connection, seed: List[str], full: List[str]) -> List[str]:
    """Liquid names first, then the ROTATE others looked at longest ago (never first)."""
    known = {t: (d, tried) for t, d, tried in conn.execute("SELECT ticker, dollar, tried FROM names")}
    liquid = {t for t, (d, _) in known.items() if d is not None and d >= KEEP_DOLLAR}
    active = liquid | {t for t in seed if t not in known or known[t][0] is None}
    rest = sorted((set(full) | set(known)) - active, key=lambda t: (known[t][1] if t in known else "", t))
    return [t for t in sorted(active) + rest[:ROTATE] if t != XIU]


def liquid_names(conn: sqlite3.Connection) -> Set[str]:
    return {r[0] for r in conn.execute("SELECT ticker FROM names WHERE dollar >= ?", [MIN_DOLLAR])}


# ── Scan ─────────────────────────────────────────────────────────────────────

def clean(d: pd.DataFrame, session: date) -> pd.DataFrame:
    d = d.dropna()
    d = d[(d["Close"] > 0) & (d["Volume"] >= 0)]
    return after_seam(d[~d.index.duplicated(keep="last")].sort_index().loc[:pd.Timestamp(session)])


def dollar_volume(d: pd.DataFrame, cal: pd.DatetimeIndex, day: pd.Timestamp) -> float:
    """20-session mean on XIU's sessions; a session without a bar counts as zero."""
    last = cal[cal <= day][-20:]
    return float((d["Close"] * d["Volume"]).reindex(last).fillna(0.0).mean())


def find(d: pd.DataFrame, cal: pd.DatetimeIndex, days: List[pd.Timestamp]) -> List[Dict]:
    """Eligible breakouts on `days` from one name's bars up to the session."""
    if len(d) < MIN_HISTORY or d["Close"].iloc[-(len(days) + 1):].max() < MIN_PRICE:
        return []
    tail = d.iloc[-NEED:]
    first = cal[max(0, len(cal) - AGAIN - len(days))]
    hits = [(tail.index[b.bar], b) for b in scan(tail) if tail.index[b.bar] >= first]
    out = []
    for day, b in hits:
        k = d.index.get_loc(day)
        if day not in days or b.close < MIN_PRICE or k + 1 < MIN_HISTORY or dollar_volume(d, cal, day) < MIN_DOLLAR:
            continue
        f = stock_features(d, k, b.level, k - b.bars, b.low, b.target, b.lfd_low)
        slope = slope200(d["Close"].to_numpy(float), k)
        before = [x for x, _ in hits if x < day and 0 < cal.get_loc(day) - cal.searchsorted(x) <= AGAIN]
        out.append({"date": day.date().isoformat(), "rising": None if np.isnan(slope) else int(slope > 0), "slope200": float(slope),
                    "close": b.close, "line": float(b.level), "confirm": float(b.confirm), "tests": b.tests, "bars": b.bars,
                    "depth": float(f["depth"]), "lfd_low": b.lfd_low, "target": float(b.target), "vol_x": float(f["vol_x"]),
                    "day_ret": float(f["day_ret"]), "atr": float(f["atr"]), "previous": before[-1].date().isoformat() if before else None})
    return out


Fetch = Callable[[List[str]], Dict[str, pd.DataFrame]]


def yahoo(tickers: List[str]) -> Dict[str, pd.DataFrame]:
    return LiveDataProvider().download_batch_with_reasons(tickers, period=PERIOD, interval="1d", auto_adjust=False)[0]


def scan_all(names: List[str], session: date, done: List[str], known_liquid: Set[str], fetch: Fetch, pacer: Pacer, progress=print
             ) -> Tuple[List[Dict], Dict[str, Optional[float]]]:
    """(breakouts of the sessions not scanned yet, {name: 20-day dollar volume, None without data}). Stops if Yahoo refuses."""
    def get(chunk):
        pacer.wait(len(chunk))
        return fetch(chunk)

    x = get([XIU]).get(XIU)
    if x is None or x.empty:
        raise YahooBlocked("no benchmark bars at the start")
    cal = clean(x, session).index
    last = pd.Timestamp(session)
    if cal[-1] != last:
        raise NotNow(f"{XIU} has no bar for {session} yet")
    since = pd.Timestamp(max([s for s in done if s < session.isoformat()], default=cal[-2].date().isoformat()))
    days = list(cal[cal > since][-CATCH_UP:])
    found, dollars, liquid, fresh = [], {}, 0, 0
    for i in range(0, len(names), CHUNK):
        chunk = names[i:i + CHUNK]
        data = get(chunk)
        if not data and not get([XIU]):
            raise YahooBlocked(f"no bars at name {i} of {len(names)}")
        for t in chunk:
            d = clean(data[t], session) if t in data else None
            if d is None or d.empty:
                dollars[t] = None
                liquid += t in known_liquid
                continue
            dollars[t] = dollar_volume(d, cal, last)
            if dollars[t] >= MIN_DOLLAR:
                liquid, fresh = liquid + 1, fresh + (d.index[-1] == last)
            found += [{"ticker": t, **r} for r in find(d, cal, days)]
        progress(f"{min(i + CHUNK, len(names))}/{len(names)}")
    if not liquid or fresh / liquid < MIN_FRESH:
        raise NotNow(f"only {fresh} of {liquid} liquid names have a bar for {session}")
    return found, dollars


def save(conn: sqlite3.Connection, session: date, found: List[Dict], dollars: Dict[str, Optional[float]], now: datetime) -> None:
    stamp = now.isoformat(timespec="seconds")
    with conn:
        day, gone = session.isoformat(), (session - timedelta(days=GONE_DAYS)).isoformat()
        for t, v in dollars.items():
            if v is not None:
                conn.execute("INSERT OR REPLACE INTO names VALUES (?, ?, ?, ?)", (t, v, day, day))
            else:                        # a passing failure must not drop a name; one gone for weeks is dropped
                conn.execute("INSERT INTO names VALUES (?, NULL, NULL, ?) ON CONFLICT (ticker) DO UPDATE SET tried = excluded.tried, "
                             "dollar = CASE WHEN seen >= ? THEN dollar END", (t, day, gone))
        for r in found:
            cols = ", ".join(r)
            conn.execute(f"INSERT OR IGNORE INTO breakouts ({cols}, found_at) VALUES ({', '.join('?' * len(r))}, ?)", [*r.values(), stamp])
        conn.execute("INSERT OR REPLACE INTO runs VALUES (?, ?, ?)", (session.isoformat(), stamp, len(dollars)))


def unsent(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM breakouts WHERE emailed = 0 ORDER BY date DESC, rising DESC, slope200 DESC", conn)


# ── Email ────────────────────────────────────────────────────────────────────

def describe(r) -> str:
    state = {1: "RISING", 0: "FALLING"}.get(None if pd.isna(r.rising) else int(r.rising), "not known (short history)")
    change = "" if pd.isna(r.slope200) else f" ({r.slope200:+.1%} in a month)"
    lines = [f"{r.ticker}   200-day average {state}{change}",
             f"  closed {r.close:,.2f}, {r.close / r.line - 1:+.1%} over the line {r.line:,.2f} (confirmed at {r.confirm:,.2f})",
             f"  base {r.bars / 21:.1f} months, {r.tests} tests of the line, {r.depth:.0%} deep",
             f"  day {r.day_ret:+.1%}, volume {r.vol_x:.1f}x its 50-day average, daily range {r.atr:.1%}",
             f"  Last Full Day low {r.lfd_low:,.2f} ({r.lfd_low / r.close - 1:+.1%}), measured target {r.target:,.2f} ({r.target / r.close - 1:+.1%})"]
    if r.previous:
        lines.append(f"  also broke out on {r.previous}")
    return "\n".join(lines)


def message(rows: pd.DataFrame, session: date) -> Tuple[str, str]:
    up = rows[rows["rising"] == 1]
    tail = len(rows) - len(up)
    names = ", ".join(up["ticker"]) + (f" (+{tail} not rising)" if len(up) and tail else "") or f"{tail} not rising"
    parts = []
    for day, g in rows.groupby("date", sort=False):
        late = "" if day == session.isoformat() else "  (found late: an earlier scan was missed)"
        parts.append(f"=== {day}{late} ===\n\n" + "\n\n".join(describe(r) for r in g.itertuples()))
    return f"{SUBJECT} {session}: {names}", "\n\n".join(parts) + "\n\n" + RECORD + "\n"


# ── On the timer ─────────────────────────────────────────────────────────────

def full_list() -> List[str]:
    from research.triple_screen.batch import load_tickers
    return candidate_tickers(load_tickers(MOMENTUM_RAW_TICKERS_URL))


def auto(dry_run: bool = False, path: Path = DB, fetch: Fetch = yahoo, full: Callable[[], List[str]] = full_list,
         send: Optional[Callable[[str, str], bool]] = None, pacer: Optional[Pacer] = None, seed: Optional[List[str]] = None) -> int:
    """Scans the last finished session if that has not been done, then emails every
    breakout not emailed yet. A failure is emailed and left for the next scheduled try."""
    if send is None:
        from send_report import send_text_email as send
    session = session_at(market_now())
    conn = connect(path)
    try:
        done = finished(conn)
        if session.isoformat() in done and not dry_run:
            print(f"{session} is already scanned")
        else:
            try:
                try:
                    others = full()
                except Exception as exc:
                    print(f"full list not available ({type(exc).__name__}); known names only")
                    others = []
                names = choose(conn, seed if seed is not None else SEED.read_text().split(), others)
                found, dollars = scan_all(names, session, done, liquid_names(conn), fetch, pacer or Pacer(YAHOO_PER_SECOND))
            except Exception as exc:
                print(f"STOPPED: {type(exc).__name__}: {exc}")
                if not dry_run:
                    send(f"{SUBJECT} {session}: scan NOT done", f"{type(exc).__name__}: {exc}\n\nThe next scheduled run tries again.")
                return 2
            print(f"{session}: {len(names)} names scanned, {len(found)} breakouts")
            if dry_run:
                if found:
                    print("\n".join(message(pd.DataFrame(found).sort_values(["date", "rising"], ascending=False), session)))
                return 0
            save(conn, session, found, dollars, market_now())
        rows = unsent(conn)
        if rows.empty:
            return 0
        subject, body = message(rows, session)
        print(subject, body, sep="\n")
        if not send(subject, body):
            print("email not sent; kept for the next run")
            return 2
        with conn:
            conn.executemany("UPDATE breakouts SET emailed = 1 WHERE ticker = ? AND date = ?", rows[["ticker", "date"]].itertuples(index=False, name=None))
        return 0
    finally:
        conn.close()


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Daily scan for confirmed breakouts")
    p.add_argument("action", choices=["auto", "list"])
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    if a.action == "auto":
        return auto(a.dry_run)
    conn = connect()
    try:
        rows = pd.read_sql("SELECT date, ticker, rising, close, line, lfd_low, target, previous FROM breakouts ORDER BY date DESC, ticker", conn)
    finally:
        conn.close()
    print(rows.round(3).to_string(index=False) if len(rows) else "none yet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
