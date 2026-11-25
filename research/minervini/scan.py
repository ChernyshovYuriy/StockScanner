"""
Every evening: Canadian stocks that meet Minervini's setup within 3% of the pivot, emailed with entry, loss point
and take profit (LIVE_PLAN.md). No orders, no positions, no capital. Own database, data/minervini.db, shown on the
dashboard's /minervini tab for one week. Reads the bars the breakout scan saved; asks Yahoo for nothing.

    python -m research.minervini.scan auto             # the Pi's timer
    python -m research.minervini.scan auto --dry-run   # print, write and send nothing
"""
from __future__ import annotations

import argparse
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import MINERVINI_DB_PATH
from research.breakouts.backtest import XIU
from research.breakouts.live import BARS, session_at
from research.minervini import method as mm
from research.minervini.backtest import Study
from time_utils import market_now, next_trading_day, previous_trading_day

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / MINERVINI_DB_PATH
SUBJECT = "Minervini candidates"
KEEP_SESSIONS = 30                 # a pivot emailed within this many sessions is not emailed again

RECORD = """Past record of exactly this (2012-2026, after 0.5% costs, only companies that still exist): half of the
emailed setups were bought within the week; 30% of those won, average +19.4% / -6.1%, +1.5% a trade, typically
about two months for a winner. As a portfolio 8.1% a year, about the same as XIU without its dividends: it
did not prove better than the index. Not advice. Check news and earnings dates yourself."""


class NotNow(RuntimeError):
    """The session's bars are not there."""


def connect(path: Path = DB) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS candidates (ticker TEXT NOT NULL, signal TEXT NOT NULL, pivot REAL NOT NULL, low REAL NOT NULL,
            close REAL NOT NULL, rank REAL, rest INTEGER, depth REAL, dry REAL, state TEXT NOT NULL, state_date TEXT, entry REAL,
            stop REAL NOT NULL, target REAL NOT NULL, last REAL, last_date TEXT, expires TEXT, emailed INTEGER NOT NULL DEFAULT 0,
            found_at TEXT NOT NULL, PRIMARY KEY (ticker, signal));
        CREATE TABLE IF NOT EXISTS runs (session TEXT PRIMARY KEY, finished_at TEXT NOT NULL);
    """)
    return conn


def memory_copy(path: Path) -> sqlite3.Connection:
    """The database as it is, in memory: a dry run leaves nothing behind."""
    conn = connect(Path(":memory:"))
    if path.exists():
        disk = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        disk.backup(conn)
        disk.close()
    return conn


def load_bars(session: date, path: Path = BARS) -> Dict[str, pd.DataFrame]:
    if not path.exists():
        raise NotNow("no saved bars: has the breakout scan run?")
    got = pd.read_pickle(path)
    if got["session"] != session.isoformat():
        raise NotNow(f"the saved bars are of {got['session']}, not {session}")
    return got["bars"]


# ── Candidates ───────────────────────────────────────────────────────────────

def levels(pivot: float, low: float) -> Tuple[float, float]:
    """(loss point, take profit) of a buy at the pivot."""
    return mm.initial_stop(pivot, low), 1.20 * pivot


def fresh(st: Study, session: date) -> List[Dict]:
    """Every setup of tonight within 3% of its pivot."""
    day, out = pd.Timestamp(session), []
    for t, setups in st.setups.items():
        i, a = st.idx[t], st.a[t]
        k = len(i) - 1
        if i[k] != day or k not in setups:
            continue
        pivot, low = setups[k]
        if not mm.near(pivot, a["Close"][k]):
            continue
        stop, target = levels(pivot, low)
        h = a["High"][k - mm.DEFAULT.pivot_bars + 1:k + 1]
        out.append({"ticker": t, "signal": session.isoformat(), "pivot": pivot, "low": low, "close": float(a["Close"][k]),
                    "rank": float(st.rank.at[day, t]), "rest": int(len(h) - 1 - np.argmax(h)), "depth": 1 - low / pivot,
                    "dry": float(a["vol_dry"][k] / a["vol50"][k]), "state": "waiting", "stop": stop, "target": target})
    return sorted(out, key=lambda r: -r["rank"])


def follow(a: Dict[str, np.ndarray], t: int, pivot: float, low: float) -> Dict:
    """What has happened to an order placed after bar t, on the bars so far."""
    stop, target = levels(pivot, low)
    state, k, price = mm.order(a, np.ones(len(a["Open"]), bool), t, pivot, low, cancel=False)
    out = {"state": state, "bar": k if state != "waiting" else None, "entry": price, "stop": stop, "target": target}
    if state == "bought":
        out["stop"], out["target"] = mm.initial_stop(price, low), 1.20 * price
        x = mm.leave(a, k, price, out["stop"], "t20")
        if x["why"] in ("stop", "target"):
            out["state"], out["bar"] = {"stop": "stopped", "target": "target"}[x["why"]], x["end"]
    return out


def sessions_between(cal: pd.DatetimeIndex, a: str, b: date) -> int:
    return int(cal.searchsorted(pd.Timestamp(b)) - cal.searchsorted(pd.Timestamp(a)))


def update(conn: sqlite3.Connection, st: Study, session: date) -> None:
    """States of the setups still in their week."""
    cal = st.m.cal
    rows = conn.execute("SELECT ticker, signal, pivot, low, close FROM candidates").fetchall()
    for t, signal, pivot, low, close in rows:
        if not 0 < sessions_between(cal, signal, session) <= mm.WEEK or t not in st.a or pd.Timestamp(signal) not in st.idx[t]:
            continue
        i, a = st.idx[t], st.a[t]
        k = i.get_loc(pd.Timestamp(signal))
        f = a["Close"][k] / close                              # a split since then
        x = follow(a, k, pivot * f, low * f)
        conn.execute("UPDATE candidates SET pivot = ?, low = ?, close = ?, state = ?, state_date = ?, entry = ?, stop = ?, target = ?, last = ?, "
                     "last_date = ? WHERE ticker = ? AND signal = ?",
                     (pivot * f, low * f, close * f, x["state"], None if x["bar"] is None else i[x["bar"]].date().isoformat(), x["entry"],
                      x["stop"], x["target"], float(a["Close"][-1]), i[-1].date().isoformat(), t, signal))


def is_new(conn: sqlite3.Connection, cal: pd.DatetimeIndex, c: Dict) -> bool:
    """Not while the name's last setup is in its week and open, nor the same pivot again."""
    session = date.fromisoformat(c["signal"])
    for signal, pivot, state in conn.execute("SELECT signal, pivot, state FROM candidates WHERE ticker = ?", [c["ticker"]]):
        age = sessions_between(cal, signal, session)
        if age < KEEP_SESSIONS and abs(pivot / c["pivot"] - 1) < 1e-6:
            return False
        if age < mm.WEEK and state in ("waiting", "bought", "stopped", "target"):
            return False
    return True


def shift(day: date, n: int) -> date:
    """n sessions later (or earlier, if negative)."""
    for _ in range(abs(n)):
        day = next_trading_day(day) if n > 0 else previous_trading_day(day)
    return day


# ── Email ────────────────────────────────────────────────────────────────────

def describe(r) -> str:
    return "\n".join([
        f"{r.ticker}   close {r.close:,.2f}   strength rank {r.rank:.0%}   rest {r.rest} days, {r.depth:.1%} deep, volume {r.dry:.1f}x",
        f"  ENTRY        buy above {r.pivot:,.2f}   ({r.pivot / r.close - 1:+.1%} from the close; not above {mm.DEFAULT.chase * r.pivot:,.2f}; good for {mm.WEEK} sessions)",
        f"  LOSS POINT   {r.stop:,.2f}   ({r.stop / r.pivot - 1:+.1%} from the entry)",
        f"  TAKE PROFIT  {r.target:,.2f}   (+20%)"])


STATES = {"waiting": "waiting for the entry", "bought": "entry hit", "stopped": "entry hit, then the loss point", "target": "entry hit, then +20%",
          "expired": "entry not reached in a week", "ran": "opened too far above the entry"}


def message(new: pd.DataFrame, others: pd.DataFrame, session: date) -> Tuple[str, str]:
    body = "\n\n".join(describe(r) for r in new.itertuples())
    body += "\n\nBuying higher than the entry: loss point " + f"{mm.DEFAULT.max_risk:.0%} below your price at most (or just under the rest's low), take profit +20%."
    if len(others):
        body += "\n\nEarlier this week (also on the dashboard's Minervini tab):\n" + "\n".join(
            f"  {r.ticker} of {r.signal}, entry {r.pivot:,.2f}: {STATES[r.state]}" + (f" on {r.state_date}" if pd.notna(r.state_date) else "")
            for r in others.itertuples())
    return f"{SUBJECT} {session}: {', '.join(new['ticker'])}", body + "\n\n" + RECORD + "\n"


# ── On the timer ─────────────────────────────────────────────────────────────

def auto(dry_run: bool = False, path: Path = DB, bars_path: Path = BARS, send: Optional[Callable[[str, str], bool]] = None) -> int:
    """Scans the last finished session once; emails the new candidates, nothing if there are none."""
    if send is None:
        from send_report import send_text_email as send
    now = market_now()
    session = session_at(now)
    conn = memory_copy(path) if dry_run else connect(path)
    try:
        if conn.execute("SELECT 1 FROM runs WHERE session = ?", [session.isoformat()]).fetchone() and not dry_run:
            print(f"{session} is already scanned")
        else:
            try:
                bars = load_bars(session, bars_path)
                st = Study(bars)
                cal = st.m.cal
                if cal[-1] != pd.Timestamp(session):
                    raise NotNow(f"{XIU}'s last saved bar is of {cal[-1].date()}")
                with conn:
                    update(conn, st, session)
                    found = [c for c in fresh(st, session) if is_new(conn, cal, c)]
                    ends = shift(session, mm.WEEK).isoformat()
                    for c in found:
                        c = {**c, "expires": ends, "last": c["close"], "last_date": c["signal"], "found_at": now.isoformat(timespec="seconds")}
                        conn.execute(f"INSERT INTO candidates ({', '.join(c)}) VALUES ({', '.join('?' * len(c))})", list(c.values()))
                    conn.execute("INSERT OR REPLACE INTO runs VALUES (?, ?)", (session.isoformat(), now.isoformat(timespec="seconds")))
            except Exception as exc:
                print(f"STOPPED: {type(exc).__name__}: {exc}")
                if not dry_run and now.hour >= 21:                     # the last try of the evening
                    send(f"{SUBJECT} {session}: scan NOT done", f"{type(exc).__name__}: {exc}\n")
                return 2
            print(f"{session}: {len(st.a)} names, {len(found)} new candidates")
        new = pd.read_sql("SELECT * FROM candidates WHERE emailed = 0 ORDER BY signal DESC, rank DESC", conn)
        new = new[new["state"] == "waiting"]
        if new.empty:
            return 0
        rows = pd.read_sql("SELECT * FROM candidates WHERE emailed = 1 ORDER BY signal DESC, rank DESC", conn)
        others = rows[rows["signal"] >= shift(session, -mm.WEEK).isoformat()]
        subject, body = message(new, others, session)
        print(subject, body, sep="\n")
        if dry_run:
            return 0
        if not send(subject, body):
            print("email not sent; kept for the next run")
            return 2
        with conn:
            conn.executemany("UPDATE candidates SET emailed = 1 WHERE ticker = ? AND signal = ?", new[["ticker", "signal"]].itertuples(index=False, name=None))
        return 0
    finally:
        conn.close()


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Evening scan for Minervini setups")
    p.add_argument("action", choices=["auto"])
    p.add_argument("--dry-run", action="store_true")
    return auto(p.parse_args(argv).dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
