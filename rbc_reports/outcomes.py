"""
rbc_reports/outcomes.py
=======================
Step 5 of the RBC research tracker: how did the stocks do after RBC's
notes? Scores every priced company note (bullish, neutral and bearish
alike -- the comparison groups matter as much as the bullish one) by its
forward return over HORIZONS trading days, against a benchmark over the
same days: SPY for a US listing, XIU.TO for a Canadian one.

No lookahead, and backfillable from bars alone:
- Entry is the first close the user could actually have traded at: a note
  published after the 16:00 ET close enters at the NEXT session's close
  (the digest arrives ~19:00), one published during or before the session
  at that day's close.
- The stock's state at entry (% off its 52-week high, turning up) is
  prices.price_context() over the bars up to the entry day only -- the
  same rule the Candidates table applies today.
A horizon that hasn't elapsed yet stays NULL; every run recomputes the
notes still inside config.RBC_OUTCOME_WINDOW_DAYS, so they fill in.

summarize() groups the scored notes the way the /rbc "Track record" table
shows them, one event per stock per entry day (two notes on the same name
the same day are one trade, not two).
"""
from __future__ import annotations

from datetime import datetime
from statistics import median

from config import RBC_CANDIDATE_DIP_PCT

HORIZONS = (5, 20, 60)
BENCHMARK_US = "SPY"
BENCHMARK_CA = "XIU.TO"


def benchmark_for(yahoo_symbol: str) -> str:
    return BENCHMARK_CA if yahoo_symbol.endswith((".TO", ".V")) else BENCHMARK_US


def entry_index(df, published_at: str):
    """Position in df of the first tradeable close after publication, or
    None if that session hasn't happened yet."""
    published = datetime.fromisoformat(published_at)
    dates = df.index.date
    day = published.date()
    for i, d in enumerate(dates):
        if d > day or (d == day and published.hour < 16):
            return i
    return None


def _close_on_or_before(bench, date):
    closes = bench["Close"][bench.index <= date]
    return float(closes.iloc[-1]) if len(closes) else None


def compute_outcome(df, bench, published_at: str, horizons=HORIZONS) -> dict | None:
    """{"entry_date", "entry_close", "off_high", "turning_up", "ret_<h>",
    "bench_<h>"} -- returns are fractions, NULL (None) where the horizon
    hasn't elapsed or a benchmark close is missing. None if there is no
    entry yet."""
    from rbc_reports.prices import price_context

    df = df[["Close"]].dropna()
    i = entry_index(df, published_at)
    if i is None:
        return None
    entry_date = df.index[i]
    entry_close = float(df["Close"].iloc[i])
    ctx = price_context(df, as_of=entry_date)
    out = {
        "entry_date": entry_date.strftime("%Y-%m-%d"), "entry_close": round(entry_close, 4),
        "off_high": ctx["pct_off_high"] if ctx else None,
        "turning_up": ctx["turning_up"] if ctx else None,
    }
    bench_entry = _close_on_or_before(bench, entry_date) if bench is not None else None
    for h in horizons:
        out[f"ret_{h}"] = out[f"bench_{h}"] = None
        if i + h >= len(df):
            continue
        exit_date = df.index[i + h]
        out[f"ret_{h}"] = round(float(df["Close"].iloc[i + h]) / entry_close - 1, 4)
        bench_exit = _close_on_or_before(bench, exit_date) if bench is not None else None
        if bench_entry and bench_exit:
            out[f"bench_{h}"] = round(bench_exit / bench_entry - 1, 4)
    return out


def score_notes(notes, bars: dict) -> dict:
    """notes: [(report_key, yahoo_symbol, published_at)]; bars: {symbol:
    DataFrame} incl. the benchmarks. Returns {report_key: outcome}."""
    out = {}
    for key, ys, published in notes:
        df = bars.get(ys)
        if df is None or df.empty:
            continue
        o = compute_outcome(df, bars.get(benchmark_for(ys)), published)
        if o:
            out[key] = {**o, "yahoo_symbol": ys, "benchmark": benchmark_for(ys)}
    return out


# Track-record groups, in display order: (label, predicate over an event).
# "bullish" = the Candidates table's own definition (tone bullish, or an
# upgrade / PT raise / initiation / beat).
def _beaten(e):
    return e["off_high"] is not None and e["off_high"] <= -RBC_CANDIDATE_DIP_PCT


GROUPS = (
    ("Bullish, beaten down, turning up (the Candidates rule)",
     lambda e: e["bullish"] and _beaten(e) and e["turning_up"] == 1),
    ("Bullish, beaten down", lambda e: e["bullish"] and _beaten(e)),
    ("Bullish, not beaten down", lambda e: e["bullish"] and not _beaten(e) and e["off_high"] is not None),
    ("All bullish", lambda e: e["bullish"]),
    ("Neutral", lambda e: not e["bullish"] and e["tone"] == "neutral"),
    ("Bearish", lambda e: not e["bullish"] and e["tone"] == "bearish"),
)


def summarize(events: list, horizons=HORIZONS) -> list:
    """events: dicts with yahoo_symbol, entry_date, published_at, bullish,
    tone, off_high, turning_up, ret_<h>, bench_<h>. One event per stock per
    entry day (the earliest note). Returns per group: {"label", "events",
    "by_horizon": {h: {"n", "avg_excess", "median_excess", "hit_rate",
    "avg_return"} or None}} -- `excess` is return minus benchmark over the
    same days, `hit_rate` the share that beat the benchmark."""
    seen, unique = set(), []
    for e in sorted(events, key=lambda e: e["published_at"]):
        k = (e["yahoo_symbol"], e["entry_date"])
        if k not in seen:
            seen.add(k)
            unique.append(e)
    rows = []
    for label, pred in GROUPS:
        members = [e for e in unique if pred(e)]
        by_h = {}
        for h in horizons:
            done = [e for e in members if e[f"ret_{h}"] is not None and e[f"bench_{h}"] is not None]
            if not done:
                by_h[h] = None
                continue
            excess = [e[f"ret_{h}"] - e[f"bench_{h}"] for e in done]
            by_h[h] = {
                "n": len(done),
                "avg_excess": sum(excess) / len(excess),
                "median_excess": median(excess),
                "hit_rate": sum(x > 0 for x in excess) / len(excess),
                "avg_return": sum(e[f"ret_{h}"] for e in done) / len(done),
            }
        rows.append({"label": label, "events": len(members), "by_horizon": by_h})
    return rows
