"""
news_watchlist/outcomes.py
============================
Outcome scoring for EVERY press_release_tracker catch with a ticker -- not
just the human-confirmed 'watching' items (those keep their own daily
price_history; this is a separate, automatic measurement). The question it
answers: do the parser/analyst labels (materiality, category, analyst
verdict, dilution level) actually predict what the stock does next?

Each release is scored from DAILY BARS after the fact, not from snapshots
taken on the day -- a missed run self-heals on the next one, and the whole
history since the feed started gets scored on the first run.

Entry rule (tradeable, no lookahead): the first session OPEN you could
actually have bought at after the release was published --
  published before 09:30 ET  -> that day's open (or the next session's, if
                                published on a weekend/holiday)
  published 09:30 or later   -> the NEXT session's open
Same next-open entry the volume_breakout_backtest.py family uses.

Horizons are counted in XIU.TO sessions (a thinly traded name can skip
days; its own bar count would stretch the window): horizon h ends on the
h-th XIU session counting the entry session as 1, so h=1 is entry-day
open->close. The stock's exit is its last close on or before that date.
The benchmark return is XIU.TO's own open->close over the same sessions,
so excess = ret - bench_ret.

Only bars strictly before today are used, so every number stored is from a
finished session. A row is finalized ('complete') once its longest horizon
is in; until then it's recomputed every run.

`python -m news_watchlist.outcomes` prints the summary report.
"""
from __future__ import annotations

import math
import statistics
from datetime import date, datetime, time, timezone
from email.utils import parsedate_to_datetime

import pandas as pd

from time_utils import TSX_TZ

HORIZONS = (1, 5, 20, 60)
BENCHMARK = "XIU.TO"
MARKET_OPEN = time(9, 30)

# Group-by keys the report breaks results down by -- each is a column on
# news_watchlist/store.py's release_outcomes table.
REPORT_GROUPS = ("materiality", "category", "verdict", "dilution_level")


def parse_published(pubdate: str | None) -> datetime | None:
    """seen_items.pubdate (RFC 822, see press_release_tracker/feeds.py) ->
    an aware datetime in TSX time, or None if missing/unparseable."""
    if not pubdate:
        return None
    try:
        dt = parsedate_to_datetime(pubdate)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(TSX_TZ)


def symbol_candidates(ticker: str) -> list[str]:
    """Yahoo symbols to try for an as-parsed ticker, in order. Same
    disambiguation idea as news_watchlist_service._resolve_market_price()
    (a bare Canadian symbol often collides with an unrelated US listing,
    so the Canadian suffixes go first), plus .CN -- GlobeNewswire's Canada
    feed carries many CSE names. An already-suffixed ticker is used as-is."""
    ticker = ticker.strip().upper()
    if "." in ticker:
        return [ticker]
    return [ticker + ".TO", ticker + ".V", ticker + ".CN", ticker]


def _finished(df: pd.DataFrame | None, today: date) -> pd.DataFrame | None:
    """Bars strictly before today, sorted -- today's bar may still be
    filling."""
    if df is None or df.empty:
        return None
    df = df.sort_index()
    df = df[df.index.date < today]
    return df if not df.empty else None


def compute_outcome(bars: pd.DataFrame, bench: pd.DataFrame, published: datetime,
                    today: date) -> dict | None:
    """Forward returns for one release. None when the entry session
    hasn't happened yet (or there's no data). Otherwise a dict with
    entry_date/prior_close/entry_open, ret_{h}/bench_ret_{h} per horizon
    (None until that horizon's session has finished), and complete (all
    horizons in)."""
    bars = _finished(bars, today)
    bench = _finished(bench, today)
    if bars is None or bench is None:
        return None

    pub_day = published.date()
    stock_dates = bars.index.date
    if published.time() < MARKET_OPEN:
        eligible = stock_dates >= pub_day
    else:
        eligible = stock_dates > pub_day
    if not eligible.any():
        return None
    entry_pos = int(eligible.argmax())
    entry_date = stock_dates[entry_pos]
    entry_open = float(bars["Open"].iloc[entry_pos])
    if not entry_open > 0:
        return None
    prior_close = float(bars["Close"].iloc[entry_pos - 1]) if entry_pos > 0 else None

    bench_dates = bench.index.date
    bench_on_or_after = bench_dates >= entry_date
    if not bench_on_or_after.any():
        return None
    b0 = int(bench_on_or_after.argmax())
    bench_open = float(bench["Open"].iloc[b0])

    out = {
        "entry_date": entry_date.isoformat(),
        "prior_close": prior_close,
        "entry_open": entry_open,
    }
    for h in HORIZONS:
        bi = b0 + h - 1
        if bi >= len(bench):
            out[f"ret_{h}d"] = None
            out[f"bench_ret_{h}d"] = None
            continue
        exit_date = bench_dates[bi]
        stock_close = bars["Close"][stock_dates <= exit_date].iloc[-1]
        out[f"ret_{h}d"] = float(stock_close) / entry_open - 1.0
        out[f"bench_ret_{h}d"] = float(bench["Close"].iloc[bi]) / bench_open - 1.0
    out["complete"] = out[f"ret_{HORIZONS[-1]}d"] is not None
    return out


# ── report ───────────────────────────────────────────────────────────────

def dedupe_events(rows: list[dict]) -> list[dict]:
    """One event per (symbol, entry_date): GlobeNewswire repeats a release
    in fr/de, and a busy story can produce several releases the same day
    -- counting each would overweight it. Prefers the row that carries an
    analyst verdict (the English original that was analysed)."""
    best: dict[tuple, dict] = {}
    for r in rows:
        key = (r["yahoo_ticker"], r["entry_date"])
        if key not in best or (r.get("verdict") and not best[key].get("verdict")):
            best[key] = r
    return list(best.values())


def _stats(values: list[float]) -> dict:
    n = len(values)
    if n == 0:
        return {"n": 0}
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if n > 1 else 0.0
    t = mean / (sd / math.sqrt(n)) if n > 1 and sd > 0 else None
    return {
        "n": n,
        "mean": mean,
        "median": statistics.median(values),
        "beat": sum(v > 0 for v in values) / n,
        "t": t,
    }


def summarize(rows: list[dict], group_by: str) -> dict:
    """{group value: {horizon: stats of excess return}} over rows whose
    horizon is in. Excess = stock return - XIU.TO return, same window."""
    groups: dict[str, dict[int, list[float]]] = {}
    for r in rows:
        key = r.get(group_by) or "(none)"
        for h in HORIZONS:
            ret, bench = r.get(f"ret_{h}d"), r.get(f"bench_ret_{h}d")
            if ret is None or bench is None:
                continue
            groups.setdefault(key, {}).setdefault(h, []).append(ret - bench)
    return {k: {h: _stats(v.get(h, [])) for h in HORIZONS} for k, v in groups.items()}


def format_report(rows: list[dict]) -> str:
    events = dedupe_events(rows)
    lines = [
        f"Press-release outcomes: {len(events)} events "
        f"({len(rows)} scored releases before de-duplication)",
        "Excess return vs XIU.TO from the first tradeable open. "
        "Each cell: mean / median / %beat XIU (n, t).",
        "UNVALIDATED until n is in the hundreds -- |t| < 2 is noise.",
    ]
    for group_by in ("all",) + REPORT_GROUPS:
        summary = summarize(events, group_by) if group_by != "all" else summarize(
            [{**r, "all": "all events"} for r in events], "all")
        lines.append("")
        lines.append(f"== by {group_by} ==")
        for key in sorted(summary, key=lambda k: -summary[k][HORIZONS[0]].get("n", 0)):
            cells = []
            for h in HORIZONS:
                s = summary[key][h]
                if not s["n"]:
                    cells.append(f"{h}d: -")
                    continue
                t = f"{s['t']:+.1f}" if s["t"] is not None else "n/a"
                cells.append(f"{h}d: {s['mean']:+.1%} / {s['median']:+.1%} / "
                             f"{s['beat']:.0%} (n={s['n']}, t={t})")
            lines.append(f"  {key:<24} " + " | ".join(cells))
    return "\n".join(lines)


if __name__ == "__main__":
    from news_watchlist import store

    conn = store.connect()
    print(format_report(store.list_scored_outcomes(conn)))
