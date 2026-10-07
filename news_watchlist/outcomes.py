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
MARKET_CLOSE = time(16, 0)

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
    # Last close of a session that had FINISHED when the release came out
    # -- the market price a financing's issue price is compared against.
    # Differs from prior_close for a release published during the session
    # (prior_close is then the release day's own, post-news close).
    if published.time() >= MARKET_CLOSE:
        finished_before = stock_dates <= pub_day
    else:
        finished_before = stock_dates < pub_day
    pre_close = float(bars["Close"][finished_before].iloc[-1]) if finished_before.any() else None

    bench_dates = bench.index.date
    bench_on_or_after = bench_dates >= entry_date
    if not bench_on_or_after.any():
        return None
    b0 = int(bench_on_or_after.argmax())
    bench_open = float(bench["Open"].iloc[b0])

    out = {
        "entry_date": entry_date.isoformat(),
        "prior_close": prior_close,
        "pre_close": pre_close,
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
    analyst verdict / financing terms (the English original that was
    analysed)."""
    def info(r):
        return bool(r.get("verdict")) + bool(r.get("financing"))

    best: dict[tuple, dict] = {}
    for r in rows:
        key = (r["yahoo_ticker"], r["entry_date"])
        if key not in best or info(r) > info(best[key]):
            best[key] = r
    return list(best.values())


# ── financing terms (press_release_tracker/financing.py) ─────────────────

_CANADIAN_SUFFIXES = (".TO", ".V", ".CN", ".NE")


def financing_metrics(terms: dict, pre_close: float | None, yahoo_ticker: str | None) -> dict:
    """The two derived numbers, computed here rather than by the LLM:
    discount_pct = issue price vs the last close before publication
    (negative = priced below market), only when the deal is priced in CAD
    and the symbol trades in Canada (else the currencies may differ);
    dilution_pct = securities offered (or gross / issue price) as a % of
    shares outstanding."""
    price = terms.get("issue_price")
    discount = None
    if (price and pre_close and (terms.get("currency") in (None, "CAD"))
            and yahoo_ticker and yahoo_ticker.endswith(_CANADIAN_SUFFIXES)):
        discount = price / pre_close - 1.0
    offered = terms.get("securities_offered")
    if not offered and price and terms.get("gross_proceeds"):
        offered = terms["gross_proceeds"] / price
    shares_out = terms.get("shares_outstanding")
    dilution = offered / shares_out if offered and shares_out else None
    return {"discount_pct": discount, "dilution_pct": dilution}


def _discount_bucket(d):
    if d is None:
        return None
    if d >= 0:
        return "a premium (>= 0%)"
    if d > -0.10:
        return "b 0 to -10%"
    if d > -0.25:
        return "c -10 to -25%"
    return "d below -25%"


def _dilution_bucket(d):
    if d is None:
        return None
    if d < 0.05:
        return "a < 5%"
    if d < 0.15:
        return "b 5-15%"
    if d < 0.30:
        return "c 15-30%"
    return "d >= 30%"


def _yes_no(v):
    return None if v is None else ("yes" if v else "no")


def attach_financing(rows: list[dict], terms_by_guid: dict) -> list[dict]:
    """Rows with their financing terms (if any) under "financing" and the
    report's financing group keys filled in. Only releases the extraction
    confirmed are a financing (is_financing) get the group keys."""
    out = []
    for r in rows:
        r = dict(r)
        t = terms_by_guid.get(r["guid"])
        if t and t.get("is_financing"):
            m = financing_metrics(t, r.get("pre_close"), r.get("yahoo_ticker"))
            r["financing"] = {**t, **m}
            r["fin_offering_type"] = t.get("offering_type")
            r["fin_deal_stage"] = t.get("deal_stage")
            r["fin_flow_through"] = _yes_no(t.get("flow_through"))
            wc = t.get("warrant_coverage")
            r["fin_warrants"] = None if wc is None else ("yes" if wc > 0 else "no")
            r["fin_brokered"] = _yes_no(t.get("brokered"))
            r["fin_insiders"] = _yes_no(t.get("insider_participation"))
            r["fin_discount"] = _discount_bucket(m["discount_pct"])
            r["fin_dilution"] = _dilution_bucket(m["dilution_pct"])
        out.append(r)
    return out


FINANCING_GROUPS = ("fin_offering_type", "fin_deal_stage", "fin_flow_through", "fin_warrants",
                    "fin_brokered", "fin_insiders", "fin_discount", "fin_dilution")


def _stats(values: list[float], pool_beat: float | None = None) -> dict:
    """median, 1%-trimmed mean and beat rate lead -- a handful of penny-stock
    or split-artifact moves (+6,884% in the archive) carry the plain mean
    and its t-statistic, so neither is shown. z tests the beat rate against
    pool_beat (the same horizon's beat rate over everything being compared),
    not against 50%: most events lag XIU, so 50% would flag every group."""
    n = len(values)
    if n == 0:
        return {"n": 0}
    ordered = sorted(values)
    k = n // 100
    beat = sum(v > 0 for v in values) / n
    z = None
    if pool_beat is not None and 0 < pool_beat < 1:
        z = (beat - pool_beat) / math.sqrt(pool_beat * (1 - pool_beat) / n)
    return {
        "n": n,
        "mean": statistics.fmean(values),
        "trimmed": statistics.fmean(ordered[k:n - k]),
        "median": statistics.median(values),
        "beat": beat,
        "z": z,
    }


def _excess(r: dict, h: int) -> float | None:
    ret, bench = r.get(f"ret_{h}d"), r.get(f"bench_ret_{h}d")
    return None if ret is None or bench is None else ret - bench


def pool_beat_rates(rows: list[dict]) -> dict:
    """{horizon: share of rows beating XIU.TO}, the baseline z is measured against."""
    out = {}
    for h in HORIZONS:
        xs = [x for x in (_excess(r, h) for r in rows) if x is not None]
        out[h] = sum(x > 0 for x in xs) / len(xs) if xs else None
    return out


def summarize(rows: list[dict], group_by: str, pool_beat: dict | None = None) -> dict:
    """{group value: {horizon: stats of excess return}} over rows whose
    horizon is in. Excess = stock return - XIU.TO return, same window."""
    groups: dict[str, dict[int, list[float]]] = {}
    for r in rows:
        key = r.get(group_by) or "(none)"
        for h in HORIZONS:
            x = _excess(r, h)
            if x is None:
                continue
            groups.setdefault(key, {}).setdefault(h, []).append(x)
    pool_beat = pool_beat or {}
    return {k: {h: _stats(v.get(h, []), pool_beat.get(h)) for h in HORIZONS}
            for k, v in groups.items()}


def _format_groups(events: list[dict], groups, lines: list[str], sort_by_key=False) -> None:
    pool = pool_beat_rates(events)
    for group_by in groups:
        summary = summarize(events, group_by, pool)
        lines.append("")
        lines.append(f"== by {group_by} ==")
        keys = sorted(summary) if sort_by_key else sorted(
            summary, key=lambda k: -summary[k][HORIZONS[0]].get("n", 0))
        for key in keys:
            cells = []
            for h in HORIZONS:
                s = summary[key][h]
                if not s["n"]:
                    cells.append(f"{h}d: -")
                    continue
                z = f"{s['z']:+.1f}" if s["z"] is not None else "n/a"
                cells.append(f"{h}d: {s['median']:+.1%} / {s['trimmed']:+.1%} / "
                             f"{s['beat']:.0%} (n={s['n']}, z={z})")
            lines.append(f"  {key:<24} " + " | ".join(cells))


def format_report(rows: list[dict]) -> str:
    events = dedupe_events(rows)
    lines = [
        f"Press-release outcomes: {len(events)} events "
        f"({len(rows)} scored releases before de-duplication)",
        "Excess return vs XIU.TO from the first tradeable open. "
        "Each cell: median / 1%-trimmed mean / %beat XIU (n, z of %beat vs the whole section).",
        "Plain means are left out: a few penny-stock/split outliers carry them.",
        "UNVALIDATED until n is in the hundreds -- |z| < 2 is noise, and with dozens "
        "of groups a few |z| near 3 turn up by chance.",
    ]
    _format_groups([{**r, "all": "all events"} for r in events], ("all",) + REPORT_GROUPS, lines)

    fin = [r for r in events if r.get("financing")]
    if fin:
        lines.append("")
        lines.append(f"######## Financings with extracted terms: {len(fin)} events ########")
        lines.append("discount = issue price vs last close before the release (CAD deals on "
                     "Canadian symbols only); dilution = securities offered / shares outstanding.")
        lines.append("Groups not stated in a release are shown as (none).")
        _format_groups(fin, FINANCING_GROUPS, lines, sort_by_key=True)
    return "\n".join(lines)


def read_financing_terms(pr_db_path) -> dict:
    """{guid: terms dict} from press_releases.db's financing_terms, read
    read-only (this package never writes there) -- {} if the DB or the
    table doesn't exist yet."""
    import sqlite3

    try:
        conn = sqlite3.connect(f"file:{pr_db_path}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        return {}
    try:
        cur = conn.execute("SELECT * FROM financing_terms")
        cols = [d[0] for d in cur.description]
        return {r[0]: dict(zip(cols, r)) for r in cur.fetchall()}
    except sqlite3.OperationalError:
        return {}
    finally:
        conn.close()


if __name__ == "__main__":
    from config import PRESS_RELEASE_DB_PATH
    from news_watchlist import store

    conn = store.connect()
    rows = attach_financing(store.list_scored_outcomes(conn), read_financing_terms(PRESS_RELEASE_DB_PATH))
    print(format_report(rows))
