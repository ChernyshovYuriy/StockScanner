"""
rbc_reports/prices.py
=====================
Step 3 of the RBC research tracker: price context for the stocks RBC
writes about, so a bullish note on a beaten-down or turning-up name (the
user's own pattern: GS/JPM at their lows, Celestica off the bottom) stands
out on the /rbc tab's Candidates table.

Three parts, all run once per trading day by rbc_reports_service.py:

1. resolve_symbols() -- RBC prints a bare symbol ("GS", "L", "CHR") and
   only sometimes a Bloomberg exchange code. Each one is mapped to a Yahoo
   symbol by trying North-American listings in turn (.TO, then US bare,
   then .V) and accepting the first whose Yahoo company name matches the
   name RBC printed -- so "L" (Loblaw) is not taken as Loews (and each
   (symbol, company) pair is mapped separately: CFR is both Cullen/Frost
   and Richemont). A European/
   Australian listing is left unresolved (no price): the user's account
   trades North America. Stored in `symbol_map`, never re-checked (a miss
   too, unless --retry; --recheck-before re-resolves the found ones after
   a rule change).
2. sync_prices() -- daily bars for every resolved subject ticker of a
   company note in the last RBC_OUTCOME_WINDOW_DAYS (plus the SPY/XIU.TO
   benchmarks), through market_data_cache.py's DuckDB cache
   (data/market_cache.db -- topped up, not re-fetched). The same bars
   score each note's forward return (outcomes.py, step 5).
3. price_context() -- per symbol: last close, 52-week high/low, % off the
   high, position in the 52-week range, 20-day return, close vs its
   50-day SMA, and "turning up" (close above a rising 20-day SMA). Stored
   in `price_context`; each note's base close (the last close before RBC
   published it) in `report_prices`, so the tab can show the move since.

The dashboard only ever reads those rbc_reports.db tables -- never
market_cache.db, whose DuckDB writer lock a read could collide with.

Yahoo budget: every request (an info lookup, or one symbol's bars) is
paced at RBC_YAHOO_PER_SECOND (0.4/s, the repo's bulk-job ceiling), and a
rate-limit error stops the step at once rather than prolong a block. A
step that stops early simply carries on next run.

  python -m rbc_reports.prices               # resolve + sync + context (ignores the once-a-day gate)
  python -m rbc_reports.prices --retry       # also re-check symbols that failed to resolve
  python -m rbc_reports.prices --recheck-before 2026-11-11   # re-resolve mappings found before that day's rule change
"""
from __future__ import annotations

import argparse
import re
import time
from datetime import datetime, timedelta

from config import (RBC_OUTCOME_WINDOW_DAYS, RBC_RESOLVE_MAX_PER_RUN, RBC_YAHOO_CHUNK,
                    RBC_YAHOO_PER_SECOND)
from rbc_reports import store
from time_utils import last_trading_day_on_or_before, market_now

# Bloomberg exchange code -> Yahoo suffixes to try, in order. A code not
# listed here is a non-North-American listing: left unresolved.
# TSX first: an interlisted Canadian name (CNQ, SU, ENB, SLF) is priced in
# CAD against XIU.TO -- the listing the user holds -- not as its NYSE line
# in USD against SPY. A US company has no .TO listing (a CDR is quoteType
# ETF and skipped), so it still lands on its bare US symbol.
_SUFFIXES_BY_CODE = {"": (".TO", "", ".V"), "US": ("",), "CN": (".TO", ".V")}
HISTORY_DAYS = 400          # calendar days of bars: a full 52 weeks plus SMA warm-up
_NAME_STOPWORDS = frozenset("""
    the inc incorporated corp corporation co company ltd limited plc group holdings holding
    sa ag nv se spa s a p l trust reit lp llc and of class ordinary shares common stock
    de cv ab asa oyj the
""".split())
# A place word in one name but not the other means a different company,
# however many other words they share: "Sandfire Resources Limited" (ASX)
# is not "Sandfire Resources America Inc." (TSXV).
_PLACE_WORDS = frozenset("""
    america american americas canada canadian usa australia australian asia pacific europe
    european uk britain british africa african mexico brazil latin
""".split())


class YahooRateLimited(RuntimeError):
    """Yahoo is refusing requests; stop rather than make the block longer."""


def _is_rate_limit(exc: Exception) -> bool:
    return "RateLimit" in type(exc).__name__ or "Too Many Requests" in str(exc)


class _Pacer:
    def __init__(self, per_second: float, sleep=time.sleep):
        self._gap, self._next, self._sleep = 1.0 / per_second, 0.0, sleep

    def wait(self, n: int = 1):
        now = time.monotonic()
        delay = max(0.0, self._next - now)
        self._next = max(now, self._next) + self._gap * n
        if delay:
            self._sleep(delay)


def _name_tokens(name: str) -> list:
    words = re.findall(r"[a-z0-9]+", (name or "").lower().replace("&", " and ").replace("+", " "))
    return [w for w in words if w not in _NAME_STOPWORDS]


def names_match(rbc_name: str, yahoo_name: str) -> bool:
    """Same first significant word, at least half of the shorter name's
    words shared, and two of them unless a name is a single word, and no
    place word (_PLACE_WORDS) in only one of them --
    "Goldman Sachs Group Inc." matches "The Goldman Sachs Group, Inc.";
    "First Horizon" never matches "First Industrial Realty"."""
    a, b = set(_name_tokens(rbc_name)), set(_name_tokens(yahoo_name))
    if not a or not b or _name_tokens(rbc_name)[0] != _name_tokens(yahoo_name)[0]:
        return False
    if (a ^ b) & _PLACE_WORDS:
        return False
    shared, shorter = len(a & b), min(len(a), len(b))
    return shared / shorter >= 0.5 and (shared >= 2 or shorter == 1)


def yahoo_candidates(symbol: str, exchange: str) -> list:
    """Yahoo symbols to try for an RBC symbol, e.g. ("CTC.A", "CN") ->
    ["CTC-A.TO", "CTC-A.V"]. A share-class dot becomes Yahoo's dash."""
    suffixes = _SUFFIXES_BY_CODE.get(exchange)
    base = symbol.strip(".").replace(".", "-")
    if not suffixes or not base or not re.fullmatch(r"[A-Z0-9-]+", base):
        return []
    return [base + s for s in suffixes]


def resolve_symbols(conn, limit: int | None = RBC_RESOLVE_MAX_PER_RUN, retry: bool = False,
                    info_fn=None, pacer=None) -> dict:
    """Maps not-yet-checked subject tickers to Yahoo symbols. Returns
    {"checked", "resolved", "stopped"} -- `stopped` is True when Yahoo
    rate-limited the run (the rest wait for the next one)."""
    if info_fn is None:
        from market_data import DEFAULT_PROVIDER
        info_fn = DEFAULT_PROVIDER.get_info
    pacer = pacer or _Pacer(RBC_YAHOO_PER_SECOND)
    todo = store.unresolved_subjects(conn, retry=retry, limit=limit)
    result = {"checked": 0, "resolved": 0, "stopped": False}
    now = market_now().isoformat()
    for symbol, exchange, company in todo:
        try:
            found = _lookup(symbol, exchange, company, info_fn, pacer)
        except YahooRateLimited:
            result["stopped"] = True
            break
        store.save_symbol(conn, symbol, exchange, company, *found, checked_at=now, commit=False)
        result["checked"] += 1
        result["resolved"] += found[0] is not None
    conn.commit()
    return result


def _lookup(symbol, exchange, company, info_fn, pacer) -> tuple:
    """(yahoo_symbol, yahoo_name, currency) of the first candidate listing
    whose name matches, else (None, None, None). Raises YahooRateLimited."""
    for cand in yahoo_candidates(symbol, exchange):
        pacer.wait()
        try:
            info = info_fn(cand) or {}
        except Exception as exc:
            if _is_rate_limit(exc):
                raise YahooRateLimited(f"{cand}: {exc}") from exc
            continue
        # A Canadian Depositary Receipt (HON.TO: Honeywell, priced in
        # CAD at ~1/20th of a share) is quoteType ETF -- a different
        # price series, so only a real share listing counts.
        if info.get("quoteType", "EQUITY") != "EQUITY":
            continue
        names = [info.get("longName"), info.get("shortName")]
        if any(n and names_match(company, n) for n in names):
            return cand, next(n for n in names if n), info.get("currency")
    return None, None, None


def recheck_symbols(conn, before: str, info_fn=None, pacer=None, commit_every: int = 20) -> dict:
    """Re-resolves every FOUND mapping last checked before `before` -- the
    ISO date the rules changed (2026-11-11: TSX-first order, the place-word
    name check) -- with the current rules. A row that no longer matches
    becomes a miss. Resumable: a re-checked row's checked_at is now, past
    `before` (so `before` must not be later than today), so a run
    stopped by a rate limit carries on where it left off. Commits every
    `commit_every` rows so the scheduled service's writes aren't blocked
    for the whole (long, paced) run. Returns {"checked", "changed",
    "stopped"}."""
    if info_fn is None:
        from market_data import DEFAULT_PROVIDER
        info_fn = DEFAULT_PROVIDER.get_info
    pacer = pacer or _Pacer(RBC_YAHOO_PER_SECOND)
    rows = conn.execute(
        "SELECT symbol, exchange, company, yahoo_symbol FROM symbol_map "
        "WHERE yahoo_symbol IS NOT NULL AND checked_at < ? ORDER BY symbol", [before]).fetchall()
    result = {"checked": 0, "changed": 0, "stopped": False}
    for symbol, exchange, company, old in rows:
        try:
            found = _lookup(symbol, exchange, company, info_fn, pacer)
        except YahooRateLimited:
            result["stopped"] = True
            break
        store.save_symbol(conn, symbol, exchange, company, *found,
                          checked_at=market_now().isoformat(), commit=False)
        result["checked"] += 1
        result["changed"] += found[0] != old
        if result["checked"] % commit_every == 0:
            conn.commit()
    conn.commit()
    return result


def price_context(df, as_of=None) -> dict | None:
    """Context from daily bars (Close column, date index, oldest first).
    None with fewer than 60 bars -- too little for a 50-day SMA and a
    meaningful 52-week range."""
    closes = df["Close"].dropna()
    if as_of is not None:
        closes = closes[closes.index <= as_of]
    if len(closes) < 60:
        return None
    last_date = closes.index[-1]
    year = closes[closes.index > last_date - timedelta(days=365)]
    close, hi, lo = float(closes.iloc[-1]), float(year.max()), float(year.min())
    sma20 = closes.rolling(20).mean()
    sma50 = float(closes.rolling(50).mean().iloc[-1])
    rising = float(sma20.iloc[-1]) > float(sma20.iloc[-6])
    return {
        "as_of": last_date.strftime("%Y-%m-%d"),
        "close": round(close, 4),
        "high_52w": round(hi, 4),
        "low_52w": round(lo, 4),
        "pct_off_high": round(close / hi - 1, 4) if hi else None,
        "range_pos": round((close - lo) / (hi - lo), 4) if hi > lo else None,
        "ret_20d": round(close / float(closes.iloc[-21]) - 1, 4),
        "above_sma50": int(close > sma50),
        "turning_up": int(close > float(sma20.iloc[-1]) and rising),
    }


def base_close(df, published_at: str):
    """(date, close) of the last close before RBC published the note: a
    note out after 16:00 ET is measured from that day's close, an earlier
    one from the previous session's."""
    published = datetime.fromisoformat(published_at)
    closes = df["Close"].dropna()
    day = published.date()
    eligible = closes[closes.index.date <= day] if published.hour >= 16 else \
        closes[closes.index.date < day]
    if eligible.empty:
        return None
    return eligible.index[-1].strftime("%Y-%m-%d"), round(float(eligible.iloc[-1]), 4)


def sync_prices(conn, now: datetime | None = None, sync_fn=None, load_fn=None,
                pacer=None, cache_conn=None) -> dict:
    """Tops up bars for every resolved symbol in the window, then writes
    price_context, report_prices and note_outcomes. Returns {"symbols",
    "with_context", "scored", "stale", "stopped"} -- `stale` counts symbols
    whose last bar is behind their benchmark's (delisted names, or a
    download that failed). sync_fn/load_fn/cache_conn are test seams over
    market_data_cache.sync_tickers/load_from_cache (cache_conn=None: each
    call opens and closes its own DuckDB connection, so the cache's writer
    lock is never held between chunks)."""
    import market_data_cache as mdc
    from rbc_reports import outcomes

    now = now or market_now()
    since = (now - timedelta(days=RBC_OUTCOME_WINDOW_DAYS)).strftime("%Y-%m-%dT%H:%M")
    notes = store.company_notes_since(conn, since)
    symbols = sorted({ys for _, ys, _ in notes})
    result = {"symbols": len(symbols), "with_context": 0, "scored": 0, "stale": 0, "stopped": False}
    if not symbols:
        return result
    benchmarks = (outcomes.BENCHMARK_US, outcomes.BENCHMARK_CA)
    symbols = list(benchmarks) + [s for s in symbols if s not in benchmarks]
    sync_fn = sync_fn or mdc.sync_tickers
    load_fn = load_fn or mdc.load_from_cache
    pacer = pacer or _Pacer(RBC_YAHOO_PER_SECOND)
    start = (now - timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d")
    end = now.strftime("%Y-%m-%d")
    bench_last = {}
    for i in range(0, len(symbols), RBC_YAHOO_CHUNK):
        chunk = symbols[i:i + RBC_YAHOO_CHUNK]
        pacer.wait(len(chunk))
        try:
            sync_fn(chunk, start, end, conn=cache_conn, quiet=True)
        except Exception as exc:
            if _is_rate_limit(exc):
                result["stopped"] = True
                break
            raise
        # sync_tickers() swallows download errors, so a rate limit shows
        # up as a chunk where no symbol got its latest bar: no bars at all
        # for a new symbol, only the already-cached older ones for a known
        # one (2026-11-10: the alphabetical tail SN..ZWS stayed a day behind
        # while the run reported success).
        got = load_fn(chunk, start, end, conn=cache_conn)
        for b in benchmarks:
            if b in got and not got[b].empty:
                bench_last[b] = got[b].index[-1]
        others = [s for s in chunk if s not in benchmarks]
        if others and not any(_is_fresh(s, got, bench_last) for s in others):
            result["stopped"] = True
            break
    bars = load_fn(symbols, start, end, conn=cache_conn)
    result["stale"] = sum(not _is_fresh(s, bars, bench_last) for s in symbols if s not in benchmarks)

    contexts, bases = {}, []
    for ys in symbols:
        df = bars.get(ys)
        if df is None or df.empty:
            continue
        ctx = price_context(df)
        if ctx:
            contexts[ys] = ctx
    for key, ys, published in notes:
        df = bars.get(ys)
        b = base_close(df, published) if df is not None and not df.empty else None
        if b:
            bases.append((key, ys, *b))
    scored = outcomes.score_notes(notes, bars)
    store.save_price_context(conn, contexts, now.isoformat(), commit=False)
    store.save_report_prices(conn, bases, commit=False)
    store.save_outcomes(conn, scored, now.isoformat(), commit=False)
    conn.commit()
    result["with_context"] = len(contexts)
    result["scored"] = len(scored)
    return result


def _is_fresh(symbol: str, bars: dict, bench_last: dict) -> bool:
    """True when `symbol` has bars up to its benchmark's last bar (SPY for a
    US listing, XIU.TO for a Canadian one -- same exchange calendar, so a
    US-only holiday doesn't read as stale). No benchmark bars to compare
    with: can't tell, so True."""
    from rbc_reports import outcomes

    df = bars.get(symbol)
    if df is None or df.empty:
        return False
    ref = bench_last.get(outcomes.benchmark_for(symbol))
    return ref is None or df.index[-1] >= ref


def due_today(conn, now: datetime | None = None) -> bool:
    """True once per trading day: the last price run was before the
    latest trading day (so the 23:30 run, and weekend runs, skip)."""
    now = now or market_now()
    last = store.get_meta(conn, "prices_last_run")
    latest = last_trading_day_on_or_before(now.date())
    return last is None or last < latest.isoformat()


def run(conn=None, retry: bool = False, force: bool = False) -> dict | None:
    """resolve_symbols() then sync_prices(), once per trading day unless
    `force`. None when not due."""
    conn = conn or store.connect()
    if not force and not due_today(conn):
        return None
    # The day the run STARTED: a 23:30 run can take over half an hour, and
    # stamping it with the after-midnight end date made the next evening's
    # run skip (2026-11-12 -- that night's notes went unpriced).
    started = market_now().date().isoformat()
    resolved = resolve_symbols(conn, retry=retry)
    synced = {"symbols": 0, "with_context": 0, "stopped": True}
    if not resolved["stopped"]:
        synced = sync_prices(conn)
    if not resolved["stopped"] and not synced["stopped"]:
        store.set_meta(conn, "prices_last_run", started)
    return {"resolve": resolved, "sync": synced}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Price context for RBC-covered stocks.")
    p.add_argument("--retry", action="store_true", help="re-check symbols that failed to resolve")
    p.add_argument("--all", action="store_true", help="resolve every pending symbol (no per-run cap)")
    p.add_argument("--recheck-before", metavar="DATE",
                   help="re-resolve found mappings last checked before DATE (after a rule change), then exit")
    args = p.parse_args(argv)
    conn = store.connect()
    if args.recheck_before:
        r = recheck_symbols(conn, args.recheck_before)
        print(f"recheck: {r}")
        return 1 if r["stopped"] else 0
    started = market_now().date().isoformat()   # see run()
    resolved = resolve_symbols(conn, retry=args.retry, limit=None if args.all else RBC_RESOLVE_MAX_PER_RUN)
    print(f"resolve: {resolved}")
    if resolved["stopped"]:
        print("Yahoo rate-limited -- stopped; run again later")
        return 1
    synced = sync_prices(conn)
    print(f"prices: {synced}")
    if not synced["stopped"]:
        store.set_meta(conn, "prices_last_run", started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
