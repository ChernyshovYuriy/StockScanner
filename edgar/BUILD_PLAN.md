# BUILD PLAN — start here

This `edgar` module is a **working, tested foundation** for EDGAR domain
logic plus **build specs**. It will be integrated into the existing
**StockScanner** repo (github.com/ChernyshovYuriy/StockScanner) as a separate
operational service that REUSES StockScanner's infrastructure.

Read the specs in this order:
1. **INTEGRATION_SPEC.md** — how this becomes a 4th StockScanner service,
   reusing config/email/logging/scheduling. READ FIRST — it sets context and
   supersedes the Py3.6 section of the collector spec (Nano runs Python 3.12).
2. **COLLECTOR_SPEC.md** — the daily collector design: two-loop architecture,
   daily cadence, dedup, flag rules, digest. (Ignore §0 Py3.6 constraints — see
   integration spec. The architecture and flag logic still apply.)
3. **CONVERGENCE_SPEC.md** — the later analysis layer: stack independent signals,
   flag where multiple agree. Build after collection runs and has history.

## What already works (validated)
- `edgar/core.py` — ticker→CIK, rate-limited fetch (SEC 10 req/s), cache.
- `edgar/fundamentals.py` — XBRL metrics; period-aware EPS; stale-value
  rejection; liabilities = assets−equity fallback. (Verified on MU, CDE.)
- `edgar/insiders.py` — Form 4 parse; isolates open-market buys (code P);
  ignores sells/grants/holdings. (Verified on real MU Form 4s: all 10b5-1
  sells → correctly zero buys.)
- `edgar/scanner.py` — daily-index scan; routes 4/13D/13G/8-K/13F; watchlist or
  all-filers.
- `edgar/store.py` — SQLite keyed on CIK.
- `edgar/run.py` — CLI (`ticker`, `watchlist`, `scan`) for dev/testing.
- `tests/` — offline logic tests, 3 passing.

## Target
Nano (Python 3.12, ARM). EDGAR collector runs as a service beside StockScanner's
`main.py` / `virtual_buy.py` / `position_monitor.py`, using its `send_report.py`,
`config.py`, `log_utils.py`, `time_utils.py`. Daily, business days, after US
market close. Email digest of flagged hits only — quiet day, no email.

## Build order
1. Integrate `edgar/` package into StockScanner; wire config + logging.
2. COLLECTOR_SPEC steps 1–3: harden scanner (missing-index tolerance, 5-day
   backfill, dedup), insider valuation + flags (10b5-1 detection, split-filing
   merge, cluster query), 13D flagging from index/header.
3. Digest builder → StockScanner's `send_report.py`; quiet-day skip; email_log
   dedup.
4. Schedule as a service (match the mechanism the existing 3 services use).
5. Later: fundamentals YoY, 13F quarterly, convergence report; eventual
   cross-feed with momentum screen (BLOCKED on US ticker coverage — see
   integration spec).

## Guiding principle (don't lose this)
Every filing is a lagged disclosure — the tool surfaces *footprints* of big
money earlier and more systematically than the crowd, as research triggers. Not
a price predictor, not financial advice. This EDGAR/ownership work is the
deliberate fundamentals counterweight to StockScanner's technical/momentum
approach. Keep that framing in code comments, README, and the email footer.
