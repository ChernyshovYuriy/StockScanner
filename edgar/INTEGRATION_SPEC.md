# INTEGRATION — EDGAR collector as a StockScanner service

**Decision:** the EDGAR collector is a SEPARATE operational service that REUSES
StockScanner's existing infrastructure. It does not duplicate email, logging,
config, or scheduling. It runs alongside `main.py` / `virtual_buy.py` /
`position_monitor.py` as a fourth service.

This supersedes the Python 3.6 constraints in COLLECTOR_SPEC.md §0 — the Nano
runs **Python 3.12** (`/usr/local/bin/python3.12`). Write normal modern Python.
COLLECTOR_SPEC's architecture (two loops, daily cadence, dedup, flag rules)
still applies; only the language floor and the "build your own email/scheduler"
parts change — those are reused from StockScanner.

---

## Reuse from StockScanner (DO NOT reimplement)

Existing modules in the StockScanner repo to import/extend:

- **`send_report.py`** — email delivery. The EDGAR digest goes out through this,
  not a fresh `smtplib` implementation. Match its function signature; pass it
  the EDGAR digest body + subject.
- **`config.py`** — central settings. Add an EDGAR section (USER_AGENT,
  MIN_BUY_VALUE, forms of interest, backfill window, DB path). Don't scatter
  config across the EDGAR modules.
- **`log_utils.py`** — logging setup. EDGAR service logs through the same
  mechanism (rotating file, headless-friendly).
- **`time_utils.py`** — timezone / market-close timing. Reuse for "after US
  market close" scheduling and any date math, so behavior matches the existing
  services on the Nano's clock.
- **`concurrent_utils.py`** — if the lazy fundamentals fetch ever needs
  parallelism, reuse this rather than rolling new threadpool code (mind the SEC
  10 req/s limit — the existing rate limiter in edgar/core.py must still gate).
- **Report patterns** — `report_html.py` / the URGENT/WATCH/FORMING/EXPIRED tier
  vocabulary maps directly onto EDGAR flags (see below). Reuse the tiering idea.

## Bring in from edgar (the new domain logic)

The `edgar/` package is the part StockScanner doesn't have — EDGAR fetch,
Form 4 parsing, fundamentals, scanner, store. Drop it in as a subpackage
(e.g. `edgar/` inside the StockScanner repo) and build the collector entrypoint
(`edgar_service.py`, analogous to `main.py`) on top of it.

---

## Service shape (mirror main.py's structure)

`edgar_service.py`:
1. load EDGAR config from `config.py`
2. set up logging via `log_utils.py`
3. run the **event loop** (COLLECTOR_SPEC §1): daily-index scan, 5-day backfill,
   parse, store to SQLite (dedup by accession)
4. (lazy) fetch fundamentals for tickers with a flagged hit
5. build the digest of FLAGGED HITS ONLY
6. if non-empty → hand body+subject to `send_report.py`; else exit quietly
7. wrap in try/except → log + optional error email (match how main.py handles
   failure if it does)

## Flag → tier mapping (reuse StockScanner's vocabulary)

- 🔴 **URGENT** — fresh `SC 13D` (activist intent), OR a CLUSTER of discretionary
  insider buys (>=2 insiders, 90d window).
- 🟡 **WATCH** — single discretionary open-market buy >= MIN_BUY_VALUE; or
  `SC 13G` (passive >5%).
- 🟢 **FORMING** — accumulation building but below thresholds (optional; can be
  silent in v1).
- (sells / 10b5-1 / grants → stored, never tiered, never emailed.)

## Scheduling on the Nano

Add a service entry next to the existing three (the README documents them as
operational services run on a schedule). Daily, Mon–Fri, after US market close
(use `time_utils.py` so it matches the Nano's TZ handling). systemd timer or
cron — whatever the existing three services use; match that, don't introduce a
new mechanism.

---

## The cross-feed (LATER — note the blocker)

The high-value future step: a ticker flagged by EDGAR (insider cluster / 13D)
that ALSO appears in StockScanner's momentum screen = fundamentals-conviction
AND technical setup agreeing. That intersection beats either signal alone.

**Blocker to record now:** StockScanner is **TSX/Canadian** (Yahoo, `XIU.TO`
benchmark); EDGAR is **US filers only**. There is currently NO ticker overlap,
so the cross-feed does nothing until the momentum screen is extended to US
tickers. Don't design the EDGAR service assuming StockScanner tickers will
match — keep it US-standalone for now; treat the cross-feed as a separate later
project gated on US momentum coverage.

---

## What stays in the standalone edgar module

The `edgar` module (with its specs, tests, CLI) remains the development/
testing sandbox for the EDGAR domain logic. Iterate and test there with the CLI
(`python -m edgar.run ticker MU`), then the validated `edgar/` package + the
collector entrypoint get integrated into StockScanner as the service. Keeps the
domain logic unit-tested in isolation while the service lives where the infra is.
