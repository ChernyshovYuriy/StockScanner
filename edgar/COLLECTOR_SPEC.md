# EDGAR Collector System — Build Spec (for Claude Code)

A scheduled collector that runs on a **Jetson Nano (JetPack 4.x, Python 3.6)**,
sweeps SEC EDGAR daily for "big money" movements, stores them, and emails a
digest of **flagged hits only** via Gmail SMTP. Quiet day = no email.

This spec is the build plan. The existing `edgar/` package (core, fundamentals,
insiders, scanner, store) is the foundation; this adds the collection,
digest, email, and scheduling layers on top, and hardens everything for the
Nano. Build in the numbered order; each step is testable against live EDGAR.

---

## 0. HARD PLATFORM CONSTRAINTS (read first — these are not optional)

**Target: Python 3.6 on ARM (aarch64), JetPack 4.x.** All code MUST run there.

- `requires-python = ">=3.6"` in pyproject.toml (NOT 3.10).
- **No** walrus `:=`, **no** `match/case`, **no** `dict1 | dict2` merge, **no**
  `X | None` type unions (use `Optional[X]`), **no** `from __future__` tricks
  needed but f-strings are fine (3.6 has them).
- Dependencies: `requests` only (pure-Python, installs fine on ARM). Everything
  else stdlib: `sqlite3`, `smtplib`, `email`, `xml`, `json`, `datetime`,
  `argparse`, `logging`. **Do NOT add pandas/numpy** — heavy to build on ARM,
  unnecessary here.
- Low memory: process filings one at a time, stream, never load a full day's
  filings into a giant list if it can be avoided. The Nano has limited RAM.
- Assume the job runs headless under cron/systemd with no terminal.

---

## 1. Architecture: TWO loops, not per-ticker

Do NOT walk all ~6,000 tickers fetching each. That's 30k–90k requests per sweep.
Instead:

### Event loop (DAILY, business days, after close)
The spine. Pull EDGAR's **daily index** (one file lists every filing
market-wide), filter to forms of interest, parse, store. A few requests, runs
in seconds.
- Forms: `4` (insider), `SC 13D`/`SC 13D/A` (activist), `SC 13G`/`SC 13G/A`
  (passive), `8-K` (catalyst).
- This is market-wide by default — the whole point is catching names you don't
  already watch. No watchlist filter for collection.

### Fundamentals loop (MONTHLY or lazy)
Per-ticker `companyfacts` is the only thing that needs a per-ticker fetch.
Companies report quarterly, so refresh monthly — OR lazily fetch only for
tickers that surfaced in the event loop with a flagged hit. Prefer lazy: it
keeps request volume tiny.

### 13F loop (QUARTERLY)
Institutional holdings update only quarterly. Daily/weekly is pointless. Run
after each quarter's 45-day deadline. Lowest priority; build last.

---

## 2. Cadence (decided — implement exactly this)

- **Event loop: daily, Mon–Fri, ~6:30pm ET** (after EDGAR's daily index for the
  day is complete). NOT weekly — weekly adds up to 7 days staleness on top of
  the 4–10 day filing lag, and you'd have to stitch 5 daily indices anyway.
- **Backfill on each run:** re-scan the last **5 business days**, deduped by
  accession number. This catches anything missed during downtime/holidays so
  the system self-heals instead of silently losing days.
- **Fundamentals: lazy** (fetch on flagged hit) + a monthly full refresh of
  watchlisted/seen CIKs.
- **13F: quarterly.**
- Weekends/holidays: index may be absent. **Handle a missing index by skipping
  and logging — never crash.**

---

## 3. What counts as a "flagged hit" (what triggers an email)

The email contains ONLY these. Everything else is stored silently for later
analysis but does not generate noise.

### Insider buy flag
- transaction code `P` (open-market purchase), acquired (`A`).
- dollar value (shares x price) >= configurable `MIN_BUY_VALUE` (default
  $250,000 — tune later).
- label `discretionary` vs `scheduled`: scheduled if `aff10b5One=true` OR any
  transaction footnote matches /10b5-1/i. Discretionary buys rank above
  scheduled.
- **Cluster bonus:** if >=2 distinct insiders bought the same issuer within a
  rolling 90-day window (query the DB), mark CLUSTER — strongest insider signal.
- **Merge split filings:** same owner+issuer+period across multiple Form 4s =
  one logical buy (the "first/second of two" case). Dedup/merge before valuing.

### Activist stake flag
- any `SC 13D` or `SC 13D/A` filed → flag (activist intent = strong).
- `SC 13G` → store, lower priority (passive); include only if you want.
- (Body parsing of filer name + stake % is a refinement — see step 6. v1 can
  flag "13D filed on TICKER by <filer from header>" using the index/header
  alone.)

Sells (code S), grants (A under plans), option exercises (M), tax withholding
(F) → stored, NEVER flagged. They carry little signal and are mostly 10b5-1 /
liquidity.

---

## 4. Storage (extend existing store.py)

Keep SQLite, keyed on CIK + accession. Tables:
- `scan_hits` (exists) — every form-of-interest filing seen (the raw log).
- `insider_buys` (exists) — parsed P-code buys; ADD columns: `dollar_value`,
  `is_scheduled` (0/1), `merged_accessions` (for split filings).
- `activist_filings` (NEW) — 13D/G hits: cik, ticker, form, filer, pct,
  filing_date, accession, url.
- `fundamentals` (exists) — latest snapshot per cik+metric+period.
- `email_log` (NEW) — date, hit_count, sent (0/1) — so you never double-send and
  can see history.

**Dedup is by accession number everywhere.** Re-running a day must be
idempotent — `INSERT OR IGNORE`.

---

## 5. Email digest (Gmail SMTP, stdlib only)

- `smtplib.SMTP_SSL("smtp.gmail.com", 465)` + Gmail **app password** (NOT the
  account password). Credentials from env vars (`GMAIL_USER`, `GMAIL_APP_PW`,
  `DIGEST_TO`) — never hardcode, never commit.
- Build digest from TODAY's flagged hits (after backfill dedup, only genuinely
  new ones — check `email_log`/sent state).
- **If zero flagged hits → exit without sending.** Quiet day = silence.
- Plain-text body (Nano-friendly, no HTML needed). Group by category:
  ```
  EDGAR digest — 2026-06-05

  INSIDER BUYS (discretionary)
    CDE  COEUR MINING   Dir. J. Smith   $480,000   2026-06-02   [CLUSTER: 2 insiders/60d]
  ACTIVIST STAKES (13D)
    XYZ  filed by <Filer>   2026-06-03   <url>

  (research triggers — not recommendations. one filing is never enough to act.)
  ```
- Subject line carries the headline count: `EDGAR: 2 insider buys, 1 activist`.

---

## 6. Build order (each step testable on live EDGAR)

1. **Port to Py3.6 + harden core/scanner** — fix pyproject floor, scrub any
   3.10-only syntax, make `scan_day` tolerate a missing/partial daily index
   (skip+log, no crash). Add 5-day backfill with accession dedup.
2. **Insider buy valuation + flags** — dollar value, scheduled-vs-discretionary
   (use the two MU Form 4s as fixtures — both 10b5-1 sells, so they must produce
   ZERO flags), split-filing merge, 90-day cluster query.
3. **Activist (13D) flagging** — v1 from index/header (form + issuer + filer
   name). Store to `activist_filings`. (Body parse of stake % = refinement.)
4. **Email digest** — assemble from flagged hits, Gmail SMTP, quiet-day skip,
   email_log dedup.
5. **Scheduler + deployment** — systemd timer (preferred on Nano) OR cron entry,
   Mon–Fri ~6:30pm ET. Env-var config. Logging to a rotating file.
6. **(Later) Fundamentals YoY + 13F quarterly + convergence report** — per
   CONVERGENCE_SPEC.md. Not needed for v1 collection; the digest works without
   them. Collection first, analysis next, exactly as planned.

---

## 7. Deployment notes (Jetson Nano)

- Use a venv: `python3 -m venv ~/edgar-venv && source ~/edgar-venv/bin/activate
  && pip install -e .` (pure-Python deps build fine on ARM).
- **systemd timer** is more robust than cron on a headless Nano (survives
  reboots cleanly, easier logging). Provide both a `.service` and a `.timer`
  unit in a `deploy/` folder.
- Set `USER_AGENT` (SEC requires real contact) and the three Gmail env vars in
  the service's `Environment=` or an `EnvironmentFile=`.
- Timezone: the Nano may be UTC. Schedule in the Nano's actual TZ so "after US
  market close" lands correctly (≈22:30–23:30 UTC depending on DST). Note this
  in the unit file.
- SEC rate limit (10 req/s) is trivially satisfied by a daily index job, but the
  limiter in core.py stays in place for the lazy fundamentals fetches.
- Robustness: wrap the whole run in try/except → log + (optional) email an error
  notice to yourself, so a silent failure doesn't go unnoticed for weeks.

---

## 8. Honest limits (keep in README and in the email footer)

- Every filing is a LAGGED disclosure (4–10+ days). You follow footprints, not
  live position. You won't beat the filing-day price move; you gain systematic,
  market-wide attention earlier than most retail.
- A flagged buy is a RESEARCH TRIGGER, not a recommendation. One filing is never
  enough. Sells carry little signal and are intentionally never flagged.
- The collector's value compounds: after weeks it holds a queryable history that
  the analysis layer (convergence) runs against. That's why collection comes
  first even before analysis is good.
- This is not financial advice and not a price predictor. The decision — business
  quality, valuation, your risk tolerance — stays with you.
```
