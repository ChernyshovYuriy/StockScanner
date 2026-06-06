# edgar-screener

Follows disclosed "big money" footprints in SEC EDGAR — insider buys, activist
stakes, material events — and joins them with company fundamentals. Everything
keys on **CIK**, the SEC's permanent company identifier, so the signals join
cleanly.

## The honest premise (read this first)

Price/pattern systems (P&F, swing) are downstream of what actually moves stocks:
flow, ownership, disclosed positioning. This tool scans the *disclosures*. But
every public filing is a **lagged disclosure of a decision already made** — you
follow footprints, never live position. Real-time order flow / dark-pool prints
are paid Tier-1 data or simply not disclosed. No public scanner gets you there,
and any tool claiming to "detect big-money moves in real time" is selling the
fantasy. Treat this as research tooling that tilts where you look, not an edge
and not a predictor of any individual name.

## Signals, by how much survives reality

| Form        | What it is                      | Timeliness        | Signal quality |
|-------------|---------------------------------|-------------------|----------------|
| 4           | Insider transactions            | ~5 days           | Clustered **open-market buys** = real; sells = noise |
| SC 13D      | Activist >5% stake              | ~10 days          | High — an activist arriving is a catalyst |
| SC 13G      | Passive >5% stake               | periodic          | Medium — big holder crossing threshold |
| 8-K         | Material event                  | ~4 business days  | High — these *are* the catalysts (M&A, exec change, guidance) |
| 13F-HR      | Institutional holdings (qtrly)  | up to 45 days     | Low — stale; context only |

## Layout

```
edgar/
  core.py          identity (ticker->CIK) + rate-limited fetch + cache
  fundamentals.py  XBRL concepts; period-aware EPS; same-period pinning
  insiders.py      Form 4 parse -> open-market buys + cluster flag
  scanner.py       daily-index scan -> route forms for a watchlist
  store.py         SQLite (keyed on CIK) for diffing over time
  run.py           CLI
tests/             offline logic tests (no network)
```

## Setup

```bash
cd edgar
python -m venv .venv && source .venv/bin/activate   # recommended
pip install -e ".[dev]"
# EDIT USER_AGENT in edgar/core.py to a real contact, or EDGAR returns 403.
```

## Use

```bash
python -m edgar.run ticker MU                       # fundamentals + insider buys
python -m edgar.run watchlist MU,AMD,KEY,PLTR       # persist a watchlist
python -m edgar.run scan --days 5                   # watchlist filings, last 5 days
python -m edgar.run scan --days 1 --forms "SC 13D,SC 13D/A" --all   # all activist stakes today
pytest -q                                           # run offline tests
```

## What's verified

- EPS period fix, Form 4 buy isolation, cluster flag, SQLite round-trip — all
  unit-tested offline (`pytest -q`, 3 passing). The EPS fix specifically picks
  the ~90-day quarterly figure over an annual one (the Micron `eps=17` bug).
- Live network calls were NOT run in the build environment (its sandbox blocks
  sec.gov). They work on any normal machine. A 403 on real EDGAR almost always
  means a missing/invalid `USER_AGENT`.

## Known gotchas / next steps (good Claude Code tasks)

- **XBRL tag coverage.** `fundamentals.METRIC_TAGS` is a fallback list; extend
  it when a metric shows `<not tagged>` for names you care about.
- **Insider path, positive case.** Tested empty (large-caps rarely open-market
  buy) and against a fixture; confirm against a real small/mid-cap with recent
  's' code 'P' buys.
- **13D/8-K parsers.** The scanner *routes* these by form type and gives you the
  filing URL; it doesn't yet parse 13D bodies (who/what %) or 8-K item codes.
  Parsing those is the next real layer.
- **The join.** With store.py populated, the query worth running is the
  intersection: watchlist names with recent clustered insider buys AND/OR a
  fresh 13D AND improving fundamentals. Build that as a `report` subcommand.
```
