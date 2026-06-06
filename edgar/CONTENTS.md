# edgar — package contents

Validated EDGAR domain logic + build specs for integration into StockScanner.

## Read order (specs)
1. BUILD_PLAN.md       — start here; orientation + build order
2. INTEGRATION_SPEC.md — how this becomes a StockScanner service (reuse infra)
3. COLLECTOR_SPEC.md   — daily collector design (ignore §0 Py3.6; Nano is 3.12)
4. CONVERGENCE_SPEC.md — later analysis layer

## Code (validated, tested)
- edgar/core.py        — ticker->CIK, rate-limited fetch (SEC 10/s), cache
- edgar/fundamentals.py — XBRL metrics; period-aware EPS; stale rejection;
                          liabilities=assets-equity fallback
- edgar/insiders.py    — Form 4 parse; isolates open-market buys (code P)
- edgar/scanner.py     — daily-index scan; routes 4/13D/13G/8-K/13F
- edgar/store.py       — SQLite keyed on CIK
- edgar/run.py         — CLI: ticker / watchlist / scan
- tests/test_logic.py  — offline logic tests (3 passing)

## Project files
- pyproject.toml       — installable package (requires-python >=3.10)
- requirements.txt     — requests only; rest stdlib
- watchlist.example.txt
- .gitignore

## Quick start (dev/testing in this standalone repo)
    python -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"
    # set USER_AGENT in edgar/core.py to a real contact email
    python -m edgar.run ticker MU
    pytest -q

## Integration target
Becomes a 4th service in StockScanner (github.com/ChernyshovYuriy/StockScanner),
reusing send_report.py / config.py / log_utils.py / time_utils.py. Runs daily on
the Jetson Nano (Python 3.12), emails a digest of flagged hits only.

## Guiding principle
Filings are LAGGED disclosures — this surfaces footprints of big money as
research triggers, earlier/more systematically than the crowd. Not a price
predictor, not financial advice. The fundamentals/ownership counterweight to
StockScanner's technical/momentum approach.
