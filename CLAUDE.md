# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# First-time init — create the database and set starting capital (run once)
python -c "from db import init_db, set_cash; init_db(); set_cash(50_000)"

# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_db.py -v            # database layer unit tests
pytest tests/test_integration.py -v   # service integration tests
pytest tests/test_backtest_refactor.py -v

# Run tests by phase gate (backtest refactor)
pytest -v -m phase1        # clock injection (time_utils.py)
pytest -v -m phase2        # MarketDataProvider interface
pytest -v -m phase3        # PortfolioState abstraction
pytest -v -m phase4        # BacktestRunner day-by-day simulation
pytest -v -m phase5        # HTML backtest report generation
pytest -v -m phase6        # CLI entry point (run_backtest.py)
pytest -v -m characterization  # golden-value business logic locks

# Run the three scheduled services manually
python main.py   # end-of-day pipeline (4:30 PM); --tickers-url overrides CAN_TICKERS_URL from config.py
python virtual_buy.py                             # virtual entry execution (9:45 AM)
python position_monitor.py --mode pre-close       # position monitoring with sells (3:50 PM)
python position_monitor.py --mode post-close      # informational EOD run

# EDGAR collector — separate 4th service (US SEC filings; ~6:30 PM ET)
python edgar_service.py                            # scan + store + email digest of flagged hits
python edgar_service.py --dry-run                  # build & print the digest, send nothing
python -m edgar.run watchlist MU,KEY,AMD           # set the insider-buy watchlist (US tickers)
python edgar_report.py --dry-run                   # on-demand digest of already-collected 13D hits (independent of the daily collector)

# Inspect live state — console report of the trading database
python report.py

# Run backtest
python run_backtest.py --start 2022-01-01 --end 2024-01-01
python run_backtest.py --start 2022-01-01 --end 2024-01-01 --sweep              # exit-param sweep (time_stop × stop_atr)
python run_backtest.py --start 2022-01-01 --end 2025-01-01 --walk-forward-gap   # walk-forward gap filter optimization
python run_backtest.py --tickers https://your-host/path/can_tickers.txt --start 2022-01-01 --end 2024-01-01

# Run the screener standalone
python canadian_stock_screener.py --tickers https://your-host/path/can_tickers.txt

# Query the live database
python -c "
from db import init_db, get_cash, get_open_positions, load_pending_intents
init_db()
print('Cash    :', get_cash())
print('Intents :', len(load_pending_intents()), 'pending')
print('Positions:', get_open_positions())
"
```

## Architecture

### Three scheduled services (do not merge)

The system runs as three separate scheduled entrypoints that must remain independent:

- **`main.py`** — End-of-day pipeline (4:30 PM). Runs: regime check (XIU.TO vs 200-day SMA) → universe builder (`swing_tickers.py`) → screener → entry pipeline → send report. Skips signal generation in bear regime but still sends a report.
- **`virtual_buy.py`** — Virtual entry execution (9:45 AM). Reads pending intents and cash from the database, sizes positions using `RISK_PER_TRADE_PCT` from `config.py`, and writes new positions back to the database, persisting each intent's planned `stop_price` onto the position. Skips execution outside TSX trading hours via `is_market_open()` (a `--dry-run` is still allowed any time).
- **`position_monitor.py`** — Position monitoring / virtual exits (3:50 PM pre-close and post-close). Reads open positions from the database and applies exit rules, honouring the position's persisted `stop_price` as the initial stop when present. Off-hours pre-close runs fall back to daily bars and suppress sells via `is_market_open()`.

### Data flow

```
<CAN_TICKERS_URL>  (remote URL, one ticker per line)
  → canadian_stock_screener.py (StockScreener + DataManager)
  → out/screener_out/YYYYMMDD_HHMM.csv
  → auto_pipeline.py (pattern detectors + signal state machine)
  → out/alerts/, out/report.html
  → data/trading.db [signals, intents tables]
  → virtual_buy.py → data/trading.db [positions, transactions, account tables]
  → position_monitor.py → data/trading.db [trades, transactions tables]
```

### Key modules

- **`config.py`** — `CAN_TICKERS_URL` (the single ticker list URL), output paths, and trading parameters (`MAX_POSITIONS`, `RISK_PER_TRADE_PCT`, `GAP_FILTER_PCT`, `PositionMonitorMode`).
- **`db.py`** — DuckDB persistence layer. All live-mode state lives in `data/trading.db`. Call `init_db()` at the start of each service. Six tables: `account` (cash), `positions` (open, incl. the planned `stop_price`), `trades` (closed, append-only), `signals` (pipeline state machine), `intents` (buy queue), `transactions` (unified BUY+SELL ledger). Schema changes are straightforward — DuckDB supports full `ALTER TABLE`; `init_db()` self-migrates the `stop_price` column via `ADD COLUMN IF NOT EXISTS`.
- **`time_utils.py`** — Injectable backtest clock. `set_backtest_clock(dt)` pins time for simulation; `None` restores live wall-clock. All time-dependent code calls `market_now()` / `market_today()` from here. `is_market_open()` (weekday + 09:30–16:00 ET + TSX holiday set) guards the live services from transacting off-hours and is deterministic under the backtest clock.
- **`market_data.py`** — Two data providers behind a structural interface (`MarketDataProvider`): `LiveDataProvider` (wraps yfinance for live mode) and `HistoricalSliceProvider` (pre-loaded dict with strict `as_of` cutoff to prevent lookahead bias).
- **`backtest_runner.py`** `BacktestConfig.gap_filter_pct` — `None` (default) = no gap filter, matching pre-2026-05 backtest behaviour. Set to e.g. `2.0` to simulate the live `GAP_FILTER_PCT` from `config.py` (skip buys where open > intent_entry × 1.02).
- **`portfolio.py`** — In-memory `PortfolioState` for backtesting only. No file I/O; the live-mode equivalent is `db.py`.
- **`backtest_runner.py`** — Day-by-day simulation. For each day: after-close screener + pipeline, then next-open buys at open price, then daily position monitor.
- **`canadian_stock_screener.py`** — Multi-factor momentum screener (Weinstein Stage II, RS vs XIU.TO, MACD, OBV, ADX, VAM, 52w proximity). Universe and weights controlled by `CONFIG` dict inside the file.
- **`auto_pipeline.py`** — Consumes screener CSV outputs, runs three pattern detectors (VCP, EMA pullback reclaim, base breakout), advances a signal state machine (`FORMING → AT_PIVOT → CONFIRMED → ACTIVE/FAILED`), and writes confirmed intents to the database.
- **`send_report.py`** — Gmail SMTP sender. Has two distinct responsibilities: `send_report()` sends the daily HTML pipeline report (called from `main.py` and `position_monitor.py`); `send_transaction_email()` / `build_transaction_html()` sends a trade notification email after every buy or sell (called from `virtual_buy.py` and `position_monitor.py`). `send_text_email(subject, body)` sends a plain-text email (used by the EDGAR collector). Credentials are read from `.env` (`GMAIL_SENDER`, `GMAIL_APP_PASSWORD`, `GMAIL_RECIPIENT`). Silently skips if not configured.
- **`edgar_service.py`** / **`edgar/`** — Separate 4th service (US SEC EDGAR), the fundamentals/ownership counterweight to the TSX momentum system. Daily after the US close it sweeps EDGAR's daily index, stores filings in its OWN SQLite DB (`data/edgar.db`, keyed on CIK+accession — NOT `trading.db`), and emails a plain-text digest of flagged hits (interim: watchlist insider open-market buys + all SC 13D/13G) via `send_report.send_text_email()`. Quiet day = no email. Reuses shared config/email/logging/lock; stays independent of the TSX trio (US filers only). Config in the `EDGAR_*` block of `config.py`; dev sandbox + specs in `edgar/`. Every filing is a lagged disclosure — a research trigger, not a price predictor. `edgar_report.py` is a separate on-demand command that parses already-collected 13D bodies from `scan_hits` and emails a digest.
- **Support modules** — `report.py` (console dump of the live trading DB: cash, positions, trades, signals, intents), `report_html.py` (light-theme HTML email report — Gmail forcibly overrides dark backgrounds), `schema_keys.py` (shared column/field keys used across pipeline, buy, monitor, and report modules), `concurrent_utils.py` (fcntl file lock in `out/locks/` preventing overlapping service runs), `log_utils.py` (JSON run logging via `market_now()`).

### Scope discipline

This codebase applies strict scope discipline (see `AGENTS.md`):
- Make the minimum change required; do not improve nearby code.
- Preserve all existing comments verbatim unless a code change makes them factually wrong.
- Do not rename, move, or refactor unless explicitly requested.
- Do not convert in-code constants to CLI arguments, env vars, or config files.
- If a change outside the immediate scope seems necessary, stop and explain before making it.

### Deployment (systemd)

`system/` holds `.service`/`.timer` units for the four services (main, buy, monitor, edgar). All timers deliberately use `Persistent=false`: a slot missed while the host was down is skipped, not run late on stale prices. Do not change this to `true` — off-hours catch-up runs previously caused fills at stale quotes; `is_market_open()` is the second layer of the same defence. Install/status commands are in `README.md` ("Systemd deployment").

### State files (live mode)

- `data/trading.db` — DuckDB database; single source of truth for all live-mode state (cash, positions, trades, signals, intents, transactions). Query directly with `duckdb.connect("data/trading.db")`.
- `out/` — generated outputs (screener CSVs, alerts, HTML report, logs, locks)

### Querying the database

```python
import duckdb
conn = duckdb.connect("data/trading.db")

conn.execute("SELECT * FROM transactions ORDER BY trade_date").df()   # full ledger
conn.execute("SELECT * FROM positions").df()                          # open positions
conn.execute("SELECT * FROM trades ORDER BY sell_date").df()          # closed trades
conn.execute("SELECT * FROM intents WHERE intent_status = 'PENDING'").df()
conn.close()
```
