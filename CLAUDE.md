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
python virtual_buy.py                             # virtual entry execution (9:30 AM)
python position_monitor.py --mode pre-close       # position monitoring with sells (3:30 PM)
python position_monitor.py --mode post-close      # informational EOD run

# Run backtest
python run_backtest.py --start 2022-01-01 --end 2024-01-01
python run_backtest.py --start 2022-01-01 --end 2024-01-01 --sweep   # parameter sweep
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

- **`main.py`** — End-of-day pipeline (4:30 PM). Runs: regime check (XIU.TO vs 200-day SMA) → universe builder → screener → entry pipeline → send report. Skips signal generation in bear regime but still sends a report.
- **`virtual_buy.py`** — Virtual entry execution (9:30 AM). Reads pending intents and cash from the database, sizes positions using `RISK_PER_TRADE_PCT` from `config.py`, and writes new positions back to the database.
- **`position_monitor.py`** — Position monitoring / virtual exits (3:30 PM pre-close and post-close). Reads open positions from the database and applies exit rules.

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
- **`db.py`** — DuckDB persistence layer. All live-mode state lives in `data/trading.db`. Call `init_db()` at the start of each service. Six tables: `account` (cash), `positions` (open), `trades` (closed, append-only), `signals` (pipeline state machine), `intents` (buy queue), `transactions` (unified BUY+SELL ledger). Schema changes are straightforward — DuckDB supports full `ALTER TABLE`.
- **`time_utils.py`** — Injectable backtest clock. `set_backtest_clock(dt)` pins time for simulation; `None` restores live wall-clock. All time-dependent code calls `market_now()` / `market_today()` from here.
- **`market_data.py`** — Two data providers behind a structural interface (`MarketDataProvider`): `LiveDataProvider` (wraps yfinance for live mode) and `HistoricalSliceProvider` (pre-loaded dict with strict `as_of` cutoff to prevent lookahead bias).
- **`portfolio.py`** — In-memory `PortfolioState` for backtesting only. No file I/O; the live-mode equivalent is `db.py`.
- **`backtest_runner.py`** — Day-by-day simulation. For each day: after-close screener + pipeline, then next-open buys at open price, then daily position monitor.
- **`canadian_stock_screener.py`** — Multi-factor momentum screener (Weinstein Stage II, RS vs XIU.TO, MACD, OBV, ADX, VAM, 52w proximity). Universe and weights controlled by `CONFIG` dict inside the file.
- **`auto_pipeline.py`** — Consumes screener CSV outputs, runs three pattern detectors (VCP, EMA pullback reclaim, base breakout), advances a signal state machine (`FORMING → AT_PIVOT → CONFIRMED → ACTIVE/FAILED`), and writes confirmed intents to the database.
- **`send_report.py`** — Gmail SMTP sender. Has two distinct responsibilities: `send_report()` sends the daily HTML pipeline report (called from `main.py` and `position_monitor.py`); `send_transaction_email()` / `build_transaction_html()` sends a trade notification email after every buy or sell (called from `virtual_buy.py` and `position_monitor.py`). Credentials are read from `.env` (`GMAIL_SENDER`, `GMAIL_APP_PASSWORD`, `GMAIL_RECIPIENT`). Silently skips if not configured.

### Scope discipline

This codebase applies strict scope discipline (see `AGENTS.md`):
- Make the minimum change required; do not improve nearby code.
- Preserve all existing comments verbatim unless a code change makes them factually wrong.
- Do not rename, move, or refactor unless explicitly requested.
- Do not convert in-code constants to CLI arguments, env vars, or config files.
- If a change outside the immediate scope seems necessary, stop and explain before making it.

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
