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

# Manual sell — close one open position at the current market price, on demand
python manual_sell.py SLF.TO                      # sells SLF.TO now
python manual_sell.py SLF.TO --dry-run             # preview only, no DB write

# Web dashboard — live positions + history + sell button (LAN-only, no auth)
python dashboard_app.py                            # dev server on 0.0.0.0:8080 (config.py DASHBOARD_HOST/PORT); /momentum shows the momentum sleeve read-only

# Momentum sleeve — separate DB/capital/services (see config.py MOMENTUM_*)
python momentum_pipeline.py                        # relaxed-ATR universe + screener + entry pipeline (4:30 PM)
python momentum_buy.py                             # virtual entry execution (9:45 AM)
python momentum_monitor.py --mode pre-close        # position monitoring with sells (3:50 PM)
python momentum_monitor.py --mode post-close       # informational EOD run

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
- **`virtual_buy.py`** — Virtual entry execution (9:45 AM). Reads pending intents and cash from the database, sizes positions using `RISK_PER_TRADE_PCT` from `config.py`, and writes new positions back to the database, persisting each intent's planned `stop_price` onto the position. Skips execution outside TSX trading hours via `is_market_open()` (a `--dry-run` is still allowed any time). Also enforces `MAX_POSITIONS_PER_SECTOR` (via `sector_lookup.get_sector()`), skipping a candidate whose GICS sector already holds the cap's worth of open positions — added 2026-08 after a live cluster of same-sector earnings-week stop-outs; 16-fold walk-forward showed no return cost but a significant max-drawdown reduction.
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

- **`config.py`** — `CAN_TICKERS_URL` (the single ticker list URL), output paths, and trading parameters (`MAX_POSITIONS`, `RISK_PER_TRADE_PCT`, `GAP_FILTER_PCT`, `MAX_POSITIONS_PER_SECTOR`, `PositionMonitorMode`).
- **`sector_lookup.py`** — `get_sector(ticker)`: GICS sector lookup for the `MAX_POSITIONS_PER_SECTOR` cap in `virtual_buy.py`, via yfinance `.info['sector']`, cached to `cache/sector_cache.json` (sectors change rarely).
- **`db.py`** — DuckDB persistence layer. All live-mode state lives in `data/trading.db`. Call `init_db()` at the start of each service. Six tables: `account` (cash), `positions` (open, incl. the planned `stop_price`), `trades` (closed, append-only), `signals` (pipeline state machine), `intents` (buy queue), `transactions` (unified BUY+SELL ledger). Schema changes are straightforward — DuckDB supports full `ALTER TABLE`; `init_db()` self-migrates the `stop_price` column via `ADD COLUMN IF NOT EXISTS`.
- **`time_utils.py`** — Injectable backtest clock. `set_backtest_clock(dt)` pins time for simulation; `None` restores live wall-clock. All time-dependent code calls `market_now()` / `market_today()` from here. `is_market_open()` (weekday + 09:30–16:00 ET + TSX holiday set) guards the live services from transacting off-hours and is deterministic under the backtest clock.
- **`market_data.py`** — Two data providers behind a structural interface (`MarketDataProvider`): `LiveDataProvider` (wraps yfinance for live mode) and `HistoricalSliceProvider` (pre-loaded dict with strict `as_of` cutoff to prevent lookahead bias).
- **`backtest_runner.py`** `BacktestConfig.gap_filter_pct` — `None` (default) = no gap filter, matching pre-2026-05 backtest behaviour and the live `GAP_FILTER_PCT` (set to `None` 2026-07 after a fixed-value head-to-head showed the filter reduced returns at every level; was `2.0`). Set to e.g. `2.0` to simulate a gap filter (skip buys where open > intent_entry × 1.02).
- **`backtest_runner.py`** `BacktestConfig.max_per_sector` / `.sector_map` — `None` (default) = no sector cap. Pass a cap (e.g. `2`) and a `{ticker: sector}` dict to simulate `MAX_POSITIONS_PER_SECTOR`; a candidate is skipped once its sector holds the cap's worth of open positions, mirroring `virtual_buy.py`'s live enforcement.
- **`portfolio.py`** — In-memory `PortfolioState` for backtesting only. No file I/O; the live-mode equivalent is `db.py`.
- **`backtest_runner.py`** — Day-by-day simulation. For each day: after-close screener + pipeline, then next-open buys at open price, then daily position monitor.
- **`canadian_stock_screener.py`** — Multi-factor momentum screener (Weinstein Stage II, RS vs XIU.TO, MACD, OBV, ADX, VAM, 52w proximity). Universe and weights controlled by `CONFIG` dict inside the file.
- **`auto_pipeline.py`** — Consumes screener CSV outputs, runs three pattern detectors (VCP, EMA pullback reclaim, base breakout), advances a signal state machine (`FORMING → AT_PIVOT → CONFIRMED → ACTIVE/FAILED`), and writes confirmed intents to the database.
- **`send_report.py`** — Gmail SMTP sender. Has two distinct responsibilities: `send_report()` sends the daily HTML pipeline report (called from `main.py` and `position_monitor.py`); `send_transaction_email()` / `build_transaction_html()` sends a trade notification email after every buy or sell (called from `virtual_buy.py` and `position_monitor.py`). `send_text_email(subject, body)` sends a plain-text email (used by the EDGAR collector). Credentials are read from `.env` (`GMAIL_SENDER`, `GMAIL_APP_PASSWORD`, `GMAIL_RECIPIENT`). Silently skips if not configured.
- **`edgar_service.py`** / **`edgar/`** — Separate 4th service (US SEC EDGAR), the fundamentals/ownership counterweight to the TSX momentum system. Daily after the US close it sweeps EDGAR's daily index, stores filings in its OWN SQLite DB (`data/edgar.db`, keyed on CIK+accession — NOT `trading.db`), and emails a plain-text digest of flagged hits (interim: watchlist insider purchases (Form 4 code 'P' — open-market or private, not distinguished in the data) + all SC 13D/13G, plus new Section 16 filers via Form 3) via `send_report.send_text_email()`. Quiet day = no email. Reuses shared config/email/logging/lock; stays independent of the TSX trio (US filers only). Config in the `EDGAR_*` block of `config.py`; dev sandbox + specs in `edgar/`. Every filing is a lagged disclosure — a research trigger, not a price predictor. `edgar_report.py` is a separate on-demand command that parses already-collected 13D bodies from `scan_hits` and emails a digest.
- **`demand_signals_service.py`** / **`demand_signals/`** — 5th service, added 2026-09: a "real buyer demand" tracker normalizing three US-market sources into one schema (`demand_signals.schema.DemandSignal`: ticker/us_ticker/date/source/signal_type/direction/strength/lag_days/detail, persisted in its OWN SQLite DB `data/demand_signals.db` via `demand_signals/store.py`) so they're screenable together per ticker. `source='edgar_insider'` (`edgar_adapter.py` reads-only from `edgar.db`'s `insider_buys` — never re-fetches, never writes back), `source='finra_darkpool'` (`darkpool.py`: FINRA's free Query API, but OAuth2 client-credentials, not an anonymous GET like SEC's — `FINRA_CLIENT_ID`/`FINRA_CLIENT_SECRET` read from `.env` by `darkpool.py` itself, same self-contained pattern as `send_report.py`'s `GMAIL_*`, since `config.py` doesn't load `.env`; flags a ticker whose weekly ATS-volume ratio has risen `DEMAND_DARKPOOL_RISING_WEEKS` (3) consecutive weeks — WEEKLY data with its own ~2-week publication lag, a confirmation signal only, never a live trigger), `source='options_flow'` (`options_flow.py`: a pluggable `OptionsFlowProvider` ABC, one free implementation `YahooOptionsProvider` via yfinance's option chain — a SNAPSHOT, not a trade tape, no sweep/block detection; a paid vendor plugs in as a second implementation without touching callers). `demand_signals/http_cache.py` generalizes `edgar/core.py`'s rate-limit+cache pattern but is a separate implementation with its own per-source `RateLimiter`/cache-dir instances (`edgar/core.py` itself untouched) — sharing one rate budget across SEC/FINRA/Yahoo would incorrectly couple three unrelated hosts' limits. `demand_signals/ticker_map.py` is a hand-curated CAN→US symbol table for interlisted names (`config.py`'s EDGAR_FORMS/watchlist precedent, not automated inference); a Canadian-only name (no US line) gets no `finra_darkpool`/`options_flow` signal until a SEDI (Canadian insider) source exists — the module's own extension-point comment marks where a 4th `source` value would go. Config in the `DEMAND_*` block of `config.py`. No email digest yet, same "interim" staging EDGAR itself started with — `demand_signals.store.signals_for_ticker()` is the screener-facing read path.
- **Support modules** — `report.py` (console dump of the live trading DB: cash, positions, trades, signals, intents), `report_html.py` (light-theme HTML email report — Gmail forcibly overrides dark backgrounds), `schema_keys.py` (shared column/field keys used across pipeline, buy, monitor, and report modules), `concurrent_utils.py` (fcntl file lock in `out/locks/` preventing overlapping service runs), `log_utils.py` (JSON run logging via `market_now()`).
- **`manual_sell.py`** — on-demand CLI to close one open position at the current market price (live 5-min intraday snapshot, falling back to the last daily close). Takes a single ticker argument; reuses `position_monitor.execute_virtual_sells()` so the position removal, cash credit, trade record, and transaction email match every other sell path. `--dry-run` previews without writing. Core logic lives in `sell_position(ticker, dry_run=False) -> dict` so both the CLI and the web dashboard call the same code.
- **`dashboard_app.py`** / **`dashboard_positions.py`** — Flask + Waitress web dashboard (LAN-only, no auth — deliberate for a home-network Jetson deployment). `dashboard_positions.build_live_positions()` is a read-only, TTL-cached mirror of `position_monitor.py`'s live per-position pipeline (never calls `execute_virtual_sells` itself — a "SELL" status shown here is informational only). Routes: `/` (Monitor), `/history` (closed trades + transaction ledger), `/momentum` (momentum sleeve, read-only), `POST /api/positions/<ticker>/sell` (calls `manual_sell.sell_position()`), `/healthz`. Templates in `templates/`, assets in `static/`. Deployed via `system/stockscanner-dashboard.service` (the first always-on `Type=simple` unit in this repo — the other four are `Type=oneshot` + `.timer`).
- **Momentum sleeve** (`momentum_pipeline.py`, `momentum_buy.py`, `momentum_monitor.py`, `momentum_dashboard_data.py`) — a second, fully isolated paper account (`data/momentum.db`, `config.py` `MOMENTUM_*`), added 2026-08 after diagnosing that the core sleeve's universe builder (hard `atr_pct_14 > 5%` rejection in `swing_tickers.py`) and its three pattern detectors (all require a basing/consolidation structure) structurally cannot buy a vertical sector move (e.g. the 2026-08 gold/silver miner rally). `momentum_pipeline.py` builds its own universe each run via `swing_tickers.run_universe_builder()` with `MOMENTUM_MAX_ATR_PCT` (0.20) against the raw pre-filter ticker source (`MOMENTUM_RAW_TICKERS_URL`, since the published `CAN_TICKERS_URL` is already ATR-filtered upstream), runs with `regime_filter=False` (a broad-index gate would block exactly the sector-specific rallies this sleeve targets), and `enable_momentum_breakout=False` (see below). `momentum_buy.py` is a structural fork of `virtual_buy.py` (own `MOMENTUM_MAX_POSITIONS`/`MOMENTUM_RISK_PER_TRADE_PCT`, own DB) rather than a parameterisation, matching the "services stay independent" precedent already set by EDGAR. `momentum_monitor.py` reuses `position_monitor.compute_signals()`/`execute_virtual_sells()` by import with a wide chandelier `ExitParams` (`MOMENTUM_CHAND_TRAIL_ATR_K=4.0` vs the core's 2.5 — the one exit parameter actually validated by the 2026-08 walk-forward; initial stop distance is untouched, still `PipelineConfig.atr_stop_mult`'s default). `momentum_dashboard_data.py` deliberately never imports `db.py` — it opens its own read-only DuckDB connection to `MOMENTUM_DB_PATH` so the dashboard's single long-running process can serve both sleeves concurrently without racing on `db.py`'s module-global `DB_PATH`; no manual-sell route exists for this sleeve yet for the same reason. `auto_pipeline.py`'s `_detect_momentum_breakout` (a 4th detector, no base-range requirement) exists behind `PipelineConfig.enable_momentum_breakout` but ships **off**: a 16-fold walk-forward (2021-06→2025-06) found it added trades with no statistically significant return edge (p=0.836) and a directionally worse drawdown. The sleeve itself (relaxed universe + wide chandelier trail, 3 existing detectors) was walk-forward validated on the same fold structure as the core sleeve's own 2026-08 risk% test: ~2x the core sleeve's average per-fold return, ~1.5x the average drawdown — a real trade-off, which is why it runs as a separate, walled-off account instead of a core-sleeve config change.

### Scope discipline

This codebase applies strict scope discipline (see `AGENTS.md`):
- Make the minimum change required; do not improve nearby code.
- Preserve all existing comments verbatim unless a code change makes them factually wrong.
- Do not rename, move, or refactor unless explicitly requested.
- Do not convert in-code constants to CLI arguments, env vars, or config files.
- If a change outside the immediate scope seems necessary, stop and explain before making it.

### Deployment (systemd)

`system/` holds `.service`/`.timer` units for the four core services (main, buy, monitor, edgar) plus the three momentum-sleeve services (momentum-pipeline, momentum-buy, momentum-monitor — same schedule as their core counterparts). All timers deliberately use `Persistent=false`: a slot missed while the host was down is skipped, not run late on stale prices. Do not change this to `true` — off-hours catch-up runs previously caused fills at stale quotes; `is_market_open()` is the second layer of the same defence. Install/status commands are in `README.md` ("Systemd deployment") and `system/info`.

### State files (live mode)

- `data/trading.db` — DuckDB database; single source of truth for all live-mode state (cash, positions, trades, signals, intents, transactions). Query directly with `duckdb.connect("data/trading.db")`.
- `data/momentum.db` — same schema (via `db.py`'s `init_db(path=...)`), fully separate momentum-sleeve state. Query with `duckdb.connect("data/momentum.db")`.
- `out/` — generated outputs (screener CSVs, alerts, HTML report, logs, locks); momentum-sleeve equivalents are prefixed `momentum_` (see `config.py` `MOMENTUM_*_PATH`).

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
