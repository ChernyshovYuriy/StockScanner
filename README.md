# StockScanner

An automated **paper-trading and market-intelligence system for TSX/TSXV stocks**,
plus one tool for a real brokerage account. It runs as eleven independent
scheduled services (screeners, paper-trading sleeves, and news/filing
collectors) that all write to their own database and feed one Flask
dashboard. Nothing here places a real trade except `conviction_watchlist/`,
which only records trades the user made by hand elsewhere.

> Data source: Yahoo Finance (`yfinance`), FRED, SEC EDGAR, FINRA, and
> GlobeNewswire RSS. All sleeves below `virtual_buy.py`/`momentum_buy.py`/
> `macro_buy.py`/`kangaroo_buy.py` are **virtual** — no brokerage connection.

For full architecture detail, config keys, and the research/backtests behind
each design choice, see [`CLAUDE.md`](CLAUDE.md).

---

## What's in here

**Paper-trading sleeves** (each has its own cash, positions, and exit rules —
independent virtual accounts, never merged):

| Sleeve | Entry mechanic | DB |
|---|---|---|
| **Core** (`main.py` / `virtual_buy.py` / `position_monitor.py`) | Weinstein Stage II + RS + MACD + OBV + ADX screener → VCP / EMA-pullback / base-breakout pattern detection | `data/trading.db` |
| **Momentum** (`momentum_*.py`) | Same detectors, relaxed ATR universe, wide trailing stop — catches vertical moves the core sleeve's basing requirement rejects | `data/momentum.db` |
| **Macro** (`macro_buy.py` / `macro_monitor.py`) | FRED yield-curve/credit-spread/Fed-balance-sheet regime gate; concentrated bets on the *core* sleeve's own candidates when risk-on | `data/macro.db` |
| **Kangaroo Tail** (`kangaroo_*.py`) | Breakout-entry (stop = tail low, target = R-multiple) off a bullish reversal candle — the one entry mechanic that backtested with a real edge | `data/kangaroo.db` |

**Research trackers** (no capital, no positions — track an idea, not a trade):

| Service | Question it answers | DB |
|---|---|---|
| **Triple Screen tracker** (`triple_screen_tracker_service.py`) | After an Elder Triple Screen BUY signal, how does price behave until it first closes below entry? | `data/triple_screen_tracker.db` |
| **Ticker Indicator Board** (`scanner_pipeline.py`) | One row per ticker, one column per Elder indicator — a read-only screen, deliberately no composite score | `data/scanner_board.db` |
| **Volume spike scanner** (`volume_spike_scanner.py`) | Which tickers are trading above their own 20-day average volume, on rising price — on-demand only, no schedule | *(none — live scan)* |

**Collectors** (external data, normalized and screenable):

| Service | Source | DB |
|---|---|---|
| **EDGAR collector** (`edgar_service.py`) | SEC insider buys (Form 4) + 13D/13G ownership filings | `data/edgar.db` |
| **Demand signals** (`demand_signals_service.py`) | EDGAR insider buys + FINRA dark-pool + FINRA short-volume + options flow, normalized into one schema | `data/demand_signals.db` |
| **Press-release tracker** (`press_release_service.py`) | GlobeNewswire RSS, LLM-classified (ticker/category/materiality), emailed within minutes of publication | `data/press_releases.db` |
| **News watchlist** (`news_watchlist_service.py`) | Human-curated follow-through tracker on top of the press-release tracker's own catches (inbox → watching → dismissed) | `data/news_watchlist.db` |

**Real account** (`conviction_watchlist/`): a quality filter + 52-week-dip
entry screen + trailing-stop sell flag for the user's own RBC account.
Nothing here executes a trade — holdings are entered by hand after the user
actually trades. State: `data/conviction_*.json`.

**Dashboard** (`dashboard_app.py`, Flask, LAN-only, no auth):

| Route | Sleeve/service | Read/write |
|---|---|---|
| `/` , `/history` | Core | read-only + manual sell button |
| `/momentum` | Momentum sleeve | read-only |
| `/macro` | Macro sleeve | read-only |
| `/kangaroo` | Kangaroo Tail sleeve | read-only |
| `/triple-screen` | Triple Screen tracker | read-only |
| `/scanner` | Ticker Indicator Board | read-only |
| `/volume-spikes` | Volume spike scanner | read-only (live scan) |
| `/demand` | Demand signals | read-only |
| `/news-watchlist` | News watchlist | read/write (confirm/dismiss/note) |
| `/conviction` | Real account | read/write (edit holdings, refresh candidates) |

---

## Installation

**Requirements:** Python 3.10+, internet access.

```bash
git clone <repo>
cd StockScanner
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## First-time setup

Initialize the core sleeve's database and starting capital (run once):

```bash
python -c "from db import init_db, set_cash; init_db(); set_cash(50_000)"
```

Every other sleeve seeds its own capital on its first run
(`MOMENTUM_INITIAL_CAPITAL` / `MACRO_INITIAL_CAPITAL` / `KANGAROO_INITIAL_CAPITAL`
in `config.py`) — nothing else to run by hand. The research trackers and
collectors create their databases on first run with no capital step at all.

Run `main.py` once (e.g. over a weekend) before the first live trading day so
`virtual_buy.py` has something to buy Monday morning:

```bash
python main.py
```

### Optional `.env` keys

Nothing below is required — every integration degrades gracefully (skips its
own step, keeps running) if its key is absent.

| Key | Used by | Effect if unset |
|---|---|---|
| `GMAIL_SENDER` / `GMAIL_APP_PASSWORD` / `GMAIL_RECIPIENT` | trade + digest emails across every service | emails are skipped, everything else runs |
| `FINRA_CLIENT_ID` / `FINRA_CLIENT_SECRET` | `demand_signals/darkpool.py` | dark-pool signal skipped; insider-buy and options-flow signals unaffected |
| `FRED_API_KEY` | `macro_regime.py` | every regime vote reads 0 → macro sleeve stays in cash |
| `OPENAI_API_KEY` | `press_release_tracker/llm_parser.py` | releases still emailed, just unclassified (raw RSS title/link only) |

Gmail requires an **App Password**, not your normal password — enable
2-Step Verification at <https://myaccount.google.com/security>, then create one.

---

## Daily operation

The core sleeve's three services, runnable manually or via systemd:

```bash
python main.py                                    # 4:30 PM — screen, detect, queue candidates
python virtual_buy.py                              # 9:45 AM — size and execute queued buys
python position_monitor.py --mode pre-close        # 3:50 PM — evaluate exits, execute sells
python position_monitor.py --mode post-close       # informational only, no sells
```

`virtual_buy.py` and `position_monitor.py` both support `--dry-run` (preview,
no DB writes). All live services no-op outside TSX trading hours
(`is_market_open()`), so an off-schedule manual run is always safe.

Every other service's manual-run command is listed in its own `--help`, and
summarized in `CLAUDE.md`'s Commands section.

---

## Systemd deployment (Linux)

`system/` holds one `.service`/`.timer` pair per scheduled job (33 units
covering all eleven services) plus one always-on `.service` for the
dashboard. `system/info` has the full install/enable/status/journalctl
command list; the short version:

```bash
sudo cp system/*.service system/*.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now stockscanner-main.timer stockscanner-buy.timer stockscanner-monitor.timer
sudo systemctl enable --now stockscanner-dashboard.service    # always-on, not timer-driven
```

Repeat `enable --now` for whichever other sleeves/collectors you want
running (`stockscanner-momentum-*`, `stockscanner-macro-*`,
`stockscanner-kangaroo-*`, `stockscanner-edgar.timer`,
`stockscanner-demand-signals.timer`, `stockscanner-triple-screen-tracker.timer`,
`stockscanner-scanner-pipeline.timer`, `stockscanner-press-release.timer`,
`stockscanner-news-watchlist.timer`, `stockscanner-news-watchlist-seed.timer`).

The dashboard listens on `DASHBOARD_HOST:DASHBOARD_PORT` from `config.py`
(default `0.0.0.0:8080`, LAN-only, no auth — deliberate for a home-network
deployment).

> Timers use `Persistent=false` — a slot missed while the host is down is
> skipped, not run late on stale prices. Don't change this; it's a
> deliberate second layer of defence alongside `is_market_open()`.

---

## Querying a database

Every sleeve's DB is a plain DuckDB (core/momentum/macro/kangaroo) or SQLite
(everything else) file under `data/` — open it directly:

```python
import duckdb
conn = duckdb.connect("data/trading.db")   # or momentum.db / macro.db / kangaroo.db
conn.execute("SELECT * FROM positions").df()
conn.execute("SELECT * FROM trades ORDER BY sell_date").df()
conn.execute("SELECT * FROM intents WHERE intent_status = 'PENDING'").df()
conn.close()
```

Core/momentum/macro/kangaroo share one schema: `account` (cash), `positions`
(open), `trades` (closed, append-only), `transactions` (unified BUY+SELL
ledger), `signals` (pipeline state machine), `intents` (buy queue). The
collectors and research trackers each have their own unrelated schema — see
`CLAUDE.md`'s "State files" section.

---

## Screener (core sleeve)

`canadian_stock_screener.py` scores each ticker in the universe on Weinstein
Stage II alignment, RS vs XIU.TO, MACD, OBV slope, ADX, volatility-adjusted
momentum, and 52-week proximity. Universe: `CAN_TICKERS_URL` (`config.py`).
Output: `out/screener_out/YYYYMMDD_HHMM.csv`. Weights/thresholds: the
`CONFIG` dict inside the file.

`auto_pipeline.py` then runs three pattern detectors (VCP, EMA pullback
reclaim, base breakout) and advances a `FORMING → AT_PIVOT → CONFIRMED →
ACTIVE/FAILED` state machine; `CONFIRMED` setups become `PENDING` intents for
`virtual_buy.py`. Before generating new signals, `main.py` checks XIU.TO
against its 200-day SMA — in a bear regime, no new signals, existing
positions still monitored.

### Exit rules (`position_monitor.py`)

| Rule | Trigger |
|---|---|
| Initial stop | Persisted `stop_price` from the buy intent, or `Entry − 1.5×ATR(14)` if none was stored |
| Chandelier trail | Highest high since entry − 2.5×ATR(14) |
| Profit giveback | Peak profit ≥ 6%, current profit ≥ 3pts below that peak |
| Time stop | ≥ 20 trading days held, profit < 0% |

---

## Backtesting (core sleeve only)

```bash
python run_backtest.py --start 2022-01-01 --end 2024-01-01
python run_backtest.py --start 2022-01-01 --end 2024-01-01 --sweep              # exit-param grid
python run_backtest.py --start 2022-01-01 --end 2025-01-01 --walk-forward-gap    # gap-filter optimization
python run_backtest.py --help
```

Outputs land in `out/`: an HTML report with equity curve, a trade-log CSV,
and a day-by-day equity CSV.

---

## Running tests

```bash
pytest tests/ -v

# By phase gate (backtest refactor)
pytest -v -m phase1   # clock injection
pytest -v -m phase2   # MarketDataProvider
pytest -v -m phase3   # PortfolioState
pytest -v -m phase4   # BacktestRunner
pytest -v -m phase5   # HTML report
pytest -v -m phase6   # CLI entry point
pytest -v -m characterization  # golden-value business logic locks
```

---

## Directory layout

```
.
├── main.py / virtual_buy.py / position_monitor.py   # core sleeve
├── momentum_pipeline.py / momentum_buy.py / momentum_monitor.py
├── macro_regime.py / macro_buy.py / macro_monitor.py
├── kangaroo_pipeline.py / kangaroo_buy.py / kangaroo_monitor.py
├── edgar_service.py            edgar/
├── demand_signals_service.py   demand_signals/
├── triple_screen_tracker_service.py   triple_screen_tracker/   research/triple_screen/
├── scanner_pipeline.py         scanner_board/
├── volume_spike_scanner.py
├── press_release_service.py    press_release_tracker/
├── news_watchlist_service.py   news_watchlist/
├── conviction_watchlist/       # real-account tool, no scheduled service
├── auto_pipeline.py            # core sleeve pattern detection + state machine
├── canadian_stock_screener.py
├── run_backtest.py             db.py             # backtest CLI / DuckDB persistence
├── market_data.py               # sole yfinance access point (LiveDataProvider/HistoricalSliceProvider)
├── send_report.py               config.py         # email sender / all path+param constants
├── dashboard_app.py             templates/        static/
├── data/                        # all *.db + conviction *.json (gitignored)
├── out/                         # screener CSVs, alerts, logs, locks
├── system/                      # systemd .service/.timer units + system/info
└── tests/
```

---

## Notes / limitations

- `yfinance` hits live Yahoo Finance endpoints — intermittent rate limits or
  missing data can occur.
- The core screener and its benchmark (`XIU.TO`) are TSX-specific.
- Every sleeve above is virtual paper trading. `conviction_watchlist/` never
  places trades either — it only records what the user already did in their
  real account.
- For research and education only. Not financial advice. Trading involves
  risk of loss.

---

## License

MIT
