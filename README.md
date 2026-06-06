# TSX Swing Trading System

A fully automated virtual swing-trading system for **Canadian (TSX) stocks**.
Screens the market daily, detects technical entry patterns, executes virtual
buys and sells, and emails a report after every trade.

> Data source: Yahoo Finance via `yfinance`. All transactions are virtual —
> no real brokerage connection.

---

## How it works

Three scheduled services cooperate across the trading day:

| Time (ET) | Service | What it does |
|---|---|---|
| **4:30 PM** | `main.py` | Checks market regime → rebuilds universe → runs screener → detects patterns → queues buy candidates |
| **9:45 AM** | `virtual_buy.py` | Reads the candidate queue, fetches live prices, sizes and executes virtual buys → sends trade email |
| **3:50 PM** | `position_monitor.py` | Evaluates open positions against exit rules using intraday data → executes virtual sells → sends trade email |

> The live services no-op outside TSX trading hours (weekday, 09:30–16:00 ET,
> non-holiday) via `is_market_open()`, so an off-schedule trigger never
> transacts on stale prices.

All state — cash, positions, trades, signals, intents — lives in a single
**DuckDB database** at `data/trading.db`.

---

## Installation

**Requirements:** Python 3.10+, internet access (Yahoo Finance).

```bash
git clone <repo>
cd StockScanner

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## First-time setup

### 1. Set your starting capital

Run once before anything else. Replace `50000` with your paper-trading capital:

```bash
python -c "from db import init_db, set_cash; init_db(); set_cash(50_000)"
```

Verify the database was created and the balance is correct:

```bash
python -c "
from db import init_db, get_cash, get_open_positions
init_db()
print('Cash     :', get_cash())
print('Positions:', get_open_positions())
"
```

### 2. Configure email (optional but recommended)

The system sends a trade notification email after every buy or sell.
Gmail requires an **App Password** (not your regular password).

1. Enable 2-Step Verification at <https://myaccount.google.com/security>
2. Create an App Password → copy the 16-character code
3. Create a `.env` file in the repo root:

```
GMAIL_SENDER=you@gmail.com
GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx
GMAIL_RECIPIENT=you@gmail.com
```

If `.env` is absent the system runs normally — it just skips emails.

### 3. Run `main.py` before the first trading day

Run this on the weekend before you want to start:

```bash
python main.py
```

This populates `data/trading.db` with the first batch of signals and queues
any `CONFIRMED` setups as pending buy intents. `virtual_buy.py` will consume
them Monday morning.

---

## Daily operation

Once the system is running, the three services execute automatically on their
schedule. To run them manually:

```bash
# End-of-day (after 4:30 PM ET)
python main.py

# Next morning (at/after 9:45 AM ET open)
python virtual_buy.py

# Pre-close (around 3:50 PM ET, market still open)
python position_monitor.py --mode pre-close

# Optional post-close informational run
python position_monitor.py --mode post-close
```

### Dry-run mode

Both `virtual_buy.py` and `position_monitor.py` support `--dry-run` to
preview activity without touching the database:

```bash
python virtual_buy.py --dry-run
python position_monitor.py --mode pre-close --dry-run  # not yet wired — informational only
```

---

## Systemd deployment (Linux)

The `system/` directory contains `.service` and `.timer` unit files for
running the three services automatically on a Linux host.

### First-time install

```bash
# Copy unit files to systemd
sudo cp system/*.service system/*.timer /etc/systemd/system/

# Reload systemd so it sees the new units
sudo systemctl daemon-reload

# Enable timers so they survive reboots
sudo systemctl enable stockscanner-main.timer
sudo systemctl enable stockscanner-buy.timer
sudo systemctl enable stockscanner-monitor.timer

# Start the timers now
sudo systemctl start stockscanner-main.timer
sudo systemctl start stockscanner-buy.timer
sudo systemctl start stockscanner-monitor.timer
```

### After editing a .service or .timer file

```bash
# Copy updated files
sudo cp system/stockscanner-main.service /etc/systemd/system/
# (repeat for whichever files changed)

# Tell systemd to reload its configuration
sudo systemctl daemon-reload

# Restart the affected service if it is currently running
sudo systemctl restart stockscanner-main.service
```

### Useful status commands

```bash
# Check whether timers are active and when they next fire
systemctl list-timers stockscanner-*

# View recent output for a service
journalctl -u stockscanner-main.service -n 50

# Check the current status of a service
systemctl status stockscanner-main.service

# Run a service immediately (outside its schedule)
sudo systemctl start stockscanner-main.service
```

> The timers use `Persistent=false`, so a slot missed while the host is down is
> skipped rather than run late on stale prices. Combined with the
> `is_market_open()` guard, starting a `*.service` manually outside trading
> hours is a safe no-op for live buys/sells (a report may still be generated).

### Tickers URL

The main service is started with `--tickers-url` pointing to a remote file
(one ticker per line). To change the URL, edit
`system/stockscanner-main.service` and re-run the install steps above.
The default URL is also set in `config.py` (`CAN_TICKERS_URL`) and is used
when running services manually from the command line without `--tickers-url`.

---

## Querying the database

Use DuckDB directly for ad-hoc queries:

```python
import duckdb
conn = duckdb.connect("data/trading.db")

conn.execute("SELECT * FROM transactions ORDER BY trade_date").df()     # full ledger
conn.execute("SELECT * FROM positions").df()                             # open positions
conn.execute("SELECT * FROM trades ORDER BY sell_date").df()             # closed trades
conn.execute("SELECT * FROM account").df()                               # cash balance
conn.execute("SELECT * FROM intents WHERE intent_status = 'PENDING'").df()

conn.close()
```

### Database tables

| Table | Contents |
|---|---|
| `account` | Current cash balance |
| `positions` | Open virtual positions (ticker, entry date, price, shares, planned stop) |
| `trades` | Permanent closed-trade log with full P&L |
| `transactions` | Unified ledger — every BUY and SELL in chronological order |
| `signals` | Pipeline signal state machine (rebuilt each run) |
| `intents` | Buy candidate queue with execution history |

---

## Screener

`canadian_stock_screener.py` scores each ticker in the TSX universe with a
weighted factor stack:

- **Weinstein Stage II alignment**
- **Relative Strength vs XIU.TO**
- **MACD momentum**
- **OBV slope** (volume accumulation)
- **ADX trend strength**
- **Volatility-adjusted momentum (VAM)**
- **52-week high proximity / breakout pressure**

**Universe:** loaded from `CAN_TICKERS_URL` in `config.py` (one ticker per line).

**Output:** `out/screener_out/YYYYMMDD_HHMM.csv`

**Configure** via `CONFIG` dict inside `canadian_stock_screener.py`:
`top_n`, `min_price`, `min_avg_volume`, `weights`, `lookback_days`.

---

## Entry pipeline

`auto_pipeline.py` reads screener CSVs, tracks tickers across days, and runs
three pattern detectors:

1. **VCP** — Volatility Contraction Pattern
2. **EMA pullback reclaim** — EMA21 and EMA50 variants
3. **Base breakout** — tight range + volume confirmation

### Signal state machine

```
FORMING  →  AT_PIVOT  →  CONFIRMED  →  ACTIVE
                     ↘              ↘  FAILED / EXPIRED
```

`CONFIRMED` setups are written to the `intents` table as `PENDING` buy
candidates. `virtual_buy.py` processes them the following morning.

### Market regime filter

Before generating new signals, `main.py` checks whether **XIU.TO** is above
its 200-day SMA. In a bear regime, signal generation is skipped — no new
positions are opened. Existing positions continue to be monitored.

---

## Exit rules (`position_monitor.py`)

Each position is evaluated against four exit rules (defaults):

| Rule | Trigger |
|---|---|
| Initial stop | Entry − 1.5 × ATR(14) |
| Chandelier trail | Highest high since entry − 2.5 × ATR(14) |
| Profit giveback | Max profit ≥ 6% and current profit drops ≥ 3% below peak |
| Time stop | ≥ 20 trading days held with profit < 0% |

When a position carries a `stop_price` persisted from its buy intent (the
swing-low-aware planned stop), that level is used as the initial stop instead
of the `Entry − 1.5 × ATR(14)` formula, so the exit matches the stop the trade
was sized against. Legacy positions without a stored stop fall back to the ATR
formula.

---

## Backtesting

Run a historical backtest over any date range:

```bash
# Single run
python run_backtest.py --start 2022-01-01 --end 2024-01-01

# Custom tickers (URL or file)
python run_backtest.py --tickers https://example.com/tickers.txt --start 2022-01-01 --end 2024-01-01

# Exit-param sweep: time_stop_days × stop_atr (4×4=16 combinations)
python run_backtest.py --start 2022-01-01 --end 2024-01-01 --sweep

# Walk-forward gap filter optimization (find best GAP_FILTER_PCT via sliding windows)
python run_backtest.py --start 2022-01-01 --end 2025-01-01 --walk-forward-gap
python run_backtest.py --start 2022-01-01 --end 2025-01-01 --walk-forward-gap --wf-in-days 84 --wf-out-days 21

python run_backtest.py --help
```

The backtest simulates the live gap filter when `gap_filter_pct` is set in `BacktestConfig`
(default `None` = no filter, matching historical behaviour before May 2026).

Output files are written to `out/`:
- `backtest_DATES_TIMESTAMP.html` — HTML report with equity curve and trade log
- `backtest_trades_TIMESTAMP.csv` — full trade log
- `backtest_equity_TIMESTAMP.csv` — day-by-day equity curve
- `backtest_wf_gap_DATES_TIMESTAMP.csv` — walk-forward gap optimization results

---

## Directory layout

```
.
├── main.py                  # end-of-day service (4:30 PM)
├── virtual_buy.py           # morning buy execution (9:45 AM)
├── position_monitor.py      # pre/post-close monitor (3:50 PM)
├── auto_pipeline.py         # pattern detection + signal state machine
├── canadian_stock_screener.py
├── run_backtest.py          # backtest CLI
├── db.py                    # DuckDB persistence layer
├── send_report.py           # Gmail email sender
├── config.py                # path constants + trading parameters
├── data/
│   └── trading.db           # all live state (auto-created on first run)
└── out/
    ├── screener_out/        # daily screener CSVs
    ├── alerts/              # daily alert CSVs + HTML report
    ├── logs/                # service run logs
    └── locks/               # process lock files
```

---

## Running tests

```bash
pytest tests/ -v

# By phase gate (backtest refactor)
pytest -v -m phase1    # clock injection
pytest -v -m phase2    # MarketDataProvider
pytest -v -m phase3    # PortfolioState
pytest -v -m phase4    # BacktestRunner
pytest -v -m phase5    # HTML report
pytest -v -m phase6    # CLI entry point

# Specific files
pytest tests/test_db.py -v           # database layer
pytest tests/test_integration.py -v  # service integration
```

---

## Notes / limitations

- `yfinance` depends on Yahoo Finance endpoints — intermittent rate limits
  or missing data can occur.
- The screener and benchmark (`XIU.TO`) are designed for **TSX tickers**.
- The pipeline caps tracked tickers (default 40) to limit API calls.
- All transactions are virtual — this system does **not** connect to any
  brokerage.

---

## Disclaimer

For research and education only. Does **not** constitute financial advice.
Trading involves risk of loss.

---

## License

MIT
