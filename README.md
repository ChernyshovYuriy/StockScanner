# TSX Canadian Stock Screener + Auto Entry Pipeline

Python tools for **screening Canadian (TSX) stocks** and running a **daily swing-trading “entry pipeline”** that tracks candidates across days and emits actionable alerts when setups move through a signal state machine.

This repo contains two primary scripts:

- **`canadian_stock_screener.py`** — multi-factor momentum screener that ranks a predefined TSX universe and saves a *top-N* CSV.
- **`auto_pipeline.py`** — consumes daily screener outputs, tracks tickers over time, detects technical entry patterns, and produces alerts + a persistent signal database.

> Data source: Yahoo Finance via `yfinance`.

---

## Repo layout

```
.
├── canadian_stock_screener.py
├── auto_pipeline.py
├── config.py
├── requirements.txt
└── data/
    └── can_tickers        # TSX universe list (one ticker per line)
```

---

## 1) Canadian stock screener

### What it does
`canadian_stock_screener.py` downloads ~2 years of daily data and scores each ticker with a weighted stack (see `CONFIG["weights"]` in the script):

- **Weinstein Stage II alignment**
- **Relative Strength vs benchmark** (`XIU.TO`)
- **MACD momentum**
- **OBV slope** (volume accumulation)
- **ADX trend strength**
- **Volatility-adjusted momentum (VAM)**
- **52-week high proximity / breakout pressure**

It also applies basic filters such as **minimum price** and **minimum average volume**.

### Output
- Prints a table to the console.
- Saves the top picks to:
  - `screener_outputs/YYYYMMDD_HHMM.csv`

### Run
```bash
python canadian_stock_screener.py
```

### Configure
Edit `CONFIG` inside `canadian_stock_screener.py`:
- `top_n` (default 10)
- `min_price` (default 2.0 CAD)
- `min_avg_volume` (default 100,000)
- `weights` (factor weighting)
- `lookback_days` (default 504 trading days)

Universe list:
- `data/can_tickers`

---

## 2) Automated entry pipeline

### What it does
`auto_pipeline.py` reads daily screener outputs, tracks tickers across days, detects entry patterns, and alerts on signal state transitions.

**Accepted inputs (flexible):**
- Any CSV with a **`Ticker`** or **`symbol`** column
- Example names (from the script header):
  - `top10_tsx_20260218_1430.csv`
  - `my_tickers_2026-02-18.csv`
- JSON is also supported for some upstream tools (see `auto_pipeline.py` header).

### Patterns detected
The pipeline currently runs **three detectors**:

1. **VCP (Volatility Contraction Pattern)**
2. **EMA pullback reclaim** (EMA21 and EMA50 variants)
3. **Base breakout** (tight range + volume confirmation)

### Signal state machine
```
FORMING  →  AT_PIVOT  →  CONFIRMED  →  ACTIVE / FAILED
```

### Alert priority
- 🔴 **URGENT** — state jumped to **CONFIRMED** today (often “enter next session open” style)
- 🟡 **WATCH** — state reached **AT_PIVOT** today (often “place buy-stop” / watch volume)
- 🟢 **FORMING** — setup building, check again
- ⚫ **EXPIRED** — invalidated/expired

### Directory layout (auto-created)
On first run, the pipeline creates:

```
<base_dir>/
  screener_outputs/      # drop daily top-N CSVs here
  signal_db/
    signal_history.csv   # persistent signal state across days (auto-managed)
  alerts/
    alerts_YYYYMMDD.csv  # actionable alerts for each run
    report_YYYYMMDD.txt  # human-readable daily report
```

### Run (single run)
```bash
python auto_pipeline.py
```

### Run (daily scheduler)
Runs once per day at market close time (default **16:05 ET**):

```bash
python auto_pipeline.py --schedule
```

### Debug a single ticker
```bash
python auto_pipeline.py --ticker CNQ.TO
```

### Key CLI options
```bash
python auto_pipeline.py --base-dir .
python auto_pipeline.py --account 100000
python auto_pipeline.py --risk 1.0
python auto_pipeline.py --min-days 1
python auto_pipeline.py --max-tickers 40
python auto_pipeline.py --schedule --schedule-time 16:05
```

---

## Installation

### Requirements
- Python **3.10+** recommended
- Internet access (Yahoo Finance via `yfinance`)

### Install deps
```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Notes / limitations

- `yfinance` depends on Yahoo endpoints; intermittent rate limits or missing data can happen.
- The screener is designed around **TSX tickers** and uses **`XIU.TO`** as the benchmark.
- The pipeline caps tracked tickers (`--max-tickers`, default 40) to avoid excessive API calls.

---

## Disclaimer

This repository is for research/education and does **not** constitute financial advice. Trading involves risk.

---

## License

- MIT

