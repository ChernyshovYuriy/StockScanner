from enum import Enum
from pathlib import Path

# repo root
ROOT_DIR = Path(__file__).resolve().parent

DATA_PATH = ROOT_DIR / "data"
OUT_PATH = ROOT_DIR / "out"
CACHE_PATH = ROOT_DIR / "cache"

# URL for the ticker list (one ticker per line); used by all services.
CAN_TICKERS_URL = "https://raw.githubusercontent.com/ChernyshovYuriy/Financing/refs/heads/main/data/can_tickers_swing_universe"
SCREENER_OUT_PATH = OUT_PATH / "screener_out"
REPORT_PATH = OUT_PATH / "report.html"
REPORT_POSITION_PATH = OUT_PATH / "position_monitor_report.html"
ALERTS_PATH = OUT_PATH / "alerts"
LOGS_PATH = Path(OUT_PATH / "logs")
LOCKS_PATH = Path(OUT_PATH / "locks")

# Maximum number of positions the portfolio can hold simultaneously.
MAX_POSITIONS = 8

# Fraction of total funds risked on each trade (used by virtual_buy.py).
# Shares = (funds * RISK_PER_TRADE_PCT/100) / (entry_price - stop_price)
# Position value is additionally capped at funds / MAX_POSITIONS.
RISK_PER_TRADE_PCT = 1.0

# Maximum % a stock's open price may exceed the planned entry before the buy
# is skipped. Protects against gap-ups that destroy the signal's R:R.
GAP_FILTER_PCT = 2.0


class PositionMonitorMode(Enum):
    PRE_CLOSE = 1
    POST_CLOSE = 2


# ─────────────────────────────────────────────────────────────────────────────
# EDGAR collector (separate 4th service — see edgar_service.py)
# ─────────────────────────────────────────────────────────────────────────────
# The EDGAR collector is the fundamentals/ownership counterweight to this TSX
# momentum system: it surfaces *footprints* of big money (insider open-market
# buys, activist stakes) from SEC filings. Every filing is a lagged disclosure —
# a research trigger, never a price predictor or financial advice.

# SEC fair-access requires a real contact in the User-Agent or requests are 403'd.
EDGAR_USER_AGENT = "StockScanner-EDGAR/0.1 (chernyshov.yuriy@gmail.com)"

# EDGAR keeps its OWN SQLite store (US filers, keyed on CIK + accession), kept
# separate from the DuckDB trading.db so the two domains stay independent.
EDGAR_DB_PATH = DATA_PATH / "edgar.db"

# On-disk cache for SEC JSON/idx payloads (ticker map, companyfacts, daily index).
EDGAR_CACHE_PATH = CACHE_PATH / "edgar"

# Minimum open-market insider buy ($ = shares × price) to flag in the digest.
EDGAR_MIN_BUY_VALUE = 250_000

# SEC forms the daily-index event loop collects.
EDGAR_FORMS = ("4", "SC 13D", "SC 13D/A", "SC 13G", "SC 13G/A", "8-K")

# Each run re-scans this many business days (accession-deduped) to self-heal
# after downtime/holidays.
EDGAR_BACKFILL_DAYS = 5
