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
