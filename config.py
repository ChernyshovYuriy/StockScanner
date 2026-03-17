from enum import Enum
from pathlib import Path

# repo root
ROOT_DIR = Path(__file__).resolve().parent

DATA_PATH = ROOT_DIR / "data"
OUT_PATH = ROOT_DIR / "out"
CACHE_PATH = ROOT_DIR / "cache"

CAN_TICKERS_PATH = DATA_PATH / "can_tickers"
CAN_TICKERS_UNIVERSE_PATH = DATA_PATH / "can_tickers_universe"
CANDIDATES_QUEUE_PATH = DATA_PATH / "candidates_queue.csv"
FUNDS_PATH = DATA_PATH / "funds"
OWN_PATH = DATA_PATH / "own.csv"

CAN_TICKERS_ONE_LINE_PATH = OUT_PATH / "can_tickers_one_line"
CAN_TICKERS_REJECTED_PATH = OUT_PATH / "can_tickers_rejected.csv"
SCREENER_OUT_PATH = OUT_PATH / "screener_out"
REPORT_PATH = OUT_PATH / "report.html"
REPORT_POSITION_PATH = OUT_PATH / "position_monitor_report.html"
ALERTS_PATH = OUT_PATH / "alerts"
LOGS_PATH = Path(OUT_PATH / "logs")
LOCKS_PATH = Path(OUT_PATH / "locks")


class PositionMonitorMode(Enum):
    PRE_CLOSE = 1
    POST_CLOSE = 2
