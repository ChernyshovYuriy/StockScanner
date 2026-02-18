from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent  # repo root
DATA_PATH = ROOT_DIR / "data"
CAN_TICKERS_PATH = DATA_PATH / "can_tickers"
SCREENER_OUTPUT = "screener_outputs"
