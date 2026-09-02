import os
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
# None disables the filter — fixed-value head-to-head backtest (2022→2026-07,
# live parity) showed the filter reduced returns at every tested level.
GAP_FILTER_PCT = None

# Maximum number of concurrently open positions in the same GICS sector
# (see sector_lookup.py). None disables the cap. 16-fold walk-forward
# (2021-06→2025-06, live-parity backtest, see backtest_runner.py
# max_per_sector/sector_map): no return cost (p=0.995) but a statistically
# significant reduction in max drawdown (14/16 folds, p=0.003) — prevents
# correlated same-sector clusters (e.g. Canadian bank earnings week) from
# entering and exiting together, which is what actually hurt the live
# account in 2026-08.
MAX_POSITIONS_PER_SECTOR = 2


class PositionMonitorMode(Enum):
    PRE_CLOSE = 1
    POST_CLOSE = 2


# ─────────────────────────────────────────────────────────────────────────────
# Momentum sleeve (separate, isolated live experiment — see momentum_*.py)
# ─────────────────────────────────────────────────────────────────────────────
# The core sleeve above is structurally built for defined-risk continuation
# trades: swing_tickers.py's universe builder hard-rejects atr_pct_14 > 5%, and
# every pattern detector requires a basing/consolidation structure. That's a
# deliberate design, not a bug — but it means the core sleeve can never catch a
# vertical sector move (e.g. the 2026-08 gold/silver miner rally: EDR.TO was
# rejected from the universe with atr_pct_14=0.0657 > 0.05). This block is a
# fully separate paper account — own DB, own capital, own detector, own wide
# stops — to test whether a genuinely different, higher-volatility-tolerant
# strategy can capture what the core sleeve is designed to avoid. Backtest +
# walk-forward validated before any live wiring (see CLAUDE.md).
MOMENTUM_DB_PATH = DATA_PATH / "momentum.db"
MOMENTUM_INITIAL_CAPITAL = 10_000.0

# Smaller, more concentrated book than the core sleeve's MAX_POSITIONS=8 —
# matches the smaller capital base.
MOMENTUM_MAX_POSITIONS = 5

# Backtest-validated value for this sleeve's smaller book (see below) — not
# copied from the core sleeve's RISK_PER_TRADE_PCT=1.0.
MOMENTUM_RISK_PER_TRADE_PCT = 2.0

# Universe ATR ceiling for this sleeve's own swing_tickers.py run — vs the core
# sleeve's hard 0.05 (5%) ceiling. This is the actual unblock: without raising
# it, EDR.TO-style vertical movers never enter the universe at all, regardless
# of detector logic. Other swing_tickers.py gates (liquidity, above_50d,
# staleness) are kept.
MOMENTUM_MAX_ATR_PCT = 0.20

# Wide chandelier trail only — accepts more give-back than the core sleeve's
# CHAND_TRAIL_ATR_K=2.5 in exchange for room to let a vertical move develop
# instead of being stopped out by normal noise on a high-ATR name. This is
# the one exit parameter the 2026-08 walk-forward actually varied (~2x avg
# fold return vs the core sleeve, ~1.5x avg drawdown) — the *initial* stop
# distance was left at the same 1.5x-ATR (PipelineConfig.atr_stop_mult
# default) as the core sleeve in that test, so it stays untouched here too;
# widening it further is a follow-up experiment, not something to deploy
# unvalidated. CHAND_ARM_PCT is kept at the core sleeve's backtest-validated
# value (see position_monitor.py) — "don't trail too early" applies here too.
MOMENTUM_CHAND_TRAIL_ATR_K = 4.0
MOMENTUM_CHAND_ARM_PCT = 8.0

# Raw, pre-filter TSX/TSXV/CSE ticker list (same GitHub repo that publishes
# CAN_TICKERS_URL, which is *already* ATR-filtered upstream and so cannot be
# reused here — see the diagnosis above). momentum_pipeline.py drops the .NE
# NEO-exchange interlisting duplicates (same underlying security as the
# .TO/.V/.CN listing) before running swing_tickers.run_universe_builder()
# against it with the relaxed MOMENTUM_MAX_ATR_PCT ceiling.
MOMENTUM_RAW_TICKERS_URL = "https://raw.githubusercontent.com/ChernyshovYuriy/Financing/refs/heads/main/data/can_tickers_full"

# Output paths — kept fully separate from the core sleeve's out/ files.
MOMENTUM_UNIVERSE_OUT_PATH = OUT_PATH / "can_tickers_momentum"
MOMENTUM_SCREENER_OUT_PATH = OUT_PATH / "momentum_screener_out"
MOMENTUM_REPORT_PATH = OUT_PATH / "momentum_report.html"
MOMENTUM_REPORT_POSITION_PATH = OUT_PATH / "momentum_position_monitor_report.html"
MOMENTUM_ALERTS_PATH = OUT_PATH / "momentum_alerts"


# ─────────────────────────────────────────────────────────────────────────────
# Web dashboard (Jetson, LAN-only, no auth — deliberate choice)
# ─────────────────────────────────────────────────────────────────────────────
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 8080
DASHBOARD_SNAPSHOT_CACHE_TTL_SECONDS = 15  # server-side cache for build_live_positions()


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

# Minimum insider purchase ($ = shares × price; Form 4 code 'P' -- open-market
# or private, the two aren't distinguished in the data) to flag in the digest.
EDGAR_MIN_BUY_VALUE = 250_000

# SEC forms the daily-index event loop collects. The daily index labels the
# Schedule 13D/G forms as "SCHEDULE 13D" (the submissions API uses "SC 13D"),
# so both spellings are accepted defensively.
EDGAR_FORMS = (
    "3", "4",
    "SC 13D", "SC 13D/A", "SC 13G", "SC 13G/A",
    "SCHEDULE 13D", "SCHEDULE 13D/A", "SCHEDULE 13G", "SCHEDULE 13G/A",
    "8-K",
)

# Each run re-scans this many business days (accession-deduped) to self-heal
# after downtime/holidays.
EDGAR_BACKFILL_DAYS = 5

# ─────────────────────────────────────────────────────────────────────────────
# demand_signals/ — a 5th, structurally independent service: normalizes EDGAR
# insider buys, FINRA ATS dark-pool volume, and an options-flow proxy into one
# "real buyer demand" schema, screenable per ticker. US-market sources; see
# demand_signals/ticker_map.py for the CAN interlisting gap. Own DB, own cache,
# same conventions as EDGAR_* above -- kept separate rather than folded into
# edgar_service.py, matching this repo's "services stay independent" precedent.

DEMAND_DB_PATH = DATA_PATH / "demand_signals.db"
DEMAND_CACHE_PATH = CACHE_PATH / "demand_signals"

# Fair-access identification for FINRA/Yahoo requests (same spirit as
# EDGAR_USER_AGENT; neither FINRA nor Yahoo mandate this the way SEC does,
# but identifying the client politely costs nothing).
DEMAND_USER_AGENT = "StockScanner-DemandSignals/0.1 (chernyshov.yuriy@gmail.com)"

# FINRA's Query API is free but requires a (free) registered app -- OAuth2
# client-credentials, not an anonymous GET like SEC EDGAR's endpoints.
# FINRA_CLIENT_ID / FINRA_CLIENT_SECRET are read from .env by
# demand_signals/darkpool.py itself (config.py isn't the one that loads
# .env -- send_report.py's GMAIL_* does its own load_dotenv() the same
# way, since config.py is imported before that in most entrypoints);
# darkpool.py skips its fetch silently, logging why, when unset.

# Consecutive rising weekly dark-pool-ratio readings needed to flag a ticker.
DEMAND_DARKPOOL_RISING_WEEKS = 3

# volume/open-interest ratio above which an options chain leg is flagged
# "unusual" for the options_flow proxy.
DEMAND_OPTIONS_UNUSUAL_VOL_OI_RATIO = 2.0
