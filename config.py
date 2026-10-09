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
# The raw list CAN_TICKERS_URL is filtered from each week (swing_tickers.py's
# filters). run_backtest.py screens this list point-in-time by default, since
# CAN_TICKERS_URL itself is only ever today's selection.
BACKTEST_RAW_TICKERS_URL = "https://raw.githubusercontent.com/ChernyshovYuriy/Financing/refs/heads/main/data/can_tickers_full"
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

# Volume spike scanner — on-demand dashboard-only feature (no scheduled
# service, no DB; see volume_spike_scanner.py). Same raw, unfiltered
# full TSX/TSXV/CSE list as MOMENTUM_RAW_TICKERS_URL (a spike can happen on
# a ticker CAN_TICKERS_URL's ATR pre-filter would have excluded) — its own
# constant rather than reusing MOMENTUM_RAW_TICKERS_URL, keeping this
# feature decoupled from the momentum sleeve.
VOLUME_SPIKE_TICKERS_URL = "https://raw.githubusercontent.com/ChernyshovYuriy/Financing/refs/heads/main/data/can_tickers_full"

# Output paths — kept fully separate from the core sleeve's out/ files.
MOMENTUM_UNIVERSE_OUT_PATH = OUT_PATH / "can_tickers_momentum"
MOMENTUM_SCREENER_OUT_PATH = OUT_PATH / "momentum_screener_out"
MOMENTUM_REPORT_PATH = OUT_PATH / "momentum_report.html"
MOMENTUM_REPORT_POSITION_PATH = OUT_PATH / "momentum_position_monitor_report.html"
MOMENTUM_ALERTS_PATH = OUT_PATH / "momentum_alerts"


# ─────────────────────────────────────────────────────────────────────────────
# Macro conviction sleeve (6th, fully isolated paper account — see
# macro_regime.py / macro_buy.py / macro_monitor.py and CLAUDE.md)
# ─────────────────────────────────────────────────────────────────────────────
# A concentrated, top-down sleeve loosely inspired by Stanley Druckenmiller's
# approach: read the macro/liquidity backdrop first (macro_regime.py, FRED-
# based), then take 1-2 concentrated positions ONLY when the backdrop is
# supportive, sized much larger per-position than the core/momentum sleeves
# so a real conviction bet actually moves the book. Long-only (no shorting
# infrastructure exists in this repo) — a risk-off regime reading means "go
# to cash" (force-liquidate, see macro_monitor.py), never "go short". Own DB,
# own capital, own report/alert paths — same isolation precedent as the
# momentum sleeve. No own screener/pipeline: reads the core sleeve's own
# already-confirmed intents (read-only cross-DB, see macro_buy.py) instead of
# duplicating swing_tickers.py/canadian_stock_screener.py/auto_pipeline.py.
MACRO_DB_PATH = DATA_PATH / "macro.db"
MACRO_CACHE_PATH = CACHE_PATH / "macro_regime"

# Fair-access identification for FRED's API (St. Louis Fed) — same spirit as
# DEMAND_USER_AGENT / EDGAR_USER_AGENT.
MACRO_USER_AGENT = "StockScanner-MacroRegime/0.1 (chernyshov.yuriy@gmail.com)"

MACRO_INITIAL_CAPITAL = 10_000.0

# Hard concentration cap — the defining feature of this sleeve. 1-2 names
# max, never a diversified book. Not backtested (no history exists yet for
# this sleeve) — this is a starting point driven by the stated design goal
# (concentrated conviction), not a walk-forward result.
MACRO_MAX_POSITIONS = 2

# Sized deliberately much higher than the core sleeve's RISK_PER_TRADE_PCT=1.0
# or the momentum sleeve's 2.0 -- with MAX_POSITIONS=2 and a starting book of
# $10,000, remaining_slots-based sizing (see macro_buy.py) already puts up to
# ~$5,000 (half the book) into a single name at max_position_value alone; a
# risk-based cap of 5% still allows a full-size fill on any setup with >=10%
# stop distance (dollar_risk / per_share_risk), while still preventing an
# unusually tight-stop candidate from being oversized relative to its own
# risk. Starting point, not yet backtested -- no live/backtest history exists
# for this sleeve yet; revisit once live/backtest data accumulates.
MACRO_RISK_PER_TRADE_PCT = 5.0

# Per-series consecutive-trend lookback windows for macro_regime.py's vote
# logic (native frequency: T10Y2Y and BAMLH0A0HYM2 are daily, WALCL is
# weekly). 5 daily sessions (~1 trading week) and 3 weekly readings (~3
# weeks) are starting points -- not backtested; chosen to require a real,
# sustained move rather than single-day/week noise, mirroring the spirit of
# DEMAND_DARKPOOL_RISING_WEEKS=3 / DEMAND_SHORTVOL_TREND_DAYS=3 without
# copying their validated values (different data, different sleeve).
MACRO_CURVE_TREND_DAYS = 5
MACRO_CREDIT_TREND_DAYS = 5
MACRO_LIQUIDITY_TREND_WEEKS = 3

# Output paths -- kept fully separate from the core and momentum sleeves'
# out/ files. No MACRO_REPORT_PATH (momentum's pipeline-report equivalent):
# this sleeve has no own screener/pipeline, so there's no pipeline report to
# send -- only the position-monitor report below.
MACRO_REPORT_POSITION_PATH = OUT_PATH / "macro_position_monitor_report.html"
MACRO_ALERTS_PATH = OUT_PATH / "macro_alerts"


# ─────────────────────────────────────────────────────────────────────────────
# Kangaroo Tail sleeve (breakout-entry, defined-risk paper account — see
# research/kangaroo_tail/, kangaroo_pipeline.py / kangaroo_buy.py /
# kangaroo_monitor.py and CLAUDE.md)
# ─────────────────────────────────────────────────────────────────────────────
# A fully isolated 8th sleeve/service, added 2026-09. Unlike every other
# sleeve, an entry here is NOT "buy at next open off a confirmed intent" —
# a Kangaroo Tail detection is a PENDING BREAKOUT setup: risk is only taken
# once price actually trades through the tail candle's own high (a
# buy-stop trigger), with the stop at the tail's low and the target a
# multiple of that risk (KANGAROO_RR_TARGET). Phase 2's 16-fold walk-forward
# + R-multiple breakout backtest found a real, if modest, edge for exactly
# this entry mechanic (win 56.6%, avg_R +0.246, PF 1.72, p=0.033 at rr=1.5,
# n=143) — same-day "buy at the tail's own close" showed no edge under any
# config tested and is NOT what this sleeve trades (see
# research/kangaroo_tail/KANGAROO_TAIL_VERIFICATION_FINDINGS.md).
KANGAROO_DB_PATH = DATA_PATH / "kangaroo.db"
KANGAROO_INITIAL_CAPITAL = 10_000.0

# Sizing starting points, not yet backtested at the portfolio level (only
# the per-trade R-multiple edge above is validated) — same honesty
# precedent as MACRO_MAX_POSITIONS/MACRO_RISK_PER_TRADE_PCT. Matches the
# momentum sleeve's own starting values for a similarly-scaled book.
KANGAROO_MAX_POSITIONS = 5
KANGAROO_RISK_PER_TRADE_PCT = 2.0

# Reward:risk target for the fixed take-profit (target = trigger + RR *
# (trigger - stop)). 1.5 is the value the findings doc's headline numbers
# above were computed at; 1.0/2.0 were also swept with a similar
# significance pattern (see the findings doc) if this is ever revisited.
KANGAROO_RR_TARGET = 1.5

# Time stop: cancel a still-pending (untriggered) breakout intent, or exit
# a triggered position that has hit neither stop nor target, after this
# many TRADING days — matches Phase 2's own max_hold_bars exactly, so live
# behaviour matches what was actually backtested.
KANGAROO_MAX_HOLD_DAYS = 10

# Output paths — kept fully separate from every other sleeve's out/ files.
KANGAROO_REPORT_PATH = OUT_PATH / "kangaroo_report.html"
KANGAROO_REPORT_POSITION_PATH = OUT_PATH / "kangaroo_position_monitor_report.html"
KANGAROO_ALERTS_PATH = OUT_PATH / "kangaroo_alerts"


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
# insider buys, FINRA ATS dark-pool volume, FINRA's daily short-sale-volume
# file, and an options-flow proxy into one "real buyer demand" schema,
# screenable per ticker. US-market sources; see demand_signals/ticker_map.py
# for the CAN interlisting gap. Own DB, own cache, same conventions as
# EDGAR_* above -- kept separate rather than folded into edgar_service.py,
# matching this repo's "services stay independent" precedent.

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

# FINRA daily short-sale-volume file (short_volume.py): the daily
# short-volume/total-volume ratio needs no auth and no OAuth app, unlike
# darkpool.py's ATS weekly summary -- see demand_signals/short_volume.py.
# Consecutive rising/falling daily readings needed to flag a ticker.
DEMAND_SHORTVOL_TREND_DAYS = 3
# Day-over-day ratio change treated as "full strength" (1.0) for the
# short_volume_covering/short_volume_pressure signal.
DEMAND_SHORTVOL_STRENGTH_SCALE = 0.05

# ─────────────────────────────────────────────────────────────────────────────
# Ticker Indicator Board — 9th service, a read-only research board built on
# Elder's "The New Trading for a Living" (see scanner_board/PLAN.md). No
# capital/positions at all (unlike every paper sleeve above) -- one snapshot
# row per ticker per run, every column a raw indicator reading or a named,
# book-cited label. Own SQLite DB, same isolation precedent as the Triple
# Screen tracker above. Scans the same CAN_TICKERS_URL universe every other
# service uses (no separate SCANNER_*_TICKERS_URL needed).
SCANNER_BOARD_DB_PATH = DATA_PATH / "scanner_board.db"


# ─────────────────────────────────────────────────────────────────────────────
# Press-release tracker — 10th service, a news aggregator/collector (see
# press_release_tracker/ + press_release_service.py). No capital or
# positions at all, same "pure collector" shape as EDGAR/demand-signals --
# polls RSS newswire feeds, parses each new item with an LLM, and emails a
# digest of what's new. Grew directly out of the Volume spike scanner's own
# research above: the same-day intraday edge is real, but acting on it needs
# the catalyst detected fast, and a press release is often that catalyst.
# Built as a live, forward-observing collector (no historical press-release
# archive was confirmed available to backtest against) rather than a
# backtested sleeve -- see MEMORY for the OMI.V case study this grew from.
PRESS_RELEASE_DB_PATH = DATA_PATH / "press_releases.db"

# One feed to start (GlobeNewswire's "News from Canada" feed) -- a plain
# list so a second/third source (e.g. Newsfile Corp, CNW) is just another
# URL appended here later, no code change needed in press_release_service.py.
# TMX Newsfile was added 2026-09 then removed the same month -- it flooded
# News Watchlist's inbox with noise, so back to the single GlobeNewswire feed.
# TMX Newsfile re-added 2026-10 per industry (it has no working all-news
# feed): every item is still fetched, parsed, stored and scored, but only
# a materiality 'high' one is analysed, emailed or seeded into News
# Watchlist's inbox -- see PRESS_RELEASE_HIGH_ONLY_FEEDS below.
PRESS_RELEASE_NEWSFILE_FEEDS = [
    "https://feeds.newsfilecorp.com/industry/mining-metals",
    "https://feeds.newsfilecorp.com/industry/technology",
    "https://feeds.newsfilecorp.com/industry/cannabis",
    "https://feeds.newsfilecorp.com/industry/oil-gas",
]
# CSE company news (added 2026-10): every CSE-listed company's releases,
# whatever wire they used (ACCESS Newswire etc. have no public feed). Not
# RSS -- a JSON file of the whole archive, read via an HTTP Range request
# of its first PRESS_RELEASE_CSE_NEWS_RANGE_BYTES (~2 days of releases);
# see press_release_tracker/cse_news.py. Title-only: no full-article read.
PRESS_RELEASE_CSE_NEWS_FEED = "https://webapi-backup.thecse.com/news-releases/en/news-releases.json"
PRESS_RELEASE_CSE_NEWS_RANGE_BYTES = 80_000
PRESS_RELEASE_FEEDS = [
    "https://www.globenewswire.com/RssFeed/country/Canada/feedTitle/GlobeNewswire%20-%20News%20from%20Canada",
    *PRESS_RELEASE_NEWSFILE_FEEDS,
    PRESS_RELEASE_CSE_NEWS_FEED,
]
# Feeds whose items reach the user (email, News Watchlist inbox) and get a
# full-article read only when the classifier rates them materiality 'high';
# the rest are filed silently (still in press_releases.db, still scored by
# news_watchlist/outcomes.py, so the gate itself can be checked later).
PRESS_RELEASE_HIGH_ONLY_FEEDS = frozenset([*PRESS_RELEASE_NEWSFILE_FEEDS, PRESS_RELEASE_CSE_NEWS_FEED])

# Fair-access identification for the RSS fetch (same spirit as
# EDGAR_USER_AGENT / DEMAND_USER_AGENT / MACRO_USER_AGENT).
PRESS_RELEASE_USER_AGENT = "StockScanner-PressRelease/0.1 (chernyshov.yuriy@gmail.com)"

# OPENAI_API_KEY is read from .env by press_release_tracker/llm_parser.py
# itself (same self-contained pattern as FINRA_CLIENT_ID/SECRET and
# FRED_API_KEY -- config.py itself doesn't load .env). Unconfigured means
# each item still gets emailed with its raw RSS title/link, just without
# the LLM's ticker/company/category/materiality/summary fields -- same
# "degrade gracefully, don't crash" convention as every other optional
# credential in this repo.

# Cheapest/smallest OpenAI text tier -- this task (short-text classify +
# one-sentence summary) doesn't need a frontier model, and token cost
# scales with model tier more than task difficulty here. Verify current
# pricing/model availability at https://platform.openai.com/pricing before
# relying on any cost estimate; this is a one-line change if a
# cheaper/better tier appears later.
PRESS_RELEASE_LLM_MODEL = "gpt-5-nano"

# Truncate a release's RSS description before sending it to the LLM --
# caps token spend per item regardless of how long a given press release
# runs.
PRESS_RELEASE_LLM_MAX_DESCRIPTION_CHARS = 2000

# Two delivery lanes (see press_release_service.py): a materiality='high'
# item emails immediately (its own run, no batching -- the whole point of
# this sleeve is catching a market-moving catalyst fast); everything else
# (medium/low/unclassified) accumulates and flushes in one digest at most
# this often, so a busy newswire morning doesn't produce an email every 5
# minutes (the timer's own poll interval) for routine releases.
PRESS_RELEASE_BATCH_INTERVAL_MINUTES = 60

# Analyst read (press_release_tracker/analyst.py, added 2026-10): the FULL
# article of an important release (a ticker, an English page, and one of
# these categories or materiality 'high') goes to a stronger model than the
# classifier above -- reading figures, tables and dilution terms is a
# harder task than classifying a teaser. ~30-60 items a day; verify
# current pricing before relying on a cost estimate.
PRESS_RELEASE_ANALYSIS_MODEL = "gpt-5-mini"
PRESS_RELEASE_ANALYSIS_CATEGORIES = ("earnings", "financing", "ma_acquisition", "contract_award")
# Caps tokens per analysis; a full results release with its statement
# tables is typically 10-25k characters.
PRESS_RELEASE_ANALYSIS_MAX_BODY_CHARS = 24000
# Minimum gap between two article-page fetches from the same wire host
# (press_release_tracker/article.py). Added 2026-10 after the archive
# backfill's ~860 GlobeNewswire pages/hour got the home IP refused by
# Akamai (403), and a burst of ~6 Newsfile pages drew an AWS WAF challenge.
PRESS_RELEASE_ARTICLE_MIN_INTERVAL_SECONDS = 10.0
# Historical backfill (press_release_tracker/archive.py): GlobeNewswire's
# monthly English sitemaps, Canadian-listed releases only, in their OWN
# DB -- never read by the live services, so 3-year-old releases can't
# leak into the News Watchlist inbox or mix with live outcome numbers.
PRESS_RELEASE_ARCHIVE_DB_PATH = DATA_PATH / "press_release_archive.db"
PRESS_RELEASE_ARCHIVE_CACHE_PATH = CACHE_PATH / "press_release_archive"


# ─────────────────────────────────────────────────────────────────────────────
# News watchlist — 11th service, a follow-through tracker for
# press_release_tracker's own catches (see news_watchlist/__init__.py). No
# capital or positions, same "pure collector" shape as press_release_tracker
# above. Reads data/press_releases.db read-only (never writes back) to
# auto-seed an "inbox" of ticker candidates; a human confirms an item into
# "watching" before it gets any daily price tracking at all -- the triage
# step stays manual by design, so automating the bookkeeping doesn't drown
# the judgment call in noise. Two independently-scheduled modes (see
# news_watchlist_service.py's --mode): `seed` runs every ~10 minutes and
# emails an immediate alert the moment something new lands in the inbox
# (the user explicitly asked to keep getting notified fast, on top of the
# dashboard tab); `update-prices` runs once daily and appends today's price
# to every already-confirmed watching item.
NEWS_WATCHLIST_DB_PATH = DATA_PATH / "news_watchlist.db"
# A candidate's own RSS pubDate older than this many days is skipped by
# seed_inbox() rather than seeded -- this service is about fast follow-
# through on a fresh catalyst; a stale article surfacing late (e.g. after
# the service was down, or a delayed LLM parse) has no such catalyst left
# to follow, and would otherwise get stamped with today's date/price as if
# it just broke.
NEWS_WATCHLIST_MAX_ARTICLE_AGE_DAYS = 7
