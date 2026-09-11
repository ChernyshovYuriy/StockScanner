# Ticker Indicator Board — Plan

A new, isolated, **read-only** 9th service. One row per ticker, many columns.
Each column is either a raw indicator reading or a named, book-cited rule
("MACD-H slope agrees with price → trend is safe"). No column is a
synthesized buy/sell verdict — Elder himself warns against that: "beginners
look for a magic bullet... markets are too complex to be analyzed with a
single indicator" (Ch.39). The closest thing to a verdict is the **Triple
Screen Alignment** column, and that's literally Elder's own published
decision table (Ch.39), not something invented here.

Source material: Dr. Alexander Elder, *The New Trading for a Living* (2014),
chapters 22–30 (Moving Averages → Force Index) and 39–40 (Triple Screen,
Impulse System). Cross-checked against `/home/yurii/dev/pythonfintech`'s
notebooks (`compute-slope-series`, `market-stage-detection`,
`trading-view-stochastic-rsi`, `computing-simple-moving-averages`) and this
repo's existing `canadian_stock_screener.py`.

Decisions already made (asked of the user 2026-09-11):
- **Delivery**: new dashboard tab (`/scanner`), same pattern as `/momentum`,
  `/triple-screen`, `/kangaroo`.
- **Timeframes**: daily **and** weekly, from v1 — Triple Screen/Impulse are
  inherently multi-timeframe methods.
- **Cadence**: its own scheduled service (9th isolated service, own systemd
  timer), same "services stay independent" precedent as every other
  sleeve/collector in this repo.
- **Universe**: the full `config.CAN_TICKERS_URL` universe.

## Single-shared-logic rule for this build

No indicator math is copy-pasted between modules. Concretely:
- Every indicator already implemented in
  `canadian_stock_screener.TechnicalIndicators` (`sma`, `ema`, `rsi`, `macd`,
  `true_range`, `atr`, `adx`, `obv`, `linear_regression_slope`,
  `weekly_resample`) is reused by **import**, never re-derived, by every
  `scanner_board` module.
- `true_range`/`atr` did not previously exist as their own methods — they
  were inlined only inside `adx()`. Phase 1 extracted them out (additive,
  behavior-preserving refactor of `canadian_stock_screener.py`, verified by
  the full existing `TestAdx` suite still passing unchanged) so this board's
  ATR/ATR-channel columns reuse the exact same computation `adx()` already
  relies on, instead of a second copy of the True Range formula.
- Any indicator genuinely new to this codebase (Stochastic, Accumulation/
  Distribution, Force Index, the slope→label classifier) is implemented
  exactly once, inside `scanner_board/`, built out of the above primitives
  wherever possible (e.g. Stochastic's %D smoothing reuses
  `TechnicalIndicators.sma`; Force Index's smoothing reuses
  `TechnicalIndicators.ema`) rather than hand-rolling a second moving
  average.
- Every new function has its own test file under `tests/`, following this
  repo's existing adversarial-test conventions (ground-truth hand traces,
  boundary cases, property-based tests via `hypothesis`).

## File layout

```
scanner_board/
  PLAN.md
  __init__.py
  slope.py           # Phase 1: Rising/Falling/Flat classification
  indicators.py       # Phase 1: Stochastic, Accumulation/Distribution, Force Index
  divergence.py       # Phase 2: shared pivot-based divergence detector
  thesis_rules.py      # Phase 3: book-thesis -> labeled-column functions
  triple_screen.py    # Phase 3: Impulse System color + Triple Screen alignment
  store.py             # Phase 4: own SQLite store, data/scanner_board.db
scanner_pipeline.py    # Phase 4: daily entrypoint (fetch, compute, persist)
scanner_dashboard_data.py   # Phase 5: read-only mirror for the dashboard
templates/scanner.html      # Phase 5: new /scanner tab
system/stockscanner-scanner-pipeline.{service,timer}   # Phase 6
```

`canadian_stock_screener.TechnicalIndicators` gained two new static methods
in Phase 1 (`true_range`, `atr`) — the one deliberate cross-module addition
this plan calls for; everything else in `scanner_board/` is new, additive
code that only *imports* existing modules.

## Phase 1 — raw indicator math + slope classification (THIS PHASE)

- `scanner_board/slope.py`: `classify_slope(series, period, threshold) ->
  Rising/Falling/Flat`, built on `TechnicalIndicators.linear_regression_slope`
  (not a second slope implementation — see "single-shared-logic rule"
  above). Threshold reuses the `0.001` cutoff already used for this exact
  normalized-slope quantity elsewhere in this codebase
  (`score_macd`'s `hist_slope`, `score_stage2`'s `ma30_slope`).
  Cites Ch.22: "the single most important message of a moving average
  comes from the direction of its slope."
- `scanner_board/indicators.py`:
  - `stochastic()` — Slow Stochastic, Ch.26. Fast %K from a rolling
    high/low window; %D smoothing done via `TechnicalIndicators.sma`
    (the book explicitly says %D smoothing "can be done in several ways,"
    so the plain-SMA variant — what most charting software does — was
    chosen over the book's own sum-of-differences formula, for one
    fewer NaN-propagation edge case at no book-fidelity cost). Slow %K =
    Fast %D, Slow %D = SMA(Slow %K) again, exactly as the book describes
    constructing Slow Stochastic.
  - `accumulation_distribution()` — Ch.29. `(Close-Open)/(High-Low) *
    Volume`, cumulative sum. A `High==Low` bar is treated as a zero
    contribution (not a crash, not Inf) rather than dividing by zero.
  - `force_index()` — Ch.30. Raw `Volume * (Close_today - Close_yesterday)`,
    smoothed by `TechnicalIndicators.ema` at both the book's short
    (2-day, "pinpoints entry/exit points") and long (13-day, "confirms
    trends") windows.
- Extracted `TechnicalIndicators.true_range` / `.atr` in
  `canadian_stock_screener.py` (see above).
- Tests: `tests/test_scanner_board_slope.py`,
  `tests/test_scanner_board_indicators.py`, plus two new test classes
  (`TestTrueRange`, `TestAtr`) added to the existing
  `tests/test_adversarial_technical_indicators.py`.

## Phase 2 — divergence detector

Divergences are called out repeatedly as the single strongest signal type
for RSI, Stochastic, MACD-H, OBV, A/D, and Force Index — one shared,
well-tested primitive rather than five ad-hoc copies.

- Find the last two significant pivot lows (bullish) / highs (bearish) in
  price over a lookback window (Kerry Lovvorn's research, cited Ch.23: 20–40
  bars apart, "the closer to 20, the better").
- Compare the indicator's value at those same two pivots.
- Enforce the book's explicit centerline-crossing requirement for MACD-H
  divergences (Ch.23): "if there is no crossover, there is no divergence."
  Easy to get wrong; worth its own stress-test pass in the spirit of
  `research/triple_screen/TRIPLE_SCREEN_STRESS_FINDINGS.md`.
- Output: `Bullish` / `Bearish` / `None` per (price, indicator) pair.

## Phase 3 — book-thesis columns + Triple Screen / Impulse System

One function per labeled column, each docstring citing its chapter. Full
column list (grouped): Identity/Price; Trend/MA (`MA Slope`, `Price vs MA`,
`Value Zone`); MACD (`MACD Cross`, `MACD-H Slope`, **`Trend Health`**,
`MACD-H Divergence`, `MACD-H New 3mo Extreme`); Directional System/ADX
(`DI Bias`, `ADX Trend`, `ADX Regime`); Oscillators (`Stochastic Zone`,
`RSI Zone`, plus divergence columns — reference lines calibrated per-ticker
via the book's 5% rule, not a hardcoded 80/20 or 70/30, recalibrated
quarterly); Volume (`Volume vs Avg`, `OBV Trend`+divergence, `A/D
Trend`+divergence, `Force(2) Zone`, `Force(13) Bias`+divergence);
Multi-timeframe (`Impulse (Daily/Weekly)`, `Triple Screen` alignment,
`Weinstein Stage`); Volatility (`ATR`/`ATR%`, ±1/2/3 ATR band).

`triple_screen.py` implements:
- **Impulse System** color (Ch.40): EMA slope + MACD-H slope combined per
  Elder's own 4-row table (green/red/blue).
- **Triple Screen Alignment** (Ch.39): weekly trend × daily trend → Elder's
  own published 4-row summary table (Stand-aside / Long-setup /
  Short-setup).

## Phase 4 — pipeline + storage

`scanner_pipeline.py`: daily fetch (weekly resampled from cached daily bars
via `market_data_cache.py`, not a second yfinance round-trip — mirrors how
Elder himself performs "weekly studies each day" in Ch.23) → compute every
Phase 1–3 column → persist to `scanner_board/store.py`'s own SQLite DB,
`data/scanner_board.db` (own schema, no capital/positions — like
`triple_screen_tracker.db`, not `trading.db`).

## Phase 5 — dashboard

`scanner_dashboard_data.py` (read-only mirror, same isolation pattern as
`momentum_dashboard_data.py`/`kangaroo_dashboard_data.py` — its own
connection, never reaches into `store.py`'s writer path) + `/scanner` route
+ `templates/scanner.html`. Visual conventions: Elder's own Impulse colors
(green/red/blue) reused for every trend-agreement column, not a separate
palette; ▲▼– glyphs for slope columns; grouped column headers (Trend | MACD
| ADX | Oscillators | Volume | Multi-timeframe); default sort by ticker, not
by any score, to avoid re-introducing a magic composite ranking.

## Phase 6 — deployment

`system/stockscanner-scanner-pipeline.{service,timer}` (~17:15 ET, after the
Triple Screen tracker's 17:00 slot — same slot-stacking convention as the
rest of `system/`), `install-services.sh` entry, `CLAUDE.md` update.

## Explicitly out of scope for v1

No auto-generated composite score, no buy/sell button, no position sizing,
no email digest (could follow the Triple Screen tracker's pattern later),
no A-trade grading (Ch.55 — a discretionary checklist, not a computable
column).
