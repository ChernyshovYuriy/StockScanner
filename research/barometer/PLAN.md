# Barometer: calibrated move probabilities for TSX stocks

A barometer doesn't forecast the weather. It reads the pressure and gives the
odds of a storm. Barometer does the same for stocks: for every liquid TSX
stock on every day, it estimates **probabilities** of what the next 20
trading days hold. It does not call direction.

Started 2026-10-12. This is a research package, not a sleeve: it has no
capital, positions or schedule. It becomes something live only through
Phase 6, and only if the earlier gates pass.

## Why this framing

Every direction-based test in this repo has found no edge: entry rules,
volume spikes, news categories, timing. Direction is the crowd's "tug of
war", and what little signal is left gets traded away. Volatility and
tail risk, by contrast, cluster strongly. That is the most robust
statistical fact about markets, and it is crowd psychology too, since fear
is contagious. The 2026-10 live review found the sleeves' *entries* carry
negative alpha while the exits are fine. So the most plausible payoff here
is fewer bad entries, not finding winners.

## Targets (horizon H = 20 trading days, from the day after the features)

| # | Label | Definition |
|---|---|---|
| T1 | `y_stop_hit` | Any close in t+1..t+H below the core pipeline's own stop (`auto_pipeline.compute_levels`: min(entry − 1.5×ATR14, 20-day low × 0.99), floored at entry × 0.93), with entry = close at t |
| T2 | `y_up10` / `y_dn10` | 20-day forward return above +10% / below −10% |
| T3 | `fwd_rvol` | Std of daily log returns over t+1..t+H |
| T4 | `y_beat_mkt` | 20-day forward return above XIU.TO's over the same window |

## Phases

0. **Dataset** (`dataset.py`): one row per liquid stock-day with
   past-only features and future-only labels. Liquid means a close of at
   least $2, at least $1M 20-day average dollar volume, and a trade that day.
   Leakage is guarded by tests (`tests/test_barometer_dataset.py`) and by a
   real-data truncation check (`--check-lookahead`).
1. **Baselines**: base rate, and a volatility rule of thumb with no model.
2. **Models**: logistic regression first. Gradient boosting only if it beats
   logistic out of sample. Features are cross-sectionally ranked per day.
3. **Walk-forward evaluation**: expanding train window, yearly test
   years 2023 / 2024 / 2025 / 2026, with a 20-session embargo between train
   and test. Metrics: Brier skill vs both baselines, calibration, AUC, and
   per-year and per-regime breakdowns. Statistics are clustered by date.
4. **Decision gate** (fixed before any results):
   - T1/T2: beat both baselines in at least 3 of 4 test years, AUC ≥ 0.60,
     calibration within ±20%.
   - T3: beat the volatility rule of thumb in at least 3 of 4 years.
   - T4: top-vs-bottom group spread above 1% per 20 days (costs) in at
     least 3 of 4 years.
5. **Core-sleeve A/B** (only if T1 or T2 pass): 16-fold walk-forward, core
   sleeve as-is vs. skipping high-risk candidates. Needs an optional,
   default-off entry-filter hook in `backtest_runner.py`, which is a repo
   change to ask about first.
6. **Live** (only if Phase 5 passes): a display-only column on `/scanner`
   first.

## Known biases (stated up front)

- **Survivorship**: `data/market_cache.db` holds today's tickers only. Delisted
  losers are missing, which flatters the down-side labels.
- **Adjusted prices**: Yahoo's auto-adjusted history applies later
  dividends and splits backwards. This is mild, but it nudges the $2 price
  floor and the price levels.
- **`days_to_earnings`** uses the actual report date. Companies usually
  announce dates a few weeks ahead, so this is near-known within ~30
  sessions but is lookahead beyond that. It is clipped to 30 and excluded
  from the truncation check. Phase 3 must compare results with and without it.
- **Live vs panel entry**: the live pipeline enters at pivot × 1.005, but the
  panel uses that day's close. T1 is "would the standard stop survive from
  here", not a replay of actual intents. Phase 5 handles real intents.
