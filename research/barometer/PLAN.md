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

## Phase 1 results (2026-10-12, `python -m research.barometer.baselines`)

Walk-forward, base rate vs. volatility rule. Skill is the Brier-score improvement of
the rule over the base rate; t is HAC-corrected and date-clustered (|t| > 2 is significant).

| Target | Rule skill (2023 / 24 / 25 / 26) | t-stat | Rule AUC | Rule calibration error |
|---|---|---|---|---|
| Stop hit | 4.9% / 8.6% / 3.8% / 11.1% | −1.8 / −4.1 / −1.5 / −2.5 | 0.62–0.68 | 6–22% |
| Up 10% | 3.2% / 2.9% / 8.7% / 0.2% | −3.9 / −2.2 / −3.5 / −0.1 | 0.60–0.71 | 22–46% |
| Down 10% | 8.0% / 7.9% / 1.9% / 9.3% | −7.3 / −6.1 / −0.4 / −1.8 | 0.70–0.73 | 15–36% |
| Beat XIU | about 0 (the rule is a constant 0.5) | n.s. | 0.50 | — |
| Volatility (QLIKE) | 36% / 28% / 47% / 51% | −5.6 / −2.0 / −7.6 / −7.6 | — | — |

What this means for Phase 2:
- **The volatility rule is the bar to beat, not the base rate.** It already ranks
  risk usefully (AUC above 0.60 every year) and wins on volatility by a wide margin.
- **The AUC ≥ 0.60 gate is already met by the rule itself**, so it doesn't
  separate models from the rule. The binding gate is "beat both baselines in at
  least 3 of 4 years". The gate is left as written, since it was fixed before
  any results.
- **The rule is badly calibrated.** It is mostly off by more than 20%, and by 46%
  on +10% in the 2025 rally, because it assumes zero drift and constant
  volatility. Calibration is the clearest place a fitted model can add value.
- **Beating XIU is unpredictable here, as expected.**

## Phase 2 results (2026-10-12, `python -m research.barometer.models [--earnings]`)

Same walk-forward splits as Phase 1. Skill is the Brier improvement vs. the volatility
rule (QLIKE for volatility); t is HAC-corrected and date-clustered, negative = model better.

| Target | Logistic skill vs rule (2023 / 24 / 25 / 26) | Best t in the model's favour | GBM | Verdict |
|---|---|---|---|---|
| Stop hit | −2.2% / −2.3% / +2.0% / +0.2% | −0.5 | same or worse | **does not beat the rule** |
| Up 10% | −1.8% / +1.2% / −1.5% / +1.7% | −1.4 | ≈ rule | **does not beat the rule** |
| Down 10% | −0.8% / −2.4% / +2.9% / +2.3% | −1.3 | worse in 2024 (t = +2.9) | **does not beat the rule** |
| Beat XIU | about 0, AUC 0.50–0.54 | — | same | **unpredictable** |
| Volatility (ridge) | **+35% / +52% / +34% / +29%** | **−3.7 to −7.9** | (same ridge) | **beats the rule every year** |

- Models beat the base rate on stop hit and tails (like the rule), but they add
  **nothing significant over the closed-form rule**. Their calibration is not
  better either: it is often worse, because linear extrapolation of market
  features fails in regimes outside the training range. The synthetic tests
  showed the same pattern (2024, dn10).
- Gradient boosting never beats logistic regression reliably, so per the plan
  it is dropped. Before its leaves were sized for overlapping labels, it overfit
  pure noise (−5 to −8% vs the rule).
- `days_to_earnings` changes nothing material (all differences within about ±1
  point), so its mild lookahead is moot. It stays off.
- In the `fwd_rvol` rows, "logistic" and "gbm" are the same ridge model; the
  model name only applies to the binary targets.

Preview of the Phase 4 gate (formal check in Phase 3/4): T1 and T2 fail (no
year-over-year win against the rule), T4 fails, **T3 passes clearly.** The
practical reading is that for stop-hit and tail odds, the free closed-form rule
is as good as a fitted model, while volatility forecasts benefit from fitting.

## Phases 3–4: the gate (2026-10-12, `python -m research.barometer.gate`)

The model is logistic regression (ridge for volatility), with no earnings feature.
Interpretations pinned in `gate.py` before the run: "beats" means lower loss
that year, and every criterion must hold in at least 3 of 4 test years.
Significance (HAC t < −2) is reported but not required.

| Gate | Target | Beats base | Beats rule (significant) | AUC ≥ 0.60 | Calibration ≤ 20% | Verdict |
|---|---|---|---|---|---|---|
| T1 | Stop hit | 4/4 | 2/4 (0) | 4/4 | 3/4 | **FAIL** |
| T2 | Up 10% | 4/4 | 2/4 (0) | 4/4 | 0/4 | **FAIL** |
| T2 | Down 10% | 4/4 | 2/4 (0) | 4/4 | 1/4 | **FAIL** |
| T3 | Volatility | — | **4/4 (4)** | — | — | **PASS** |
| T4 | Beat XIU | spread > 1% in 2/4 years: +0.02 / +0.81 / +1.35 / +1.03%; pooled +0.78%, t = 1.40 | | | | **FAIL** |

Details:
- **T1 breaks down in bear markets.** With XIU below its 200-day average (6.5%
  of rows), the model is 12% *worse* than the rule. In bull markets it ties
  the rule (+0.3%). This is the regime-extrapolation weakness seen in the
  synthetic tests.
- **Calibration (stop hit, pooled deciles):** the model over-forecasts the
  riskiest names (68.5% predicted vs 54.7% observed in the top decile). The
  rule under-forecasts the safest names (7.9% vs 14.5%) but is closer in
  between.
- **T4's small spread** comes mainly from low ATR and momentum ranks (the
  well-known low-volatility and momentum factors). It is positive in all 4
  years but below the 1% cost bar in 2 of them, and not significant overall.
- **The gate procedure itself was tested on synthetic worlds:** a world with
  truly predictable direction passes T4, and pure noise fails T2 and T4. On
  noise, the model can earn a *small real* stop-hit edge from a better
  volatility estimate (it also sees 60-day vol), which is why significance
  is reported next to every win.

**Decision, per the plan:** Phase 5 was conditional on T1 or T2 passing.
Neither did, so **Phase 5 (core-sleeve filter A/B) does not run on the model.**
The only target that passed is T3, volatility forecasting. Using the volatility
rule itself, or the T3 forecast, in the sleeves would be a *new* hypothesis
outside this plan and would need its own pre-registered test.
