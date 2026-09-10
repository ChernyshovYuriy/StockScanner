# Triple Screen — Adversarial Stress Test Findings

**Status: all 3 confirmed defects FIXED.** Originally: `tests/test_triple_screen_stress.py`
(54 tests: 51 passed, 3 xfailed — the 3 defects below, each demonstrated failing against the
pre-fix code). After the fixes: same 54 tests, **all 54 pass, 0 xfail**. Run together with the
existing suite: `tests/test_triple_screen_{batch,edge_cases,engine,screens}.py` → **105 passed**,
0 failures, 0 xfail.

Every claim below is backed by a specific test in that file — no conclusion here rests on reading the code.

---

## Defects (ranked: silent-wrong first, then crashes, then contract mismatches, then robustness gaps) — all FIXED

### 1. SILENT-WRONG (most severe) — reverse-chronological bars invert the trend with no error
- **Status: FIXED.** `ema_slope_direction`, `force_index_pullback`, and `prior_bar_breakout` (`indicators.py`) now call `.sort_index()` on their input before reading off "latest"/"prior" positionally, applied uniformly since all three functions shared the identical vulnerability. Test now **PASSES** (was XFAIL).
- **Test:** `test_part3_reverse_chronological_bars_must_not_silently_invert_the_trend`
- **Input:** A textbook 60-bar uptrend, rows reversed (`bars.iloc[::-1]`) so the index is descending instead of ascending.
- **Expected:** `Direction.UP` (it's the same uptrend data — order in the array shouldn't matter to what "the trend" means; at minimum, a clear error rather than a confident wrong answer).
- **Actual:** `Direction.DOWN`, `{'ema': 106.38, 'ema_prior': 109.60}` — a fully plausible-looking result, no error, no warning.
- **Why:** Every screen (`indicators.py`) uses positional `.iloc[-1]` / `.iloc[-2]` and never checks the index is ascending, despite `types.py`'s `PriceData` docstring documenting "sorted ascending by date" as the contract. Nothing in `DataProvider`, `PriceData`, or any screen enforces or even checks it.
- **Impact:** Any provider bug, pandas version change, or manual `PriceData` construction that hands bars in descending order produces the *exact opposite* signal, silently. This is the single worst class of failure the spec calls out.

### 2. SILENT-WRONG — `+inf` in the latest bar's High/Low fires the trigger with no guard
- **Status: FIXED.** `prior_bar_breakout` now checks `math.isfinite()` on both the latest and prior value after the (already-existing) `dropna()`, since `dropna()` only ever caught NaN, never inf; a non-finite value now degrades to the same safe default (`False, {}`) as missing data, rather than being compared as if it were real. `force_index_pullback` needed no change — it's already incidentally protected by pandas' own `.ewm().mean()`, verified separately by `test_part3_inf_in_midseries_close_does_not_corrupt_the_trend_read` (unaffected by this fix). Test now **PASSES** (was XFAIL).
- **Test:** `test_part3_inf_in_latest_high_must_not_silently_fire_the_trigger`
- **Input:** A clean 20-bar uptrend with the latest bar's `High` set to `float("inf")`.
- **Expected:** `triggered=False` (or a raised error) — a non-finite OHLC value should never be able to justify a live signal.
- **Actual:** `triggered=True`, `indicator_values={'today_high': inf, 'prior_high': 108.79}`, reported as if it were a real breakout.
- **Why:** `prior_bar_breakout` (`indicators.py:65-81`) does a plain `today > prior` comparison with no `np.isfinite` check. Contrast with `ema_slope_direction`, which is *accidentally* protected — verified separately (`test_part3_inf_in_midseries_close_does_not_corrupt_the_trend_read`, PASSED) — because pandas' own `.ewm().mean()` happens to treat non-finite values as missing internally. That protection is incidental (a pandas implementation detail, not something this package does on purpose) and doesn't extend to the trigger screen's plain comparison at all.
- **Impact:** A single corrupted OHLCV value from a bad feed can, combined with `Direction.UP` + `pullback_present`, produce a confident live **BUY** with no red flag anywhere in the output.

### 3. CONTRACT MISMATCH — `run_batch` has no per-ticker error isolation
- **Status: FIXED.** `run_batch` (`batch.py`) now wraps each ticker's `provider.get_bars()` + `engine.evaluate()` in `try/except Exception`, printing a `WARNING: skipping {ticker} (...)` and continuing to the next ticker rather than letting the exception propagate out of the loop. Mirrors the existing `WARNING:`-print style already used in `yfinance_provider.py`. Test now **PASSES** (was XFAIL).
- **Test:** `test_part3_one_ticker_provider_failure_does_not_abort_the_whole_batch`
- **Input:** Three tickers; the fake provider raises `RuntimeError` for the middle one only.
- **Expected:** `{"GOOD1": ..., "GOOD2": ...}` — the two good tickers still resolve.
- **Actual:** The exception propagates out of `run_batch` entirely; **no** ticker gets a result, not even the two good ones.
- **Why:** `batch.py`'s `run_batch()` loop has no `try/except` around `provider.get_bars(ticker, ...)`. The shipped `YFinanceDataProvider` happens to never raise (its own `fetch_bars` catches everything internally), which is why this has never surfaced in practice — but the isolation guarantee lives entirely in one concrete provider, not in `run_batch` itself, where the spec (and the module's own "one bad ticker never aborts the batch" framing elsewhere in this package) says it should.
- **Impact:** A future provider (rate-limited API, different data source, or even a bug in a timeframe mapping raising `ValueError`) takes down the entire batch run instead of degrading per-ticker.

---

## Confirmed NOT defects (proven, not assumed)

| Claim | Test | Result |
|---|---|---|
| `Direction.FLAT` never yields BUY/SELL (all pullback/trigger/position combos) | `test_part1_flat_trend_never_yields_buy_or_sell` | PASS |
| UP trend never yields SELL; DOWN trend never yields BUY | `test_part1_up_trend_never_yields_sell_and_down_trend_never_yields_buy` | PASS |
| BUY requires all 3 conditions; any one missing blocks it | `test_part1_buy_requires_all_three_missing_any_one_blocks_it` | PASS |
| Position state alone (not indicators) decides HOLD-vs-BUY and WAIT-vs-SELL | `test_part1_position_state_not_indicators_decides_hold_vs_buy_and_wait_vs_sell` | PASS |
| EMA-slope exact-zero boundary reads FLAT (not UP/DOWN) | `test_part2_ema_slope_exactly_zero_is_flat_not_up_or_down` | PASS |
| Force Index exactly 0 → pullback absent (strict `<`, not `<=`) | `test_part2_force_index_exactly_zero_is_pullback_absent` | PASS |
| Trigger uses strict `>`; exact equality to prior high does not fire | `test_part2_trigger_exact_equality_to_prior_high_does_not_fire` | PASS |
| Minimum viable trend data is exactly `ema_period + lag` (16 bars); 15 degrades to FLAT | `test_part2_minimum_viable_trend_data_boundary_is_exactly_ema_period_plus_lag` | PASS |
| Minimum viable entry data (Screen 2) is 2 bars; 1 degrades to `False` | `test_part2_minimum_viable_entry_data_boundary_is_two_bars` | PASS |
| ~~Minimum viable trigger data (Screen 3) is 2 bars~~ **SUPERSEDED 2026-09** — see addendum below: Screen 3 now needs 30 bars | `test_part2_minimum_viable_trigger_confirmation_data_boundary_is_30_bars` | PASS |
| Trend read at a transition point depends only on the prefix up to that point (no leakage from bars appended later in the same array) | `test_part2_trend_transition_up_to_down_to_flat_no_stale_leakage` | PASS |
| One-bar spike then flat: no crash | `test_part2_single_spike_then_flat_does_not_crash` | PASS |
| EntryScreen verdict is unaffected by the trend series (screen independence) | `test_part2_entry_screen_unaffected_by_trend_series_length_or_content`, `test_part4_screen_independence_trend_result_unaffected_by_entry_series` | PASS |
| One timeframe totally empty, the other full: degrades safely, no crash | `test_part2_one_timeframe_empty_other_full_degrades_safely` | PASS |
| Empty DataFrame on all 3 screens: safe defaults, no crash | `test_part3_empty_dataframe_all_three_screens_degrade_safely` | PASS |
| `None` where a series is expected: raises (fails loudly, not silently) `AttributeError` | `test_part3_none_series_raises_rather_than_silently_misbehaving` | PASS |
| Missing `Volume` column: raises `KeyError` (fails loudly) | `test_part3_missing_volume_column_raises_rather_than_silently_misbehaving` | PASS |
| Zero volume: pullback absent, no crash (Force Index = 0 exactly) | `test_part3_zero_volume_is_pullback_absent_not_a_crash` | PASS |
| Duplicate tickers in a batch collapse to one result (dict semantics, not a bug) | `test_part3_duplicate_tickers_in_batch_collapse_to_one_result_per_ticker` | PASS |
| Empty ticker list: `run_batch` → `{}`, table → `""`, no crash | `test_part3_empty_batch_list_returns_empty_results_and_empty_table` | PASS |
| Output is always exactly one of the 4 `Signal` values, never an exception, over 60 random valid series | `test_part4_output_always_one_of_four_signals_never_exception` (hypothesis) | PASS |
| Determinism: identical input → identical output, over 60 random cases | `test_part4_determinism_same_input_same_output` (hypothesis) | PASS |
| No-lookahead: shared-prefix series with diverging tails give identical truncated verdicts, over 60 random cases | `test_part4_no_lookahead_shared_prefix_gives_identical_verdict` (hypothesis) | PASS |
| Direction-permission invariants hold over 60 randomly generated (trend, pullback, trigger, position) combos, not just hand-picked cases | `test_part4_direction_permission_invariant_holds_for_generated_inputs` (hypothesis) | PASS |

---

## Robustness gaps (documented, not classified as defects — no contract was ever stated)

- **No OHLC invariant validation.** `test_part3_ohlc_invariant_violation_high_below_low_silently_accepted` (PASS — documents current behavior): a bar with `High < Low` is accepted without error; the trigger screen just compares whatever numbers it's given. No screen or `DataProvider` implementation has ever promised to validate this, so this isn't marked as a defect, but it's worth knowing: a corrupted feed produces no error and no warning here either, for the same underlying reason as defect #2.
- **No domain check on negative prices.** `test_part3_negative_prices_do_not_crash_and_direction_follows_the_math` (PASS): a monotonically-rising-but-negative price series reads `UP`, arithmetically correct, no crash — documented as an absence of a sanity check, not a "should."
- **String-dtype columns behave inconsistently but not dangerously** (explored, not included as a formal test): a numeric-string `Close` column is silently coerced correctly by `.ewm()` and by `prior_bar_breakout`'s explicit `float()` cast; `force_index_pullback`'s `.diff()` on strings raises `TypeError` (fails loudly). Inconsistent, but neither path produces a wrong answer, so not classified as a defect.

---

## Verdict per area

| Area | Verdict |
|---|---|
| **Part 1 — main logic** | **PROVEN CORRECT.** Independently re-derived the 24-row truth table from Elder's rules (not copied from `engine.py`'s docstring) and it matches the implementation exactly, plus 4 dedicated load-bearing-rule tests, plus a hypothesis property test over 60 random combos. 29/29 pass. |
| **Part 2 — corner cases** | **PROVEN CORRECT.** All 9 corner-case tests pass: exact-equality boundaries on both `<`/`>` comparisons, exact min-viable-data boundaries (16 bars / 2 bars), trend-transition truncation safety, degenerate series, and cross-timeframe alignment gaps all degrade exactly as expected, no crashes. |
| **Part 3 — invalid/broken input** | **NOW PROVEN CORRECT — 11/11 pass.** 8/11 always passed (empty data, `None`, missing column, zero volume, negative prices, OHLC violations, duplicate tickers, empty batch all degrade or fail loudly as expected). The 3 confirmed defects above (reverse-order bars, `inf` in High/Low, `run_batch` batch-abort) are now fixed and their tests pass without the `xfail` marker. |
| **Part 4 — invariants** | **PROVEN CORRECT** for everything the current design can be asked to prove: determinism, always-a-valid-signal, no-lookahead (shared-prefix invariance), screen independence, and direction-permission all hold over both hand-picked and randomly generated inputs. 5/5 pass. Note: the property test proves *truncation* safety (the only form of "lookahead" this code's pure-function design can even exhibit) — it does not and cannot speak to the separate, already-known issue that a live "weekly" fetch's *current* bar is itself a moving target intraweek (flagged in the prior review, not re-litigated here since it's a data-provider concern, not something these functions could detect from their own inputs).

---

## What was fixed

All three defects are fixed in `research/triple_screen/indicators.py` (defects 1 & 2) and
`research/triple_screen/batch.py` (defect 3). Each fix's test previously ran `xfail(strict=True)`
against the old code (captured in the first pytest run of this file, tail of this repo's history)
and now runs as a plain passing assertion against the new code — the "fails on old / passes on
new" pair requested, both halves demonstrated. No other implementation file was touched; the
robustness gaps documented above (no OHLC-invariant validation, no domain check on negative
prices) were left as-is since they were never classified as defects against any stated contract.

---

## Addendum 2026-09 — Screen 3 now requires true-breakout confirmation

Elder's book (_Come Into My Trading Room_) is explicit that a breakout must be confirmed by
heavy volume and by indicators reaching new extremes in the trend direction (divergence marks a
false breakout) — the pre-addendum `prior_bar_breakout` only checked the bare price cross. Fixed
by adding two confirmation checks (`indicators.volume_confirms_breakout`,
`indicators.price_momentum_new_extreme`), both required alongside the price cross for
`triggered=True`:

- **Volume confirmation**: today's volume > 1.5x the average of the 20 bars strictly before
  today (today's own volume never inflates its own baseline).
- **Indicator-extreme confirmation**: a 10-bar price Rate-of-Change makes a fresh 20-bar extreme
  in the trend direction. If price makes a new extreme but momentum doesn't, that mismatch IS
  the price/indicator divergence the false-breakout warning describes.

**A real design conflict was found and resolved before landing this, via the same "prove it"
discipline as the defects above.** The first design used a longer-period (13-bar) Force Index
for the extreme check — Elder's own dual-Force-Index convention, alongside the 2-bar Force Index
Screen 2 already uses for its pullback. That turned out to be mathematically incompatible with
Screen 2 on the shared latest bar: the raw value needed today to push a 13-bar EMA to a fresh
extreme is always ~2.6x bigger than the ceiling that keeps a 2-bar EMA of the *same underlying
series* negative — a fixed ratio set by the two EMA spans, not fixable by tuning the breakout's
size. Verified three ways before abandoning it: hand-derived the ratio, 20,000 randomized
synthetic bar sequences (0 satisfied both simultaneously), and a live check against 18 real,
volatile tickers' current data (0 satisfied both). Switched to a plain price-momentum indicator
(no shared formula with Screen 2) instead, which resolved it — verified via
`triple_screen_fixtures.buy_ready_entry_bars`, a constructed-and-checked fixture where Screen 2's
pullback and Screen 3's confirmed trigger both fire on the identical bar.

New minimum viable data for Screen 3: 30 bars (`max(volume_lookback + 1, momentum_period +
momentum_lookback)` = `max(21, 30)`), up from 2. Tests: `tests/test_triple_screen_screens.py`
(confirmed/light-volume/divergent breakout cases, both directions),
`tests/test_triple_screen_stress.py` (`test_part2_minimum_viable_trigger_confirmation_data_boundary_is_30_bars`,
`test_part2_trigger_price_cross_alone_at_two_bars_no_longer_fires`), `tests/test_triple_screen_batch.py`
and `tests/test_triple_screen_tracker_service.py` (both now use `buy_ready_entry_bars` for a real
end-to-end BUY). Full suite: 836 + new Screen-3 tests, 0 failures (see git history for the exact
count at the commit that lands this).
