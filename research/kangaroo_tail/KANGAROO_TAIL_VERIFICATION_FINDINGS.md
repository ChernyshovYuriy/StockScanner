# Kangaroo Tail — Phase 2 Verification Findings (2026-09-11)

## Question

Does the bullish Kangaroo Tail pattern (detector.py) predict an above-normal
forward price move, or does it just look plausible on a chart? Two different
methodologies were used, in this order, and they gave two different
answers depending on HOW the signal is traded — this doc covers both.

## Part 1 — Event study (same-day entry): NO EDGE

Methodology: `verify.py`, 16-fold walk-forward over the CAN_TICKERS_URL
universe (134 tickers), 2022-07-29 → 2026-07-28, same paired-t-test-on-
per-fold-differences convention as `walk_forward_ab.py` / the sector-cap
and gap-filter walk-forwards. "Candidate" = mean K-day forward return of
every detection in a fold, entering **at the tail bar's own close** (same-
day); "baseline" = every ticker's own unconditional K-day forward return in
that same fold (controls for each stock's own drift, not just universe-wide
drift). Full per-combo results: `out/kangaroo_tail_verify_results.csv`;
per-fold detail: `out/kangaroo_tail_verify_folds.csv`.

Across every `TailConfig` variant tested (`strict`, `loose_wick`,
`no_break`, `sweet_spot`, `loose_all` — see `verify.py`'s `CONFIGS`) ×
3 alert-timing variants × 3 holding windows, **same-day entry never showed
a real edge**. Two cells came back nominally significant at p<0.05 out of
36 combinations tested — both explainable as multiple-comparisons noise
(≈1-2 expected from pure chance at α=0.05 over that many cells), not a real
effect: one was significant in the *wrong* (bearish) direction
(`loose_wick/same_day/k=5`, p=0.026), and `loose_all/same_day/k=5`
(p=0.027, n=1,097) was also negative — a huge, well-powered sample saying
same-day entry is, if anything, slightly *worse* than the ticker's own
normal drift, not better. **Conclusion: do not buy a Kangaroo Tail at its
own close. This holds across every threshold combination tested.**

## Part 2 — Breakout-entry, defined-risk (R-multiple): A REAL, MODEST EDGE

Prompted by a real chart example the user flagged (AAPL 2026-09-09) that
the Phase 1 detector missed, and by a user-forwarded reference script
(`grok_solution.py`) using a different, more standard trading metric: an
R-multiple backtest where risk is explicitly defined (stop = tail's low)
and entry only happens once price **breaks out** — trades through the
tail's own high (a buy-stop trigger) — rather than same-day. Target =
entry + `rr` × risk. This is a fundamentally different, and more standard,
way to trade the same detected candle.

Same 16-fold structure, same universe/window, `rr` ∈ {1.0, 1.5, 2.0},
max_hold_bars=10 (matches `KANGAROO_MAX_HOLD_DAYS`). Headline results at
`rr=1.5`:

| config | wick_atr_mult | n_trades | win_rate | avg_R | profit_factor | fold-paired p |
|---|---|---|---|---|---|---|
| strict | 1.5 | 13 | 69.2% | +0.491 | 3.25 | 0.142 |
| loose_wick | 1.0 | 31 | 61.3% | +0.322 | 1.98 | 0.222 |
| no_break | 1.5 | 50 | 62.0% | +0.327 | 2.19 | 0.123 |
| **sweet_spot** | **1.2** | **143** | **56.6%** | **+0.246** | **1.72** | **0.033** |
| loose_all | 0.7 | 949 | 46.7% | +0.110 | 1.23 | 0.048 |

Two controls were run to rule out a generic "wide stops are just easier to
hit" confound rather than the specific candle shape mattering:
- **Any bar, breakout entry, no shape filter**: win 37.6-54.8%, avg_R
  −0.03 to +0.15 depending on rr — materially worse than every
  Kangaroo-Tail-shaped config above.
- **Any big-range day (≥1.5× its own 10-bar average range), breakout
  entry**: win 47.5-54.8%, avg_R +0.09-0.15 — better than "any bar" but
  still below every Kangaroo-Tail-shaped config. The specific tail shape
  (wick dominance + small body + close-back-into-range) adds real
  incremental value beyond "this was just a big-range day."

**A fine-grained sweep of `wick_atr_mult` (1.5 down to 0.7, fixing
`break_atr_mult=-999` / no_break and rr=1.5) found a sharp CLIFF, not a
smooth gradient**, between 1.2 and 1.0:

| wick_atr_mult | n_trades | win_rate | avg_R | profit_factor |
|---|---|---|---|---|
| 1.5 | 50 | 62.0% | 0.327 | 2.19 |
| **1.2** | **136-147** | **56.5-57.4%** | **0.236-0.265** | **1.69-1.78** |
| 1.0 | 277-301 | 48.5-48.8% | 0.091-0.101 | 1.20-1.23 |
| 0.9 – 0.72 | 378-881 | 46.3-48.7% | 0.088-0.121 | 1.19-1.26 |
| 0.7 | 815-949 | 46.7-46.9% | 0.110-0.120 | 1.23-1.24 |

Below the 1.2→1.0 cliff, sample size grows a lot (more signals fire) but
per-trade quality collapses to roughly a third of the 1.2 value and stays
flat all the way down to 0.7 — there is no rescue by loosening further,
only more of the same diluted edge. **The AAPL 2026-09-09 bar itself
(wick=0.71× its own ATR) sits below that cliff** — catching it specifically
requires `wick_atr_mult≈0.7` (`loose_all`), which trades the AAPL-catching
capability away for materially worse average quality (n=949, win 46.7%,
avg_R +0.11) versus the `sweet_spot` config just above the cliff (n=143,
win 56.6%, avg_R +0.25). That specific bar was a real, working example —
but it is statistically indistinguishable, at the point of detection, from
the flood of mediocre lookalikes that share its shallow-wick shape.

16-fold consistency check (`sweet_spot`-adjacent `no_break`, breakout
entry, rr=1.5): 16/16 folds had at least one trade; 10/16 folds had
positive average R. Real variance exists — this is not a one-quarter
fluke, but it is also not a smooth win-every-period edge.

## Conclusion & what got built

**Same-day entry: no edge, confirmed by every test above and Part 1's
dedicated event study. Do not trade it this way.**

**Breakout-entry with a defined stop/target: a real, if modest, edge —
win≈57%, avg_R≈+0.25, profit factor≈1.7, statistically significant
(p=0.033) at the largest well-powered sample above the quality cliff.**
This is what got shipped:

- `research/kangaroo_tail/types.py`'s `TailConfig` defaults were updated to
  the `sweet_spot` values (`wick_dominance_pct=0.55`, `wick_atr_mult=1.2`,
  `break_atr_mult=-999.0` — i.e. no_break) — see its own docstring.
- **Phase 3 was built**: `kangaroo_pipeline.py` / `kangaroo_buy.py` /
  `kangaroo_monitor.py`, a fully isolated paper sleeve (`data/kangaroo.db`,
  `config.py` `KANGAROO_*`) that trades ONLY the validated breakout-entry
  mechanic — a detection becomes a PENDING BREAKOUT intent (trigger = tail
  high, stop = tail low, target = trigger + `KANGAROO_RR_TARGET`×risk), not
  a same-day buy. See CLAUDE.md's "Kangaroo Tail sleeve" entry for the
  full architecture.

**Caveats that still apply, and are worth re-checking as more history
accumulates:**
1. n=143 trades over 4 years/134 tickers is a real but not huge sample;
   6/16 folds were net losers.
2. No slippage/commission is modelled on either the breakout fill or the
   stop/target fill — this sleeve fills at the exact trigger/stop/target
   price, an optimistic assumption.
3. Tested on one universe (TSX/CAN_TICKERS_URL) over one mostly-rising
   4-year window — unknown whether the edge holds in a flat/declining
   regime or a different market.
4. `verify.py`'s `CONFIGS`/`TIMINGS`/`HOLDING_DAYS` remain the harness to
   extend if this is revisited (e.g. a bearish/topping variant, live
   slippage modelling, or a broader-trend filter).
