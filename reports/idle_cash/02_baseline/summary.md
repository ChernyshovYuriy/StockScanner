# Phase 2: Backtester baseline fixes

**Setup:** the unchanged strategy, the same 16 point-in-time folds (2022-10-03 → 2026-10-02), $50k per fold with no compounding across folds, and the same screener snapshots. Each fix is switched on alone, then all together.
- The `off` variant reproduces Phase 1 trade-for-trade (254 closed trades, identical).
- Scripts: `analysis/run_walk_forward_trades.py <variants>` and `analysis/phase2_baseline.py`.
- Full tables: `metrics.csv` and `exit_mix.csv`.

## Switches added (`BacktestConfig`; every default is the old behaviour)

| Switch | Behaviour when on |
|---|---|
| `costs_enabled` | Fill price moves against the trade by `stock_spread_bps` (10 bps per side) on every buy and sell; `commission_per_order` ($0, per the user's decision) is paid from cash. |
| `cash_yield_source="cash_etf"` | Each day's closing cash earns CASH.TO's daily total return (`ref/xiu_cash_daily.csv`). |
| `same_bar_rule="conservative"` | The stop is a resting order at the level computed from bars before today, so today's high can't arm or raise it. It fills first when today's low reaches it; the close-based rules (giveback, time stop, close below the stop) apply only if it didn't. |
| `gap_fills` | Conservative rule only: an open below the resting stop fills at the open, not at the stop. |

**No price-basis switch.** It would have nothing to change: stocks, XIU and the benchmark already share one auto-adjusted total-return basis. Evidence: a fresh adjusted download matches the cache to 0.0 pp over the 4-year window for XIU and for all 19 of the 20 most-traded backtest tickers that Yahoo still serves (`ref/price_basis_check.csv`). The Phase 0 cache-drift concern does not materialise for this window.

## Results

| Variant | Compounded | Mean fold | Positive folds | Mean max DD | Worst DD | Win rate | Avg exit P&L | Costs | Cash yield |
|---|---|---|---|---|---|---|---|---|---|
| off (Phase 1) | **+24.6%** | +1.47% | 10/16 | −3.71% | −6.15% | 40.2% | −0.45% | $0 | $0 |
| costs | +12.3% | +0.82% | 9/16 | −4.06% | −6.29% | 38.8% | −0.75% | $2,560 | $0 |
| yield | +33.7% | +1.92% | 10/16 | −3.61% | −5.87% | 40.2% | −0.45% | $0 | $3,566 |
| conservative | +30.5% | +1.76% | 9/16 | −3.35% | −7.48% | 38.0% | −0.23% | $0 | $0 |
| conservative + gap fills | +28.0% | +1.64% | 9/16 | −3.43% | −7.53% | 38.0% | −0.32% | $0 | $0 |
| **all_on** | **+25.1%** | +1.49% | 8/16 | −3.55% | −7.40% | 37.1% | −0.56% | $2,693 | $3,635 |

**Per-fold comparison with `off`** (paired t-test on the fold return differences):

| Variant | Return wins | Mean Δ | p | Drawdown better | Mean Δ DD |
|---|---|---|---|---|---|
| costs | 0/16 | −0.66 pp | 0.015 | 1/16 | −0.35 pp |
| yield | 16/16 | +0.44 pp | <0.001 | 16/16 | +0.10 pp |
| conservative | 13/16 | +0.29 pp | 0.59 | 10/16 | +0.36 pp |
| conservative + gap | 13/16 | +0.17 pp | 0.76 | 9/16 | +0.28 pp |
| all_on | 10/16 | +0.02 pp | 0.97 | 9/16 | +0.16 pp |

## What each fix removed or added

- **Costs removed half the result:** compounded +24.6% → +12.3% (−12.3 pp; equity gain −$5,224). The direct spread was $2,560.
  - The rest is path dependence. 246 trades whose exits didn't change lost $2,188 to the spread; 6 trades exited on different days and lost $1,025 between them; the remaining differences come from a slightly different trade set (2 trades dropped, 6 added) and fold-end marks.
  - The 0% time stop is knife-edge. A 10 bps higher entry left EXE.TO and SGR-UN.TO at −0.1% on day 20 instead of break-even, so the time stop fired and cut two winners (+$740 and +$343 without costs).
  - Costs never helped a fold (0/16, p=0.015).
- **Cash yield added +9.1 pp** (+$3,566), in every fold (16/16). This is pure accounting: the strategy holds ~40–50% cash, which previously earned nothing.
- **Conservative same-bar rule: "exits look fine" survives.** It doesn't make exits worse: compounded +30.5% vs +24.6%; +28.0% with gap fills.
  - The resting stop fires more often (128 exits vs 93 close-triggered stop exits), because an intraday touch that recovers by the close now counts. But each exit loses less (mean −3.67% vs −6.19%): filling at the stop level beats waiting for a close that is often well below it.
  - Gap fills give back 2.5 pp (17 of 128 resting-stop exits gapped below the stop).
  - The improvement isn't statistically significant (13/16 folds, p=0.59), and the worst fold drawdown is deeper (−7.5% vs −6.2%).
- **All fixes together:** +25.1%, essentially unchanged from +24.6%. The costs (−12.3 pp) roughly cancel the cash yield (+9.1 pp) plus the stop-fill effect (+3.3 pp with gap fills). This is the realistic baseline for Phase 3 (variant A).

## Observations (not acted on, per the "don't tune" rule)

1. **Resting stops would have done better than the close trigger.** That contradicts the 2026-07 stop-trigger A/B, which rejected a low trigger. That test filled low-trigger exits at the close rather than at the stop level, so it measured a different mechanic. The live paper system has no resting orders; it sells at the 15:50 price. **This is a hypothesis for the user, not a recommendation.** It's not significant here and was found while testing something else.
2. **The time stop's 0% threshold is sensitive to costs.** Trades sitting near break-even on day 20 flip between being held and being cut on a 10 bps difference.

## Limitations

- **Fills:** resting-stop fills are exactly at the stop level (plus the spread under `costs`); real stop-market orders can slip further on fast moves. Exits decided on the close still fill at that same close (accepted; 00_recon §3).
- **Inherited from the Phase 1 baseline:** no sector cap, `top_n_buys=3`, each fold skips its last trading day, `TSX_HOLIDAYS` covers only 2026–2027, and survivorship bias from today's raw ticker list.
- **Cash yield data:** CASH.TO has one bad bar (2023-05-03, −0.78%, reversed the next day: net ≈ 0). It's left as is.
- **Over-tested window:** 2022-10 → 2026-10 has been backtested many times.
