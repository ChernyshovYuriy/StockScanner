# Phase 1: Per-trade excess over XIU

**Question:** for each trade, did the money do better in the stock than it would have in XIU.TO over the same days?

**Formula:**
```
excess = trade_pnl − entry_notional × (XIU_close(exit) / XIU_open(entry) − 1) − costs
```

**Basis:**
- **Total return on both sides.** XIU comes from one fresh auto-adjusted download. Live trades add the stock's dividends during the hold. Backtest trades use the cache's adjusted bars; over the window, cached XIU matches the fresh download exactly (+103.1%).
- **Costs:** commission $0 (a parameter, per the user's decision); spread/slippage per side on the stock notional of 0 / 10 / 20 bps (zero / realistic / adverse).
- **Excess %** is excess divided by entry notional. The CI is a 10,000-resample bootstrap of the per-trade mean.

## Sample correction: open positions must be included

A closed-trades-only sample is biased against the strategy:
- In the walk-forward, the 16 folds' equity grew **+$11,754**, but closed trades summed to **−$4,655**. The other **+$16,409** is unrealized gains on the 110 positions still open when folds ended.
- Losers exit fast on stops, while winners are still running at the fold boundary (folds are ~65 trading days; average hold is 22 calendar days).
- Live trading has the same effect: its 7 open positions include SOIL.TO at +42%.

**"All" (closed plus open positions marked to market) is the primary sample**, since it reconciles exactly to equity: backtest gross P&L $11,754 = the sum of fold equity gains. Closed-only figures are shown for reference.

## Results

| Source | Sample | Trades | Gross P&L | XIU-matched P&L | Mean excess (realistic) | 95% CI | Beat XIU |
|---|---|---|---|---|---|---|---|
| Backtest | **all** | 364 | +$11,754 | +$16,451 | **−0.41%** | **−1.15% … +0.33%** | 45% |
| Backtest | closed | 254 | −$4,655 | +$9,008 | −1.44% | −2.38% … −0.51% | 41% |
| Live | **all** | 52 | +$303 | +$389 | **−0.71%** | **−2.74% … +1.60%** | 37% |
| Live | closed | 45 | −$123 | +$295 | −1.59% | −3.29% … +0.12% | 36% |

Cost sensitivity (backtest, all): zero costs −0.21% (CI −0.93 … +0.53), realistic −0.41%, adverse −0.61% (CI −1.34 … +0.12). The CI includes zero in all three scenarios.

Total excess (backtest, all, realistic): **−$7,739** across 16 folds × $50k. Excess was positive in 6 of 16 folds; folds 11–13 (2025-03 → 2025-12) were the only clearly positive stretch.

Concentration: the 10 best trades account for 86% of the backtest's gross P&L.

## Breakdowns (all sample, realistic costs)

**Setup type.** No slice has a CI above zero.

| Setup | Backtest n | Mean | 95% CI | Live n | Mean | 95% CI |
|---|---|---|---|---|---|---|
| PB-EMA21 | 224 | −0.25% | −1.21 … +0.74 | 42 | −0.82% | −3.18 … +1.94 |
| BASE | 82 | −0.41% | −1.81 … +1.08 | 9 | −0.85% | −3.64 … +2.26 |
| PB-EMA50 | 56 | −0.93% | −2.63 … +0.83 | 1 | — | — |
| VCP | 2 | −3.53% | — | 0 | — | — |

**Stop width.** <6%, 6–8% and >8% are all indistinguishable from zero (backtest −0.69%, −0.50%, +0.01%).

**Exit reason.** These slices are largely set by the rules themselves (a giveback can only fire after a ≥6% gain; a stop exit is a loss by construction), so they describe the exits rather than test them.

| Exit | Backtest n | Mean excess |
|---|---|---|
| GIVEBACK | 92 | +5.5% |
| Open at fold end | 110 | +2.0% |
| TIME_STOP | 69 | −3.5% |
| STOP_HIT | 93 | −6.8% |

**Regime.** Every trade in both samples was entered with XIU above its 200-day average (the entry gate), so the regime split can't be tested here.

**Sector.** Live only (backtest has no offline sector map). Every sector CI includes zero except Consumer Cyclical and Utilities on closed live trades (n = 4 and 3). These are too small to mean anything.

## Decision gate

| Plan criterion | Result |
|---|---|
| Clearly negative (CI entirely below zero after costs) | **No** for the primary sample: backtest −1.15 … +0.33, live −2.74 … +1.60. Yes only for the biased closed-only backtest sample. |
| One setup type clearly positive | **No.** No setup has a CI above zero. |
| Positive or inconclusive | **Inconclusive.** The point estimates are negative (the trades did ~0.4% worse than XIU per trade), but zero is inside the CI in every cost scenario. |

**Per the plan: inconclusive → recommend continuing to Phase 2.**

Phase 1 is a per-trade comparison only. The account-level gap to XIU (+24.6% vs about +103% over the window) is mostly idle cash earning nothing, not per-trade underperformance. That's what Phases 2–3 measure.

## Known limitations

- **Exit fills:** backtest exits decide and fill on the same close (accepted by the user as an approximation of the live 15:50 sell; 00_recon §3). Live XIU fills are approximated by XIU's open (for the ~09:45 entry) and close (for the ~15:50 exit).
- **Open positions are marked, not exited.** Their final outcome is unknown, and the fold boundary truncates them.
- **Not exactly live:** no sector cap, `top_n_buys=3`, and each fold skips its last trading day. All three are inherited from the baseline that reproduced +22% (this run: +24.6%, mean fold +1.47%).
- **Holidays:** `TSX_HOLIDAYS` only covers 2026–2027, so earlier holidays are simulated as trading days. A buy dated on a holiday still fills at the next real open.
- **Universe:** today's raw ticker list (`can_tickers_full`), screened point-in-time each week. Survivorship bias remains: names delisted before today are absent.
- **Live sample:** small (52 trades, 5 months), one regime, and SES.TO is excluded (no valid price since 2026-09-02).
- **Re-testing the same window:** 2022-10 → 2026-10 has been backtested many times, so this result can't confirm anything new by itself.

## Files

- `trades_with_excess.csv`: one row per trade or open position, both sources.
- `totals.csv`, `breakdowns.csv`: the tables above, including the closed-only sample.
- `*_excess_hist.png`, `*_cumulative_excess.png`: per-source charts.
- `backtest_trades.csv`, `backtest_open.csv`, `backtest_equity.csv`, `backtest_folds.csv`: from `analysis/run_walk_forward_trades.py`.
- Reference data in `../ref/`: XIU/CASH.TO daily bars, dividends, sectors, live marks.
