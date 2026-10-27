# Phase 5: Entry slices vs XIU

**Question:** Phases 1 and 3 found no edge for the entries as a whole. Is there any slice of them (setup type, stop width, market regime, sector) that beats XIU after costs?

**Answer: no.** None of the 34 slices with ≥10 trades has a 95% CI above zero, in either the backtest or the live sample. **No new entry rule is proposed for implementation.** The hypotheses below are for you to choose from, or to reject.

## Method

**Data:** the Phase 1 trade table (`01_trade_vs_xiu/trades_with_excess.csv`), using its primary "all" sample: closed trades plus positions still open at fold end, marked to market.
- Backtest: 364 trades. Live: 52 trades.
- Metric: excess over XIU after realistic costs, as a % of entry notional.
- Script: `analysis/phase5_entries.py`.

**Slices:**
- **Setup type:** PB-EMA21, PB-EMA50, BASE, VCP.
- **Stop width:** <6%, 6–8%, >8% below entry.
- **Regime:** the plan's split (XIU above/below its 200-day at entry) can't be tested, because the entry gate only admits "above". The nearest testable proxy, XIU above/below its **50-day** at the prior close, was used instead.
- **Sector:**
  - Backtest: 179 of 181 tickers mapped via `analysis/fetch_backtest_sectors.py`, a paced Yahoo fetch (3 s per request, no rate limiting) saved to `ref/backtest_sectors.json`.
  - Live: `ref/live_sectors.json`.

**Statistics:**
- Bootstrap 95% CI, 10,000 resamples of trades.
- A Bonferroni-adjusted CI for the number of slices tested with ≥10 trades: **17 backtest, 7 live (24 in total)**. At 5% each, about 1 of 24 would look significant by chance alone.
- A slice is flagged only if its plain 95% CI is above zero and it has ≥10 trades. **None is flagged.**

Full table: `slices.csv`.

## Ranking (backtest, realistic costs; slices with ≥10 trades)

| Dimension | Slice | Trades | Mean excess | 95% CI | Live: trades, mean |
|---|---|---|---|---|---|
| Sector | Basic Materials | 77 | **+0.99%** | −0.76 … +2.74 | 6, +0.62% |
| Sector | Financial Services | 67 | +0.81% | −0.53 … +2.23 | 10, **−1.73%** |
| Sector | Technology | 20 | +0.33% | −3.97 … +5.26 | 4, +0.72% |
| Stop width | >8% | 108 | +0.01% | −1.50 … +1.52 | 14, −0.15% |
| Sector | Consumer Cyclical | 24 | −0.17% | −2.67 … +2.26 | 5, −5.03% |
| Setup | PB-EMA21 | 224 | −0.25% | −1.22 … +0.73 | 42, −0.82% |
| Regime | XIU above 50d | 290 | −0.30% | −1.18 … +0.56 | 44, −1.46% |
| Setup | BASE | 82 | −0.41% | −1.80 … +1.09 | 9, −0.85% |
| Stop width | 6–8% | 138 | −0.50% | −1.64 … +0.68 | 20, −0.00% |
| Sector | Utilities | 12 | −0.65% | −4.07 … +2.55 | 3, −1.89% |
| Stop width | <6% | 118 | −0.69% | −1.84 … +0.46 | 18, **−1.94%** |
| Regime | XIU below 50d | 74 | −0.84% | −2.03 … +0.37 | 8, **+3.42%** |
| Setup | PB-EMA50 | 56 | −0.93% | −2.61 … +0.80 | 1, — |
| Sector | Real Estate | 10 | −1.11% | −4.19 … +2.11 | 4, +0.25% |
| Sector | Energy | 66 | −1.23% | −2.96 … +0.60 | 12, +1.99% |
| Sector | Industrials | 53 | −1.85% | −3.64 … +0.01 | 6, −2.17% |
| Sector | Consumer Defensive | 18 | −2.39% | **−4.90 … −0.04** | 0 |

Too small to rank (<10 trades): VCP (2), Healthcare (6), Communication Services (9).

**The best slice is not good enough.** Even if Basic Materials' +0.99% per trade held, the strategy is ~50% invested with a beta of ~0.3. Adding about +1% per trade to the stock side would not close the ~14 points a year by which A trails D (Phase 3).

## Candidate hypotheses (for the user to choose from; none implemented)

1. **Avoid tight stops (<6%).** This slice is negative in both samples: backtest −0.69% (n=118), live −1.94% (n=18).
   - The wider buckets are about zero.
   - Weak: the CI spans zero in both samples. A tight stop may only stand in for low-volatility names, which have less upside to give.
   - Test: a forward paper split, or a pre-registered walk-forward with `min_stop_pct` as a single fixed value.
2. **Avoid Consumer Defensive and Industrials.** These are the two worst backtest sectors: −2.39% (CI just below zero) and −1.85%.
   - Live has no Consumer Defensive trades. Industrials is −2.17% live, but with only 6 trades.
   - The Consumer Defensive CI loses significance after Bonferroni (−6.34 … +0.88), so this is likely the expected chance hit among 24 slices.
3. **Basic Materials tilt.** The best backtest sector (+0.99%), and positive in live too (+0.62%, n=6).
   - The window included a gold/silver rally (2025–26), so this is probably the window, not the setup.
   - The momentum sleeve already targets that kind of move.
4. **Not worth pursuing:**
   - The 50-day regime proxy points opposite ways in the two samples (backtest below 50d −0.84%, live +3.42%).
   - Financial Services points opposite ways too (backtest +0.81%, live −1.73%).
   - Setup type: no pattern is positive in either sample.

**Bottom line:** none of these is strong enough to turn the swing layer into an edge over XIU. Testing any of them should be pre-registered, in the same way Phase 3 was, before a single result is seen.
