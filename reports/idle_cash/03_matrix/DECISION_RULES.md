# Phase 3 decision rules (pre-registered)

Written and committed **before** any Phase 3 matrix result exists. `summary.md` applies these rules mechanically. Nothing below may be changed after results are seen. Any deviation will be reported as a deviation, not folded in.

## The rules (copied from the execution plan)

> **Keep the swing layer only if all are true, after costs:**
> 1. Phase 1 mean excess after costs is above zero.
> 2. B beats D **or** C beats E on the stitched out-of-sample curve, with a block-bootstrap 95% CI of the daily return difference above zero at every block length tested.
> 3. Regressing the chosen variant's daily returns on XIU's gives a positive intercept (alpha), Newey-West t > 2.
> 4. The result holds in the adverse-cost scenario (spreads/slippage doubled).
>
> **Otherwise:** the recommendation is to hold XIU (or E, if E beats D on Calmar and max drawdown) and keep the swing system as paper research only.

## How each rule is evaluated

**Rule 1.** Uses the Phase 1 primary sample: backtest, all trades (closed plus open marked to market), realistic costs. That figure is already known: mean excess **−0.41%** (95% CI −1.15 … +0.33; `01_trade_vs_xiu/summary.md`). **Rule 1 therefore already fails**, so the verdict cannot be "keep" whatever the matrix shows. The matrix still runs:
- Rules 2–4 are reported for the record.
- The "otherwise" branch needs the E-vs-D comparison.
- The attribution shows where the money comes from.

**Rule 2.**
- **Inputs:** the stitched daily return series of B and D, and of C and E, under realistic costs.
- **Difference:** d_t = r_B,t − r_D,t (resp. C − E).
- **Bootstrap:** circular moving-block bootstrap of the mean of d_t. Block lengths 5, 10 and 20 days; 10,000 resamples each; fixed seed 20261027; 95% percentile CI.
- **Pass:** "B beats D" passes only if the CI lower bound is > 0 at all three block lengths; "C beats E" likewise.
- Rule 2 passes if either comparison passes. A passing comparison makes its strategy variant (B or C) a **candidate**.

**Rule 3.**
- **Model:** for each candidate, OLS of daily excess returns on XIU's daily excess returns: (r_v − r_cash) = α + β (r_XIU − r_cash) + ε. r_cash is CASH.TO's daily total return (Jensen's alpha).
- **Why excess returns:** a raw-return intercept would credit a low-beta variant with (1 − β) × the cash rate, which is not skill.
- **Standard errors:** Newey-West HAC with Bartlett weights and lag L = floor(4 (T/100)^(2/9)).
- **Pass:** α > 0 and t(α) > 2.
- **Chosen variant:** the candidate that passes. If both pass, the one with the higher t(α). Raw-return regressions are reported as a secondary table only.

**Rule 4.** The chosen variant must pass rules 2 and 3 again with every run, strategy and benchmarks alike, re-done under adverse costs (stock 20 bps per side, XIU 4 bps per side). Rule 1's adverse figure is −0.61% (CI −1.34 … +0.12), so it fails there too.

**Otherwise branch (E vs D).** Under realistic costs, the recommendation is E if E's stitched curve has **both** a higher Calmar ratio **and** a smaller maximum drawdown than D's. Otherwise it is D (hold XIU).

## Fixed setup (unchanged from Phases 1–2)

**Folds.** The same 16 non-overlapping ~62-day folds over 2022-10-03 → 2026-10-02, each starting from a fresh $50,000.
- Each fold simulates its own day list, with the last day of each fold except the final one skipped (inherited).
- The stitched curve chains each fold's daily returns in date order. A fold's first-day return is measured against its $50,000 start.
- The folds don't overlap, so no segments need dropping.

**Strategy settings.** Unchanged: the Phase 2 all-fixes configuration, i.e. costs, CASH.TO cash yield, the conservative same-bar rule and gap fills. The swing signals, entry gate (`regime_filter`), sizing, slots and exits are identical in A, B and C.

**Cost scenarios:**

| Scenario | Stock spread per side | XIU spread per side | Commission |
|---|---|---|---|
| realistic | 10 bps | 2 bps | $0 |
| adverse | 20 bps | 4 bps | $0 |
| zero (diagnostic only, never used by a rule) | 0 | 0 | $0 |

## Variant mechanics

**IDLE_CASH_MODE** (`BacktestConfig.idle_cash_mode`): `cash` is A (the default and current behaviour); `xiu_always` is B; `xiu_gated` is C.

**Shadow cash ledger.** The swing book is sized from the existing `PortfolioState.cash`, which behaves exactly as in A, including A's CASH.TO yield. So under each cost scenario, A, B and C size and select the same stock trades.
- The real account (actual cash plus whole XIU shares) is a separate ledger.
- XIU gains and losses never reach the sizing base.
- If the real account can't fund a buy, even after selling all its XIU, that buy is skipped (no margin; actual cash never goes negative). Every such skip is counted and reported.

**Funding (morning, at the stock-entry fill: D's open).** The morning's stock buys are netted into one XIU order at the open, with buffer = 1% and threshold = 3% of the prior close's real equity:
- If the park gate is closed, sell all XIU.
- Otherwise, if cash minus today's stock purchases is negative, sell just enough whole shares to cover the shortfall plus the buffer.
- Otherwise, if cash minus purchases minus the buffer exceeds the threshold, buy XIU with that excess.

**Exits (close, at the stock-exit fill: D's close).** Sale proceeds go to cash. If the park gate is open and cash minus the buffer exceeds 3% of real equity, the excess buys XIU in one order at the close.

**Real cash.** Whatever is left earns CASH.TO's daily total return on the closing balance, exactly like A's cash.

**Park gate.**
- **B:** always open.
- **C:** open when XIU's prior completed close is ≥ its 200-day SMA, the same test as the entry gate (`_is_market_in_uptrend` on D−1). It is evaluated each morning and acted on at that morning's open.
- The entry gate stays exactly as it is in all variants.

**XIU fills.** At the bar's open or close, moved against the trade by the XIU spread; the commission is paid per XIU order. Prices come from the same adjusted (total-return) cache as the stocks.

**Benchmarks.** Same fold days, $50,000 per fold, whole XIU shares, and leftover cash earning CASH.TO yield:
- **D: buy-and-hold.** Buy XIU at the fold's first open with all cash.
- **E: gated.** Hold XIU when the prior close is ≥ its SMA200, else cash. Switch at the open.
- **F: 55/45 mix (descriptive only).** 55% XIU, 45% cash. Bought at the first open and rebalanced back to 55% at the first open of each calendar month.

## Metrics (all on stitched curves)

**Return and risk:**
- Net CAGR, using 252 trading days per year.
- Maximum drawdown.
- Sharpe and Sortino, on daily returns in excess of CASH.TO, annualized by √252.
- Calmar = CAGR / |max DD|.
- Ulcer Index.

**Activity and costs:**
- Average exposure: (stocks + XIU) / equity.
- Turnover: traded notional ÷ average equity, per year.
- Total costs.

**Paired comparisons** (B−D, C−E, E−D, B−A, C−B):
- Annualized mean difference, tracking error and information ratio.
- The block bootstrap above.
- Per-fold win counts, as a secondary diagnostic.

**Alpha regressions.** Jensen's alpha and beta, with Newey-West t, for A, B and C.

**Attribution for B and C.** Cumulative dollars split into:
- XIU sleeve P&L, gross of spread.
- Stock P&L, split into an XIU-matched part (stock value at the prior close × XIU's close-to-close return, plus today's buys × XIU's open-to-close return) and stock excess over XIU (the remainder).
- Cash yield.
- Costs.
