# Idle cash: final report

**Verdict: don't keep the swing layer as a way to invest money. Hold XIU.** Keep the swing system as paper research only.
- This is the outcome of the decision rules pre-registered in `03_matrix/DECISION_RULES.md` (committed in `76f2f06`, before any Phase 3 result existed).
- No rule was changed after results were seen.

**Answer to the original question ("what to do with idle cash"):**
- If the swing system keeps running, parking its idle cash in XIU instead of cash is clearly better: +8.9 points a year in the backtest, significant at every block length, 14/16 folds.
- But holding XIU outright did better still. With the idle cash parked in XIU, the swing layer becomes a costlier, more active way to hold ~84% of XIU's exposure, minus the stock-picking shortfall.

## Phase by phase

| Phase | Question | Result |
|---|---|---|
| 0 Recon | How does the backtest fill, and what's missing? | Exits decide and fill on the same close (accepted as the stand-in for the live 15:50 sell). No costs, no cash interest, no setup/stop/exit-reason in the backtest logs. Adjusted (total-return) prices for both stocks and XIU. |
| 1 Per-trade excess | Did each trade's money do better in the stock than in XIU over the same days? | **No edge.** Backtest −0.41% per trade (CI −1.15 … +0.33, n=364). Live −0.71% (CI −2.74 … +1.60, n=52). Inconclusive, but negative in both. |
| 2 Baseline fixes | How much of the old +24.6% survives realistic mechanics? | All of it, by coincidence. Costs −12.3 pp; cash yield +9.1 pp; conservative resting stops +5.9 pp (n.s.). With everything on: **+25.1%**. |
| 3 Matrix | Does the swing layer beat XIU once idle cash is put to work? | **No.** B (idle cash in XIU) +76.5% vs D (XIU) +104.6%. C (gated) +45.3% vs E (gated XIU) +63.2%. Every alpha t is negative. |
| 5 Entry slices | Is any setup, stop width, regime or sector positive? | **No.** 0 of 34 slices with ≥10 trades has a CI above zero. A few weak hypotheses are listed for you to choose from. |

## The numbers that decide it (realistic costs, stitched over 16 folds, 2022-10 → 2026-10)

| | Total | CAGR | Max DD | Sharpe | Alpha/yr (NW t) |
|---|---|---|---|---|---|
| A strategy, idle cash earns cash yield | +25.0% | 5.7% | −11.8% | 0.31 | −1.9% (−0.58) |
| B strategy, idle cash in XIU | +76.5% | 15.0% | −11.5% | 0.94 | −1.4% (−0.46) |
| C strategy, idle cash in XIU above 200d | +45.3% | 9.6% | −12.2% | 0.59 | −3.2% (−0.89) |
| **D XIU buy-and-hold** | **+104.6%** | **19.3%** | −11.6% | **1.22** | — |
| E XIU above 200d, else cash | +63.2% | 12.8% | −12.8% | 0.85 | — |

**Where B's money came from** (sum over 16 folds × $50k):

| Source | Amount |
|---|---|
| XIU sleeve | +$22.2k |
| Stocks' market exposure | +$15.2k |
| Stocks' excess over XIU | **−$4.2k** |
| Costs | −$3.4k |

## Decision rules

| Rule | Outcome |
|---|---|
| 1. Phase 1 excess after costs > 0 | Fail (−0.41%) |
| 2. B > D or C > E, with block-bootstrap CI > 0 at every block length | Fail (−3.7 and −2.9 pp/yr; CIs span zero) |
| 3. Alpha > 0 with NW t > 2 | Fail (no candidate; all t < 0) |
| 4. Holds under adverse costs | Fail |
| Otherwise: E if it beats D on both Calmar and max DD | E loses on both → **D** |

## What was not tested, and why

- **Sizing redesign and more slots** (higher `MAX_POSITIONS`, more risk per trade, equity-based sizing with a heat cap). These are out of scope per the plan:
  - More slots make idle cash worse when signals are scarce.
  - Higher risk per trade was already rejected (drawdown worse in 16/16 folds).
  - Heat-cap sizing deploys about what is deployed now.
  - None of them would fix a negative per-trade excess. They would only scale it up.
- **Any change to the live paper account.** By design, nothing in this project touched live settings. Every new backtest flag defaults to the old behaviour.
- **A different 200-day gate.** E's whipsaw losses come from the single pre-registered rule (prior close vs SMA200). No other gate was tried, to avoid tuning on results.

## Known limitations

- **Daily bars approximate the live fills.** Entries use the open, not 09:45. Exits decide and fill on the same close, not the 15:50 price. Resting stops fill at the stop level, plus the spread; real stop orders can slip further.
- **A strong, over-tested window.** 2022-10 → 2026-10 starts near a market bottom (XIU +105%), which favours whatever holds the most market exposure. It has also been backtested many times, so it can't confirm anything new; only forward paper data can.
- **Backtest vs live differences:**
  - No sector cap; `top_n_buys=3`.
  - Each fold skips its last day.
  - Pre-2026 TSX holidays are simulated as trading days.
  - Survivorship bias from today's raw ticker list (flatters the strategy).
- **Live sample:** 52 trades over 5 months, in one regime. SES.TO is excluded (no valid price since 2026-09-02).
- **Costs:** commission is $0 (the user's choice: platform-specific). Spreads are assumed (10 bps per side for stocks, 2 bps for XIU), not measured.

## Changes left in the code (all default to the old behaviour)

`BacktestConfig`:
- `costs_enabled`, `commission_per_order`, `stock_spread_bps`
- `cash_yield_source`, `cash_yield_returns`
- `same_bar_rule`, `gap_fills`
- `idle_cash_mode`, `xiu_spread_bps`, `idle_buffer_pct`, `idle_min_trade_pct`

Trade logs now carry `pattern`, `stop_price` and `exit_reason`. `BacktestResults` returns open positions, costs, cash yield and the idle-account log. Tests: `tests/test_idle_cash_phase2.py`, `tests/test_idle_cash_phase3.py`.

## Folder map

| Folder | Contents |
|---|---|
| `00_recon/` | Code and data recon, plus the user decisions |
| `01_trade_vs_xiu/` | Per-trade excess, breakdowns, charts |
| `02_baseline/` | Backtester fixes, one at a time and together |
| `03_matrix/` | Pre-registered rules, the A–F matrix, attribution, verdict |
| `05_entries/` | Entry slices and candidate hypotheses |
| `ref/` | XIU/CASH.TO bars, dividends, sectors, live marks |
