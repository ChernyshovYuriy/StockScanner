# Phase 3: Comparison matrix

**Verdict (applying `DECISION_RULES.md` mechanically): don't keep the swing layer. Hold XIU (D).**
- Rule 1 failed before the matrix ran. Rules 2–4 fail as well.
- E doesn't beat D on Calmar or max drawdown, so the "otherwise" branch picks D, not E.

The rules were committed in `76f2f06` before any matrix result existed. Machine-readable verdict: `decision.json`.

## Setup

- **Folds:** the same 16 non-overlapping folds (2022-10-03 → 2026-10-02, 1,022 simulated days), each starting from a fresh $50k.
- **Stitched curve:** each variant's daily returns are chained across folds in date order.
- **Strategy runs (A/B/C):** the Phase 2 all-fixes configuration; `analysis/run_walk_forward_trades.py <scenario>_<A|B|C>`, outputs in `runs/`.
- **Benchmarks (D/E/F):** `analysis/idle_cash_benchmarks.py`, on the same fold days.
- **Statistics:** `analysis/phase3_matrix.py`.

**Checks:**
- `realistic_A` reproduces Phase 2's `all_on` fold for fold (+25.1%).
- Within each cost scenario, A, B and C made the **same stock trades** (275 / 272 / 271 closed trades, plus 110–111 open at fold ends).
- The real account never had to skip a buy (0 skips), so the shadow ledger did its job: B and C differ from A only in where the idle money sat.

## Results (realistic costs: stocks 10 bps per side, XIU 2 bps per side)

| Variant | Total | CAGR | Max DD | Sharpe | Calmar | Ulcer | Exposure | Turnover/yr | Costs |
|---|---|---|---|---|---|---|---|---|---|
| A strategy, idle cash earns cash yield | +25.0% | 5.7% | −11.8% | 0.31 | 0.48 | 4.5 | 50% | 13.2× | $2,693 |
| B strategy, idle cash always in XIU | +76.5% | 15.0% | −11.5% | 0.94 | 1.31 | 3.9 | 99% | 29.7× | $3,388 |
| C strategy, idle cash in XIU above 200d | +45.3% | 9.6% | −12.2% | 0.59 | 0.79 | 5.4 | 90% | 30.7× | $3,420 |
| **D XIU buy-and-hold** | **+104.6%** | **19.3%** | −11.6% | **1.22** | 1.67 | 2.9 | 100% | 3.8× | $160 |
| E XIU above 200d, else cash | +63.2% | 12.8% | −12.8% | 0.85 | 1.00 | 5.2 | 89% | 7.1× | $295 |
| F 55% XIU / 45% cash (descriptive only) | +59.1% | 12.1% | −6.2% | 1.22 | 1.95 | 1.3 | 55% | 2.2× | $91 |

**Other scenarios** (total return; full table in `metrics.csv`):

| Variant | Adverse costs | Zero costs |
|---|---|---|
| A | +17.8% | +37.4% |
| B | +63.9% | +93.7% |
| C | +34.8% | +59.7% |
| D | +104.0% | +105.3% |

D wins in every scenario, including zero costs.

## Paired comparisons (realistic; daily return differences, annualized)

| Pair | Ann. diff | Tracking err. | IR | 95% CI, 5d blocks | 10d | 20d | Fold wins |
|---|---|---|---|---|---|---|---|
| B − D | −3.7 pp | 6.5% | −0.57 | −9.9 … +2.5 | −10.2 … +2.6 | −10.7 … +2.8 | 6/16 |
| C − E | −2.9 pp | 6.5% | −0.44 | −9.4 … +3.6 | −9.4 … +3.8 | −9.9 … +3.8 | 6/16 |
| E − D | −5.8 pp | 6.2% | −0.93 | −11.8 … +0.0 | −11.5 … −0.4 | −11.7 … −0.5 | 1/16 |
| B − A | +8.9 pp | 7.5% | +1.19 | +1.5 … +16.1 | +2.2 … +15.6 | +2.9 … +15.1 | 14/16 |
| C − B | −4.9 pp | 5.4% | −0.92 | −10.5 … +0.2 | −10.1 … −0.2 | −10.2 … −0.5 | 1/16 |

**Alpha regressions** (Jensen's: daily excess over CASH.TO, regressed on XIU's; Newey-West, 6 lags):

| Variant | Realistic: alpha/yr | t | beta | Adverse: alpha/yr | t | Zero: alpha/yr | t |
|---|---|---|---|---|---|---|---|
| A | −1.9% | −0.58 | 0.28 | −3.4% | −1.02 | +0.4% | 0.13 |
| B | −1.4% | −0.46 | 0.84 | −3.3% | −1.06 | +1.0% | 0.32 |
| C | −3.2% | −0.89 | 0.63 | −5.1% | −1.39 | −0.8% | −0.22 |

## Decision rules applied

| Rule | Result | Pass? |
|---|---|---|
| 1. Phase 1 mean excess after costs > 0 | −0.41% realistic, −0.61% adverse | **No** |
| 2. B beats D or C beats E (CI > 0 at every block length) | B−D: −3.7 pp, CI spans 0. C−E: −2.9 pp, CI spans 0. Both point estimates are negative. | **No** |
| 3. Chosen variant's alpha > 0 with NW t > 2 | No candidate came out of rule 2. For the record, every t is negative (−0.46 to −0.89). | **No** |
| 4. Holds under adverse costs | Nothing to hold. Adverse is worse on every comparison. | **No** |
| Otherwise: E if E beats D on both Calmar and max DD | E Calmar 1.00 vs D 1.67; E max DD −12.8% vs D −11.6% | E loses on both → **D** |

## Where the money came from (attribution, B and C, realistic)

Cumulative dollars summed over 16 folds × $50k (`attribution.csv`, `attribution.png`):

| Component | B | C |
|---|---|---|
| XIU sleeve (parked idle money) | +$22,191 | +$11,491 |
| Stocks: what the same money would have made in XIU | +$15,225 | +$15,225 |
| **Stocks: excess over XIU** | **−$4,245** | **−$4,245** |
| Cash yield | +$75 | +$897 |
| Costs (stock + XIU) | −$3,388 | −$3,420 |
| **Total** | **+$29,859** | **+$19,948** |

**Market exposure produced all of B's gain.** The stock picks were worth −$4.2k relative to holding XIU over the same days, and they cost $3.4k to trade. Adding them on top of a fully parked account (B) costs about 3.7 pp a year against holding XIU outright (D).

The excess line was near zero or positive until 2026-02 and then fell ~$7k in March–April 2026. A single stretch can swing it.

## Observations (not acted on)

1. **Parking idle cash in XIU is the one clear improvement** (B − A: +8.9 pp/yr, CI above zero at every block length, 14/16 folds). It's simply more market exposure in a window where XIU doubled. It doesn't add skill: B's alpha is still negative, and its beta is 0.84.
2. **The 200-day gate hurt here** (E − D −5.8 pp/yr; C − B −4.9 pp/yr; 1/16 folds each). Whipsaws around the 200-day in late 2022, fall 2023 and April 2025 sold low and bought back higher. It didn't reduce drawdown either (E −12.8% vs D −11.6%).
3. **F (55/45) has the best Calmar (1.95) and the shallowest drawdown (−6.2%)**, with the same Sharpe as D. That's just a lower-risk mix, and F was chosen after seeing live data, so it is descriptive only, per the plan.

## Limitations

- **A strong bull window.** 2022-10 → 2026-10 started near a market bottom (XIU +105%). Any variant with more market exposure wins in a window like this, and a 200-day gate looks worst. The window has also been tested many times, so it can't confirm anything new.
- **Inherited from Phases 1–2:**
  - Close-decided exits fill at the same close.
  - No sector cap; `top_n_buys=3`.
  - Each fold skips its last day.
  - Pre-2026 TSX holidays are simulated as trading days.
  - Survivorship bias from today's ticker list.
- **Fold restarts.** Every fold restarts from $50k, so B, C, D and E re-buy XIU 16 times. That adds a small cost, the same for all.
- **Attribution approximation.** The XIU-matched part for stocks uses prior-close stock value × XIU's close-to-close return, plus each day's buys × XIU's open-to-close return. Exit-day intraday timing isn't separated.

## Files

- `DECISION_RULES.md`: the pre-registered rules.
- `decision.json`: each rule's inputs and outcome.
- `metrics.csv`, `pairs.csv`, `regressions.csv`, `attribution.csv`, `stitched_returns.csv`.
- `equity_curves.png`, `drawdowns.png`, `attribution.png`.
- `runs/<scenario>_<A|B|C>/`: per-run trades, open positions, equity, folds and the real-account log (`backtest_idle.csv`).
