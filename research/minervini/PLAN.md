# Minervini's method on Canadian stocks: entry, exit, profit taking

Owner's request, 2026-11-25. Written before any result. Research only: no capital, nothing scheduled.
The method as published (trend template, tight rest, pivot, small stop, let winners run). Their own chart reading,
news and fundamentals judgment are not reproduced; what is tested is a mechanical copy.

## Data
`cache/breakout_prices.db` (513 TSX/TSXV/CSE names, 2011-11..2026-11, split- but not dividend-adjusted, survivors only).
Eligible as in the breakout study: close >= $1, 20-day dollar volume >= $1M, 252 sessions. 0.5% a round trip.
Periods by entry date: **A** to 2018-11-30 (choose), **B** 2019-01-02..2023-06-30, **C** 2024-01-02.. (check). These years
were already used for other ideas (VCP score, breakouts), so B and C are fresh only for these rules.

## Entry (at the close of day t; all must hold)
1. Trend template: close > 150- and 200-day averages; 150 > 200; 200 higher than 21 sessions ago; 50 > 150; close > 50;
   close >= 1.30 x 52-week low; close >= 0.75 x 52-week high; strength rank >= 70 (12-month return, last quarter
   double, percentile among eligible names).
2. Rest: pivot P = the highest high of the last 30 sessions, made 5 to 29 sessions ago; since then the lowest low is at
   most 10% under P; that depth is smaller than the deepest fall of the 60 sessions before the pivot (it contracts).
3. Dry volume: the last 5 sessions average less than the 50-day average.
**Buy:** "stop" entry: next session, as price passes P: fill max(open, P), not if the open is above 1.05 P. "volume" entry:
a close above P on >= 1.4x the 50-day volume; next open, not above 1.05 P. One position per name at a time.
The owner's example (TGTX, pivot 68.03, bought 2026-11-24) must be found by the rules. It was seen first; 10%, 1.4x, 30% and
70 are the published numbers.

## Exit
Initial stop: just under the rest's low, never more than 8% below the entry. Touched = sold (gap: the open); if stop and target
fall in one bar, the stop. At most 252 sessions. Profit taking, all reported:
| x | rule |
|---|---|
| t10 / t20 | all at +10% / +20% |
| 2R / 3R | all at 2 / 3 times the risk |
| half2R / half20 | half at 2R / +20%, then stop to breakeven, the rest on a close under the 50-day average |
| ma50 / ma20 | stop to breakeven after 2R; out on a close under the 50 / 20-day average (next open) |
| m21 | sold at the close of session 21 |
| m21t10 | all at +10%, else the close of session 21 (the owner's month) |

## Judging
Portfolio: 1% of equity risked per trade, at most 12.5% and 8 positions, strongest rank first, idle cash 0%.
**Choice on A only:** the entry x exit (2 x 10) with the best portfolio growth rate, at least 100 trades.
Then, for that choice, one filter at a time (kept only if A's growth rate rises with >= 100 trades): market (XIU above its rising
200-day average), strength rank >= 90, strength line (stock/XIU) at a 252-session high, latest quarter's EPS up >= 25% on the
year before (Yahoo cache; 2016 on).
**Passes** if in B and in C: mean net return per trade > 0 with a 95% interval (13-week blocks) above zero; the same for its
excess over the 10 same-day trend-template names of nearest ATR% bought at that day's open with the same % stop and exit; and
the portfolio grows faster than XIU.TO (price) over the period. All 20 and all filters are reported whatever they say.

## Log
- **2026-11-25, data and tests.** 72 tests; 62 injected bugs, caught except six that cannot change a result (two template
  conditions implied by the others, and the like). TGTX 2026-11-24 is found (pivot 68.03).
- **Result of the 20 (run once): chosen on A: stop entry, sell at +20%.** Verdict by the rule above: **does not pass** (the
  intervals of mean net per trade include zero in B and C). Portfolio CAGR: A 13.6% (XIU 4.1%), B 8.4% (8.1%), C 21.7% (17.9%).
  No filter raised A.
- Checks added afterwards (not part of the rule): survivor benchmarks, random slot choice, slippage, +/-20% of each setting, years.
- **Checks (after the verdict).** Every one of 1,253 trades recomputed separately: no difference. Survivors: the same eligible
  names held equally (no method) made 7.2 / 11.6 / 20.5% a year in A/B/C (XIU 4.1 / 8.1 / 17.9), so the data itself adds 2.6-3.5
  points; all trend-template names, monthly, no stops or pivots, no costs: 13.4 / 21.7 / 23.5%. Book with random choice among
  signals: 12 / 8.5 / 22.5% (5-95%: 9-14 / 7-10 / 20-26). Extra 0.5% cost on exits: 11.0 / 5.4 / 17.4%. All +/-20% settings
  stay positive per trade; book A 7.6-17.0%, B 5.8-11.7%, C 17.3-24.8%. XIU's dividends (~3%) are not in these figures.
- **All years together** (1,253 trades, includes the years the exit was chosen on): stop entry, sell at +20%: 32% win, +19.4% /
  -6.6%, +1.82% a trade (+0.59..+3.00), winners take a median 39 sessions, losers 14; over same-day trend-template names +0.58%
  (-0.05..+1.23); book 11.9% a year (max drawdown -25%), XIU price 7.8% (-36%; about 11% with dividends), equal-weight survivors
  10.8%. Shorter profit taking is worse: +10% 5.6% a year, 2R 5.6%, one month 6.2%, one month or +10% 6.8%.
