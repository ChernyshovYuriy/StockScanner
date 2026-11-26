# US test of the Canadian rules (owner's request, 2026-11-25; written before any US price was seen)

Nothing is tuned: the rules are the ones frozen on Canadian data (`research/minervini/PLAN.md`, `research/breakouts/PLAN.md`).
Two hypotheses only.

**List.** `tickers.csv`: all 5,226 common stocks and ADRs in the Nasdaq Trader directory on 2026-11-25 (no size filter);
`sp500_members.csv`: S&P 500 members by date (fja05680/sp500), 219 of 821 no longer listed. (iShares' dated holdings, used for
Canada, stopped being served in October 2026. A current index list was rejected: membership is earned by having risen.)
**Prices.** Yahoo, split- but not dividend-adjusted, 15 years, 0.4 requests/s, `cache/us_prices.db`.
**Eligible on a day:** close >= $5, 20-day dollar volume >= $10M, 252 sessions. (Canada: $1 and $1M; a US listing is
traded more, and $5 is the usual floor. Set before seeing prices.) Cost 0.5% a round trip (a US-dollar account: no conversion).
Survivors: the same comparisons repeated on **S&P 500 members on the day** (the members Yahoo still has; their share is reported).

**H1 Minervini:** stop entry at the pivot, stop at most 8%, sell at +20%; signals 2012-11-01..2026-11-24.
Passes if, all years: mean net per trade > 0 and its excess over the 10 same-day trend-template names of nearest ATR% > 0, both 95%
intervals above zero; the 8-slot portfolio grows faster than SPY (price) over all years and in at least two of 2012-18 / 2019-23 /
2024-26. Shown beside: SPY with dividends, equal-weight eligible, the one-week order used by the live scan, the ten exits.
**H2 rising lows + volume:** ascending triangle, close above the level x 1.01 on >= 1.5x volume, 60 sessions. Passes if its excess over
the same-day eligible names is > 0 with a 95% interval above zero, its median excess is > 0, and it beats the same pattern without
rising lows on the same volume (interval above zero).
If either fails, it is dropped for the US; a near miss stays a near miss.

## Log

**2026-11-25 results (US list 3,882 names; prices unadjusted from Yahoo, survivors only -- flatters every rule).**
SPY price 12.2%/yr, equal-weight eligible 9.2%/yr (2012-26).
- **H1 Minervini FAILED** every check. All names, net of 0.5%: 10,243 trades, 26% win, +0.12%/trade (-0.7..+0.9), excess vs
  same-day template peers -0.5%, book -0.4%/yr (B -7.2%/yr, -53% DD). S&P 500 members only: +0.15%/trade, book -2.7%/yr.
  One-week order -2.2%/yr. None of the ten exits gets a book above +2.1%/yr.
- **H2 rising lows + volume FAILED**: 60 sessions excess -0.7% (-1.4..+0.1), median -1.4%; no better than the same pattern
  without rising lows. S&P 500 only: same.
- **Every sleeve (before costs): none "better".** Book a year: vol3 quiet 60 11.5%, breakout +5% month 11.4%, triangle 60 10.0%,
  vol5 quiet 60 7.3%, breakout month 6.4%, with 200d rising 6.1%, vol3 60 5.2%, kangaroo 4.2%, Minervini +20% 3.2%, conviction 1.6%,
  Minervini one-week 1.3%; momentum 12-1 top ten 5.4% (2024-26 -14.8%). No excess interval above zero.
  The Canadian leaders do not repeat (vol3 60: CA 24.3% vs US 5.2%; Minervini +20%: 15.1% vs 3.2%).
- **Core sleeve (BacktestRunner):** quarters mean +2.55% vs SPY +4.89%, ahead 4/16, p=0.144; one 4-year run +7.9% vs +108%.
  **Momentum sleeve:** +4.55% vs +4.89%, ahead 7/16, p=0.845; one run +28.2% vs +108%. (Canada: core +1.35% vs +3.64%; momentum
  +1.57% vs +3.64%.)
- **Scripts:** volume breakout variants no edge (all p>=0.1, breakout-only worse than baseline); volume spike hold 1-10 days no
  edge (p>=0.1); persistence: confirmation worse (D4 confirmed -4.0%/60d, p=0.03); target exits walk-forward no edge (p>=0.18);
  RS rotation walk-forward -5.8%/yr vs equal weight 8.1% (p=0.46).
- **VCP score:** 71+ minus gate-only at 60 sessions +0.26% (CI -0.27..+0.84), positive in 0.6 of years -- fails c1.
Verdict: nothing in this project has an edge on US stocks either; the US index itself was the hard bar.

## Every sleeve on the US list (owner, 2026-11-25; costs left out at the owner's request; written before the results)
Same code and settings as in this project, SPY in place of XIU.TO, the eligibility above. `sleeves.py`:
kangaroo tail (its breakout entry), conviction (first close 20% under the 52-week closing high; out on a close 25% under the peak close;
without its quality filter, which only knows today's fundamentals), breakouts 3% (21 sessions; with a rising 200-day average; and
+5%), volume breakouts (3x and 5x, quiet base), Minervini, triangles, momentum 12-1 top ten monthly. Core and momentum sleeves
through the project's own BacktestRunner (2022-10..2026-10). The volume-spike, persistence, target and rotation scripts and the VCP
score as they are. Yardsticks for each: mean per trade less all eligible names over the same days (95% interval), and an 8-slot
book against SPY and the equal-weight list. **Better** means: that interval above zero, and the book ahead of SPY over all years and in
two of three periods. About 20 strategies are looked at: one or two would look good by chance. Not run: macro (its regime history
needs FRED), EDGAR / demand / news / RBC (no price rule to test), earnings drift (no US earnings dates here).
