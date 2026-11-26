# Breakouts: detecting what techcharts.net calls a confirmed breakout

Goal (owner, 2026-11-25): a subsystem that detects breakouts and nothing else, built
only on evidence from techcharts.net and the owner's 22 screenshots (20 charts: XPEL and
SCOP appear twice), then backtested. No capital, schedule or dashboard. Upside only
(every chart is upside).

A detection is a fact about past bars: a horizontal level was tested, held, and today
closed 3% above it. Whether price then goes on is not a fact; the backtest measures it.

## Rules and their evidence

| Rule | Value | Evidence |
|---|---|---|
| Confirmation | daily close >= level x 1.03 | FAQ: "a 3% margin above the resistance (and record a daily close)". 17 of 19 printed confirmation levels are 2.6-3.05% above the line (PPC 1.7%, IONQ 5.6%; XD prints none). On 19 charts the marked day is the first close >= line x 1.03 |
| Horizontal boundary | one price | FAQ, 2017 review: horizontal boundaries are more reliable |
| Duration | 52-504 sessions from the first test | FAQ: 3-24 months. 52: its "3 month" charts start at about 2.5 |
| Tested several times | >= 2 tests | 2017 review gives no number; charts show 2 or more. **Set from the charts** |
| A test | a 5-bar swing high within 2% of the level; 5% below it in between | not published. **Set from the charts** |
| The line | the highest test, one spike forgiven with >= 3 tests | not published; how the lines are drawn. **Set from the charts** |
| Inside | no close >= the confirmation level; at most 2% of closes above the line before the last 10 bars | follows from the first rule. **Numbers set from the charts** |
| Which pattern | the most recent long enough, highest line | **Set from the charts** (KALU, CON, CSX) |
| Target | line + (line - lowest low) | not published; measured on the charts (exact on CON, XPEL, CAE, PRSU, ATT, BAER) |
| Last Full Day low | low of the last bar wholly below the line | 2026-07-14 framework post (after Peter Brandt) |
| Types 1-4 | see backtest | same post; its "negation level" is not defined publicly |
| Volume | not used | 2017 volume post: "good" but "not a must" |

Not found publicly (members only): success rates, the negation rule, trend filters.
Prices must not be dividend-adjusted (the charts are not).

**Frozen 2026-11-25** on the 20 charts, before any return was computed. Variants looked at
while setting geometry: tolerance 1.5/2/2.5/3%, pullback 4/5/6%, pivot 3/5/7, tests 2/3,
closes above 0/2/5%, median or highest line, four tie rules. Result: 20 of 20 detected
within -4..+3 sessions, 19 lines within 2% of the drawn one.
**The charts are all successes picked by their author: they measure whether breakouts are
found, not how often a detection is worthless.**

## Backtest, fixed before any result

**Data.** Unadjusted daily bars 2011-11..2026-11 (Yahoo), 513 TSX/TSXV/CSE names of the
existing cache that were ever liquid. Survivors only: absolute returns read too high;
comparisons on the same day are less affected. Price returns (no dividends) on both sides.
**Eligible on a day:** close >= $1, 20-day dollar volume >= $1M, 252 sessions of history.
**Event.** Detection on an eligible name; then none for that ticker for 60 sessions. Buy at
the next open; return to the close 20/60/120 sessions later; a name that stops trading is
closed at its last bar. Cost 0.5% a round trip, where stated.
**Development** signals up to 2023-06-30. **Holdout** from 2024-01-02, opened once, after the
development verdict is written here.
**Comparisons, same day and horizon:** (a) all eligible names; (b) eligible names making a
52-session closing high without a detection (does the pattern add to a plain new high?);
(c) XIU.TO.

**Pass rule (60 sessions, development), all five:**
1. mean excess over (a) > 0, 95% interval (13-week blocks of signal dates) above zero;
2. the same over (b);
3. mean excess over (a) > 0 in at least 60% of calendar years;
4. mean excess over (a) > 0 in at least 80% of the +/-20% runs of every setting;
5. mean net return after cost above XIU.TO's, at least 100 events and 50 tickers.
**Then** the holdout must have a positive mean excess over (a) and (b).

**Trade as published (reported, not part of the rule).** Stop just below the LFD low
(intraday), target the measured move, at most 252 sessions, stop first if both are hit in
one bar, gaps filled at the open. Compared with 20 same-day names of the closest ATR% with
the same stop and target in ATRs. Types: 1 target without touching the line; 2 touched the
line, LFD low intact; 3 LFD low broken, pattern low held; 4 pattern low broken first;
unresolved otherwise. (The pattern low stands in for the unpublished negation level; a
real one is higher, so type 3 is overstated.)

**If it fails** the detector stays as a detector of facts; no claim that it makes money.

## Log
- **2026-11-25, data.** 514 names, 1.57M bars, no gaps in coverage; CAE matches its chart; daily
  moves agree with the existing cache (2.4 in 100,000 differ by more than 5%: special dividends).
  Added before any result: a name's history starts after any one-day rise of 2.5x or more
  (unadjusted consolidations); falls are kept.
- **Detector tests:** 85, 36 injected bugs, all caught but one that cannot change a result
  (a speed shortcut). **Backtest tests:** 40, 43 injected bugs, likewise one equivalent.
- **The 20 charts afterwards:** 20 of 20 reached the target, 19 before closing back below the
  line. Picked winners; not evidence of an edge.
- **2026-11-25, development run: does not pass** (2,030 separate events, 302 names, 2012-11..2023-06).
  60 sessions: mean +2.5%, median +1.2%, 52.9% up. Excess over all eligible names +0.91%
  (-0.17..+2.05), median -0.26%: **fails 1**. Over plain 52-session closing highs +0.35%
  (-0.49..+1.27): **fails 2**. Meets 3 (67% of years), 4 (100% of 17 runs), 5 (+0.88% net vs XIU.TO).
  Without its best 20 events the excess over (a) is -0.47%. As published (stop under the LFD
  low, measured target): 45.3% reach the target, mean +0.19R; same-day names of equal
  volatility 43.2%, +0.15R; difference -0.03..+0.11R. Types 1/2/3/4: 30/16/17/25%, unresolved 13%.
  Survivors only, so all of it reads high. 200 events and 30 same-day means recomputed
  independently: no differences. Looked at afterwards (not evidence): >= 3 tests +0.4%, 6+ months +3.8%
  (444), 12+ months +1.1%.
- **Holdout not opened**: there is no development pass for it to confirm.
- `python -m research.breakouts.recent` lists current detections, as facts.

## One-month take-profit test (owner's actual goal, 2026-11-25; written before its result)

The 60-session horizon above was the assistant's choice. The owner's use: find a breakout, buy,
take profit within a month. Development span only, same events, entry at the next open, all
sold at the close of session 21 at the latest, 0.5% cost. **Main:** take profit at the measured
target, stop under the LFD low. Shown beside it: take profit at +5%, +10%, or none; with that
stop or with none (all eight, none chosen afterwards). Compared with the 20 same-day names of
closest ATR%, traded with the same stop and target in ATRs. **Passes** if the main version's mean
net return is > 0 and its excess over those names has a 95% interval above zero.
(20-session returns were already printed above: +1.02%, excess +0.41%, -0.27..+1.12.)
- **2026-11-25, one month: does not pass.** Main (target, LFD stop, 21 sessions; 2,006 trades): 47.8%
  profitable, target reached 19.9%, stopped 29.1%, mean net +0.01% (-0.59..+0.62); same-day names
  +0.10%, target 19.3%; difference -0.09% (-0.49..+0.32). All eight versions: mean net -0.17%..+0.52%,
  difference -0.17%..+0.26%, every interval spans zero. +5% without a stop wins 62.2% of the time
  and still averages -0.12%. reports/breakouts/month_dev.csv. Check: the version with neither stop
  nor target equals the 20-session return above less cost. Of the 20 charts, 12 reached the
  target within 21 sessions.

## What do the successful breakouts share? (owner's question, 2026-11-25; written before its result)

Every feature is known at the breakout's close (`features.py`, FEATURES: the base, the breakout day, volume,
trend, strength against the market, market, sector, event). Winners are compared with losers, never
looked at alone: what both have is not a feature of success.
**Outcome:** the month return already defined (next open to the close of session 21; r20). **Gained** >= +5%,
**lost** <= -5%, **big winner** >= +15%. Scoring uses y = (r20 - same-day mean of eligible names) / ATR%, so
that "volatile" (which makes both winners and losers) does not pass for "good"; rank correlation, so one
monster cannot decide.
**Periods:** A = to 2018-11-30 (search), B = 2019-01-02..2023-06-30 (confirmation), C = 2024-01-02 on, unopened.
**A feature counts if** in A its rank correlation with y survives Benjamini-Hochberg at 10% over all features
(13-week block bootstrap), and in B it has the same sign with one-sided p < 0.05. Counting features are then
checked once on C: same sign, p < 0.05.
**Also reported, same periods:** (1) for each feature, the share of gained / lost having it (best third by A's
cut points and sign), in A and B; (2) the widest threshold 90% of A's big winners meet, and how many losers meet
it too; (3) combinations: ridge on all ranked features and a depth-3 tree (leaf >= 60), fitted on A only, the
best fifth/leaf in B; a combination counts if its B mean net return (0.5% cost) is > 0 and its excess over the
same-day names has a 95% interval above zero; then once on C.
**The 20 charts:** their features against the other breakouts of the same 20 stocks. Description only (picked).
Earnings dates: Yahoo cache, covers about 82% of events from 2015; sector: XIC holdings, else Yahoo cache (96%).
A missing feature is left out of that feature's test. No settings will be changed after the result.
- **2026-11-25, A and B.** 62 features. Count: `slope200`, `ma_gap`, `rs252` (a long uptrend already in place). Not:
  `overhead`, `prior_up`, `ret252`, `rs126` passed A and failed B; every other feature failed A, among them volume, base
  length, depth, tightness, tests, gap, cross speed, market, sector and earnings. Neither combination counts
  (ridge B net +3.7%, excess +3.2% (-0.5..+8.7); tree +3.4% (-2.3..+13.4)).
- **Before opening C** (fixed here): p is two-sided. Events from 2024-01-02 with a finished month. Beside each rule's best
  third (A's cut point): the same-day eligible names in the same range of that feature with no detection in 20 sessions
  (is it the breakout or only the trend?), in B and C. All three features are near-copies, so also the one rule "all three
  in their best third". Nothing will be changed after C.
- **2026-11-25, C opened once (862 events, 2024-01..2026-10): all three count** (two-sided p: slope200 0.003, rs252 0.037,
  ma_gap 0.011; rank correlation 0.06-0.11, as in A and B at 0.10-0.14). What winners share more than losers is a long uptrend
  that was already in place. **As money it is not proven:** in C their best thirds net +1.3..+1.4% a month (XIU.TO
  +0.9..+1.1%), excess over the same-day names +0.65..+1.0%, every interval spans zero; and over names with the same uptrend
  but no breakout +0.4..+0.6%, every interval spans zero (B: +2.2..+2.7%, only ma_gap's excludes zero). All three
  together in C: net +1.4% (median +1.1%), 36% gained >= 5%, 28% lost >= 5%; without its ten best trades -1.25%.
  Nothing is in all winners: 72% of those that gained >= 5% had a rising 200-day average, and so did 58% of the losers.
- **Seen afterwards, not evidence:** 200-day average falling: 44/43/50% up in A/B/C and a negative mean net in each; rising:
  59/55/58%. 20 charts against the 55 other breakouts of the same stocks: shallower base (14% vs 19%), at the 3-year high
  (+1% vs -10%), 200-day slope +1.9% vs +0.7% (75% vs 59% rising), 2.5 vs 2 tests; same in volume, tightness, gap, speed.
- Feature tests: 36; 48 injected bugs, all caught but one that cannot change a result. slope200 and ma_gap recomputed
  from raw closes for all events: no differences. reports/breakouts/features_*.csv.
- **2026-11-25, owner's question: chance of +5%/+10% in one or two weeks with a rising 200-day average** (2,017 breakouts,
  all years; description, the filter was seen on this data). Touched from the next open, before costs: +5% 24.1% / 37.9%,
  +10% 7.5% / 14.0%, +15% 2.6% / 5.6% (5 / 10 sessions); touched -5% 20.7% / 31.6%, -10% 5.8% / 11.5%. +5% before -5%:
  22.0 v 19.4% / 33.2 v 28.2%. Same-day rising names of equal ATR%: +5% 20.1% / 34.5%, +10% 5.3% / 12.1%. Sell at +5%
  or at the close of session 10: net +0.14% (-0.15..+0.41); +10%: +0.30% (-0.03..+0.61).
- **2026-11-25, where the 20 charts come from (owner):** Tech Charts' video "BEST 20 BREAKOUT ALERTS FOR 2025 2026",
  published 2026-06-02, after every one of them (the last broke out 2026-04-14). Selected on the outcome: settled. They are
  good evidence of what a breakout looks like (detection) and none about the odds. How many alerts they were picked from is
  not public.

## Live scan (owner's decision, 2026-11-25)
`live.py`, on the Pi weekdays 17:45 (21:00 repeats a failed scan): emails each confirmed breakout with the 200-day average
rising/falling. Nothing is traded. Over four sample weeks (2020, 2024, 2026 twice) it gives exactly the backtest's 46
eligible detections from 3 years of bars. 42 tests; 35 injected bugs, all caught but three that cannot change a result.
`recent.py` shows the same mark.

## Ascending triangle (owner's question, 2026-11-25; definition written before its result)
Two 1930s charts (Hudson Motor, Johns Manville): flat top, rising lows, falling volume, 5-8 weeks. At bar t: level = highest of at
least 2 swing highs (3 bars either side, confirmed before t) within 2% of it, with a fall of >= 3% between them; first touch 20-126
sessions ago; no close > level x 1.01 since; t closes above level x 1.01 (variant 1.03). The lows between touches and from the last
touch to t-1 (>= 2): **ascending** if each is >= 1.01 x the one before; **falling** if the last is < 0.99 x the first; else **flat**.
Eligible names, 20 sessions between events of a name, next open, 20 and 60 sessions, 0.5% cost. Compared with same-day eligible
names, plain new highs, and above all the same pattern without rising lows. Variants shown: volume falling (last third of the
base below the first), breakout volume >= 1.5x. Trade: stop under the last low, target level + (level - first low), against same-day
names of equal volatility. **Relevant** if ascending beats both same-day names and the non-ascending ones (60 sessions, all years)
with 95% intervals above zero, and is ahead in A, B and C. All years were used for other ideas before.
- **2026-11-25, ascending triangle: does not pass** (7,323 flat-top breakouts; 2,464 ascending). Ascending, 60 sessions: excess over
  same-day names +0.33% (-0.78..+1.53); over the same pattern without rising lows +0.66% (-0.54..+1.94); A +0.04, B +1.38, C -0.82%.
  20 sessions: -0.07% (-0.67..+0.52); 53% up. Trade to the measured target, stop under the last low: 49.9% reach it, same-day names
  of equal volatility 50.6%. 3% margin: the same. One variant stands out, breakout volume >= 1.5x (776): +2.98% (+0.33..+6.02) at
  60 sessions, but one of many, see the check below.
  Check: with volume its median excess is +0.3% and +0.6% without its ten best trades (top five +153..+395%); the same pattern without
  rising lows and with volume -0.1% (median -1.0%); rising lows on lower volume -0.9%. By period +2.5 / +4.3 / +1.7%, each spanning
  zero. Suggestive, found after looking; the textbook pattern does ask for volume on the breakout. Scripts not kept in the repo.
