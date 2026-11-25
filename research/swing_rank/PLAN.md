# Swing rank: monthly top-10 ETFs and top-10 stocks

Two ranked lists (iShares funds, Canadian stocks), built from *151 Trading
Strategies* (Kakushadze & Serur 2018) and what this repo already knows. Goal set
by the owner: **beat XIU.TO**. Research only until Phase 7. Started 2026-11-24.

## Fixed before any result (Phase 0, approved by the owner 2026-11-24)

**Universes.** A: iShares Canada equity and commodity funds (bonds only where
the strategy is multi-asset); USD units dropped; of a hedged/unhedged pair the
unhedged one is kept. B: index members as they were on each date. C: today's
liquid survivors, for live scanning only, never used to claim an edge.

**Timing.** Rank on the month's last close, trade at the next session's open.
Long only, equal weight, 10 names; a holding stays while ranked in the top 15.
Stocks: at most 2 per sector. Funds: 20-day dollar volume >= $100k and 273
sessions of history on the date. Cash earns CMR.TO.

**Costs per side.** Funds 0.10%, stocks 0.20%; stress run doubles them.

**ETF strategies (Phase 2).**
- E1 sector rotation (book 4.1): 12-month return.
- E1b (4.1.1): E1, only funds above their 200-day average; empty slots in cash.
- E2 dual momentum (4.1.2): E1 while XIU.TO is above its 200-day average, else XBB.TO.
- E3 multi-asset trend (4.6): funds incl. bonds with 12-month return > 0 and price
  above the 200-day average; top 10 by return/volatility, weights in proportion
  (eq. 372), each at most 20%, remainder in cash.
- E4 volatility targeting (6.5): XIU.TO weight = min(1, target / 20-day volatility),
  target = the expanding mean of past 20-day volatility; changed only when it moves
  by more than 10%; rest in cash.

**Stock ingredients (Phase 3), each alone.**
- S1 momentum 12-1 (3.1). S2 low volatility, 252 days (3.4).
- S3 residual momentum (3.7): 36-month monthly regression on XIU.TO and the
  sector's equal-weight return; sum of residuals over months 12..2 back, over
  their standard deviation.

**Overlays (Phase 4), one at a time, on the Phase 2/3 survivors.** Factor trend is
+1 if above its 200-day average with a positive 63-day return, -1 if the reverse,
else 0. A name's link is its 52-week weekly-return correlation, counted if |t| >= 2.
A name with a headwind (link sign x trend < 0) is skipped. Factors: gold, silver,
oil, copper, semiconductors, US dollar, US 10-year yield, real estate, VIX, credit.

**Pass rule, per list, after costs, all six:**
1. annualized mean excess over XIU.TO > 0, 95% interval (6-month blocks) above zero;
2. beats the 95th percentile of 1,000 random top-10 portfolios (before costs);
3. ahead of XIU.TO in at least 60% of calendar years;
4. maximum drawdown at most 1.25x XIU.TO's;
5. excess > 0 in at least 80% of the +/-20% runs, and with doubled costs;
6. stocks only: still excess > 0 when missing index members are assumed at
   bankrupt -100%, acquired = last price, unknown -50%.

**Gates.** Phase 2 and Phase 3 each continue only with strategies that pass 1-5
(phase 3: against the equal-weight universe as well). Phase 4 overlay kept only if
it improves the same ranking with a 99.5% interval above zero (10 overlays).
Also reported, not a pass: lower drawdown at a similar return. Every variant run
is counted.

## Log
- **2026-11-24, Phase 1 (funds).** `cache/swing_rank_prices.db`: 148 iShares funds and
  14 factors from 2009. Engine: 34 tests, 36/36 mutations. ETF strategies: 41 tests,
  36/36 mutations.
- **Phase 1 (stocks): point-in-time membership found.** BlackRock serves dated XIC
  holdings from December 2006, with sector and price for every member, dead ones
  included. A first version filtered on the Exchange column, which older files fill
  with foreign listings (RY as New York); caught by the member-count check before
  any use, now filtered on currency.
- **2026-11-24, Phase 2 done: no ETF strategy passes** (198 months, 2010-05..2026-10,
  XIU.TO 9.85% a year, max drawdown -22.8%; 56 strategy runs).
  E1 11.2% a year, excess +1.4% (-2.9..+5.6), fails 1 only. E1b 9.1%, E2 10.0%, E3 6.3%,
  E4 8.7%: fail 1, 3 and 5. Second goal (not a pass): E2 has XIU's return with a
  -10.6% drawdown, Sharpe 0.97 vs 0.85. ETF track ends; no overlays on it.
- **Phase 1 (stocks), membership done.** 231 month-ends, 2006-12..2026-10, 203-335
  members each, 795 companies. BlackRock published none for 2007-08, 2007-10 and
  2017-01..06: the last earlier file stands in (staler, never later). Two more
  errors caught before use: two companies under one label (IAG) in older files,
  one of which a de-duplication dropped.
- **Phase 1 (stocks) gate passed.** 694 companies; 262 matched to Yahoo by their
  returns, the rest priced from the index files. At least 95.5% of members priced in
  every month (mean 99.1%). The index rebuilt from this data returns 8.29% a year
  against XIU.TO's 8.39% (correlation 0.985), with Sino-Forest, Oilexco and Poseidon in
  it. Yahoo alone could price about 35% of the 2010 members. Fixed before any
  result: missed splits, dividends of file-priced names (credited at their sector's
  average), renames that change ticker and name, files dated on a holiday, and 82
  companies listed twice in December 2011. Harsh test: 346 unknown endings.
  Members and stocks: 88 tests + 1; 30/30 and 43/43 mutations (one after a new test).
- **2026-11-24, Phase 3 done: no stock ranking passes** (195 months, 2010-08..2026-10,
  XIU.TO 10.2% a year, equal-weight universe 9.4%; 29 strategy runs).
  S1 momentum 15.8%, excess +6.4% (-1.2..+14.2): meets 2, 4, 5 and 6, fails 1, 3 (58.8% of
  years, needs 60%) and the universe test (lower bound -0.2%). S2 11.6%, excess +1.2%
  (-3.7..+6.5). S3 13.4%, excess +3.4% (-1.6..+8.3), drawdown -34.7%.
  S1 recomputed independently: identical every month. t = 1.55; without its best
  10 months the excess is -0.3%; second half +2.8%. Two of its worst months
  (FirstService 2015, Dundee 2013, file-priced) look like unadjusted spin-offs, which
  understates it a little; not changed after seeing the result.
- **Stopped, per the rules.** Phases 2 and 3 both failed, so Phases 4-7 did not run.
  85 strategy runs in all (56 funds, 29 stocks).
- **2026-11-25, identity bug found after the Phase 3 result, while building the forward
  record.** `company_ids` compared a new label with the one that *started* before it,
  not the one seen last, so a short-lived spelling cut a company in two and its 12-month
  history was lost. 42 cuts: POU 2018, TRP 2019, T 2026-02 and 39 from 2026-08, when
  BlackRock dropped INC/CORP from names (about 50 of 216 members could not be ranked
  on a live list). Fixed; 654 companies, 261 matched, coverage unchanged, rebuilt index
  8.33% against 8.39%. Phase 3 rerun with nothing else changed. First result above
  stands as the record of what was seen; the corrected one is in phase3_stocks.csv:
  S1 16.1%, excess +6.6% (-0.8..+14.5), t = 1.61, years 64.7% (now meets 3), universe test
  lower bound +0.03% (now just meets it), first half +10.0%, second +3.3%, without its
  best 10 months -0.2%, harsh +2.9%: fails 1 only. S2 12.3%, excess +1.7% (-2.8..+6.9).
  S3 13.4%, excess +3.4% (-1.5..+8.3), drawdown -34.3%. **Verdict unchanged: nothing passes.**
  Independent S1 recomputation identical every month.

## Forward record of S1 (owner's request, 2026-11-25)

Paper only, no money, no service. `python -m research.swing_rank.forward record` after
each month's last close and before the next open; it writes the ten names to
`reports/swing_rank/forward_s1.csv`, which is committed the same day. First signal
2026-11-30, entry 2026-12-01; it starts with no holdings. Rules fixed in advance:
- The list is S1 exactly as tested (same code, `DEFAULT` settings); no changes while it runs.
- Entry is the first open after recording. More than 5 sessions late: the month is
  skipped and the last list is held. A recorded month is never rewritten.
- Members: BlackRock's latest list up to the signal, at most 10 days old. Prices: a
  fresh full download each month, at the paced rate; 90% of members must reach the signal day.
- Costs 0.20% a side, benchmark XIU.TO, both as in the backtest.
- **What it can show.** S1's excess varies by 16.6% a year, so confirming +6.6% at t = 2
  takes about 25 years; no pass can come from this record, and none will be claimed. It
  checks that the list can be made in advance and traded at the open. Dropped if
  after 36 months the annualized excess is below -12.6% (two standard errors under the
  backtest). No real or paper account follows it on this evidence.
- Checked 2026-11-25 by a dry run with freshly downloaded data for 2026-10-30: the
  same ten names in the same order as the backtest. Not recorded, that month was over.
