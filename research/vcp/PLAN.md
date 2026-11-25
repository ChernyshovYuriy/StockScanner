# VCP score: execution plan

Score how closely a TSX/TSXV/CSE stock matches Mark Minervini's Volatility
Contraction Pattern (0-100), then test whether a high score leads to better
forward returns. The score is not assumed to work. Research package only: no
capital, positions, schedule or dashboard column. Started 2026-11-24.

Everything up to Phase 6 runs on the dev machine. No existing file is edited.

## Phases and their gates

A phase starts only after the previous gate passes. "Mutation check" means:
inject a deliberate bug, confirm a test fails, restore.

| # | Build | Gate to pass before moving on |
|---|---|---|
| 1 | `prices.py`: paced one-time download into `cache/vcp_prices.db` | Offline tests (pacing, resume, stop on a block, no writes to `market_cache.db`); every mutation caught |
| 2 | `vcp.py`: gate, base, zigzag, score | Synthetic cases; hand-computed values; property tests (bounds, price/volume scale invariance); **future bars never change a past score**; every mutation caught |
| 3 | Real-ticker check, 5 names (existing cache) | Components recomputed independently from raw bars and equal; printed for a chart check |
| 4 | Run the download, audit the data | Coverage, duplicate/NaN/OHLC checks, jump list, spot check against `market_cache.db` |
| 5 | `backtest_vcp.py` | Tests on a synthetic market with a planted edge (must find it) and with none (must not); truncation check; hand-checked trades; every mutation caught |
| 6 | Development run, sensitivity, then one holdout run | Verdict by the rule below; report |
| 7 | Dashboard | Only after the owner has seen Phase 6; separate plan |

## Fixed before any result

**Data.** Bars 2018-01-01..2026-11-20. Universe: April 2026 raw list joined
with `can_tickers_full`, minus `.NE`, warrants, units, rights, debentures and
preferreds. Scored weekly on the last session of each week, from the first
date with a year of history.

**Return.** Buy at the next session's open, sell at the close 20/60/120
sessions after entry. A name that stops trading is closed at its last bar.
Cost shown separately: 0.5% round trip.

**Groups.** Gate failed; gate passed without a valid VCP (*gate-only*, the
fair baseline); score 1-40, 41-70, 71-85, 86-100. Also `score_stage2 > 0` and
XIU.TO.

**Separate events.** One per ticker per group per 60 sessions.

**Excess.** Event return minus the mean return of gate-only names on the same
date. Confidence intervals resample 13-week blocks of dates.

**Pass rule (60-session horizon, score >= 71, separate events).** All four:
1. mean excess > 0 and its 95% interval is above zero;
2. at least 100 events across at least 50 tickers;
3. mean excess > 0 in at least 60% of calendar years;
4. mean excess > 0 in at least 80% of the +/-20% sensitivity runs.

**Holdout.** Signals from 2025-05-20 on are not looked at until the
development verdict is written. Development signals stop 121 sessions before
it, so no development return reaches into the holdout. Weights and thresholds
are not tuned on returns. Pattern *frequency* may be checked (it uses no
returns) before Phase 5; any change made for that reason is listed here.

## Known biases

- Yahoo serves few delisted names: absolute returns are too high.
- Prices are adjusted back in time, so a later reverse split lifts an old
  price over the $1 floor.
- Overlapping windows: the date blocks handle it, but effective samples are
  far smaller than the row counts.

## Log

- **2026-11-24, Phase 1 done.** 21 tests, 14/14 mutations caught.
- **2026-11-24, Phases 2-3 done.** 155 tests, every mutation caught (69 in all);
  an independent reimplementation agreed on all 325 liquid names of 2026-10-02.
- **Change made in Phase 3, before any return was computed.** Counting every
  zigzag swing gave 10-30 "contractions" in an ordinary base, so only 3-week
  bases could qualify. Contractions are now merged (high to the lowest low
  after it, then the highest high after that low, ...), and each must be at
  most 0.85x the one before (`_detect_vcp`'s ratio), because merging makes
  "any smaller" nearly automatic. The every-swing method is reported as a
  variant. Frequency on the old cache, 2021-09..2025-05, every 20 sessions:
  12,014 liquid rows, 2,215 pass the gate, 1,355 valid VCPs (median base 41
  days), scores 71+ on 919. **Detector frozen here.**
- The development run cuts the DATA at the holdout date, so no embargo rule is
  needed: a return that would reach the holdout can't be computed.
- Baselines reported beside gate-only: all gate passers, `score_stage2 > 0`,
  XIU.TO. Gate-only (not a valid VCP) overlaps the 0 and 1-40 buckets by design.
- **2026-11-24, Phase 5 built** ahead of Phase 4 (the download waits for the
  market close); its tests use synthetic markets only, so no real return was
  seen. 49 tests.
- **2026-11-24, Phase 5 done.** 51 tests. The first run caught 47 of 52 injected
  bugs (one was a redundant condition, removed); a full re-run then exposed a
  weak sampled test (a lookahead in the Stage 2 flag got through), so every
  per-row check now covers all rows. Final full run: 51 of 51. Real bars: rows
  rebuilt from data cut at three dates are identical (73,344 rows, scores only).
- **2026-11-24, Phase 4 done.** 2,550 names asked, 2,397 with bars plus XIU.TO (153
  have none, confirmed by a retry; log shows no rate limiting). 4.38M rows, 2018-01-02..2026-11-20.
  No duplicates, weekend bars or off-calendar dates; no broken high/low bar in 560,863
  liquid ticker-days. Daily returns agree with `market_cache.db` on effectively all
  of 1.2M shared rows. Six unadjusted reverse splits (TSLA.TO, RFA.TO, TLO.TO, TLRY.TO,
  KAS.CN, KRY.V) are rescaled by `fix_split_errors` (9 more tests, 11/11 mutations).
  Unadjusted forward splits look like crashes and can't be told apart: not handled.
- **Survivorship is worse than hoped:** only 3 of 2,397 names stop before November
  2026. Yahoo drops delisted names entirely, so the April list barely helps.
- **2026-11-24, development verdict: FAIL** (written before the holdout was run).
  71+ at 60 sessions: 1,485 events, 384 tickers, mean excess -0.12%, 95% interval
  -1.97..+1.89%, ahead in 4 of 7 years (57%), positive in 9 of 44 sensitivity
  runs (20%). Only the sample-size condition is met.
- **2026-11-24, holdout (run once): also FAIL.** 525 events, 284 tickers, mean excess
  +1.13%, interval -1.85..+4.47%. The 86-100 bucket did well (weekly +3.67%,
  0.97..7.07; separate events +2.73%, -1.41..8.19), but it wasn't a pre-set test,
  rests on about 6 blocks, and the same bucket showed nothing in development
  (-0.12% weekly, +0.32% events). Phase 7 not recommended.
