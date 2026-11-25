# Daily Minervini scan of Canadian stocks, by email

Alert only: no orders, no positions, no capital. The rules are exactly those backtested (PLAN.md), which did not pass their
check; every email says so.

## Every weekday evening
1. Reuse the bars the 17:45 breakout scan downloads (no extra Yahoo requests): that scan saves the session's bars to
   `cache/` when it finishes; this one reads them. Separate service, so neither can break the other.
2. Eligible names -> trend template (rank among all eligible names) -> tight rest -> dry volume. Same functions as the
   backtest (`method.py`).
3. Candidate: setup valid at tonight's close and price within 5% under the pivot (94% of past trades triggered from there).
4. Email only if at least one candidate is new (this name and pivot not emailed before). Nothing else is ever sent, except a
   failure notice. Expected: about 1 new candidate a day, no email on about half of days.

## Email, per candidate
Entry: buy above the pivot; do not pay more than pivot x 1.05. Loss: max(rest low x 0.999, entry x 0.92). Take profit: entry x 1.20.
Also: close and distance to the pivot, risk %, rest length and depth, strength rank, volume. Then earlier candidates that are
still valid, and those to cancel (rest broken / volume no longer dry / template lost). Footer: past odds.

## Phases
1. Test one question first, rule written before: an order left standing 5 sessions (how a person would use an email) against the
   tested one-session order. The email then states whichever rule was tested.
2. `scan.py`: candidates from bars; own SQLite `data/minervini.db` (candidates, runs); tests; equality with the backtest's setups.
3. Email text, new/valid/cancelled logic, failure notice; tests with injected bugs.
4. Dry run on the Pi against a known session; units `stockscanner-minervini.{service,timer}` (18:20, 21:40 if the first had no bars).
5. Commit, push, enable; CLAUDE.md and system/info lines.
6. Later, optional: record whether each candidate triggered, stopped or reached +20%, so the live record can be judged after
   50-100 trades.

## Failure handling
Breakout scan not finished or incomplete: wait for the second slot, then one failure email. Missed day: next run uses that
evening's setups only (yesterday's are stale).

## Owner's decisions (2026-11-25)
Within **3%** of the pivot. No cancel emails: a dashboard tab (`/minervini`) shows each setup's state for one week (5 sessions), then
drops it. The order therefore stands one week, which the backtest did not test. **Test, written before its result:** the first
evening a pivot is a valid setup within 3%, an order for the next 5 sessions, filled as before (not if the open is above 1.05 x
pivot: the order ends). *keep*: stands whatever happens. *cancel*: also ends when the day's low reaches the loss point before a fill
or the trend template / eligibility fails at a close. One setup per name in its week. Exit as chosen (+20%). Used live: the one with
the better mean net per trade in A (to 2018-11-30); B and C reported. Its numbers go into the email footer.
- **Result: keep** (A: +1.31% a trade against +1.23% with cancel). One-week order, all years: 2,702 emails (0.75 a day, none on 55% of
  days), 52% bought within the week; of those 1,401: 30% win, +19.4% / -6.1%, +1.54% a trade (+0.44..+2.65; A +1.31, B +0.98, C
  +2.80, each interval includes zero), book 8.1% a year (A 7.1, B 4.8, C 11.2) against XIU price 7.8% (4.1, 8.1, 17.9). Weaker than
  the tested fresh-every-evening order (11.9%): B and C trail XIU. States on the tab: waiting, bought, stopped, target, expired, ran.
- **Built 2026-11-25.** The live scan, run evening by evening on 3 years of bars, gives exactly the backtest's candidates
  and states (93 of 93 over two 40-session stretches). 45 + 3 tests; 23 injected bugs, caught but three duplicated safety checks.
