# Phase 0: Recon

Read-only pass over the code and data as of commit `1531b7c` (2026-10-27). No code changes.

## 1. Backtest loop and folds

- **Loop:** `backtest_runner.py`, `BacktestRunner.run()`. For each trading day D:
  1. Buy yesterday's intents at D's open (`_execute_buys`).
  2. Run the monitor on D's completed bar and sell at D's close (`_run_monitor_step`).
  3. Run the screener (weekly, `screener_frequency=5`).
  4. Run pattern detection and create intents for D+1.
  5. Log equity (cash plus open positions marked at D's close).
- **Folds:** `walk_forward_ab.py`. Sixteen contiguous, **non-overlapping** windows of about 65 trading days each.
  - Each fold starts with fresh `initial_cash` (no compounding across folds), and open positions at fold end are marked to market, not closed.
  - There is no train/test split. It's an A/B of two fixed configs, with nothing fitted per fold.
  - Screener scores are computed once for the whole window and reused.
- **The +22% vs XIU +110% result** (2026-10-08/09) came from scratch scripts (`pit_wf.py`, `run_wf_ab.py`) that were never committed and no longer exist. That run used the point-in-time universe, the live 20d/0% time stop, and bars from the cache only.
- **The committed `walk_forward_ab.py` does not match live:** window 2022-08-26 → 2026-08-26, time stop 14d/+0.5% (live is 20d/0%), no sector cap (live has `MAX_POSITIONS_PER_SECTOR=2`), and `top_n_buys=3` new buys per day (live processes up to 10 intents per morning, limited only by free slots).

## 2. Trade logs

- **Live paper trading** (`data/trading.db`, DuckDB):
  - `trades` table: ticker, entry/sell date, entry/sell price, shares, proceeds, pnl $/%, exit `reason`.
  - The setup type is not in `trades`; it's in `transactions.reason` on the BUY row.
  - The planned stop is in `intents.stop_price` (EXECUTED rows), and in `positions.stop_price` while the position is open.
  - 45 closed trades (2026-05-28 → 2026-10-22). Sector comes from `cache/sector_cache.json` (Pi) or a Yahoo lookup.
- **Backtest:** `portfolio.ClosedTrade` / `BacktestResults.trade_log_df()` holds only ticker, dates, prices, shares and pnl.
  - **No setup type, no stop, no exit reason.**
  - No point-in-time walk-forward trade log is saved anywhere. The newest trade CSVs in `out/` are from July, on the old non-point-in-time universe.
  - **Phase 1 cannot run on backtest trades without adding these fields to `ClosedTrade` and re-running the walk-forward.**

## 3. Fill convention

| | Backtest | Live paper |
|---|---|---|
| Entry | Open of D+1 (`_next_open_price`) | Last 1-minute bar of the session at about 09:45 (`get_session_quote`) |
| Exit | **Close of D, decided on D's completed bar** (`_day_close_price`) | Latest 5-minute bar at about 15:50, decided on that same price |

- **Exit look-ahead (plan STOP condition).** The backtest decides an exit on D's close and fills at that same close. You can't know a close before it prints, so this is same-bar look-ahead.
  - Live does the closest real equivalent: it decides and sells at the 15:50 price, about 10 minutes before the close. So the backtest models live's mechanics, but replaces the 15:50 price with the actual close it couldn't have known.
  - Size of the effect: unknown. It's the price move in the last ~10 minutes, plus the selection effect of deciding on the close rather than the 15:50 price.
- **Yahoo daily bars have no 09:45 price.** Backtest entries use the open, so the 15 minutes from 09:30 to 09:45 aren't modelled.

## 4. Same-bar handling

- Stops trigger on **close only** (`STOP_TRIGGER="close"`): exit if close < stop. The bar's low is never used to trigger an exit, so low-vs-high ordering within a bar never arises.
- The chandelier level uses the highest high *including* D, and is armed when the peak close including D reaches +8%. Both checks happen at D's close, after D's high has happened. Arming can only raise the stop, so this is not optimistic.
- Giveback and the time stop also use D's close. **The same-bar rule is not optimistic.**
- **There are no gap fills,** because no stop order exists. A gap through the stop exits at that day's close, which may be better or worse than the open.

## 5. Price basis

- Every download uses `auto_adjust=True`: OHLC adjusted for splits and dividends, for both stocks and XIU. The backtest is therefore on an approximately total-return basis for both, so its +110% XIU benchmark is approximately total return.
- **Caveat:** `market_data_cache.py` only adds new bars to the end of each series and never re-fetches older ones. Dividends paid after a ticker's first fetch are therefore not re-adjusted into older bars, and the basis drifts toward price-only for those. Its size for XIU is unverified. Local cache: 958 tickers, 2020-09-01 → 2026-10-02.
- **Live paper is price-only:** fills use raw quotes and no dividends are credited (dividends were deliberately left out). The earlier live review compared it with XIU price-only, a consistent basis.
- For Phase 1, the live comparison must treat both sides the same way: both price-only, or both total return with stock dividends added to the trade.

## 6. Costs and cash interest

None anywhere: no commission, spread or slippage in `backtest_runner.py`, `portfolio.py`, `virtual_buy.py` or `position_monitor.py`. Uninvested cash earns 0%, in both backtest and live.

## 7. Regime gate

- **Backtest** (`_is_market_in_uptrend`): XIU's close on D ≥ its 200-day simple moving average through D. Evaluated after D's close and applied to D+1 morning buys only. Sells are never blocked.
- **Live:** `main.py` (16:30) and `auto_pipeline` both check XIU's latest close against its 200-day SMA. A failing check means no new intents, so no buys the next morning. Same timing as the backtest.

## STOP conditions (per plan)

| Condition | Status |
|---|---|
| Fills use same-day close for same-day signals | **Yes, for exits** (§3). Entries are fine (D close signal → D+1 open). |
| Same-bar rule optimistic | No (§4) |
| XIU benchmark on a different return basis than the strategy | No within each dataset (backtest: both adjusted; live: both raw), with the cache-tail caveat (§5) |

**Stopping here** for the user's decision on the exit-fill issue, plus the broker facts the plan says to ask for.

## User decisions (2026-10-27)

| Question | Decision |
|---|---|
| Exit fill at D's close | **Accept** as an approximation of the live 15:50 sell. Listed as a known limitation in every report. |
| Commission | Platform-specific, so not part of the framework. Modelled as a per-order parameter **set to $0** in all runs; costs are spread/slippage only. |
| Spread/slippage | Plan defaults: stocks 10 bps per side, XIU 2 bps per side. Adverse scenario: 20 / 4 bps. |
| Cash yield | CASH.TO total return (Yahoo daily, adjusted). |

## Other gaps found

1. Phase 1 needs backtest trade logs that include setup type, stop and exit reason. Today's code doesn't record them, and no saved log exists.
2. The committed walk-forward settings differ from live (time stop, sector cap, buys per day, window). The +22%/+110% numbers came from a scratch wrapper that fixed some of these.
3. Delisted or stale tickers in the backtest are force-exited at that day's close or, failing that, **at the entry price** (a flat exit). This rarely affects liquid names.
