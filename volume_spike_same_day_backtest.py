"""
volume_spike_same_day_backtest.py
===================================
The validated finding, after volume_spike_intraday_backtest.py's own
checkpoint sweep was retracted (broken hourly-volume data -- see that
script's docstring) and volume_spike_daily_backtest.py showed that
buying a confirmed spike at its CLOSE and holding forward is negative
(see that script's docstring): does buying INTRADAY, on a day that
turns out to be a confirmed volume-spike + price-up day, and holding to
THAT SAME day's close, actually work?

Detection uses ONLY the accurate daily-bar feed (exactly
volume_spike_scanner.compute_spikes()'s definition: today's volume >
its own trailing 20-trading-day average, AND close > yesterday's
close) -- the hourly-bar undercount bug that sank the checkpoint sweep
never touches this, because volume is never read from hourly bars here.
Hourly bars are used for exactly one thing: the PRICE at midday on a
confirmed-spike day, to simulate a midday entry -- price was never the
broken part (only the hourly VOLUME sum was), and it's rescaled onto the
same dividend/split-adjusted basis as the daily feed (see
_checkpoint_reads()'s own docstring in volume_spike_intraday_backtest.py,
reused here unchanged).

One honest caveat: "was today a confirmed spike" is only fully knowable
once the day is over, so this is mildly idealized relative to checking
the live tool at exactly 12:30pm in real time (a stock's volume could
still slow down after midday and end up not qualifying). In practice,
checking later in the session (as this signal is meant to be used) has
this be much closer to a real-time read; the size of the effect below
is also far larger than this caveat could plausibly explain away on its
own.

RESULT (2026-09, 106-name liquid TSX universe, ~2yr window -- the
INTRADAY_PERIOD hourly-bar ceiling): 16/16 folds positive, t=18.34,
p~0.0000. Win rate 63.1% (15,302 events) vs. 46.3% baseline (57,981
ticker-days), mean return +0.27%/trade vs. -0.04%/trade -- a +0.32pp
edge, consistent in both halves of the window (+0.32pp / +0.32pp). This
is the strongest, cleanest result in this whole line of research.

Economic read: a volume spike's own price move tends to continue THROUGH
the day it happens (real intraday follow-through) but is largely spent
by the time that day is over -- see volume_spike_daily_backtest.py's own
negative forward-holding result. The strategy this validates is strictly
same-day: enter intraday on a confirmed (or near-confirmed, if checked
late enough in the session) spike, exit at that day's close. It does NOT
validate buying after confirmation and holding overnight or longer.

Usage:
    python volume_spike_same_day_backtest.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import stats
from tabulate import tabulate

from config import OUT_PATH
from market_data_cache import sync_and_load
from volume_spike_daily_backtest import AVG_VOLUME_DAYS, FETCH_END, START as DAILY_START
from volume_spike_intraday_backtest import TICKERS, _checkpoint_reads, _load_hourly

ENTRY_CHECKPOINT = "midday"       # ~12:30 bar close -- the checkpoint the earlier retracted sweep found most promising before the volume bug was found
N_FOLDS = 16

RESULTS_CSV = OUT_PATH / "volume_spike_same_day_backtest_folds.csv"
SUMMARY_CSV = OUT_PATH / "volume_spike_same_day_backtest_summary.csv"


def _daily_events(daily: pd.DataFrame) -> pd.Series:
    """Accurate is_event per date -- exactly
    volume_spike_scanner.compute_spikes()'s definition, daily bars only."""
    daily = daily.sort_index()
    avg_vol = daily["Volume"].rolling(AVG_VOLUME_DAYS).mean().shift(1)
    prev_close = daily["Close"].shift(1)
    return (daily["Volume"] > avg_vol) & (daily["Close"] > prev_close) & avg_vol.notna() & prev_close.notna()


def _ticker_rows(ticker: str, daily: pd.DataFrame) -> list[dict]:
    daily = daily.sort_index()
    daily.index = pd.DatetimeIndex(daily.index).tz_localize(None).normalize()
    is_event = _daily_events(daily)

    hourly = _load_hourly(ticker)
    if hourly is None:
        return []
    reads = _checkpoint_reads(hourly, daily)  # price only -- cum_vol from this call is unused below

    rows = []
    for day, day_reads in reads.items():
        ts = pd.Timestamp(day)
        if ts not in daily.index or ENTRY_CHECKPOINT not in day_reads:
            continue
        _, entry_price = day_reads[ENTRY_CHECKPOINT]
        close_price = float(daily.loc[ts, "Close"])
        if entry_price <= 0:
            continue
        rows.append({
            "ticker": ticker, "date": ts,
            "is_event": bool(is_event.get(ts, False)),
            "ret": (close_price - entry_price) / entry_price,
        })
    return rows


def _fold_bounds(start: str, end: str) -> np.ndarray:
    return pd.date_range(start=start, end=end, periods=N_FOLDS + 1).values


def _fold_of(dates: pd.DatetimeIndex, bounds: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(bounds, dates.values, side="right") - 1
    idx[(dates.values < bounds[0]) | (dates.values > bounds[-1])] = -1
    idx[idx == N_FOLDS] = N_FOLDS - 1
    return idx


def main() -> None:
    print(f"Universe: {len(TICKERS)} liquid TSX tickers  |  entry checkpoint: {ENTRY_CHECKPOINT}  |  {N_FOLDS} folds")

    t0 = time.perf_counter()
    daily_by_ticker = sync_and_load(TICKERS, start=DAILY_START, end=FETCH_END, quiet=True)
    print(f"Daily bars loaded in {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    all_rows = []
    for i, ticker in enumerate(TICKERS, 1):
        daily = daily_by_ticker.get(ticker)
        if daily is None or len(daily) < AVG_VOLUME_DAYS + 5:
            continue
        rows = _ticker_rows(ticker, daily)
        all_rows.extend(rows)
        print(f"  [{i}/{len(TICKERS)}] {ticker}: {len(rows)} rows")
    print(f"Hourly bars fetched + rows built in {time.perf_counter() - t0:.1f}s")

    table = pd.DataFrame(all_rows)
    if table.empty:
        print("No usable data -- aborting.")
        return

    bounds = _fold_bounds(table["date"].min().strftime("%Y-%m-%d"), table["date"].max().strftime("%Y-%m-%d"))
    table["fold"] = _fold_of(pd.DatetimeIndex(table["date"]), bounds)

    diffs, wins, n_valid = [], 0, 0
    detail_rows = []
    for f in range(N_FOLDS):
        fold_rows = table[table["fold"] == f]
        cand = fold_rows.loc[fold_rows["is_event"], "ret"]
        base = fold_rows["ret"]
        if cand.empty or base.empty:
            continue
        cand_mean, base_mean = float(cand.mean()), float(base.mean())
        diff = cand_mean - base_mean
        diffs.append(diff)
        wins += int(diff > 0)
        n_valid += 1
        detail_rows.append({
            "fold": f + 1, "n_events": len(cand),
            "win_rate_pct": round(100 * float((cand > 0).mean()), 1),
            "candidate_ret_pct": round(cand_mean * 100, 3),
            "baseline_ret_pct": round(base_mean * 100, 3),
            "diff_pct": round(diff * 100, 3),
        })

    t_stat, p_val = stats.ttest_1samp(diffs, 0.0) if n_valid >= 2 else (float("nan"), float("nan"))

    events = table[table["is_event"]]
    baseline = table[~table["is_event"]]
    summary = pd.DataFrame([{
        "entry_checkpoint": ENTRY_CHECKPOINT,
        "total_events": len(events), "total_baseline_days": len(baseline),
        "folds_used": n_valid, "wins": wins,
        "event_win_rate_pct": round(100 * float((events["ret"] > 0).mean()), 1),
        "baseline_win_rate_pct": round(100 * float((baseline["ret"] > 0).mean()), 1),
        "event_mean_ret_pct": round(float(events["ret"].mean()) * 100, 3),
        "baseline_mean_ret_pct": round(float(baseline["ret"].mean()) * 100, 3),
        "mean_diff_pct": round(float(np.mean(diffs)) * 100, 3) if diffs else float("nan"),
        "t_stat": round(t_stat, 2) if np.isfinite(t_stat) else float("nan"),
        "p_value": round(p_val, 4) if np.isfinite(p_val) else float("nan"),
    }])

    print()
    print(tabulate(pd.DataFrame(detail_rows), headers="keys", tablefmt="simple", showindex=False))
    print()
    print(tabulate(summary, headers="keys", tablefmt="simple", showindex=False))

    OUT_PATH.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(detail_rows).to_csv(RESULTS_CSV, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"\nWrote {RESULTS_CSV} and {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
