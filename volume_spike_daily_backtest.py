"""
volume_spike_daily_backtest.py
================================
Corrected replacement for volume_spike_intraday_backtest.py, whose
checkpoint/hold sweep is RETRACTED (see that script's own docstring):
its "cumulative volume so far" reads were summed from yfinance hourly
bars, which turned out to materially undercount real TSX trading volume
(confirmed empirically -- CNQ.TO's hourly-bar sum for a single day came
back 6.5x-12x BELOW that same day's real consolidated daily volume, most
other names 1.1x-2.4x, the ratio erratic even for one ticker across
consecutive days). Every checkpoint that script tested, including
"close", used that same broken hourly sum -- none of it reproduced what
the live scanner (volume_spike_scanner.py, which correctly uses the
accurate daily-bar feed) actually measures.

This script tests the ACTUAL production signal instead: current day's
volume > its own trailing 20-trading-day average (today excluded) AND
close > yesterday's close -- exactly volume_spike_scanner.compute_spikes()'s
definition, using ONLY daily bars (the accurate feed; no hourly data at
all, so the undercount bug above cannot affect this). The tradeoff this
resolves the intraday version's own tradeoff into: this can no longer
answer "what time of day should I check" (a historical daily bar has no
recoverable intraday timestamp -- yfinance only gives the final settled
volume for a past date, never a reconstructible partial-day snapshot),
but it CAN use the full multi-year daily history and the FULL
VOLUME_SPIKE_TICKERS_URL universe cheaply (sync_and_load batches daily
bars; no more per-ticker hourly fetch, so no more ~40-name universe cap).

Same event-study methodology as research/kangaroo_tail/verify.py /
volume_spike_intraday_backtest.py: self-baseline (each ticker's own
mean K-day forward return, event or not) + 16 non-overlapping
chronological folds + a paired one-sample t-test on the per-fold
(candidate - baseline) differences.

RESULT (2026-09, 900 tickers, 4yr window, n=121,764 events): buying at
a CONFIRMED spike day's CLOSE and holding 1/2/5/10 trading days forward
is NEGATIVE at every horizon (diff -0.37pp to -0.90pp vs. the ticker's
own normal drift, t -7.1 to -10.8, p~0, negative in EVERY one of 16
folds, both halves of the window). Buying after a spike is confirmed and
holding forward underperforms -- the move is already largely priced in
by the time you could act on this signal at the close.

THIS IS NOT THE SAME QUESTION as "does riding the spike DURING the day
it happens work" -- it doesn't, and initially this result was reported
as settling that broader question too. It doesn't: see
volume_spike_same_day_backtest.py, which tests buying INTRADAY on a
confirmed-spike day and exiting that SAME day's close, and finds a
strong POSITIVE result (16/16 folds, p~0). The two don't contradict each
other -- a spike day's own price action tends to continue through that
day (real intraday momentum/follow-through), but by the time the day is
over the move is already spent, and subsequent days revert to (or below)
normal. The actionable signal is same-day only; this script's own
negative finding about holding PAST that day stands as accurate and
important context for why the strategy has to be intraday, not
"buy today, hold overnight."

Usage:
    python volume_spike_daily_backtest.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import stats
from tabulate import tabulate

from config import OUT_PATH, VOLUME_SPIKE_TICKERS_URL
from market_data_cache import sync_and_load
from research.triple_screen.batch import load_tickers
from time_utils import market_today

AVG_VOLUME_DAYS = 20              # matches volume_spike_scanner.py's production definition exactly
ROUND_TRIP_COST_BPS = 5
HOLDING_DAYS = [1, 2, 5, 10]       # trading days forward; 1 = "next close", the closest a daily-bar study can get to "same day"
N_FOLDS = 16                      # full 4-year window now available -- back to this repo's usual fold count (see kangaroo_tail/verify.py)

FORWARD_BUFFER_DAYS = 20
_END_TS = market_today() - pd.Timedelta(days=FORWARD_BUFFER_DAYS)
END = _END_TS.strftime("%Y-%m-%d")
START = (_END_TS - pd.Timedelta(days=365 * 4)).strftime("%Y-%m-%d")
FETCH_END = market_today().strftime("%Y-%m-%d")

RESULTS_CSV = OUT_PATH / "volume_spike_daily_backtest_folds.csv"
SUMMARY_CSV = OUT_PATH / "volume_spike_daily_backtest_summary.csv"


def _events_and_returns(ticker: str, daily: pd.DataFrame) -> pd.DataFrame:
    """Long-format (date, k, is_event, ret) rows for one ticker -- ret is
    the K-trading-day forward return from that date's close, for every
    date (not just events), so the same table serves both the candidate
    and self-baseline pools downstream."""
    daily = daily.sort_index()
    avg_vol = daily["Volume"].rolling(AVG_VOLUME_DAYS).mean().shift(1)
    prev_close = daily["Close"].shift(1)
    is_event = (daily["Volume"] > avg_vol) & (daily["Close"] > prev_close) & avg_vol.notna() & prev_close.notna()

    cost = ROUND_TRIP_COST_BPS / 10_000
    frames = []
    for k in HOLDING_DAYS:
        fwd = daily["Close"].shift(-k) / daily["Close"] - 1.0 - cost
        df = pd.DataFrame({"date": daily.index, "k": k, "is_event": is_event.to_numpy(), "ret": fwd.to_numpy()})
        frames.append(df.dropna(subset=["ret"]))
    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "ticker", ticker)
    return out


def _fold_bounds() -> np.ndarray:
    return pd.date_range(start=START, end=END, periods=N_FOLDS + 1).values


def _fold_of(dates: pd.DatetimeIndex, bounds: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(bounds, dates.values, side="right") - 1
    idx[(dates.values < bounds[0]) | (dates.values > bounds[-1])] = -1
    idx[idx == N_FOLDS] = N_FOLDS - 1
    return idx


def main() -> None:
    tickers = [t for t in load_tickers(VOLUME_SPIKE_TICKERS_URL) if not t.endswith(".NE")]
    print(f"Universe: {len(tickers)} tickers  |  window {START} .. {END}  |  {N_FOLDS} folds")

    t0 = time.perf_counter()
    daily_by_ticker = sync_and_load(tickers, start=START, end=FETCH_END, quiet=True)
    print(f"Daily bars loaded in {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    frames = []
    n_ok = 0
    for ticker in tickers:
        df = daily_by_ticker.get(ticker)
        if df is None or len(df) < AVG_VOLUME_DAYS + max(HOLDING_DAYS) + 5:
            continue
        n_ok += 1
        frames.append(_events_and_returns(ticker, df))
    print(f"{n_ok}/{len(tickers)} tickers usable  |  built in {time.perf_counter() - t0:.1f}s")

    table = pd.concat(frames, ignore_index=True)
    bounds = _fold_bounds()
    table["fold"] = _fold_of(pd.DatetimeIndex(table["date"]), bounds)
    table = table[table["fold"] >= 0]

    summary_rows, detail_rows = [], []
    for k in HOLDING_DAYS:
        sub = table[table["k"] == k]
        diffs, wins, n_valid, n_events_used = [], 0, 0, 0
        cand_means, base_means = [], []
        for f in range(N_FOLDS):
            fold_rows = sub[sub["fold"] == f]
            cand = fold_rows.loc[fold_rows["is_event"], "ret"]
            base = fold_rows["ret"]
            if cand.empty or base.empty:
                continue
            cand_mean, base_mean = float(cand.mean()), float(base.mean())
            diff = cand_mean - base_mean
            diffs.append(diff)
            wins += int(diff > 0)
            n_valid += 1
            n_events_used += len(cand)
            cand_means.append(cand_mean)
            base_means.append(base_mean)
            detail_rows.append({
                "hold_days": k, "fold": f + 1, "n_events": len(cand),
                "candidate_ret_pct": round(cand_mean * 100, 3),
                "baseline_ret_pct": round(base_mean * 100, 3),
                "diff_pct": round(diff * 100, 3),
            })

        if n_valid >= 2:
            t_stat, p_val = stats.ttest_1samp(diffs, 0.0)
            mean_diff = float(np.mean(diffs))
        else:
            t_stat, p_val = float("nan"), float("nan")
            mean_diff = float(np.mean(diffs)) if diffs else float("nan")

        summary_rows.append({
            "hold_days": k, "total_events": int(sub["is_event"].sum()),
            "events_used": n_events_used, "folds_used": n_valid, "wins": wins,
            "candidate_ret_pct": round(float(np.mean(cand_means)) * 100, 3) if cand_means else float("nan"),
            "baseline_ret_pct": round(float(np.mean(base_means)) * 100, 3) if base_means else float("nan"),
            "mean_diff_pct": round(mean_diff * 100, 3) if np.isfinite(mean_diff) else float("nan"),
            "t_stat": round(t_stat, 2) if np.isfinite(t_stat) else float("nan"),
            "p_value": round(p_val, 3) if np.isfinite(p_val) else float("nan"),
        })

    summary = pd.DataFrame(summary_rows)
    print()
    print(tabulate(summary, headers="keys", tablefmt="simple", showindex=False))

    OUT_PATH.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(detail_rows).to_csv(RESULTS_CSV, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"\nWrote {RESULTS_CSV} and {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
