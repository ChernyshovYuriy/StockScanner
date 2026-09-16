"""
volume_spike_magnitude_timing_backtest.py
===========================================
Follow-up to volume_spike_same_day_backtest.py's validated finding (buy
intraday on a confirmed volume-spike + price-up day, sell that same
day's close -- 16/16 folds positive, p~0.0000, see that script's
docstring). That result treated every qualifying event identically (a
20%-over-average spike and a 300%-over-average spike both just
`is_event = True`) and entered at a single fixed checkpoint (midday,
~12:30). This script asks the two natural next questions on top of that
same, already-validated same-session-close exit:

  1. MAGNITUDE -- does a bigger spike (volume further above its own
     20-day average) predict a bigger same-day move, or does the edge
     plateau / decay for extreme readings (which could just as easily be
     a news-driven blow-off as "real" buying interest)?
  2. TIMING -- how much of the edge is already present at an EARLIER
     checkpoint (open_hour ~10:30, morning ~11:30) vs. waiting for
     midday/afternoon confirmation? This quantifies the "act as soon as
     possible" vs. "wait for confirmation" tradeoff directly, instead of
     leaving it to real-time visual judgement (Yahoo/RBC DI/WealthSimple
     charts).

Reuses, unchanged: the 106-name curated liquid large/mid-cap TSX
universe and hourly-bar checkpoint machinery (TICKERS, _load_hourly,
_checkpoint_reads, CHECKPOINTS, AVG_VOLUME_DAYS, ROUND_TRIP_COST_BPS)
from volume_spike_intraday_backtest.py, and the daily-bar fetch window
(START/FETCH_END) from volume_spike_daily_backtest.py -- exactly
volume_spike_same_day_backtest.py's own reuse pattern. Volume is READ
ONLY from daily bars (the accurate feed); hourly bars are used only for
intraday PRICE and to know how much of the day's volume has accumulated
by a given hour, per _checkpoint_reads()'s own adjustment-factor logic --
the same "hourly volume undercounts, hourly price doesn't" split that
volume_spike_intraday_backtest.py's retraction and
volume_spike_same_day_backtest.py both document. Only the SAME-SESSION-
CLOSE hold is tested here (next_day_close / 2day_close were already
shown negative by volume_spike_daily_backtest.py -- no need to re-test).
Excludes the "close" checkpoint (same_session_close is degenerate there
-- entry == exit).

Three analyses, in increasing order of statistical rigor:

  Part A (checkpoint main effect, 16-fold paired t-test): for each of
  the 4 entry checkpoints independently, candidate = is_event rows at
  that checkpoint, baseline = ALL ticker-days at that checkpoint (same
  methodology as volume_spike_same_day_backtest.py, just split out by
  checkpoint instead of pooling midday only).

  Part B (magnitude bins, pooled across checkpoints, 16-fold paired
  t-test): for each spike_pct bucket, candidate = is_event rows in that
  bucket (any checkpoint), baseline = ALL ticker-day-checkpoint rows.
  Tests whether bigger spikes predict bigger moves, independent of when
  in the day they're read.

  Part C (checkpoint x magnitude cross-tab, DESCRIPTIVE ONLY -- pooled
  two-sample stats, not fold-paired): splitting by both dimensions at
  once leaves too few events per cell per fold for the Part A/B fold
  methodology to be reliable, so this is reported as a plain pooled
  mean/win-rate/t-stat table, explicitly not held to the same
  significance bar as Parts A/B. Meant to be read directionally.

Usage:
    python volume_spike_magnitude_timing_backtest.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import stats
from tabulate import tabulate

from config import OUT_PATH
from market_data_cache import sync_and_load
from volume_spike_daily_backtest import FETCH_END, START as DAILY_START
from volume_spike_intraday_backtest import (
    AVG_VOLUME_DAYS,
    CHECKPOINTS,
    ROUND_TRIP_COST_BPS,
    TICKERS,
    _checkpoint_reads,
    _load_hourly,
)

ENTRY_CHECKPOINTS = [c for c in CHECKPOINTS if c != "close"]  # same_session_close is degenerate at "close"

# (label, low_inclusive, high_exclusive) on spike_pct = (cum_vol/avg_vol - 1) * 100
MAGNITUDE_BINS = [
    ("0-50%", 0.0, 50.0),
    ("50-100%", 50.0, 100.0),
    ("100-200%", 100.0, 200.0),
    (">200%", 200.0, float("inf")),
]

N_FOLDS = 16

RESULTS_CHECKPOINT_CSV = OUT_PATH / "volume_spike_magnitude_timing_checkpoint.csv"
RESULTS_MAGNITUDE_CSV = OUT_PATH / "volume_spike_magnitude_timing_magnitude.csv"
RESULTS_CROSS_CSV = OUT_PATH / "volume_spike_magnitude_timing_cross.csv"


def _bucket(spike_pct: float) -> str | None:
    for label, lo, hi in MAGNITUDE_BINS:
        if lo <= spike_pct < hi:
            return label
    return None


def _ticker_rows(ticker: str, daily: pd.DataFrame, hourly: pd.DataFrame) -> list[dict]:
    """One row per (date, checkpoint) this ticker has hourly data for --
    spike_pct/is_event/ret, same_session_close hold only."""
    daily = daily.sort_index()
    daily.index = pd.DatetimeIndex(daily.index).tz_localize(None).normalize()

    avg_volume_20 = daily["Volume"].rolling(AVG_VOLUME_DAYS).mean().shift(1)
    prev_close = daily["Close"].shift(1)
    session_close = daily["Close"]
    cost = ROUND_TRIP_COST_BPS / 10_000

    reads = _checkpoint_reads(hourly, daily)

    rows = []
    for day, day_reads in reads.items():
        ts = pd.Timestamp(day)
        if ts not in daily.index:
            continue
        avg_vol = avg_volume_20.get(ts)
        p_close = prev_close.get(ts)
        exit_price = session_close.get(ts)
        if (avg_vol is None or not np.isfinite(avg_vol) or avg_vol <= 0
                or p_close is None or not np.isfinite(p_close)
                or exit_price is None or not np.isfinite(exit_price)):
            continue

        for cp_name in ENTRY_CHECKPOINTS:
            if cp_name not in day_reads:
                continue
            cum_vol, cp_price = day_reads[cp_name]
            if cp_price <= 0:
                continue
            spike_pct = (cum_vol / avg_vol - 1.0) * 100.0
            is_event = (cum_vol > avg_vol) and (cp_price > p_close)
            ret = (exit_price - cp_price) / cp_price - cost
            rows.append({
                "ticker": ticker, "date": ts, "checkpoint": cp_name,
                "spike_pct": spike_pct, "is_event": is_event, "ret": ret,
            })
    return rows


def _fold_bounds(dates: pd.Series) -> np.ndarray:
    return pd.date_range(start=dates.min(), end=dates.max(), periods=N_FOLDS + 1).values


def _fold_of(dates: pd.DatetimeIndex, bounds: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(bounds, dates.values, side="right") - 1
    idx[(dates.values < bounds[0]) | (dates.values > bounds[-1])] = -1
    idx[idx == N_FOLDS] = N_FOLDS - 1
    return idx


def _paired_fold_test(table: pd.DataFrame, cand_mask: pd.Series) -> dict:
    """16-fold paired t-test: candidate mean return vs. baseline (ALL
    rows in `table`) mean return, per fold, skipping folds where either
    side is empty -- exactly volume_spike_same_day_backtest.py's own
    per-fold methodology."""
    diffs, wins, n_valid, total_events = [], 0, 0, 0
    for f in range(N_FOLDS):
        fold_rows = table[table["fold"] == f]
        cand = fold_rows.loc[cand_mask.reindex(fold_rows.index, fill_value=False), "ret"]
        base = fold_rows["ret"]
        if cand.empty or base.empty:
            continue
        diff = float(cand.mean()) - float(base.mean())
        diffs.append(diff)
        wins += int(diff > 0)
        n_valid += 1
        total_events += len(cand)
    t_stat, p_val = stats.ttest_1samp(diffs, 0.0) if n_valid >= 2 else (float("nan"), float("nan"))
    all_cand = table.loc[cand_mask.reindex(table.index, fill_value=False), "ret"]
    return {
        "n_events": total_events,
        "win_rate_pct": round(100 * float((all_cand > 0).mean()), 1) if len(all_cand) else float("nan"),
        "mean_ret_pct": round(float(all_cand.mean()) * 100, 3) if len(all_cand) else float("nan"),
        "baseline_mean_ret_pct": round(float(table["ret"].mean()) * 100, 3),
        "folds_used": n_valid, "wins": wins,
        "mean_diff_pct": round(float(np.mean(diffs)) * 100, 3) if diffs else float("nan"),
        "t_stat": round(t_stat, 2) if np.isfinite(t_stat) else float("nan"),
        "p_value": round(p_val, 4) if np.isfinite(p_val) else float("nan"),
    }


def main() -> None:
    print(f"Universe: {len(TICKERS)} curated liquid TSX tickers  |  "
          f"checkpoints: {ENTRY_CHECKPOINTS}  |  hold: same_session_close  |  {N_FOLDS} folds")

    t0 = time.perf_counter()
    daily_by_ticker = sync_and_load(TICKERS, start=DAILY_START, end=FETCH_END, quiet=True)
    print(f"Daily bars loaded in {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    all_rows = []
    for i, ticker in enumerate(TICKERS, 1):
        daily = daily_by_ticker.get(ticker)
        if daily is None or len(daily) < AVG_VOLUME_DAYS + 5:
            continue
        hourly = _load_hourly(ticker)
        if hourly is None:
            continue
        rows = _ticker_rows(ticker, daily, hourly)
        all_rows.extend(rows)
        print(f"  [{i}/{len(TICKERS)}] {ticker}: {len(rows)} ticker-day-checkpoint rows")
    print(f"Hourly bars fetched + rows built in {time.perf_counter() - t0:.1f}s")

    table = pd.DataFrame(all_rows)
    if table.empty:
        print("No usable data -- aborting.")
        return

    bounds = _fold_bounds(table["date"])
    table["fold"] = _fold_of(pd.DatetimeIndex(table["date"]), bounds)
    table["bucket"] = table["spike_pct"].where(table["is_event"]).apply(lambda v: _bucket(v) if pd.notna(v) else None)

    # ---- Part A: checkpoint main effect ----------------------------------
    part_a_rows = []
    for cp in ENTRY_CHECKPOINTS:
        sub = table[table["checkpoint"] == cp].reset_index(drop=True)
        if sub.empty:
            continue
        stats_row = _paired_fold_test(sub, sub["is_event"])
        stats_row = {"checkpoint": cp, **stats_row}
        part_a_rows.append(stats_row)
    part_a = pd.DataFrame(part_a_rows)

    # ---- Part B: magnitude bins, pooled across checkpoints ----------------
    part_b_rows = []
    for label, _lo, _hi in MAGNITUDE_BINS:
        mask = table["bucket"] == label
        stats_row = _paired_fold_test(table, mask)
        stats_row = {"magnitude_bucket": label, **stats_row}
        part_b_rows.append(stats_row)
    part_b = pd.DataFrame(part_b_rows)

    # ---- Part C: checkpoint x magnitude cross-tab, descriptive only -------
    part_c_rows = []
    for cp in ENTRY_CHECKPOINTS:
        cp_table = table[table["checkpoint"] == cp]
        baseline_ret = cp_table["ret"]
        for label, _lo, _hi in MAGNITUDE_BINS:
            cell = cp_table[cp_table["bucket"] == label]
            if cell.empty:
                part_c_rows.append({
                    "checkpoint": cp, "magnitude_bucket": label, "n_events": 0,
                    "win_rate_pct": float("nan"), "mean_ret_pct": float("nan"),
                    "baseline_mean_ret_pct": round(float(baseline_ret.mean()) * 100, 3) if len(baseline_ret) else float("nan"),
                    "diff_pct": float("nan"), "t_stat": float("nan"), "p_value": float("nan"),
                })
                continue
            t_stat, p_val = (stats.ttest_ind(cell["ret"], baseline_ret, equal_var=False)
                              if len(cell) >= 2 else (float("nan"), float("nan")))
            part_c_rows.append({
                "checkpoint": cp, "magnitude_bucket": label, "n_events": len(cell),
                "win_rate_pct": round(100 * float((cell["ret"] > 0).mean()), 1),
                "mean_ret_pct": round(float(cell["ret"].mean()) * 100, 3),
                "baseline_mean_ret_pct": round(float(baseline_ret.mean()) * 100, 3),
                "diff_pct": round((float(cell["ret"].mean()) - float(baseline_ret.mean())) * 100, 3),
                "t_stat": round(t_stat, 2) if np.isfinite(t_stat) else float("nan"),
                "p_value": round(p_val, 4) if np.isfinite(p_val) else float("nan"),
            })
    part_c = pd.DataFrame(part_c_rows)

    print("\n=== Part A: checkpoint main effect (16-fold paired t-test, baseline = all rows at that checkpoint) ===")
    print(tabulate(part_a, headers="keys", tablefmt="simple", showindex=False))

    print("\n=== Part B: magnitude bins, pooled across checkpoints (16-fold paired t-test, baseline = all rows) ===")
    print(tabulate(part_b, headers="keys", tablefmt="simple", showindex=False))

    print("\n=== Part C: checkpoint x magnitude cross-tab (DESCRIPTIVE ONLY -- pooled two-sample t-test, not fold-paired) ===")
    print(tabulate(part_c, headers="keys", tablefmt="simple", showindex=False))

    OUT_PATH.mkdir(parents=True, exist_ok=True)
    part_a.to_csv(RESULTS_CHECKPOINT_CSV, index=False)
    part_b.to_csv(RESULTS_MAGNITUDE_CSV, index=False)
    part_c.to_csv(RESULTS_CROSS_CSV, index=False)
    print(f"\nWrote {RESULTS_CHECKPOINT_CSV}, {RESULTS_MAGNITUDE_CSV}, {RESULTS_CROSS_CSV}")


if __name__ == "__main__":
    main()
